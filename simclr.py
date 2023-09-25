# adapted from: https://github.com/sthalles/SimCLR

import logging
import os
import sys
import wandb

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint

from torchvision.transforms import transforms
from torchvision import transforms, datasets

from utils import get_embeddings
from evaluate import train_linear_layer, train_knn


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']

        assert self.args.distance in ['JS', 'InfoNCE']

        if self.args.distance == 'JS':
            self.loss_distance = self.info_JS_loss
        elif self.args.distance == 'InfoNCE':
            self.loss_distance = self.info_nce_loss

        
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels


    def info_JS_loss(self, features):
        """
        Jensen Shannon loss divergence instead of cosine similarity
        """
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = (features - features.min(1, keepdim=True)[0])/(features.max(1, keepdim=True)[0] - features.min(1, keepdim=True)[0])

        # features = F.normalize(features, dim=1, p=1.0)

        lam = lambda a, b: torch.where(a*b > 0, a * torch.log2(a / b), torch.where(a > 0, -a*torch.log2(1.0000001 - a), 0.0)).sum(-1)

        m = (features.unsqueeze(0) + features.unsqueeze(1))
        
        similarity_matrix = 1.0 - ((lam(features.unsqueeze(0), m) + lam(features.unsqueeze(1), m)) / 2)

        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels


    def train(self, train_loader, test_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch_counter in range(self.args.epochs):
            data = dict()
            for images, _ in tqdm(train_loader):
                images = images.to(self.args.device)
                color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
                data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=images.size(-1), antialias=True),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.RandomApply([color_jitter], p=0.8),
                                                    transforms.RandomGrayscale(p=0.2),
                                                    transforms.GaussianBlur(kernel_size=int(0.1 * images.size(-1)) - ((1 + int(0.1 * images.size(-1))) % 2))
                                                    ])

                images = torch.cat([data_transforms(images) for i in range(self.args.n_views)], dim=0)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.loss_distance(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                # TODO: evaluations here


                if n_iter % self.args.log_every_n_steps == 0:
                    embeds = get_embeddings(self.model, train_loader, test_loader)
                    if int(os.environ["RANK"]) == 0:
                        ll_accuracy, _  = train_linear_layer(embeds, device=0)
                        knn_accuracy, _ = train_knn(embeds)
                    else:
                        knn_accuracy, ll_accuracy = 0, 0

                    agree = torch.Tensor([ll_accuracy, knn_accuracy]).to(int(os.environ["RANK"]) % torch.cuda.device_count())
                    torch.distributed.barrier()
                    torch.distributed.all_reduce(agree)
                    print('agreement reduced')
                    knn_accuracy, ll_accuracy = agree[1], agree[0]

                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    data['top1'] = float(top1[0])
                    data['top5'] = float(top5[0])
                    data['loss'] = float(loss)
                    data['knn'] = float(knn_accuracy)
                    data['linear'] = float(ll_accuracy)
                    data['learning_rate'] = float(self.optimizer.param_groups[0]['lr'])
                    wandb.log(data, commit=True, step=epoch_counter)
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], global_step=n_iter)
                    

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step(loss)
            if epoch_counter:
                print(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
