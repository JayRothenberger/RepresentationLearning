# adapted from: https://github.com/sthalles/SimCLR

import os
import shutil
import gc

import torch
import yaml
from tqdm import tqdm
import torch.distributed as dist


def get_embeddings(model, loader, val_loader):
    global_embeds = []
    global_labels = []
    gc.collect()
    with torch.no_grad():
        for images, labels in tqdm(loader):
            local_embeds, _ = model(images)
            labels = labels.to(int(os.environ["RANK"]) % torch.cuda.device_count())
            world_embeds = [local_embeds for i in range(int(os.environ["WORLD_SIZE"]))]
            world_labels = [labels for i in range(int(os.environ["WORLD_SIZE"]))]
            dist.all_gather(world_embeds, local_embeds)
            dist.all_gather(world_labels, labels)
            world_embeds = [w.cpu() for w in world_embeds]
            world_labels = [l.cpu() for l in world_labels]
            global_embeds += world_embeds
            global_labels += world_labels
    
    global_embeds_val = []
    global_labels_val = []
    gc.collect()
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            local_embeds, _ = model(images)
            labels = labels.to(int(os.environ["RANK"]) % torch.cuda.device_count())
            world_embeds = [local_embeds for i in range(int(os.environ["WORLD_SIZE"]))]
            world_labels = [labels for i in range(int(os.environ["WORLD_SIZE"]))]
            dist.all_gather(world_embeds, local_embeds)
            dist.all_gather(world_labels, labels)
            world_embeds = [w.cpu() for w in world_embeds]
            world_labels = [l.cpu() for l in world_labels]
            global_embeds_val += world_embeds
            global_labels_val += world_labels

    return (torch.concat(global_embeds, 0), torch.concat(global_labels, 0), torch.concat(global_embeds_val, 0), torch.concat(global_labels_val, 0))


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
