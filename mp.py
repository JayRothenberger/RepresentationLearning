import os

import torch
import torch.distributed as dist
import functools
import torch.multiprocessing as mp
from torch.distributed.optim import ZeroRedundancyOptimizer
from torchvision import datasets, transforms
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.backends.cudnn as cudnn

import torchvision

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import wandb

import random

from simclr import SimCLR
from argparse import Namespace

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

import flash
from myresnet import ResNet50


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def main(rank, world_size):
    print(f"Running FSPD SimCLR example on rank {rank} with world_size {world_size}.")
    setup(rank, world_size)

    transform = transforms.Compose([
                    transforms.ToTensor(),
                    # transforms.Resize(size=32, antialias=True),
                    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

    tr = torchvision.datasets.CIFAR10('../cifar-10/', train=True, download=True, transform=transform)
    val = torchvision.datasets.CIFAR10('../cifar-10/', train=False, download=True, transform=transform)

    sampler_tr = DistributedSampler(tr,
                             num_replicas=world_size,
                             rank=rank,
                             shuffle=True,  # May be True
                             seed=42, 
                             drop_last=True)
    sampler_val = DistributedSampler(val,
                             num_replicas=world_size,
                             rank=rank,
                             seed=42, 
                             shuffle=False, 
                             drop_last=True)

    n_views, batch_size = 2, 512

    train_loader = DataLoader(tr, batch_size=batch_size, shuffle=False, sampler=sampler_tr, pin_memory=True, num_workers=16, drop_last=True)
    valid_loader = DataLoader(val, batch_size=batch_size, shuffle=False, sampler=sampler_val, pin_memory=True, num_workers=16)

    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}") if torch.cuda.is_available() else torch.device("cpu")

    class MyModel(torch.nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.layers = torch.nn.ModuleList([ResNet50(num_classes=512), torch.nn.Flatten(), torch.nn.ReLU(), torch.nn.Linear(512, 512), torch.nn.Linear(512, 128)])

        
        def forward(self, x):
            x = self.layers[1](self.layers[0](x))

            return x, self.layers[-1](self.layers[2](self.layers[-2](x))), self.layers[-1](self.layers[2](self.layers[-2](x)))


    cnn_model = MyModel()

    torch.cuda.set_device(rank % torch.cuda.device_count())

    # have to send the module to the correct device first
    cnn_model.to(device)

    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=1_000_000
    )
    model = FSDP(cnn_model, 
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=torch.distributed.fsdp.MixedPrecision(
                param_dtype=torch.float16, 
                reduce_dtype=torch.float32, 
                buffer_dtype=torch.float32, 
                cast_forward_inputs=True)
            )

    opt = flash.core.optimizers.LARS(model.parameters(), lr=1.5, momentum=0.9, weight_decay=1e-2)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=20)
    # important to not use precision here because it is included in FSDP
    args = {'epochs': 1200, 'device': device, 'fp16_precision': False, 'disable_cuda': False, 'temperature': .1, 'n_views': n_views, 'batch_size': batch_size, 'log_every_n_steps': 100, 'arch': 'resnet50', 'distance': 'InfoNCE'}
    wandb.init(project='SimCLR Distances', entity='ai2es',
    name=f"{rank}: SimCLR Test",
    config={
        'experiment': 'torch_test',
        'args': args
    })
    args = Namespace(**args)

    model = SimCLR(model=model, optimizer=opt, scheduler=scheduler, args=args)
    model.train(train_loader, valid_loader)

    cleanup()


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    cudnn.deterministic = True
    cudnn.benchmark = False

    main(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]))
