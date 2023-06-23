#!/usr/bin/env python3
"""
Setup worker envs for DistributedDataParallel objects.

author: jy,
"""
import os
import torch.distributed as dist

def setup(
    device='cpu',
    master_addr:str = 'localhost',
    master_port:str = '12355',
    backend='nccl',
    rank:int=0,
    ):
    """
    Setup worker env
    Init process with nccl for GPU training, or gloo for CPU training
    """
    if str(type(device)) == "<class 'list'>":
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        dist.init_process_group(backend, rank=rank, world_size=len(device))
    # elif str(type(device)) == "<class 'str'>":
    #     if device == 'cpu':
    #         dist.init_process_group('gloo')
    return

def cleanup(device='cpu',):
    if str(type(device)) == "<class 'list'>":
        dist.destroy_process_group()
    return