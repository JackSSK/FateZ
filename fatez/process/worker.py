#!/usr/bin/env python3
"""
Setup worker envs for DistributedDataParallel objects.

author: jy,
"""
import torch
import os
import torch.distributed as dist

def setup(
        rank:int = 0,
        world_size:int = 1,
        master_addr:str = 'localhost',
        master_port:str = '2307',
        backend:str = 'nccl',
        **kwargs
    ) -> None:
    """
    Set up process groups for GPU training
    """
    if str(type(rank)) != "<class 'int'>": return

    # Make sure using properly amount of device
    assert torch.cuda.device_count() >= world_size

    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup(rank, **kwargs):
    """
    Clean up process groups for GPU training
    """
    if str(type(rank)) != "<class 'int'>": return

    dist.destroy_process_group()
