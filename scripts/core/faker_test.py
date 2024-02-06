#!/usr/bin/env python3
"""
Test modules with fake data.
Using DDP for multi-gpu training.

author: jy
"""
import time
import torch
import fatez as fz
import fatez.model as model
from fatez.tool import timer
import torch.multiprocessing as mp
from fatez.test.faker import Faker


if __name__ == '__main__':
    n_gpus = torch.cuda.device_count()
    train_epoch = 2
    tune_epoch = 2
    trainer_save_path = '../../results/faker.ckpt'
    tuner_save_path = '../../results/faker.ckpt'
    print('Using GPUs:', n_gpus)

    t0 = time.time()
    fake = Faker(
        world_size = n_gpus,
        device = list(range(n_gpus)),
        quiet = False,
    )
    print(f'Faker init time: {time.time()-t0:.4f}s\n')

    print('Testing Trainer Model.\n')
    t0 = time.time()
    mp.spawn(
        fake.test_trainer_main,
        args = (
            n_gpus,
            train_epoch,
            trainer_save_path,
        ),
        #  total number of processes - # gpus
        nprocs = n_gpus, 
    )
    print('Pre-Trainer OK.\n')
    print(f'Pre-Trainer time: {time.time()-t0:.4f}s\n')

    # print('Testing Tuner Model.\n')
    # t0 = time.time()
    # mp.spawn(
    #    fake.test_tuner_main,
    #    args = (
    #        n_gpus,
    #        trainer_save_path,
    #        train_epoch,
    #        tuner_save_path,
    #    ),
    #    #  total number of processes - # gpus
    #    nprocs = n_gpus, 
    #)
    # print('Fine-Tuner OK.\n')
    # print(f'Fine-Tuner time: {time.time()-t0:.4f}s\n') 

    # print('Testing Explainer.\n')
    # t0 = time.time()
    # fake.test_explainer(
    #     rank = 0,
    #    world_size = 1,
    #    tuner_path = tuner_save_path,
    #)
    #print('Explainer OK.\n')
    # print(f'Explainer time: {time.time()-t0:.4f}s\n')


