#!/usr/bin/env python3
"""
This is the playground for Multiomics rebuilder.

author: jy
"""
import copy
import torch
from torch.utils.data import DataLoader
import fatez.test.endo_hep_prepare as prep
import fatez.lib as lib
import fatez.model as model
import fatez.process as process
import fatez.process.worker as worker
import fatez.process.fine_tuner as fine_tuner
import fatez.process.pre_trainer as pre_trainer



def test_full_model(
    config:dict = None,
    train_dataloader:DataLoader = None,
    test_dataloader:DataLoader = None,
    train_epoch:int = 10,
    tune_epoch:int = 50,
    quiet:bool = False,
    device = 'cpu',
    dtype = torch.float32,
    ):
    # Initialize
    suppressor = process.Quiet_Mode()
    worker.setup(device)

    # Pre-train part
    if quiet: suppressor.on()
    trainer = pre_trainer.Set(config, device = device, dtype = dtype)
    print('Pre-Training:')
    for i in range(train_epoch):
        report = trainer.train(train_dataloader, report_batch = False,)
        print(f'\tEpoch {i} Loss: {report.iloc[0,0]}')
    if quiet: suppressor.off()

    # Fine tune part
    if quiet: suppressor.on()
    best_model = None
    best_loss = 99999
    tuner = fine_tuner.Set(config, trainer, device = device, dtype = dtype)
    print('Fine-Tuning:')
    for i in range(tune_epoch):
        report = tuner.train(train_dataloader, report_batch = False,)
        print(f'\tEpoch {i} Loss: {report.iloc[0,0]}')
        if report.iloc[0,0] <= best_loss:
            best_loss = report.iloc[0,0]
            best_model = copy.deepcopy(tuner)
            print('Updated best model')
    # Test fine tune model
    report = best_model.test(test_dataloader, report_batch = True,)
    print('Tuner Test Report')
    print(report)
    if quiet: suppressor.off()
    print(f'\tFine-Tuner Green.\n')


    model.Save(best_model, 'faker_test.model', save_full = True)
    tuner = fine_tuner.Set(
        config,
        prev_model = model.Load('faker_test.model'),
        device = device, dtype = dtype
    )
    report = tuner.test(test_dataloader, report_batch = True,)
    print(report)
    report = tuner.test(train_dataloader, report_batch = True,)
    print(report)

    worker.cleanup(device)
    return tuner




if __name__ == '__main__':
    device = 'cuda'
    train_dataloader, test_dataloader = prep.get_dataloaders()
    config = prep.get_config()
    test_full_model(
        config = config,
        train_dataloader = train_dataloader,
        test_dataloader = test_dataloader
        )
