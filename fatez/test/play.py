#!/usr/bin/env python3
"""
This is the playground for Multiomics rebuilder.

author: jy
"""
import torch
from torch.utils.data import DataLoader
import fatez.test.endo_hep_prepare as prep
import fatez.lib as lib
import fatez.process as process
import fatez.process.worker as worker
import fatez.process.fine_tuner as fine_tuner
import fatez.process.pre_trainer as pre_trainer



def test_full_model(
    config:dict = None,
    train_dataloader:DataLoader = None,
    test_dataloader:DataLoader = None,
    train_epoch:int = 20,
    tune_epoch:int = 10,
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
    for i in range(train_epoch):
        report = trainer.train(train_dataloader, report_batch = False,)
        print(f'\tEpoch {i} Loss: {report.iloc[0,0]}')
    if quiet: suppressor.off()

    # Fine tune part
    if quiet: suppressor.on()
    tuner = fine_tuner.Set(config, trainer, device = device, dtype = dtype)
    for i in range(tune_epoch):
        report = tuner.train(train_dataloader, report_batch = False,)
        print(f'\tEpoch {i} Loss: {report.iloc[0,0]}')
    # Test fine tune model
    report = tuner.test(test_dataloader, report_batch = True,)
    print('Tuner Test Report')
    print(report)
    if quiet: suppressor.off()
    print(f'\tFine-Tuner Green.\n')


    # model.Save(trainer, 'faker_test.model')
    # trainer = pre_trainer.Set(
    #     config,
    #     prev_model = model.Load('faker_test.model'),
    #     **self.factory_kwargs
    # )
    # print('Save Load Green')
    worker.cleanup(device)
    return trainer




if __name__ == '__main__':
    device = 'cuda'
    train_dataloader, test_dataloader = prep.get_dataloaders()
    config = prep.get_config()
    test_full_model(
        config = config,
        train_dataloader = train_dataloader,
        test_dataloader = test_dataloader
        )
