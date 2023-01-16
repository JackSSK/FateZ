#!/usr/bin/env python3
"""
Pre-train model with unlabeled data

Note: Developing~

author: nkmtmsys
"""
import torch
import torch.nn as nn
import fatez.model as model
import fatez.model.bert as bert
# import fatez.process.grn_encoder as grn_encoder


class Model(nn.Module):
    def __init__(self,
        gat = None,
        masker:model.Masker = None,
        bin_pro:model.Binning_Process = None,
        bert_model:bert.Pre_Train_Model = None,
        device:str = 'cpu',
        dtype:str = None,
        ):
        super(Model, self).__init__()
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        self.gat = gat
        self.masker = masker
        self.bin_pro= bin_pro
        self.bert_model = bert_model
        self.gat.to(self.factory_kwargs['device'])

    def forward(self, fea_mats, adj_mats,):
        output = self.gat(fea_mats, adj_mats)
        output = self.bin_pro(output)
        output = self.masker.mask(output, factory_kwargs = self.factory_kwargs)
        output = self.bert_model(output,)
        return output

    def get_gat_output(self, fea_mats, adj_mats,):
        model = self.gat.eval()
        with torch.no_grad():
            output = self.gat(fea_mats, adj_mats)
            output = self.bin_pro(output)
        return output



class Train(object):
    """
    The pre-train process.
    """
    def __init__(self,
        gat = None,
        bin_pro:model.Binning_Process = None,
        masker:model.Masker = None,
        encoder:bert.Encoder = None,
        n_class:int = 100,
        lr:float = 1e-4,
        betas:set = (0.9, 0.999),
        weight_decay:float = 0.01,
        n_warmup_steps:int = 10000,
        log_freq:int = 10,
        with_cuda:bool = True,
        cuda_devices:set = None,
        dtype:str = None,
        ):
        super(Train, self).__init__()
        # Setting device
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        self.factory_kwargs = {'n_class':n_class,'device':device,'dtype':dtype}
        self.gat = gat
        self.bin_pro = bin_pro
        self.encoder = encoder
        self.bert_model = bert.Pre_Train_Model(self.encoder, **factory_kwargs)
        sel.model = Model(
            gat = self.gat,
            bin_pro = self.bin_pro,
            bert_model = self.bert_model,
            device = self.device,
            dtype = dtype,
        )

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model, device_ids = cuda_devices)

        # Setting the Adam optimizer with hyper-param
        self.optim = optim.Adam(
            self.model.parameters(),
            lr = lr,
            betas = betas,
            weight_decay = weight_decay
        )
        self.optim_schedule = model.LR_Scheduler(
            self.optim,
            self.n_features,
            n_warmup_steps = n_warmup_steps
        )
        # Using Negative Log Likelihood Loss function
        self.criterion = nn.NLLLoss(ignore_index = 0)
        self.log_freq = log_freq

    def train(self, epoch, train_data):
        return self.iteration(epoch, train_data)

    def test(self, epoch, test_data):
        return self.iteration(epoch, test_data, train = False)

    def iteration(self, epoch, data_loader, train = True):
        # if train:
        #     self.optim_schedule.zero_grad()
        #     loss.backward()
        #     self.optim_schedule.step_and_update_lr()
        return
