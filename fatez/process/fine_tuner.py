#!/usr/bin/env python3
"""
Fine tune model with labled data.

author: jy, nkmtmsys
"""
import torch
import torch.nn as nn
import torch.optim as optim
import fatez.model as model
import fatez.model.bert as bert


def Set_Tuner(config:dict = None, factory_kwargs:dict = None):
    """
    Set up a Tuner object based on given config file

    Note:
        Do NOT use this method if loading pre-trained models.
    """
    return Tuner(
        gat = model.Set_GAT(config, factory_kwargs),
        encoder = bert.Encoder(**config['encoder'], **factory_kwargs),
        bin_pro = model.Binning_Process(**config['bin_pro']),
        **config['fine_tuner'],
        **factory_kwargs,
    )


class Model(nn.Module):
    """
    We take bert model and gat seperately and combine them here considering the
    needs of XAI mechanism.
    """
    def __init__(self,
        gat = None,
        bin_pro:model.Binning_Process = None,
        bert_model:bert.Fine_Tune_Model = None,
        device:str = 'cpu',
        dtype:str = None,
        ):
        super(Model, self).__init__()
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        self.gat = gat.to(self.factory_kwargs['device'])
        self.bert_model = bert_model.to(self.factory_kwargs['device'])
        self.encoder = self.bert_model.encoder.to(self.factory_kwargs['device'])
        self.bin_pro= bin_pro

    def forward(self, fea_mats, adj_mats):
        output = self.gat(fea_mats, adj_mats)
        output = self.bin_pro(output)
        output = self.bert_model(output)
        return output

    def get_gat_output(self, fea_mats, adj_mats,):
        with torch.no_grad():
            output = self.gat.eval()(fea_mats, adj_mats)
            output = self.bin_pro(output)
        return output

    def get_encoder_output(self, fea_mats, adj_mats,):
        with torch.no_grad():
            output = self.gat.eval()(fea_mats, adj_mats)
            output = self.bin_pro(output)
            output = self.bert_model.encoder.eval()(output)
        return output


class Tuner(object):
    """
    The fine-tune processing module.
    """
    def __init__(self,
        # Models to take
        gat = None,
        encoder:bert.Encoder = None,
        bin_pro:model.Binning_Process = None,
        n_hidden:int = 2,
        n_class:int = 100,

        # Adam optimizer settings
        lr:float = 1e-4,
        betas:set = (0.9, 0.999),
        weight_decay:float = 0.001,

        # Max norm of the gradients, to prevent gradients from exploding.
        max_norm:float = 0.5,

        # Scheduler params
        sch_T_0:int = 2,
        sch_T_mult:int = 2,
        sch_eta_min:float = 1e-4 / 50,

        # Criterion params
        ignore_index:int = -100,
        # ignore_index:int = 0, # For NLLLoss
        reduction:str = 'mean',

        # factory_kwargs
        device:str = 'cpu',
        dtype:str = None,
        ):
        super(Tuner, self).__init__()
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        self.model = Model(
            gat = gat,
            bin_pro = bin_pro,
            bert_model = bert.Fine_Tune_Model(
                encoder = encoder,
                n_hidden = n_hidden,
                n_class = n_class,
                **self.factory_kwargs
            ),
            **self.factory_kwargs,
        )

        # Setting the Adam optimizer with hyper-param
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr = lr,
            betas = betas,
            weight_decay = weight_decay
        )
        # self.optimizer = optim.SGD(
        #     self.model.parameters(),
        #     lr = lr,
        #     betas = betas,
        #     weight_decay = weight_decay
        # )

        # Gradient norm clipper param
        self.max_norm = max_norm

        # Set scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0 = sch_T_0,
            T_mult = sch_T_mult,
            eta_min = sch_eta_min,
        )

        # Using Negative Log Likelihood Loss function
        self.criterion = nn.CrossEntropyLoss(
            ignore_index = ignore_index,
            reduction = reduction,
        )

        # Not supporting parallel training now
        # # Distributed GPU training if CUDA can detect more than 1 GPU
        # if with_cuda and torch.cuda.device_count() > 1:
        #     self.model = nn.DataParallel(self.model, device_ids = cuda_devices)

    def train(self, data_loader):
        """
        Under construction...
        """
        size = len(data_loader.dataset)
        num_batches = len(data_loader)
        batch_num = 1
        self.model.train()
        train_loss, correct = 0, 0
        out_gat_data = list()
        for x,y in data_loader:
            self.optimizer.zero_grad()
            node_fea_mat = x[0].to(device)
            adj_mat = x[1].to(device)
            out_gat = self.model.get_gat_output(node_fea_mat, adj_mat)
            output = self.model(node_fea_mat, adj_mat)
            for ele in out_gat.detach().tolist(): out_gat_data.append(ele)
            loss = self.criterion(output, y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            self.optimizer.step()
            acc = (output.argmax(1)==y).type(torch.float).sum().item()
            print(f"batch: {batch_num} loss: {loss} accuracy:{acc/num_batches}")
            batch_num += 1
            train_loss += loss
            correct += acc
        self.scheduler.step()
        return out_gat_data, train_loss/num_batches, correct/size


    def test(self, data_loader):
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for x, y in data_loader:
                output = self.model(
                    x[0].to(self.factory_kwargs['device']),
                    x[1].to(self.factory_kwargs['device'])
                )
                test_loss += self.criterion(output, y).item()
                correct += (output.argmax(1)==y).type(torch.float).sum().item()

        test_loss /= len(data_loader)
        correct /= len(data_loader.dataset)
        return test_loss, correct
