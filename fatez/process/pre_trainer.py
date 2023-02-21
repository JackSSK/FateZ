#!/usr/bin/env python3
"""
Pre-train model with unlabeled data

author: jy, nkmtmsys
"""
import torch
import torch.nn as nn
import torch.optim as optim
import fatez.model as model
import fatez.model.bert as bert



def Set_Trainer(config:dict = None, factory_kwargs:dict = None):
    """
    Set up a Trainer object based on given config file

    Note:
        Do NOT use this method if loading pre-trained models.
    """
    return Trainer(
        gat = model.Set_GAT(config, factory_kwargs),
        encoder = bert.Encoder(**config['encoder'], **factory_kwargs),
        masker = model.Masker(**config['masker']),
        bin_pro = model.Binning_Process(**config['bin_pro']),
        **config['pre_trainer'],
        **factory_kwargs,
    )



class Model(nn.Module):
    """
    Full model for pre-training.
    """
    def __init__(self,
        gat = None,
        masker:model.Masker = None,
        bin_pro:model.Binning_Process = None,
        bert_model:bert.Pre_Train_Model = None,
        device:str = 'cpu',
        dtype:str = None,
        ):
        super(Model, self).__init__()
        self.factory_kwargs = {'device': device, 'dtype': dtype,}
        self.gat = gat.to(self.factory_kwargs['device'])
        self.bert_model = bert_model.to(self.factory_kwargs['device'])
        self.encoder = self.bert_model.encoder.to(self.factory_kwargs['device'])
        self.masker = masker
        self.bin_pro= bin_pro

    def forward(self, fea_mats, adj_mats,):
        output = self.gat(fea_mats, adj_mats)
        output = self.bin_pro(output)
        output = self.masker.mask(output, factory_kwargs = self.factory_kwargs)
        output = self.bert_model(output,)
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



class Trainer(object):
    """
    The pre-train processing module.
    """
    def __init__(self,
        # Models to take
        gat = None,
        encoder:bert.Encoder = None,
        masker:model.Masker = None,
        bin_pro:model.Binning_Process = None,
        n_dim_adj:int = None,

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
        reduction:str = 'mean',

        # factory_kwargs
        device:str = 'cpu',
        dtype:str = None,
        ):
        super(Trainer, self).__init__()
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        self.model = Model(
            gat = gat,
            masker = masker,
            bin_pro = bin_pro,
            bert_model = bert.Pre_Train_Model(
                encoder = encoder,
                n_dim_node = gat.d_model,
                n_dim_adj = n_dim_adj,
                **self.factory_kwargs,
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

        # Using L1 Loss for criterion
        self.criterion = nn.L1Loss(reduction = reduction)

        # Not supporting parallel training now
        # # Distributed GPU training if CUDA can detect more than 1 GPU
        # if with_cuda and torch.cuda.device_count() > 1:
        #     self.model = nn.DataParallel(self.model, device_ids = cuda_devices)

    def train(self, data_loader, print_log:bool = False):
        cur_batch = 1
        best_loss = 99
        train_loss = 0

        self.model.train()
        for x, _ in data_loader:
            self.optimizer.zero_grad()
            node_fea_mat = x[0].to(self.factory_kwargs['device'])
            adj_mat = x[1].to(self.factory_kwargs['device'])
            output_node, output_adj = self.model(node_fea_mat, adj_mat)

            # Get total loss
            loss_node = self.criterion(
                output_node,
                torch.split(node_fea_mat, output_node.shape[1], dim = 1)[0]
            )
            if output_adj is not None:
                loss_adj = self.criterion(output_adj, adj_mat)
                loss = loss_node + loss_adj
            else:
                loss = loss_node
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            self.optimizer.step()

            # Some logs
            if print_log:
                print(f"batch: {cur_batch} loss: {loss}")
                cur_batch += 1
            best_loss = min(best_loss, loss)
            train_loss += loss

        self.scheduler.step()
        # return best_loss
        return train_loss / len(data_loader)
