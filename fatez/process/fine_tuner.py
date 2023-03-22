#!/usr/bin/env python3
"""
Fine tune model with labled data.

author: jy, nkmtmsys
"""
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torchmetrics import AUROC
import fatez.model as model
import fatez.model.mlp as mlp
import fatez.model.gat as gat
import fatez.model.cnn as cnn
import fatez.model.rnn as rnn
import fatez.model.bert as bert
import fatez.model.transformer as transformer
import fatez.model.position_embedder as pe
import fatez.model.criterion as crit



def Set(config:dict = None, factory_kwargs:dict = None, prev_model = None,):
    """
    Set up a Tuner object based on given config file (and pre-trained model)
    """
    if prev_model is None:
        return Tuner(
            gat = gat.Set(config['gnn'], config['input_sizes'], factory_kwargs),
            encoder = transformer.Encoder(**config['encoder'],**factory_kwargs),
            graph_embedder = pe.Set(
                config['graph_embedder'], config['input_sizes'], factory_kwargs
            ),
            rep_embedder = pe.Set(
                config['rep_embedder'], config['input_sizes'], factory_kwargs
            ),
            **config['fine_tuner'],
            **factory_kwargs,
        )
    else:
        return Tuner(
            gat = prev_model.gat,
            encoder = prev_model.bert_model.encoder,
            graph_embedder = prev_model.graph_embedder,
            rep_embedder = prev_model.bert_model.rep_embedder,
            **config['fine_tuner'],
            **factory_kwargs,
        )



class Model(nn.Module):
    """
    We take bert model and gat seperately and combine them here considering the
    needs of XAI mechanism.
    """
    def __init__(self,
        graph_embedder = None,
        gat = None,
        bert_model:bert.Fine_Tune_Model = None,
        **kwargs
        ):
        super(Model, self).__init__()
        self.graph_embedder = graph_embedder
        self.gat = gat
        self.bert_model = bert_model

    def forward(self, fea_mats, adj_mats):
        output = self.graph_embedder(fea_mats, adj = adj_mats)
        output = self.gat(output, adj_mats)
        output = self.bert_model(output)
        return output

    def get_gat_output(self, fea_mats, adj_mats,):
        with torch.no_grad():
            output = self.graph_embedder.eval()(fea_mats, adj = adj_mats)
            output = self.gat.eval()(output, adj_mats)
        return output

    def get_encoder_output(self, fea_mats, adj_mats,):
        with torch.no_grad():
            output = self.graph_embedder.eval()(fea_mats, adj = adj_mats)
            output = self.gat.eval()(output, adj_mats)
            output = self.bert_model.encoder.eval()(output)
        return output



class Tuner(object):
    """
    The fine-tune processing module.
    """
    def __init__(self,
        # Models to take
        gat = None,
        encoder:transformer.Encoder = None,
        graph_embedder = pe.Skip(),
        rep_embedder = pe.Skip(),
        clf_type:str = 'MLP',
        clf_params:dict = {'n_hidden': 2},
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
        self.n_class = n_class
        self.model = Model(
            gat = gat,
            graph_embedder = graph_embedder,
            bert_model = bert.Fine_Tune_Model(
                encoder = encoder,
                # Will need to take this away if embed before GAT.
                rep_embedder = rep_embedder,
                classifier = self.__set_classifier(
                    n_dim = encoder.d_model,
                    n_class = n_class,
                    clf_type = clf_type,
                    clf_params = clf_params,
                ),
                **self.factory_kwargs
            ),
        )

        # Setting the Adam optimizer with hyper-param
        self.optimizer = optim.AdamW(
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

    def train(self, data_loader, report_batch:bool = False,):
        # save_gat_out:bool = False
        self.model.train()
        self.model.to(self.factory_kwargs['device'])
        nbatch = len(data_loader)
        best_loss = 99
        loss_all, acc_all = 0, 0
        out_gat_data = list()
        report = list()

        acc_crit = crit.Accuracy(requires_grad = False)

        for x,y in data_loader:
            self.optimizer.zero_grad()

            node_fea_mat = x[0].to(self.factory_kwargs['device'])
            adj_mat = x[1].to(self.factory_kwargs['device'])
            y = y.to(self.factory_kwargs['device'])
            output = self.model(node_fea_mat, adj_mat)
            # if save_gat_out:
            #     out_gat = self.model.get_gat_output(node_fea_mat, adj_mat)
            #     for ele in out_gat.detach().tolist():
            #         out_gat_data.append(ele)
            loss = self.criterion(output, y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            self.optimizer.step()

            # Accumulate
            acc = acc_crit(output, y)
            best_loss = min(best_loss, loss.item())
            loss_all += loss
            acc_all += acc
            if report_batch: report.append([loss.item(), acc])

        self.scheduler.step()
        report.append([loss_all.item() / nbatch, acc_all / nbatch])
        report = pd.DataFrame(report)
        report.columns = ['Loss', 'ACC',]
        return report

    def test(self, data_loader, report_batch = False):
        self.model.eval()
        nbatch = len(data_loader)
        report = list()
        loss_all, acc_all, auroc_all = 0, 0, 0

        acc_crit = crit.Accuracy(requires_grad = False)
        # auroc_crit = crit.AUROC(requires_grad = False)
        auroc = AUROC("multiclass", num_classes = self.n_class)
        with torch.no_grad():
            for x, y in data_loader:
                y = y.to(self.factory_kwargs['device'])
                output = self.model(
                    x[0].to(self.factory_kwargs['device']),
                    x[1].to(self.factory_kwargs['device'])
                )

                # Batch specific
                loss = self.criterion(output, y).item()
                acc = acc_crit(output, y)
                # auc_score = auroc_crit(output, y)
                auc_score = auroc(output, y).item()

                if report_batch: report.append([loss, acc, auc_score])

                # Accumulate
                loss_all += loss
                acc_all += acc
                auroc_all += auc_score

        report.append([loss_all / nbatch, acc_all / nbatch, auroc_all / nbatch])
        report = pd.DataFrame(report)
        report.columns = ['Loss', 'ACC', 'AUROC']
        return report

    def __set_classifier(self,
        n_dim:int = 4,
        n_class:int = 2,
        clf_type:str = 'MLP',
        clf_params:dict = {'n_hidden': 2},
        ):
        """
        Set up classifier model accordingly.
        """
        if clf_type.upper() == 'MLP':
            return mlp.Model(
                d_model = n_dim,
                n_class = n_class,
                **clf_params,
                **self.factory_kwargs,
            )
        elif clf_type.upper() == 'CNN_1D':
            return cnn.Model_1D(
                in_channels = n_dim,
                n_class = n_class,
                **clf_params,
                **self.factory_kwargs,
            )
        elif clf_type.upper() == 'CNN_2D':
            return cnn.Model_2D(
                in_channels = 1,
                n_class = n_class,
                **clf_params,
                **self.factory_kwargs,
            )
        elif clf_type.upper() == 'CNN_HYB':
            return cnn.Model_Hybrid(
                in_channels = 1,
                n_class = n_class,
                **clf_params,
                **self.factory_kwargs,
            )
        elif clf_type.upper() == 'RNN':
            return rnn.RNN(
                input_size = n_dim,
                n_class = n_class,
                **clf_params,
                **self.factory_kwargs,
            )
        elif clf_type.upper() == 'GRU':
            return rnn.GRU(
                input_size = n_dim,
                n_class = n_class,
                **clf_params,
                **self.factory_kwargs,
            )
        elif clf_type.upper() == 'LSTM':
            return rnn.LSTM(
                input_size = n_dim,
                n_class = n_class,
                **clf_params,
                **self.factory_kwargs,
            )
        else:
            raise model.Error('Unknown Classifier Type:', clf_type)
