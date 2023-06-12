#!/usr/bin/env python3
"""
Fine tune model with labled data.

author: jy, nkmtmsys
"""
import shap
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torchmetrics import AUROC
import fatez.model.gnn as gnn
import fatez.model.transformer as transformer
import fatez.model.position_embedder as pe
import fatez.model.criterion as crit



def Set(config:dict = None, factory_kwargs:dict = None, prev_model = None,):
    """
    Set up a Tuner object based on given config file (and pre-trained model)
    """
    if prev_model is None:
        return Tuner(
            input_sizes = config['input_sizes'],
            gat = gnn.Set(config['gnn'], config['input_sizes'], factory_kwargs),
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
            input_sizes = config['input_sizes'],
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
        bert_model:transformer.Classifier = None,
        **kwargs
        ):
        super(Model, self).__init__()
        self.graph_embedder = graph_embedder
        self.gat = gat
        self.bert_model = bert_model

    def forward(self, fea_mats, edge_index, edge_attr):
        output = self.graph_embedder(
            fea_mats, edge_index = edge_index, edge_attr = edge_attr
        )
        output = self.gat(output, edge_index, edge_attr)
        output = self.bert_model(output, )
        return output

    def get_gat_out(self, fea_mats, edge_index, edge_attr,):
        with torch.no_grad():
            output = self.graph_embedder.eval()(
                fea_mats, edge_index = edge_index, edge_attr = edge_attr
            )
            output = self.gat.eval()(output, edge_index, edge_attr)
        return output

    def get_encoder_out(self, fea_mats, edge_index, edge_attr,):
        with torch.no_grad():
            output = self.graph_embedder.eval()(
                fea_mats, edge_index = edge_index, edge_attr = edge_attr
            )
            output = self.gat.eval()(output, edge_index, edge_attr)
            output = self.bert_model.encoder.eval()(output)
        return output

    def make_explainer(self, bg_data):
        return shap.GradientExplainer(
            self.bert_model,
            self.get_gat_out(bg_data[0], bg_data[1], bg_data[2]),
        )

    def explain_batch(self, batch, explainer):
        adj_exp = self.gat.explain_batch(batch)
        reg_exp, vars = explainer.shap_values(
            self.get_gat_out(batch[0], batch[1], batch[2]),
            return_variances = True,
        )
        return adj_exp, reg_exp, vars



class Tuner(object):
    """
    The fine-tune processing module.
    """
    def __init__(self,
            input_sizes:list = None,

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
            **kwargs
            ):
        super(Tuner, self).__init__()
        self.input_sizes = input_sizes
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        self.n_class = n_class
        self.model = Model(
            gat = gat,
            graph_embedder = graph_embedder,
            bert_model = transformer.Classifier(
                encoder = encoder,
                rep_embedder = rep_embedder,
                n_class = n_class,
                clf_type = clf_type,
                clf_params = clf_params,
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
            node_fea_mat = x[0].to(self.factory_kwargs['device'])
            edge_index = x[1].to(self.factory_kwargs['device'])
            edge_attr = x[2].to(self.factory_kwargs['device'])
            y = y.to(self.factory_kwargs['device'])
            output = self.model(node_fea_mat, edge_index, edge_attr)
            print(output, y)
            loss = self.criterion(output, y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

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
                node_fea_mat = x[0].to(self.factory_kwargs['device'])
                edge_index = x[1].to(self.factory_kwargs['device'])
                edge_attr = x[2].to(self.factory_kwargs['device'])
                y = y.to(self.factory_kwargs['device'])
                output = self.model(node_fea_mat, edge_index, edge_attr)

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

    def unfreeze_encoder(self):
        self.model.bert_model.freeze_encoder = False
        return
