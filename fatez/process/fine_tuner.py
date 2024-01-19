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
import fatez.model as model
import fatez.model.gnn as gnn
import fatez.model.transformer as transformer
import fatez.model.position_embedder as pe
import fatez.model.criterion as crit
from torch.nn.parallel import DistributedDataParallel as DDP


def Set(
        config:dict=None,
        prev_model=None,
        load_full_model:bool = False,
        load_opt_sch:bool = True,
        rank:str='cpu',
        dtype:str=None,
        **kwargs
    ):
    """
    Set up a Tuner object based on given config file (and pre-trained model)
    """
    torch.cuda.empty_cache()
    net = Tuner(
        input_sizes = config['input_sizes'],
        graph_embedder = pe.Set(
            config = config['graph_embedder'],
            input_sizes = config['input_sizes'],
            latent_dim = config['latent_dim'],
            dtype=dtype
        ),
        gat = gnn.Set(
            config = config['gnn'],
            input_sizes = config['input_sizes'],
            latent_dim = config['latent_dim'],
            dtype=dtype
        ),
        rep_embedder = pe.Set(
            config = config['rep_embedder'],
            input_sizes = config['input_sizes'],
            latent_dim = config['latent_dim'],
            dtype=dtype
        ),
        encoder = transformer.Encoder(
            d_model = config['latent_dim'],
            **config['encoder'],
        ),
        dtype = dtype,
        **config['fine_tuner'],
        **kwargs,
    )
    if prev_model is not None:
        if str(type(prev_model)) == "<class 'dict'>":
            model.Load_state_dict(net, prev_model, load_opt_sch)
        else:
            print('Deprecated: Loading from a model object.')
            net = Tuner(
                input_sizes = config['input_sizes'],
                gat = prev_model.model.gat,
                encoder = prev_model.model.bert_model.encoder,
                graph_embedder = prev_model.model.graph_embedder,
                rep_embedder = prev_model.model.bert_model.rep_embedder,
                dtype = dtype,
                **config['fine_tuner'],
                **kwargs,
            )
    # Setup worker env
    net.setup(rank = rank)
    return net



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

    def forward(self, input, return_embed = False):
        embed = self.graph_embedder(input)
        output = self.gat(embed)
        output = self.bert_model(output, )
        if return_embed:
            return output, embed
        else:
            return output

    def get_gat_out(self, input,):
        with torch.no_grad():
            output = self.graph_embedder.eval()(input)
            output = self.gat.eval()(output,)
        return output

    def get_encoder_out(self, input,):
        with torch.no_grad():
            output = self.graph_embedder.eval()(input)
            output = self.gat.eval()(output,)
            output = self.bert_model.encoder.eval()(output)
        return output

    def make_explainer(self, bg_data):
        return shap.GradientExplainer(
            self.bert_model, self.get_gat_out(bg_data)
        )

    def explain_batch(self, batch, explainer):
        adj_exp = self.gat.explain_batch(batch)
        reg_exp, vars = explainer.shap_values(
            self.get_gat_out(batch), return_variances=True
        )
        return adj_exp, reg_exp, vars
    
    def to_state_dict(self, save_full:bool = False,):
        """
        Save the model as a state dict.
        """
        if save_full:
            bert_model = self.bert_model.state_dict()
            encoder = None
            rep_embedder = None
        else:
            bert_model = None
            encoder = self.bert_model.encoder.state_dict()
            rep_embedder = self.bert_model.rep_embedder.state_dict()
        return {
            'graph_embedder':self.graph_embedder.state_dict(),
            'gnn':self.gat.state_dict(),
            'encoder':encoder,
            'rep_embedder':rep_embedder,
            'bert_model':bert_model,
        }




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
            adapter:str = None,

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

            # factory kwargs
            device:str = 'cpu',
            dtype:str = None,
            **kwargs
            ):
        super(Tuner, self).__init__()
        self.input_sizes = input_sizes
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.max_norm = max_norm
        self.sch_T_0 = sch_T_0
        self.sch_T_mult = sch_T_mult
        self.sch_eta_min = sch_eta_min
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.device = device
        self.dtype = dtype
        self.n_class = n_class

        self.model = Model(
            gat = gat,
            graph_embedder = graph_embedder,
            bert_model = transformer.Classifier(
                rep_embedder = rep_embedder,
                encoder = encoder,
                adapter = adapter,
                input_sizes = self.input_sizes,
                clf_type = clf_type,
                clf_params = clf_params,
                n_class = n_class,
                dtype = self.dtype
            ),
        )
        # Torch 2
        # self.model = torch.compile(self.model)

        # Using Negative Log Likelihood Loss function
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss(
            ignore_index = ignore_index,
            reduction = self.reduction,
        )
        
    def get_encoder_out(self,dataloader):
        all_out = []
        for x, y in dataloader:
            input = [ele.to(self.device) for ele in x]
            output = self.worker.get_encoder_out(input)
            all_out.append(output)
        return torch.cat(all_out,dim=0)

    def get_gat_out(self,dataloader):
        all_out = []
        for x, y in dataloader:
            input = [ele.to(self.device) for ele in x]
            output = self.worker.get_gat_out(input)
            all_out.append(output)
        return torch.cat(all_out,dim=0)

    def train(self, data_loader, report_batch:bool = False,):
        self.worker.train(True)
        nbatch = len(data_loader)
        best_loss = 99
        loss_all, acc_all = 0, 0
        out_gat_data = list()
        report = list()

        acc_crit = crit.Accuracy(requires_grad = False)

        for x,y in data_loader:
            y = y.to(self.device)
            output = self.worker([ele.to(self.device) for ele in x])
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

    def test(self, data_loader, report_batch:bool = False,):
        self.worker.eval()
        nbatch = len(data_loader)
        report = list()
        loss_all, acc_all, auroc_all = 0, 0, 0

        acc_crit = crit.Accuracy(requires_grad = False)
        # auroc_crit = crit.AUROC(requires_grad = False)
        auroc = AUROC("multiclass", num_classes = self.n_class)
        with torch.no_grad():
            for x, y in data_loader:
                y = y.to(self.device)
                output = self.worker([ele.to(self.device) for ele in x])

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
    
    def setup(self, rank = 0, **kwargs):
        if str(type(rank)) == "<class 'int'>":
            self.device = torch.device(rank)
            self.worker = DDP(
                self.model.to(self.device), 
                device_ids = [rank],
                # Turn this on when using LORA
                # Since encoders are frozen, they are not used in the backward
                find_unused_parameters = True,
            )
        elif str(type(rank)) == "<class 'str'>":
            self.device = rank
            self.worker = self.model.to(self.device)

        # Set optimizer
        init_optimizer = optim.AdamW(
        # self.optimizer = optim.SGD(
            self.worker.parameters(),
            lr = self.lr,
            betas = self.betas,
            weight_decay = self.weight_decay
        )
        if self.optimizer is not None:
            init_optimizer.load_state_dict(self.optimizer)
            self.optimizer = init_optimizer
        else:
            self.optimizer = init_optimizer

        # Set scheduler
        init_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0 = self.sch_T_0,
            T_mult = self.sch_T_mult,
            eta_min = self.sch_eta_min,
        )
        if self.scheduler is not None:
            init_scheduler.load_state_dict(self.scheduler)
            self.scheduler = init_scheduler
        else:
            self.scheduler = init_scheduler


        return
