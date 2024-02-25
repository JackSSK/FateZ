#!/usr/bin/env python3
"""
Pre-train model with unlabeled data

author: jy, nkmtmsys
"""
import random
import shap
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import pandas as pd
import fatez.model as model
import fatez.model.gnn as gnn
import fatez.model.transformer as transformer
import fatez.model.position_embedder as pe
import fatez.lib as lib
import fatez.process.worker as worker 
from fatez.process.masker import Dimension_Masker, Feature_Masker



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
    Set up a Trainer object based on given config file and pre-trained model
    """
    torch.cuda.empty_cache()
    net = Trainer(
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
        **config['pre_trainer'],
        **kwargs,
    )
    if prev_model is not None:
        if str(type(prev_model)) == "<class 'dict'>":
            model.Load_state_dict(net, prev_model, load_opt_sch)
        else:
            print('Deprecated: Loading from a model object.')
            net = Trainer(
                input_sizes = config['input_sizes'],
                gat = prev_model.model.gat,
                encoder = prev_model.model.bert_model.encoder,
                graph_embedder = prev_model.model.graph_embedder,
                rep_embedder = prev_model.model.bert_model.rep_embedder,
                dtype = dtype,
                **config['pre_trainer'],
                **kwargs,
            )

    # Setup worker
    net.setup(rank = rank)
    return net



class Model(nn.Module):
    """
    Full model for pre-training.
    """
    def __init__(self,
        graph_embedder = None,
        gat = None,
        masker = Feature_Masker(),
        bert_model:transformer.Reconstructor = None,
        ):
        super(Model, self).__init__()
        self.graph_embedder = graph_embedder
        self.gat = gat
        self.bert_model = bert_model
        self.masker = masker

    def forward(self, input, return_embed = False):
        embed = self.graph_embedder(input)
        output = self.gat(embed)
        output = self.masker.mask(output,)
        output = self.bert_model(output,)
        if return_embed:
            return output, embed
        else:
            return output

    def get_gat_out(self, input,):
        with torch.no_grad():
            output = self.graph_embedder.eval()(input)
            output = self.gat.eval()(output)
        return output

    def get_encoder_out(self, input,):
        with torch.no_grad():
            output = self.graph_embedder.eval()(input)
            output = self.gat.eval()(output,)
            output = self.bert_model.encoder.eval()(output)
        return output

    def make_explainer(self, bg_data):
        return shap.GradientExplainer(
            self.bert_model,self.get_gat_out(bg_data[0], bg_data[1], bg_data[2])
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



class Trainer(object):
    """
    The pre-train processing module.
    """
    def __init__(self,
        input_sizes:list = None,

        # Models to take
        gat = None,
        encoder:transformer.Encoder = None,
        masker_params:dict = {'ratio': 0.15},
        graph_embedder = pe.Skip(),
        rep_embedder = pe.Skip(),
        train_adj:bool = False,
        node_recon_dim:int = None,

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

        # factory kwargs
        device:str = 'cpu',
        dtype:str = None,
        **kwargs
        ):
        super(Trainer, self).__init__()
        self.input_sizes = input_sizes
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.max_norm = max_norm
        self.sch_T_0 = sch_T_0
        self.sch_T_mult = sch_T_mult
        self.sch_eta_min = sch_eta_min
        self.reduction = reduction
        self.device = device
        self.dtype = dtype

        self.model = Model(
            gat = gat,
            masker = Feature_Masker(**masker_params),
            graph_embedder = graph_embedder,
            bert_model = transformer.Reconstructor(
                rep_embedder = rep_embedder,
                encoder = encoder,
                input_sizes = self.input_sizes,
                train_adj = train_adj,
                node_recon_dim = node_recon_dim,
                dtype = self.dtype,
            ),
        )
        # Torch 2
        # self.model = torch.compile(self.model)

        # Using L1 Loss for criterion
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.L1Loss(reduction = self.reduction)

    def train(self, 
            data_loader, 
            report_batch:bool = False, 
            rank=None,
            device='cpu',
        ) -> pd.DataFrame:

        self.worker.train(True)
        best_loss = 99
        loss_all = 0
        report = list()

        for x, _ in data_loader:
            input = [ele.to(self.device) for ele in x]
            recon, embed_data = self.worker(input, return_embed = True)

            # Get the input tensors
            node_mat = torch.stack([ele.x for ele in embed_data], 0)
            # The reg only version
            # node_mat = torch.split(
            #     torch.stack([ele.x for ele in input], 0),
            #     recon[0].shape[1],
            #     dim = 1
            # )[0]
            loss = self.criterion(recon[0], node_mat)

            # For Adjacent Matrix reconstruction
            if recon[1] is not None:
                size = self.input_sizes
                adj_mat = lib.get_dense_adjs(
                    embed_data, (size['n_reg'],size['n_node'],size['edge_attr'])
                )
                loss += self.criterion(recon[1], adj_mat)

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Accumulate
            best_loss = min(best_loss, loss.item())
            loss_all += loss.item()

            # Some logs
            if report_batch: report.append([loss.item()])

        # Ending current epoch
        self.scheduler.step()
        report.append([loss_all / len(data_loader)])
        report = pd.DataFrame(report)
        report.columns = ['Loss', ]
        return report

    def setup(self, rank = 0, **kwargs):
        if str(type(rank)) == "<class 'int'>":
            self.device = torch.device(rank)
            self.worker = DDP(
                self.model.to(self.device), 
                device_ids = [rank],
                # find_unused_parameters = True,
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

        
