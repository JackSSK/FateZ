#!/usr/bin/env python3
"""
Impute missing modality.

Q: How to get GRN if one modality is missing?

author: jy
"""
import random
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
from fatez.process.masker import Dimension_Masker, Feature_Masker



def Set(
    config:dict=None,
    prev_model=None,
    load_full_model:bool = False,
    device:str='cpu',
    dtype:str=None,
    **kwargs
    ):
    """
    Set up a Imputer object based on given config file and pre-trained model
    """
    net = Imputer(
        input_sizes = config['input_sizes'],
        gat = gnn.Set(config['gnn'], config['input_sizes'], dtype=dtype),
        encoder = transformer.Encoder(**config['encoder'],),
        graph_embedder = pe.Set(
            config['graph_embedder'], config['input_sizes'], dtype=dtype
        ),
        rep_embedder = pe.Set(
            config['rep_embedder'], config['input_sizes'], dtype=dtype
        ),
        dtype = dtype,
        **config['imputer'],
        **kwargs,
    )
    if prev_model is not None and str(type(prev_model)) == "<class 'dict'>":
        model.Load_state_dict(net, prev_model)
    elif prev_model is not None:
        net = Imputer(
            input_sizes = config['input_sizes'],
            gat = prev_model.model.gat,
            encoder = prev_model.model.bert_model.encoder,
            graph_embedder = prev_model.model.graph_embedder,
            rep_embedder = prev_model.model.bert_model.rep_embedder,
            dtype = dtype,
            **config['imputer'],
            **kwargs,
        )
    # Setup worker env
    net.setup(device=device)
    return net



class Model(nn.Module):
    """
    Full model for pre-training.
    """
    def __init__(self,
        graph_embedder = None,
        gat = None,
        masker:Dimension_Masker = Dimension_Masker(ratio = 0.0),
        bert_model:transformer.Reconstructor = None,
        ):
        super(Model, self).__init__()
        self.graph_embedder = graph_embedder
        self.gat = gat
        self.bert_model = bert_model
        self.masker = masker

    def forward(self, input, return_embed = False):
        embed = self.graph_embedder(input)
        output = self.masker.mask(embed,)
        output = self.gat(output)
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



class Imputer(object):
    """
    The imputation module.
    """
    def __init__(self,
        input_sizes:list = None,

        # Models to take
        gat = None,
        encoder:transformer.Encoder = None,
        masker_params:dict = {'mask_token': 0},
        graph_embedder = pe.Skip(),
        rep_embedder = pe.Skip(),
        train_adj:bool = False,
        node_recon_dim:int = None,
        impute_dim:int = None,

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
        super(Imputer, self).__init__()
        self.input_sizes = input_sizes
        self.impute_dim = impute_dim
        assert self.impute_dim >= 0
        self.device = device
        self.dtype = dtype
        self.model = Model(
            gat = gat,
            masker = Dimension_Masker(dim = impute_dim, **masker_params),
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

        # Setting the Adam optimizer with hyper-param
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr = lr,
            betas = betas,
            weight_decay = weight_decay
        )

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

    def train(self, data_loader, report_batch:bool = False,):
        self.worker.train(True)
        best_loss = 99
        loss_all = 0
        report = list()

        for x, _ in data_loader:
            input = [ele.to(self.device) for ele in x]
            recon, embed_data = self.worker(input, return_embed = True)

            # Get the input tensors
            node_mat = torch.stack([ele.x for ele in embed_data], 0)
            impute_mat = node_mat[:,:, self.impute_dim:self.impute_dim+1]
            loss = self.criterion(recon[0], impute_mat)

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Accumulate
            best_loss = min(best_loss, loss.item())
            loss_all += loss.item()

            # Some logs
            if report_batch: report.append([loss.item()])

        self.scheduler.step()
        report.append([loss_all / len(data_loader)])
        report = pd.DataFrame(report)
        report.columns = ['Loss', ]
        return report

    def test(self, data_loader, report_batch:bool = False,):
        self.worker.eval()
        best_loss = 99
        loss_all = 0
        report = list()

        with torch.no_grad():
            for x, _ in data_loader:
                input = [ele.to(self.device) for ele in x]
                recon, embed_data = self.worker(input, return_embed = True)

                # Get the input tensors
                node_mat = torch.stack([ele.x for ele in embed_data], 0)
                impute_mat = node_mat[:,:, self.impute_dim:self.impute_dim+1]
                loss = self.criterion(recon[0], impute_mat)

                # Accumulate
                best_loss = min(best_loss, loss.item())
                loss_all += loss.item()

                # Some logs
                if report_batch: report.append([loss.item()])

        report.append([loss_all / len(data_loader)])
        report = pd.DataFrame(report)
        report.columns = ['Loss', ]
        return report

    def setup(self, device='cpu',):
        if str(type(device)) == "<class 'list'>":
            self.device = torch.device('cuda:0')
            self._setup()
            self.worker = DDP(self.model, device_ids = device)
        elif str(type(device)) == "<class 'str'>":
            self.device = device
            self._setup()
            self.worker = self.model
        return

    def _setup(self):
        self.model = self.model.to(self.device)
        self._set_state_values(self.optimizer.state.values())

    def _set_state_values(self, state_values):
        for state in state_values:
            for k,v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)
