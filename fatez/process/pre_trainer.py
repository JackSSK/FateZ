#!/usr/bin/env python3
"""
Pre-train model with unlabeled data

author: jy, nkmtmsys
"""
import random
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import fatez.model as model
import fatez.model.gnn as gnn
import fatez.model.transformer as transformer
import fatez.model.position_embedder as pe
import fatez.lib as lib
from torch.nn.parallel import DistributedDataParallel as DDP



def Set(
    config:dict=None,
    prev_model=None,
    device:str='cpu',
    dtype:str=None,
    **kwargs
    ):
    """
    Set up a Trainer object based on given config file and pre-trained model
    """
    if prev_model is None:
        net = Trainer(
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
            **config['pre_trainer'],
            **kwargs,
        )
    else:
        net = Trainer(
            input_sizes = config['input_sizes'],
            gat = prev_model.gat,
            encoder = prev_model.bert_model.encoder,
            graph_embedder = prev_model.graph_embedder,
            rep_embedder = prev_model.bert_model.rep_embedder,
            dtype = dtype,
            **config['pre_trainer'],
            **kwargs,
        )
    # Setup worker env
    net.setup(device=device)
    return net



class Masker(object):
    """
    Make masks for BERT encoder input.

    ToDo:
    Mask data on sparse matrices instead of full matrix.
    Then, the loss should be calculated only considering masked values?
    """
    def __init__(self, ratio, seed = None):
        super(Masker, self).__init__()
        self.ratio = ratio
        self.seed = seed
        self.choices = None

    def make_2d_mask(self, size, dtype:str = None):
        # Set random seed
        if self.seed != None:
            random.seed(self.seed)
            self.seed += 1
        # Make tensors
        answer = torch.ones(size)
        mask = torch.zeros(size[-1])
        # Set random choices to mask
        choices = random.choices(range(size[-2]), k = int(size[-2]*self.ratio))
        assert choices != None
        self.choices = choices
        # Make mask
        for ind in choices:
            answer[ind] = mask
        return answer

    def mask(self, input,):
        mask = self.make_2d_mask(input[0].size(), input.dtype).to(input.device)
        return torch.multiply(input, mask)



class Model(nn.Module):
    """
    Full model for pre-training.
    """
    def __init__(self,
        graph_embedder = None,
        gat = None,
        masker:Masker = Masker(ratio = 0.0),
        bert_model:transformer.Reconstructor = None,
        ):
        super(Model, self).__init__()
        self.graph_embedder = graph_embedder
        self.gat = gat
        self.bert_model = bert_model
        self.masker = masker


    def forward(self, input):
        output = self.graph_embedder(input)
        print('Passed GE.')
        output = self.gat(output)
        print('Passed GNN.')
        output = self.masker.mask(output,)
        print('Passed Masker.')
        output = self.bert_model(output,)
        print('Pre-Trainer Done.')
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
        self.device = device
        self.dtype = dtype
        self.model = Model(
            gat = gat,
            masker = Masker(**masker_params),
            graph_embedder = graph_embedder,
            bert_model = transformer.Reconstructor(
                rep_embedder = rep_embedder,
                encoder = encoder,
                input_sizes = self.input_sizes,
                train_adj = train_adj,
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

    def train(self, data_loader, report_batch:bool = False,):
        self.worker.train(True)
        best_loss = 99
        loss_all = 0
        report = list()

        for x, _ in data_loader:
            input = [ele.to(self.device) for ele in x]
            node_rec, adj_rec = self.worker(input)

            # Get total loss
            origin_nodes = torch.split(
                torch.stack([ele.x for ele in input], 0),
                node_rec.shape[1],
                dim = 1
            )[0]
            loss = self.criterion(node_rec, origin_nodes)
            if adj_rec is not None:
                size = self.input_sizes
                loss += self.criterion(
                    adj_rec,
                    lib.get_dense_adjs(
                        input, (size['n_reg'],size['n_node'],size['edge_attr'])
                    )
                )

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

    def setup(self, device='cpu',):
        if str(type(device)) == "<class 'list'>":
            self.device = torch.device('cuda:0')
            self.model = self.model.to(self.device)
            self.worker = DDP(self.model, device_ids = device)
        elif str(type(device)) == "<class 'str'>":
            self.device = device
            self.model = self.model.to(self.device)
            self.worker = self.model
        return
