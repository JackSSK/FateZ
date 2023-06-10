#!/usr/bin/env python3
"""
Generating fake data to test model performance & compatibility.

author: jy
"""
import sys
import tracemalloc
from pkg_resources import resource_filename
import shap
import torch
import torch.nn as nn
import torch_geometric.data as pyg_d
from torch.utils.data import DataLoader
import fatez.lib as lib
import fatez.tool.JSON as JSON
import fatez.model as model
import fatez.model.mlp as mlp
import fatez.model.gnn as gnn
import fatez.model.criterion as crit
import fatez.model.position_embedder as pe
import fatez.process as process
import fatez.process.fine_tuner as fine_tuner
import fatez.process.pre_trainer as pre_trainer




class Faker(object):
    """
    Test modules with fake data
    """

    def __init__(self,
        model_config:dict = None,
        n_sample:int = 10,
        batch_size:int = 5,
        simpler_samples:bool = False,
        device:str = 'cpu',
        dtype:type = torch.float32,
        ):
        super(Faker, self).__init__()
        if model_config is None:
            path = '../data/config/gat_bert_config.json'
            # path = '../data/config/gat_bert_cnn1d_config.json'
            # path = '../data/config/gat_bert_rnn_config.json'
            self.config = JSON.decode(resource_filename(__name__, path))
        else:
            self.config = model_config
        self.n_sample = n_sample
        self.batch_size = batch_size
        self.simpler_samples = simpler_samples
        self.factory_kwargs = {'device': device, 'dtype': dtype,}

    def make_data_loader(self,):
        """
        Generate random data with given parameters and
        set up a PyTorch DataLoader.

        :return:
            torch.utils.data.DataLoader
        """
        assert self.config['fine_tuner']['n_class'] == 2
        samples = list()

        def rand_sample():
            input_sz = self.config['input_sizes']
            fea_m = torch.randn(
                (input_sz['n_node'], input_sz['node_attr']),
                dtype = self.factory_kwargs['dtype']
            )
            adj_m = torch.randn(
                (input_sz['n_reg'], input_sz['n_node'], input_sz['edge_attr']),
                dtype = self.factory_kwargs['dtype']
            )
            # Zero all features if making simple samples
            if self.simpler_samples:
                fea_m = fea_m * 0 + 1
                adj_m = adj_m * 0 + 1
            # Take last two features as testing features
            fea_m[-2:] *= 0
            adj_m[:,-2:] *= 0
            return fea_m, adj_m

        def append_sample(samples, fea_m, adj_m, label):
            inds, attrs = lib.get_sparse_coo(adj_m)
            samples.append(
                pyg_d.Data(
                    x = fea_m,
                    edge_index = inds,
                    edge_attr = attrs,
                    y = label,
                    shape = adj_m.shape,
                )
            )

        # Prepare type_0 samples
        for i in range(int(self.n_sample / 2)):
            fea_m, adj_m = rand_sample()
            fea_m[-1] += 9
            adj_m[:,-1] += 9
            append_sample(samples, fea_m, adj_m, label = 0)

        # Prepare type_1 samples
        for i in range(self.n_sample - int(self.n_sample / 2)):
            fea_m, adj_m = rand_sample()
            fea_m[-1] += 1
            adj_m[:,-1] += 1
            append_sample(samples, fea_m, adj_m, label = 1)

        return DataLoader(
            lib.FateZ_Dataset(samples), batch_size=self.batch_size, shuffle=True
        )

    def test_gat(self, config:dict = None, decision = None):
        """
        Function to test whether GAT in FateZ is performing properly or not.

        :param config:
            The configuration of testing model.

        :param decision:
            The decision model.

        :return:
            GAT model
        """
        print('Testing GAT.\n')
        suppressor = process.Quiet_Mode()
        # Initialize
        device = self.factory_kwargs['device']
        # tracemalloc.start()
        data_loader = self.make_data_loader()
        # first_size, first_peak = tracemalloc.get_traced_memory()
        # tracemalloc.reset_peak()
        # print(f"{first_size=}, {first_peak=}")

        if config is None: config = self.config
        # graph_embedder = pe.Skip()
        gat_model = gnn.Set(
            config['gnn'],
            config['input_sizes'],
            self.factory_kwargs
        )
        if decision is None:
            mlp_param = {
                'd_model': self.config['gnn']['params']['en_dim'],
                'n_hidden': 4,
                'n_class': self.config['fine_tuner']['n_class'],
            }
            decision = mlp.Model(**mlp_param, **self.factory_kwargs).to(device)
        criterion = nn.CrossEntropyLoss()
        # criterion = crit.Accuracy(requires_grad = True)

        # Using data loader to train
        for x,y in data_loader:
            out = gat_model(x[0], x[1], x[2])
            output = decision(out)
            loss = criterion(output, y.to(device))
            loss.backward()
        print(f'\tGAT Green.\n')

        gat_explain = gat_model.explain_batch([a.to(device) for a in x])
        # print(gat_explain)
        shap_values = shap.GradientExplainer(decision, out).shap_values(out)
        # print(shap_values)
        print(f'\tExplainer Green.\n')
        return gat_model

    def test_full_model(self, config:dict=None, epoch:int=50, quiet:bool=True):
        """
        Function to test whether FateZ is performing properly or not.

        :param config:
            The configuration of testing model.

        :return:
            Pre-trainer model
        """
        print('Testing Full Model.\n')
        suppressor = process.Quiet_Mode()
        # Initialize
        device = self.factory_kwargs['device']
        data_loader = self.make_data_loader()
        if config is None: config = self.config

        # Pre-train part
        if quiet: suppressor.on()
        trainer = pre_trainer.Set(config, self.factory_kwargs)
        for i in range(epoch):
            report = trainer.train(data_loader, report_batch = False)
            print(f'Epoch {i} Loss: {report.iloc[0,0]}')
        if quiet: suppressor.off()
        print(f'\tPre-Trainer Green.\n')

        # Fine tune part
        if quiet: suppressor.on()
        tuner = fine_tuner.Set(config, self.factory_kwargs, trainer.model)
        for i in range(epoch):
            report = tuner.train(data_loader, report_batch = False)
            print(f'Epoch {i} Loss: {report.iloc[0,0]}')
        # Test fine tune model
        report = tuner.test(data_loader, report_batch = True)
        print('Tuner Test Report')
        print(report)
        if quiet: suppressor.off()
        print(f'\tFine-Tuner Green.\n')

        # Test explain
        size = self.config['input_sizes']
        adj_exp = torch.zeros((size['n_reg'], size['n_node']))
        reg_exp = torch.zeros((size['n_reg'],self.config['encoder']['d_model']))
        # Make background data
        bg = [a for a,_ in DataLoader(data_loader.dataset, self.n_sample)][0]
        # Set explainer through taking input data from pseudo-dataloader
        tuner.unfreeze_encoder()
        explain = tuner.model.make_explainer([a.to(device) for a in bg])

        for x,_ in data_loader:
            data = [a.to(device) for a in x]
            adj_temp, reg_temp, _ = tuner.model.explain_batch(data, explain)
            adj_exp += adj_temp
            # Only taking explainations for class 0
            for exp in reg_temp[0]: reg_exp += abs(exp)
            break

        reg_exp = torch.sum(reg_exp, dim = -1)
        node_exp = torch.matmul(reg_exp, adj_exp.type(reg_exp.dtype))
        print('Edge Explain:\n', adj_exp, '\n')
        print('Reg Explain:\n', reg_exp, '\n')
        print('Node Explain:\n', node_exp, '\n')
        print(f'\tExplainer Green.\n')

        return trainer.model, tuner.model
