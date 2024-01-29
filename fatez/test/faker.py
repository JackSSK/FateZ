#!/usr/bin/env python3
"""
Generating fake data to test model performance & compatibility.

author: jy
"""
import sys
import tracemalloc
import warnings
from pkg_resources import resource_filename
import shap
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_geometric.data as pyg_d
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import fatez.lib as lib
# from fatez.tool import timer
import fatez.tool.JSON as JSON
import fatez.model as model
import fatez.model.mlp as mlp
import fatez.model.gnn as gnn
import fatez.model.criterion as crit
import fatez.model.position_embedder as pe
import fatez.process as process
import fatez.process.worker as worker
import fatez.process.fine_tuner as fine_tuner
import fatez.process.pre_trainer as pre_trainer



class Faker(object):
    """
    Test modules with fake data
    """

    def __init__(self,
            warning_filter:str = 'default',
            quiet:bool = True,
            model_config:dict = None,
            n_sample:int = 10,
            batch_size:int = 5,
            simpler_samples:bool = False,
            device:str = 'cpu',
            world_size:int = 1,
            dtype:type = torch.float32,
        ):
        super(Faker, self).__init__()
        self.quiet = quiet
        self.warning_filter = warning_filter
        # warnings.filterwarnings(self.warning_filter)
        self.suppressor = process.Quiet_Mode()
        self.factory_kwargs = {
            'device': device,
            'world_size': world_size,
            'dtype': dtype,
        }

        if model_config is None:
            path = '../data/config/gat_bert_config.json'
            # path = '../data/config/gat_bert_cnn1d_config.json'
            # path = '../data/config/gat_bert_rnn_config.json'
            self.config = JSON.decode(resource_filename(__name__, path))
        else:
            self.config = model_config
        
        self.n_sample = n_sample
        self.data_loader = self.make_data_loader(
            simpler_samples, n_sample, batch_size
        )
    
    def make_data_loader(self,
            simpler_samples:bool = True,
            n_sample:int = 10,
            batch_size:int = 5
        ):
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
            fea_m = torch.abs(torch.randn(
                (input_sz['n_node'], input_sz['node_attr']),
                dtype = self.factory_kwargs['dtype']
            ))
            adj_m = torch.randn(
                (input_sz['n_reg'], input_sz['n_node'], input_sz['edge_attr']),
                dtype = self.factory_kwargs['dtype']
            )
            # Zero all features if making simple samples
            if simpler_samples:
                fea_m = fea_m * 0 + 1
                adj_m = adj_m * 0 + 1
            # Take last two features as testing features
            fea_m[-2:] *= 0
            fea_m[-2:] += 1
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
        for i in range(int(n_sample / 2)):
            fea_m, adj_m = rand_sample()
            fea_m[-1] += 8
            adj_m[:,-1] += 9
            append_sample(samples, fea_m, adj_m, label = 0)

        # Prepare type_1 samples
        for i in range(n_sample - int(n_sample / 2)):
            fea_m, adj_m = rand_sample()
            adj_m[:,-1] += 1
            append_sample(samples, fea_m, adj_m, label = 1)

        return DataLoader(
            lib.FateZ_Dataset(samples),
            batch_size = batch_size,
            collate_fn = lib.collate_fn,
            shuffle = (self.factory_kwargs['world_size'] <= 1),
        )

    def test_gat(self, config:dict = None, decision = None):
        """
        Function to test whether GAT in FateZ is performing properly or not.

        NEED TO BE REVISED

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
        # first_size, first_peak = tracemalloc.get_traced_memory()
        # tracemalloc.reset_peak()
        # print(f"{first_size=}, {first_peak=}")

        if config is None: config = self.config
        # graph_embedder = pe.Skip()
        gat_model = gnn.Set(
            config['gnn'],
            config['input_sizes'],
            **self.factory_kwargs
        ).to(device)
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
        for x,y in self.data_loader:
            out = gat_model(x)
            output = decision(out)
            loss = criterion(output, y.to(device))
            loss.backward()
        print(f'\tGAT OK.\n')

        gat_explain = gat_model.explain_batch([a.to(device) for a in x])
        # print(gat_explain)
        shap_values = shap.GradientExplainer(decision, out).shap_values(out)
        # print(shap_values)
        print(f'\tExplainer OK.\n')
        return gat_model

    def test_trainer_main(self,
            rank:int = 0,
            world_size:int = 1,
            train_epoch:int = 20,
            save_path:str = None,
        ):
        # Initialize
        worker.setup(rank, world_size)
        sampler = DistributedSampler(
            self.data_loader.dataset,
        ) if world_size > 1 else None

        # Pre-train part
        if self.quiet: self.suppressor.on()
        trainer = pre_trainer.Set(
            config = self.config,
            rank = rank,
            dtype = self.factory_kwargs['dtype'],
        )
        for i in range(train_epoch):
            # Activate distributed sampler
            if sampler is not None:
                sampler.set_epoch(i)
            # Process training
            report = trainer.train(
                self.data_loader, 
                report_batch = False,
            )
            print(f'Rank {rank} Epoch {i} Loss: {report.iloc[0,0]}')
        if self.quiet: self.suppressor.off()
        dist.destroy_process_group()

        # Testing Save Load
        if save_path is not None and rank == 0:
            model.Save(trainer, save_path)
            trainer = pre_trainer.Set(
                self.config,
                prev_model = model.Load(save_path),
                load_opt_sch = True,
                rank = 'cpu',
                dtype = self.factory_kwargs['dtype'],
            )
            print('Trainer Save Load OK.\n')
        return

    def test_tuner_main(self,
            rank:int = 0,
            world_size:int = 1,
            trainer_path:str = None, 
            tune_epoch:int = 10,
            save_path:str = None,
        ):
        """
        Function to test whether FateZ is performing properly or not.

        :return:
            Fine-Tuner model
        """
        # Initialize
        worker.setup(rank, world_size)
        sampler = DistributedSampler(
            self.data_loader.dataset,
        ) if world_size > 1 else None

        # Fine tune part
        if self.quiet: self.suppressor.on()
        if trainer_path is not None:
            trainer = model.Load(trainer_path)
        else:
            trainer = None

        tuner = fine_tuner.Set(
            config = self.config,
            prev_model =  trainer,
            load_opt_sch = False,
            rank = rank,
            dtype = self.factory_kwargs['dtype'],
        )
        for i in range(tune_epoch):
            # Activate distributed sampler
            if sampler is not None:
                sampler.set_epoch(i)
            # Process training
            report = tuner.train(
                self.data_loader,
                report_batch = False,
            )
            print(f'Rank {rank} Epoch {i} Loss: {report.iloc[0,0]}')
        # Test fine tune model
        report = tuner.test(
            self.data_loader,
            report_batch = True,
        )
        print('Tuner Test Report')
        print(report)
        if self.quiet: self.suppressor.off()
        dist.destroy_process_group()

         # Testing Save Load
        if save_path is not None and rank == 0:
            model.Save(tuner, save_path, save_full = True)
            # No need to set barrier since only testing in rank 0
            # dist.barrier()
            tuner = fine_tuner.Set(
                self.config,
                prev_model = model.Load(save_path),
                load_opt_sch = True,
                rank = 'cpu',
                dtype = self.factory_kwargs['dtype'],
            )
            print('Tuner Save Load OK.\n')
        return 

    def test_explainer(self,
            rank:int = 0,
            world_size:int = 1,
            tuner_path:str = None, 
        ):
        """
        Function to test whether FateZ is performing properly or not.
        """
        # Init
        worker.setup(rank, world_size)
        tuner = fine_tuner.Set(
            config = self.config,
            prev_model =  model.Load(tuner_path),
            load_opt_sch = True,
            rank = rank,
            dtype = self.factory_kwargs['dtype'],
        )
        size = self.config['input_sizes']
        adj_exp = torch.zeros(
            (size['n_reg'], size['n_node'])
        )
        reg_exp = torch.zeros(
            (size['n_reg'], self.config['latent_dim'])
        )
        if str(type(rank)) == "<class 'list'>":
            rank = torch.device('cuda:0')
        # Make background data
        bg = [a for a,_ in DataLoader(
                self.data_loader.dataset,
                self.n_sample,
                collate_fn = lib.collate_fn,
            )][0]

        # Set explainer through taking input data from pseudo-dataloader
        # Unfreeze encoder to get explainations if using LoRA
        tuner.unfreeze_encoder()
        explain = tuner.model.make_explainer(
            [a.to(rank) for a in bg]
        )

        for x,_ in self.data_loader:
            data = [a.to(rank) for a in x]
            adj_temp, reg_temp, _ = tuner.model.explain_batch(data, explain)
            adj_exp += adj_temp
            # Only taking explainations for class 0
            for exp in reg_temp[0]:
                reg_exp += abs(exp)
            break

        reg_exp = torch.sum(
            reg_exp,
            dim = -1
        )
        node_exp = torch.matmul(
            reg_exp, adj_exp.type(reg_exp.dtype)
        )
        print('Edge Explain:\n', adj_exp, '\n')
        print('Reg Explain:\n', reg_exp, '\n')
        print('Node Explain:\n', node_exp, '\n')
        return
