import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import fatez.lib as lib
import fatez.test as test
import fatez.model as model
import fatez.tool.JSON as JSON
import fatez.process as process
import fatez.process.worker as worker
import fatez.process.fine_tuner as fine_tuner
import fatez.process.pre_trainer as pre_trainer
from pkg_resources import resource_filename
import torch_geometric.data as pyg_d
import fatez.tool.PreprocessIO as PreprocessIO
import fatez.process.early_stopper as es
from fatez.process.scale_network import scale_network
from sklearn.model_selection import train_test_split
import numpy as np

class rebuilder(object):
    def __init__(self,
        node_train_dir = None,
        label_idr = None,
        edge_dir = None,
        train_cell = None
        ):
        self.matrix1 = PreprocessIO.input_csv_dict_df(
            node_train_dir,
            df_type='node', order_cell=False)
        self.matrix2 = PreprocessIO.input_csv_dict_df(
            edge_dir,
            df_type='edge', order_cell=False)
        for i in list(self.matrix2.keys()):
            edge = scale_network(self.matrix2[i])
            edge = torch.from_numpy(edge)
            edge = edge.to(torch.float32)
            self.matrix2[i] = edge
        self.edge_label = pd.read_table(
            label_idr)
        self.edge_label.index = self.edge_label['sample']
        self.train_cell = train_cell
        self.pertubation_dataloader = None
        self.result_dataloader = None
        self.predict_dataloader = None
        self.predict_true_dataloader = None
        self.model = None
        self.train_batch_loss = []
        self.train_epoch_loss = []
        self.train_batch_cor = []
        self.train_epoch_cor = []
        self.predict_cor = []

    def __tensor_cor_atac(self,tensor1, tensor2):
        all_cor = []
        for i in range(tensor1.shape[0]):
            tensor1_use = tensor1[i]
            tensor2_use = tensor2[i]
            column1 = tensor1_use[:, 0].detach().numpy()
            column2 = tensor2_use[:, 0].detach().numpy()
            correlation1 = np.corrcoef(column1, column2)[0, 1]
            all_cor.append(correlation1)
        return (np.array(all_cor).mean())

    def load_data(self,batch_size = 10):
        samples1 = []
        samples2 = []
        samples3 = []
        samples4 = []
        for i in range(len(self.matrix1)):
            sample_name = list(self.matrix1.keys())[i]
            m1 = torch.from_numpy(self.matrix1[sample_name].to_numpy()).to(
                torch.float32)
            m2 = torch.from_numpy(self.matrix1[sample_name].to_numpy()).to(
                torch.float32)
            m2[:, 1] = 0
            dict_key = sample_name.split('#')[0]
            label = self.edge_label['label'][str(sample_name)]
            edge_name = self.edge_label.loc[sample_name, 'label']
            key_use = dict_key + '#' + str(edge_name)
            inds, attrs = lib.get_sparse_coo(self.matrix2[key_use])
            if sample_name in self.train_cell:
                samples1.append(
                    pyg_d.Data(
                        x=m2,
                        edge_index=inds,
                        edge_attr=attrs,
                        y=0,
                        shape=self.matrix2[key_use].shape,
                    )
                )
                samples2.append(
                    pyg_d.Data(
                        x=m1,
                        edge_index=inds,
                        edge_attr=attrs,
                        y=0,
                        shape=self.matrix2[key_use].shape,
                    )
                )
            else:
                samples3.append(
                    pyg_d.Data(
                        x=m2,
                        edge_index=inds,
                        edge_attr=attrs,
                        y=0,
                        shape=self.matrix2[key_use].shape,
                    )
                )
                samples4.append(
                    pyg_d.Data(
                        x=m1,
                        edge_index=inds,
                        edge_attr=attrs,
                        y=0,
                        shape=self.matrix2[key_use].shape,
                    )
                )
        self.pertubation_dataloader = DataLoader(
            lib.FateZ_Dataset(samples=samples1),
            batch_size=batch_size,
            collate_fn=lib.collate_fn,
            shuffle=False
        )

        self.result_dataloader = DataLoader(
            lib.FateZ_Dataset(samples=samples2),
            batch_size=batch_size,
            collate_fn=lib.collate_fn,
            shuffle=False
        )
        self.predict_dataloader = DataLoader(
            lib.FateZ_Dataset(samples=samples3),
            batch_size=batch_size,
            collate_fn=lib.collate_fn,
            shuffle=False
        )
        self.predict_true_dataloader = DataLoader(
            lib.FateZ_Dataset(samples=samples4),
            batch_size=batch_size,
            collate_fn=lib.collate_fn,
            shuffle=False
        )
    def train(self,config,prev_model_dir,device,epoch=5):
        trainer = pre_trainer.Set(
            config,
            dtype=torch.float32,
            device=device,
            prev_model=model.Load(prev_model_dir)
        )
        trainer = pre_trainer.Set(
            config,
            node_recon_dim=1,
            dtype=torch.float32,
            device=device,
            prev_model=trainer
        )
        self.train_batch_loss = []
        self.train_epoch_loss = []
        self.train_batch_cor = []
        self.train_epoch_cor = []
        trainer.setup()
        size = trainer.input_sizes
        for i in range(epoch):
            trainer.worker.train(True)
            best_loss = 99
            loss_all = 0
            cor_all = 0
            print('epoch-------' + str(i + 1))
            for x, y in self.pertubation_dataloader:

                # Prepare input data as always
                input = [ele.to(trainer.device) for ele in x]

                node_rec, adj_rec = trainer.worker(input)

                # Prepare pertubation result data using a seperate dataloader
                y = [self.result_dataloader.dataset.samples[ele].to(trainer.device)
                     for ele in y]

                node_results = torch.stack([ele.x for ele in y], 0)

                """
                Need to select training feature here by partioning node_results
                """

                # The input is not LogSoftmax-ed?
                node_results = nn.LogSoftmax(dim=-2)(node_results)
                node_results = node_results[:, :, 1]
                node_results = node_results.reshape(node_results.shape[0], 1103,
                                                    1)
                adj_results = lib.get_dense_adjs(
                    y, (size['n_reg'], size['n_node'], size['edge_attr'])
                )
                cor_atac = self.__tensor_cor_atac(node_rec.cpu(), node_results.cpu())
                # cor_rna = tensor_cor_rna(node_rec.cpu(), node_results.cpu())
                # all_cor_rna.append(cor_rna)
                # Get total loss
                loss = trainer.criterion(node_rec, node_results)
                if adj_rec is not None:
                    loss += trainer.criterion(adj_rec, adj_results)

                # Some backward stuffs here
                loss.backward()
                nn.utils.clip_grad_norm_(trainer.model.parameters(),
                                         trainer.max_norm)
                trainer.optimizer.step()
                trainer.optimizer.zero_grad()

                # Accumulate
                best_loss = min(best_loss, loss.item())
                loss_all += loss.item()
                cor_all += cor_atac

                # Some logs
                self.train_batch_loss.append([loss.item()])
                self.train_batch_loss.append(cor_atac)
            self.model = trainer
            self.train_epoch_loss.append(loss_all / len(self.pertubation_dataloader))
            self.train_epoch_cor.append(cor_all / len(self.pertubation_dataloader))

    def predict(self):

        self.model.model.eval()
        self.predict_cor = []
        all_predict = []
        for x, y in self.predict_dataloader:
            # Prepare input data as always
            input = [ele.to(self.model.device) for ele in x]
            # Mute some debug outputs
            node_rec, adj_rec = self.model.worker(input)
            y = [self.predict_true_dataloader.dataset.samples[ele].to(self.model.device)
                 for ele
                 in y]
            node_results = torch.stack([ele.x for ele in y], 0)
            node_results = nn.LogSoftmax(dim=-2)(node_results)
            node_results = node_results[:, :, 1]
            node_results = node_results.reshape(node_results.shape[0], 1103, 1)
            cor_atac = self.__tensor_cor_atac(node_rec.cpu(), node_results.cpu())
            self.predict_cor.append(cor_atac)
            node_rec = node_rec.reshape([node_rec.shape[0], 1103])
            all_predict.append(node_rec)

        return torch.cat(all_predict,dim=0)
