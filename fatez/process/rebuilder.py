
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import fatez.lib as lib
import fatez.model as model
import fatez.process.pre_trainer as pre_trainer
from sklearn.decomposition import PCA
import torch_geometric.data as pyg_d
import fatez.tool.PreprocessIO as PreprocessIO
import fatez.process.early_stopper as es
import fatez.process.worker as worker
from fatez.process.scale_network import scale_network
import numpy as np
import pickle
import umap
from sklearn.decomposition import PCA

class rebuilder(object):
    def __init__(self,
        node_dir = None,
        edge_label = None,
        edge_dir = None,
        ):
        self.matrix1 = PreprocessIO.input_csv_dict_df(
            node_dir,
            df_type='node', order_cell=False)
        self.matrix2 = PreprocessIO.input_csv_dict_df(
            edge_dir,
            df_type='edge', order_cell=False)
        for i in list(self.matrix2.keys()):
            edge = scale_network(self.matrix2[i])
            edge = torch.from_numpy(edge)
            edge = edge.to(torch.float32)
            self.matrix2[i] = edge
        self.edge_label = edge_label
        self.edge_label.index = self.edge_label['sample']
        self.pertubation_dataloader = None
        self.result_dataloader = None
        self.predict_dataloader = None
        self.predict_true_dataloader = None
        self.trainer = None
        self.train_batch_loss = []
        self.train_epoch_loss = []
        self.train_batch_cor = []
        self.train_epoch_cor = []
        self.predict_cor = []
        self.predict_loss = []
        self.train_cell_name = []
        self.test_cell_name = []
    def __tensor_cor_atac(self,tensor1, tensor2,dim=0):
        all_cor = []
        for i in range(tensor1.shape[0]):
            tensor1_use = tensor1[i]
            tensor2_use = tensor2[i]
            column1 = tensor1_use[:, dim].detach().numpy()
            column2 = tensor2_use[:, dim].detach().numpy()
            correlation1 = np.corrcoef(column1, column2)[0, 1]
            all_cor.append(correlation1)
        return (np.array(all_cor).mean())

    def load_data(self,batch_size = 10,cell_train = None,cell_predict = None):
        samples1 = []
        samples2 = []
        samples3 = []
        samples4 = []
        ### check whether label is numeric
        if isinstance(self.edge_label['label'][0],int):
            self.edge_label['label2'] = self.edge_label['label'].to_list()
        else:
            self.edge_label['label2'] = \
                self.__convert_label(self.edge_label['label'].to_list())

        ### load data
        for i in range(len(self.matrix1)):
            sample_name = list(self.matrix1.keys())[i]
            m1 = torch.from_numpy(self.matrix1[sample_name].to_numpy()).to(
                torch.float32)
            m2 = torch.from_numpy(self.matrix1[sample_name].to_numpy()).to(
                torch.float32)
            # m1 = nn.LogSoftmax(dim=-2)(m1)
            # m2 = nn.LogSoftmax(dim=-2)(m2)
            m2[:,1] = 0
            dict_key = sample_name.split('#')[0]
            label = self.edge_label['label'][str(sample_name)]
            label2 = self.edge_label['label2'][str(sample_name)]
            label2 = torch.tensor(label2).long()
            edge_name = self.edge_label.loc[sample_name, 'label']
            key_use = dict_key + '#' + str(edge_name)
            inds, attrs = lib.get_sparse_coo(self.matrix2[key_use])
            if sample_name in cell_train:
                samples1.append(
                    pyg_d.Data(
                        x=m2,
                        edge_index=inds,
                        edge_attr=attrs,
                        y=label2,
                        shape=self.matrix2[key_use].shape,
                    )
                )
                samples2.append(
                    pyg_d.Data(
                        x=m1,
                        edge_index=inds,
                        edge_attr=attrs,
                        y=label2,
                        shape=self.matrix2[key_use].shape,
                    )
                )
                self.train_cell_name.append(sample_name)
            elif sample_name in cell_predict:
                samples3.append(
                    pyg_d.Data(
                        x=m2,
                        edge_index=inds,
                        edge_attr=attrs,
                        y=label2,
                        shape=self.matrix2[key_use].shape,
                    )
                )
                samples4.append(
                    pyg_d.Data(
                        x=m1,
                        edge_index=inds,
                        edge_attr=attrs,
                        y=label2,
                        shape=self.matrix2[key_use].shape,
                    )
                )
                self.test_cell_name.append(sample_name)

        pertubation_dataloader = DataLoader(
            lib.FateZ_Dataset(samples=samples1),
            batch_size=batch_size,
            collate_fn=lib.collate_fn,
            shuffle=False
        )

        result_dataloader = DataLoader(
            lib.FateZ_Dataset(samples=samples2),
            batch_size=batch_size,
            collate_fn=lib.collate_fn,
            shuffle=False
        )
        predict_dataloader = DataLoader(
            lib.FateZ_Dataset(samples=samples3),
            batch_size=batch_size,
            collate_fn=lib.collate_fn,
            shuffle=False
        )
        predict_true_dataloader = DataLoader(
            lib.FateZ_Dataset(samples=samples4),
            batch_size=batch_size,
            collate_fn=lib.collate_fn,
            shuffle=False
        )
        return pertubation_dataloader,result_dataloader,predict_dataloader,predict_true_dataloader

    def __convert_label(self,label):

        new_label = []
        label_corr = pd.Series(range(len(set(label))),index=list(set(label)))
        for i in label:
            new_label.append(label_corr[i])
        return new_label

    def set_model(self,config,prev_model_dir = None,device = 'cuda',
                  node_recon_dim = 1,mode = 'train'):
        worker.setup(device)
        if prev_model_dir ==None:
            self.trainer = pre_trainer.Set(
                config,
                node_recon_dim=node_recon_dim,
                dtype=torch.float32,
                device=device
            )
        else:
            if  mode == 'train':
                trainer = pre_trainer.Set(
                    config,
                    dtype=torch.float32,
                    device=device,
                    prev_model=model.Load(prev_model_dir)
                )
                self.trainer = pre_trainer.Set(
                    config,
                    node_recon_dim=node_recon_dim,
                    dtype=torch.float32,
                    device=device,
                    prev_model=trainer
                )
            elif mode =='predict':
                self.trainer = pre_trainer.Set(config,
                                prev_model=model.Load(prev_model_dir),
                                dtype=torch.float32,
                                node_recon_dim=node_recon_dim,
                                device=device)

    def get_encoder_out(self,dataloader):
        all_out = []
        for x, y in dataloader:
            input = [ele.to(self.trainer.device) for ele in x]
            output = self.trainer.model.get_encoder_out(input)
            all_out.append(output)
        return torch.cat(all_out,dim=0)
    def get_gat_out(self,dataloader):
        all_out = []
        for x, y in dataloader:
            input = [ele.to(self.trainer.device) for ele in x]
            output = self.trainer.model.get_gat_out(input)
            all_out.append(output)
        return torch.cat(all_out,dim=0)

    def umap_embedding(self,pca_num=None,input=None):
        pca = PCA(n_components=pca_num)
        if pca_num == None:
            umap_obj = umap.UMAP()
            umap_result = umap_obj.fit_transform(input)
        else:
            reduced_matrix = pca.fit_transform(input)
            umap_obj = umap.UMAP()
            umap_result = umap_obj.fit_transform(reduced_matrix)

    def train(self,epoch=5,
              pertubation_dataloader = None,
              result_dataloader = None,
              node_recon_dim = 1):

        self.train_batch_loss = []
        self.train_epoch_loss = []
        self.train_batch_cor = []
        self.train_epoch_cor = []

        size = self.trainer.input_sizes
        for i in range(epoch):
            self.trainer.model.train(True)
            best_loss = 99
            loss_all = 0
            cor_all = 0
            print('epoch-------' + str(i + 1))
            for x, y in pertubation_dataloader:

                # Prepare input data as always
                input = [ele.to(self.trainer.device) for ele in x]

                node_rec, adj_rec = self.trainer.model(input)

                # Prepare pertubation result data using a seperate dataloader
                y = [result_dataloader.dataset.samples[ele].to(self.trainer.device)
                     for ele in y]

                node_results = torch.split(
                    torch.stack([ele.x for ele in y], 0),
                    node_rec.shape[1],
                    dim=1
                )[0]

                """
                Need to select training feature here by partioning node_results
                """

                # The input is not LogSoftmax-ed?
                if node_recon_dim == 1:
                    node_results = nn.LogSoftmax(dim=-2)(node_results)
                    node_results = node_results[:, :, 1]
                    node_results = node_results.reshape(node_results.shape[0], 1103,
                                                        1)
                    cor_atac = self.__tensor_cor_atac(node_rec.cpu(),
                                                      node_results.cpu(),dim=0)
                elif node_recon_dim ==2:
                    node_results = nn.LogSoftmax(dim=-2)(node_results)
                    cor_atac = self.__tensor_cor_atac(node_rec.cpu(),
                                                      node_results.cpu(),dim=1)
                adj_results = lib.get_dense_adjs(
                    y, (size['n_reg'], size['n_node'], size['edge_attr'])
                )


                # cor_rna = tensor_cor_rna(node_rec.cpu(), node_results.cpu())
                # all_cor_rna.append(cor_rna)
                # Get total loss
                loss = self.trainer.criterion(node_rec, node_results)
                if adj_rec is not None:
                    loss += self.trainer.criterion(adj_rec, adj_results)

                # Some backward stuffs here
                loss.backward()
                nn.utils.clip_grad_norm_(self.trainer.model.parameters(),
                                         self.trainer.max_norm)
                self.trainer.optimizer.step()
                self.trainer.optimizer.zero_grad()

                # Accumulate
                best_loss = min(best_loss, loss.item())
                loss_all += loss.item()
                cor_all += cor_atac

                # Some logs
                self.train_batch_loss.append(float(loss.item()))
                self.train_batch_cor.append(cor_atac)

            self.trainer.scheduler.step()
            self.train_epoch_loss.append(loss_all / len(pertubation_dataloader))
            self.train_epoch_cor.append(cor_all / len(pertubation_dataloader))

    def predict(self,predict_dataloader,predict_true_dataloader,node_recon_dim = 1):

        self.trainer.model.eval()
        self.predict_cor = []
        self.predict_loss = []
        all_predict = []
        all_true = []
        for x, y in predict_dataloader:
            # Prepare input data as always
            input = [ele.to(self.trainer.device) for ele in x]
            # Mute some debug outputs
            node_rec, adj_rec = self.trainer.model(input)
            y = [predict_true_dataloader.dataset.samples[ele].to(self.trainer.device)
                 for ele
                 in y]
            node_results = torch.stack([ele.x for ele in y], 0)

            if node_recon_dim == 1:
                node_results = nn.LogSoftmax(dim=-2)(node_results)
                node_results = node_results[:, :, 1]
                node_results = node_results.reshape(node_results.shape[0], 1103,
                                                    1)
                cor_atac = self.__tensor_cor_atac(node_rec.cpu(),
                                                  node_results.cpu(), dim=0)
                loss = self.trainer.criterion(node_rec, node_results)
                node_rec = node_rec.reshape([node_rec.shape[0], 1103])
                print(cor_atac)
            elif node_recon_dim == 2:
                node_results = nn.LogSoftmax(dim=-2)(node_results)
                cor_atac = self.__tensor_cor_atac(node_rec.cpu(),
                                                  node_results.cpu(), dim=1)
                loss = self.trainer.criterion(node_rec, node_results)
            self.predict_cor.append(cor_atac)
            self.predict_loss.append(loss.item())
            all_predict.append(node_rec)
            all_true.append(node_results)
        print(self.predict_cor)
        return torch.cat(all_predict,dim=0),torch.cat(all_true,dim=0)

    def get_umap_embedding(self,tensor):

        reshaped_tensor = np.reshape(tensor, (740, 1103 * 32))
        umap_obj = umap.UMAP()
        umap_result = umap_obj.fit_transform(reshaped_tensor)

    def output_report(self,outputdir = '/',prefix = ''):

        if len(self.train_batch_loss)>1:

            batch_report = pd.DataFrame({'loss':self.train_batch_loss
                             ,'cor':self.train_batch_cor})
            epoch_report = pd.DataFrame({'loss':self.train_epoch_loss
                             ,'cor':self.train_epoch_cor})
            batch_report.to_csv(outputdir + prefix + '_batch_report.csv')
            epoch_report.to_csv(outputdir + prefix + '_epoch_report.csv')

        if len(self.predict_cor)>1:

            predict_report = pd.DataFrame({'loss':self.predict_loss,'cor':self.predict_cor})
            predict_report.to_csv(outputdir + prefix + '_predict_report.csv')

        torch.save(
            self.trainer,
            outputdir + prefix + '_torch_save_model.model'
        )
        with open(outputdir + prefix + '_model.pkl', 'wb') as file:
            pickle.dump(self.trainer, file)
        model.Save(
            self.trainer,
            outputdir + prefix + '_model_full.model',
            save_full=True
        )
        model.Save(
            self.trainer,
            outputdir + prefix + '_model_part.model',
            save_full=True
        )