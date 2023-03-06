#!/usr/bin/env python3
"""
Fine tune model with labled data.

author: jy, nkmtmsys
"""
import torch
import torch.nn as nn
import torch.optim as optim
import fatez.model as model
import fatez.model.mlp as mlp
import fatez.model.gat as gat
import fatez.model.cnn as cnn
import fatez.model.rnn as rnn
import fatez.model.bert as bert
import fatez.model.transformer as transformer
import fatez.process.position_embedder as pe
from torchmetrics import AUROC


def Set(config:dict = None, factory_kwargs:dict = None, prev_model = None,):
    """
    Set up a Tuner object based on given config file (and pre-trained model)
    """
    if prev_model is None:
        return Tuner(
            gat = gat.Set(config['gat'], config['input_sizes'], factory_kwargs),
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
        device:str = 'cpu',
        dtype:str = None,
        ):
        super(Model, self).__init__()
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        self.graph_embedder = graph_embedder.to(self.factory_kwargs['device'])
        self.gat = gat.to(self.factory_kwargs['device'])
        self.bert_model = bert_model.to(self.factory_kwargs['device'])

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
            **self.factory_kwargs,
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

    def train(self,
        data_loader,
        print_log:bool = False,
        save_gat_out:bool = False
        ):
        self.model.train()
        num_batches = len(data_loader)
        cur_batch = 1
        best_loss = 99
        train_loss = 0
        correct = 0
        out_gat_data = list()

        for x,y in data_loader:
            self.optimizer.zero_grad()
            node_fea_mat = x[0].to(self.factory_kwargs['device'])
            adj_mat = x[1].to(self.factory_kwargs['device'])

            output = self.model(node_fea_mat, adj_mat)

            # if save_gat_out:
            #     out_gat = self.model.get_gat_output(node_fea_mat, adj_mat)
            #     for ele in out_gat.detach().tolist():
            #         out_gat_data.append(ele)

            loss = self.criterion(output, y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            self.optimizer.step()

            best_loss = min(best_loss, loss)
            train_loss += loss
            acc = (output.argmax(1)==y).type(torch.float).sum().item()
            correct += acc

            # Some logs
            if print_log:
                print(f"Batch:{cur_batch} Loss:{loss} ACC:{acc/num_batches}")
                cur_batch += 1

        self.scheduler.step()
        # return best_loss
        return train_loss/num_batches, correct/len(data_loader.dataset)

    def test(self, data_loader,write_result = False,out_dir = './'):
        self.model.eval()
        test_loss, acc_all,auroc_all = 0, 0, 0
        with torch.no_grad():
            for x, y in data_loader:
                output = self.model(
                    x[0].to(self.factory_kwargs['device']),
                    x[1].to(self.factory_kwargs['device'])
                )
                loss = self.criterion(output, y).item()
                test_loss += loss
                correct = (output.argmax(1)==y).type(torch.float).sum().item()
                acc_all += correct
                auroc = AUROC(task="binary")
                roc = auroc(output, x[1].to(self.factory_kwargs['device']))
                auroc_all += roc
                if write_result:
                    with open(out_dir + 'report.txt', 'w+') as f1:
                        f1.write('loss' + '\t' + 'acc' + '\t' + 'auroc' + '\n')
                        f1.write(
                            str(loss) + '\t' + str(correct) + '\t' + str(
                                roc) + '\n')
        test_loss /= len(data_loader)
        acc_all /= len(data_loader.dataset)
        auroc_all /= len(data_loader.dataset)
        if write_result:
            with open(out_dir + 'report.txt', 'w+') as f1:
                f1.write(
                    str(test_loss) + '\t' + str(correct) + '\t' + str(
                        auroc_all) + '\n')
        return test_loss, correct, auroc_all

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
                in_channels = n_dim,
                n_class = n_class,
                **clf_params,
                **self.factory_kwargs,
            )
        elif clf_type.upper() == 'CNN_HYB':
            return cnn.Model_Hybrid(
                in_channels = n_dim,
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
