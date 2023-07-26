from fatez.process.rebuilder import rebuilder
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import fatez.lib as lib
import fatez.model as model
import fatez.process.pre_trainer as pre_trainer
import torch_geometric.data as pyg_d
import fatez.tool.PreprocessIO as PreprocessIO
import fatez.process.early_stopper as es
from fatez.process.scale_network import scale_network
import numpy as np
import pickle
import umap
import fatez.tool.JSON as JSON
import torch
### input
node_dir = '/storage/peiweikeLab/jiangjunyao/fatez/fine_tune/rebuild/data/GSE231674_kft1_anno/node/'
edge_dir = '/storage/peiweikeLab/jiangjunyao/fatez/pp/celloracle_edge_tf/'
edge_label = pd.read_table('/storage/peiweikeLab/jiangjunyao/fatez/pp/label2/GSE231674_kft1_anno.txt')
edge_label.index = edge_label['sample']
config = JSON.decode('/storage/peiweikeLab/jiangjunyao/fatez/fine_tune/config/config1.json')
outdir = '/storage/peiweikeLab/jiangjunyao/fatez/fine_tune/rebuild/result/'
prefix = 'test_overfit_kft_-1_out_script'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epoch = 30
training = True
rebuilding = rebuilder(node_dir=node_dir,
                       edge_label=edge_label,
                       edge_dir=edge_dir)

prev_model_dir = '/storage/peiweikeLab/jiangjunyao/fatez/pre_train/model_tf/epoch2/config1.json_pretrainindex9_pre_train.model'
cell_use = edge_label.loc[edge_label['label']=='club-cell']
train_number = int(int(cell_use.shape[0]) * 0.8)
train_cell = list(cell_use.index)[0:train_number]
test_cell = list(cell_use.index)[train_number:cell_use.shape[0]]
pertubation_dataloader,result_dataloader,predict_dataloader,predict_true_dataloader = rebuilding.load_data(cell_train=train_cell,cell_predict=test_cell)

###set model
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
    prev_model=trainer)
trainer.setup()

### train
def tensor_cor_atac(tensor1, tensor2, dim=0):
    all_cor = []
    for i in range(tensor1.shape[0]):
        tensor1_use = tensor1[i]
        tensor2_use = tensor2[i]
        column1 = tensor1_use[:, dim].detach().numpy()
        column2 = tensor2_use[:, dim].detach().numpy()
        correlation1 = np.corrcoef(column1, column2)[0, 1]
        all_cor.append(correlation1)
    return (np.array(all_cor).mean())
size = trainer.input_sizes
train_batch_loss = []
train_batch_cor = []
train_epoch_loss = []
train_epoch_cor = []
for i in range(epoch):
    trainer.model.train(True)
    best_loss = 99
    loss_all = 0
    cor_all = 0
    print('epoch-------' + str(i + 1))
    for x, y in pertubation_dataloader:
    # Prepare input data as always
        input = [ele.to(trainer.device) for ele in x]

        node_rec, adj_rec = trainer.model(input)

        # Prepare pertubation result data using a seperate dataloader
        y = [result_dataloader.dataset.samples[ele].to(trainer.device)
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

        node_results = nn.LogSoftmax(dim=-2)(node_results)
        node_results = node_results[:, :, 1]
        node_results = node_results.reshape(node_results.shape[0], 1103,
                                            1)
        cor_atac = tensor_cor_atac(node_rec.cpu(),
                                          node_results.cpu(), dim=0)

        adj_results = lib.get_dense_adjs(
            y, (size['n_reg'], size['n_node'], size['edge_attr'])
        )

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
        train_batch_loss.append(float(loss.item()))
        train_batch_cor.append(cor_atac)

    trainer.scheduler.step()
    train_epoch_loss.append(loss_all / len(pertubation_dataloader))
    train_epoch_cor.append(cor_all / len(pertubation_dataloader))

### predict
trainer.model.eval()
predict_cor = []
predict_loss = []
all_predict = []
all_true = []
for x, y in predict_dataloader:
    # Prepare input data as always
    input = [ele.to(trainer.device) for ele in x]
    # Mute some debug outputs
    node_rec, adj_rec = trainer.model(input)
    y = [predict_true_dataloader.dataset.samples[ele].to(trainer.device)
         for ele
         in y]
    node_results = torch.stack([ele.x for ele in y], 0)

    node_results = nn.LogSoftmax(dim=-2)(node_results)
    node_results = node_results[:, :, 1]
    node_results = node_results.reshape(node_results.shape[0], 1103,
                                        1)
    cor_atac = tensor_cor_atac(node_rec.cpu(),
                                      node_results.cpu(), dim=0)
    loss = trainer.criterion(node_rec, node_results)
    node_rec = node_rec.reshape([node_rec.shape[0], 1103])

    predict_cor.append(cor_atac)
    predict_loss.append(loss.item())
    all_predict.append(node_rec)
    all_true.append(node_results)

predict_tensor = torch.cat(all_predict,dim=0)
truth_tensor = torch.cat(all_true,dim=0)
### output
torch.save(predict_tensor,outdir+prefix+'predict_tensor.pt')
torch.save(truth_tensor,outdir+prefix+'truth_tensor.pt')
batch_report = pd.DataFrame({'loss':train_batch_loss
                             ,'cor':train_batch_cor})
epoch_report = pd.DataFrame({'loss':train_epoch_loss
                 ,'cor':train_epoch_cor})
batch_report.to_csv(outdir + prefix + '_batch_report.csv')
epoch_report.to_csv(outdir + prefix + '_epoch_report.csv')
predict_report = pd.DataFrame({'loss':predict_loss,'cor':predict_cor})
predict_report.to_csv(outdir + prefix + '_predict_report.csv')
model.Save(trainer,outdir + prefix + '_model_para.model',
            save_full=True)