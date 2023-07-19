from fatez.process.rebuilder import rebuilder
import pandas as pd
import fatez.tool.JSON as JSON
import torch
### input
node_dir = '/storage/peiweikeLab/jiangjunyao/fatez/fine_tune/rebuild/data/GSE231674_kft1_anno/node/'
edge_dir = '/storage/peiweikeLab/jiangjunyao/fatez/pp/celloracle_edge_tf/'
edge_label = pd.read_table('/storage/peiweikeLab/jiangjunyao/fatez/pp/label2/GSE231674_kft1_anno.txt')
edge_label.index = edge_label['sample']
config = JSON.decode('/storage/peiweikeLab/jiangjunyao/fatez/pre_train/tf_config/config1.json')
prev_model_dir = '/storage/peiweikeLab/jiangjunyao/fatez/fine_tune/rebuild/result/test_overfit_kft_model.model'
outdir = '/storage/peiweikeLab/jiangjunyao/fatez/fine_tune/rebuild/result/'
prefix = 'test_overfit_kft_direct_predict'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epoch = 30
training = False
rebuilding = rebuilder(node_dir=node_dir,
                       edge_label=edge_label,
                       edge_dir=edge_dir)


if training:
    cell_use = edge_label.loc[edge_label['label']=='club-cell']
    train_number = int(int(cell_use.shape[0]) * 0.8)
    train_cell = list(cell_use.index)[0:train_number]
    test_cell = list(cell_use.index)[train_number:cell_use.shape[0]]
    pertubation_dataloader,result_dataloader,predict_dataloader,predict_true_dataloader = rebuilding.load_data(cell_train=train_cell,cell_predict=test_cell)
else:
    cell_use = edge_label.loc[edge_label['label']=='club-cell']
    train_cell = []
    test_cell = list(cell_use.index)[0:cell_use.shape[0]]
    pertubation_dataloader,result_dataloader,predict_dataloader,predict_true_dataloader = rebuilding.load_data(cell_train=train_cell,cell_predict=test_cell,batch_size=10)
rebuilding.set_model(config=config,
                     prev_model_dir=prev_model_dir,
                     device=device,)
if training:

    rebuilding.train(epoch=epoch,
                         pertubation_dataloader=pertubation_dataloader,
                         result_dataloader=result_dataloader)

result,truth = rebuilding.predict(predict_dataloader,predict_true_dataloader)
rebuilding.output_report(outputdir=outdir,prefix=prefix)
torch.save(result,outdir+prefix+'result.pt')
torch.save(truth,outdir+prefix+'truth.pt')