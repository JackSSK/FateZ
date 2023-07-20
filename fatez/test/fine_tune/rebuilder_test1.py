import torch
import pandas as pd
import fatez.tool.JSON as JSON
from fatez.process.rebuilder import rebuilder
import fatez.model as model

### input
header = '/storage/peiweikeLab/jiangjunyao/fatez/'
node_dir = header + 'fine_tune/rebuild/data/GSE231674_kft1_anno/node/'
edge_dir = header + 'pp/celloracle_edge_tf/'
edge_label = pd.read_table(header + 'pp/label2/GSE231674_kft1_anno.txt')
edge_label.index = edge_label['sample']
config = JSON.decode(header + 'pre_train/tf_config/config1.json')
prev_model_dir = header+'fine_tune/rebuild/result/test_overfit_kft_model.model'
outdir = header + 'fine_tune/rebuild/result/'
prefix = 'test_overfit_kft_direct_predict'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epoch = 30
training = False
rebuilding = rebuilder(
    node_dir = node_dir,
    edge_label = edge_label,
    edge_dir = edge_dir
)

train_cell = list()
cell_use = edge_label.loc[edge_label['label']=='club-cell']

if training:
    train_number = int(int(cell_use.shape[0]) * 0.8)
    train_cell = list(cell_use.index)[0:train_number]
    test_cell = list(cell_use.index)[train_number:cell_use.shape[0]]
    pert_dl, res_dl, pred_dl, pred_true_dl = rebuilding.load_data(
        cell_train = train_cell,
        cell_predict = test_cell
    )
else:
    test_cell = list(cell_use.index)[0:cell_use.shape[0]]
    pert_dl, res_dl, pred_dl, pred_true_dl = rebuilding.load_data(
        cell_train = train_cell,
        cell_predict = test_cell,
        batch_size = 10
    )

rebuilding.set_model(
    config=config,
    prev_model_dir=prev_model_dir,
    device=device,
)

if training:
    rebuilding.train(
        epoch=epoch,
        pertubation_dataloader=pert_dl,
        result_dataloader=res_dl
    )

result, truth = rebuilding.predict(pred_dl, pred_true_dl)
rebuilding.output_report(outputdir = outdir, prefix = prefix)

model.Save(result, outdir+prefix+'result.pt', save_full = True)
model.Save(truth, outdir+prefix+'truth.pt', save_full = True)

"""
full_model = model.Load(#path)
pretrainer.Set(config, prev_model = full_model, **factory_kwargs)
"""
