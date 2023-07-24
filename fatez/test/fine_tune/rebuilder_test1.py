from fatez.process.rebuilder import rebuilder
import pandas as pd
import fatez.process.imputer as imputer
import fatez.tool.JSON as JSON
import torch
import fatez.process.pre_trainer as pre_trainer
import fatez.process.worker as worker
import fatez.model as model
import copy
import torch
from torch.utils.data import DataLoader
import fatez.test.endo_hep_prepare as prep
import fatez.lib as lib
import fatez.model as model
import fatez.process as process
import fatez.process.worker as worker
import fatez.process.fine_tuner as fine_tuner
import fatez.process.pre_trainer as pre_trainer
import fatez.process.imputer as imputer
suppressor = process.Quiet_Mode()
### input
node_dir = '/storage/peiweikeLab/jiangjunyao/fatez/fine_tune/rebuild/data/GSE231674_kft1_anno/node/'
edge_dir = '/storage/peiweikeLab/jiangjunyao/fatez/pp/celloracle_edge_tf/'
edge_label = pd.read_table('/storage/peiweikeLab/jiangjunyao/fatez/pp/label2/GSE231674_kft1_anno.txt')
edge_label.index = edge_label['sample']
config = JSON.decode('/storage/peiweikeLab/jiangjunyao/fatez/fine_tune/config/config1.json')
outdir = '/storage/peiweikeLab/jiangjunyao/fatez/fine_tune/rebuild/result/'
prefix = 'test_overfit_kft_-1_2_load_predict3'
device = 'cuda'
dtype = torch.float32
print(device)

tune_epoch = 10
training = True
rebuilding = rebuilder(node_dir=node_dir,
                       edge_label=edge_label,
                       edge_dir=edge_dir)
if training:

    prev_model_dir = '/storage/peiweikeLab/jiangjunyao/fatez/pre_train/model_tf/epoch2/config1.json_pretrainindex9_pre_train.model'
    cell_use = edge_label.loc[edge_label['label']=='club-cell']
    train_number = int(int(cell_use.shape[0]) * 0.7)
    train_cell = list(cell_use.index)[0:train_number]
    test_cell = list(cell_use.index)[train_number:cell_use.shape[0]]

    pertubation_dataloader,result_dataloader,predict_dataloader,predict_true_dataloader = rebuilding.load_data(cell_train=train_cell,cell_predict=test_cell)
    #### training here
    worker.setup(device)
    trainer = pre_trainer.Set(config, device=device, dtype=dtype,
                              prev_model = model.Load(prev_model_dir))

    imput_model = imputer.Set(config, trainer, device=device, dtype=dtype)
    for i in range(tune_epoch):
        report = imput_model.train(result_dataloader, report_batch = False,)
        print(f'\tEpoch {i} Loss: {report.iloc[0,0]} Cor: {report.iloc[0,1]}')
    report = imput_model.test(predict_true_dataloader, report_batch=True, )
    print('Test Data Report:\n', report)
    worker.cleanup(device)

    # rebuilding.set_model(config=config,
    #                      prev_model_dir=prev_model_dir,
    #                      device=device, mode = 'predict')
    # rebuilding.train(epoch=epoch,
    #                  pertubation_dataloader=pertubation_dataloader,
    #                  result_dataloader=result_dataloader)
else:
    prev_model_dir = '/storage/peiweikeLab/jiangjunyao/fatez/fine_tune/rebuild/result/test_overfit_kft_-1_2_predict_model_para.model'
    cell_use = edge_label.loc[edge_label['label']=='club-cell']
    train_cell = []
    test_cell = list(cell_use.index)[0:cell_use.shape[0]]
    pertubation_dataloader,result_dataloader,predict_dataloader,predict_true_dataloader = rebuilding.load_data(cell_train=train_cell,cell_predict=test_cell,batch_size=10)
    rebuilding.set_model(config=config,
                         prev_model_dir=prev_model_dir,
                         device=device, mode = 'predict')





# result,truth = rebuilding.predict(predict_dataloader,predict_true_dataloader)
# rebuilding.output_report(outputdir=outdir,prefix=prefix)
# torch.save(result,outdir+prefix+'result.pt')
# torch.save(truth,outdir+prefix+'truth.pt')
# result,truth = rebuilding.predict(pertubation_dataloader,result_dataloader)
# rebuilding.output_report(outputdir=outdir,prefix=prefix+'_predict')
