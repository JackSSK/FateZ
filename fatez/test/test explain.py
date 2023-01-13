import fatez.tool.JSON as JSON
import fatez.model.mlp as mlp
import torch
import fatez.process.explainer as explainer
import shap
import fatez.model as model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mlp_param = {
    'd_model': 8,
    'n_hidden': 4,
    'n_class': 2,
    'device': device,
    'dtype': torch.float32,
}

out_gat = JSON.decode('E:\\out_gat\\nhead0_nhidden1_weight_decay0/out_gat\\99.pt')
model_gat = model.Load('E:\\out_gat\\nhead0_nhidden1_weight_decay0/gat.model')
# u need to use trained model
decision = mlp.Classifier(**mlp_param)
epoch_out_gat=torch.tensor(out_gat[0])
print(epoch_out_gat.shape)
print(epoch_out_gat[0][1].shape)
model_gat.explain(epoch_out_gat[0], epoch_out_gat[1])
explain = shap.GradientExplainer(decision, epoch_out_gat)
shap_values = explain.shap_values(out_gat)
print(shap_values)
explain = explainer.Gradient(decision, out_gat)
shap_values = explain.shap_values(out_gat, return_variances=True)
