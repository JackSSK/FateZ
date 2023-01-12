import fatez.tool.JSON as JSON
import fatez.model.mlp as mlp
import torch
import fatez.process.explainer as explainer
import shap

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mlp_param = {
    'd_model': 8,
    'n_hidden': 4,
    'n_class': 2,
    'device': device,
    'dtype': torch.float32,
}

out_gat = JSON.decode('D:\\Westlake\\pwk lab\\fatez\\gat_gradient\\nhead0_nhidden1_weight_decay0\\out_gat\\99.pt')
decision = mlp.Classifier(**mlp_param)
epoch_out_gat=torch.tensor(out_gat[0])
explain = shap.GradientExplainer(decision, epoch_out_gat)
shap_values = explain.shap_values(out_gat)
print(shap_values)
explain = explainer.Gradient(decision, out_gat)
shap_values = explain.shap_values(out_gat, return_variances=True)
