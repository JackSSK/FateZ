from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn.functional as torch_F

def explain_feature(explain_weight,method = 'softmax'):

    if method == 'scale':

        scaler = MinMaxScaler(feature_range=(0, 1))
        scale_weight = scaler.fit_transform(explain_weight)
        scale_weight = np.array(scale_weight)
        return scale_weight

    elif method == 'add':

        scale_weight = np.ones(len(explain_weight))
        return scale_weight

    elif method == 'softmax':

        t_data = torch.from_numpy(explain_weight.astype(np.float32))
        scores = torch_F.softmax(t_data, dim=-1)
        scale_weight = scores.squeeze(0).data.cpu().numpy()[:, 1]
        return scale_weight
