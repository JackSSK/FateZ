#!/usr/bin/env python3
"""
Pearson Correlation as loss

author: jy
"""
import torch
import torch.nn as nn
from scipy.stats import pearsonr
import fatez.model.criterion as criterion



class Pearson(nn.Module):
    """
    Pearson scoring
    """

    def __init__(self, requires_grad:bool = False, **kwargs):
        super(Pearson, self).__init__()
        self.requires_grad = requires_grad
        self.kwargs = kwargs

    def forward(self, input, refer, ):
        cor_score = 0
        p_value = 0
        for index,value in enumerate(input):
            temp_cor = 0
            temp_p = 0
            for i in range(value.shape[-1]):
                cor = pearsonr(
                    value[:,i].cpu().detach().numpy(),
                    refer[index][:,i].cpu().detach().numpy()
                )
                temp_cor += cor[0]
                temp_p += cor[1]

            cor_score += (temp_cor / value.shape[-1])
            p_value += (temp_p / value.shape[-1])

        cor_score /= len(input)
        p_value /= len(input)

        if not self.requires_grad:
            return cor_score, p_value
        else:
            # Make predictions by solving least square problem
            preds = criterion.probs_to_preds(input, preds)
            return criterion.preds_to_scores(preds, cor_score)

# if __name__ == '__main__':
#     a = torch.randn(3,10,1)
#     b = torch.randn(3,10,1)
#     # print(a,b)
#     layer = Pearson()
#     layer(a,b)
