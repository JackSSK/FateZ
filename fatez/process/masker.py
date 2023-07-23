#!/usr/bin/env python3
"""
Objects for masking inputs for models or parts of models.

author: jy
"""
import copy
import torch
import random



class Dimension_Masker(object):
    """
    Makes maskers on matrix dimensions.

    ToDo:
    Mask data on sparse matrices instead of full matrix.
    Then, the loss should be calculated only considering masked values?
    """
    def __init__(self, dim:int = -1, mask_token = 0, **kwargs):
        super(Dimension_Masker, self).__init__()
        self.dim = dim
        self.mask_token = mask_token

    def mask(self, input):
        result = copy.deepcopy(input)
        for ele in result:
            ele.x[:,self.dim] = ele.x[:,self.dim] * 0 + self.mask_token
        return result



class Feature_Masker(object):
    """
    Makes maskers on features.

    ToDo:
    Mask data on sparse matrices instead of full matrix.
    Then, the loss should be calculated only considering masked values?
    """
    def __init__(self, ratio:float=None, seed:int=None, mask_token=0, **kwargs):
        super(Feature_Masker, self).__init__()
        self.ratio = ratio
        self.seed = seed
        self.choices = None
        self.mask_token = mask_token

    def mask(self, input,):
        # Skip if no need to mask
        if self.ratio == 0 or self.ratio is None: return input

        # Set random seed
        if self.seed is not None:
            random.seed(self.seed)
            self.seed += 1
        # Make tensors
        size = input[0].size()
        mask = torch.ones(size)
        # Set random choices to mask
        choices = random.choices(range(size[-2]), k = int(size[-2]*self.ratio))
        assert choices is not None
        self.choices = choices
        # Make mask
        for ind in choices: mask[ind] = torch.zeros(size[-1])

        return torch.multiply(input, mask.to(input.device))
