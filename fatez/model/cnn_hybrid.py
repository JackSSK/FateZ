#!/usr/bin/env python3
"""
This file contains classes to build CNN based classifiers

ToDo:
Implement _earlystopping

Note 1
Just in case if LazyLinear having problem
You may want to try torchlayers
# import torchlayers as tl

Note 2
If using SHAP deepLIFT based method or Cuptum,
repeated layers cannot be interpreted.
Thus, layer set loop-adding feature is removed in limited versions
which has layer set max fixed being 3, should be enough

author: jy, nkmtmsys
"""

import os
import math
import torch
import itertools
import torch.nn as nn
from warnings import warn
import torch.optim as optim
import torch.nn.functional as func
from torch.utils.data import DataLoader
import ageas.classifier as classifier



class Limited(nn.Module):
    """
    Defining a CNN model treating input as 2D data
    with given hyperparameters
    then using 2 1D convolution kernels to generate layers
    Layer set number limited to max == 3
    """
    def __init__(self, id, param, d_model = 1, n_class = 2,):
        super().__init__()

        # Initialization
        self.id = id
        self.model_type = 'CNN_Hybrid_Limited'
        self.matrixSize = param['matrix_size']
        self.num_layers = param['num_layers']
        self.loss_func = nn.CrossEntropyLoss()


        model_verti = OrderedDict([
            ('conv0', nn.Conv2d(
                d_model,
                param['conv_kernel_num'],
                (1, self.matrixSize[1])
            )),
            ('relu0', nn.ReLu(inplace = False)),
            ('pool0', nn.MaxPool2d((1, param['maxpool_kernel_size'])))
        ])

        model_horiz = OrderedDict([
            ('conv0', nn.Conv2d(
                d_model,
                param['conv_kernel_num'],
                (self.matrixSize[0], 1)
            )),
            ('relu0', nn.ReLu(inplace = False)),
            ('pool0', nn.MaxPool2d((param['maxpool_kernel_size'], 1)))
        ])

        self.poolVer1 = nn.MaxPool2d((1, param['maxpool_kernel_size']))
        self.convVer1 = nn.Conv2d(
            param['conv_kernel_num'],
            param['conv_kernel_num'],
            (int(
            self.matrixSize[1]/pow(param['maxpool_kernel_size'],self.num_layers)
            ), 1)
        )
        self.poolHor1 = nn.MaxPool2d((param['maxpool_kernel_size'], 1))
        self.convHor1 = nn.Conv2d(
            param['conv_kernel_num'],
            param['conv_kernel_num'],
            (1, int(
            self.matrixSize[0]/pow(param['maxpool_kernel_size'],self.num_layers)
            ))
        )

        # Layer set 3
        self.poolVer2 = nn.MaxPool2d((1, param['maxpool_kernel_size']))
        self.convVer2 = nn.Conv2d(
            param['conv_kernel_num'],
            param['conv_kernel_num'],
            (int(
            self.matrixSize[1]/pow(param['maxpool_kernel_size'],self.num_layers)
            ), 1)
        )
        self.poolHor2 = nn.MaxPool2d((param['maxpool_kernel_size'], 1))
        self.convHor2 = nn.Conv2d(
            param['conv_kernel_num'],
            param['conv_kernel_num'],
            (1, int(
            self.matrixSize[0]/pow(param['maxpool_kernel_size'],self.num_layers)
            ))
        )


        ### Same problem as 1D model ###
        # flattenLength = int(featureNum / pow(maxpool_kernel_size, num_layers))
        # self.dense = nn.Linear(flattenLength, densed_size)
        self.dense = nn.LazyLinear(param['densed_size'])
        self.decision = nn.Linear(param['densed_size'], n_class)
        self.optimizer = optim.SGD(self.parameters(), param['learning_rate'])

    # Overwrite the forward function in nn.Module
    def forward(self, input):
        input = self.reshape(input)
        temp0 = self.poolVer(func.relu(self.convHor(input)))
        temp1 = self.poolHor(func.relu(self.convVer(input)))
        if self.num_layers > 1:
            temp0 = self.poolVer1(func.relu(self.convHor1(temp0)))
            temp1 = self.poolHor1(func.relu(self.convVer1(temp1)))
        if self.num_layers > 2:
            temp0 = self.poolVer2(func.relu(self.convHor2(temp0)))
            temp1 = self.poolHor2(func.relu(self.convVer2(temp1)))
        if self.num_layers > 3:
            raise classifier.Error('CNN Model with more than 3 layer sets')
        temp0 = torch.flatten(temp0, start_dim = 1)
        temp1 = torch.flatten(temp1, start_dim = 1)
        input = torch.cat((temp0, temp1), dim = 1)
        input = func.relu(self.dense(input))
        input = func.softmax(self.decision(input),dim = 1)
        return input

    # Overwrite the forward function in nn.Module
    def forward(self, input):
        return self.model(input)


    # transform input(1D) into a 2D matrix
    def reshape(self, input):
        return torch.reshape(
            input,
            (
                input.shape[0],
                input.shape[1],
                self.matrixSize[0],
                self.matrixSize[1]
            )
        )


class Unlimited(nn.Module):
    """
    Defining a CNN model treating input as 2D data
    with given hyperparameters
    then using 2 1D convolution kernels to generate layers
    """
    def __init__(self, id, param):
        super().__init__()
        # Initialization
        self.id = id
        self.model_type = 'CNN_Hybrid_Unlimited'
        self.mat_size = param['matrix_size']
        self.num_layers = param['num_layers']
        self.loss_func = nn.CrossEntropyLoss()
        self.pool0 = nn.MaxPool2d((1, param['maxpool_kernel_size']))
        self.pool1 = nn.MaxPool2d((param['maxpool_kernel_size'], 1))
        dividen = pow(param['maxpool_kernel_size'], self.num_layers)
        self.conv0 = nn.Conv2d(
            1,
            param['conv_kernel_num'],
            (self.mat_size[0],1)
        )
        self.conv0_recur = nn.Conv2d(
            param['conv_kernel_num'],
            param['conv_kernel_num'],
            (1, max(1, int(self.mat_size[0] / dividen)))
        )
        self.conv1 = nn.Conv2d(
            1,
            param['conv_kernel_num'],
            (1,self.mat_size[1])
        )
        self.conv1_recur = nn.Conv2d(
            param['conv_kernel_num'],
            param['conv_kernel_num'],
            (max(1, int(self.mat_size[1] / dividen)), 1)
        )

        ### Same problem as 1D model ###
        # flattenLength = int(featureNum / pow(maxpool_kernel_size, num_layers))
        # self.dense = nn.Linear(flattenLength, densed_size)

        self.dense = nn.LazyLinear(param['densed_size'])
        self.decision = nn.Linear(param['densed_size'], n_class)
        self.optimizer = optim.SGD(self.parameters(), param['learning_rate'])

    # Overwrite the forward function in nn.Module
    def forward(self, input):
        input = self.reshape(input)
        temp0 = self.pool0(func.relu(self.conv0(input)))
        for i in range(self.num_layers - 1):
            temp0 = self.pool0(func.relu(self.conv0_recur(temp0)))

        temp0 = torch.flatten(temp0, start_dim = 1)

        temp1 = self.pool1(func.relu(self.conv1(input)))
        for i in range(self.num_layers - 1):
            temp1 = self.pool1(func.relu(self.conv1_recur(temp1)))
        temp1 = torch.flatten(temp1, start_dim = 1)
        input = torch.cat((temp0, temp1), dim = 1)
        input = func.relu(self.dense(input))
        input = func.softmax(self.decision(input),dim = 1)
        return input

    # transform input(1D) into a 2D matrix
    def reshape(self, input):
        return torch.reshape(input, (input.shape[0], input.shape[1],
                                    self.mat_size[0], self.mat_size[1]))



class Make(classifier.Make_Template):
    """
    Analysis the performances of CNN based approaches
    with different hyperparameters
    Find the top settings to build CNN
    """
    def __init__(self, config, cpu_mode, grp_amount):
        self.configs = config
        self.__check_input_matrix_size(grp_amount)
        self.models = []
        self.cpu_mode = cpu_mode

    # Perform classifier training process for given times
    def train(self, dataSets, test_split_set):
        num_features = len(dataSets.dataTest[0])
        for id in self.configs:
            if self.configs[id]['config']['num_layers'] < 3:
                model = Limited(id, self.configs[id]['config'])
            else:
                model = Unlimited(id, self.configs[id]['config'])
            epoch = self.configs[id]['epoch']
            batch_size = self.configs[id]['batch_size']
            self._train_torch(epoch, batch_size, model, dataSets)
            model_record = self._evaluate_torch(
                model,
                dataSets.dataTest,
                dataSets.labelTest,
                test_split_set
            )
            self.models.append(model_record)

    # Check whether matrix sizes are reasonable or not
    def __check_input_matrix_size(self, grp_amount):
        matrix_dim = int(math.sqrt(grp_amount))
        # m is square shaped data dimmensions
        square_size = [matrix_dim, matrix_dim]
        for id in self.configs:
            mat_size = self.configs[id]['config']['matrix_size']
            if mat_size is not None:
                if mat_size[0] * mat_size[1] != grp_amount:
                    warn('Ignored illegal matrixsize config:' + str(mat_size))
                    self.configs[id]['config']['matrix_size'] = square_size

            elif mat_size is None:
                warn('No valid matrix size in config')
                warn('Using 1:1 matrix size: ' + str(idealMatSize))
                self.configs[id]['config']['matrix_size'] = square_size

            if len(mat_size) != 2:
                warn('No valid matrix size in config')
                warn('Using 1:1 matrix size: ' + str(idealMatSize))
                self.configs[id]['config']['matrix_size'] = square_size
