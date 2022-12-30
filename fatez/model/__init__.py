#!/usr/bin/env python3
"""
This folder contains machine learning models.

author: jy, nkmtmsys
"""
import re
import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np


def Save(model, file_path:str = 'a.model', device:str = 'cpu',):
	"""
	Saving a model

	:param model: the model to save
	:param file_path: model output path
	:param device: device to load model
	"""
	model_type = str(type(model))

	if (re.search(r'torch.nn.modules.', model_type) or
		re.search(r'fatez.model.bert.', model_type) or
		re.search(r'fatez.model.sparse_gat.Spare_GAT', model_type) or
		re.search(r'fatez.model.gat.GAT', model_type)
		):
		torch.save(model.cpu(), file_path)
		model.to(device)
	else:
		raise Error('Not Supporting Save ' + model_type)
	return model

def Load(file_path:str = 'a.model', mode:str = 'torch', device:str = 'cpu',):
	"""
	Loading a model

	:param file_path: path to load model
	:param device: device to load model
	"""
	model = None
	# PyTorch Loading method
	if mode == 'torch':
		model = torch.load(file_path)
		model.to(device)
	else:
		raise Error('Not Supporting Load Mode ' + mode)
	return model



class Error(Exception):
	"""
	Error handling
	"""
	pass



class Binning_Process(nn.Module):
	"""
	Binning data output by GAT.

	We'd better implement this part after having a general idea about the
	overall data distribution of GAT output in real world cases.
	"""

	def __init__(self, n_bin):
		super(Binning_Process, self).__init__()
		self.n_bin = n_bin

	def forward(self, input):
		return input



class LR_Scheduler(object):
	"""
	Automatically adjust learning rate based on learning steps.
	"""

	def __init__(self,
		optimizer,
		n_features:int = None,
		n_warmup_steps:int = None
		):
		"""
		:param optimizer: <Default = None>
			The gradient optimizer.

		:param n_features: <int Default = None>
			The number of expected features in the model input.

		:param n_warmup_steps: <int Default = None>
			Number of warming up steps. Gradually increase learning rate.
		"""

		self.optimizer = optimizer
		self.n_warmup_steps = n_warmup_steps
		self.n_current_steps = 0
		self.init_learning_rate = np.power(n_features, -0.5)

	def step_and_update_lr(self):
		self._update_learning_rate()
		self.optimizer.step()

	def zero_grad(self):
		self.optimizer.zero_grad()

	def _get_lr_scale(self):
		return np.min(
			[
				np.power(self.n_current_steps, -0.5),
				np.power(self.n_warmup_steps, -1.5) * self.n_current_steps
			]
		)

	def _update_learning_rate(self):
		self.n_current_steps += 1
		lr = self.init_learning_rate * self._get_lr_scale()

		for param_group in self.optimizer.param_groups:
			param_group['lr'] = lr
