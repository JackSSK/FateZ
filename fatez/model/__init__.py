#!/usr/bin/env python3
"""
This folder contains machine learning models.

author: jy, nkmtmsys
"""
import numpy as np



class Error(Exception):
	"""
    Error handling
	"""
	pass



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
