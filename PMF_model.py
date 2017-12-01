from __future__ import print_function
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import os
import torch.nn.init as initilize
import pickle
from numpy.random import RandomState
import numpy as np

class PMF(nn.Module):
	def __init__(self, n_users, n_items, n_factors=20, is_sparse=False, no_cuda=None):
		super(PMF, self).__init__()
		self.n_users = n_users
		self.n_items = n_items
		self.n_factors = n_factors
		self.no_cuda = no_cuda
		self.random_state = RandomState(1)

		self.user_embeddings = nn.Embedding(n_users, n_factors, sparse=is_sparse)
		self.user_embeddings.weight.data = torch.from_numpy(0.1 * self.random_state.rand(n_users, n_factors)).float()

		self.item_embeddings = nn.Embedding(n_items, n_factors, sparse=is_sparse)
		self.item_embeddings.weight.data = torch.from_numpy(0.1 * self.random_state.rand(n_items, n_factors)).float()


		self.relu = nn.ReLU()

	def forward(self, users_index, items_index):
		user_h1 = self.user_embeddings(users_index)
		item_h1 = self.item_embeddings(items_index)
		R_h = (user_h1 * item_h1).sum(1)

		return R_h



	def __call__(self, *args):
		return self.forward(*args)


	def predict(self, users_index, items_index):
		preds = self.forward(users_index, items_index)
		return preds
