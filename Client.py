import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import random
import copy
import numpy as np

from Nets import LocalFCFModel


class Client(object):
    def __init__(self, args, user_rating):
        self.args = args
        if args.model == 'fcf':
            self.local_fcf_model = LocalFCFModel(args, user_rating).to(args.device)

    def set_model(self, item_factor):
        self.local_fcf_model.set_item_factor(item_factor)

    def train(self):
        optimizer = torch.optim.SGD(self.local_fcf_model.parameters(), lr=self.args.client_lr)
        for _ in range(self.args.local_epochs):
            optimizer.zero_grad()
            loss = self.local_fcf_model.calculate_loss()
            loss.backward()
            optimizer.step()
        return self.local_fcf_model.item_factor.grad.clone().detach()

    def predict(self, item_factor):
        return self.local_fcf_model.forward(item_factor)

