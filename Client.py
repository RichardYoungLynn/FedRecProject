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
            self.local_fcf_model = LocalFCFModel(args.client_lr, args.feature_num,
                                                 user_rating, args.Lambda).to(args.device)

    def get_data_num(self):
        return self.data_num

    def get_local_model(self):
        if self.args.model == 'fcf':
            return self.local_fcf_model

    def train(self, global_model):
        self.local_model.load_state_dict(copy.deepcopy(global_model))
        self.local_model.train()
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.args.learning_rate,
                                    momentum=self.args.momentum)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.args.learning_rate_decay)
        epoch_loss = []
        for epoch in range(self.args.local_epochs):
            batch_loss = []
            for batch_id, (images, labels) in enumerate(self.train_dataloader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                self.local_model.zero_grad()
                log_probs = self.local_model(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return self.get_local_model(), sum(epoch_loss) / len(epoch_loss)
