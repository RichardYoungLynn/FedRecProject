import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import random
import copy
import numpy as np

from Nets import LocalFCFModel, NCF
from DataProcess import DatasetSplit, get_train_instances


class Client(object):
    def __init__(self, args, train_dataset):
        self.args = args
        if args.model == 'fcf':
            self.local_fcf_model = LocalFCFModel(args, train_dataset).to(args.device)
        elif args.model == 'ncf':
            # self.local_model = NCF(args).to(args.device)
            self.local_model = NCF(args)
            self.local_model = nn.DataParallel(self.local_model)
            self.local_model = self.local_model.cuda()
            self.loss_func = nn.BCELoss().cuda()
            self.user_input, self.item_input, self.labels = get_train_instances(args, train_dataset)
            self.train_dataloader = DataLoader(DatasetSplit(args, self.user_input, self.item_input, self.labels),
                                               batch_size=self.args.local_batch_size, shuffle=True)

    # def set_model(self, item_factor):
    #     self.local_fcf_model.set_item_factor(item_factor)
    #
    # def train(self):
    #     optimizer = torch.optim.SGD(self.local_fcf_model.parameters(), lr=self.args.client_lr)
    #     for _ in range(self.args.local_epochs):
    #         optimizer.zero_grad()
    #         loss = self.local_fcf_model.calculate_loss()
    #         loss.backward()
    #         optimizer.step()
    #     return self.local_fcf_model.get_item_factor().grad.clone().detach()
    #
    # def predict(self, item_factor):
    #     return self.local_fcf_model.forward(item_factor)

    def get_data_num(self):
        return len(self.user_input)

    def get_local_model(self):
        return self.local_model.state_dict()

    def train(self, global_model):
        self.local_model.load_state_dict(global_model)
        self.local_model.train()
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.args.learning_rate,
                                    momentum=self.args.momentum)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.args.learning_rate_decay)
        epoch_loss = []
        for epoch in range(self.args.local_epochs):
            batch_loss = []
            for batch_id, (user_input, item_input, labels) in enumerate(self.train_dataloader):
                # user_input, item_input, labels = user_input.to(self.args.device), item_input.to(self.args.device), labels.to(self.args.device)
                user_input, item_input, labels = user_input.cuda(), item_input.cuda(), labels.cuda()
                self.local_model.zero_grad()
                predict = self.local_model(user_input, item_input)
                loss = self.loss_func(predict, labels.float())
                loss.backward()
                optimizer.step()
                scheduler.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return self.get_local_model(), sum(epoch_loss) / len(epoch_loss)
