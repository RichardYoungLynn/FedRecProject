import math
import random
import time

import numpy as np
import copy
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn, autograd
import torch.nn.functional as F
from sklearn.cluster import KMeans
from pulp import *

from Nets import ServerFCFModel


class Server(object):
    def __init__(self, args, test_dataset, candidates):
        self.args = args
        self.participants = []
        self.candidates = candidates
        self.args = args
        self.test_dataset = test_dataset
        if args.model == 'fcf':
            self.server_fcf_model = ServerFCFModel(args.server_lr, test_dataset.shape[1], args.feature_num).to(
                args.device)

    def reset(self):
        self.participants = []

    def get_global_model(self):
        return self.global_model.state_dict()

    def client_selection(self):
        if self.args.client_selection == 'random':
            participant_ids = np.random.choice(self.args.candidate_num, self.args.participant_num, replace=False)
            for participant_id in participant_ids:
                self.participants.append(self.candidates[participant_id])

    def init_aggregation_weights(self):
        aggregation_weights = np.ones(self.args.participant_num, dtype='float64')
        for i in range(len(self.participants)):
            aggregation_weights[i] = self.participants[i].get_data_num()
        return aggregation_weights

    def model_aggregation(self, local_models, aggregation_weights):
        global_model = copy.deepcopy(local_models[0])
        for k in global_model.keys():
            for i in range(0, len(local_models)):
                if i == 0:
                    global_model[k] = global_model[k] * aggregation_weights[i]
                else:
                    global_model[k] += local_models[i][k] * aggregation_weights[i]
            global_model[k] = torch.div(global_model[k], sum(aggregation_weights))
        self.global_model.load_state_dict(global_model)

    def test(self):
        self.global_model.eval()
        test_loss = 0
        correct = 0
        for batch_id, (data, target) in enumerate(self.test_dataloader):
            data, target = data.to(self.args.device), target.to(self.args.device)
            log_probs = self.global_model(data)
            test_loss += self.loss_func(log_probs, target).item()
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            y_pred = torch.squeeze(y_pred)
            correct += torch.eq(y_pred, target).sum().float().item()
        test_loss /= len(self.test_dataloader)
        test_accuracy = 100.0 * correct / len(self.test_dataloader.dataset)
        return test_accuracy, test_loss

    def train(self):
        for epoch_index in range(self.args.epochs):
            test_accuracy, test_loss = self.test()
            self.reset()
            self.client_selection()
            print('epoch {} test accuracy {:.4f} test loss {:.4f}'.format(epoch_index, test_accuracy, test_loss))
            self.global_model.train()
            local_models = []
            local_losses = []
            for participant in self.participants:
                local_model, local_loss = participant.train(self.get_global_model())
                local_models.append(local_model)
                local_losses.append(local_loss)
            aggregation_weights = self.init_aggregation_weights()
            self.model_aggregation(local_models, aggregation_weights)

    def client_update(self, user_id):
        optimizer = torch.optim.SGD(self.participants[user_id].parameters(), lr=self.args.client_lr)
        for _ in range(self.args.local_epochs):
            optimizer.zero_grad()
            loss = self.participants[user_id].get_local_model().calculate_loss()
            loss.backward()
            optimizer.step()
        return self.participants[user_id].item_factor.grad.clone().detach()

    def global_update(self):
        grads = []
        for uid in range(self.client_num):
            grads.append(self.client_update(uid))
            self.clients[uid].add_his_grad(grads[-1])
            torch.cuda.empty_cache()
        # update the alpha
        if self.detection_alg:
            # pass    # TODO: implement shilling attack detection algorithm
            self.FoolsGold()
            for i in range(len(grads)):
                grads[i] *= self.alpha[i]
        grads = sum(grads) / self.client_num
        self.server_model.update(grads)
        self.broad_cast()

    def calculate_rmse(self):
        predicts = []
        for participant in self.participants:
            predicts.append(participant.forward(self.server_fcf_model.get_item_factor()).cpu().detach())
        loss = 0.0
        for (user_id, item_id, target) in self.test_dataset:
            loss += (predicts[user_id][0][user_id] - target) ** 2
        loss /= float(len(self.test_dataset))
        rmse = np.sqrt(loss)
        return rmse
