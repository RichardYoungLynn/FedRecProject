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
        self.participants = [-1] * self.args.candidate_num
        self.candidates = candidates
        self.args = args
        self.test_dataset = test_dataset
        if args.model == 'fcf':
            self.server_fcf_model = ServerFCFModel(args).to(args.device)

    def reset(self):
        self.participants = [-1] * self.args.candidate_num

    def client_selection(self):
        if self.args.client_selection == 'random':
            participant_ids = np.random.choice(self.args.candidate_num, self.args.participant_num, replace=False)
            for participant_id in participant_ids:
                self.participants[participant_id] = self.candidates[participant_id]

    def broad_cast_model(self):
        item_factor = self.server_fcf_model.get_item_factor()
        for participant in self.participants:
            if participant == -1:
                continue
            participant.set_model(item_factor)

    def train(self):
        for epoch_index in range(self.args.epochs):
            self.reset()
            self.client_selection()
            self.broad_cast_model()
            rmse = self.test()
            gradients = []
            for participant in self.participants:
                if participant == -1:
                    continue
                gradient = participant.train()
                gradients.append(gradient)
                torch.cuda.empty_cache()
            gradients = sum(gradients) / self.args.participant_num
            self.server_fcf_model.update(gradients)
            print('epoch {} rmse {:.4f}'.format(epoch_index, rmse))

    def test(self):
        predicts = []
        for participant in self.participants:
            if participant == -1:
                predicts.append(-1)
            else:
                predicts.append(participant.predict(self.server_fcf_model.get_item_factor()).cpu().detach())
        loss = 0.0
        for (user_id, item_id, target) in self.test_dataset:
            if self.participants[user_id] == -1:
                continue
            loss += (predicts[user_id][0][user_id] - target) ** 2
        loss /= float(len(self.test_dataset))
        rmse = np.sqrt(loss)
        return rmse
