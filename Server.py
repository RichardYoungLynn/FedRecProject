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
import heapq

from Nets import ServerFCFModel, NCF
from DataProcess import DatasetSplit, get_train_instances


class Server(object):
    def __init__(self, args, test_dataset, test_negatives, candidates):
        self.args = args
        self.participants = [-1] * self.args.candidate_num
        self.candidates = candidates
        self.args = args
        if args.model == 'fcf':
            self.server_fcf_model = ServerFCFModel(args).to(args.device)
        elif args.model == 'ncf':
            self.global_model = NCF(args).to(args.device)
            self.test_dataset = test_dataset
            self.test_negatives = test_negatives
            # self.test_dataset = torch.tensor(np.array(test_dataset)).to(args.device)
            # self.test_negatives = torch.tensor(np.array(test_negatives)).to(args.device)
            self.loss_func = nn.BCELoss()

    def reset(self):
        self.participants = [-1] * self.args.candidate_num

    def get_global_model(self):
        return self.global_model.state_dict()

    def client_selection(self):
        if self.args.client_selection == 'random':
            participant_ids = np.random.choice(self.args.candidate_num, self.args.participant_num, replace=False)
            for participant_id in participant_ids:
                self.participants[participant_id] = self.candidates[participant_id]

    def init_aggregation_weight(self):
        aggregation_weight = np.array([1] * self.args.participant_num, dtype='float64')
        j = 0
        for i in range(len(self.participants)):
            if self.participants[i] == -1:
                continue
            if self.args.client_selection == 'ramdom':
                aggregation_weight[j] = self.participants[i].get_data_num()
            j += 1
        return aggregation_weight

    def model_aggregation(self, local_models, aggregation_weight):
        global_model = copy.deepcopy(local_models[0])
        for k in global_model.keys():
            for i in range(0, len(local_models)):
                if i == 0:
                    global_model[k] = global_model[k] * aggregation_weight[i]
                else:
                    global_model[k] += local_models[i][k] * aggregation_weight[i]
            global_model[k] = torch.div(global_model[k], sum(aggregation_weight))
        return global_model

    # def broad_cast_model(self):
    #     item_factor = self.server_fcf_model.get_item_factor()
    #     for participant in self.participants:
    #         if participant == -1:
    #             continue
    #         participant.set_model(item_factor)
    #
    # def train(self):
    #     for epoch_index in range(self.args.epochs):
    #         self.reset()
    #         self.client_selection()
    #         self.broad_cast_model()
    #         rmse = self.test()
    #         gradients = []
    #         for participant in self.participants:
    #             if participant == -1:
    #                 continue
    #             gradient = participant.train()
    #             gradients.append(gradient)
    #             torch.cuda.empty_cache()
    #         gradients = sum(gradients) / self.args.participant_num
    #         self.server_fcf_model.update(gradients)
    #         print('epoch {} rmse {:.4f}'.format(epoch_index, rmse))
    #
    # def test(self):
    #     predicts = []
    #     for participant in self.participants:
    #         if participant == -1:
    #             predicts.append(-1)
    #         else:
    #             predicts.append(participant.predict(self.server_fcf_model.get_item_factor()).cpu().detach())
    #     loss = 0.0
    #     for (user_id, item_id, target) in self.test_dataset:
    #         if self.participants[user_id] == -1:
    #             continue
    #         loss += (predicts[user_id][0][user_id] - target) ** 2
    #     loss /= float(len(self.test_dataset))
    #     rmse = np.sqrt(loss)
    #     return rmse

    def train(self):
        self.global_model.train()
        for epoch_index in range(self.args.epochs):
            (hits, ndcgs) = self.test()
            print("epoch:", epoch_index, "hits:", np.array(hits).mean(), "ndcgs:", np.array(ndcgs).mean())
            self.reset()
            self.client_selection()
            local_models = []
            local_losses = []
            for participant in self.participants:
                if participant == -1:
                    continue
                local_model, local_loss = participant.train(copy.deepcopy(self.get_global_model()))
                local_models.append(local_model)
                local_losses.append(local_loss)
            aggregation_weight = self.init_aggregation_weight()
            self.global_model.load_state_dict(self.model_aggregation(local_models, aggregation_weight))


    def eval_one_rating(self, idx):
        rating = self.test_dataset[idx]
        items = self.test_negatives[idx]
        u = rating[0]
        gtItem = rating[1]
        items.append(gtItem)
        # Get prediction scores
        map_item_score = {}
        users = np.full(len(items), u)
        users = torch.tensor(np.array(users), dtype=torch.long).to(self.args.device)
        items = torch.tensor(np.array(items), dtype=torch.long).to(self.args.device)
        predictions = self.global_model(users, items)
        for i in range(len(items)):
            item = items[i]
            map_item_score[item] = predictions[i]
        # items.pop()

        # Evaluate top rank list
        ranklist = heapq.nlargest(self.args.topK, map_item_score, key=map_item_score.get)
        hr = self.getHitRatio(ranklist, gtItem)
        ndcg = self.getNDCG(ranklist, gtItem)
        return (hr, ndcg)

    def getHitRatio(self, ranklist, gtItem):
        for item in ranklist:
            if item == gtItem:
                return 1
        return 0

    def getNDCG(self, ranklist, gtItem):
        for i in range(len(ranklist)):
            item = ranklist[i]
            if item == gtItem:
                return math.log(2) / math.log(i + 2)
        return 0

    def test(self):
        self.global_model.eval()
        hits, ndcgs = [], []
        for idx in range(len(self.test_dataset)):
            (hr, ndcg) = self.eval_one_rating(idx)
            hits.append(hr)
            ndcgs.append(ndcg)
        return (hits, ndcgs)
