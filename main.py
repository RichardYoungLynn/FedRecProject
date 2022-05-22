import math
import time

import torch
from torchvision import datasets, transforms
import numpy as np
import random
import ssl

from Server import Server
from Client import Client
from Options import args_parser
from DataProcess import load_dataset

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    if args.dataset == 'movielens':
        train_dataset, test_dataset = load_dataset(args)
        candidates = []
        for user_id in range(args.candidate_num):
            candidates.append(Client(args, train_dataset[user_id]))
        server = Server(args, test_dataset, candidates)
        server.train()

    # data_1 = np.load('dataset_sample/tiktok/train_sample.npy')
    # data_2 = np.load('dataset_sample/tiktok/test_sample.npy')
    # data_3 = np.load('dataset_sample/tiktok/val_sample.npy')
    # data_4 = torch.load('dataset_sample/tiktok/a_feat_sample.pt')
    # data_5 = torch.load('dataset_sample/tiktok/t_feat_sample.pt')
    # data_6 = torch.load('dataset_sample/tiktok/v_feat_sample.pt')
    # data_7 = np.load('dataset_sample/tiktok/user_item_dict_sample.npy')


