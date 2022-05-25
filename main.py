import math
import time

import torch
from torchvision import datasets, transforms
import numpy as np
import random
import ssl
import os
from torch import nn

from Server import Server
from Client import Client
from Options import args_parser
from DataProcess import load_dataset
from Nets import ServerFCFModel, NCF


if __name__ == '__main__':
    args = args_parser()
    # args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    if args.dataset == 'ml-100k':
        train_dataset, test_dataset = load_dataset(args)
    elif args.dataset == 'ml-1m':
        train_dataset, test_dataset, test_negatives = load_dataset(args)
    candidates = []
    # for user_id in range(args.candidate_num):
    #     candidates.append(Client(args, train_dataset[user_id]))
    # server = Server(args, test_dataset, test_negatives, candidates)
    # server.train()

    server = Server(args, test_dataset, test_negatives, candidates)
    server.train_centralized(train_dataset)

