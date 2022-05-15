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
        train_dataset, test_dataset, mask = load_dataset(args)

    # dataset_train, dataset_test, num_shards, num_imgs = init_dataset(args)
    # candidate_filename = "logs/2022_05_02/" + args.dataset + "/candidates.txt"
    # candidates = load_candidate_noniid(args, dataset_train, num_shards, num_imgs, candidate_filename)
    # server = Server(args, dataset_test, candidates)
    # server.train()
