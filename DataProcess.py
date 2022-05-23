import pandas as pd
import numpy as np
import os
import random
import torch
import scipy.sparse as sp
from torch.utils.data import DataLoader, Dataset


def load_dataset(args):
    if args.dataset == 'ml-100k':
        rating_matrix = load_movielens_dataset()
        train_dataset, test_dataset = split_movielens_dataset(rating_matrix)
        train_dataset = torch.tensor(train_dataset).to(args.device)
        return train_dataset, test_dataset
    elif args.dataset == 'ml-1m':
        file_path = 'datasets/movielens/' + args.dataset + '/' + args.dataset
        train_dataset = load_rating_file_as_matrix(file_path + ".train.rating")
        test_dataset = load_rating_file_as_list(file_path + ".test.rating")
        test_negatives = load_negative_file(file_path + ".test.negative")
        return train_dataset, test_dataset, test_negatives
        # assert len(testRatings) == len(testNegatives)
        # num_users, num_items = trainMatrix.shape


def load_movielens_dataset(type='ml-100k'):
    file_path = 'datasets/movielens/' + type + '/'
    if type == 'ml-100k':
        if os.path.exists(file_path + 'rating.npy'):
            rating_matrix = np.load(file_path + 'rating.npy')
        else:
            data = pd.read_csv(file_path + 'u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'time'])
            rating_matrix = np.zeros((943, 1682), dtype=np.float32)
            for index, row in data.iterrows():
                row_id = int(row['user_id']) - 1
                col_id = int(row['item_id']) - 1
                rating_matrix[row_id][col_id] = row['rating']
            np.save(file_path + 'rating.npy', rating_matrix)
        return rating_matrix
    elif type == 'ml-1m':
        ratings_df = pd.read_csv(file_path + 'ratings.dat',
                                 sep='::',
                                 names=['userId', 'movieId', 'rating', 'Time'])
        movies_df = pd.read_csv(file_path + 'movies.dat',
                                sep='::',
                                names=['movieId', 'name', 'type'])
    elif type == 'ml-latest-small':
        ratings_df = pd.read_csv(file_path + 'ratings.csv')
        movies_df = pd.read_csv(file_path + 'movies.csv')
    else:
        print("ERROR: data file NOT FOUND")
        return None
    user_num = ratings_df['userId'].drop_duplicates().size
    num_movies = movies_df.index.size
    movie_id_mapping = {}
    id_count = 1
    for i in movies_df['movieId']:
        movie_id_mapping[int(i)] = int(id_count)
        id_count += 1
    rating = np.zeros((user_num, num_movies), dtype=np.float32)
    for index, row in ratings_df.iterrows():
        row_id = int(row['userId']) - 1
        col_id = movie_id_mapping[int(row['movieId'])] - 1
        rating[row_id][col_id] = row['rating']
    np.save(file_path + 'rating.npy', rating)
    return rating


def split_movielens_dataset(rating_matrix):
    user_rating_num = ((rating_matrix > 0) * 1.0).sum(1)
    test_dataset = []
    for user_id in range(rating_matrix.shape[0]):
        all_index = list(np.where(rating_matrix[user_id] > 0)[0])
        test_index = random.sample(all_index, int(0.2 * user_rating_num[user_id]))
        for item_id in test_index:
            test_dataset.append((user_id, item_id, rating_matrix[user_id][item_id]))
            rating_matrix[user_id][item_id] = 0
    train_dataset = rating_matrix
    return train_dataset, test_dataset


def load_rating_file_as_list(filename):
    ratingList = []
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            user, item = int(arr[0]), int(arr[1])
            ratingList.append([user, item])
            line = f.readline()
    return ratingList


def load_negative_file(filename):
    negativeList = []
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            negatives = []
            for x in arr[1:]:
                negatives.append(int(x))
            negativeList.append(negatives)
            line = f.readline()
    return negativeList


def load_rating_file_as_matrix(filename):
    '''
    Read .rating file and Return dok matrix.
    The first line of .rating file is: num_users\t num_items
    '''
    # Get number of users and items
    num_users, num_items = 0, 0
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            u, i = int(arr[0]), int(arr[1])
            num_users = max(num_users, u)
            num_items = max(num_items, i)
            line = f.readline()
    # Construct matrix
    mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
            if (rating > 0):
                mat[user, item] = 1.0
            line = f.readline()
    return mat


def get_train_instances(args, train_dataset):
    user_input, item_input, labels = [], [], []
    for (u, i) in train_dataset.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(args.num_neg):
            j = np.random.randint(args.num_items)
            # while train_dataset.has_key((u, j)):
            while (u, j) in train_dataset.keys():
                j = np.random.randint(args.num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels


class DatasetSplit(Dataset):
    def __init__(self, args, user_input, item_input, labels):
        self.args = args
        self.user_input = user_input
        self.item_input = item_input
        self.labels = labels

    def __len__(self):
        return len(self.user_input)

    def __getitem__(self, index):
        user_input, item_input, label = self.user_input[index], self.item_input[index], self.labels[index]
        # return np.eye(self.args.candidate_num, dtype=int)[user_input], np.eye(self.args.num_items, dtype=int)[
        #     item_input], np.array(label)
        return user_input, item_input, label


if __name__ == '__main__':
    mat = load_rating_file_as_matrix('datasets/movielens/ml-1m/ml-1m.train.rating')
    print(mat.shape)
    print("yyc yyds:")
    # print(mat.has_key((0,32)))
    # print(mat.keys())
