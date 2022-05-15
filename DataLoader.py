import pandas as pd
import numpy as np
import os


def load_data(args):
    if args.dataset == 'movielens':
        load_movielens_data()


def load_movielens_data(type='ml-100k'):
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
