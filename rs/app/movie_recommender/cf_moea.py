from .base import Recommender
import pandas as pd
import numpy as np
# Libraries for text preprocessing
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
#nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
from datasets import dataset_word2vec, dataset_ratings_user
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import RidgeCV
import json

class CF_MOEA(Recommender):
    df_movies = None
    df_movies_u = None
    model = None
    X_train = None
    y_train = None
    X_test = None
    y_test = None
    train = None
    test = None
    last_user = -1
    rs = None
    df_ratings_wt = None
    


    def __init__(self, n_ratings):
        self.n_ratings = n_ratings
        self.load_ratings(n_ratings)

    def reset_users(self):
        data = {}
        with open('./app/datasets/index.txt') as json_file:
            data = json.load(json_file)
        data['last_user'] = -1  
        with open('./app/datasets/index.txt', 'w') as outfile:
            json.dump(data, outfile) 
        
    def min(self):
        return np.zeros(15, dtype=np.uint8)
        
    def max(self):
        return np.ones(15, dtype=np.uint8)*(self.test.shape[0]-1)

    def n_users(self):
        data = {}
        with open('./app/datasets/index.txt') as json_file:
            data = json.load(json_file)
        users = list(data.keys())
        self.last_user = data.get('last_user', -1)
        n = len(users)
        return n - (self.last_user + 1)

    def users(self):
        data = {}
        with open('./app/datasets/index.txt') as json_file:
            data = json.load(json_file)
        
        self.last_user = data.get('last_user',  None)
        if self.last_user:
            del data['last_user']
        else:
            self.last_user = -1
        users = list(data.keys())
        return users[(self.last_user + 1):]

    def get_user(self):
        data = {}
        with open('./app/datasets/index.txt') as json_file:
            data = json.load(json_file)
        users = list(data.keys())
        last_user = data.get('last_user',-1)
        user = users[last_user + 1]
        return int(user)

    def fit(self):
        self.rs = self.get_all_rij(self.train, self.test, self.user, 10)
    
    def split(self, user, df_movies):
        data = {}
        with open('./app/datasets/index.txt') as json_file:
            data = json.load(json_file)
        user_index = data[str(user)]
        df_movies_user = self.df_ratings[self.df_ratings['userId'] == int(user)]
        df_movies_user =  df_movies_user.set_index('movieId')
        train = df_movies_user.loc[user_index['train'], :]
        test = df_movies_user.loc[user_index['test'], :]
        return train, test

    def set_user(self, user):
        data = {}
        with open('./app/datasets/index.txt') as json_file:
            data = json.load(json_file)
        last_user = data.get('last_user',-1)  
        data['last_user'] = last_user + 1 
        self.last_user = last_user
        with open('./app/datasets/index.txt', 'w') as outfile:
            json.dump(data, outfile) 
        self.train, self.test = self.split(user, self.df_movies_u)   
        self.user = user
        self.fit()
    
    def get_all_rij(self, train, test, user, n):
        movies_user_train = train.index
        movies_user_test = test.index
        df_ratings_wt = self.df_ratings[~self.df_ratings.index.isin(movies_user_test)] 
        self.df_ratings_wt = df_ratings_wt
        df_ratings_user = self.df_ratings[self.df_ratings['userId'] == user]
        itens_rating = pd.pivot_table(df_ratings_wt[df_ratings_wt['movieId'].isin(df_ratings_user['movieId'])], values='rating', index='movieId', columns='userId').fillna(0)
        sim = pd.DataFrame(cosine_similarity(itens_rating, itens_rating), index=itens_rating.index, columns=itens_rating.index)
        item_item_u = sim.loc[movies_user_test, movies_user_train]
        b_u = train.rating.std()
        b_i = df_ratings_wt[['movieId', 'rating']].groupby(['movieId']).std().fillna(0)
        beta = self.df_ratings.rating.std()
        b_ui = beta + b_u + b_i
        rs = pd.DataFrame(np.zeros(movies_user_test.shape[0]), index=movies_user_test, columns=['rating'])
        for i in movies_user_test:
            rs.loc[i, 'rating'] = self.rij(item_item_u, i, train, b_ui, n)
        return rs
    
    def rij(self, item_item_u, i, train, b_ui, n):
        s_ij = item_item_u.loc[i, :].sort_values(ascending=False)[:n]
        r_ij = train.loc[s_ij.index,:].rating
        b_uj = b_ui.loc[s_ij.index,:].rating
        num = s_ij*(r_ij - b_uj)
        den = s_ij.abs().sum()
        return b_ui.loc[i, :].values[0] + (num.sum()/den.sum())

    def popularity(self, movieId):
        mean = self.df_ratings_wt[self.df_ratings_wt.movieId==movieId].rating.mean()
        std = self.df_ratings_wt[self.df_ratings_wt.movieId==movieId].rating.std() 
        if np.isnan(std):
            std = 0.0
        f2 = (1/ (mean*((std+1)**2))) 
        return f2

    def evaluate_solution(self, solution):
        f1 = 0
        f2 = 0
        n = len(solution)
        for item in solution:
            index = self.test.index[item]
            f1 += self.rs.loc[index, 'rating']/n
            f2 += self.popularity(index)/n
        return [f1, f2]

    
    def filtering(self, solutions):
        solution = solutions[0, :]
        pop_index = [int(self.test.index[item]) for item in solution]
        data = {}
        if self.last_user >= 0:
            with open('./app/recomendacoes/cf-moea-recomendacoes.txt') as json_file:
                data = json.load(json_file)
        data[str(self.user)] = pop_index
        with open('./app/recomendacoes/cf-moea-recomendacoes.txt', 'w') as outfile:
            json.dump(data, outfile)
        return pop_index

    