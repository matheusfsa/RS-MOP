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

class CB_MOP(Recommender):
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


    def __init__(self, n_ratings):
        self.n_ratings = n_ratings
        self.load_ratings(n_ratings)
        self.fit()
    
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
        self.df_movies, _ = dataset_word2vec(['genres', 'rating', 'runtimes', 'year'], op='sum', n_features=300)
        self.df_movies.set_index('movieId')
    
    def split(self, user, df_movies):
        data = {}
        with open('./app/datasets/index.txt') as json_file:
            data = json.load(json_file)
        user_index = data[str(user)]
        train = df_movies.loc[user_index['train'], :]
        test = df_movies.loc[user_index['test'], :]
        return train.dropna(subset=['rating_user']), test.dropna(subset=['rating_user'])

    def set_user(self, user):
        data = {}
        with open('./app/datasets/index.txt') as json_file:
            data = json.load(json_file)
        last_user = data.get('last_user',-1)  
        data['last_user'] = last_user + 1 
        self.last_user = last_user
        with open('./app/datasets/index.txt', 'w') as outfile:
            json.dump(data, outfile) 
        self.df_movies_u = dataset_ratings_user(self.df_movies, df_ratings=self.df_ratings, user=user)
        self.train, self.test = self.split(user, self.df_movies_u)
        self.X_train = self.train.drop(columns=['rating_user','title'])
        self.y_train = self.train['rating_user']
        self.X_test = self.test.drop(columns=['rating_user','title'])
        self.y_test = self.test['rating_user']
        self.model = RidgeCV(cv=5).fit(self.X_train, self.y_train)
        self.user = user

        

    def evaluate_solutions(self, solutions, data, *args, **kwargs):
        y = np.zeros((solutions.shape[0], 3))
        y[:, 0] = -self.model.predict(solutions) 
        mating = kwargs.get('mating', None)
        if mating is not None:
            y[:, 1] = -self.get_diversity_offspring(mating, solutions)
        else:
            y[:, 1] = -self.get_diversity(solutions)
        y[:, 2] = -self.get_novelty(solutions, data)
        
        return y


    def get_diversity_offspring(self, mating, solutions):
        a = np.append(solutions, mating, axis=0)
        sim = cosine_similarity(a,a)
        np.fill_diagonal(sim, 0)
        n = solutions.shape[0] - 1 + mating.shape[0]
        return ((1 - sim).sum(axis=1) * 1/n)[:solutions.shape[0]].reshape(1, -1)

    def get_diversity(self, solutions):
        sim = cosine_similarity(solutions, solutions)
        np.fill_diagonal(sim, 0)
        return (1 - sim).sum(axis=1) * 1/(solutions.shape[0] - 1)


    def get_novelty(self, solutions, data):
        sim = cosine_similarity(solutions, data)
        return (1-sim).max(axis=1) #* 1/(data.shape[0] - 1)

    def filtering(self, solutions):
        sim = -cosine_similarity(solutions, self.X_test.values)
        pop_index = []
        def get_best(bests, pop_index):
            for x in bests:
                    if x not in pop_index:
                        return x
        for i in range(solutions.shape[0]):
            bests = sim[i, :].argsort() 
            pop_index.append(get_best(bests, pop_index))
        res = self.test.iloc[pop_index, :].drop_duplicates()
        data = {}
        if self.last_user >= 0:
            with open('./app/recomendacoes/cb-moea-recomendacoes.txt') as json_file:
                data = json.load(json_file)
        data[str(self.user)] = res.index.values.tolist()
        with open('./app/recomendacoes/cb-moea-recomendacoes.txt', 'w') as outfile:
            json.dump(data, outfile)
        return res.index

    