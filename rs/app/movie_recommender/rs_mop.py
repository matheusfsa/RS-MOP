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
from app.datasets import dataset_word2vec, dataset_ratings_user, dataset_tfid
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import RidgeCV
import json
from sklearn.metrics.pairwise import cosine_similarity
import itertools
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
    sim_users = None
    combination = 16
    n_features = 2000
    index_src = './app/datasets/index.txt'
    name = 'tfid'
    new_feats = None
    end_user = 1329
    
    def __init__(self, n_ratings):
        self.experiments_src = './app/recomendacoes/' + self.name + '-'+str(self.n_features) + '-'+str(self.combination) + '.txt'
        all_feats = ['genres', 'rating', 'runtimes', 'year']
        self.combinations = []
        for i in range(5):
            for c in itertools.combinations(all_feats,i):
                self.combinations.append(list(c))
        print(self.combinations)
        #features = ['genres', 'rating', 'runtimes', 'year']
        self.features = self.combinations[self.combination-1]

        self.n_ratings = n_ratings
        self.load_ratings(n_ratings)
        self.fit()
    
    def reset_users(self):
        data = {}
        with open(self.index_src) as json_file:
            data = json.load(json_file)
        data['last_user'] = -1  
        with open(self.index_src, 'w') as outfile:
            json.dump(data, outfile) 

    def n_users(self):
        data = {}
        with open(self.index_src) as json_file:
            data = json.load(json_file)
        users = list(data.keys())
        self.last_user = data.get('last_user', -1)
        n = len(users)
        return n - (self.last_user + 1)

    
    def min(self):
        return np.min(self.X_test, axis=0)
    
    def max(self):
        return np.max(self.X_test, axis=0)
    
    def n_variables(self):
        return self.X_test.shape[1]

    def users(self):
        data = {}
        with open(self.index_src) as json_file:
            data = json.load(json_file)
        
        self.last_user = data.get('last_user',  None)
        if self.last_user:
            del data['last_user']
        else:
            self.last_user = -1
            del data['last_user']
        users = list(data.keys())
        self.end_user = int(users[-1])
        self.reset_users()
        return users[(self.last_user + 1):]

    def get_user(self):
        data = {}
        with open(self.index_src) as json_file:
            data = json.load(json_file)
        users = list(data.keys())
        last_user = data.get('last_user',-1)
        user = users[last_user + 1]
        return int(user)

    def fit(self):
        if self.new_feats is None:
            self.df_movies, self.new_feats = dataset_tfid(self.features, op='sum', n_features=self.n_features, n_words=self.n_features)
        else:
            self.df_movies, self.new_feats = dataset_tfid(self.features, op='sum', n_features=self.n_features, new_feats=self.new_feats,n_words=self.n_features)
    
    def split(self, user, df_movies):
        data = {}
        with open(self.index_src) as json_file:
            data = json.load(json_file)
        user_index = data[str(user)]
        train = df_movies.loc[user_index['train'], :]
        test = df_movies.loc[user_index['test'], :]
        return train.dropna(subset=['rating_user']), test.dropna(subset=['rating_user'])

    def set_sim_users(self, n):
        test = self.X_test.index
        df_ratings = self.df_ratings
        df_ratings_wt = df_ratings[~df_ratings.movieId.isin(test)] 
        df_ratings_user = df_ratings[df_ratings['userId'] == int(self.user)]
        itens_rating = pd.pivot_table(df_ratings_wt[df_ratings_wt['movieId'].isin(df_ratings_user['movieId'])], values='rating', index='userId', columns='movieId').fillna(0)
        user_ratings = itens_rating[itens_rating.index == int(self.user)]
        rest_ratings = itens_rating[itens_rating.index != int(self.user)]
        sim = pd.DataFrame(cosine_similarity(rest_ratings, user_ratings), index=rest_ratings.index, columns=user_ratings.index)
        sim = sim.sort_values(by=int(self.user), ascending=False)[int(self.user)]
        sim = sim[:n]
        self.sim_users = sim/sim.sum()

    def get_model_from_user(self, user):
        df_movies_u = dataset_ratings_user(self.df_movies, df_ratings=self.df_ratings, user=user)
        train, test = self.split(user, df_movies_u)
        X_train = train.drop(columns=['rating_user','title'])
        y_train = train['rating_user']
        X_test = test.drop(columns=['rating_user','title'])
        y_test = test['rating_user']
        return RidgeCV(cv=5).fit(X_train, y_train)

    def set_user(self, user):
        data = {}
        with open(self.index_src) as json_file:
            data = json.load(json_file)
        last_user = data.get('last_user',-1)  
        data['last_user'] = last_user + 1 
        self.last_user = last_user
        with open(self.index_src, 'w') as outfile:
            json.dump(data, outfile) 
        self.df_movies_u = dataset_ratings_user(self.df_movies, df_ratings=self.df_ratings, user=user)
        self.train, self.test = self.split(user, self.df_movies_u)
        self.X_train = self.train.drop(columns=['rating_user','title'])
        self.y_train = self.train['rating_user']
        self.X_test = self.test.drop(columns=['rating_user','title'])
        self.y_test = self.test['rating_user']
        self.model = RidgeCV(cv=5).fit(self.X_train, self.y_train)
        self.user = user
        #self.set_sim_users(10)
        #self.models_sim = [(self.get_model_from_user(i), v) for i, v in self.sim_users.iteritems()]

        #print(self.X_test)
        

    def evaluate_solutions(self, solutions, data, *args, **kwargs):
        y = np.zeros((solutions.shape[0], 3))
        y[:, 0] = -self.model.predict(solutions) 
        mating = kwargs.get('mating', None)
        #cf = np.zeros(solutions.shape[0])
        #for model, v in self.models_sim:
        #    cf += model.predict(solutions)*v
        #y[:, 1]  = -cf   
        
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
        try:
            with open(self.experiments_src) as json_file:
                data = json.load(json_file)
        except FileNotFoundError:
            pass
        data[str(self.user)] = res.index.values.tolist()
        with open(self.experiments_src, 'w') as outfile:
            json.dump(data, outfile)
            print('Salvo em:', self.experiments_src)
        #aprint(self.user)
        #if int(self.user) == self.end_user and self.combination <= len(self.combinations):
        #    self.combination += 1
        #    self.experiments_src = './app/recomendacoes/'+ self.name + '-'+str(self.n_features) + '-'+str(self.combination) + '.txt'
        #    self.fit()
        return res.index

    