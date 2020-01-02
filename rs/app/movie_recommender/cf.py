from .base import Recommender
import pandas as pd
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

class CF(Recommender):
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
    user = -1

    def __init__(self, n_ratings):
        self.n_ratings = n_ratings
        self.load_ratings(n_ratings)
    
    def reset_index(self):
        data = {}
        with open('./app/datasets/index.txt') as json_file:
            data = json.load(json_file)
        data['last_user'] = -1
        with open('./app/datasets/index.txt', 'w') as outfile:
            json.dump(data, outfile)
        

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
    
    def set_user(self, user):
        data = {}
        self.user = int(user)
        with open('./app/datasets/index.txt') as json_file:
            data = json.load(json_file)
        last_user = data.get('last_user',-1)  
        data['last_user'] = last_user + 1 
        self.last_user = last_user
        with open('./app/datasets/index.txt', 'w') as outfile:
            json.dump(data, outfile) 
        
        

    def split(self, user):
        df_movies = self.df_ratings[self.df_ratings['userId'] == user]
        df_movies = df_movies.set_index('movieId')
        data = {}
        with open('./app/datasets/index.txt') as json_file:
            data = json.load(json_file)
        user_index = data[str(user)]
        train = df_movies.loc[user_index['train'], :]
        test = df_movies.loc[user_index['test'], :]
        return train, test
    
    def rij(self, item_item_u, i, train, b_ui, n):
        s_ij = item_item_u.loc[i, :].sort_values(ascending=False)[:n]
        r_ij = train.loc[s_ij.index,:].rating
        b_uj = b_ui.loc[s_ij.index,:].rating
        num = s_ij*(r_ij - b_uj)
        den = s_ij.abs().sum()
        return b_ui.loc[i, :].values[0] + (num.sum()/den.sum())
    
    def get_all_rij(self, train, test, user, n):
        movies_user_train = train.index
        movies_user_test = test.index
        df_ratings_wt = self.df_ratings[~self.df_ratings.index.isin(movies_user_test)] 
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
    
    def recommend(self, n, *args, **kwargs):
        train, test = self.split(self.user)
        movies =  self.get_all_rij(train, test, self.user, n)
        return movies.sort_values(by='rating',ascending=False).iloc[:15, :].index.values.tolist()
    
    
def test_1_user():
    cf = CF('100k')
    cf.reset_index()
    users = cf.users()
    user = users[0]
    cf.set_user(user)
    return cf.recommend(10)

def execute(reset):
    cf = CF('100k')
    cf.reset_index()
    users = cf.users()
    n = len(users)
    for i in range(n):
        print('faltam', n - i, 'users')
        user = users[i]
        cf.set_user(user)
        data = {}
        if cf.last_user >= 0:
            with open('./app/recomendacoes/cf-recomendacoes.txt') as json_file:
                data = json.load(json_file)
        data[str(user)] = cf.recommend(10)
        with open('./app/recomendacoes/cf-recomendacoes.txt', 'w') as outfile:
            json.dump(data, outfile)
