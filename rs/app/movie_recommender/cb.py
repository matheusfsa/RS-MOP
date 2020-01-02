from .base import Recommender
import pandas as pd
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

class CB(Recommender):
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
        self.df_movies = self.load_movies()
        self.genres, self.cast, self.directors, self.years = self.get_weights()
        
    
    def load_movies(self):
        df_movies = pd.read_csv('./app/datasets/movies_imdb.csv')
        df_movies = df_movies.dropna(subset=['genres', 'cast','directors', 'runtimes', 'title', 'year'])
        df_movies = df_movies.loc[:, ['movieId', 'genres','cast', 'directors', 'runtimes', 'title', 'year']]
        return df_movies

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

    def get_weights(self):
        def add_in_dict(d, l):
            for e in l:
                n = d.get(e, 0)
                d[e] = n+1
        genres = {}
        cast = {}
        directors = {}
        for index, movie in self.df_movies.iterrows():
            movie_genres = movie['genres'].split('|')
            add_in_dict(genres, movie_genres)
            movie_cast = movie['cast'].split('|')
            add_in_dict(cast, movie_cast)
            movie_directors = movie['directors'].split('|')
            add_in_dict(directors, movie_directors)
        genres = pd.Series(genres, name='genres')
        cast = pd.Series(cast, name='cast')
        directors = pd.Series(directors, name='directors')
        years = self.df_movies['year'].value_counts()
        return genres, cast, directors, years

    def get_gcdy_preference(self, train, df_movies):
        def add_in_list(d, l):
            for e in l:
                if e not in d:
                    d.append(e)
                
        df = df_movies.loc[train[train.rating>=3].index, :]
        genres = []
        cast = []
        directors = []
        for index, movie in df.iterrows():
            print(movie['genres'])
            movie_genres = movie['genres'].split('|')
            add_in_list(genres, movie_genres)
            movie_cast = movie['cast'].split('|')
            add_in_list(cast, movie_cast)
            movie_directors = movie['directors'].split('|')
            add_in_list(directors, movie_directors)  
        return genres, cast, directors, df.year.unique().tolist()
    
    def get_wi(self, lista, preference, all_movies, total, inc):
        wi = 0.0
        for valor in lista:
            wi += all_movies[valor]/total
            if valor in preference:
                wi += inc
        return wi
    def get_wa(self, lista, c, cast, total):
        return get_wi(lista, c, cast, total, 0.4)

    def get_wd(self, lista, d, directors, total):
        return get_wi(lista, d, directors, total, 0.3)
    
    def get_wg(self, lista, g, genres, total):
        return get_wg(lista, g, directors, total, 0.2)

    def get_wy(self, lista, y, years, total):
        return get_wi(lista, y, years, total, 0.1)
            
    def get_wr(self, movieId, df_ratings):
        ratings = df_ratings[df_ratings['movieId'] == movieId].rating.value_counts()
        wr = 0.0
        
        for index, count in ratings.iteritems():
            if index >=3:
                if count <= 100:
                    wr += index
                elif count <= 1000:
                    wr += index*2
                else:
                    wr += index*3
            else:
                if count <= 100:
                    wr += 1
                elif count <= 1000:
                    wr += 2
                else:
                    wr += 3
        return wr
    
    def get_user_weights(self, user, train, test, df_movies):
        total = df_movies.shape[0]
        g, c, d, y = self.get_gcdy_preference(train, df_movies)
        ratings_test = self.df_ratings[self.df_ratings.userId == user][self.df_ratings.movieId.isin(test.index)]
        df_ratings_w_test = self.df_ratings[~self.df_ratings.index.isin(ratings_test.index)]
        weights = []
        for index, movie in df_movies.iterrows():
            wa = self.get_wa(movie['cast'].split('|'), c, self.cast, total)
            wd = self.get_wd(movie['directors'].split('|'), d, self.directors, total)
            wg = self.get_wd(movie['genres'].split('|'), g, self.genres, total)
            wy = self.get_wy([movie['year']], y, self.years, total)
            wr = self.get_wr(movie['movieId'], df_ratings_w_test)
            weights.append([wa, wd,wg, wy, wr])
        weights = pd.DataFrame(weights, index=df_movies.movieId, columns=['Wa', 'Wd', 'Wg','Wy', 'Wr'])
        return weights
    
    def recommend(self, n):
        train, test = self.split(self.user)
        weights = self.get_user_weights(self.user, train, test, self.df_movies)
        X = weights[weights.index.isin(test.index)].iloc[:30, :]
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        X['cluster'] = kmeans.labels_
        X['W'] = X['Wa'] + X['Wd'] + X['Wg'] + X['Wy'] + X['Wr']
        recommender_cluster = X.groupby('cluster').mean()['W'].sort_values(ascending=False).index[0]
        movies = X[X.cluster == recommender_cluster].index.tolist()
        if len(movies) > 15:
            return movies[:15]
        else:
            return movies

def test_1_user():
    cb = CB('100k')
    cb.reset_index()
    users = cb.users()
    user = users[0]
    cb.set_user(user)
    return cb.recommend(15)


def execute(reset):
    cf = CF('100k')
    if reset:
        cf.reset_index()
    users = cf.users()
    n = len(users)
    for i in range(n):
        print('faltam', n - i, 'users')
        user = users[i]
        cf.set_user(user)
        data = {}
        if cf.last_user >= 0:
            with open('./app/recomendacoes/cb-recomendacoes.txt') as json_file:
                data = json.load(json_file)
        data[str(user)] = cb.recommend(10)
        with open('./app/recomendacoes/cb-recomendacoes.txt', 'w') as outfile:
            json.dump(data, outfile)