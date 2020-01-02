import pandas as pd
import numpy as np
import json
from sklearn.cluster import KMeans


df_ratings = pd.read_table('./app/datasets/ml-1m/ratings.dat', delimiter='::', names=['userId', 'movieId', 'rating', 'timestamp'], engine='python')
df_movies = pd.read_csv('./app/datasets/movies_imdb.csv')
df_movies = df_movies.dropna(subset=['genres', 'cast','directors', 'runtimes', 'title', 'year'])
df_movies = df_movies.loc[:, ['movieId', 'genres','cast', 'directors', 'runtimes', 'title', 'year']]
df_ratings = df_ratings[df_ratings['movieId'].isin(df_movies['movieId'])]

last_user = -1
data = {}
with open('./app/datasets/index.txt') as json_file:
    data = json.load(json_file)
last_user = data.get('last_user',  None)
if last_user:
    del data['last_user']
else:
    last_user = -1
users = list(data.keys())
data = None
        

def get_weights(df_movies):
    def add_in_dict(d, l):
        for e in l:
            n = d.get(e, 0)
            d[e] = n+1

    genres = {}
    cast = {}
    directors = {}
    for index, movie in df_movies.iterrows():
        movie_genres = movie['genres'].split('|')
        add_in_dict(genres, movie_genres)
        movie_cast = movie['cast'].split('|')
        add_in_dict(cast, movie_cast)
        movie_directors = movie['directors'].split('|')
        add_in_dict(directors, movie_directors)
    genres = pd.Series(genres, name='genres')
    cast = pd.Series(cast, name='cast')
    directors = pd.Series(directors, name='directors')
    years = df_movies['year'].value_counts()
    return genres, cast, directors, years


def split(df_ratings, user):
        df_movies = df_ratings[df_ratings['userId'] == user]
        df_movies = df_movies.set_index('movieId')
        data = {}
        with open('./app/datasets/index.txt') as json_file:
            data = json.load(json_file)
        user_index = data[str(user)]
        train = df_movies.loc[user_index['train'], :]
        test = df_movies.loc[user_index['test'], :]
        return train, test

def get_gcdy_preference(train, df_movies):
    def add_in_list(d, l):
        for e in l:
            if e not in d:
                d.append(e)
            
    df = df_movies.loc[train[train.rating>=3].index, :]
    genres = []
    cast = []
    directors = []
    for index, movie in df.iterrows():
        movie_genres = movie['genres'].split('|')
        add_in_list(genres, movie_genres)
        movie_cast = movie['cast'].split('|')
        add_in_list(cast, movie_cast)
        movie_directors = movie['directors'].split('|')
        add_in_list(directors, movie_directors)  
    return genres, cast, directors, df.year.unique().tolist()

def get_wi(lista, preference, all_movies, total, inc):
    wi = 0.0
    for valor in lista:
        wi += all_movies[valor]/total
        if valor in preference:
            wi += inc
    return wi
def get_wa(lista, c, cast, total):
    return get_wi(lista, c, cast, total, 0.4)

def get_wd(lista, d, directors, total):
    return get_wi(lista, d, directors, total, 0.3)

def get_wg(lista, g, genres, total):
    return get_wg(lista, g, genres, total, 0.2)

def get_wy(lista, y, years, total):
    return get_wi(lista, y, years, total, 0.1)
        
def get_wr(movieId, df_ratings):
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

def get_user_weights(user, train, test, df_movies):
    total = df_movies.shape[0]
    genres, cast, directors, years = get_weights(df_movies)
    g, c, d, y = get_gcdy_preference(train, df_movies)
    ratings_test = df_ratings[df_ratings.userId == user][df_ratings.movieId.isin(test.index)]
    df_ratings_w_test = df_ratings[~df_ratings.index.isin(ratings_test.index)]
    weights = []
    for index, movie in df_movies.iterrows():
        wa = get_wa(movie['cast'].split('|'), c, cast, total)
        wd = get_wd(movie['directors'].split('|'), d, directors, total)
        wg = get_wg(movie['genres'].split('|'), g, genres, total)
        wy = get_wy([movie['year']], y, years, total)
        wr = get_wr(movie['movieId'], df_ratings_w_test)
        weights.append([wa, wd, wg, wy, wr])
    weights = pd.DataFrame(weights, index=df_movies.movieId, columns=['Wa', 'Wd', 'Wg', 'Wy', 'Wr'])
    return weights


def recommend(user):
    train, test = split(df_ratings, user)
    weights = get_user_weights(user, train, test, df_movies)
    X = weights[weights.index.isin(test.index)]
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    X['cluster'] = kmeans.labels_
    X['W'] = X['Wa'] + X['Wd'] + X['Wg'] + X['Wy'] + X['Wr']
    recommender_cluster = X.groupby('cluster').mean()['W'].sort_values(ascending=False).index[0]
    movies = X[X.cluster == recommender_cluster].index.tolist()
    if len(movies) > 15:
            return movies[:15]
    else:
        return movies

recommend(users[0])