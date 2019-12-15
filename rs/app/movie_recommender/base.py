import pandas as pd
class Recommender:
    df_ratings = None
    user = None

    def load_ratings(self, n_ratings):
        if n_ratings == '20M':
            self.df_ratings = pd.read_csv('.app/datasets/ml-20m/ratings.csv')
        if n_ratings == '100k':
            self.df_ratings = pd.read_table('./app/datasets/ml-1m/ratings.dat', delimiter='::', names=['userId', 'movieId', 'rating', 'timestamp'])
        
    
    def set_user(self, user):
        raise NotImplementedError

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def recommend(self, n, *args, **kwargs):
        raise NotImplementedError

