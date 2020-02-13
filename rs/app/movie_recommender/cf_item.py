from .base import Recommender
import pandas as pd
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

class CF(Recommender):
    
    def __init__(self, df_ratings, item_id_col, user_id_col, rating_col):
        self.df_ratings = df_ratings
        self.item_id_col = item_id_col
        self.user_id_col = user_id_col
        self.rating_col = rating_col
        
    def rij(self, item_item_u, i, train, b_ui, n):
        s_ij = item_item_u.loc[i, :].sort_values(ascending=False)[:n]
        r_ij = train.loc[s_ij.index,:].rating
        b_uj = b_ui.loc[s_ij.index,:].rating
        num = s_ij*(r_ij - b_uj)
        den = s_ij.abs().sum()
        print(num, den)
        return b_ui.loc[i, :].values[0] + (num.sum()/den.sum())
    
    def get_itens_rating_table(self, df_ratings_wt, user, test_index):
        df_ratings_user = self.df_ratings[self.df_ratings[self.user_id_col] == user]
        table = df_ratings_wt[df_ratings_wt[self.item_id_col].isin(df_ratings_user[self.item_id_col])]
        itens_rating = pd.pivot_table(table,
                         values=self.rating_col, 
                         index=self.item_id_col, 
                         columns=self.user_id_col).fillna(0)
        return itens_rating                         

    def get_item_item_u(self, itens_rating, movies_user_test, movies_user_train):
        sim = pd.DataFrame(cosine_similarity(itens_rating, itens_rating), index=itens_rating.index, columns=itens_rating.index)
        item_item_u = sim.loc[movies_user_test, movies_user_train]
        return item_item_u

    def get_all_rij(self, train, test, user, n):
        df_ratings_wt = self.df_ratings[~self.df_ratings.index.isin(test.index)] 
        itens_rating = self.get_itens_rating_table(df_ratings_wt, user, test.index)
        item_item_u = self.get_item_item_u(itens_rating, test.index, train.index)
        b_u = train.rating.std()
        b_i = df_ratings_wt[[self.item_id_col, self.rating_col]].groupby([self.item_id_col]).std().fillna(0)
        beta = self.df_ratings.rating.std()
        b_ui = beta + b_u + b_i
        rs = pd.DataFrame(np.zeros(test.index.shape[0]), index=test.index, columns=[self.rating_col])
        for i in test.index:
            rs.loc[i, 'rating'] = self.rij(item_item_u, i, train, b_ui, n)
        return rs
    
    def recommend(self,train, test, user, n, *args, **kwargs):
        itens =  self.get_all_rij(train, test, user, n)
        return itens.sort_values(by='rating',ascending=False).iloc[:15, :].index.values.tolist()
 
'''
from app.movie_recommender.cf_item import CF
import numpy as np
ratings = [[1, 2, 4], [1,1,3], [1,3,4], [1,4,4], [2, 1,4],[2, 3,3]]
import pandas as pd
df_ratings = pd.DataFrame(ratings, columns=['userId', 'itemId', 'rating'])
test = pd.DataFrame([4, 3], index=[2, 1], columns=['rating'])

train = pd.DataFrame([4], index=[3], columns=['rating'])
cf = CF(df_ratings, 'itemId', 'userId', 'rating')
cf.recommend(train, test, 1, 1)

'''

