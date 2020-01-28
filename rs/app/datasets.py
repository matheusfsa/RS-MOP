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
from sklearn.feature_extraction.text import CountVectorizer
from .tfidf import tf_id
from .plot2vec import plot2vec
from .word2vec import word2vec

def get_df_movies():
    df_movies = pd.read_csv('./app/datasets/movies_imdb.csv')
    df_movies = df_movies.dropna(subset=['plots', 'genres', 'directors', 'rating', 'runtimes', 'title', 'year'])
    df_movies = df_movies.loc[:, ['movieId', 'plots', 'genres', 'directors', 'rating', 'runtimes', 'title', 'year']]
    return df_movies

def get_count(df_movies, **kwargs):
    df_movies['plots'] = df_movies['plots'].apply(lambda x: str(x).replace("|"," "))
    ##Creating a list of stop words and adding custom stopwords
    #nltk.download('wordnet')
    #nltk.download('stopwords')
    stop_words = set(stopwords.words("english"))
    corpus = []
    for i in df_movies.index:
        #Remove punctuations
        text = re.sub('[^a-zA-Z]', ' ', df_movies['plots'][i])
        
        #Convert to lowercase
        text = text.lower()
        
        #remove tags
        text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
        
        # remove special characters and digits
        text=re.sub("(\\d|\\W)+"," ",text)
        
        ##Convert to list from string
        text = text.split()
        
        ##Stemming
        ps=PorterStemmer()
        #Lemmatisation
        lem = WordNetLemmatizer()
        text = [lem.lemmatize(word) for word in text if not word in  
                stop_words] 
        text = " ".join(text)
        corpus.append(text)
    n_features = kwargs.get('n_features', 2000)
    cv = CountVectorizer(max_df=0.8,stop_words=stop_words, max_features=n_features, ngram_range=(1,3))
    return cv.fit_transform(corpus), corpus


def get_dataset(feats, new_feats_func, **kwargs):
    df_movies = get_df_movies()
    n_words = kwargs.get('n_words', 2000)
    count, corpus = get_count(df_movies, n_features=n_words)
    kwargs['corpus'] = corpus
    kwargs['movies'] = df_movies
    new_feats = kwargs.get('new_feats', None)
    if new_feats is None:
        new_feats = new_feats_func(count, **kwargs)
    X = pd.DataFrame(new_feats, index=df_movies.index, columns=['feat_'+ str(i) for i in range(new_feats.shape[1])])
    if feats:
        df_movies_all = df_movies.loc[:, feats]
        #df_movies_all = pd.get_dummies(df_movies_all)
        if 'genres' in feats:
            genres = df_movies_all.genres.str.get_dummies()

            for g in genres.columns:
                df_movies_all[g] = genres[g]
            df_movies.drop(columns=['genres'])
        df_movies_all = pd.get_dummies(df_movies_all)    
        for col in X.columns:
            df_movies_all[col] = X[col]
    else:
        df_movies_all  = X 
    def min_max(X):
        return (X - X.min())/ (X.max() - X.min())
    if 'runtimes' in feats:
        df_movies_all['runtimes'] = min_max(df_movies_all['runtimes'])
    if 'year' in feats:
        df_movies_all['year'] = min_max(df_movies_all['year'])      
    if 'rating' in feats:
        df_movies_all['rating'] = min_max(df_movies_all['rating'])     
    df_movies_all['title'] = df_movies['title'] 
    df_movies_all['movieId'] = df_movies['movieId'] 
    return df_movies_all, new_feats

#def movie2vec(feats)


def dataset_tfid(feats, **kwargs):
    return get_dataset(feats, tf_id, **kwargs)


def dataset_plot2vec(feats, **kwargs):
    return get_dataset(feats, plot2vec, **kwargs)

def dataset_word2vec(feats, **kwargs):
    return get_dataset(feats, word2vec, **kwargs)

def dataset_ratings_user(X, **kwargs):
    user = kwargs.get('user', 1)
    df_ratings = kwargs.get('df_ratings', None)
    if df_ratings is None:
        df_ratings = pd.read_csv('.app/datasets/ml-20m/ratings.csv')
    df_ratings_u = df_ratings[df_ratings['userId'] == user]
    df_ratings_u.head()
    df_ratings_u = df_ratings_u.drop(columns=['userId'])
    df_ratings_u.index = df_ratings_u['movieId']
    df_ratings_u = df_ratings_u.drop(columns=['movieId', 'timestamp'])
    df_movies_u = X.drop(columns=['movieId'])
    df_movies_u.index = X['movieId']
    df_movies_u['title'] = X['title'] 
    df_movies_u['rating_user'] = df_ratings_u['rating']
    df_movies_u = df_movies_u[df_movies_u['rating_user'].notna()]
    return df_movies_u