from flask import Flask
from flask import request
from flask import jsonify
from datasets import dataset_word2vec, dataset_ratings_user
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
import evaluate_solutions as ev
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class Recommender:
    X_train = None
    y_train = None
    X_test = None
    y_test = None
    model = None

app = Flask(__name__)
df_ratings = pd.read_csv('./app/datasets/ml-20m/ratings.csv')
X, new_feats = dataset_word2vec(['genres', 'rating', 'runtimes', 'year'], op='sum', n_features=300)

recommender = Recommender()
@app.route('/user', methods=['POST'])
def set_user():
    req_data = request.get_json()
    userId = int(req_data['user'])
    df_movies_u = dataset_ratings_user(X, df_ratings=df_ratings, user=userId) 
    train, test = train_test_split(df_movies_u , test_size=0.2)
    recommender.X_train = train.drop(columns=['rating_user','title'])
    recommender.y_train = train['rating_user']
    recommender.X_test = test.drop(columns=['rating_user','title'])
    recommender.y_test = test['rating_user']
    recommender.model = RidgeCV(cv=5).fit(recommender.X_train, recommender.y_train)
    return {'response': []}

@app.route('/', methods=['GET', 'POST'])
def parse_request():
    req_data = request.get_json()
    return jsonify({'solucoes':req_data})

@app.route('/max', methods=['GET', 'POST'])
def get_max():
    max_v = np.max(recommender.X_test, axis=0)
    return {'response': max_v.tolist()}


@app.route('/min', methods=['GET', 'POST'])
def get_min():
    min_v = np.min(recommender.X_test, axis=0)
    return {'response': min_v.tolist()}


@app.route('/n-variables', methods=['GET', 'POST'])
def n_variables():
    return {'response': [recommender.X_test.shape[1]]}

@app.route('/evaluate-solutions', methods=['GET', 'POST'])
def evaluate_solutions():
    message = request.get_json(silent=True)
    solucoes = message["solucoes"]
    solucoes = np.array(solucoes)
    y = ev.evaluate_solutions(solucoes, recommender.X_train, recommender)
    return {'response': y.tolist()}

@app.route('/evaluate-solutions-offspring', methods=['GET', 'POST'])
def evaluate_solutions_offspring():
    message = request.get_json(silent=True)
    solucoes = np.array(message["solucoes"])
    mating = np.array(message["mating"])
    y = ev.evaluate_solutions_offspring(mating, solucoes, recommender.X_train, recommender)
    return {'response': y.tolist()}


@app.route('/filtering', methods=['GET', 'POST'])
def filtering():
    message = request.get_json(silent=True)
    solucoes = message["solucoes"]
    sim = cosine_similarity(solucoes, recommender.X_test.values)
    pop_index = sim.argmax(axis=1)
    res = recommender.X_test.iloc[pop_index, :].drop_duplicates()
    #res.to_csv('./datasets/Recomendacoes.csv')
    y = ev.evaluate_solutions(res, recommender.X_test, recommender)
    objs = pd.DataFrame()
    objs['Acurácia'] = y[:, 0]
    objs['Diversidade'] = y[:, 1]
    objs['Novidade'] = y[:, 2]
    objs.index = res.index
    print(objs)
    return {'response': []}