from flask import Flask
from flask import request
from flask import jsonify
from .datasets import dataset_word2vec, dataset_ratings_user
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from .movie_recommender.rs_mop import CB_MOP


app = Flask(__name__)

recommender = CB_MOP('100k')
recommender.reset_users()
@app.route('/users', methods=['POST'])
def get_users():
    users = [int(u) for u in recommender.users()]
    return {'response': users}

@app.route('/user', methods=['POST'])
def set_user():
    req_data = request.get_json()
    userId = int(req_data['user'])
    recommender.set_user(userId)
    return {'response': []}

@app.route('/', methods=['GET', 'POST'])
def parse_request():
    req_data = request.get_json()
    return jsonify({'solucoes':req_data})

@app.route('/max', methods=['GET', 'POST'])
def get_max():
    #max_v = np.max(recommender.X_test, axis=0)
    return {'response': recommender.max().tolist()}


@app.route('/min', methods=['GET', 'POST'])
def get_min():
    #min_v = np.min(recommender.X_test, axis=0)
    return {'response': recommender.min().tolist()}


@app.route('/n-variables', methods=['GET', 'POST'])
def n_variables():
    return {'response': [recommender.n_variables()]}

@app.route('/evaluate-solution', methods=['GET', 'POST'])
def evaluate_solution():
    message = request.get_json(silent=True)
    solution = message["solution"]
    y = recommender.evaluate_solution(solution)
    return {'response': y}

@app.route('/evaluate-solutions', methods=['GET', 'POST'])
def evaluate_solutions_req():
    message = request.get_json(silent=True)
    solucoes = message["solucoes"]
    solucoes = np.array(solucoes)
    y = recommender.evaluate_solutions(solucoes, recommender.X_train, mating=None)
    return {'response': y.tolist()}

@app.route('/evaluate-solutions-offspring', methods=['GET', 'POST'])
def evaluate_solutions_offspring():
    message = request.get_json(silent=True)
    solucoes = np.array(message["solucoes"])
    mating = np.array(message["mating"])
    y = recommender.evaluate_solutions(solucoes, recommender.X_train, mating=mating)
    return {'response': y.tolist()}

@app.route('/filtering', methods=['GET', 'POST'])
def filtering():
    message = request.get_json(silent=True)
    solucoes = np.array(message["solucoes"])
    print(recommender.filtering(solucoes))
    #res.to_csv('./datasets/Recomendacoes.csv')
    '''
    y = ev.evaluate_solutions(res, recommender.X_test, recommender)
    objs = pd.DataFrame()
    objs['Acurácia'] = y[:, 0]
    objs['Diversidade'] = y[:, 1]
    objs['Novidade'] = y[:, 2]
    objs.index = res.index
    print(objs)
    '''
    return {'response': []}