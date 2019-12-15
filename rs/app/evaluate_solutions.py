import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def evaluate_solutions(solutions, data, recommender):
    y = np.zeros((solutions.shape[0], 3))
    y[:, 0] = -recommender.model.predict(solutions) 
    y[:, 1] = -get_diversity(solutions)
    y[:, 2] = -get_novelty(solutions, data)
    # print('solutions:', solutions[0:5])
    # print('acurácia:', y[:, 0])
    # print('diversidade:', y[:, 1])
    # print('novelty:', y[:, 2])
    return y

def evaluate_solutions_offspring(mating, solutions, data, recommender):
    y = np.zeros((solutions.shape[0], 3))
    y[:, 0] = -recommender.model.predict(solutions) 
    y[:, 1] = -get_diversity_offspring(mating, solutions)
    y[:, 2] = -get_novelty(solutions, data)
    return y


def get_diversity_offspring(mating, solutions):
    a = np.append(solutions, mating, axis=0)
    sim = cosine_similarity(a,a)
    np.fill_diagonal(sim, 0)
    n = solutions.shape[0] - 1 + mating.shape[0]
    return ((1 - sim).sum(axis=1) * 1/n)[:solutions.shape[0]].reshape(1, -1)

def get_diversity(solutions):
    sim = cosine_similarity(solutions, solutions)
    np.fill_diagonal(sim, 0)
    return (1 - sim).sum(axis=1) * 1/(solutions.shape[0] - 1)


def get_novelty(solutions, data):
    sim = cosine_similarity(solutions, data)
    
    return (1-sim).max(axis=1) #* 1/(data.shape[0] - 1)


def popularity(df_ratings, i):
    mean = df_ratings[['movieId', 'rating']].groupby(['movieId']).mean()
    std = df_ratings[['movieId', 'rating']].groupby(['movieId']).std()
    f2 = (1/ (mean*((std+1)**2))).sum()
    return f2