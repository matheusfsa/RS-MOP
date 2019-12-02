from time import time
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
import multiprocessing
import numpy as np
def get_setences(docs):
    phrases = Phrases(docs, min_count=30, progress_per=10000)
    sentences = phrases[docs]
    return sentences

def build_model(sentences, n_feats):
    cores = multiprocessing.cpu_count()
    w2v_model = Word2Vec(min_count=10,
                        window=3,
                        size=n_feats,
                        sample=6e-5, 
                        alpha=0.03, 
                        min_alpha=0.0007, 
                        negative=20,
                        workers=cores-1)
    t = time()
    w2v_model.build_vocab(sentences, progress_per=10000)
    print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))
    return w2v_model

def train_model(w2v_model, sentences):
    t = time()
    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
    print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
    return w2v_model

def plot2vecAgg(w2v_model, plot, op):
    span_plot = w2v_model.wv[[w for w in plot if w in w2v_model.wv.vocab]]
    if op == 'sum':
        return np.sum(span_plot, axis=0)/len(span_plot)
    elif op == 'min':
        return np.min(span_plot, axis=0)
    else:
        return np.max(span_plot, axis=0)


def compute_vectors(w2v_model,sentences, n_feats, op='sum'):
    #X_rref = np.zeros((df_movies.shape[0], 300))
    n = len(sentences)
    X_sum = np.zeros((n, n_feats))
    t = time()
    for i in range(n):
        #X_rref[i] = plot2vecRref(sentences[i])
        X_sum[i] = plot2vecAgg(w2v_model, sentences[i], op)
    print('Time to compute vectors: {} mins'.format(round((time() - t) / 60, 2)))
    return X_sum

def word2vec(count, **kwargs):
    corpus = kwargs['corpus']
    n_feats = kwargs.get('n_features', 300)
    op = kwargs.get('op', 'sum')
    sentences = get_setences(corpus)
    w2v_model = build_model(sentences, n_feats)
    w2v_model = train_model(w2v_model, sentences)
    return compute_vectors(w2v_model, sentences, n_feats, op)

