from datasets import dataset_word2vec, dataset_ratings_user
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, RidgeCV, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, precision_score, recall_score
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd

def evaluate(X, user, df_ratings):
    df_movies_u = dataset_ratings_user(X,df_ratings=df_ratings, user=user) 
    train, test = train_test_split(df_movies_u , test_size=0.2)
    X_train = train.drop(columns=['rating_user','title'])
    y_train = train['rating_user']
    X_test = test.drop(columns=['rating_user','title'])
    y_test = test['rating_user']
    df_movies_u = None
    return train_predict(X_train, y_train, X_test, y_test)




def train_predict(X_train, y_train, X_test, y_test):

    precision = {}
    recall = {}
    y_true = (y_test >= 3) * 1

    def get_precision_recall(name, model):
        model = model.fit(X_train, y_train)
        y_pred = (model.predict(X_test) >= 3)*1
        precision[name] = precision_score(y_true, y_pred)
        recall[name] = recall_score(y_true, y_pred)
    ridge = RidgeCV(cv=5)
    get_precision_recall('ridge', RidgeCV(cv=5))
    print(ridge.alpha_)
    #get_precision_recall('kr', KernelRidge())
    #get_precision_recall('ada',AdaBoostRegressor(random_state=0, n_estimators=100))
    #get_precision_recall('gbr',GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,max_depth=1, random_state=0, loss='ls'))
    return precision, recall


def evaluate_method(X, users, df_ratings):
    precisions, recalls = [], []
    columns = ['ridge', 'kr', 'ada', 'gbr']
    for user in users:
        precision, recall = evaluate(X, user, df_ratings)
        precisions.append(precision)
        recalls.append(recall)
    df_precisions = pd.DataFrame(precisions, index=users, columns=columns)
    df_recalls = pd.DataFrame(recalls, index=users, columns=columns)
    return df_precisions, df_recalls


