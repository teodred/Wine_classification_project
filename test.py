from utils.dataloader import DataLoader
import numpy as np
from utils.trainer import Estimator
from sklearn.metrics import accuracy_score
import pickle
import pandas as pd
import json
from utils.predictor import Predictor
from utils.dataset import Dataset
from settings.constants import TRAIN_CSV, VAL_CSV
from sklearn.preprocessing import StandardScaler
import requests


dataloader = DataLoader()
with open('settings/specifications.json') as f:
    specifications = json.load(f)
def split_data(train_dataset, test_dataset):
    features = train_dataset.columns[:-1]
    output = train_dataset.columns[-1]
    X_train = train_dataset[features].values
    y_train = train_dataset[output].values
    X_test = test_dataset[features].values
    y_test = test_dataset[output].values
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X=X_train, y=None)
    X_test = sc_X.transform(X_test)
    return X_train, X_test, y_train, y_test
#X_train, X_test, y_train, y_test = split_data(train_dataset,test_dataset)

#model = Estimator.fit(X_train,y_train)

#prediction = Estimator.predict(model,X_test)
#prediction1 = Predictor().predict(X_test)
#loaded_model = pickle.load(open('models/KNN.pickle', 'rb'))
#print(loaded_model.score(test_set[x_columns].values, test_set[y_column].values))
#print(accuracy_score(y_test,prediction))
#print(accuracy_score(y,prediction1))
#print(X_test)


PREDICT_ROUTE = "http://127.0.0.1:8000/predict"
info = specifications['description']
x_columns, y_column, metrics = info['X'], info['y'], info['metrics']
train_set = pd.read_csv(TRAIN_CSV, header=0)
test_set = pd.read_csv(VAL_CSV, header=0)
train_x, train_y = train_set[x_columns], train_set[y_column]
test_x, test_y = test_set[x_columns], test_set[y_column]
loader = DataLoader()
loader.fit(train_x)
train_processed = loader.load_data()
loader = DataLoader()
loader.fit(test_x)
test_processed = loader.load_data()
trained = Estimator.fit(train_processed, train_y)
trained_predict = Estimator.predict(trained, test_processed)
trained_score = round(eval(metrics)(test_y, trained_predict), 2)
req_data = {'data': json.dumps(test_x.to_dict())}
response = requests.get(PREDICT_ROUTE, data=req_data)
api_predict = response.json()['prediction']
api_score = round(eval(metrics)(test_y, api_predict), 2)
print(trained_score)
print(api_score)
assert trained_score == api_score
