from utils.dataloader import DataLoader
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier as KNN #K-Nearest Neighbors
from sklearn.model_selection import GridSearchCV
import pickle
import json

class Estimator:
    @staticmethod
    def fit(train_x, train_y):
        parameters_KNN = {
            "n_neighbors": [2, 5, 7, 15],
            "weights": ('uniform', 'distance'),
            "algorithm": ('auto', 'ball_tree', 'kd_tree', 'brute'),
            'p': [1, 2, 5]

        }
        model_KNN = KNN(n_jobs=-1)
        model_KNN_with_best_params = GridSearchCV(model_KNN, parameters_KNN)
        return model_KNN_with_best_params.fit(train_x, train_y)

    @staticmethod
    def predict(trained, test_x):
        return trained.predict(test_x)


