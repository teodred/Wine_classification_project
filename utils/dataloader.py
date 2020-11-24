import pandas as pd
from sklearn.model_selection import train_test_split
from settings.constants import TRAIN_CSV, VAL_CSV
from sklearn.preprocessing import StandardScaler


class DataLoader(object):
    def fit(self, dataset):
        self.dataset = dataset.copy()

    def get_title(self):
        pass

    def load_data(self):
        extra_col = "Unnamed: 0"
        if extra_col in self.dataset.columns:
            self.dataset = self.dataset.drop(['Unnamed: 0'],axis=1)
        return self.dataset



