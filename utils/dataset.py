import pandas as pd
import csv

class Dataset:
    def __init__(self, csv_file):
        self.csv_file = csv_file

    def len(self):
        with open(self.csv_file, 'rt') as f:
            return sum(1 for row in f) - 1

    def columns(self):
        with open(self.csv_file, 'rt') as f:
            columns = f.readline().rstrip().split(',')
        del columns[0]
        return columns

    def get_item(self, index):
        idx = index - 1
        # Load data and get label
        with open(self.csv_file, 'rt') as f:
            reader = csv.reader(f)
            for line in reader:
                if str(idx) in line:
                    break
        y = line[-1]
        del line[-1]
        x = line
        return x, y

    def get_items(self, items_number):
        data = pd.read_csv(self.csv_file, nrows=items_number)
        y = data['is_good']
        x = data.drop(['is_good','Unnamed: 0'], axis=1)
        return x, y


