from utils import Predictor
from utils import DataLoader
from settings.constants import TRAIN_CSV, VAL_CSV
from flask import Flask, request, jsonify, make_response

import pandas as pd
import json


app = Flask(__name__)


@app.route('/predict', methods=['GET'])
def predict():
    #received_keys = sorted(list(request.form.keys()))
    #if len(received_keys) > 1 or 'data' not in received_keys:
    #    err = 'Wrong request keys'
    #    return make_response(jsonify(error=err), 400)

    #data = json.loads(request.form.get(received_keys[0]))

    #with open('settings/specifications.json') as f:
    #    specifications = json.load(f)
    #info = specifications['description']
    #x_columns = info['X']
    #y_column = info['y']
    #test_set = pd.read_csv(VAL_CSV, header=0)
    #x, y = test_set[x_columns], test_set[y_column]
    #data = {'data': json.dumps(x.to_dict())}
    #df = pd.DataFrame(columns=x_columns, data=x.values)
    #predictor = Predictor()
    #response_dict = {'prediction': predictor.predict(df).tolist()}
    PREDICT_ROUTE = "/predict"
    #response = requests.get(PREDICT_ROUTE, data=data)
    received_keys = sorted(list(request.form.keys()))
    if len(received_keys) > 1 or 'data' not in received_keys:
        err = 'Wrong request keys'
        return make_response(jsonify(error=err), 400)

    data = json.loads(request.form.get(received_keys[0]))
    df = pd.DataFrame.from_dict(data)

    loader = DataLoader()
    loader.fit(df)
    processed_df = loader.load_data()

    predictor = Predictor()
    response_dict = {'prediction': predictor.predict(processed_df).tolist()}


    return make_response(jsonify(response_dict), 200)





if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=8000)
    #app.run(host='0.0.0.0',port=8000)