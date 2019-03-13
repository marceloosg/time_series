import pickle
from flask import Flask, abort, request
import numpy as np
import json
import hashlib
import keras.models  as km
from keras.layers import Dense, Dropout, LSTM
import keras
import tensorflow as tf
import os


class model_store():
    scales = (0, 0)
    model = None
    valid = False
    x = None

    def __init__(self):
        print(os.listdir())
        with open('app/scales.pkl', 'rb') as mp:
            output = pickle.load(mp)
            self.scales = output

        self.model = km.load_model('app/model.h5py')
        self.graph = tf.get_default_graph()
        print("MODEL LOADED")

    def scale(self, x):
        return (x - self.scales[1]) / (self.scales[0] - self.scales[1])

    def input(self, x):
        self.valid = len(x) == 60
        if self.valid:
            scaled_data = self.scale(np.array(x))
            self.x = np.reshape(scaled_data, (1, 60, 1))

    def predict(self):
        if not self.valid:
            return False
        with self.graph.as_default():
            y=self.model.predict(self.x)
        return y


app = Flask(__name__)
model = model_store()


@app.route("/")
def hello():
    return "Server is up"


@app.route("/math/", methods=['GET', 'POST'])
def math():
    if not request.json:
        abort(400)
    print(request.json)
    x=request.json['load']
    print(hashlib.md5(x.__str__().encode()).digest())
    model.input(x)
    y=model.predict()
    print(keras.__version__)
    #json.dumps({'loads':str(model.x)})

    #return #json.dumps({'load':hashlib.md5(x.__str__().encode()).digest()})
    return json.dumps({'y':str(y)})

@app.route("/data/", methods=['GET', 'POST'])
def data():
    if not request.json:
        abort(400)
    print(request.json)
    x=request.json['load']
    print(hashlib.md5(x.__str__().encode()).digest())
    model.input(x)
    y=model.predict()
    print(keras.__version__)
    #json.dumps({'loads':str(model.x)})

    #return #json.dumps({'load':hashlib.md5(x.__str__().encode()).digest()})
    return json.dumps({'y':str(y)})

@app.route("/stock/", methods=['GET', 'POST'])
def stock():
    if not request.json:
        abort(400)
    print(request.json)
    x=request.json['load']
    print(hashlib.md5(x.__str__().encode()).digest())
    model.input(x)
    y=model.predict()
    print(keras.__version__)
    #json.dumps({'loads':str(model.x)})

    #return #json.dumps({'load':hashlib.md5(x.__str__().encode()).digest()})
    return json.dumps({'y':str(y)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5051)
