from flask import Flask, abort, request
import json
import hashlib
import keras
from lib import PriceModel, StockModel

app = Flask(__name__)
model = PriceModel()
model_stock = StockModel('data3.model.h5py',train_model=False)

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

@app.route("/data", methods=['GET', 'POST'])
def data():
    print('data!!!')
    print(request)
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

@app.route("/stock", methods=['GET', 'POST'])
def stock():
    if not request.json:
        abort(400)
    print(request.json)
    x=request.json['load']
    print(hashlib.md5(x.__str__().encode()).digest())
    #model.input(x)
    print(model_stock)
    y=model_stock.predict(x)
    print(keras.__version__)
    #json.dumps({'loads':str(model.x)})

    #return #json.dumps({'load':hashlib.md5(x.__str__().encode()).digest()})
    yl=[z[0] for z in list(y)]
    print(yl)
    return json.dumps({'y':str(yl)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5051)
