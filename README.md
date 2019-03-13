# time_series
Description

Tiem series prediction models

Installation

This application works with Python via pip

pip install -r requirements.txt

or you can pip install packages in your home directory instead, which doesn't require any special privileges.

pip install -r requirements.txt --user

Usage

python3 server.py

Docker

docker build -t timeseries .

docker run -p 5051:5051 timeseries

Open http://localhost:5051


# Client

Dataset2:
data2.test.json has one key: 'load'
its value is an one-dimensional array of length multiple of 60
the model arranges every 60 elements into one individual input to be predicted

curl -H "Content-Type: application/json" --data @tests/data2.test.json http://localhost:5051/data

Dataset3:
data3.test.json has one key: 'load'
its value is a list where each element is another list that follows the pattern of data3.input.columns.

curl -H "Content-Type: application/json" --data @tests/data3.test.json http://localhost:5051/stock

