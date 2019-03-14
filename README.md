# 4 Layer Artificial Recurrent Neural Network (Long short-term memory) applied to stock market Time Series data

Description

## Time series prediction models

### Organization:

*	**notebooks/**: contains the notebooks, html files that answers the stakeholders questions
*	**test/**: sample request and answers json files for api testing
*	**app/**: application files
	*	**app/server.py**: Flask server
	*	**app/models**: Static model saved informations
	*	**app/lib**: Model helper classes


## Installation

This application works with Python via pip

	pip install -r requirements.txt

or you can pip install packages in your home directory instead, which doesn't require any special privileges.

	pip install -r requirements.txt --user

## Server local Usage:

	python3 server.py

## Docker

	docker build -t timeseries .

	docker run -p 5051:5051 timeseries

	Open http://localhost:5051


## Client

### Local Usage:

### Dataset2:
-	**data2.test.json** has one key: 'load'
-	its value is an one-dimensional array of length multiple of 60
-	the model arranges every 60 elements into one individual input to be predicted
	
-	json: '{"load": [ ... ] }'

-	curl -H "Content-Type: application/json" --data @tests/data2.test.json http://localhost:5051/data

### Dataset3:
-	**data3.test.json** has one key: 'load'
-	its value is a list where each element is another list that follows the pattern of data3.input.columns.

-	json:  '{"load": [[ ... ], [ ... ], [ ... ], etc }'

-	curl -H "Content-Type: application/json" --data @tests/data3.test.json http://localhost:5051/stock


For remote access exchange "localhost:5051" to the relevant ip, see the Makefile, make data2.remote, make data3.remote
