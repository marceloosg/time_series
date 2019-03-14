ip=$(shell cat ip)
version=$(shell cat version)
project=$(shell cat project)

all:	upversion update

upversion:	version
	docker image list | grep v${version} && touch upversion || touch app/Dockerfile

build:	app/Dockerfile 
	cd app && docker build -t gcr.io/${project}/timeseries:v${version} . && cd .. && touch build

run:  build
	docker ps|cut -d ' ' -f 1 | tail -n 1|grep "CONTAINER" || docker kill $(shell docker ps|cut -d ' ' -f 1 | tail -n 1) 
	cd app
	docker run -p 80:5051  gcr.io/cryptic-bolt-122517/timeseries:v${version}  &
	cd ..
	touch run

test:	run
	sleep 3
	curl -H "Content-Type: application/json" --data @tests/data2.test.json http://localhost/data	> test

push: build
	docker push gcr.io/${project}/timeseries:v${version} && touch push

update: push
	kubectl set image deployment/hello-web hello-web=gcr.io/${project}/timeseries:v${version} && touch update

resize: 
	gcloud container clusters resize hello-cluster --node-pool default-pool --size=3 --region=us-east1

data2.test: tests/data2.test.json run
	curl -H "Content-Type: application/json" --data @tests/data2.test.json http://localhost/data > test
	diff test tests/data2.answer.json && touch data2.test || rm run &&  docker kill $(shell docker ps|cut -d ' ' -f 1 | tail -n 1) 

data3.test: tests/data3.test.json run
	curl -H "Content-Type: application/json" --data @tests/data3.test.json http://localhost/stock > test
	diff test tests/data3.answer.json && touch data3.test || (rm run &&  docker kill $(shell docker ps|cut -d ' ' -f 1 | tail -n 1))

data2.remote: tests/data2.test.json
	curl -H "Content-Type: application/json" --data @tests/data2.test.json http://${ip}/data > remote
	diff remote tests/data2.answer.json && touch data2.remote 

data3.remote: tests/data3.test.json
	curl -H "Content-Type: application/json" --data @tests/data3.test.json http://${ip}/stock > remote
	diff remote tests/data3.answer.json && touch data3.remote 
