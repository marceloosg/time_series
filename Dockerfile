FROM python:3.6
WORKDIR /app
#RUN apt update -y && apt install python3-dev python3-pip -y
RUN pip3 install --upgrade pip

ADD requirements.txt app/requirements.txt

RUN pip3 install -r app/requirements.txt

ADD server.py app/server.py
ADD model.h5py app/model.h5py
ADD scales.pkl app/scales.pkl

EXPOSE 5051

ENTRYPOINT ["python", "app/server.py"]~
