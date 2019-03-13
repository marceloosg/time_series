FROM python:3.6
WORKDIR /app
#RUN apt update -y && apt install python3-dev python3-pip -y
RUN pip3 install --upgrade pip

ADD requirements.txt 
RUN pip3 install -r requirements.txt

ADD server.py 
ADD model.h5py 
ADD scales.pkl 
EXPOSE 5051

ENTRYPOINT ["python", "server.py"]~
