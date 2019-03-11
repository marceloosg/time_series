FROM python:3.6
WORKDIR /app
#RUN apt update -y && apt install python3-dev python3-pip -y
RUN pip3 install --upgrade pip

ADD requirements.txt app/requirements.txt

RUN pip3 install -r app/requirements.txt

COPY server.py model.pkl scales.pkl app/
CMD ["python", "server.py"]~
