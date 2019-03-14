FROM python:3.6
#RUN apt update -y && apt install python3-dev python3-pip -y
RUN pip3 install --upgrade pip
ADD requirements.txt .
RUN pip3 install -r requirements.txt 
WORKDIR "/app"
ADD lib lib
ADD models models
ADD server.py server.py

EXPOSE 5051

ENTRYPOINT ["python", "server.py"]~
