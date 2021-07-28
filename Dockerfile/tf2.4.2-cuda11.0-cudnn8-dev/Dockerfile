FROM tensorflow/tensorflow:2.4.2-gpu-jupyter
MAINTAINER Uijeong jeong <uiui.jeong@dgu.ac.kr>

RUN apt-get -y update
RUN apt-get -y install python-dev libgl1-mesa-glx gcc

WORKDIR /workspace
RUN chmod -R a+w /workspace

COPY requirements.txt /workspace/requirements.txt
RUN pip install -r requirements.txt

EXPOSE 8888

CMD jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root

