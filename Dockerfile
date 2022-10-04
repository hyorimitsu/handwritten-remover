FROM python:3

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y libgl1-mesa-dev

RUN pip install opencv-python && \
    pip install numpy
