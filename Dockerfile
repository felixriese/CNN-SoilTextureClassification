FROM tensorflow/tensorflow:2.0.0-gpu-py3

WORKDIR /app

COPY /. /app

EXPOSE 8088

RUN apt-get -y update \
    && apt-get install -y --no-install-recommends \
        git \
        tmux \
        emacs \
        nano \
        graphviz \
    &&  apt-get clean \
    && rm -rf /tmp/* /var/tmp/* /var/lib/apt/lists/*

RUN pip3 install -r requirements.txt
