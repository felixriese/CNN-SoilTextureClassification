FROM tensorflow/tensorflow:2.0.0-gpu-py3

WORKDIR /app

COPY /. /app

EXPOSE 8088

RUN apt-get -y update
RUN apt-get install -y --no-install-recommends \
        wget \
        git \
        tmux \
        emacs \
        nano

RUN pip3 install --upgrade pip
RUN pip3 install pandas

RUN apt-get clean && rm -rf /tmp/* /var/tmp/*
