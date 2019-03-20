FROM tensorflow/tensorflow:1.12.0-gpu-py3

WORKDIR /app

COPY /. /app

EXPOSE 8088

RUN apt-get -y update
RUN apt-get install -y \
    git \
    tmux \
    emacs \
    nano \
    graphviz \
    texlive-latex-base \
    texlive-science
RUN pip3 install \
    seaborn \
    scikit-learn \
    requests \
    keras \
    pydot
RUN apt-get clean && rm -rf /tmp/* /var/tmp/*
