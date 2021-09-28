FROM python:3.7-slim

ENV PATH="~/.local/bin:${PATH}"
ENV PATH="/home/testuser/.local/bin:${PATH}"

#Mirror host user
#Efficently mirror host user and share files between host and docker
#https://stackoverflow.com/a/44683248/298240
ARG UNAME=testuser
ARG UID=1000
ARG GID=1000
#FIXES https://stackoverflow.com/a/63377623/298240
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 python3-opencv sudo && groupadd -g $GID -o $UNAME && useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME && echo "testuser:testuser" | chpasswd && adduser testuser sudo
USER $UNAME
WORKDIR /home/$UNAME/workspace
RUN pip install --upgrade h5py==2.10.0 fawkes opencv-python
