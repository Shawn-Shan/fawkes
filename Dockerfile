# set base image
FROM python:3.7-slim

# set the working directory in the container
WORKDIR /app

# install fawkes
RUN pip install fawkes

# downgrade h5py: https://github.com/Shawn-Shan/fawkes/issues/75
RUN pip install --upgrade h5py==2.10.0

# copy the files you want to edit with fawkes
COPY imgs imgs