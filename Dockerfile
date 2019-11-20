FROM tensorflow/tensorflow:1.10.0-gpu-py3 
LABEL maintainer="Brian Kenji Iwana"

RUN apt-get update && apt-get install -y --no-install-recommends git=1:2.7.4-0ubuntu1.6 \
    && apt-get install -y python3-tk \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install git+https://github.com/albermax/innvestigate@1.0.8.3
RUN pip install scikit-image==0.15.0
RUN pip install tqdm

