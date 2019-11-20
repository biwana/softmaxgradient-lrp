FROM tensorflow/tensorflow:1.10.0-gpu-py3 
LABEL maintainer="Brian Kenji Iwana"

ENV PYTHONPATH "${PYTHONPATH}:/work"
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git \
    && apt-get install -y python3-tk

RUN pip install git+https://github.com/albermax/innvestigate@1.0.8.3
RUN pip install scikit-image==0.15.0
RUN pip install tqdm

