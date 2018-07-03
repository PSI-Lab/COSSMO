FROM tensorflow/tensorflow:latest-gpu
MAINTAINER Hannes Bretschneider <hannes@deepgenomics.com>
RUN curl -sSL https://sdk.cloud.google.com | bash
ENV PATH=$PATH:/root/google-cloud-sdk/bin/
ADD . /COSSMO
RUN pip install -r /COSSMO/requirements.txt
RUN pip install -e /COSSMO
WORKDIR /run
