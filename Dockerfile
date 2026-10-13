ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY . /opt/dsv41-ssd
RUN python3 /opt/dsv41-ssd/install.py apply && python3 /opt/dsv41-ssd/install.py verify
ENV PYTHONHASHSEED=0
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
