#FROM registry.aicloud.sbcp.ru/base/jupyter-cuda10.1-cudnn8-tf2.3.0-gpu
FROM cr.msk.sbercloud.ru/aicloud-jupyter/jupyter-cuda10.1-tf2.3.0-gpu-mlspace:latest

USER root

ENV TZ=Europe/Moscow

COPY . /home/jovyan/segformer/

WORKDIR /home/jovyan/segformer

RUN pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -r requirements/build_ml_space.txt




