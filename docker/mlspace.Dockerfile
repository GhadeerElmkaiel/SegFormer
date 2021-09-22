FROM registry.aicloud.sbcp.ru/base/jupyter-cuda10.1-cudnn8-tf2.3.0-gpu
#FROM cr.msk.sbercloud.ru/aicloud-jupyter/jupyter-cuda10.1-tf2.3.0-gpu-mlspace:latest

USER root

ENV TZ=Europe/Moscow

COPY . /home/jovyan/segformer/

WORKDIR /home/jovyan/segformer

RUN pip3 install -r requirements/build.txt
RUN pip3 install  -e . 

USER jovyan


#RUN groupadd --gid 1000 node && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash juvyan

# WORKDIR /
# COPY ./docker/script.sh /home/juvyan
# RUN chmod +x /home/juvyan/script.sh

# USER juvyan 1000

# WORKDIR /home/juvyan

# CMD ["/home/juvyan/script.sh"]

#RUN groupadd --gid 1000 node && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash jovyan
#
#WORKDIR /
#COPY ./docker/script.sh /home/jovyan
#RUN chmod +x /home/jovyan/script.sh
#
#USER jovyan 1000
#
#WORKDIR /home/jovyan
#
#CMD ["/home/jovyan/script.sh"]

#CMD ["bash"]
