FROM nvcr.io/nvidia/pytorch:20.03-py3

USER root

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Moscow

RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 tmux ncdu htop \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

#RUN conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
RUN pip install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# install mmdetection3d
RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmdetection3d.git /mmdetection3d
WORKDIR /mmdetection3d
# checkout to mmdetection3d v0.7.0
RUN git checkout 37ce1871eac08cb6c1e5cb9e01d633583a9f3a49
RUN cat mmdet3d/version.py
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install llvmlite --ignore-installed
RUN pip install --no-cache-dir -e .

# uninstall pycocotools installed by nuscenes-devkit and reinstall mmpycocotools
RUN pip uninstall pycocotools --no-cache-dir -y
RUN pip install mmpycocotools --no-cache-dir --force --no-deps

WORKDIR /
RUN git clone https://github.com/open-mmlab/mmcv.git /mmcv
WORKDIR /mmcv
# checkout to mmcv v1.2.1
RUN git checkout 91a7fee03a3973a56cb5f687a6859ef0aaacf15e
RUN MMCV_WITH_OPS=1 pip install -e .

ENV PYTHONPATH "${PYTHONPATH}:/perception"
ENV PYTHONPATH "${PYTHONPATH}:/mmdetection3d/tools"
ENV PYTHONPATH "${PYTHONPATH}:/home/jovyan/stereo_perception"

RUN groupadd --gid 1000 node && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash juvyan

WORKDIR /
COPY ./docker/script.sh /home/juvyan
RUN chmod +x /home/juvyan/script.sh

USER juvyan 1000

WORKDIR /home/juvyan

CMD ["/home/juvyan/script.sh"]

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
