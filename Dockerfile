ARG DOCKER_BASE="ubuntu18.04-anaconda3/cuda-11.0:latest"
FROM $DOCKER_BASE

ARG ENV_NAME="spd"
ARG USER_HOME="/home/docker"

USER root:root
ENV DEBIAN_FRONTEND=noninteractive

# update NVIDIA-KEY due to the rotation of the key recently.
RUN apt-key del 7fa2af80 && \
    rm /etc/apt/sources.list.d/cuda.list && \
    rm /etc/apt/sources.list.d/nvidia-ml.list && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb

# dependencies installation
RUN apt-get update && apt-get --no-install-recommends install -yq git cmake build-essential \
    libgl1-mesa-dev libsdl2-dev \
    libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
    libdirectfb-dev libst-dev mesa-utils xvfb x11vnc \
    python3-pip

COPY ./third_party/ $USER_HOME/envs
RUN conda create -n $ENV_NAME python=3.8 && \
    source deactivate && conda activate $ENV_NAME && \
    pip install --upgrade pip setuptools wheel && \
    pip install psutil && pip install $USER_HOME/envs/football && pip install $USER_HOME/envs/smac && pip install pettingzoo[mpe]==1.17.0 && \
    rm -r $USER_HOME/envs && \
    pip install sacred numpy scipy gym matplotlib seaborn pyyaml==5.3.1 pygame pytest probscale imageio snakeviz tensorboard-logger pyvirtualdisplay tqdm protobuf==3.20.1 && \
    pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html && \
    echo -e "\nconda activate $ENV_NAME" >> $USER_HOME/.bashrc

# COPY . $USER_HOME/pymarl2
RUN chown -R docker:docker $USER_HOME

USER docker:docker
RUN source $USER_HOME/.bashrc && source /etc/profile
WORKDIR "$USER_HOME/spd"
