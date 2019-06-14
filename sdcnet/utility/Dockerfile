# ===========
# base images
# ===========
FROM FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04


# ===============
# system packages
# ===============
RUN apt-get update
RUN apt-get install -y bash-completion \
    emacs \
    ffmpeg \
    git \
    graphviz \
    htop \
    libopenexr-dev \
    openssh-server \
    rsync \
    wget

# ===========
# latest apex
# ===========
RUN pip uninstall -y apex
RUN git clone https://github.com/NVIDIA/apex.git ~/apex && \
    cd ~/apex && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

# ============
# pip packages
# ============
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install --upgrade boto3
RUN pip install --upgrade cffi
RUN pip install --upgrade colorama==0.3.7
RUN pip install --upgrade Cython
RUN pip install --upgrade dominate
RUN pip install --upgrade ffmpeg
RUN pip install --upgrade graphviz
RUN pip install --upgrade imageio
RUN pip install --upgrade ipython
RUN pip install --upgrade matplotlib
RUN pip install --upgrade natsort
RUN pip install --upgrade nltk
RUN pip install --upgrade numpy
RUN pip install --upgrade openexr
RUN pip install --upgrade packaging
RUN pip install --upgrade pandas
RUN pip install --upgrade pillow
RUN pip install --upgrade pylint
RUN pip install --upgrade pytz
RUN pip install --upgrade pyyaml
RUN pip install --upgrade requests
RUN pip install --upgrade scikit-image
RUN pip install --upgrade scikit-learn
RUN pip install --upgrade scipy
RUN pip install --upgrade sentencepiece
RUN pip install --upgrade setproctitle
RUN pip install --upgrade tensorboard
RUN pip install --upgrade tensorboardX
RUN pip install --upgrade tensorflow
RUN pip install --upgrade torchvision
RUN pip install --upgrade tqdm
RUN pip install --upgrade youtube_dl
RUN pip install --opencv-python
