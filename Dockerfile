FROM nvcr.io/nvidia/pytorch:19.10-py3

RUN pip install --no-cache-dir runx==0.0.6
RUN pip install --no-cache-dir numpy
RUN pip install --no-cache-dir sklearn
RUN pip install --no-cache-dir h5py
RUN pip install --no-cache-dir jupyter
RUN pip install --no-cache-dir scikit-image
RUN pip install --no-cache-dir pillow
RUN pip install --no-cache-dir piexif
RUN pip install --no-cache-dir cffi
RUN pip install --no-cache-dir tqdm
RUN pip install --no-cache-dir dominate
RUN pip install --no-cache-dir opencv-python
RUN pip install --no-cache-dir nose
RUN pip install --no-cache-dir ninja

RUN apt-get update && apt-get install libgtk2.0-dev -y && rm -rf /var/lib/apt/lists/*

# Install Apex
RUN cd /home/ && git clone https://github.com/NVIDIA/apex.git apex && cd apex && python setup.py install --cuda_ext --cpp_ext
WORKDIR /home/
