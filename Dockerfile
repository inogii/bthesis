FROM arm64v8/ubuntu:20.04

RUN apt-get update && apt-get install -y wget git libxcursor1 libxcursor-dev libxdamage1 libxdamage-dev vim tar curl autoconf automake libtool g++ make libpthread-stubs0-dev sudo librdkafka-dev zlib1g-dev libbz2-dev libcurl4-openssl-dev 

COPY . /app/
WORKDIR /app

RUN ldconfig

RUN curl -LO https://github.com/LibtraceTeam/wandio/archive/refs/tags/4.2.4-1.tar.gz
RUN tar zxf 4.2.4-1.tar.gz
WORKDIR /app/wandio-4.2.4-1
RUN ./bootstrap.sh 
RUN ./configure
RUN make
RUN sudo make install
RUN sudo ldconfig
WORKDIR /app

RUN curl -LO https://github.com/CAIDA/libbgpstream/releases/download/v2.2.0/libbgpstream-2.2.0.tar.gz
RUN tar zxf libbgpstream-2.2.0.tar.gz
WORKDIR /app/libbgpstream-2.2.0
RUN LIBPTHREAD_PATH='/lib/aarch64-linux-gnu/libc.so'
RUN ./configure --without-kafka
RUN make
RUN make check
RUN sudo make install
RUN sudo ldconfig
WORKDIR /app

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O miniconda.sh && \
    bash miniconda.sh -b -u -p /root/miniconda3 && \
    rm miniconda.sh

# Set environment variables for Miniconda
ENV PATH="/root/miniconda3/bin:${PATH}"

RUN conda create --name train -c conda-forge graph-tool
RUN conda install -n train -y python=3.11
RUN conda init

RUN conda install -n train -y pip

RUN /root/miniconda3/envs/train/bin/pip install pybgpstream

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get clean && apt-get update && apt-get install -y nodejs npm
RUN npm install sass

# install build utilities
RUN git clone https://github.com/inogii/nni.git
WORKDIR /app/nni
ENV NNI_RELEASE=3.0
RUN python setup.py build_ts
RUN python setup.py bdist_wheel
WORKDIR /app/nni/dist
RUN /root/miniconda3/envs/train/bin/pip install nni-3.0-py3-none-any.whl
WORKDIR /app

RUN apt-get install -y build-essential libssl-dev
RUN /root/miniconda3/envs/train/bin/pip install dgl -f https://data.dgl.ai/wheels/repo.html
RUN /root/miniconda3/envs/train/bin/pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html
RUN /root/miniconda3/envs/train/bin/pip install torch==2.0.1
RUN /root/miniconda3/envs/train/bin/pip install torchvision==0.15.0
RUN /root/miniconda3/envs/train/bin/pip install torch_geometric==2.3.0
RUN /root/miniconda3/envs/train/bin/pip install tqdm==4.65.0
RUN /root/miniconda3/envs/train/bin/pip install imbalanced_learn==0.10.1
RUN /root/miniconda3/envs/train/bin/pip install matplotlib==3.7.1
RUN /root/miniconda3/envs/train/bin/pip install networkx==3.0
RUN /root/miniconda3/envs/train/bin/pip install pycountry_convert==0.7
RUN /root/miniconda3/envs/train/bin/pip install tabulate==0.9.0
RUN /root/miniconda3/envs/train/bin/pip install xgboost==1.7.5
RUN /root/miniconda3/envs/train/bin/pip install gitpython==3.1
RUN /root/miniconda3/envs/train/bin/pip install pytricia==1.0.2

RUN /root/miniconda3/envs/train/bin/pip install git+https://github.com/inogii/scikit-learn@modified-v1.2.2

RUN echo "conda activate train" >> ~/.bashrc





