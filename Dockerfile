FROM ubuntu:18.04

RUN apt-get update && apt-get upgrade -y && apt-get clean
RUN apt-get install -y curl python3.7 python3.7-dev python3.7-distutils
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1
RUN update-alternatives --set python /usr/bin/python3.7

RUN curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py --force-reinstall && \
    rm get-pip.py

# RUN apt-get update && \
#     DEBIAN_FRONTEND=noninteractive \
#     apt-get -y install default-jre-headless && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*

RUN DEBIAN_FRONTEND=noninteractive \
    apt-get -y install default-jre-headless
    #  && \
    # apt-get clean && \
    # rm -rf /var/lib/apt/lists/*

RUN apt-get install -y \
    build-essential \
    python-all-dev \
    libpq-dev \
    libgeos-dev \
    wget \
    curl \
    sqlite3 \
    cmake \
    libtiff-dev \
    libsqlite3-dev \
    libcurl4-openssl-dev \
    pkg-config \
    ffmpeg \
    libsm6 \
    libxext6 
    # openjdk-17-jdk \
    # openjdk-17-jre

RUN curl https://download.osgeo.org/proj/proj-8.2.1.tar.gz | tar -xz &&\
    cd proj-8.2.1 &&\
    mkdir build &&\
    cd build && \
    cmake .. &&\
    make && \
    make install

RUN wget http://download.osgeo.org/gdal/3.4.1/gdal-3.4.1.tar.gz
RUN tar xvfz gdal-3.4.1.tar.gz
WORKDIR ./gdal-3.4.1
RUN ./configure --with-python --with-pg --with-geos &&\
    make && \
    make install && \
    ldconfig

# RUN export CPLUS_INCLUDE_PATH=/usr/include/gdal
# RUN export C_INCLUDE_PATH=/usr/include/gdal

WORKDIR /global-agland-2015/
RUN python -m pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt
