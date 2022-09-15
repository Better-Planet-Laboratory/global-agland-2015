FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y && apt-get clean

RUN apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    sqlite3 \
    gfortran \
    libbz2-dev \
    libsqlite3-dev \
    libtiff-dev \
    libcurl4-openssl-dev \
    libpq-dev \
    libpcre2-dev \
    libgeos-dev \
    libhdf4-dev \
    pkg-config \
    cmake \
    r-base \
    r-cran-randomforest \
    xvfb \
    xauth \
    xfonts-base \
    python3.7 \
    python3.7-distutils \
    python3.7-dev \
    default-jre-headless

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1
RUN update-alternatives --set python /usr/bin/python3.7

RUN curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py --force-reinstall && \
    rm get-pip.py

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
RUN ./configure --with-python --with-pg --with-geos && \
    make && \
    make install && \
    ldconfig

RUN wget https://cran.r-project.org/src/base/R-4/R-4.0.2.tar.gz
RUN tar -xf R-4.0.2.tar.gz
WORKDIR ./R-4.0.2
RUN ./configure --with-readline=no --with-x=no && \
    make -j9 && \
    make install && \
    ldconfig

WORKDIR /global-agland-2015/
RUN python -m pip install --upgrade pip
COPY requirements.txt .
COPY constraints.txt .
RUN pip install -c constraints.txt markupsafe==2.0.1 numpy==1.21.6
RUN pip install -r requirements.txt

COPY requirements.r .
RUN Rscript requirements.r