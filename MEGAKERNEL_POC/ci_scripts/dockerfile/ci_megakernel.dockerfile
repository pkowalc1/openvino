FROM ubuntu:24.04

# Set proxies
ENV http_proxy=http://proxy-igk.intel.com:911 \
  https_proxy=http://proxy-igk.intel.com:912 \
  HTTP_PROXY=http://proxy-igk.intel.com:911 \
  HTTPS_PROXY=http://proxy-igk.intel.com:912 \
  no_proxy=localhost,127.0.0.1,sclab.intel.com,.corp.intel.com,corp.intel.com,ubit-artifactory-or.intel.com,.sclab.intel.com,devtools.intel.com,.devtools.intel.com,icloud.intel.com,.icloud.intel.com,appsecapi.intel.com,onecloudapi.intel.com,oneclouddemoapi.intel.com \
  NO_PROXY=localhost,127.0.0.1,sclab.intel.com,.corp.intel.com,corp.intel.com,ubit-artifactory-or.intel.com,.sclab.intel.com,devtools.intel.com,.devtools.intel.com,icloud.intel.com,.icloud.intel.com,appsecapi.intel.com,onecloudapi.intel.com,oneclouddemoapi.intel.com

RUN apt update

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN apt-get -y install tzdata
RUN apt-get install -y --no-install-recommends \
        `# for python3-pip` \
        ca-certificates \
        `# build tools` \
        build-essential \
        git \
        cmake \
        gzip \
        `# openvino main dependencies` \
        libtbb-dev \
        libpugixml-dev \
        `# OpenCL for GPU` \
        ocl-icd-opencl-dev \
        opencl-headers \
        rapidjson-dev \
        `# GPU plugin extensions` \
        libva-dev \
        `# For TF FE saved models` \
        libsnappy-dev \
        `# python API` \
        python3-pip \
        libpython3-dev \
        pybind11-dev \
        wget \
        gdb \
        vim \
        clinfo \
        python3-venv

# Install GPU UMD drivers and libs
RUN apt-get update && \
    apt-get install -y --no-install-recommends ocl-icd-libopencl1 && \
    apt-get clean ; \
    rm -rf /var/lib/apt/lists/* && rm -rf /tmp/*
RUN mkdir /tmp/gpu_deps && cd /tmp/gpu_deps && \
  wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.30.1/intel-igc-core-2_2.30.1+20950_amd64.deb && \
  wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.30.1/intel-igc-opencl-2_2.30.1+20950_amd64.deb && \
  wget https://github.com/intel/compute-runtime/releases/download/26.09.37435.1/intel-ocloc-dbgsym_26.09.37435.1-0_amd64.ddeb && \
  wget https://github.com/intel/compute-runtime/releases/download/26.09.37435.1/intel-ocloc_26.09.37435.1-0_amd64.deb && \
  wget https://github.com/intel/compute-runtime/releases/download/26.09.37435.1/intel-opencl-icd-dbgsym_26.09.37435.1-0_amd64.ddeb && \
  wget https://github.com/intel/compute-runtime/releases/download/26.09.37435.1/intel-opencl-icd_26.09.37435.1-0_amd64.deb && \
  wget https://github.com/intel/compute-runtime/releases/download/26.09.37435.1/libigdgmm12_22.9.0_amd64.deb && \
  wget https://github.com/intel/compute-runtime/releases/download/26.09.37435.1/libze-intel-gpu1-dbgsym_26.09.37435.1-0_amd64.ddeb && \
  wget https://github.com/intel/compute-runtime/releases/download/26.09.37435.1/libze-intel-gpu1_26.09.37435.1-0_amd64.deb

RUN cd /tmp/gpu_deps && dpkg -i *.deb
RUN rm -Rf /tmp/gpu_deps