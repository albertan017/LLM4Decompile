# PyTorch base image with CUDA, cuDNN, and PyTorch preinstalled
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

# Set working directory
WORKDIR /app

ARG GHIDRA_VERSION=11.0.3
ARG GHIDRA_BUILD_DATE=20240410
ARG GHIDRA_NAME=ghidra_${GHIDRA_VERSION}_PUBLIC

ENV DEBIAN_FRONTEND=noninteractive
ENV GHIDRA_DIR=/app/ghidra/${GHIDRA_NAME}
ENV CONDA_ENV_NAME=llm4decompile

# Install system packages
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
 wget ca-certificates unzip git curl bzip2 build-essential vim \
 openjdk-17-jdk-headless tzdata libxext6 libxrender1 libxtst6 libxi6 \
 && rm -rf /var/lib/apt/lists/*

# Install Ghidra
RUN wget -q https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_${GHIDRA_VERSION}_build/${GHIDRA_NAME}_${GHIDRA_BUILD_DATE}.zip -O /tmp/ghidra.zip \
 && unzip /tmp/ghidra.zip -d /app/ghidra \
 && rm /tmp/ghidra.zip

# Add Ghidra to PATH
ENV PATH=${GHIDRA_DIR}:$PATH

# Create conda environment from base
RUN conda create -n ${CONDA_ENV_NAME} --clone base && conda clean -a -y

# Copy dependency file
COPY requirements-docker.txt .

# Install pip dependencies in the new conda environment
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate ${CONDA_ENV_NAME} && \
    pip install --no-cache-dir -r requirements-docker.txt && \
    conda clean -a -y

# Copy source code
COPY . .

# Add conda environment activation to bashrc
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> /etc/bash.bashrc && \
    echo "if [[ \$- == *i* ]]; then conda activate ${CONDA_ENV_NAME}; fi" >> /etc/bash.bashrc

# Create entrypoint script to activate conda environment
RUN echo '#!/bin/bash\n\
source /opt/conda/etc/profile.d/conda.sh\n\
conda activate llm4decompile\n\
exec "$@"' > /entrypoint.sh  \
 && chmod +x /entrypoint.sh

# Set container entrypoint
ENTRYPOINT ["/entrypoint.sh"]
SHELL ["/bin/bash"]
