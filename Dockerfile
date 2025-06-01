# Gunakan image dasar yang ringan
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install dependencies sistem & git
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# Salin file project ke container
COPY MLProject/ .

# Install MLflow (gunakan pip)
RUN pip install --upgrade pip && pip install mlflow

# Set default command
ENTRYPOINT ["mlflow", "run", "."]
