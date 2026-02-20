FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    git-lfs \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Initialization for Git LFS
RUN git lfs install

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Copy application code
COPY . .

# Pull LFS files specifically because HF custom Dockerfiles don't do it automatically
RUN git lfs pull

# Expose Streamlit port
EXPOSE 7860

# Run the start script directly with bash
CMD ["bash", "start.sh"]
