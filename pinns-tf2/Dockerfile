# Use the NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Update and install essential dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    wget \
    curl \
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory to your project
WORKDIR /workspace

# Copy your project files into the container
COPY . /workspace

# Validate GPU access
RUN python3 -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"

# Keep the container running
CMD ["bash"]
