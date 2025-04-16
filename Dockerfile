# Base image
FROM ubuntu:24.10

# Avoid prompts during package installs
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies (you can add more as needed)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /app

# Copy all files, excluding some dirs using .dockerignore
COPY . /app

RUN pip install -r requirements.txt

# Default command
CMD ["/bin/bash"]
