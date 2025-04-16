# Base image (slim version)
FROM ubuntu:22.04

# Avoid prompts during package installs
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Expose FastAPI port (optional but recommended)
EXPOSE 8000

# Default command
CMD ["python3", "server.py"]
