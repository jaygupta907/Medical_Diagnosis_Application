FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

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

#Cope requirements.txt
COPY requirements.txt /app

#Copy all python files
COPY *.py /app

COPY training /app/training
COPY tuning /app/tuning


# Install Python dependencies
RUN pip3 install .

# Expose FastAPI port (optional but recommended)
EXPOSE 8000

RUN python3 download.py

CMD ["python3", "server.py"]
