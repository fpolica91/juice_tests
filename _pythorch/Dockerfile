# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
  libvulkan1 libgl1 libglib2.0-0 build-essential wget libatomic1 \
  && rm -rf /var/lib/apt/lists/*

# Download and install Juicify
RUN wget https://juice-public-storage.s3.us-east-2.amazonaws.com/2023.12.14-1912.bf1ae07e/juice-gpu-linux.tar.gz \
  && tar -xzvf juice-gpu-linux.tar.gz -C /usr/local/bin \
  && ldconfig

# Add the local bin directory to PATH for any user scripts
ENV PATH="/usr/local/bin:${PATH}"

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Since we want to run the container interactively, we set the entrypoint to a shell
# This will allow us to execute the juicify command or any other command interactively
ENTRYPOINT ["/bin/bash"]

