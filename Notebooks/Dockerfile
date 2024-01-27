# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install system dependencies
# This might include software-properties-common, wget, or other utilities needed for Juicify.
RUN apt-get update && apt-get install -y \
  libvulkan1 libgl1 libglib2.0-0 build-essential wget \
  && rm -rf /var/lib/apt/lists/*

# Download and install Juicify
RUN wget https://juice-public-storage.s3.us-east-2.amazonaws.com/2023.12.14-1912.bf1ae07e/juice-gpu-linux.tar.gz \
  && tar -xzvf juice-gpu-linux.tar.gz \
  && mv juice /usr/local/bin \
  && mv juicify /usr/local/bin \
  && mv lib* /usr/local/lib \
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



# Use Juicify to run Jupyter Notebook with the specified command
CMD ["sh", "-c", "juicify --address electra.juicelabs.co --access-token  --pool-id your_pool_id --log-file stdout -- jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root"]
