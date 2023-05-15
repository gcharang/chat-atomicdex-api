FROM python:3.9.16

# Install some basic utilities.
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    build-essential \
    libssl-dev \
    cmake \
    gcc \
    g++ \
    wget \
 && rm -rf /var/lib/apt/lists/*

 # Create a working directory.
RUN mkdir /app
WORKDIR /app

COPY ./requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

#CMD ["gunicorn", "main:app", "--bind", "[::]:8080", "-k", "uvicorn.workers.UvicornWorker"]