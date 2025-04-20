# Use the official Python 3.9 slim image
FROM python:3.9-slim

# Avoids prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for building dlib and other packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    g++ \
    wget \
    curl \
    git \
    libgl1-mesa-glx \
    libboost-all-dev \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt first to leverage Docker layer caching
COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel --timeout=100 \
    && pip install --no-cache-dir -r requirements.txt --timeout=100

# Copy the rest of the application
COPY . .

# Expose port 8000 for the application
EXPOSE 8000

# Run the application
CMD ["python", "main.py", "--host=0.0.0.0", "--port=8000"]
