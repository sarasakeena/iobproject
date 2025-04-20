# Use the official Python 3.9 slim image
FROM python:3.9-slim

# Install system dependencies for building dlib and other packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    wget \
    libgl1-mesa-glx \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt first to leverage Docker layer caching
COPY requirements.txt .

# Upgrade pip, setuptools, and wheel, then install the dependencies
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy the application code into the Docker image
COPY . .

# Expose port 8000 for the application
EXPOSE 8000

# Command to run the application
CMD ["python", "main.py", "--host=0.0.0.0", "--port=8000"]
