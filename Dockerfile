# Dockerfile
FROM python:3.9-slim

# Install system dependencies for dlib/OpenCV
RUN apt-get update && apt-get install -y \
    cmake \
    libboost-all-dev \
    libx11-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y libpq-dev
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "main.py"]  # Use JSON array syntax