FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=100

# Install system packages + Rust (and everything dlib needs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    build-essential \
    git \
    cmake \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    python3-dev \
    pkg-config \
    libffi-dev \
    libssl-dev \
    ffmpeg \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (for packages like cryptography, or others needing cargo)
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Optional: Install CMake manually if specific version is needed
RUN wget -q https://github.com/Kitware/CMake/releases/download/v3.22.1/cmake-3.22.1-linux-x86_64.sh \
    && mkdir /opt/cmake \
    && sh cmake-3.22.1-linux-x86_64.sh --prefix=/opt/cmake --skip-license \
    && ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake \
    && rm cmake-3.22.1-linux-x86_64.sh

# Set work directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
COPY models/shape_predictor_68_face_landmarks.dat /app/models/


RUN pip install --upgrade pip==24.0 setuptools==70.0.0 wheel
RUN pip install --no-cache-dir -r requirements.txt --timeout=300

# Copy the rest of your code
COPY . .

# Expose port
EXPOSE 8000

# Run app
CMD ["python", "main.py", "--host=0.0.0.0", "--port=8000"]
