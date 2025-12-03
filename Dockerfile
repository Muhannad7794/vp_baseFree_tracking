# Dockerfile
FROM python:3.11

# Prevent Python from buffering stdout (nicer logs)
ENV PYTHONUNBUFFERED=1

WORKDIR /vp_basefree_tracking_dockerized

# Install system deps (optional, but good to have for scientific stack)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project code (but NOT the data directory – that will be mounted)
COPY . .
