FROM debian:bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System packages:
# - python3 + pip: runtime
# - python3-picamera2 + python3-libcamera + libcamera-apps: Pi Camera Module 3 on Pi 5
# - alsa-utils: provides aplay for deterrent sound playback
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-picamera2 \
    python3-libcamera \
    libcamera-apps \
    alsa-utils \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages -r /app/requirements.txt

COPY main.py /app/main.py

# Create mount points for model and sound assets.
RUN mkdir -p /app/models /app/assets

CMD ["python3", "/app/main.py"]
