# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      ffmpeg \
      libportaudio2 \
      python3 \
      python3-dev \
      python3-pip \
      python3-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 13000

CMD ["python3", "-m", "backend.media_gateway.stream_server", "--host", "0.0.0.0", "--port", "13000", "--audio-model-path", "assets/weights/voice_model.pth", "--audio-index-path", "assets/indices/voice_model.index"]
