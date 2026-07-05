from __future__ import annotations

import argparse
import json
import logging
import queue
import socket
import threading
import time
from dataclasses import dataclass

import numpy as np

from backend.media_gateway.protocol import (
    Codec,
    MediaPacket,
    PacketReassembler,
    StreamType,
    packetize_payload,
)


logger = logging.getLogger("preview_client")
DEFAULT_SESSION_ID = "default"


try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

try:
    import sounddevice as sd  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    sd = None


@dataclass(frozen=True)
class PreviewConfig:
    host: str
    gateway_host: str
    gateway_port: int
    session_id: bytes
    audio_port: int
    video_port: int
    audio_sample_rate: int
    audio_block_samples: int


def parse_args() -> PreviewConfig:
    parser = argparse.ArgumentParser(description="Preview client for media gateway audio/video outputs.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--gateway-host", required=True)
    parser.add_argument("--gateway-port", type=int, default=12000)
    parser.add_argument("--session-id", default=DEFAULT_SESSION_ID)
    parser.add_argument("--audio-port", type=int, default=11001)
    parser.add_argument("--video-port", type=int, default=11002)
    parser.add_argument("--audio-sample-rate", type=int, default=48000)
    parser.add_argument("--audio-block-samples", type=int, default=12000)
    args = parser.parse_args()
    return PreviewConfig(
        host=args.host,
        gateway_host=args.gateway_host,
        gateway_port=args.gateway_port,
        session_id=args.session_id.encode("utf-8")[:16].ljust(16, b"\x00"),
        audio_port=args.audio_port,
        video_port=args.video_port,
        audio_sample_rate=args.audio_sample_rate,
        audio_block_samples=args.audio_block_samples,
    )


def send_registration(sock: socket.socket, cfg: PreviewConfig, stream: str, port: int) -> None:
    payload = json.dumps({"kind": "register_return", "stream": stream, "port": port}).encode("utf-8")
    for packet in packetize_payload(
        stream_type=StreamType.CONTROL,
        codec=Codec.JSON,
        session_id=cfg.session_id,
        sequence_number=0,
        timestamp_us=time.time_ns() // 1000,
        payload=payload,
    ):
        sock.sendto(packet.to_bytes(), (cfg.gateway_host, cfg.gateway_port))


def registration_loop(sock: socket.socket, cfg: PreviewConfig, stream: str, port: int) -> None:
    logger.info("registering %s return path via udp://%s:%s", stream, cfg.gateway_host, cfg.gateway_port)
    while True:
        send_registration(sock, cfg, stream, port)
        time.sleep(2.0)


def audio_listener(cfg: PreviewConfig, audio_queue: queue.Queue) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((cfg.host, cfg.audio_port))
    threading.Thread(
        target=registration_loop,
        args=(sock, cfg, "audio", cfg.audio_port),
        daemon=True,
    ).start()
    reassembler = PacketReassembler()
    logger.info("listening for audio on udp://%s:%s", cfg.host, cfg.audio_port)
    while True:
        data, _ = sock.recvfrom(65535)
        packet = reassembler.push(MediaPacket.from_bytes(data))
        if packet is None:
            continue
        if packet.header.stream_type == StreamType.AUDIO and packet.header.codec == Codec.PCM16:
            audio_queue.put(packet.payload)


def video_listener(cfg: PreviewConfig, video_queue: queue.Queue) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((cfg.host, cfg.video_port))
    threading.Thread(
        target=registration_loop,
        args=(sock, cfg, "video", cfg.video_port),
        daemon=True,
    ).start()
    reassembler = PacketReassembler()
    logger.info("listening for video on udp://%s:%s", cfg.host, cfg.video_port)
    while True:
        data, _ = sock.recvfrom(2 * 1024 * 1024)
        packet = reassembler.push(MediaPacket.from_bytes(data))
        if packet is None:
            continue
        if packet.header.stream_type == StreamType.VIDEO:
            video_queue.put(packet.payload)


def audio_playback(audio_queue: queue.Queue, sample_rate: int, block_samples: int) -> None:
    if sd is None:
        logger.warning("sounddevice not installed; audio preview disabled")
        while True:
            audio_queue.get()
        return

    def callback(outdata, frames, time_info, status) -> None:
        del frames, time_info
        if status:
            logger.warning("audio playback status: %s", status)
        try:
            payload = audio_queue.get_nowait()
            pcm = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
            outdata[:, 0] = pcm[: outdata.shape[0]]
        except queue.Empty:
            outdata.fill(0)

    with sd.OutputStream(
        samplerate=sample_rate,
        blocksize=block_samples,
        channels=1,
        dtype="float32",
        callback=callback,
    ):
        while True:
            time.sleep(1)


def video_preview(video_queue: queue.Queue) -> None:
    if cv2 is None:
        logger.warning("opencv-python not installed; video preview disabled")
        while True:
            video_queue.get()
        return

    while True:
        payload = video_queue.get()
        np_buffer = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
        if frame is None:
            continue
        cv2.imshow("media-gateway-preview", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=64)
    video_queue: queue.Queue[bytes] = queue.Queue(maxsize=64)

    threads = [
        threading.Thread(target=audio_listener, args=(args, audio_queue), daemon=True),
        threading.Thread(target=video_listener, args=(args, video_queue), daemon=True),
        threading.Thread(
            target=audio_playback,
            args=(audio_queue, args.audio_sample_rate, args.audio_block_samples),
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()
    video_preview(video_queue)


if __name__ == "__main__":
    main()
