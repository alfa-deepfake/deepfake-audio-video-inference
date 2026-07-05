from __future__ import annotations

import argparse
import logging
import socket
import threading
import time
import uuid
import wave
from dataclasses import dataclass
from typing import Optional

import numpy as np

from backend.media_gateway.protocol import Codec, StreamType, packetize_payload


logger = logging.getLogger("capture_client")


try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

try:
    import sounddevice as sd  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    sd = None


@dataclass(frozen=True)
class CaptureConfig:
    gateway_host: str
    gateway_port: int
    session_id: bytes
    audio_sample_rate: int
    audio_block_samples: int
    audio_device: Optional[int]
    video_device: int
    video_width: int
    video_height: int
    video_fps: float
    jpeg_quality: int
    source_wav: Optional[str]


def parse_args() -> CaptureConfig:
    parser = argparse.ArgumentParser(description="Send webcam and microphone streams to the media gateway over UDP.")
    parser.add_argument("--gateway-host", required=True)
    parser.add_argument("--gateway-port", type=int, default=11000)
    parser.add_argument("--session-id", default="")
    parser.add_argument("--audio-sample-rate", type=int, default=48000)
    parser.add_argument("--audio-block-samples", type=int, default=12000)
    parser.add_argument("--audio-device", type=int, default=None)
    parser.add_argument("--video-device", type=int, default=0)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=360)
    parser.add_argument("--video-fps", type=float, default=20.0)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--source-wav", default=None, help="Read audio from wav file instead of microphone")
    args = parser.parse_args()
    session_id = args.session_id.encode("utf-8")[:16] if args.session_id else uuid.uuid4().bytes
    session_id = session_id.ljust(16, b"\x00")
    return CaptureConfig(
        gateway_host=args.gateway_host,
        gateway_port=args.gateway_port,
        session_id=session_id,
        audio_sample_rate=args.audio_sample_rate,
        audio_block_samples=args.audio_block_samples,
        audio_device=args.audio_device,
        video_device=args.video_device,
        video_width=args.video_width,
        video_height=args.video_height,
        video_fps=args.video_fps,
        jpeg_quality=args.jpeg_quality,
        source_wav=args.source_wav,
    )


class UdpMediaSender:
    def __init__(self, cfg: CaptureConfig) -> None:
        self.cfg = cfg
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.audio_sequence = 0
        self.video_sequence = 0

    def send_audio(self, payload: bytes) -> None:
        for packet in packetize_payload(
            stream_type=StreamType.AUDIO,
            codec=Codec.PCM16,
            session_id=self.cfg.session_id,
            sequence_number=self.audio_sequence,
            timestamp_us=time.time_ns() // 1000,
            payload=payload,
        ):
            self.sock.sendto(packet.to_bytes(), (self.cfg.gateway_host, self.cfg.gateway_port))
        self.audio_sequence += 1

    def send_video(self, payload: bytes) -> None:
        for packet in packetize_payload(
            stream_type=StreamType.VIDEO,
            codec=Codec.MJPEG,
            session_id=self.cfg.session_id,
            sequence_number=self.video_sequence,
            timestamp_us=time.time_ns() // 1000,
            payload=payload,
        ):
            self.sock.sendto(packet.to_bytes(), (self.cfg.gateway_host, self.cfg.gateway_port))
        self.video_sequence += 1


def microphone_loop(cfg: CaptureConfig, sender: UdpMediaSender) -> None:
    if cfg.source_wav:
        wav_loop(cfg, sender)
        return
    if sd is None:
        raise RuntimeError("sounddevice is required for microphone capture")

    def callback(indata, frames, time_info, status) -> None:
        del time_info
        if status:
            logger.warning("audio status: %s", status)
        mono = indata[:, 0]
        pcm = np.clip(mono, -1.0, 1.0)
        sender.send_audio((pcm * 32767.0).astype("<i2", copy=False).tobytes())

    with sd.InputStream(
        samplerate=cfg.audio_sample_rate,
        blocksize=cfg.audio_block_samples,
        device=cfg.audio_device,
        channels=1,
        dtype="float32",
        callback=callback,
    ):
        logger.info("microphone capture started")
        while True:
            time.sleep(1)


def wav_loop(cfg: CaptureConfig, sender: UdpMediaSender) -> None:
    with wave.open(cfg.source_wav, "rb") as wav_file:
        if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
            raise ValueError("source wav must be mono 16-bit PCM")
        if wav_file.getframerate() != cfg.audio_sample_rate:
            raise ValueError("source wav sample rate must match audio-sample-rate")
        block_bytes = cfg.audio_block_samples * 2
        while True:
            chunk = wav_file.readframes(cfg.audio_block_samples)
            if not chunk:
                wav_file.rewind()
                continue
            if len(chunk) < block_bytes:
                chunk += b"\x00" * (block_bytes - len(chunk))
            sender.send_audio(chunk)
            time.sleep(cfg.audio_block_samples / cfg.audio_sample_rate)


def webcam_loop(cfg: CaptureConfig, sender: UdpMediaSender) -> None:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for webcam capture")
    cap = cv2.VideoCapture(cfg.video_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.video_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.video_height)
    cap.set(cv2.CAP_PROP_FPS, cfg.video_fps)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open webcam device {cfg.video_device}")

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), cfg.jpeg_quality]
    frame_interval = 1.0 / cfg.video_fps
    logger.info("webcam capture started")
    try:
        while True:
            started = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                logger.warning("failed to read webcam frame")
                continue
            ok, encoded = cv2.imencode(".jpg", frame, encode_params)
            if ok:
                sender.send_video(encoded.tobytes())
            elapsed = time.perf_counter() - started
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
    finally:
        cap.release()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = parse_args()
    sender = UdpMediaSender(cfg)
    threads = [
        threading.Thread(target=microphone_loop, args=(cfg, sender), daemon=True),
        threading.Thread(target=webcam_loop, args=(cfg, sender), daemon=True),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
