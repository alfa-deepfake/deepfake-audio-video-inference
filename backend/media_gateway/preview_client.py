from __future__ import annotations

import argparse
import logging
import queue
import socket
import threading
import time

import numpy as np

from backend.media_gateway.protocol import Codec, MediaPacket, PacketReassembler, StreamType


logger = logging.getLogger("preview_client")


try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

try:
    import sounddevice as sd  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    sd = None


def parse_args():
    parser = argparse.ArgumentParser(description="Preview client for media gateway audio/video outputs.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--audio-port", type=int, default=11001)
    parser.add_argument("--video-port", type=int, default=11002)
    parser.add_argument("--audio-sample-rate", type=int, default=48000)
    parser.add_argument("--audio-block-samples", type=int, default=12000)
    return parser.parse_args()


def audio_listener(host: str, port: int, audio_queue: queue.Queue) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    reassembler = PacketReassembler()
    logger.info("listening for audio on udp://%s:%s", host, port)
    while True:
        data, _ = sock.recvfrom(65535)
        packet = reassembler.push(MediaPacket.from_bytes(data))
        if packet is None:
            continue
        if packet.header.stream_type == StreamType.AUDIO and packet.header.codec == Codec.PCM16:
            audio_queue.put(packet.payload)


def video_listener(host: str, port: int, video_queue: queue.Queue) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    reassembler = PacketReassembler()
    logger.info("listening for video on udp://%s:%s", host, port)
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
        threading.Thread(target=audio_listener, args=(args.host, args.audio_port, audio_queue), daemon=True),
        threading.Thread(target=video_listener, args=(args.host, args.video_port, video_queue), daemon=True),
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
