from __future__ import annotations

import argparse
import logging
import queue
import socket
import threading
from dataclasses import dataclass, field

from deepfake_media_transport import (
    Codec,
    MediaPacket,
    PacketReassembler,
    StreamType,
    packetize_payload,
    read_frame,
    write_frame,
)

from backend.media_gateway.audio_engine import AudioEngineConfig, AudioInferenceEngine
from backend.media_gateway.signature_cli import add_signature_verifier_args, trusted_signature_keys_from_args
from backend.media_gateway.signature_policy import verify_packet_signature
from backend.media_gateway.stream_signature import (
    SignatureStatus,
    StreamSignatureVerifier,
    VerificationResult,
)
from backend.media_gateway.video_engine import VideoEngineConfig, VideoInferenceEngine


logger = logging.getLogger("media_gateway.stream_server")


@dataclass(frozen=True)
class StreamServerConfig:
    host: str = "127.0.0.1"
    port: int = 13000
    audio_model_path: str = ""
    audio_index_path: str = ""
    audio_sample_rate: int = 48000
    audio_block_time: float = 0.25
    audio_index_rate: float = 0.0
    audio_f0method: str = "fcpe"
    video_dlc_root: str = ""
    video_source_face: str = ""
    video_execution_provider: str = "cuda"
    video_camera_fps: float = 15.0
    video_python_path: str = ""
    video_cuda_lib_root: str = ""
    signature_policy: str = "off"
    trusted_signature_keys: dict[str, bytes] = field(default_factory=dict)


class StreamServer:
    def __init__(self, cfg: StreamServerConfig) -> None:
        self.cfg = cfg
        self.audio_engine = AudioInferenceEngine(
            AudioEngineConfig(
                model_path=cfg.audio_model_path,
                index_path=cfg.audio_index_path,
                sample_rate=cfg.audio_sample_rate,
                block_time=cfg.audio_block_time,
                index_rate=cfg.audio_index_rate,
                f0method=cfg.audio_f0method,
            )
        )
        self.video_engine = (
            VideoInferenceEngine(
                VideoEngineConfig(
                    deep_live_cam_root=cfg.video_dlc_root,
                    python_path=cfg.video_python_path or None,
                    execution_provider=cfg.video_execution_provider,
                    camera_fps=cfg.video_camera_fps,
                    source_face_path=cfg.video_source_face or None,
                    cuda_lib_root=cfg.video_cuda_lib_root or None,
                )
            )
            if cfg.video_dlc_root
            else None
        )

    def serve_forever(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.cfg.host, self.cfg.port))
        server.listen()
        logger.info("stream server listening on tcp://%s:%s", self.cfg.host, self.cfg.port)
        while True:
            client, addr = server.accept()
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            logger.info("stream client connected: %s", addr)
            StreamSession(self, client, addr).start()


class StreamSession:
    def __init__(self, server: StreamServer, sock: socket.socket, addr) -> None:
        self.server = server
        self.sock = sock
        self.addr = addr
        self.reassembler = PacketReassembler()
        self.audio_queue: queue.Queue[MediaPacket] = queue.Queue(maxsize=4)
        self.video_queue: queue.Queue[MediaPacket] = queue.Queue(maxsize=1)
        self.write_lock = threading.Lock()
        self.signature_verifier = StreamSignatureVerifier(server.cfg.trusted_signature_keys)
        self.signature_status_logged: set[tuple[int, SignatureStatus, str]] = set()

    def start(self) -> None:
        for target in (self.audio_worker, self.video_worker, self.reader):
            threading.Thread(target=target, daemon=True).start()

    def reader(self) -> None:
        try:
            while True:
                packet = self.reassembler.push(MediaPacket.from_bytes(read_frame(self.sock)))
                if packet is None:
                    continue
                verification = self.verify_packet(packet)
                if verification is None:
                    continue
                packet = verification.packet
                if packet.header.stream_type == StreamType.AUDIO:
                    put_latest(self.audio_queue, packet)
                elif packet.header.stream_type == StreamType.VIDEO:
                    put_latest(self.video_queue, packet)
                else:
                    logger.info("control from %s: %s", self.addr, packet.payload.decode("utf-8", errors="replace"))
        except Exception as exc:
            logger.info("stream client disconnected %s: %s", self.addr, exc)
            self.sock.close()

    def verify_packet(self, packet: MediaPacket) -> VerificationResult | None:
        return verify_packet_signature(
            packet=packet,
            verifier=self.signature_verifier,
            policy=self.server.cfg.signature_policy,
            logger=logger,
            peer=self.addr,
            logged_statuses=self.signature_status_logged,
            color=True,
        )

    def audio_worker(self) -> None:
        while True:
            packet = self.audio_queue.get()
            if packet.header.codec != Codec.PCM16:
                logger.warning("unsupported audio codec from %s: %s", self.addr, packet.header.codec)
                continue
            output = self.server.audio_engine.process_pcm16(packet.payload)
            self.write_responses(
                packetize_payload(
                    stream_type=StreamType.AUDIO,
                    codec=Codec.PCM16,
                    session_id=packet.header.session_id,
                    sequence_number=packet.header.sequence_number,
                    timestamp_us=packet.header.timestamp_us,
                    payload=output,
                )
            )

    def video_worker(self) -> None:
        while True:
            packet = self.video_queue.get()
            output = packet.payload
            if self.server.video_engine is not None and packet.header.codec == Codec.MJPEG:
                output = self.server.video_engine.process_mjpeg(packet.payload)
            self.write_responses(
                packetize_payload(
                    stream_type=StreamType.VIDEO,
                    codec=packet.header.codec,
                    session_id=packet.header.session_id,
                    sequence_number=packet.header.sequence_number,
                    timestamp_us=packet.header.timestamp_us,
                    payload=output,
                )
            )

    def write_responses(self, packets: list[MediaPacket]) -> None:
        with self.write_lock:
            for packet in packets:
                write_frame(self.sock, packet.to_bytes())


def put_latest(packet_queue: queue.Queue, packet: MediaPacket) -> None:
    try:
        packet_queue.put_nowait(packet)
    except queue.Full:
        try:
            packet_queue.get_nowait()
        except queue.Empty:
            pass
        packet_queue.put_nowait(packet)


def parse_args() -> StreamServerConfig:
    parser = argparse.ArgumentParser(description="Low-latency stream server for SSH-tunneled media inference.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=13000)
    parser.add_argument("--audio-model-path", required=True)
    parser.add_argument("--audio-index-path", default="")
    parser.add_argument("--audio-sample-rate", type=int, default=48000)
    parser.add_argument("--audio-block-time", type=float, default=0.25)
    parser.add_argument("--audio-index-rate", type=float, default=0.0)
    parser.add_argument("--audio-f0method", default="fcpe")
    parser.add_argument("--video-dlc-root", default="")
    parser.add_argument("--video-source-face", default="")
    parser.add_argument("--video-execution-provider", default="cuda")
    parser.add_argument("--video-camera-fps", type=float, default=15.0)
    parser.add_argument("--video-python-path", default="")
    parser.add_argument("--video-cuda-lib-root", default="")
    add_signature_verifier_args(parser)
    args = parser.parse_args()
    return StreamServerConfig(
        host=args.host,
        port=args.port,
        audio_model_path=args.audio_model_path,
        audio_index_path=args.audio_index_path,
        audio_sample_rate=args.audio_sample_rate,
        audio_block_time=args.audio_block_time,
        audio_index_rate=args.audio_index_rate,
        audio_f0method=args.audio_f0method,
        video_dlc_root=args.video_dlc_root,
        video_source_face=args.video_source_face,
        video_execution_provider=args.video_execution_provider,
        video_camera_fps=args.video_camera_fps,
        video_python_path=args.video_python_path,
        video_cuda_lib_root=args.video_cuda_lib_root,
        signature_policy=args.signature_policy,
        trusted_signature_keys=trusted_signature_keys_from_args(args),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    StreamServer(parse_args()).serve_forever()


if __name__ == "__main__":
    main()
