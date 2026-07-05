from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import dataclass

from backend.media_gateway.audio_engine import AudioEngineConfig, AudioInferenceEngine
from backend.media_gateway.protocol import (
    Codec,
    MediaPacket,
    PacketHeader,
    PacketReassembler,
    StreamType,
    packetize_payload,
)
from backend.media_gateway.session import SessionState
from backend.media_gateway.video_engine import VideoEngineConfig, VideoInferenceEngine


logger = logging.getLogger("media_gateway")


@dataclass(frozen=True)
class GatewayConfig:
    host: str = "0.0.0.0"
    port: int = 11000
    audio_return_port_offset: int = 1
    video_return_port_offset: int = 2
    audio_model_path: str = ""
    audio_index_path: str = ""
    audio_sample_rate: int = 48000
    audio_block_time: float = 0.25
    audio_index_rate: float = 0.0
    audio_f0method: str = "fcpe"
    video_dlc_root: str = ""
    video_source_face: str = ""
    video_execution_provider: str = "cuda"
    video_camera_fps: float = 20.0
    video_python_path: str = ""
    video_cuda_lib_root: str = ""


class MediaGatewayProtocol(asyncio.DatagramProtocol):
    def __init__(self, cfg: GatewayConfig) -> None:
        self.cfg = cfg
        self.transport = None
        self.sessions: dict[bytes, SessionState] = {}
        self.reassembler = PacketReassembler()
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

    def connection_made(self, transport) -> None:
        self.transport = transport
        logger.info("media gateway listening on udp://%s:%s", self.cfg.host, self.cfg.port)

    def datagram_received(self, data: bytes, addr) -> None:
        try:
            packet = self.reassembler.push(MediaPacket.from_bytes(data))
        except Exception as exc:
            logger.warning("dropping invalid packet from %s: %s", addr, exc)
            return
        if packet is None:
            return

        session = self.sessions.setdefault(
            packet.header.session_id, SessionState(session_id=packet.header.session_id)
        )
        session.touch(addr)

        if packet.header.stream_type == StreamType.AUDIO:
            self._handle_audio(packet, addr, session)
        elif packet.header.stream_type == StreamType.VIDEO:
            self._handle_video(packet, addr, session)
        else:
            self._handle_control(packet, addr, session)

    def _handle_audio(self, packet: MediaPacket, addr, session: SessionState) -> None:
        session.audio.packets_received += 1
        session.audio.last_sequence_number = packet.header.sequence_number
        session.audio.last_timestamp_us = packet.header.timestamp_us
        if packet.header.codec != Codec.PCM16:
            logger.warning("unsupported audio codec from %s: %s", addr, packet.header.codec)
            return
        output_payload = self.audio_engine.process_pcm16(packet.payload)
        return_addr = session.audio_return_addr or (addr[0], addr[1] + self.cfg.audio_return_port_offset)
        for response in packetize_payload(
            stream_type=StreamType.AUDIO,
            codec=Codec.PCM16,
            session_id=packet.header.session_id,
            sequence_number=packet.header.sequence_number,
            timestamp_us=packet.header.timestamp_us,
            payload=output_payload,
        ):
            self.transport.sendto(response.to_bytes(), return_addr)

    def _handle_video(self, packet: MediaPacket, addr, session: SessionState) -> None:
        session.video.packets_received += 1
        session.video.last_sequence_number = packet.header.sequence_number
        session.video.last_timestamp_us = packet.header.timestamp_us
        output_payload = packet.payload
        if self.video_engine is not None and packet.header.codec == Codec.MJPEG:
            output_payload = self.video_engine.process_mjpeg(packet.payload)
        return_addr = session.video_return_addr or (addr[0], addr[1] + self.cfg.video_return_port_offset)
        for response in packetize_payload(
            stream_type=StreamType.VIDEO,
            codec=packet.header.codec,
            session_id=packet.header.session_id,
            sequence_number=packet.header.sequence_number,
            timestamp_us=packet.header.timestamp_us,
            payload=output_payload,
        ):
            self.transport.sendto(response.to_bytes(), return_addr)
        logger.debug(
            "video passthrough session=%s seq=%s codec=%s size=%s",
            packet.header.session_id.hex(),
            packet.header.sequence_number,
            packet.header.codec.name,
            len(packet.payload),
        )

    def _handle_control(self, packet: MediaPacket, addr, session: SessionState) -> None:
        try:
            payload = json.loads(packet.payload.decode("utf-8"))
        except Exception:
            payload = {"raw": packet.payload.decode("utf-8", errors="replace")}
        if payload.get("kind") == "register_return":
            stream = payload.get("stream")
            if stream == "audio":
                session.audio_return_addr = addr
                logger.info("registered audio return addr for session=%s -> %s", session.session_id.hex(), addr)
                return
            if stream == "video":
                session.video_return_addr = addr
                logger.info("registered video return addr for session=%s -> %s", session.session_id.hex(), addr)
                return
        logger.info("control packet from %s session=%s payload=%s", addr, session.session_id.hex(), payload)


def parse_args() -> GatewayConfig:
    parser = argparse.ArgumentParser(description="Realtime UDP media gateway for voice and face deepfake pipelines.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=11000)
    parser.add_argument("--audio-model-path", required=True)
    parser.add_argument("--audio-index-path", default="")
    parser.add_argument("--audio-sample-rate", type=int, default=48000)
    parser.add_argument("--audio-block-time", type=float, default=0.25)
    parser.add_argument("--audio-index-rate", type=float, default=0.0)
    parser.add_argument("--audio-f0method", default="fcpe")
    parser.add_argument("--video-dlc-root", default="")
    parser.add_argument("--video-source-face", default="")
    parser.add_argument("--video-execution-provider", default="cuda")
    parser.add_argument("--video-camera-fps", type=float, default=20.0)
    parser.add_argument("--video-python-path", default="")
    parser.add_argument("--video-cuda-lib-root", default="")
    args = parser.parse_args()
    return GatewayConfig(
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
    )


async def run_gateway(cfg: GatewayConfig) -> None:
    loop = asyncio.get_running_loop()
    await loop.create_datagram_endpoint(
        lambda: MediaGatewayProtocol(cfg),
        local_addr=(cfg.host, cfg.port),
    )
    while True:
        await asyncio.sleep(3600)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = parse_args()
    asyncio.run(run_gateway(cfg))


if __name__ == "__main__":
    main()
