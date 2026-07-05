from __future__ import annotations

import enum
import struct
from dataclasses import dataclass
from typing import Dict, Tuple


MAGIC = b"DF"
VERSION = 1
HEADER_STRUCT = struct.Struct("!2sBB16sQQHIHH")
MAX_DATAGRAM_SIZE = 60_000
MAX_PAYLOAD_SIZE = MAX_DATAGRAM_SIZE - HEADER_STRUCT.size


class StreamType(enum.IntEnum):
    AUDIO = 1
    VIDEO = 2
    CONTROL = 3


class Codec(enum.IntEnum):
    PCM16 = 1
    MJPEG = 2
    H264 = 3
    JSON = 255


@dataclass(frozen=True)
class PacketHeader:
    stream_type: StreamType
    codec: Codec
    session_id: bytes
    sequence_number: int
    timestamp_us: int
    payload_size: int
    fragment_index: int = 0
    fragment_count: int = 1

    def to_bytes(self) -> bytes:
        return HEADER_STRUCT.pack(
            MAGIC,
            VERSION,
            int(self.stream_type),
            self.session_id,
            self.sequence_number,
            self.timestamp_us,
            int(self.codec),
            self.payload_size,
            self.fragment_index,
            self.fragment_count,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "PacketHeader":
        if len(data) < HEADER_STRUCT.size:
            raise ValueError("packet too short for media gateway header")
        magic, version, stream_type, session_id, sequence_number, timestamp_us, codec, payload_size, fragment_index, fragment_count = HEADER_STRUCT.unpack(
            data[: HEADER_STRUCT.size]
        )
        if magic != MAGIC:
            raise ValueError("invalid packet magic")
        if version != VERSION:
            raise ValueError(f"unsupported protocol version: {version}")
        return cls(
            stream_type=StreamType(stream_type),
            codec=Codec(codec),
            session_id=session_id,
            sequence_number=sequence_number,
            timestamp_us=timestamp_us,
            payload_size=payload_size,
            fragment_index=fragment_index,
            fragment_count=fragment_count,
        )


@dataclass(frozen=True)
class MediaPacket:
    header: PacketHeader
    payload: bytes

    def to_bytes(self) -> bytes:
        if len(self.payload) != self.header.payload_size:
            raise ValueError("payload length does not match header")
        return self.header.to_bytes() + self.payload

    @classmethod
    def from_bytes(cls, data: bytes) -> "MediaPacket":
        header = PacketHeader.from_bytes(data)
        payload = data[HEADER_STRUCT.size :]
        if len(payload) != header.payload_size:
            raise ValueError("payload size mismatch")
        return cls(header=header, payload=payload)


def packetize_payload(
    *,
    stream_type: StreamType,
    codec: Codec,
    session_id: bytes,
    sequence_number: int,
    timestamp_us: int,
    payload: bytes,
    max_payload_size: int = MAX_PAYLOAD_SIZE,
) -> list[MediaPacket]:
    if max_payload_size <= 0:
        raise ValueError("max_payload_size must be positive")
    if not payload:
        chunks = [b""]
    else:
        chunks = [
            payload[start : start + max_payload_size]
            for start in range(0, len(payload), max_payload_size)
        ]
    fragment_count = len(chunks)
    return [
        MediaPacket(
            header=PacketHeader(
                stream_type=stream_type,
                codec=codec,
                session_id=session_id,
                sequence_number=sequence_number,
                timestamp_us=timestamp_us,
                payload_size=len(chunk),
                fragment_index=fragment_index,
                fragment_count=fragment_count,
            ),
            payload=chunk,
        )
        for fragment_index, chunk in enumerate(chunks)
    ]


class PacketReassembler:
    def __init__(self) -> None:
        self._pending: Dict[Tuple[bytes, int, int, int], list[bytes | None]] = {}

    def push(self, packet: MediaPacket) -> MediaPacket | None:
        if packet.header.fragment_count <= 1:
            return packet
        key = (
            packet.header.session_id,
            int(packet.header.stream_type),
            packet.header.sequence_number,
            packet.header.timestamp_us,
        )
        parts = self._pending.setdefault(
            key,
            [None] * packet.header.fragment_count,
        )
        if len(parts) != packet.header.fragment_count:
            self._pending.pop(key, None)
            raise ValueError("fragment count mismatch")
        if packet.header.fragment_index >= packet.header.fragment_count:
            self._pending.pop(key, None)
            raise ValueError("fragment index out of range")
        parts[packet.header.fragment_index] = packet.payload
        if any(part is None for part in parts):
            return None
        self._pending.pop(key, None)
        payload = b"".join(part for part in parts if part is not None)
        return MediaPacket(
            header=PacketHeader(
                stream_type=packet.header.stream_type,
                codec=packet.header.codec,
                session_id=packet.header.session_id,
                sequence_number=packet.header.sequence_number,
                timestamp_us=packet.header.timestamp_us,
                payload_size=len(payload),
            ),
            payload=payload,
        )
