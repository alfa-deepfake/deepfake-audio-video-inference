from __future__ import annotations

import time

from deepfake_media_transport import Codec, MediaPacket, PacketHeader, StreamType, packetize_payload

from backend.media_gateway.stream_signature import StreamSigner


def signed_packetize_payload(
    *,
    signer: StreamSigner,
    stream_type: StreamType,
    codec: Codec,
    session_id: bytes,
    sequence_number: int,
    payload: bytes,
    timestamp_us: int | None = None,
) -> list[MediaPacket]:
    timestamp_us = timestamp_us if timestamp_us is not None else time.time_ns() // 1000
    signed_packet = signer.sign_packet(
        MediaPacket(
            header=PacketHeader(
                stream_type=stream_type,
                codec=codec,
                session_id=session_id,
                sequence_number=sequence_number,
                timestamp_us=timestamp_us,
                payload_size=len(payload),
            ),
            payload=payload,
        )
    )
    return packetize_payload(
        stream_type=stream_type,
        codec=codec,
        session_id=session_id,
        sequence_number=sequence_number,
        timestamp_us=timestamp_us,
        payload=signed_packet.payload,
    )
