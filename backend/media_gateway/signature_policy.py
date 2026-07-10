from __future__ import annotations

import logging
from typing import Hashable

from deepfake_media_transport import MediaPacket

from backend.media_gateway.stream_signature import SignatureStatus, StreamSignatureVerifier, VerificationResult


COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_RED = "\033[91m"
COLOR_RESET = "\033[0m"
BAD_SIGNATURE_STATUSES = {
    SignatureStatus.UNTRUSTED_KEY,
    SignatureStatus.INVALID,
    SignatureStatus.TAMPERED,
    SignatureStatus.REPLAY,
    SignatureStatus.CHAIN_MISMATCH,
}


def verify_packet_signature(
    *,
    packet: MediaPacket,
    verifier: StreamSignatureVerifier,
    policy: str,
    logger: logging.Logger,
    peer,
    logged_statuses: set[Hashable],
    include_session_in_log_key: bool = False,
    color: bool = False,
) -> VerificationResult | None:
    if policy == "off":
        return VerificationResult(SignatureStatus.DISABLED, packet)

    try:
        verification = verifier.verify_and_strip(packet)
    except Exception as exc:
        logger.warning("invalid signature envelope from %s: %s", peer, exc)
        if policy == "block":
            return None
        return VerificationResult(SignatureStatus.INVALID, packet, str(exc))

    if verification.status in BAD_SIGNATURE_STATUSES:
        _log_bad_signature(logger, packet, verification, peer, color=color)
        return None if policy == "block" else verification

    if verification.status in {SignatureStatus.ABSENT, SignatureStatus.TRUSTED}:
        log_key = _signature_log_key(packet, verification, include_session=include_session_in_log_key)
        if log_key not in logged_statuses:
            logged_statuses.add(log_key)
            _log_signature_state(logger, packet, verification, peer, color=color)

    return verification


def _signature_log_key(
    packet: MediaPacket,
    verification: VerificationResult,
    *,
    include_session: bool,
) -> tuple[Hashable, int, SignatureStatus, str]:
    session_key: Hashable = packet.header.session_id if include_session else ""
    return session_key, int(packet.header.stream_type), verification.status, verification.key_id


def _log_bad_signature(
    logger: logging.Logger,
    packet: MediaPacket,
    verification: VerificationResult,
    peer,
    *,
    color: bool,
) -> None:
    prefix = f"{COLOR_RED}[SIGNATURE] BAD" if color else "stream signature"
    suffix = COLOR_RESET if color else ""
    logger.warning(
        "%s status=%s addr=%s session=%s stream=%s seq=%s key=%s reason=%s%s",
        prefix,
        verification.status.value,
        peer,
        packet.header.session_id.hex(),
        packet.header.stream_type.name,
        packet.header.sequence_number,
        verification.key_id,
        verification.reason,
        suffix,
    )


def _log_signature_state(
    logger: logging.Logger,
    packet: MediaPacket,
    verification: VerificationResult,
    peer,
    *,
    color: bool,
) -> None:
    if verification.status == SignatureStatus.TRUSTED:
        if color:
            logger.info(
                "%s[SIGNATURE] TRUSTED addr=%s session=%s stream=%s key=%s first_seq=%s%s",
                COLOR_GREEN,
                peer,
                packet.header.session_id.hex(),
                packet.header.stream_type.name,
                verification.key_id,
                packet.header.sequence_number,
                COLOR_RESET,
            )
        else:
            logger.info(
                "trusted signed stream from %s session=%s stream=%s key=%s first_seq=%s",
                peer,
                packet.header.session_id.hex(),
                packet.header.stream_type.name,
                verification.key_id,
                packet.header.sequence_number,
            )
        return

    if color:
        logger.info(
            "%s[SIGNATURE] UNSIGNED addr=%s session=%s stream=%s first_seq=%s%s",
            COLOR_YELLOW,
            peer,
            packet.header.session_id.hex(),
            packet.header.stream_type.name,
            packet.header.sequence_number,
            COLOR_RESET,
        )
    else:
        logger.info(
            "unsigned stream accepted from %s session=%s stream=%s first_seq=%s",
            peer,
            packet.header.session_id.hex(),
            packet.header.stream_type.name,
            packet.header.sequence_number,
        )
