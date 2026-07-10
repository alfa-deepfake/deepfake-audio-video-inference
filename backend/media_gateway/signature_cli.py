from __future__ import annotations

import argparse

from backend.media_gateway.stream_signature import (
    DEFAULT_ISSUER,
    DEFAULT_KEY_ID,
    SignatureConfig,
    parse_key_value_pairs,
)


def add_signature_sender_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--signature-key", default="", help="Enable test C2PA-like stream signatures with this shared secret")
    parser.add_argument("--signature-key-id", default=DEFAULT_KEY_ID)
    parser.add_argument("--signature-issuer", default=DEFAULT_ISSUER)


def signature_config_from_args(args: argparse.Namespace) -> SignatureConfig:
    return SignatureConfig(
        enabled=bool(args.signature_key),
        key_id=args.signature_key_id,
        secret=args.signature_key.encode("utf-8"),
        issuer=args.signature_issuer,
    )


def add_signature_verifier_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--signature-policy", choices=("off", "log", "block"), default="off")
    parser.add_argument(
        "--signature-trusted-key",
        action="append",
        default=[],
        help="Trusted test stream signature key in key_id=secret format. Can be passed multiple times.",
    )


def trusted_signature_keys_from_args(args: argparse.Namespace) -> dict[str, bytes]:
    return parse_key_value_pairs(args.signature_trusted_key)
