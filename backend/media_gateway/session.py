from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StreamState:
    last_sequence_number: int = -1
    last_timestamp_us: int = 0
    packets_received: int = 0
    packets_dropped: int = 0


@dataclass
class SessionState:
    session_id: bytes
    audio: StreamState = field(default_factory=StreamState)
    video: StreamState = field(default_factory=StreamState)
    last_client_addr: Optional[tuple[str, int]] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def touch(self, client_addr: tuple[str, int]) -> None:
        self.last_client_addr = client_addr
        self.updated_at = time.time()

