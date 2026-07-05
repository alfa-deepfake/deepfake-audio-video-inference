from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional

from tools.udp_infer import RealtimeUdpProcessor, UdpInferSettings, load_runtime_dependencies


@dataclass(frozen=True)
class AudioEngineConfig:
    model_path: str
    index_path: str = ""
    sample_rate: int = 48000
    block_time: float = 0.25
    crossfade_time: float = 0.05
    extra_time: float = 2.5
    pitch: int = 0
    index_rate: float = 0.0
    rms_mix_rate: float = 0.0
    f0method: str = "fcpe"
    n_cpu: int = 1
    threshold: int = -60
    device: Optional[str] = None
    is_half: Optional[bool] = None
    use_jit: bool = False


class AudioInferenceEngine:
    """Thin adapter around the existing realtime RVC processor."""

    def __init__(self, cfg: AudioEngineConfig) -> None:
        argv = sys.argv[:]
        sys.argv = sys.argv[:1]
        try:
            load_runtime_dependencies()
            from configs.config import Config
        finally:
            sys.argv = argv
        settings = UdpInferSettings(
            model_path=cfg.model_path,
            index_path=cfg.index_path,
            bind_host="127.0.0.1",
            bind_port=0,
            sample_rate=cfg.sample_rate,
            block_time=cfg.block_time,
            crossfade_time=cfg.crossfade_time,
            extra_time=cfg.extra_time,
            pitch=cfg.pitch,
            index_rate=cfg.index_rate,
            rms_mix_rate=cfg.rms_mix_rate,
            f0method=cfg.f0method,
            n_cpu=cfg.n_cpu,
            threshold=cfg.threshold,
            device=cfg.device,
            is_half=cfg.is_half,
            use_jit=cfg.use_jit,
            recv_size=65535,
        )
        argv = sys.argv[:]
        sys.argv = sys.argv[:1]
        try:
            config = Config()
        finally:
            sys.argv = argv
        if cfg.device:
            config.device = cfg.device
        if cfg.is_half is not None:
            config.is_half = cfg.is_half
        config.use_jit = cfg.use_jit
        self.processor = RealtimeUdpProcessor(settings, config)

    def process_pcm16(self, payload: bytes) -> bytes:
        return self.processor.process_pcm16(payload)
