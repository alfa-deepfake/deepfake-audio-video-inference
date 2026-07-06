from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import torch


VERSION_CONFIGS = (
    "v1/32k.json",
    "v1/40k.json",
    "v1/48k.json",
    "v2/48k.json",
    "v2/32k.json",
)


class Config:
    """Minimal runtime config for headless realtime inference."""

    def __init__(self) -> None:
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.is_half = torch.cuda.is_available()
        self.use_jit = False
        self.dml = False
        self.n_cpu = max(1, os.cpu_count() or 1)
        self.gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        self.gpu_mem = None
        self.json_config = self.load_config_json()
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    @staticmethod
    def load_config_json() -> dict:
        configs: dict[str, dict] = {}
        for config_file in VERSION_CONFIGS:
            source = Path("configs") / config_file
            target = Path("configs/inuse") / config_file
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(source, target)
            with target.open("r", encoding="utf-8") as handle:
                configs[config_file] = json.load(handle)
        return configs

    def device_config(self) -> tuple[int, int, int, int]:
        if self.device.startswith("cuda"):
            return 3, 10, 60, 65
        self.is_half = False
        return 1, 6, 38, 41
