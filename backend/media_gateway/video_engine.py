from __future__ import annotations

import os
import struct
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


LENGTH_STRUCT = struct.Struct("!I")


@dataclass(frozen=True)
class VideoEngineConfig:
    deep_live_cam_root: str
    source_face_path: str
    execution_provider: str = "cuda"
    camera_fps: float = 25.0
    python_path: Optional[str] = None
    cuda_lib_root: Optional[str] = None


class VideoInferenceEngine:
    """Out-of-process Deep-Live-Cam adapter."""

    def __init__(self, cfg: VideoEngineConfig) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        self._stderr_file = tempfile.TemporaryFile(mode="w+b")
        self._proc = self._start_worker()

    def _start_worker(self) -> subprocess.Popen:
        repo_root = Path(__file__).resolve().parents[2]
        python_path = self.cfg.python_path or "python3"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
        cmd = [
            python_path,
            "-m",
            "backend.media_gateway.video_worker",
            "--dlc-root",
            self.cfg.deep_live_cam_root,
            "--source-face",
            self.cfg.source_face_path,
            "--execution-provider",
            self.cfg.execution_provider,
            "--camera-fps",
            str(self.cfg.camera_fps),
        ]
        if self.cfg.cuda_lib_root:
            cmd.extend(["--cuda-lib-root", self.cfg.cuda_lib_root])
        return subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_file,
            bufsize=0,
            env=env,
        )

    def process_mjpeg(self, payload: bytes) -> bytes:
        with self._lock:
            if self._proc.poll() is not None:
                self._stderr_file.seek(0)
                stderr = self._stderr_file.read() or b""
                raise RuntimeError(
                    "Deep-Live-Cam worker exited unexpectedly: "
                    + stderr.decode("utf-8", errors="replace")
                )
            assert self._proc.stdin is not None
            assert self._proc.stdout is not None
            self._proc.stdin.write(LENGTH_STRUCT.pack(len(payload)))
            self._proc.stdin.write(payload)
            self._proc.stdin.flush()
            length_data = self._proc.stdout.read(LENGTH_STRUCT.size)
            if len(length_data) != LENGTH_STRUCT.size:
                raise RuntimeError("failed to read response length from Deep-Live-Cam worker")
            response_length = LENGTH_STRUCT.unpack(length_data)[0]
            response = self._proc.stdout.read(response_length)
            if len(response) != response_length:
                raise RuntimeError("incomplete response from Deep-Live-Cam worker")
            return response
