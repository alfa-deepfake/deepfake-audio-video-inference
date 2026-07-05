from __future__ import annotations

import argparse
import ctypes
import glob
import importlib
import os
import struct
import sys
import types

import numpy as np


LENGTH_STRUCT = struct.Struct("!I")


def parse_args():
    parser = argparse.ArgumentParser(description="Deep-Live-Cam MJPEG worker")
    parser.add_argument("--dlc-root", required=True)
    parser.add_argument("--source-face", required=True)
    parser.add_argument("--execution-provider", default="cuda")
    parser.add_argument("--camera-fps", type=float, default=20.0)
    parser.add_argument("--cuda-lib-root", default="")
    return parser.parse_args()


def provider_name(provider: str) -> str:
    value = provider.lower()
    mapping = {
        "cuda": "CUDAExecutionProvider",
        "cpu": "CPUExecutionProvider",
        "coreml": "CoreMLExecutionProvider",
        "dml": "DmlExecutionProvider",
        "rocm": "ROCMExecutionProvider",
    }
    return mapping.get(value, provider)


def preload_nvidia_libraries(cuda_lib_root: str) -> None:
    if not cuda_lib_root:
        return
    nvidia_dir = os.path.join(cuda_lib_root, "nvidia")
    torch_lib_dir = os.path.join(cuda_lib_root, "torch", "lib")
    for lib_dir in (torch_lib_dir,):
        if os.path.isdir(lib_dir):
            current = os.environ.get("LD_LIBRARY_PATH", "")
            if lib_dir not in current.split(os.pathsep):
                os.environ["LD_LIBRARY_PATH"] = lib_dir + (os.pathsep + current if current else "")
            for so_path in sorted(glob.glob(os.path.join(lib_dir, "lib*.so*"))):
                try:
                    ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
    if not os.path.isdir(nvidia_dir):
        return
    for package_name in os.listdir(nvidia_dir):
        lib_dir = os.path.join(nvidia_dir, package_name, "lib")
        if not os.path.isdir(lib_dir):
            continue
        current = os.environ.get("LD_LIBRARY_PATH", "")
        if lib_dir not in current.split(os.pathsep):
            os.environ["LD_LIBRARY_PATH"] = lib_dir + (os.pathsep + current if current else "")
        for so_path in sorted(glob.glob(os.path.join(lib_dir, "lib*.so*"))):
            try:
                ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


def build_processor(args):
    root = os.path.abspath(args.dlc_root)
    if root not in sys.path:
        sys.path.insert(0, root)

    core_stub = types.ModuleType("modules.core")

    def update_status(message: str, scope: str = "DLC.WORKER") -> None:
        print(f"[{scope}] {message}", file=sys.stderr, flush=True)

    core_stub.update_status = update_status
    sys.modules["modules.core"] = core_stub

    dlc_globals = importlib.import_module("modules.globals")
    dlc_globals.source_path = args.source_face
    dlc_globals.target_path = None
    dlc_globals.output_path = None
    dlc_globals.frame_processors = ["face_swapper"]
    dlc_globals.many_faces = False
    dlc_globals.map_faces = False
    dlc_globals.mouth_mask = False
    dlc_globals.keep_fps = True
    dlc_globals.keep_audio = False
    dlc_globals.keep_frames = False
    dlc_globals.headless = True
    dlc_globals.live_mirror = False
    dlc_globals.live_resizable = False
    dlc_globals.show_fps = False
    dlc_globals.execution_threads = 2
    dlc_globals.execution_providers = [provider_name(args.execution_provider)]
    dlc_globals.fp_ui["face_enhancer"] = False
    dlc_globals.fp_ui["face_enhancer_gpen256"] = False
    dlc_globals.fp_ui["face_enhancer_gpen512"] = False

    face_swapper = importlib.import_module("modules.processors.frame.face_swapper")
    if not face_swapper.pre_check():
        raise RuntimeError("Deep-Live-Cam face_swapper pre_check failed")
    if not face_swapper.pre_start():
        raise RuntimeError("Deep-Live-Cam face_swapper pre_start failed")

    live_processor_module = importlib.import_module("modules.live_processor")
    cv2 = importlib.import_module("cv2")
    processor = live_processor_module.LiveFrameProcessor(args.camera_fps)
    return processor, cv2


def read_exact(stream, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def main() -> None:
    args = parse_args()
    preload_nvidia_libraries(args.cuda_lib_root)
    processor, cv2 = build_processor(args)

    while True:
        header = read_exact(sys.stdin.buffer, LENGTH_STRUCT.size)
        if len(header) != LENGTH_STRUCT.size:
            return
        payload_size = LENGTH_STRUCT.unpack(header)[0]
        payload = read_exact(sys.stdin.buffer, payload_size)
        if len(payload) != payload_size:
            return
        np_buffer = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
        if frame is None:
            response = payload
        else:
            _, processed = processor.process(frame)
            ok, encoded = cv2.imencode(".jpg", processed, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            response = encoded.tobytes() if ok else payload
        sys.stdout.buffer.write(LENGTH_STRUCT.pack(len(response)))
        sys.stdout.buffer.write(response)
        sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
