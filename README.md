# deepfake-voice-inference

Headless realtime media inference service for the current deepfake pipeline.

The laptop captures microphone and webcam input, sends both streams through a
single SSH-tunneled TCP connection to the cluster, and receives processed audio
and video back for local preview.

The repository is intentionally trimmed down to runtime-critical pieces:

- `backend/media_gateway`: audio/video stream server, client, protocol, and
  Deep-Live-Cam adapter.
- `tools/udp_infer.py` and `tools/rvc_for_realtime.py`: realtime RVC audio
  inference path.
- `infer/lib/infer_pack`, `infer/lib/jit`, `infer/lib/rmvpe.py`: minimal RVC
  inference internals needed by the realtime processor.
- `configs`: model/runtime configuration used by RVC.
- `assets`: model locations and placeholder files. Large model weights are not
  committed.

## Server Directory

The active cluster clone is:

```bash
/home/master/work/deepfake-test/voice-module/deepfake-voice-inference
```

The local laptop clone is usually:

```bash
~/work/deepfake-voice-inference
```

## Required Runtime Assets

The stream server expects these files on the cluster:

```text
assets/weights/voice_model.pth
assets/indices/voice_model.index
assets/hubert/hubert_base.pt
```

Deep-Live-Cam is expected at:

```text
~/workspace_w9line/deep_face/extracted/Deep-Live-Cam
```

The current source face used by the documented command is:

```text
~/workspace_w9line/deep_face/extracted/Deep-Live-Cam/классный_чел_пнг.jpg
```

## Start Realtime Stream

### 1. Start The Server On The Cluster

```bash
ssh -i /tmp/deepfake_voice_cluster_key -p 22010 master@62.183.4.208
cd ~/work/deepfake-test/voice-module/deepfake-voice-inference
source .venv/bin/activate

PYTHONPATH=$PWD python -m backend.media_gateway.stream_server \
  --host 127.0.0.1 \
  --port 13000 \
  --audio-model-path assets/weights/voice_model.pth \
  --audio-index-path assets/indices/voice_model.index \
  --audio-index-rate 0.3 \
  --video-dlc-root ~/workspace_w9line/deep_face/extracted/Deep-Live-Cam \
  --video-source-face ~/workspace_w9line/deep_face/extracted/Deep-Live-Cam/классный_чел_пнг.jpg \
  --video-python-path ~/workspace_w9line/deep_face/extracted/Deep-Live-Cam/.venv_dlc/bin/python \
  --video-cuda-lib-root "$PWD/.venv/lib/python3.10/site-packages" \
  --video-execution-provider cuda \
  --video-camera-fps 15.0
```

Expected log:

```text
media_gateway.stream_server: stream server listening on tcp://127.0.0.1:13000
```

For a persistent server session, run the command inside `tmux`/`screen`, or
redirect it to `/tmp/media_gateway_stream.log`.

### 2. Open The SSH Tunnel On The Laptop

In a separate laptop terminal:

```bash
ssh -i /tmp/deepfake_voice_cluster_key -p 22010 \
  -N -L 13000:127.0.0.1:13000 master@62.183.4.208
```

Keep this terminal open while streaming.

### 3. Run The Laptop Client

In another laptop terminal:

```bash
cd ~/work/deepfake-voice-inference
source .venv/bin/activate

PYTHONPATH=$PWD python -m backend.media_gateway.stream_client \
  --gateway-host 127.0.0.1 \
  --gateway-port 13000 \
  --video-width 512 \
  --video-height 288 \
  --video-fps 15 \
  --jpeg-quality 65
```

Expected laptop log:

```text
media_gateway.stream_client: connected to tcp://127.0.0.1:13000
media_gateway.stream_client: microphone capture started
media_gateway.stream_client: webcam capture started
```

Expected server log:

```text
media_gateway.stream_server: stream client connected: ('127.0.0.1', ...)
media_gateway.stream_server: control from ... {"kind": "stream_client_started", ...}
```

The preview window is named `media-gateway-stream-preview`.

## Checks

Check whether the cluster server is listening:

```bash
ssh -i /tmp/deepfake_voice_cluster_key -p 22010 master@62.183.4.208 \
  "ss -ltnp | grep 13000 || true"
```

Watch the server log:

```bash
ssh -i /tmp/deepfake_voice_cluster_key -p 22010 master@62.183.4.208 \
  "tail -f /tmp/media_gateway_stream.log"
```

Check laptop preview dependencies:

```bash
python - <<'PY'
import tkinter
from PIL import Image, ImageTk
print("tkinter and Pillow preview dependencies are available")
PY
```

## Tuning

Lower bandwidth and latency:

```bash
PYTHONPATH=$PWD python -m backend.media_gateway.stream_client \
  --gateway-host 127.0.0.1 \
  --gateway-port 13000 \
  --video-width 512 \
  --video-height 288 \
  --video-fps 12 \
  --jpeg-quality 55
```

Better visual quality:

```bash
PYTHONPATH=$PWD python -m backend.media_gateway.stream_client \
  --gateway-host 127.0.0.1 \
  --gateway-port 13000 \
  --video-width 640 \
  --video-height 360 \
  --video-fps 15 \
  --jpeg-quality 75
```

## Legacy UDP Tools

The preferred runtime is `stream_server` plus `stream_client` through SSH.
The older UDP tools are still present for local experiments:

```bash
PYTHONPATH=$PWD python -m tools.udp_infer --help
PYTHONPATH=$PWD python -m backend.media_gateway.udp_tcp_tunnel --help
```

## License

This project keeps the upstream RVC MIT license. See `LICENSE`.
