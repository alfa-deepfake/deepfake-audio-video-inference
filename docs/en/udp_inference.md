# UDP Inference Prototype

This repository is being adapted from RVC WebUI into a lightweight, headless
voice conversion service for local realtime inference.

The first service entry point is:

```bash
python tools/udp_infer.py \
  --model-path assets/weights/model.pth \
  --index-path assets/indices/model.index \
  --bind-host 0.0.0.0 \
  --bind-port 9999 \
  --sample-rate 48000 \
  --block-time 0.25 \
  --index-rate 0.3 \
  --f0method fcpe
```

## Scope

The prototype intentionally avoids:

- Gradio
- FastAPI
- WebSocket
- browser audio devices
- `sounddevice`

It reuses the existing realtime RVC model code in `tools/rvc_for_realtime.py`
and replaces the UI/audio-device layer with a UDP socket loop.

## Wire Format

The current packet format is deliberately simple:

- transport: UDP
- input payload: mono PCM16 little-endian
- output payload: mono PCM16 little-endian
- sample rate: configured with `--sample-rate`
- frame size: configured indirectly with `--block-time`

If a packet is shorter than the configured block size, it is left-padded with
silence. If it is longer, only the newest block is processed.

For production benchmarking, the next protocol revision should add a compact
binary header with at least:

- stream id
- sequence number
- timestamp
- sample rate
- payload codec

## Runtime Path

```text
UDP PCM16 packet
-> float32 block
-> rolling input buffer
-> resample to 16 kHz for HuBERT
-> RVC realtime inference
-> optional model-to-transport resample
-> SOLA crossfade
-> UDP PCM16 response
```

## Next Steps

1. Add a small UDP client/benchmark tool that sends WAV files as fixed-size
   packets and records response latency.
2. Move reusable buffering/SOLA code out of `tools/udp_infer.py` once the
   prototype stabilizes.
3. Add a Docker image for Ubuntu 24.04 + CUDA runtime on A100.
4. Replace raw PCM with RTP or an explicit project packet header.
5. Profile `fcpe`, `rmvpe`, and `harvest` on the target GPU.
