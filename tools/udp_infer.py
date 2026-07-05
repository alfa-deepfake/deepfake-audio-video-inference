from __future__ import annotations

import argparse
import logging
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


logger = logging.getLogger("udp_infer")


@dataclass(frozen=True)
class UdpInferSettings:
    model_path: str
    index_path: str
    bind_host: str
    bind_port: int
    sample_rate: int
    block_time: float
    crossfade_time: float
    extra_time: float
    pitch: int
    index_rate: float
    rms_mix_rate: float
    f0method: str
    n_cpu: int
    threshold: int
    device: Optional[str]
    is_half: Optional[bool]
    use_jit: bool
    recv_size: int


class RealtimeUdpProcessor:
    def __init__(self, settings: UdpInferSettings, config) -> None:
        self.settings = settings
        self.config = config
        self.device = torch.device(config.device)

        self.rvc = RVC(
            settings.pitch,
            settings.model_path,
            settings.index_path,
            settings.index_rate,
            settings.n_cpu,
            None,
            None,
            config,
        )

        self.samplerate = settings.sample_rate or self.rvc.tgt_sr
        self.zc = self.samplerate // 100
        self.block_frame = (
            int(round(settings.block_time * self.samplerate / self.zc)) * self.zc
        )
        self.block_frame_16k = 160 * self.block_frame // self.zc
        self.crossfade_frame = (
            int(round(settings.crossfade_time * self.samplerate / self.zc)) * self.zc
        )
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self.zc
        self.extra_frame = (
            int(round(settings.extra_time * self.samplerate / self.zc)) * self.zc
        )
        self.skip_head = self.extra_frame // self.zc
        self.return_length = (
            self.block_frame + self.sola_buffer_frame + self.sola_search_frame
        ) // self.zc

        self.input_wav = torch.zeros(
            self.extra_frame
            + self.crossfade_frame
            + self.sola_search_frame
            + self.block_frame,
            device=self.device,
            dtype=torch.float32,
        )
        self.input_wav_res = torch.zeros(
            160 * self.input_wav.shape[0] // self.zc,
            device=self.device,
            dtype=torch.float32,
        )
        self.output_buffer = self.input_wav.clone()
        self.sola_buffer = torch.zeros(
            self.sola_buffer_frame,
            device=self.device,
            dtype=torch.float32,
        )
        self.rms_buffer = np.zeros(4 * self.zc, dtype=np.float32)

        self.fade_in_window = (
            torch.sin(
                0.5
                * np.pi
                * torch.linspace(
                    0.0,
                    1.0,
                    steps=self.sola_buffer_frame,
                    device=self.device,
                    dtype=torch.float32,
                )
            )
            ** 2
        )
        self.fade_out_window = 1 - self.fade_in_window

        self.resampler_to_16k = tat.Resample(
            orig_freq=self.samplerate,
            new_freq=16000,
            dtype=torch.float32,
        ).to(self.device)
        self.resampler_from_model = (
            tat.Resample(
                orig_freq=self.rvc.tgt_sr,
                new_freq=self.samplerate,
                dtype=torch.float32,
            ).to(self.device)
            if self.rvc.tgt_sr != self.samplerate
            else None
        )

    def process_pcm16(self, packet: bytes) -> bytes:
        pcm = np.frombuffer(packet, dtype="<i2")
        if pcm.size == 0:
            return b""
        audio = pcm.astype(np.float32) / 32768.0
        audio = self._fit_block(audio)
        out = self.process_float32(audio)
        out = np.clip(out, -1.0, 1.0)
        return (out * 32767.0).astype("<i2", copy=False).tobytes()

    def process_float32(self, indata: np.ndarray) -> np.ndarray:
        start_time = time.perf_counter()
        indata = np.asarray(indata, dtype=np.float32)
        indata = self._gate_by_threshold(indata)

        self.input_wav[:-self.block_frame] = self.input_wav[
            self.block_frame :
        ].clone()
        self.input_wav[-indata.shape[0] :] = torch.from_numpy(indata).to(self.device)

        self.input_wav_res[:-self.block_frame_16k] = self.input_wav_res[
            self.block_frame_16k :
        ].clone()
        self.input_wav_res[
            -160 * (indata.shape[0] // self.zc + 1) :
        ] = self.resampler_to_16k(self.input_wav[-indata.shape[0] - 2 * self.zc :])[
            160:
        ]

        infer_wav = self.rvc.infer(
            self.input_wav_res,
            self.block_frame_16k,
            self.skip_head,
            self.return_length,
            self.settings.f0method,
        )
        if self.resampler_from_model is not None:
            infer_wav = self.resampler_from_model(infer_wav)

        infer_wav = self._mix_rms(infer_wav)
        infer_wav = self._apply_sola(infer_wav)
        logger.debug("processed block in %.4fs", time.perf_counter() - start_time)
        return infer_wav[: self.block_frame].detach().cpu().numpy()

    def _fit_block(self, audio: np.ndarray) -> np.ndarray:
        if audio.shape[0] == self.block_frame:
            return audio
        if audio.shape[0] > self.block_frame:
            return audio[-self.block_frame :]
        return np.pad(audio, (self.block_frame - audio.shape[0], 0))

    def _gate_by_threshold(self, indata: np.ndarray) -> np.ndarray:
        if self.settings.threshold <= -60:
            return indata
        gated = np.append(self.rms_buffer, indata)
        rms = librosa.feature.rms(
            y=gated,
            frame_length=4 * self.zc,
            hop_length=self.zc,
        )[:, 2:]
        self.rms_buffer[:] = gated[-4 * self.zc :]
        gated = gated[2 * self.zc - self.zc // 2 :]
        db_threshold = librosa.amplitude_to_db(rms, ref=1.0)[0] < self.settings.threshold
        for idx, muted in enumerate(db_threshold):
            if muted:
                gated[idx * self.zc : (idx + 1) * self.zc] = 0
        return gated[self.zc // 2 :].astype(np.float32, copy=False)

    def _mix_rms(self, infer_wav: torch.Tensor) -> torch.Tensor:
        if self.settings.rms_mix_rate >= 1:
            return infer_wav
        input_wav = self.input_wav[self.extra_frame :]
        rms1 = librosa.feature.rms(
            y=input_wav[: infer_wav.shape[0]].detach().cpu().numpy(),
            frame_length=4 * self.zc,
            hop_length=self.zc,
        )
        rms1 = torch.from_numpy(rms1).to(self.device)
        rms1 = F.interpolate(
            rms1.unsqueeze(0),
            size=infer_wav.shape[0] + 1,
            mode="linear",
            align_corners=True,
        )[0, 0, :-1]
        rms2 = librosa.feature.rms(
            y=infer_wav.detach().cpu().numpy(),
            frame_length=4 * self.zc,
            hop_length=self.zc,
        )
        rms2 = torch.from_numpy(rms2).to(self.device)
        rms2 = F.interpolate(
            rms2.unsqueeze(0),
            size=infer_wav.shape[0] + 1,
            mode="linear",
            align_corners=True,
        )[0, 0, :-1]
        rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-3)
        return infer_wav * torch.pow(
            rms1 / rms2,
            torch.tensor(1 - self.settings.rms_mix_rate, device=self.device),
        )

    def _apply_sola(self, infer_wav: torch.Tensor) -> torch.Tensor:
        conv_input = infer_wav[
            None,
            None,
            : self.sola_buffer_frame + self.sola_search_frame,
        ]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(
            F.conv1d(
                conv_input**2,
                torch.ones(1, 1, self.sola_buffer_frame, device=self.device),
            )
            + 1e-8
        )
        sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
        infer_wav = infer_wav[sola_offset:]
        infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
        infer_wav[: self.sola_buffer_frame] += self.sola_buffer * self.fade_out_window
        self.sola_buffer[:] = infer_wav[
            self.block_frame : self.block_frame + self.sola_buffer_frame
        ]
        return infer_wav


def parse_args() -> UdpInferSettings:
    parser = argparse.ArgumentParser(
        description="Headless realtime RVC inference over raw UDP PCM16 mono packets."
    )
    parser.add_argument("--model-path", required=True, help="Path to the RVC .pth file")
    parser.add_argument("--index-path", default="", help="Path to the RVC .index file")
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--bind-port", type=int, default=9999)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--block-time", type=float, default=0.25)
    parser.add_argument("--crossfade-time", type=float, default=0.05)
    parser.add_argument("--extra-time", type=float, default=2.5)
    parser.add_argument("--pitch", type=int, default=0)
    parser.add_argument("--index-rate", type=float, default=0.0)
    parser.add_argument("--rms-mix-rate", type=float, default=0.0)
    parser.add_argument("--f0method", default="fcpe", choices=("pm", "harvest", "crepe", "rmvpe", "fcpe"))
    parser.add_argument("--n-cpu", type=int, default=1)
    parser.add_argument("--threshold", type=int, default=-60)
    parser.add_argument("--device", default=None)
    parser.add_argument("--is-half", choices=("true", "false"), default=None)
    parser.add_argument("--use-jit", action="store_true")
    parser.add_argument("--recv-size", type=int, default=65535)
    args = parser.parse_args()
    if args.index_rate > 0 and not args.index_path:
        parser.error("--index-path is required when --index-rate is greater than 0")
    if args.f0method == "harvest" and args.n_cpu != 1:
        parser.error("UDP harvest mode currently supports --n-cpu 1 only")
    return UdpInferSettings(
        model_path=args.model_path,
        index_path=args.index_path,
        bind_host=args.bind_host,
        bind_port=args.bind_port,
        sample_rate=args.sample_rate,
        block_time=args.block_time,
        crossfade_time=args.crossfade_time,
        extra_time=args.extra_time,
        pitch=args.pitch,
        index_rate=args.index_rate,
        rms_mix_rate=args.rms_mix_rate,
        f0method=args.f0method,
        n_cpu=args.n_cpu,
        threshold=args.threshold,
        device=args.device,
        is_half=None if args.is_half is None else args.is_half == "true",
        use_jit=args.use_jit,
        recv_size=args.recv_size,
    )


def load_runtime_dependencies() -> None:
    global F, Config, RVC, librosa, load_dotenv, np, tat, torch

    import librosa
    import numpy as np
    import torch
    import torch.nn.functional as F
    import torchaudio.transforms as tat
    from dotenv import load_dotenv

    from configs.config import Config
    from tools.rvc_for_realtime import RVC


def build_config(settings: UdpInferSettings):
    sys.argv = sys.argv[:1]
    config = Config()
    if settings.device:
        config.device = settings.device
    if settings.is_half is not None:
        config.is_half = settings.is_half
    config.use_jit = settings.use_jit
    return config


def serve(settings: UdpInferSettings, processor: RealtimeUdpProcessor) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((settings.bind_host, settings.bind_port))
    logger.info(
        "listening on udp://%s:%s, block_frame=%s samples",
        settings.bind_host,
        settings.bind_port,
        processor.block_frame,
    )
    while True:
        packet, addr = sock.recvfrom(settings.recv_size)
        response = processor.process_pcm16(packet)
        if response:
            sock.sendto(response, addr)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = parse_args()
    load_runtime_dependencies()
    load_dotenv()
    config = build_config(settings)
    processor = RealtimeUdpProcessor(settings, config)
    serve(settings, processor)


if __name__ == "__main__":
    main()
