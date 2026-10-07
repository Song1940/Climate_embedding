"""
Step 3 모델: 시간축 압축 — 위치별 시간 TCN 오토인코더 (양자화 없음, float32 저장).

입력은 step2 잠재값입니다. 위치(큐브 면의 잠재 격자점 또는 ViT 토큰) 하나의 전체
기간을 한 줄의 시퀀스 [C, T]로 보고, 1D 합성곱으로 시간 관계를 학습합니다.
stride-2 합성곱으로 시간 길이를 절반으로 줄이고 채널 수 L로 압축률을 조절합니다.

    인코더  [C, T] → conv k3 → ResBlock×b → conv k4 s2 → ResBlock → 1×1 → [L, T/2]  (+ 선형 경로 conv k2 s2)
    디코더  [L, T/2] → 1×1 → ResBlock → convT k4 s2 → ResBlock×b → conv k3 → [C, T]  (+ 선형 경로 convT k2 s2)
    압축률(값 개수) = T·C / (T/2·L) = 2C / L

모든 층이 합성곱이라 입력 길이에 묶이지 않습니다. 30일이든 2년이든 같은 모델이
그대로 돌아가고, 창으로 자르지 않으므로 창 경계 문제도 없습니다. 정규화는 시점마다
채널 방향으로만 하므로(`ChannelNorm`) 학습 때 자른 길이와 추론 때 전체 길이가 달라도
결과가 같습니다. 하루를 복원할 때는 그 근처 며칠의 잠재값만 필요합니다.

위치마다 시간 평균 μ(하루치 크기)를 빼고, 채널별 표준편차 σ로 나눈 값을 다룹니다.
μ는 기간 전체에 한 번만 저장되며 저장량에 포함합니다.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# ---------------------------------------------------------------- latent layout

def to_cube(latent, kind: str):
    """step2 latent [T, ...] -> [T, 6, C, S, S] (ViT tokens are ordered face, row, col)."""
    if kind == "cnn":
        return latent
    t, tokens, d = latent.shape
    g = int(round(math.sqrt(tokens // 6)))
    return np.ascontiguousarray(latent.reshape(t, 6, g, g, d).transpose(0, 1, 4, 2, 3))


def from_cube(cube, kind: str):
    if kind == "cnn":
        return cube
    t, _, d, g, _ = cube.shape
    return np.ascontiguousarray(cube.transpose(0, 1, 3, 4, 2).reshape(t, 6 * g * g, d))


def cube_to_sequences(cube: np.ndarray) -> np.ndarray:
    """[T, 6, C, S, S] -> [P, C, T] with P = 6·S·S positions."""
    t, f, c, s, _ = cube.shape
    return np.ascontiguousarray(cube.transpose(1, 3, 4, 2, 0).reshape(f * s * s, c, t))


def sequences_to_cube(seq: np.ndarray, side: int) -> np.ndarray:
    p, c, t = seq.shape
    return np.ascontiguousarray(seq.reshape(6, side, side, c, t).transpose(4, 0, 3, 1, 2))


# ---------------------------------------------------------------- layers

class ChannelNorm(nn.Module):
    """LayerNorm over channels at each time step of [N, C, T] (length-independent)."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class ResBlock1d(nn.Module):
    def __init__(self, channels: int, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.n1, self.c1 = ChannelNorm(channels), nn.Conv1d(channels, channels, kernel, padding=pad,
                                                            padding_mode="replicate")
        self.n2, self.c2 = ChannelNorm(channels), nn.Conv1d(channels, channels, kernel, padding=pad,
                                                            padding_mode="replicate")

    def forward(self, x):
        return x + self.c2(F.gelu(self.n2(self.c1(F.gelu(self.n1(x))))))


# ---------------------------------------------------------------- codec

class TemporalTCNCodec(nn.Module):
    """Per-position temporal autoencoder on [P, C, T] sequences; latent [P, L, T/2]."""

    def __init__(self, channels: int, mean: np.ndarray, scale: np.ndarray, hidden: int = 256,
                 latent_channels: int = 256, blocks: int = 2, linear_path: bool = True):
        super().__init__()
        self.config = {"channels": int(channels), "hidden": int(hidden),
                       "latent_channels": int(latent_channels), "blocks": int(blocks),
                       "linear_path": bool(linear_path)}
        c, h, l = channels, hidden, latent_channels
        # μ [P, C] (per-position time mean) and σ [C] are stored with the archive
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        self.enc = nn.Sequential(
            nn.Conv1d(c, h, 3, padding=1, padding_mode="replicate"),
            *[ResBlock1d(h) for _ in range(blocks)],
            nn.Conv1d(h, h, 4, stride=2, padding=1),
            ResBlock1d(h), ChannelNorm(h), nn.GELU(), nn.Conv1d(h, l, 1))
        self.dec_in = nn.Sequential(nn.Conv1d(l, h, 1), ResBlock1d(h))
        self.dec_up = nn.ConvTranspose1d(h, h, 4, stride=2, padding=1)
        self.dec_out = nn.Sequential(*[ResBlock1d(h) for _ in range(blocks)], ChannelNorm(h), nn.GELU(),
                                     nn.Conv1d(h, c, 3, padding=1, padding_mode="replicate"))
        # learned linear path (2 days -> 1 latent step and back) added to the TCN branches.
        # Without it the TCN stalled at ~10.5% error for every L, while a linear
        # 2-day projection reaches 0.1-7%: the deep branch cannot learn the near-identity part.
        self.linear_path = bool(linear_path)
        if self.linear_path:
            self.enc_linear = nn.Conv1d(c, l, 2, stride=2)
            self.dec_linear = nn.ConvTranspose1d(l, c, 2, stride=2)

    def normalize(self, z, positions=slice(None)):
        """z [N, C, T] for the given positions -> anomaly / σ."""
        return (z - self.mean[positions, :, None]) / self.scale[None, :, None]

    def denormalize(self, a, positions=slice(None)):
        return self.mean[positions, :, None] + a * self.scale[None, :, None]

    def encode(self, a):
        """a [N, C, T] (T even) -> latent [N, L, T/2]."""
        latent = self.enc(a)
        return latent + self.enc_linear(a) if self.linear_path else latent

    def decode(self, latent):
        out = self.dec_out(self.dec_up(self.dec_in(latent)))
        return out + self.dec_linear(latent) if self.linear_path else out

    def forward(self, a):
        latent = self.encode(a)
        return self.decode(latent), latent

    @classmethod
    def from_checkpoint(cls, checkpoint: dict):
        state = checkpoint["state_dict"]
        config = {"linear_path": False, **checkpoint["config"]}      # runs before the linear path
        model = cls(mean=state["mean"].numpy(), scale=state["scale"].numpy(), **config)
        model.load_state_dict(state)
        return model.eval()

    def decoder_parameter_count(self) -> int:
        """Decoder weights only; μ and σ are counted as stored data."""
        return sum(p.numel() for name, p in self.named_parameters() if name.startswith("dec"))


def pad_even(x: torch.Tensor) -> torch.Tensor:
    """[N, C, T] -> T rounded up to even by repeating the last day."""
    return x if x.shape[-1] % 2 == 0 else torch.cat([x, x[..., -1:]], dim=-1)
