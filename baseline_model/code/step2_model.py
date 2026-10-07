"""
Step 2 모델: 공간 지역성을 이용한 압축 (큐브스피어 CNN / ViT).

KIM 격자는 큐브스피어(ne=25, np=3)라서 15,002개 셀이 6개 면 각각의 51×51
격자점으로 정확히 옮겨집니다(`data_utils.build_cube_layout`). 면 모서리의 점은 두
면에 중복으로 들어가고, 복원할 때는 중복값을 평균합니다. 보간이 없는 무손실
재배열입니다. 면 경계 바깥은 이웃 면의 가장 가까운 셀 값으로 채운 halo를 붙여,
면 경계에서 문맥이 끊기지 않게 합니다.

입력은 step1 임베딩 [day, grid, embed_dim]이고 날짜마다 독립적으로 압축합니다.

- `CubeCNNCodec` (kind="cnn"): 51×51 면 이미지를 stride-2 합성곱 두 번으로 13×13까지
  줄인 뒤 `latent_channels`개 채널로 저장합니다. 디코더는 잠재 격자에도 이웃 면
  halo를 붙인 뒤 업샘플합니다. 잠재값 [day, 6, latent_channels, 13, 13].
- `CubeViTCodec` (kind="vit"): 면마다 5×5 셀을 토큰 하나로 묶어 전 지구 726개
  토큰을 만듭니다. 위치 정보로 토큰 중심의 3차원 구면 좌표를 쓰므로 면 경계와
  무관하게 전역 attention이 가능합니다. 토큰마다 `latent_dim`개 숫자로 줄입니다.
  잠재값 [day, 726, latent_dim].
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from data_utils import cube_center_halo_index, cube_points


class CubeCodecBase(nn.Module):
    kind = "base"

    def __init__(self, layout: dict, embed_dim: int, input_halo: int):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.n = int(layout["n"])                       # 50 -> 51 nodes per face side
        self.nodes = self.n + 1
        full_halo = int(layout["halo"])
        if input_halo > full_halo:
            raise ValueError(f"layout halo {full_halo} < required {input_halo}")
        crop = full_halo - input_halo
        index = np.asarray(layout["node_index"])
        index = index[:, crop:index.shape[1] - crop, crop:index.shape[2] - crop]
        self.input_halo = int(input_halo)
        self.register_buffer("input_index", torch.as_tensor(index, dtype=torch.long), persistent=False)
        inner = index[:, input_halo:input_halo + self.nodes, input_halo:input_halo + self.nodes]
        self.register_buffer("inner_index", torch.as_tensor(inner.reshape(-1), dtype=torch.long),
                             persistent=False)
        self.register_buffer("counts", torch.as_tensor(np.asarray(layout["counts"]), dtype=torch.float32),
                             persistent=False)
        self.axes = np.asarray(layout["axes"], dtype=np.float64)
        self.grid_count = int(len(layout["counts"]))

    def to_faces(self, embeddings):
        """[T, grid, D] -> [T, 6, D, S, S] with the input halo."""
        faces = embeddings[:, self.input_index]                  # [T, 6, S, S, D]
        return faces.permute(0, 1, 4, 2, 3)

    def from_faces(self, faces):
        """[T, 6, D, 51, 51] (no halo) -> [T, grid, D], averaging duplicated edge nodes."""
        t, _, d = faces.shape[:3]
        flat = faces.permute(0, 1, 3, 4, 2).reshape(t, -1, d).float()
        out = flat.new_zeros((t, self.grid_count, d))
        out.index_add_(1, self.inner_index, flat)
        return out / self.counts[None, :, None]

    def forward(self, embeddings):
        latent = self.encode(embeddings)
        return self.decode(latent), latent

    @classmethod
    def from_checkpoint(cls, checkpoint: dict, layout: dict):
        model_cls = {"cnn": CubeCNNCodec, "vit": CubeViTCodec}[checkpoint["kind"]]
        model = model_cls(layout, **checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model.eval()

    def decoder_parameter_count(self) -> int:
        return sum(p.numel() for name, p in self.named_parameters() if name.startswith("dec"))


# ---------------------------------------------------------------- CNN

class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.GroupNorm(8, channels), nn.GELU(), nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels), nn.GELU(), nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x)


class CubeCNNCodec(CubeCodecBase):
    """51x51 face images -> (51 / 2^stages) latent grid per face, cube halos at both ends.

    stages=2: halo 4, 59 -> 30 -> 15, crop 1 -> 13x13 latent (nodes 0, 4, ..., 48)
    stages=1: halo 2, 55 -> 28,       crop 1 -> 26x26 latent (nodes 0, 2, ..., 50)
    """

    kind = "cnn"

    def __init__(self, layout: dict, embed_dim: int, hidden: int = 128,
                 latent_channels: int = 64, blocks: int = 2, stages: int = 2):
        if stages not in (1, 2):
            raise ValueError("stages must be 1 or 2")
        super().__init__(layout, embed_dim, 2 ** stages)
        self.config = {"embed_dim": int(embed_dim), "hidden": int(hidden),
                       "latent_channels": int(latent_channels), "blocks": int(blocks),
                       "stages": int(stages)}
        h, step = hidden, 2 ** stages
        self.stages = stages
        self.latent_side = -(-self.nodes // step)          # ceil(51 / step): 13 or 26
        encoder = [nn.Conv2d(embed_dim, h, 3, padding=1), *[ResBlock(h) for _ in range(blocks)]]
        for _ in range(stages):
            encoder += [nn.Conv2d(h, h, 3, stride=2, padding=1), *[ResBlock(h) for _ in range(blocks)]]
        encoder += [nn.GroupNorm(8, h), nn.GELU(), nn.Conv2d(h, latent_channels, 1)]
        self.enc = nn.Sequential(*encoder)
        self.dec_in = nn.Sequential(nn.Conv2d(latent_channels, h, 1),
                                    *[ResBlock(h) for _ in range(blocks)])
        self.dec_ups = nn.ModuleList(
            nn.Sequential(nn.Conv2d(h, h, 3, padding=1), *[ResBlock(h) for _ in range(blocks)])
            for _ in range(stages))
        self.dec_out = nn.Sequential(nn.GroupNorm(8, h), nn.GELU(), nn.Conv2d(h, embed_dim, 3, padding=1))
        # decoder output pixel m sits near node m - 1.5*step + 0.5, so node 0 starts here
        self.output_start = int(1.5 * step - 0.5)

        # latent position k sits on input node step*(k-1); k = 1..side are stored,
        # k = 0 and side+1 are halo filled from neighbouring faces
        latent_angles = -45.0 + (90.0 / self.n) * step * (np.arange(self.latent_side + 2) - 1)
        halo_index = cube_center_halo_index(self.axes, latent_angles, slice(1, self.latent_side + 1))
        self.register_buffer("latent_halo_index", torch.as_tensor(halo_index, dtype=torch.long),
                             persistent=False)

    def encode(self, embeddings):
        faces = self.to_faces(embeddings)
        t = faces.shape[0]
        z = self.enc(faces.flatten(0, 1))
        z = z[:, :, 1:1 + self.latent_side, 1:1 + self.latent_side]
        return z.reshape(t, 6, *z.shape[1:])                            # [T, 6, c, side, side]

    def decode(self, latent):
        t, _, c, side, _ = latent.shape
        flat = latent.permute(0, 1, 3, 4, 2).reshape(t, 6 * side * side, c)
        padded = flat[:, self.latent_halo_index].permute(0, 1, 4, 2, 3)   # [T, 6, c, side+2, side+2]
        x = self.dec_in(padded.flatten(0, 1))
        for block in self.dec_ups:
            x = block(F.interpolate(x, scale_factor=2, mode="nearest"))
        a = self.output_start
        x = self.dec_out(x)[:, :, a:a + self.nodes, a:a + self.nodes]
        return self.from_faces(x.reshape(t, 6, *x.shape[1:]))


# ---------------------------------------------------------------- ViT

class CubeViTCodec(CubeCodecBase):
    """5x5-cell tokens over all 6 faces, global attention with 3-D sphere positions."""

    kind = "vit"

    def __init__(self, layout: dict, embed_dim: int, width: int = 256, depth: int = 4,
                 heads: int = 8, latent_dim: int = 90, patch: int = 5, dropout: float = 0.0,
                 halo: int = 2):
        # (patch 5, halo 2): 55 = 11 tokens x 5 per side, 726 tokens
        # (patch 3, halo 3): 57 = 19 tokens x 3 per side, 2166 tokens
        super().__init__(layout, embed_dim, halo)
        self.INPUT_HALO = int(halo)
        self.config = {"embed_dim": int(embed_dim), "width": int(width), "depth": int(depth),
                       "heads": int(heads), "latent_dim": int(latent_dim), "patch": int(patch),
                       "dropout": float(dropout), "halo": int(halo)}
        side = self.nodes + 2 * self.INPUT_HALO
        if side % patch:
            raise ValueError(f"padded face side {side} is not divisible by patch {patch}")
        self.patch = patch
        self.tokens_per_side = side // patch
        self.token_count = 6 * self.tokens_per_side ** 2
        token_in = patch * patch * embed_dim

        # token centre = middle node of each 5x5 block, as a unit vector on the sphere
        node_angles = -45.0 + (90.0 / self.n) * (np.arange(side) - self.INPUT_HALO)
        centres = node_angles[patch // 2::patch]
        xyz = cube_points(self.axes, centres).reshape(-1, 3)
        self.register_buffer("token_xyz", torch.as_tensor(xyz, dtype=torch.float32), persistent=False)

        def transformer():
            layer = nn.TransformerEncoderLayer(width, heads, dim_feedforward=width * 4,
                                               dropout=dropout, batch_first=True, norm_first=True)
            return nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)

        def position_mlp():
            return nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, width))

        self.enc_embed = nn.Linear(token_in, width)
        self.enc_pos = position_mlp()
        self.enc_blocks = transformer()
        self.enc_out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, latent_dim))
        self.dec_embed = nn.Linear(latent_dim, width)
        self.dec_pos = position_mlp()
        self.dec_blocks = transformer()
        self.dec_out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, token_in))

    def encode(self, embeddings):
        faces = self.to_faces(embeddings)                             # [T, 6, D, 55, 55]
        t, _, d = faces.shape[:3]
        g, p = self.tokens_per_side, self.patch
        tokens = faces.reshape(t, 6, d, g, p, g, p).permute(0, 1, 3, 5, 2, 4, 6)
        tokens = tokens.reshape(t, self.token_count, d * p * p)
        x = self.enc_embed(tokens) + self.enc_pos(self.token_xyz)
        return self.enc_out(self.enc_blocks(x))                        # [T, 726, latent_dim]

    def decode(self, latent):
        t = latent.shape[0]
        g, p, d, h = self.tokens_per_side, self.patch, self.embed_dim, self.INPUT_HALO
        x = self.dec_embed(latent) + self.dec_pos(self.token_xyz)
        x = self.dec_out(self.dec_blocks(x))                           # [T, 726, D*p*p]
        faces = x.reshape(t, 6, g, g, d, p, p).permute(0, 1, 4, 2, 5, 3, 6)
        faces = faces.reshape(t, 6, d, g * p, g * p)[:, :, :, h:h + self.nodes, h:h + self.nodes]
        return self.from_faces(faces)


def build_model(kind: str, layout: dict, embed_dim: int, args) -> CubeCodecBase:
    if kind == "cnn":
        return CubeCNNCodec(layout, embed_dim, hidden=args.hidden,
                            latent_channels=args.latent_channels, blocks=args.blocks,
                            stages=args.stages)
    if kind == "vit":
        return CubeViTCodec(layout, embed_dim, width=args.width, depth=args.depth,
                            heads=args.heads, latent_dim=args.latent_dim,
                            patch=args.patch, halo=args.vit_halo)
    raise ValueError(f"unknown step2 model kind: {kind}")
