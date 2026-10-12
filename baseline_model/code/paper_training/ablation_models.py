"""Shared codecs and decoder-only export for the current CLEAR paper settings.

V is Step1Codec with its visibly marked SE update. S is CubeViTCodec. T keeps
the original residual TCN layers and uses global per-channel statistics rather
than a stored per-position mean map. See README_SE_UPDATE_KO.md.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from step1_model import Step1Codec, GroupAxisDecoder
from step2_model import CubeCodecBase, CubeViTCodec
from step3_model import TemporalTCNCodec, ResBlock1d, ChannelNorm
from data_utils import cube_points

CONDITIONS = ("V", "S", "T", "VS", "VT", "VST")


def loss_weights(widths, device="cpu"):
    return torch.cat([torch.full((w,), 1.0 / (len(widths) * w), device=device)
                      for w in widths])


def balanced_mse(prediction, target, weight):
    return ((prediction.float() - target.float()).square() * weight).sum(-1).mean()


class GlobalTemporalCodec(TemporalTCNCodec):
    """Only normalization broadcasting changes; all baseline TCN layers stay the same."""

    def normalize(self, z, positions=None):
        return (z - self.mean.reshape(1, -1, 1)) / self.scale.reshape(1, -1, 1)

    def denormalize(self, z, positions=None):
        return z * self.scale.reshape(1, -1, 1) + self.mean.reshape(1, -1, 1)


class VariableDecoder(nn.Module):
    """The exact baseline variable decoder, without any encoder weights."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.widths = tuple(config["widths"])
        self.offsets = [0]
        for w in self.widths:
            self.offsets.append(self.offsets[-1] + w)
        self.total_features = self.offsets[-1]
        self.embed_dim, self.token_dim = config["embed_dim"], config["token_dim"]
        self.group_widths = sorted(set(self.widths))
        self.decoders = nn.ModuleList()
        for i, w in enumerate(self.group_widths):
            members = [a for a, width in enumerate(self.widths) if width == w]
            features = [[self.offsets[a] + k for k in range(w)] for a in members]
            self.register_buffer(f"members_{i}", torch.tensor(members), persistent=False)
            self.register_buffer(f"features_{i}", torch.tensor(features), persistent=False)
            self.decoders.append(GroupAxisDecoder(len(members), w, self.embed_dim + self.token_dim))
        self.variable_embedding = nn.Parameter(torch.empty(len(self.widths), self.token_dim))

    _group = Step1Codec._group
    decode = Step1Codec.decode


class SpatialDecoder(CubeCodecBase):
    """The exact global ViT decoder. It always consumes ALL spatial tokens of a day."""

    def __init__(self, layout, config):
        super().__init__(layout, config["embed_dim"], config["halo"])
        self.config = config
        self.INPUT_HALO, self.patch = config["halo"], config["patch"]
        side = self.nodes + 2 * self.INPUT_HALO
        if side % self.patch:
            raise ValueError("face width must be divisible by patch size")
        self.tokens_per_side = side // self.patch
        self.token_count = 6 * self.tokens_per_side ** 2
        angles = -45.0 + 90.0 / self.n * (np.arange(side) - self.INPUT_HALO)
        xyz = cube_points(self.axes, angles[self.patch // 2::self.patch]).reshape(-1, 3)
        self.register_buffer("token_xyz", torch.as_tensor(xyz, dtype=torch.float32), persistent=False)
        width = config["width"]
        self.dec_embed = nn.Linear(config["latent_dim"], width)
        self.dec_pos = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, width))
        layer = nn.TransformerEncoderLayer(width, config["heads"], width * 4,
                                            dropout=config["dropout"], batch_first=True, norm_first=True)
        self.dec_blocks = nn.TransformerEncoder(layer, config["depth"], enable_nested_tensor=False)
        self.dec_out = nn.Sequential(nn.LayerNorm(width),
                                     nn.Linear(width, self.patch ** 2 * config["embed_dim"]))

    decode = CubeViTCodec.decode


class TemporalDecoder(nn.Module):
    def __init__(self, config, mean=None, scale=None):
        super().__init__()
        self.config = config
        c, h, d, b = (config[k] for k in ("channels", "hidden", "latent_channels", "blocks"))
        self.linear_path = config["linear_path"]
        self.register_buffer("mean", torch.as_tensor(
            np.zeros((1, c), np.float32) if mean is None else mean))
        self.register_buffer("scale", torch.as_tensor(
            np.ones(c, np.float32) if scale is None else scale))
        self.dec_in = nn.Sequential(nn.Conv1d(d, h, 1), ResBlock1d(h))
        self.dec_up = nn.ConvTranspose1d(h, h, 4, stride=2, padding=1)
        self.dec_out = nn.Sequential(*[ResBlock1d(h) for _ in range(b)], ChannelNorm(h), nn.GELU(),
                                     nn.Conv1d(h, c, 3, padding=1, padding_mode="replicate"))
        if self.linear_path:
            self.dec_linear = nn.ConvTranspose1d(d, c, 2, stride=2)

    decode = TemporalTCNCodec.decode
    denormalize = GlobalTemporalCodec.denormalize


def configs(condition, terminal_dim, widths, settings):
    """Upstream dimensions are fixed across the conditions that share a prefix."""
    f = sum(widths)
    result = {}
    if "V" in condition:
        result["V"] = dict(widths=list(widths), token_dim=settings["token_dim"],
                           embed_dim=terminal_dim if condition == "V" else settings["v_dim"],
                           variable_slots=settings["variable_slots"], heads=settings["v_heads"],
                           dropout=0.1, se_gate=True)
    if "S" in condition:
        result["S"] = dict(embed_dim=settings["v_dim"] if "V" in condition else f,
                           width=settings["s_width"], depth=settings["s_depth"], heads=settings["s_heads"],
                           latent_dim=settings["s_dim"] if condition == "VST" else terminal_dim,
                           patch=settings["patch"], dropout=0.0, halo=settings["halo"])
    if "T" in condition:
        c = settings["s_dim"] if "S" in condition else (settings["v_dim"] if "V" in condition else f)
        result["T"] = dict(channels=c, hidden=settings["t_hidden"], latent_channels=terminal_dim,
                           blocks=settings["t_blocks"], linear_path=True,
                           normalization_mode="global_per_channel")
    return result


def make_codec(stage, config, layout=None, stats=None):
    if stage == "V":
        return Step1Codec(**config)
    if stage == "S":
        return CubeViTCodec(layout, **config)
    c = config["channels"]
    mean, scale = stats if stats is not None else (np.zeros((1, c), np.float32), np.ones(c, np.float32))
    return GlobalTemporalCodec(mean=mean, scale=scale, **config)


def make_decoder(stage, config, layout=None):
    return {"V": lambda: VariableDecoder(config), "S": lambda: SpatialDecoder(layout, config),
            "T": lambda: TemporalDecoder(config)}[stage]()


def decoder_state(stage, codec):
    return {k: v.detach().float().cpu().clone() for k, v in codec.state_dict().items()
            if (k.startswith("dec") or (stage == "V" and k == "variable_embedding")
                or (stage == "T" and k in ("mean", "scale")))}


def load_decoder(stage, saved, layout, device):
    model = make_decoder(stage, saved["config"], layout)
    model.load_state_dict(saved["state_dict"], strict=True)
    return model.to(device).eval().requires_grad_(False)


def decoder_bytes(stage_configs, layout):
    # No large CPU/GPU allocation while searching dimensions.
    with torch.device("meta"):
        return sum(sum(p.numel() * p.element_size() for p in make_decoder(s, c, layout).parameters())
                   + (8 * c["channels"] if s == "T" else 0)
                   for s, c in stage_configs.items())


DEFAULT_SETTINGS = dict(token_dim=96, variable_slots=4, v_heads=4, v_dim=256,
                        s_width=512, s_depth=4, s_heads=8, s_dim=480, patch=3, halo=3,
                        t_hidden=480, t_blocks=2)
