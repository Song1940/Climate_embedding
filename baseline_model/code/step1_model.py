# [2026-10-12 변경 내역]
# - Step1Codec에 SE 방식 변수 게이트를 추가함(기본 활성화).
# - 변수 토큰에 게이트를 곱한 뒤 기존 변수 임베딩/Attention으로 전달함.
# - 디코더는 변경하지 않으며, SE 없는 이전 체크포인트도 로딩 가능하게 함.
"""
Step 1 모델: 격자 셀 하나의 1D 벡터(181개 변수, 연직층 포함 6,035개 값)를
임베딩 하나로 압축하는 코덱입니다. 공간·시간 정보는 전혀 쓰지 않고 셀마다
독립적으로 동작합니다.

- 인코더: 변수마다 전용 소형 1D Conv 인코더 → 변수 토큰(token_dim) → 학습된
  질의(variable_slots개)가 cross-attention으로 요약 → 임베딩(embed_dim).
- 디코더: 임베딩 + 변수 임베딩 → 변수마다 전용 디코더가 원래 길이로 복원.

구조는 old/all_variable_model.py의 VariableGridCodec과 같습니다(변수마다 독립
가중치). 다만 GPU 속도를 위해 너비(연직층 수)가 같은 변수들을 한 그룹으로 묶어
grouped Conv1d와 einsum으로 한 번에 계산합니다. 디코더의 선형 보간
(F.interpolate)은 같은 결과를 내는 고정 행렬곱으로 바꿨습니다. 보간 커널이
학습 시간의 90%를 차지했기 때문입니다. 결과적으로 epoch당 86초 → 약 6초입니다.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


def _uniform(shape, fan_in):
    bound = 1.0 / math.sqrt(fan_in)
    return nn.Parameter(torch.empty(shape).uniform_(-bound, bound))


class GroupLinear(nn.Module):
    """G independent Linear layers applied to x[B, G, in] -> [B, G, out]."""

    def __init__(self, groups: int, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = _uniform((groups, in_dim, out_dim), in_dim)
        self.bias = _uniform((groups, out_dim), in_dim)

    def forward(self, x):
        return torch.einsum("bgi,gio->bgo", x, self.weight) + self.bias


class GroupLayerNorm(nn.Module):
    """LayerNorm over the last dim with a separate affine per group."""

    def __init__(self, groups: int, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(groups, dim))
        self.bias = nn.Parameter(torch.zeros(groups, dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:]) * self.weight + self.bias


class GroupConv1d(nn.Module):
    """G independent Conv1d layers run as one cuDNN grouped convolution.

    x[B, G, C_in, L] -> [B, G, C_out, L_out]. Same maths and init as a separate
    nn.Conv1d per variable. Weights are stored as [G, C_in*k, C_out].
    """

    def __init__(self, groups: int, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.kernel, self.stride, self.padding = kernel, stride, padding
        fan_in = in_ch * kernel
        self.weight = _uniform((groups, fan_in, out_ch), fan_in)
        self.bias = _uniform((groups, out_ch), fan_in)

    def forward(self, x):
        batch, groups, channels, length = x.shape
        weight = (self.weight.reshape(groups, channels, self.kernel, -1)
                  .permute(0, 3, 1, 2).reshape(-1, channels, self.kernel))
        out = F.conv1d(x.reshape(batch, groups * channels, length), weight.to(x.dtype),
                       self.bias.reshape(-1).to(x.dtype), stride=self.stride,
                       padding=self.padding, groups=groups)
        return out.reshape(batch, groups, -1, out.shape[-1])


class GroupAxisEncoder(nn.Module):
    """G variables of equal width w: x[B, G, w] -> tokens[B, G, token_dim]."""

    def __init__(self, groups: int, width: int, token_dim: int, axis_hidden: int = 32):
        super().__init__()
        self.groups, self.width, self.hidden = groups, int(width), axis_hidden
        self.bins = min(8, self.width)
        if self.width == 1:
            self.scalar = GroupLinear(groups, 1, token_dim)
        else:
            self.conv1 = GroupConv1d(groups, 1, axis_hidden, 3, stride=2, padding=1)
            self.conv2 = GroupConv1d(groups, axis_hidden, axis_hidden, 3, stride=2, padding=1)
            self.project = GroupLinear(groups, axis_hidden * self.bins, token_dim)
        self.norm = GroupLayerNorm(groups, token_dim)

    def forward(self, x):
        if self.width == 1:
            hidden = self.scalar(x)
        else:
            hidden = F.gelu(self.conv2(F.gelu(self.conv1(x[:, :, None, :]))))
            batch = len(x)
            hidden = F.adaptive_avg_pool1d(hidden.flatten(1, 2), self.bins)
            hidden = self.project(hidden.reshape(batch, self.groups, self.hidden * self.bins))
        return self.norm(F.gelu(hidden))


class GroupAxisDecoder(nn.Module):
    """G variables of equal width w: hidden[B, G, in] -> values[B, G, w]."""

    def __init__(self, groups: int, width: int, input_dim: int, axis_hidden: int = 32):
        super().__init__()
        self.groups, self.width, self.hidden = groups, int(width), axis_hidden
        self.bins = min(8, self.width)
        if self.width == 1:
            self.first = GroupLinear(groups, input_dim, input_dim)
            self.second = GroupLinear(groups, input_dim, 1)
        else:
            self.seed = GroupLinear(groups, input_dim, axis_hidden * self.bins)
            # F.interpolate(linear, align_corners=False) from bins to width, as a fixed
            # matrix: identical result, but the upsample kernel was ~90% of step time.
            eye = torch.eye(self.bins)[:, None, :]
            upsample = F.interpolate(eye, size=self.width, mode="linear", align_corners=False)
            self.register_buffer("upsample", upsample[:, 0, :], persistent=False)  # [bins, w]
            self.refine1 = GroupConv1d(groups, axis_hidden, axis_hidden, 3, padding=1)
            self.refine2 = GroupConv1d(groups, axis_hidden, 1, 3, padding=1)

    def forward(self, hidden):
        if self.width == 1:
            return self.second(F.gelu(self.first(hidden)))
        batch = len(hidden)
        values = F.gelu(self.seed(hidden)).reshape(batch, self.groups, self.hidden, self.bins)
        values = values @ self.upsample.to(values.dtype)                  # [B, G, h, w]
        return self.refine2(F.gelu(self.refine1(values)))[:, :, 0]


class Step1Codec(nn.Module):
    """[cells, 6035] normalized values <-> [cells, embed_dim] embeddings."""

    def __init__(
        self,
        widths: Sequence[int],
        token_dim: int = 96,
        embed_dim: int = 256,
        variable_slots: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
        se_gate: bool = True,  # [2026-10-12 추가] SE 변수 게이트 활성화 옵션.
    ):
        super().__init__()
        self.config = {
            "widths": [int(w) for w in widths], "token_dim": int(token_dim),
            "embed_dim": int(embed_dim), "variable_slots": int(variable_slots),
            "heads": int(heads), "dropout": float(dropout),
            "se_gate": bool(se_gate),  # [2026-10-12 추가] 체크포인트에 SE 사용 여부 저장.
        }
        self.widths = tuple(int(w) for w in widths)
        offsets = [0]
        for width in self.widths:
            offsets.append(offsets[-1] + width)
        self.offsets = offsets
        self.total_features = offsets[-1]
        self.embed_dim = int(embed_dim)
        self.token_dim = int(token_dim)

        # group variables by width; each group keeps its variables' original order
        self.group_widths = sorted(set(self.widths))
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for index, width in enumerate(self.group_widths):
            members = [v for v, w in enumerate(self.widths) if w == width]
            features = torch.tensor([[offsets[v] + k for k in range(width)] for v in members])
            self.register_buffer(f"members_{index}", torch.tensor(members), persistent=False)
            self.register_buffer(f"features_{index}", features, persistent=False)
            self.encoders.append(GroupAxisEncoder(len(members), width, token_dim))
            self.decoders.append(GroupAxisDecoder(len(members), width, embed_dim + token_dim))

        # loss weights: every variable counts equally regardless of its width
        weight = torch.cat([torch.full((w,), 1.0 / (w * len(self.widths))) for w in self.widths])
        self.register_buffer("loss_weight", weight, persistent=False)

        self.variable_embedding = nn.Parameter(torch.randn(len(self.widths), token_dim) * 0.02)
        self.variable_queries = nn.Parameter(torch.randn(variable_slots, token_dim) * 0.02)
        self.variable_attention = nn.MultiheadAttention(
            token_dim, heads, dropout=dropout, batch_first=True
        )
        layer = nn.TransformerEncoderLayer(
            token_dim, heads, dim_feedforward=token_dim * 3,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.slot_transformer = nn.TransformerEncoder(layer, num_layers=2,
                                                      enable_nested_tensor=False)
        self.embed_projection = nn.Sequential(
            nn.LayerNorm(variable_slots * token_dim),
            nn.Linear(variable_slots * token_dim, embed_dim), nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

        # [2026-10-12 추가 시작: SE 게이트 생성]
        # MLP A -> max(4, A//8) -> A로 변수별 계수를 계산함(181변수: 181->22->181).
        # 마지막 weight/bias=0으로 초기화하여 2*sigmoid(0)=1인 항등 게이트로 시작함.
        # RNG를 분리하여 기존 모델 초기화와 이후 dropout 난수 순서를 유지함.
        self.se_gate = bool(se_gate)
        if self.se_gate:
            a = len(self.widths)
            with torch.random.fork_rng(devices=[]):
                self.enc_se = nn.Sequential(nn.Linear(a, max(4, a // 8)), nn.GELU(),
                                            nn.Linear(max(4, a // 8), a))
                nn.init.zeros_(self.enc_se[-1].weight)
                nn.init.zeros_(self.enc_se[-1].bias)
        # [2026-10-12 추가 끝: SE 게이트 생성] 디코더에는 SE를 추가하지 않음.

    def _group(self, index):
        return getattr(self, f"members_{index}"), getattr(self, f"features_{index}")

    @classmethod
    def from_checkpoint(cls, checkpoint: dict) -> "Step1Codec":
        # [2026-10-12 변경: 체크포인트 호환]
        # 실험 메타 옵션을 제외하고 enc_se 가중치 유무로 이전 모델의 SE 사용 여부를 판단함.
        config = {k: v for k, v in checkpoint["config"].items()
                  if not k.startswith("ablation_")}
        config.setdefault("se_gate", any(k.startswith("enc_se.") for k in checkpoint["state_dict"]))
        codec = cls(**config)
        codec.load_state_dict(checkpoint["state_dict"])
        return codec.eval()

    # [2026-10-12 추가 시작: 변수 토큰 분리 및 SE 적용]
    # 기존 토큰 생성 계산을 함수로 분리함. LayerNorm 이후 부호 평균이 상쇄되는 것을
    # 피하려고 특징 절댓값 평균을 사용하며, 게이트는 변수 identity에 곱하지 않음.
    def variable_tokens(self, values):
        batch = len(values)
        tokens = values.new_zeros((batch, len(self.widths), self.token_dim))
        for index, encoder in enumerate(self.encoders):
            members, features = self._group(index)
            group_tokens = encoder(values[:, features])            # [B, G, token]
            tokens = tokens.index_copy(1, members, group_tokens.to(tokens.dtype))
        return tokens

    def gates(self, values):
        tokens = self.variable_tokens(values)
        if not self.se_gate:
            return tokens.new_ones((len(values), len(self.widths)))
        return 2 * torch.sigmoid(self.enc_se(tokens.float().abs().mean(-1)))

    def encode(self, values):
        batch = len(values)
        tokens = self.variable_tokens(values)
        if self.se_gate:
            gate = 2 * torch.sigmoid(self.enc_se(tokens.float().abs().mean(-1)))
            tokens = tokens * gate.to(tokens.dtype)[:, :, None]
        # [2026-10-12 추가 끝: SE 적용] 아래 변수 임베딩/Attention은 기존 계산을 유지함.
        tokens = tokens + self.variable_embedding
        queries = self.variable_queries.unsqueeze(0).expand(batch, -1, -1)
        slots, _ = self.variable_attention(queries, tokens, tokens, need_weights=False)
        return self.embed_projection(self.slot_transformer(slots).flatten(1))

    def decode(self, embeddings):
        batch = len(embeddings)
        output = None
        for index, decoder in enumerate(self.decoders):
            members, features = self._group(index)
            hidden = torch.cat([
                embeddings[:, None, :].expand(-1, len(members), -1),
                self.variable_embedding[members][None].expand(batch, -1, -1).to(embeddings.dtype),
            ], dim=-1)
            values = decoder(hidden).reshape(batch, -1)            # [B, G*w]
            if output is None:
                output = values.new_zeros((batch, self.total_features))
            output = output.index_copy(1, features.reshape(-1), values.to(output.dtype))
        return output

    def forward(self, values):
        embeddings = self.encode(values)
        return self.decode(embeddings), embeddings

    def variable_balanced_mse(self, prediction, target):
        """Mean over variables of each variable's MSE (each variable counts equally)."""
        squared = (prediction.float() - target.float()).square().mean(0)
        return (squared * self.loss_weight).sum()
