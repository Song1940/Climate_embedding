"""Hybrid all-variable weather compression model.

The model deliberately separates context exchange from irreversible reduction:

1. a pretrained variable/vertical codec supplies eight 128-D tokens per grid;
2. core and halo cells exchange information while every core grid still exists;
3. learned patch queries reduce each approximately-15-cell patch to patch tokens;
4. neighbouring patches exchange information;
5. temporal TCN/attention runs at the original day resolution;
6. only then does a stride-2 convolution reduce the time axis.

The stored representation is [latent_time, patch, patch_slot, latent_dim].

정보 교환(exchange)과 축 압축(reduction)을 의도적으로
분리한 모델입니다. core(약 15셀)+halo(8셀) 단위로 먼저 문맥을 교환한 뒤 패치
토큰으로 줄이고, 인접 패치끼리 다시 교환한 다음, 시간축은 원래 해상도를
유지한 채 TCN/attention을 먼저 적용하고 나서야 stride-2로 축소합니다. 제가 시도한
방식과 달리 halo(더 넓은 문맥)를 별도로 둔 점, 패치당 슬롯을 16개
쓰는 점이 특징입니다.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from information_first_model import (
    ChunkedTemporalTransformerBlock,
    MultiTokenVariableCodec,
    SphericalGraphTransformerBlock,
    sinusoidal_positions,
)


def _encoder(dimension: int, heads: int, layers: int, dropout: float):
    layer = nn.TransformerEncoderLayer(
        dimension,
        heads,
        dim_feedforward=dimension * 4,
        dropout=dropout,
        batch_first=True,
        norm_first=True,
        activation="gelu",
    )
    return nn.TransformerEncoder(layer, num_layers=layers)


class TemporalConvBlock(nn.Module):
    """Residual dilated TCN block that does not change temporal resolution."""

    def __init__(self, dimension: int, dilation: int, dropout: float):
        super().__init__()
        self.norm = nn.GroupNorm(1, dimension)
        self.conv1 = nn.Conv1d(
            dimension,
            dimension * 2,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            dimension * 2, dimension, kernel_size=1
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, values):
        hidden = self.conv1(self.norm(values))
        hidden = F.gelu(hidden)
        hidden = self.dropout(self.conv2(hidden))
        return values + hidden


class HybridExchangeCompressAutoencoder(nn.Module):
    """Exchange at high capacity, then compress the spatial and temporal axes."""

    def __init__(
        self,
        core_cells,
        core_valid,
        combined_cells,
        combined_valid,
        core_features,
        patch_neighbours,
        patch_edge_features,
        input_token_dim: int = 128,
        grid_slots: int = 8,
        latent_dim: int = 256,
        patch_slots: int = 16,
        heads: int = 8,
        local_layers: int = 2,
        patch_exchange_layers: int = 2,
        temporal_layers: int = 2,
        temporal_decoder_layers: int = 2,
        patch_chunk_size: int = 16,
        temporal_chunk_size: int = 128,
        graph_chunk_size: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        if latent_dim % heads:
            raise ValueError("latent_dim must be divisible by heads")
        self.input_token_dim = int(input_token_dim)
        self.grid_slots = int(grid_slots)
        self.grid_dim = self.input_token_dim * self.grid_slots
        self.latent_dim = int(latent_dim)
        self.patch_slots = int(patch_slots)
        self.patch_chunk_size = int(patch_chunk_size)

        self.register_buffer(
            "core_cells", torch.as_tensor(core_cells, dtype=torch.long)
        )
        self.register_buffer(
            "core_valid", torch.as_tensor(core_valid, dtype=torch.bool)
        )
        self.register_buffer(
            "combined_cells", torch.as_tensor(combined_cells, dtype=torch.long)
        )
        self.register_buffer(
            "combined_valid", torch.as_tensor(combined_valid, dtype=torch.bool)
        )
        self.register_buffer(
            "core_features", torch.as_tensor(core_features, dtype=torch.float32)
        )
        self.register_buffer(
            "patch_neighbours", torch.as_tensor(patch_neighbours, dtype=torch.long)
        )
        self.register_buffer(
            "patch_edge_features",
            torch.as_tensor(patch_edge_features, dtype=torch.float32),
        )

        self.patch_count = int(self.core_cells.shape[0])
        self.core_width = int(self.core_cells.shape[1])
        self.local_width = int(self.combined_cells.shape[1])

        self.input_lift = nn.Sequential(
            nn.LayerNorm(input_token_dim),
            nn.Linear(input_token_dim, latent_dim),
        )
        self.output_project = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, input_token_dim),
        )
        self.coordinate_mlp = nn.Sequential(
            nn.Linear(5, latent_dim), nn.GELU(), nn.Linear(latent_dim, latent_dim)
        )
        self.local_position_mlp = nn.Sequential(
            nn.Linear(4, latent_dim), nn.GELU(), nn.Linear(latent_dim, latent_dim)
        )
        self.grid_slot_embedding = nn.Parameter(
            torch.randn(grid_slots, latent_dim) * 0.02
        )
        self.patch_embedding = nn.Parameter(
            torch.randn(self.patch_count, latent_dim) * 0.02
        )
        self.core_halo_embedding = nn.Parameter(
            torch.randn(2, latent_dim) * 0.02
        )

        self.local_encoder = _encoder(
            latent_dim, heads, local_layers, dropout
        )
        self.patch_queries = nn.Parameter(
            torch.randn(patch_slots, latent_dim) * 0.02
        )
        self.patch_pool = nn.MultiheadAttention(
            latent_dim, heads, dropout=dropout, batch_first=True
        )
        self.patch_pool_refiner = _encoder(
            latent_dim, heads, 1, dropout
        )

        self.patch_exchange = nn.ModuleList([
            SphericalGraphTransformerBlock(
                latent_dim, heads, dropout, graph_chunk_size
            )
            for _ in range(patch_exchange_layers)
        ])

        self.temporal_tcn = nn.ModuleList([
            TemporalConvBlock(latent_dim, dilation, dropout)
            for dilation in (1, 2, 4)
        ])
        self.temporal_encoder = nn.ModuleList([
            ChunkedTemporalTransformerBlock(
                latent_dim, heads, dropout, temporal_chunk_size
            )
            for _ in range(temporal_layers)
        ])
        self.temporal_down = nn.Conv1d(
            latent_dim, latent_dim, kernel_size=3, stride=2, padding=1
        )
        self.temporal_refine = nn.Conv1d(
            latent_dim, latent_dim, kernel_size=3, padding=1
        )
        self.temporal_decoder = nn.ModuleList([
            ChunkedTemporalTransformerBlock(
                latent_dim, heads, dropout, temporal_chunk_size
            )
            for _ in range(temporal_decoder_layers)
        ])

        self.cell_slot_base = nn.Parameter(
            torch.randn(grid_slots, latent_dim) * 0.02
        )
        self.patch_expand = nn.MultiheadAttention(
            latent_dim, heads, dropout=dropout, batch_first=True
        )
        self.patch_expand_refiner = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
        )

    @property
    def latent_shape_per_block(self):
        return self.patch_count, self.patch_slots, self.latent_dim

    def spatial_encoder_parameters(self) -> Iterable[nn.Parameter]:
        modules = (
            self.input_lift,
            self.coordinate_mlp,
            self.local_position_mlp,
            self.local_encoder,
            self.patch_pool,
            self.patch_pool_refiner,
            self.patch_exchange,
        )
        yield self.grid_slot_embedding
        yield self.patch_embedding
        yield self.core_halo_embedding
        yield self.patch_queries
        for module in modules:
            yield from module.parameters()

    def temporal_parameters(self) -> Iterable[nn.Parameter]:
        for module in (
            self.temporal_tcn,
            self.temporal_encoder,
            self.temporal_down,
            self.temporal_refine,
            self.temporal_decoder,
        ):
            yield from module.parameters()

    def spatial_decoder_parameters(self) -> Iterable[nn.Parameter]:
        yield self.cell_slot_base
        for module in (
            self.local_position_mlp,
            self.patch_expand,
            self.patch_expand_refiner,
            self.output_project,
        ):
            yield from module.parameters()

    def _run_local(self, sequence, padding_mask):
        return self.local_encoder(
            sequence, src_key_padding_mask=padding_mask
        )

    def _local_context_and_pool(self, grid_codes, node_features):
        """Encode one day [grid, grid_dim] to [patch, slot, latent_dim]."""
        if grid_codes.ndim != 2:
            raise ValueError("grid_codes must have shape [grid, grid_dim]")
        grid_count = int(grid_codes.shape[0])
        tokens = grid_codes.reshape(
            grid_count, self.grid_slots, self.input_token_dim
        )
        tokens = self.input_lift(tokens)
        tokens = tokens + self.coordinate_mlp(node_features)[:, None]
        tokens = tokens + self.grid_slot_embedding.to(dtype=tokens.dtype)[None]

        pooled_chunks = []
        core_width = self.core_width
        slot_count = self.grid_slots
        for start in range(0, self.patch_count, self.patch_chunk_size):
            stop = min(self.patch_count, start + self.patch_chunk_size)
            indices = self.combined_cells[start:stop].clamp_min(0)
            valid = self.combined_valid[start:stop]
            gathered = tokens[indices]

            kinds = torch.ones(
                stop - start,
                self.local_width,
                dtype=torch.long,
                device=tokens.device,
            )
            kinds[:, :core_width] = 0
            kind_embedding = self.core_halo_embedding.to(dtype=gathered.dtype)
            patch_embedding = self.patch_embedding.to(dtype=gathered.dtype)
            gathered = gathered + kind_embedding[kinds][..., None, :]
            gathered = gathered + patch_embedding[start:stop, None, None, :]
            sequence = gathered.reshape(
                stop - start,
                self.local_width * slot_count,
                self.latent_dim,
            )
            padding = (~valid).repeat_interleave(slot_count, dim=1)
            if self.training:
                sequence = activation_checkpoint(
                    self._run_local,
                    sequence,
                    padding,
                    use_reentrant=False,
                )
            else:
                sequence = self._run_local(sequence, padding)

            core = sequence[:, :core_width * slot_count]
            core_padding = (~self.core_valid[start:stop]).repeat_interleave(
                slot_count, dim=1
            )
            queries = self.patch_queries.to(dtype=core.dtype)[None].expand(
                stop - start, -1, -1
            )
            queries = queries + self.patch_embedding.to(dtype=core.dtype)[
                start:stop, None
            ]
            pooled, _ = self.patch_pool(
                queries,
                core,
                core,
                key_padding_mask=core_padding,
                need_weights=False,
            )
            pooled_chunks.append(self.patch_pool_refiner(pooled))
        return torch.cat(pooled_chunks, dim=0)

    def _patch_exchange(self, patch_latent):
        # patch_latent [patch, patch_slot, dim]
        values = patch_latent.permute(1, 0, 2)
        for block in self.patch_exchange:
            if self.training:
                values = activation_checkpoint(
                    block,
                    values,
                    self.patch_neighbours,
                    self.patch_edge_features,
                    use_reentrant=False,
                )
            else:
                values = block(
                    values, self.patch_neighbours, self.patch_edge_features
                )
        return values.permute(1, 0, 2)

    def encode_spatial_day(self, grid_codes, node_features):
        patch_latent = self._local_context_and_pool(grid_codes, node_features)
        return self._patch_exchange(patch_latent)

    def _decoder_queries(self, patch_ids):
        features = self.core_features[patch_ids]
        queries = self.local_position_mlp(features)[:, :, None, :]
        queries = queries + self.cell_slot_base.to(dtype=queries.dtype)[
            None, None, :, :
        ]
        queries = queries + self.patch_embedding.to(dtype=queries.dtype)[
            patch_ids, None, None, :
        ]
        return queries.reshape(
            len(patch_ids), self.core_width * self.grid_slots, self.latent_dim
        )

    def decode_spatial_patches(self, patch_latents, patch_ids=None):
        """Decode selected patches.

        Args:
            patch_latents: [time, selected_patch, patch_slot, latent_dim]
            patch_ids: global patch ids corresponding to selected_patch.

        Returns:
            grid_codes [time, selected_core_cells, grid_dim], core cell ids.
        """
        if patch_ids is None:
            patch_ids = torch.arange(
                self.patch_count, device=patch_latents.device
            )
        patch_ids = torch.as_tensor(
            patch_ids, dtype=torch.long, device=patch_latents.device
        )
        if patch_latents.shape[1] != len(patch_ids):
            raise ValueError("patch_latents and patch_ids disagree")

        times = patch_latents.shape[0]
        decoded_chunks = []
        cell_chunks = []
        for local_start in range(0, len(patch_ids), self.patch_chunk_size):
            local_stop = min(len(patch_ids), local_start + self.patch_chunk_size)
            ids = patch_ids[local_start:local_stop]
            values = patch_latents[:, local_start:local_stop]
            patch_batch = len(ids)
            values = values.reshape(
                times * patch_batch, self.patch_slots, self.latent_dim
            )
            queries = self._decoder_queries(ids)
            queries = queries[None].expand(times, -1, -1, -1).reshape(
                times * patch_batch,
                self.core_width * self.grid_slots,
                self.latent_dim,
            )
            decoded, _ = self.patch_expand(
                queries, values, values, need_weights=False
            )
            decoded = decoded + self.patch_expand_refiner(decoded)
            decoded = self.output_project(decoded)
            decoded = decoded.reshape(
                times,
                patch_batch,
                self.core_width,
                self.grid_slots,
                self.input_token_dim,
            )
            valid = self.core_valid[ids].reshape(-1)
            decoded = decoded.reshape(
                times,
                patch_batch * self.core_width,
                self.grid_slots,
                self.input_token_dim,
            )[:, valid]
            cells = self.core_cells[ids].reshape(-1)[valid]
            decoded_chunks.append(decoded.flatten(2))
            cell_chunks.append(cells)
        return torch.cat(decoded_chunks, dim=1), torch.cat(cell_chunks)

    def decode_spatial_day(self, patch_latent, grid_count: int):
        decoded, cells = self.decode_spatial_patches(patch_latent[None])
        output = decoded.new_zeros(1, grid_count, self.grid_dim)
        output = output.index_copy(1, cells, decoded)
        return output[0]

    def spatial_autoencode_day(self, grid_codes, node_features):
        patch_latent = self.encode_spatial_day(grid_codes, node_features)
        restored = self.decode_spatial_day(patch_latent, len(grid_codes))
        return restored, patch_latent

    def _temporal_tcn(self, values):
        # values [time, patch, slot, dim]
        times, patches, slots, dimension = values.shape
        sequence = values.permute(1, 2, 3, 0).reshape(
            patches * slots, dimension, times
        )
        for block in self.temporal_tcn:
            if self.training:
                sequence = activation_checkpoint(
                    block, sequence, use_reentrant=False
                )
            else:
                sequence = block(sequence)
        return sequence.reshape(
            patches, slots, dimension, times
        ).permute(3, 0, 1, 2)

    def _temporal_attention(self, values, blocks):
        for block in blocks:
            if self.training:
                values = activation_checkpoint(
                    block, values, use_reentrant=False
                )
            else:
                values = block(values)
        return values

    def encode_temporal(self, spatial_latents):
        positions = sinusoidal_positions(
            spatial_latents.shape[0],
            self.latent_dim,
            spatial_latents.device,
            spatial_latents.dtype,
        )[:, None, None]
        values = spatial_latents + positions
        values = self._temporal_tcn(values)
        values = self._temporal_attention(values, self.temporal_encoder)
        times, patches, slots, dimension = values.shape
        sequence = values.permute(1, 2, 3, 0).reshape(
            patches * slots, dimension, times
        )
        sequence = F.gelu(self.temporal_down(sequence))
        latent_times = sequence.shape[-1]
        return sequence.reshape(
            patches, slots, dimension, latent_times
        ).permute(3, 0, 1, 2)

    def decode_temporal(self, latent, target_times: int):
        latent_times, patches, slots, dimension = latent.shape
        sequence = latent.permute(1, 2, 3, 0).reshape(
            patches * slots, dimension, latent_times
        )
        sequence = F.interpolate(
            sequence, size=target_times, mode="linear", align_corners=False
        )
        sequence = F.gelu(self.temporal_refine(sequence))
        values = sequence.reshape(
            patches, slots, dimension, target_times
        ).permute(3, 0, 1, 2)
        return self._temporal_attention(values, self.temporal_decoder)

    def temporal_autoencode(self, spatial_latents):
        latent = self.encode_temporal(spatial_latents)
        restored = self.decode_temporal(latent, spatial_latents.shape[0])
        return restored, latent

    def encode_spatial_sequence(self, grid_codes, node_features):
        return torch.stack([
            self.encode_spatial_day(grid_codes[day], node_features)
            for day in range(grid_codes.shape[0])
        ])

    def encode(self, grid_codes, node_features):
        spatial = self.encode_spatial_sequence(grid_codes, node_features)
        return self.encode_temporal(spatial)

    def decode(self, latent, target_times: int, node_features, grid_count: int):
        spatial = self.decode_temporal(latent, target_times)
        restored = []
        for day in range(target_times):
            restored.append(
                self.decode_spatial_day(spatial[day], grid_count)
            )
        return torch.stack(restored)

    def forward(self, grid_codes, node_features):
        spatial = self.encode_spatial_sequence(grid_codes, node_features)
        restored_spatial, latent = self.temporal_autoencode(spatial)
        restored = []
        for day in range(grid_codes.shape[0]):
            restored.append(
                self.decode_spatial_day(
                    restored_spatial[day], grid_codes.shape[1]
                )
            )
        return torch.stack(restored), latent


__all__ = [
    "HybridExchangeCompressAutoencoder",
    "MultiTokenVariableCodec",
]
