"""Information-first all-variable compression model.

The expensive spatial and temporal relationships are learned before the only
axis-reducing operation (the temporal bottleneck).  The horizontal grid count
and the number of grid tokens are preserved through the exchange backbone.

제 쪽·hybrid 모델과 달리 패치를 전혀 쓰지 않고,
구면 k-최근접이웃 그래프 위에서 attention(`SphericalGraphTransformerBlock`)으로
공간 관계를 학습하는 별도 계열의 모델입니다. 15,002개 격자를 줄이기 전에
공간·시간 관계를 먼저 충분히 학습시킨다는 철학("information-first")이 핵심입니다.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint


class VariableAxisEncoder(nn.Module):
    """Encode one variable's complete non-grid axis into one token."""

    def __init__(self, width: int, token_dim: int, axis_hidden: int = 48):
        super().__init__()
        self.width = int(width)
        self.bins = min(12, self.width)
        if self.width == 1:
            self.network = nn.Sequential(
                nn.Linear(1, token_dim), nn.GELU(), nn.LayerNorm(token_dim)
            )
        else:
            self.network = nn.Sequential(
                nn.Conv1d(1, axis_hidden, 3, stride=2, padding=1), nn.GELU(),
                nn.Conv1d(axis_hidden, axis_hidden, 3, stride=2, padding=1), nn.GELU(),
                nn.AdaptiveAvgPool1d(self.bins), nn.Flatten(),
                nn.Linear(axis_hidden * self.bins, token_dim), nn.GELU(),
                nn.LayerNorm(token_dim),
            )

    def forward(self, values):
        if self.width == 1:
            return self.network(values)
        return self.network(values[:, None, :])


class VariableAxisDecoder(nn.Module):
    """Decode one contextual variable token to its original feature axis."""

    def __init__(self, width: int, input_dim: int, axis_hidden: int = 48):
        super().__init__()
        self.width = int(width)
        self.bins = min(12, self.width)
        if self.width == 1:
            self.scalar = nn.Sequential(
                nn.Linear(input_dim, input_dim), nn.GELU(), nn.Linear(input_dim, 1)
            )
            self.seed = None
        else:
            self.scalar = None
            self.seed = nn.Sequential(
                nn.Linear(input_dim, axis_hidden * self.bins), nn.GELU()
            )
            self.refine = nn.Sequential(
                nn.Conv1d(axis_hidden, axis_hidden, 3, padding=1), nn.GELU(),
                nn.Conv1d(axis_hidden, 1, 3, padding=1),
            )

    def forward(self, hidden):
        if self.scalar is not None:
            return self.scalar(hidden)
        values = self.seed(hidden).reshape(len(hidden), -1, self.bins)
        values = F.interpolate(values, size=self.width, mode="linear", align_corners=False)
        return self.refine(values).squeeze(1)


class MultiTokenVariableCodec(nn.Module):
    """All variables <-> multiple high-capacity tokens for every grid cell."""

    def __init__(
        self,
        widths: Sequence[int],
        token_dim: int = 128,
        grid_slots: int = 8,
        heads: int = 8,
        sparse_indices: Sequence[int] = (),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.widths = tuple(int(width) for width in widths)
        self.offsets = [0]
        for width in self.widths:
            self.offsets.append(self.offsets[-1] + width)
        self.total_features = self.offsets[-1]
        self.token_dim = int(token_dim)
        self.grid_slots = int(grid_slots)
        self.grid_dim = self.grid_slots * self.token_dim
        self.sparse_indices = tuple(int(index) for index in sparse_indices)

        self.encoders = nn.ModuleList(
            VariableAxisEncoder(width, token_dim) for width in self.widths
        )
        self.variable_embedding = nn.Parameter(
            torch.randn(len(self.widths), token_dim) * 0.02
        )
        self.context_queries = nn.Parameter(
            torch.randn(grid_slots, token_dim) * 0.02
        )
        self.variable_to_grid = nn.MultiheadAttention(
            token_dim, heads, dropout=dropout, batch_first=True
        )
        encoder_layer = nn.TransformerEncoderLayer(
            token_dim, heads, dim_feedforward=token_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.grid_refiner = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.grid_to_variable = nn.MultiheadAttention(
            token_dim, heads, dropout=dropout, batch_first=True
        )
        decoder_input = token_dim * 2
        self.decoders = nn.ModuleList(
            VariableAxisDecoder(width, decoder_input) for width in self.widths
        )
        self.occurrence_heads = nn.ModuleDict({
            str(index): nn.Linear(decoder_input, self.widths[index])
            for index in self.sparse_indices
        })

    def encode_tokens(self, normalized_flat):
        variable_tokens = []
        for index, (encoder, start, stop) in enumerate(
            zip(self.encoders, self.offsets[:-1], self.offsets[1:])
        ):
            token = encoder(normalized_flat[:, start:stop])
            variable_tokens.append(token + self.variable_embedding[index])
        variable_tokens = torch.stack(variable_tokens, dim=1)
        queries = self.context_queries[None].expand(len(variable_tokens), -1, -1)
        grid_tokens, _ = self.variable_to_grid(
            queries, variable_tokens, variable_tokens, need_weights=False
        )
        return self.grid_refiner(grid_tokens)

    def encode(self, normalized_flat):
        return self.encode_tokens(normalized_flat).flatten(1)

    def _decoder_hidden(self, grid_codes):
        tokens = grid_codes.reshape(len(grid_codes), self.grid_slots, self.token_dim)
        queries = self.variable_embedding[None].expand(len(tokens), -1, -1)
        contextual, _ = self.grid_to_variable(
            queries, tokens, tokens, need_weights=False
        )
        embeddings = self.variable_embedding[None].expand(len(tokens), -1, -1)
        return torch.cat([contextual, embeddings], dim=-1)

    def decode_with_occurrence(self, grid_codes):
        hidden = self._decoder_hidden(grid_codes)
        values = [
            decoder(hidden[:, index])
            for index, decoder in enumerate(self.decoders)
        ]
        logits = {
            int(index): head(hidden[:, int(index)])
            for index, head in self.occurrence_heads.items()
        }
        return torch.cat(values, dim=-1), logits

    def decode(self, grid_codes):
        return self.decode_with_occurrence(grid_codes)[0]

    def forward(self, normalized_flat):
        codes = self.encode(normalized_flat)
        prediction, logits = self.decode_with_occurrence(codes)
        return prediction, codes, logits

    def variable_balanced_mse(self, prediction, target):
        losses = []
        for start, stop in zip(self.offsets[:-1], self.offsets[1:]):
            losses.append(
                (prediction[:, start:stop].float() - target[:, start:stop].float())
                .square().mean()
            )
        return torch.stack(losses).mean()


class SphericalGraphTransformerBlock(nn.Module):
    """Local graph attention with direction and chord-distance edge bias."""

    def __init__(self, dimension: int, heads: int, dropout: float, chunk_size: int):
        super().__init__()
        if dimension % heads:
            raise ValueError("dimension must be divisible by heads")
        self.dimension = int(dimension)
        self.heads = int(heads)
        self.head_dim = self.dimension // self.heads
        self.chunk_size = int(chunk_size)
        self.norm1 = nn.LayerNorm(dimension)
        self.query = nn.Linear(dimension, dimension)
        self.key = nn.Linear(dimension, dimension)
        self.value = nn.Linear(dimension, dimension)
        self.edge_bias = nn.Sequential(
            nn.Linear(4, dimension // 2), nn.GELU(),
            nn.Linear(dimension // 2, heads),
        )
        self.output = nn.Linear(dimension, dimension)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dimension)
        self.mlp = nn.Sequential(
            nn.Linear(dimension, dimension * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dimension * 4, dimension), nn.Dropout(dropout),
        )

    def forward(self, values, neighbours, edge_features):
        # values [batch, grid, dim], neighbours [grid, k]
        normalized = self.norm1(values)
        batches, nodes, _ = normalized.shape
        query = self.query(normalized).reshape(
            batches, nodes, self.heads, self.head_dim
        )
        key = self.key(normalized).reshape(
            batches, nodes, self.heads, self.head_dim
        )
        value = self.value(normalized).reshape(
            batches, nodes, self.heads, self.head_dim
        )
        chunks = []
        scale = self.head_dim ** -0.5
        for start in range(0, nodes, self.chunk_size):
            stop = min(nodes, start + self.chunk_size)
            indices = neighbours[start:stop]
            q = query[:, start:stop]
            k = key[:, indices]
            v = value[:, indices]
            score = torch.einsum("bchd,bckhd->bckh", q, k) * scale
            bias = self.edge_bias(edge_features[start:stop])
            weight = torch.softmax(score + bias[None], dim=2)
            context = torch.einsum(
                "bckh,bckhd->bchd", self.dropout(weight), v
            ).reshape(batches, stop - start, self.dimension)
            chunks.append(context)
        context = torch.cat(chunks, dim=1)
        values = values + self.dropout(self.output(context))
        return values + self.mlp(self.norm2(values))


class ChunkedTemporalTransformerBlock(nn.Module):
    """Self-attention across time while chunking independent grid-slot sequences."""

    def __init__(self, dimension: int, heads: int, dropout: float, chunk_size: int):
        super().__init__()
        self.chunk_size = int(chunk_size)
        self.layer = nn.TransformerEncoderLayer(
            dimension, heads, dim_feedforward=dimension * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )

    def forward(self, tokens):
        # tokens [time, grid, slot, dim]
        times, grids, slots, dimension = tokens.shape
        sequence = tokens.permute(1, 2, 0, 3).reshape(grids * slots, times, dimension)
        chunks = []
        for start in range(0, len(sequence), self.chunk_size):
            stop = min(len(sequence), start + self.chunk_size)
            chunks.append(self.layer(sequence[start:stop]))
        sequence = torch.cat(chunks, dim=0)
        return sequence.reshape(grids, slots, times, dimension).permute(2, 0, 1, 3)


def sinusoidal_positions(length: int, dimension: int, device, dtype):
    position = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    scale = torch.exp(
        torch.arange(0, dimension, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / dimension)
    )
    encoding = torch.zeros(length, dimension, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * scale)
    encoding[:, 1::2] = torch.cos(position * scale[:encoding[:, 1::2].shape[1]])
    return encoding.to(dtype=dtype)


class InformationFirstAutoencoder(nn.Module):
    """Exchange spatial/temporal context before temporal bottleneck compression."""

    def __init__(
        self,
        token_dim: int = 128,
        grid_slots: int = 8,
        exchange_slots: int = 4,
        heads: int = 8,
        exchange_layers: int = 3,
        decoder_layers: int = 2,
        graph_chunk_size: int = 512,
        temporal_chunk_size: int = 256,
        slot_chunk_size: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.grid_slots = int(grid_slots)
        self.exchange_slots = int(exchange_slots)
        if not 0 < self.exchange_slots <= self.grid_slots:
            raise ValueError("exchange_slots must be in [1, grid_slots]")
        self.grid_dim = self.token_dim * self.grid_slots
        self.exchange_layers = int(exchange_layers)
        self.slot_chunk_size = int(slot_chunk_size)
        self.compressed_slot_queries = nn.Parameter(
            torch.randn(self.exchange_slots, token_dim) * 0.02
        )
        self.slot_compressor = nn.MultiheadAttention(
            token_dim, heads, dropout=dropout, batch_first=True
        )
        compress_layer = nn.TransformerEncoderLayer(
            token_dim, heads, dim_feedforward=token_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.slot_compressor_refiner = nn.TransformerEncoder(
            compress_layer, num_layers=1
        )
        self.expanded_slot_queries = nn.Parameter(
            torch.randn(self.grid_slots, token_dim) * 0.02
        )
        self.slot_expander = nn.MultiheadAttention(
            token_dim, heads, dropout=dropout, batch_first=True
        )
        expand_layer = nn.TransformerEncoderLayer(
            token_dim, heads, dim_feedforward=token_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.slot_expander_refiner = nn.TransformerEncoder(
            expand_layer, num_layers=1
        )
        self.coordinate_mlp = nn.Sequential(
            nn.Linear(5, token_dim), nn.GELU(), nn.Linear(token_dim, token_dim)
        )
        self.spatial_encoder = nn.ModuleList([
            SphericalGraphTransformerBlock(
                token_dim, heads, dropout, graph_chunk_size
            ) for _ in range(exchange_layers)
        ])
        self.temporal_encoder = nn.ModuleList([
            ChunkedTemporalTransformerBlock(
                token_dim, heads, dropout, temporal_chunk_size
            ) for _ in range(exchange_layers)
        ])
        self.temporal_down = nn.Conv1d(
            token_dim, token_dim, kernel_size=3, stride=2, padding=1
        )
        self.temporal_refine = nn.Conv1d(
            token_dim, token_dim, kernel_size=3, padding=1
        )
        self.temporal_decoder = nn.ModuleList([
            ChunkedTemporalTransformerBlock(
                token_dim, heads, dropout, temporal_chunk_size
            ) for _ in range(decoder_layers)
        ])
        self.spatial_decoder = nn.ModuleList([
            SphericalGraphTransformerBlock(
                token_dim, heads, dropout, graph_chunk_size
            ) for _ in range(decoder_layers)
        ])
        self.output_head = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, token_dim))
        self.spatial_pretrain_head = nn.Sequential(
            nn.LayerNorm(token_dim), nn.Linear(token_dim, token_dim)
        )
        self.temporal_pretrain_head = nn.Sequential(
            nn.LayerNorm(token_dim), nn.Linear(token_dim, token_dim)
        )

    def _reshape_input(self, grid_codes):
        return grid_codes.reshape(
            grid_codes.shape[0], grid_codes.shape[1], self.grid_slots, self.token_dim
        )

    def _flatten(self, tokens):
        return tokens.reshape(tokens.shape[0], tokens.shape[1], -1)

    def _compress_slots(self, grid_codes):
        input_tokens = self._reshape_input(grid_codes)
        times, grids, _, dimension = input_tokens.shape
        values = input_tokens.reshape(times * grids, self.grid_slots, dimension)
        chunks = []
        for start in range(0, len(values), self.slot_chunk_size):
            stop = min(len(values), start + self.slot_chunk_size)
            chunk = values[start:stop]
            queries = self.compressed_slot_queries[None].expand(
                stop - start, -1, -1
            )
            if self.training:
                compressed_chunk = activation_checkpoint(
                    self._compress_slot_values,
                    queries,
                    chunk,
                    use_reentrant=False,
                )
            else:
                compressed_chunk = self._compress_slot_values(queries, chunk)
            chunks.append(compressed_chunk)
        compressed = torch.cat(chunks, dim=0)
        return compressed.reshape(
            times, grids, self.exchange_slots, dimension
        )

    def _compress_slot_values(self, queries, values):
        compressed, _ = self.slot_compressor(
            queries, values, values, need_weights=False
        )
        return self.slot_compressor_refiner(compressed)

    def _expand_slots(self, compressed):
        times, grids, _, dimension = compressed.shape
        values = compressed.reshape(
            times * grids, self.exchange_slots, dimension
        )
        chunks = []
        for start in range(0, len(values), self.slot_chunk_size):
            stop = min(len(values), start + self.slot_chunk_size)
            chunk = values[start:stop]
            queries = self.expanded_slot_queries[None].expand(
                stop - start, -1, -1
            )
            if self.training:
                expanded_chunk = activation_checkpoint(
                    self._expand_slot_values,
                    queries,
                    chunk,
                    use_reentrant=False,
                )
            else:
                expanded_chunk = self._expand_slot_values(queries, chunk)
            chunks.append(expanded_chunk)
        expanded = torch.cat(chunks, dim=0)
        return expanded.reshape(times, grids, self.grid_slots, dimension)

    def _expand_slot_values(self, queries, values):
        expanded, _ = self.slot_expander(
            queries, values, values, need_weights=False
        )
        return self.slot_expander_refiner(expanded)

    def slot_bottleneck_parameters(self):
        yield self.compressed_slot_queries
        yield self.expanded_slot_queries
        for module in (
            self.slot_compressor,
            self.slot_compressor_refiner,
            self.slot_expander,
            self.slot_expander_refiner,
        ):
            yield from module.parameters()

    def _spatial(self, tokens, block, neighbours, edge_features):
        times, grids, slots, dimension = tokens.shape
        values = tokens.permute(0, 2, 1, 3).reshape(times * slots, grids, dimension)
        if self.training:
            # Full-grid joint training otherwise retains every Q/K/V and MLP
            # activation for all time x slot batches. Recompute them during
            # backward instead of exhausting a 96 GiB GPU.
            values = activation_checkpoint(
                block, values, neighbours, edge_features, use_reentrant=False
            )
        else:
            values = block(values, neighbours, edge_features)
        return values.reshape(times, slots, grids, dimension).permute(0, 2, 1, 3)

    def _temporal(self, tokens, block):
        if self.training:
            return activation_checkpoint(
                block, tokens, use_reentrant=False
            )
        return block(tokens)

    def spatial_pretrain(self, grid_codes, node_features, neighbours, edge_features, mask=None):
        tokens = self._compress_slots(grid_codes)
        if mask is not None:
            tokens = tokens.clone()
            tokens[:, mask] = 0
        tokens = tokens + self.coordinate_mlp(node_features)[None, :, None, :]
        for block in self.spatial_encoder:
            tokens = self._spatial(tokens, block, neighbours, edge_features)
        tokens = self._expand_slots(tokens)
        return self._flatten(self.spatial_pretrain_head(tokens))

    def temporal_pretrain(self, grid_codes):
        tokens = self._compress_slots(grid_codes)
        positions = sinusoidal_positions(
            tokens.shape[0], self.token_dim, tokens.device, tokens.dtype
        )[:, None, None]
        tokens = tokens + positions
        for block in self.temporal_encoder:
            tokens = self._temporal(tokens, block)
        latent = self._temporal_down(tokens)
        restored = self._temporal_up(latent, grid_codes.shape[0])
        for block in self.temporal_decoder:
            restored = self._temporal(restored, block)
        restored = self._expand_slots(restored)
        return self._flatten(self.temporal_pretrain_head(restored)), latent

    def _temporal_down(self, tokens):
        times, grids, slots, dimension = tokens.shape
        sequence = tokens.permute(1, 2, 3, 0).reshape(grids * slots, dimension, times)
        sequence = F.gelu(self.temporal_down(sequence))
        latent_times = sequence.shape[-1]
        return sequence.reshape(grids, slots, dimension, latent_times).permute(3, 0, 1, 2)

    def _temporal_up(self, latent, target_times):
        latent_times, grids, slots, dimension = latent.shape
        sequence = latent.permute(1, 2, 3, 0).reshape(
            grids * slots, dimension, latent_times
        )
        sequence = F.interpolate(
            sequence, size=target_times, mode="linear", align_corners=False
        )
        sequence = F.gelu(self.temporal_refine(sequence))
        return sequence.reshape(grids, slots, dimension, target_times).permute(3, 0, 1, 2)

    def encode(self, grid_codes, node_features, neighbours, edge_features):
        tokens = self._compress_slots(grid_codes)
        tokens = tokens + self.coordinate_mlp(node_features)[None, :, None, :]
        positions = sinusoidal_positions(
            tokens.shape[0], self.token_dim, tokens.device, tokens.dtype
        )[:, None, None]
        tokens = tokens + positions
        for spatial, temporal in zip(self.spatial_encoder, self.temporal_encoder):
            tokens = self._spatial(tokens, spatial, neighbours, edge_features)
            tokens = self._temporal(tokens, temporal)
        return self._temporal_down(tokens)

    def decode(self, latent, target_times, node_features, neighbours, edge_features):
        tokens = self._temporal_up(latent, target_times)
        for temporal, spatial in zip(self.temporal_decoder, self.spatial_decoder):
            tokens = self._temporal(tokens, temporal)
            tokens = self._spatial(tokens, spatial, neighbours, edge_features)
        tokens = self._expand_slots(tokens)
        return self._flatten(self.output_head(tokens))

    def forward(self, grid_codes, node_features, neighbours, edge_features):
        latent = self.encode(grid_codes, node_features, neighbours, edge_features)
        restored = self.decode(
            latent, grid_codes.shape[0], node_features, neighbours, edge_features
        )
        return restored, latent
