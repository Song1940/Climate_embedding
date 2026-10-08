"""SE block alone for integration into an existing variable/vertical encoder.

Input/output: [batch, variable, token_feature]. This gates named variables,
not the Conv1D hidden filters or individual vertical levels. Torch only.
Equivalent to M1's enc_se; full checkpoint-compatible model is vertical_se_model.py.
"""
from __future__ import annotations

import torch
from torch import nn


class VariableTokenSE(nn.Module):
    def __init__(self, variables: int, reduction: int = 8):
        super().__init__()
        if variables < 1 or reduction < 1:
            raise ValueError("variables and reduction must be positive")
        self.variables = int(variables)
        hidden = max(4, self.variables // int(reduction))
        self.excitation = nn.Sequential(nn.Linear(self.variables, hidden), nn.GELU(),
                                        nn.Linear(hidden, self.variables))
        # Identity at initialization: 2*sigmoid(0) = 1, not 0.5.
        nn.init.zeros_(self.excitation[-1].weight)
        nn.init.zeros_(self.excitation[-1].bias)

    def gates(self, tokens):
        if tokens.ndim != 3 or tokens.shape[1] != self.variables:
            raise ValueError(f"expected [B,{self.variables},D], got {tuple(tokens.shape)}")
        # Signed LayerNorm feature averages nearly cancel, so pool absolute values.
        descriptor = tokens.float().abs().mean(dim=-1)
        return 2.0 * torch.sigmoid(self.excitation(descriptor))

    def forward(self, tokens):
        return tokens * self.gates(tokens).to(tokens.dtype).unsqueeze(-1)
