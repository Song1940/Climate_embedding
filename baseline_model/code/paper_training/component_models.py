"""CLEAR SE factory; gate implementation is in original step1_model.py."""
from __future__ import annotations

import torch

from ablation_models import make_codec
from step1_model import Step1Codec

CONDITIONS = ("M1",)
CHANGES = {"M1": "V encoder: identity-initialized SE-style variable gate"}
FIRST_CHANGED = {"M1": "V"}


def variable_tokens(codec, values):
    """Unmodified variable tokenization and mean-absolute-feature descriptor."""
    batch = len(values)
    tokens = values.new_zeros(batch, len(codec.widths), codec.token_dim)
    for i, encoder in enumerate(codec.encoders):
        members, features = codec._group(i)
        group_tokens = encoder(values[:, features])
        tokens = tokens.index_copy(1, members, group_tokens.to(tokens.dtype))
    return tokens, tokens.float().abs().mean(-1)


SEVariableCodec = Step1Codec  # Compatibility name; not a second hidden gate.


def component_codec(stage, config, condition, layout=None, stats=None,
                    local_neighbors=8, local_query_chunk=128):
    if condition != "M1":
        raise ValueError("This delivery supports CLEAR with SE only (M1).")
    base = {k: v for k, v in config.items() if not k.startswith("ablation_")}
    if stage == "V":
        base["se_gate"] = True
    model = make_codec(stage, base, layout, stats)
    model.config = {**dict(config), **({"se_gate": True} if stage == "V" else {})}
    return model
