"""Sparse synthetic rank-two updates with an exact 2-by-3 active corner."""

from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from .manifest import FIXTURE_PROJECTIONS as PROJECTIONS


def generate(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source, native = {}, {}
    for i, p in enumerate(PROJECTIONS, start=1):
        a = np.zeros((2, p.inputs), dtype="float32")
        b = np.zeros((p.outputs, 2), dtype="float32")
        a[:, :3] = [[1, 2, -1], [0, 1, 3]]
        b[:2] = np.array([[2, 0], [-1, 2]]) * i
        source[p.kohya + ".lora_down.weight"] = a
        source[p.kohya + ".lora_up.weight"] = b
        source[p.kohya + ".alpha"] = np.array(1, dtype="float32")
        native[p.native + ".lora_A.weight"] = a
        native[p.native + ".lora_B.weight"] = b / 2
    mutated = native.copy()
    key = PROJECTIONS[1].native + ".lora_B.weight"
    mutated[key] = mutated[key] / 2
    mixed = {**mutated, "text_encoder.extra.bias": np.ones(1, dtype="float32")}
    cases = {"kohya": source, "native": native, "half": mutated, "mixed": mixed}
    if any((directory / (name + ".safetensors")).exists() for name in cases):
        raise ValueError("fixture output already exists")
    for name, tensors in cases.items():
        save_file(tensors, directory / (name + ".safetensors"))
