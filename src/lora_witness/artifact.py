"""Read bounded private snapshots, validate headers, and inventory both dialects."""

import hashlib
import json
import math
import os
import stat
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from safetensors import SafetensorError, safe_open

from .manifest import FUSED, INDEX, PROJECTIONS, SUFFIXES

MAX_BYTES = 64 * 1024**2
MAX_HEADER = 512 * 1024
MAX_ELEMENTS = 8_000_000
MAX_KEYS = 4096
MAX_RANK = 64


def unique(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate header key {key!a}")
        obj[key] = value
    return obj


def snapshot(path, dest):
    if path.suffix != ".safetensors":
        raise ValueError("expected a .safetensors file")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(fd, "rb") as source, dest.open("wb") as output:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_BYTES:
            raise ValueError("expected a regular file of at most 64 MiB")
        digest = hashlib.sha256()
        while chunk := source.read(1024 * 1024):
            if output.tell() + len(chunk) > MAX_BYTES:
                raise ValueError("file exceeds 64 MiB")
            output.write(chunk)
            digest.update(chunk)
    return digest.hexdigest()


def check_header(path):
    with path.open("rb") as f:
        size = int.from_bytes(f.read(8), "little")
        if not 2 <= size <= MAX_HEADER:
            raise ValueError("invalid or over-limit header length")
        header = json.loads(f.read(size), object_pairs_hook=unique)
    if not isinstance(header, dict) or len(header) > MAX_KEYS + 1:
        raise ValueError("invalid or over-limit key table")

    return header, 8 + size


def finite_payload(stream, start, length, dtype):
    # Scan in fixed-size slices, including IEEE/BF16 values on unsupported paths.
    formats = {
        "F16": "<f2",
        "F32": "<f4",
        "F64": "<f8",
        "BF16": "<u2",
        "F8_E5M2": "u1",
        "F8_E4M3": "u1",
        "F8_E8M0": "u1",
        "C64": "<c8",
    }
    if dtype not in formats:
        return True
    stream.seek(start)
    while length:
        chunk = stream.read(min(length, 256 * 1024))
        if not chunk:
            raise ValueError("unexpected end of tensor payload")
        values = np.frombuffer(chunk, dtype=formats[dtype])
        if dtype == "BF16":
            finite = (values & 0x7F80) != 0x7F80
        elif dtype == "F8_E5M2":
            finite = (values & 0x7C) != 0x7C
        elif dtype == "F8_E4M3":
            finite = (values & 0x7F) != 0x7F
        elif dtype == "F8_E8M0":
            finite = values != 255
        else:
            finite = np.isfinite(values)
        if not finite.all():
            return False
        length -= len(chunk)
    return True


def native_configuration_issues(keys):
    """Detect UNet-wide configuration inference in Diffusers 0.43.0."""
    # The SDXL loader first drops dora_scale entries, then filters by component.
    unet = [k.removeprefix("unet.") for k in keys if k.startswith("unet.") and "dora_scale" not in k]
    issues = []
    if any("lora_magnitude_vector" in k for k in unet):
        issues.append("UNet-wide DoRA magnitude configuration")
    if any(".lokr_" in k for k in unet):
        issues.append("UNet-wide LoKr configuration")
    if "lora_controlnet" in unet:
        issues.append("UNet-wide SAI Control LoRA configuration")
    # The loader can rewrite legacy up keys to lora_B before inferring lora_bias.
    up_bias = (".lora.up", ".to_q_lora.up", ".to_k_lora.up", ".to_v_lora.up", ".to_out_lora.up")
    if any(k.endswith(".bias") and ("lora_B" in k or any(name in k for name in up_bias)) for k in unet):
        issues.append("UNet-wide LoRA B bias configuration")
    return issues


class Artifact:
    def __init__(self, side, path, profile, stack):
        self.side = side
        self.groups = {}
        self.invalid = set()
        self.unknown = set()
        self.global_unknown = []
        self.complete = False
        self.handle = None
        self.r = {
            "file": path.name,
            "sha256": None,
            "raw_keys": None,
            "parameters": None,
            "keys": {},
            "matched_keys": [],
            "missing_keys": [],
            "extra_keys": [],
            "unpaired_keys": [],
            "ambiguous_keys": [],
            "unsupported_keys": [],
            "paired_projections": [],
            "compared_keys": [],
            "verified_attachments": [],
            "errors": [],
            "configuration_issues": self.global_unknown,
        }
        try:
            tmp = Path(stack.enter_context(TemporaryDirectory())) / "input.safetensors"
            self.r["sha256"] = snapshot(path, tmp)
            header, payload_start = check_header(tmp)
            self.handle = stack.enter_context(safe_open(tmp, framework="numpy"))
            metadata = self.handle.metadata() or {}
            if metadata not in ({}, {"format": "pt"}):
                self.global_unknown.append("unsupported header metadata")
            self.r["metadata_fields"] = sorted(metadata)
            sidecars = sorted(
                {
                    name
                    for name in (
                        "adapter_config.json",
                        "config.json",
                        "model_index.json",
                        path.with_suffix(".json").name,
                        path.name + ".json",
                    )
                    if (path.parent / name).exists()
                }
            )
            self.r["sidecars"] = sidecars
            if sidecars:
                self.global_unknown.append("configuration sidecar present")
            with tmp.open("rb") as stream:
                self.inventory(profile, header, stream, payload_start)
            self.complete = True
        except (OSError, SafetensorError, ValueError, RecursionError, OverflowError) as exc:
            self.r["errors"].append(str(exc))

    def inventory(self, profile, header, stream, payload_start):
        keys = sorted(self.handle.keys())
        self.r["raw_keys"] = len(keys)
        count = 0
        for key in keys:
            tensor = self.handle.get_slice(key)
            shape = tensor.get_shape()
            count += math.prod(shape)
            self.r["keys"][key] = {"shape": shape, "dtype": tensor.get_dtype()}
        self.r["parameters"] = count
        if count > MAX_ELEMENTS or len(keys) > MAX_KEYS:
            raise ValueError("tensor inventory exceeds resource limit")
        for key in keys:
            start, end = header[key]["data_offsets"]
            if not finite_payload(
                stream, payload_start + start, end - start, self.r["keys"][key]["dtype"]
            ):
                self.r["errors"].append(f"{key}: nonfinite value")
                if key in INDEX[self.side]:
                    self.invalid.add(INDEX[self.side][key][0])
            self.classify(key)
        if self.side == "native":
            self.global_unknown.extend(native_configuration_issues(keys))
        self.validate_groups(profile)

    def classify(self, key):
        if key in INDEX[self.side]:
            p, role = INDEX[self.side][key]
            self.groups.setdefault(p, {})[role] = key
            self.r["matched_keys"].append(key)
        else:
            self.r["unsupported_keys"].append(key)
            alias = (
                key.replace(".processor.", ".")
                .replace(".to_out_lora.", ".to_out.0.")
                .replace("_lora.", ".")
            )
            candidates = set(FUSED[self.side].get(key.partition(".lora_")[0], ()))
            for p in PROJECTIONS:
                prefix = getattr(p, self.side) + "."
                if key.startswith((prefix, getattr(p, self.side) + "_slice")) or (
                    self.side == "native" and alias.startswith(prefix)
                ):
                    candidates.add(p)
            if candidates:
                self.unknown.update(candidates)
                self.r["ambiguous_keys"].append(key)
            other_side = "kohya" if self.side == "native" else "native"
            native_blocks = (
                "lora_unet_down_blocks_",
                "lora_unet_mid_block_",
                "lora_unet_up_blocks_",
            )
            mixed = key in INDEX[other_side] or (self.side == "kohya" and key.startswith(native_blocks))
            if mixed and "mixed or unexpected block dialect" not in self.global_unknown:
                self.global_unknown.append("mixed or unexpected block dialect")

    def validate_groups(self, profile):
        ranks, alphas = set(), set()
        for p, roles in self.groups.items():
            if {"A", "B"} <= roles.keys():
                self.r["paired_projections"].append(p.native)
            else:
                self.invalid.add(p)
                self.r["unpaired_keys"].extend(roles.values())
                for suffix, role in SUFFIXES[self.side].items():
                    if role in {"A", "B"} and role not in roles:
                        self.r["missing_keys"].append(getattr(p, self.side) + "." + suffix)
                if p not in self.unknown:
                    self.r["errors"].append(f"{p.native}: missing factor")
            rank = {}
            for role, key in roles.items():
                desc = self.r["keys"][key]
                shape, dtype = desc["shape"], desc["dtype"]
                if role == "alpha":
                    valid = shape == []
                else:
                    valid = len(shape) == 2 and (
                        shape[1] == p.inputs if role == "A" else shape[0] == p.outputs
                    )
                    if valid:
                        rank[role] = shape[0 if role == "A" else 1]
                        valid = 1 <= rank[role] <= MAX_RANK
                if not valid:
                    self.invalid.add(p)
                    self.r["errors"].append(f"{key}: invalid dimensions or rank limit")
                if dtype not in {"F16", "F32"}:
                    self.unknown.add(p)
                    self.r["unsupported_keys"].append(key)
                    continue
                if dtype == "F16" and profile != "fp16":
                    self.unknown.add(p)
                    self.r["unsupported_keys"].append(key)
                if role == "alpha" and valid and p not in self.invalid:
                    alphas.add(float(self.handle.get_tensor(key)))
            if len(set(rank.values())) > 1:
                self.invalid.add(p)
                self.r["errors"].append(f"{p.native}: inconsistent factor ranks")
            elif p not in self.invalid:
                ranks.update(rank.values())
            if "alpha" not in roles:
                alphas.add(None)
        if len(ranks) > 1:
            self.global_unknown.append("heterogeneous ranks")
        if len(alphas) > 1 or any(a is not None and a <= 0 for a in alphas):
            self.global_unknown.append("mixed, partial or nonpositive alpha")

    def factors(self, p):
        roles = self.groups[p]
        a, b = (self.handle.get_tensor(roles[r]).astype(np.float64) for r in ("A", "B"))
        scale = float(self.handle.get_tensor(roles["alpha"])) / len(a) if "alpha" in roles else 1.0
        return a, b, scale
