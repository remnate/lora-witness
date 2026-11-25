"""Full-extent SDXL Base UNet projections with loader evidence."""

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Projection:
    kohya: str
    native: str
    outputs: int
    inputs: int


def entry(source, dest, suffix, outputs, inputs):
    return Projection(
        "lora_unet_" + source + "_" + suffix.replace(".", "_"),
        "unet." + dest + "." + suffix,
        outputs,
        inputs,
    )


FIXTURE_PROJECTIONS = (
    entry(
        "input_blocks_4_1", "down_blocks.1.attentions.0", "transformer_blocks.0.attn2.to_k", 640, 2048
    ),
    entry("middle_block_1", "mid_block.attentions.0", "transformer_blocks.9.attn1.to_q", 1280, 1280),
    entry("middle_block_1", "mid_block.attentions.0", "transformer_blocks.9.attn1.to_v", 1280, 1280),
    entry(
        "output_blocks_1_1",
        "up_blocks.0.attentions.1",
        "transformer_blocks.3.attn2.to_out.0",
        1280,
        1280,
    ),
)


def block_entries(kohya, native, width, depth):
    for i in range(depth):
        for attention in ("attn1", "attn2"):
            for proj in ("to_q", "to_k", "to_v", "to_out.0"):
                inputs = 2048 if attention == "attn2" and proj in {"to_k", "to_v"} else width
                yield entry(kohya, native, f"transformer_blocks.{i}.{attention}.{proj}", width, inputs)


def base_inventory():
    for i in (4, 5, 7, 8):
        block, attention = divmod(i - 1, 3)
        width, depth = (640, 2) if block == 1 else (1280, 10)
        yield from block_entries(
            f"input_blocks_{i}_1", f"down_blocks.{block}.attentions.{attention}", width, depth
        )
    yield from block_entries("middle_block_1", "mid_block.attentions.0", 1280, 10)
    for i in range(6):
        block, attention = divmod(i, 3)
        width, depth = (1280, 10) if block == 0 else (640, 2)
        yield from block_entries(
            f"output_blocks_{i}_1", f"up_blocks.{block}.attentions.{attention}", width, depth
        )


PROJECTIONS = tuple(base_inventory())
MANIFEST = [asdict(p) for p in PROJECTIONS]
MANIFEST_HASH = hashlib.sha256(json.dumps(MANIFEST, sort_keys=True).encode()).hexdigest()
SUFFIXES = {
    "kohya": {"lora_down.weight": "A", "lora_up.weight": "B", "alpha": "alpha"},
    "native": {"lora_A.weight": "A", "lora_B.weight": "B"},
}
INDEX = {
    side: {
        getattr(p, side) + "." + suffix: (p, role)
        for p in PROJECTIONS
        for suffix, role in suffixes.items()
    }
    for side, suffixes in SUFFIXES.items()
}

LOADER_EVIDENCE = {
    "sd_scripts": "88ff6b5391e89822d0463df8769591273d0d28b1",
    "diffusers": "0.43.0",
    "peft": "0.22.0",
    "sdxl_config": "462165984030d82259a11f4367a4eed129e94a7b",
    "tested_projection_count": 560,
}

# These spellings can obscure a supported identity. They are never decoded.
FUSED = {"native": {}, "kohya": {}}
for side, separator in (("native", "."), ("kohya", "_")):
    for proj in PROJECTIONS:
        name = getattr(proj, side)
        for kind in ("q", "k", "v"):
            suffix = separator + "to_" + kind
            if name.endswith(suffix):
                family = ("to_qkv", "qkv", "fused_qkv", "in_proj")
                if kind in ("k", "v"):
                    family += ("to_kv",)
                for spelling in family:
                    stem = name.removesuffix(suffix) + separator + spelling
                    FUSED[side].setdefault(stem, set()).add(proj)
