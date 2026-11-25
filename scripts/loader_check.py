"""Run the real CPU inference loaders. Invoke with scripts/check-loaders.sh."""

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import torch
from diffusers import UNet2DConditionModel
from diffusers.loaders import StableDiffusionXLLoraLoaderMixin as Loader
from library.sdxl_original_unet import SdxlUNet2DConditionModel
from networks.lora import create_network_from_weights
from peft import get_peft_model_state_dict
from safetensors.torch import load_file, save_file

# Independent attachment expectations, including nearby untouched modules.
PAIRS = [
    (
        "input_blocks.4.1.transformer_blocks.0.attn2.to_k",
        "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k",
    ),
    (
        "middle_block.1.transformer_blocks.9.attn1.to_q",
        "mid_block.attentions.0.transformer_blocks.9.attn1.to_q",
    ),
    (
        "middle_block.1.transformer_blocks.9.attn1.to_v",
        "mid_block.attentions.0.transformer_blocks.9.attn1.to_v",
    ),
    (
        "output_blocks.1.1.transformer_blocks.3.attn2.to_out.0",
        "up_blocks.0.attentions.1.transformer_blocks.3.attn2.to_out.0",
    ),
]
SENTINELS = [
    (
        "input_blocks.4.1.transformer_blocks.0.attn2.to_v",
        "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_v",
    ),
    (
        "middle_block.1.transformer_blocks.9.attn1.to_k",
        "mid_block.attentions.0.transformer_blocks.9.attn1.to_k",
    ),
    (
        "output_blocks.1.1.transformer_blocks.3.attn2.to_q",
        "up_blocks.0.attentions.1.transformer_blocks.3.attn2.to_q",
    ),
]
REV = "88ff6b5391e89822d0463df8769591273d0d28b1"
CONFIG_SHA = "30ebc70750223e59006f7f2b4e1e6c102570aa19a9c4ae3e1fbe7591332dbae6"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(model, side):
    entries = {
        name: list(mod.weight.shape)
        for name, mod in model.named_modules()
        if isinstance(mod, torch.nn.Linear)
        and name.endswith((".to_q", ".to_k", ".to_v", ".to_out.0"))
        and (".attn1." in name or ".attn2." in name)
    }
    assert len(entries) == 560
    for pair in PAIRS + SENTINELS:
        layer = model.get_submodule(pair[side])
        layer.to_empty(device="cpu")
        for param in layer.parameters():
            param.zero_()
    return {
        "count": len(entries),
        "sha256": hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest(),
    }


def inject_native(model, file):
    state, alphas, metadata = Loader.lora_state_dict(
        str(file.parent), weight_name=file.name, unet_config=model.config, return_lora_metadata=True
    )
    Loader.load_lora_into_unet(state, alphas, model, adapter_name="test", metadata=metadata)
    return model.eval()


def attach_native(model, file):
    model = inject_native(model, file)
    attached = {n for n, m in model.named_modules() if hasattr(m, "lora_A")}
    assert attached == {b for a, b in PAIRS}, attached
    return model


def attach_kohya(model, file, absent=False):
    network, state = create_network_from_weights(1.0, str(file), None, [], model, for_inference=True)
    network.apply_to([], model, False, True)
    result = network.load_state_dict(state, strict=not absent)
    if absent:
        assert set(result.missing_keys) == {
            "lora_unet_" + a.replace(".", "_") + ".alpha" for a, b in PAIRS
        }
        assert not result.unexpected_keys
    assert {m.lora_name for m in network.unet_loras} == {
        "lora_unet_" + a.replace(".", "_") for a, b in PAIRS
    }
    network.eval()
    return model.eval()


def run_modules(model, side):
    outputs = []
    for i, pair in enumerate(PAIRS + SENTINELS):
        module = model.get_submodule(pair[side])
        rng = torch.Generator().manual_seed(84 + i)
        inputs = torch.randn(32, module.in_features, generator=rng)
        seen = []
        with module.register_forward_hook(lambda _m, _args, out, seen=seen: seen.append(out.detach())):
            module(inputs)
        (output,) = seen
        if i >= len(PAIRS):
            assert torch.count_nonzero(output) == 0
        outputs.append((inputs, output))
    return outputs


def check_other_native_settings(base, fixtures, output):
    weights = load_file(fixtures / "half.safetensors")
    name = PAIRS[2][1]
    bias_file = output / "mixed-bias.safetensors"
    save_file({**weights, f"unet.{name}.lora_B.bias": torch.zeros(1280)}, bias_file)
    biased = attach_native(base(1), bias_file)
    assert biased.peft_config["test"].lora_bias
    bias_layers = [dst for _, dst in PAIRS if biased.get_submodule(dst).lora_B["test"].bias is not None]
    assert len(bias_layers) == 4

    lokr_file = output / "mixed-lokr.safetensors"
    save_file(
        {**weights, f"unet.{name}.lokr_w1": torch.eye(32), f"unet.{name}.lokr_w2": torch.eye(40)},
        lokr_file,
    )
    lokr = inject_native(base(1), lokr_file)
    assert lokr.peft_config["test"].peft_type.value == "LOKR"
    active = {n for n, m in lokr.named_modules() if hasattr(m, "lokr_w1")}
    assert active == {name}
    assert not any(hasattr(m, "lora_A") for m in lokr.modules())
    for _, dst in PAIRS:
        layer = lokr.get_submodule(dst)
        linear = layer.get_base_layer() if dst == name else layer
        x = torch.zeros(1, linear.in_features)
        x[0, 0] = 1
        y = layer(x)
        if dst == name:
            assert torch.equal(x, y)
        else:
            assert torch.count_nonzero(y) == 0
    return {
        "bias": {"sha256": sha(bias_file), "modules_with_B_bias": bias_layers},
        "lokr": {
            "sha256": sha(lokr_file),
            "active_modules": sorted(active),
            "ordinary_lora_modules": 0,
        },
    }


def check_mixed_dora(base, fixtures, output):
    source = load_file(fixtures / "kohya.safetensors")
    for key, tensor in source.items():
        if key.endswith(".lora_up.weight"):
            tensor.zero_()
    zero_file = output / "zero-source.safetensors"
    save_file(source, zero_file)
    zero_source = attach_kohya(base(0), zero_file)
    ordinary_source = attach_kohya(base(0), fixtures / "kohya.safetensors")
    evidence = {
        "source_sha256": sha(zero_file),
        "base": "unit first coefficient in each native base row",
    }
    for case in ("collinear", "half"):
        weights = load_file(
            fixtures / ("native.safetensors" if case == "collinear" else "half.safetensors")
        )
        if case == "collinear":
            for tensor in weights.values():
                tensor.zero_()
            for name in ("A", "B"):
                weights[f"unet.{PAIRS[1][1]}.lora_{name}.weight"][0, 0] = 1
        target = base(1)
        for _, name in PAIRS:
            target.get_submodule(name).weight[:, 0] = 1
        magnitude_module = PAIRS[2][1]
        weights[f"unet.{magnitude_module}.lora_magnitude_vector"] = torch.ones(
            target.get_submodule(magnitude_module).out_features
        )
        target_file = output / f"{case}-dora.safetensors"
        save_file(weights, target_file)
        target = attach_native(target, target_file)
        errors = {}
        for src, dst in PAIRS:
            layer = target.get_submodule(dst)
            assert layer.use_dora["test"]
            if case == "collinear":
                probes = torch.eye(layer.in_features)
                reference = zero_source.get_submodule(src)(probes)
                assert torch.count_nonzero(reference) == 0
            else:
                probes = torch.zeros(1, layer.in_features)
                probes[0, 0] = 1
                reference = ordinary_source.get_submodule(src)(probes)
            delta = layer(probes) - layer.base_layer(probes)
            errors[dst] = (delta - reference).abs().max().item()
        if case == "collinear":
            assert max(errors.values()) == 0
        else:
            assert errors[PAIRS[0][1]] > 1 and errors[PAIRS[3][1]] > 4
        evidence[case] = {
            "target_sha256": sha(target_file),
            "dora_enabled_modules": len(errors),
            "inputs": "complete basis" if case == "collinear" else "first basis vector",
            "max_error_against_kohya": errors,
        }
    return evidence


@torch.no_grad()
def check(config, fixtures, output):
    torch.set_num_threads(1)
    assert sha(config) == CONFIG_SHA
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "sd_scripts": REV,
        "sdxl_config_revision": "462165984030d82259a11f4367a4eed129e94a7b",
        "sdxl_config_sha256": CONFIG_SHA,
        "python": platform.python_version(),
        "libraries": {
            p: version(p)
            for p in (
                "torch",
                "torchvision",
                "diffusers",
                "transformers",
                "peft",
                "accelerate",
                "safetensors",
            )
        },
        "sentinels": SENTINELS,
        "mode": "32 seeded FP32 vectors per active and inactive module",
        "inventories": {},
        "cases": {},
        "file_sha256": {},
    }

    def base(side):
        with torch.device("meta"):
            model = (
                SdxlUNet2DConditionModel()
                if side == 0
                else UNet2DConditionModel.from_config(json.loads(config.read_text()))
            )
        report["inventories"][("kohya", "diffusers")[side]] = prepare(model, side)
        return model

    source = fixtures / "kohya.safetensors"
    report["file_sha256"]["kohya"] = sha(source)
    reference = run_modules(attach_kohya(base(0), source), 0)
    absent = output / "absent.safetensors"
    save_file({k: v for k, v in load_file(source).items() if not k.endswith(".alpha")}, absent)
    unscaled = run_modules(attach_kohya(base(0), absent, absent=True), 0)
    assert all(torch.equal(y * 2, z) for (_, y), (_, z) in zip(reference, unscaled))
    report["absent_alpha_doubles_source"] = True
    report["file_sha256"]["absent"] = sha(absent)
    native = load_file(fixtures / "native.safetensors")
    square = "unet." + PAIRS[1][1]
    a_key, b_key = square + ".lora_A.weight", square + ".lora_B.weight"
    transposed = {**native, a_key: native[b_key].T.contiguous(), b_key: native[a_key].T.contiguous()}
    trans_file = output / "transposed.safetensors"
    save_file(transposed, trans_file)
    trans_outputs = run_modules(attach_native(base(1), trans_file), 1)
    x, y = trans_outputs[1]
    corner = torch.tensor([[2.0, 4.0, -2.0], [-1.0, 0.0, 7.0]])
    torch.testing.assert_close(y[:, :3], x[:, :2] @ corner)
    assert torch.count_nonzero(y[:, 3:]) == 0 and not torch.equal(y, reference[1][1])
    assert all(
        torch.equal(y, z) for i, ((_, y), (_, z)) in enumerate(zip(reference, trans_outputs)) if i != 1
    )
    report["square_transpose_loaded"] = True
    report["file_sha256"]["transposed"] = sha(trans_file)
    padded = {
        k: torch.nn.functional.pad(t, (0, 0, 0, 1) if ".lora_A." in k else (0, 1))
        for k, t in native.items()
    }
    padded_file = output / "rank3.safetensors"
    save_file(padded, padded_file)
    padded_outputs = run_modules(attach_native(base(1), padded_file), 1)
    assert all(torch.equal(y, z) for (_, y), (_, z) in zip(reference, padded_outputs))
    report["padded_rank_three_native_matches_rank_two_source"] = True
    report["file_sha256"]["rank3"] = sha(padded_file)
    converted = output / "converted.safetensors"
    loaded = attach_native(base(1), source)
    weights = get_peft_model_state_dict(loaded, adapter_name="test")
    # Export unit-scale, folding PEFT's *actual* attached scale into each B.
    for key in list(weights):
        if key.endswith(".lora_B.weight"):
            module = loaded.get_submodule(key.removesuffix(".lora_B.weight"))
            weights[key] = weights[key] * module.scaling["test"]
    Loader.save_lora_weights(output, unet_lora_layers=weights, weight_name=converted.name)
    for case, file in (
        ("sgm_remap", source),
        ("native", fixtures / "native.safetensors"),
        ("half", fixtures / "half.safetensors"),
        ("mixed", fixtures / "mixed.safetensors"),
        ("converted", converted),
    ):
        model = loaded if case == "sgm_remap" else attach_native(base(1), file)
        target = run_modules(model, 1)
        results = {}
        for i, ((src, dst), (x, y), (_, z)) in enumerate(zip(PAIRS, reference, target), 1):
            expected = torch.zeros_like(y, dtype=torch.float64)
            corner = torch.tensor([[2, 4, -2], [-1, 0, 7]], dtype=torch.float64) * (i / 2)
            expected[:, :2] = x[:, :3].double() @ corner.T
            multiplier = 0.5 if case in ("half", "mixed") and i == 2 else 1
            assert torch.equal(z, y * multiplier)
            diff = (expected - y).abs().max().item()
            assert diff < 1e-5
            results[dst] = {
                "kohya_module": src,
                "max_loader_difference": (z - y).abs().max().item(),
                "max_source_error_vs_fp64": diff,
                "target_multiplier": multiplier,
            }
        report["cases"][case] = results
        report["file_sha256"][case] = sha(file)
    report["other_configuration_controls"] = check_other_native_settings(base, fixtures, output)
    report["dora_controls"] = check_mixed_dora(base, fixtures, output)
    (output / "loader-results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "fixtures", "output"):
        p.add_argument(name, type=Path)
    args = p.parse_args()
    check(**vars(args))
    from inventory_check import verify_all

    verify_all(args.config, args.output)
