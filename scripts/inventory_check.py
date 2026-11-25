"""Validate every manifest identity in two disjoint adapter loads per stack."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from diffusers import UNet2DConditionModel
from diffusers.loaders import StableDiffusionXLLoraLoaderMixin as Loader
from library.sdxl_original_unet import SdxlUNet2DConditionModel
from networks.lora import create_network_from_weights
from peft import get_peft_model_state_dict
from safetensors.torch import save_file

from lora_witness.compare import compare
from lora_witness.manifest import MANIFEST_HASH, PROJECTIONS


def inventory(model):
    found = {}
    zero = torch.tensor(0.0)
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear) or not any(f".attn{i}." in name for i in (1, 2)):
            continue
        if not name.endswith((".to_q", ".to_k", ".to_v", ".to_out.0")):
            continue
        # Real CPU Linear.forward; expanded zero base parameters share one scalar.
        mod.weight = torch.nn.Parameter(zero.expand(mod.weight.shape), requires_grad=False)
        if mod.bias is not None:
            mod.bias = torch.nn.Parameter(zero.expand(mod.bias.shape), requires_grad=False)
        found[name] = tuple(mod.weight.shape)
    assert len(found) == 560
    return found


def inspect_outputs(model, names, expected):
    for name, (_, inputs) in names.items():
        mod = model.get_submodule(name)
        x = torch.zeros(1, inputs)
        x[0, 0] = 1.0
        seen = []
        with mod.register_forward_hook(lambda m, args, output, seen=seen: seen.append(output.detach())):
            mod(x)
        (y,) = seen
        assert y[0, 0].item() == expected.get(name, 0) and y[:, 1:].count_nonzero() == 0, name


@torch.no_grad()
def verify_all(config_path, output):
    config = json.loads(config_path.read_text())
    cohorts = []
    for parity in range(2):
        with TemporaryDirectory() as scratch:
            d = Path(scratch)
            with torch.device("meta"):
                source = SdxlUNet2DConditionModel()
                remap = UNet2DConditionModel.from_config(config)
                target = UNet2DConditionModel.from_config(config)
            src = inventory(source)
            dst = inventory(remap)
            assert inventory(target) == dst
            source_names = {"lora_unet_" + name.replace(".", "_"): name for name in src}
            assert {p.kohya for p in PROJECTIONS} == set(source_names)
            assert {p.native for p in PROJECTIONS} == {"unet." + n for n in dst}
            expected = {}
            weights = {}
            for i, (name, (outputs, inputs)) in enumerate(sorted(src.items())):
                if i % 2 != parity:
                    continue
                stem = "lora_unet_" + name.replace(".", "_")
                expected[name] = i + 1
                a, b = torch.zeros(1, inputs), torch.zeros(outputs, 1)
                a[0, 0], b[0, 0] = 1, i + 1
                weights[stem + ".lora_down.weight"] = a
                weights[stem + ".lora_up.weight"] = b
                weights[stem + ".alpha"] = torch.tensor(1.0)
            s, t = d / "source.safetensors", d / "target.safetensors"
            save_file(weights, s)
            net, state = create_network_from_weights(1.0, str(s), None, [], source, for_inference=True)
            net.apply_to([], source, False, True)
            net.load_state_dict(state, strict=True)
            assert len(net.unet_loras) == len(expected)
            inspect_outputs(source.eval(), src, expected)
            expected_native = {}
            for p in PROJECTIONS:
                native_name = p.native.removeprefix("unet.")
                source_name = source_names[p.kohya]
                assert src[source_name] == dst[native_name] == (p.outputs, p.inputs)
                if source_name in expected:
                    expected_native[native_name] = expected[source_name]
            # The converter receives only the original SGM file and upstream config.
            for model, file in ((remap, s), (target, t)):
                state, alphas, meta = Loader.lora_state_dict(
                    str(d), weight_name=file.name, unet_config=model.config, return_lora_metadata=True
                )
                Loader.load_lora_into_unet(
                    state, alphas, model, adapter_name="inventory", metadata=meta
                )
                actual = {n for n, m in model.named_modules() if hasattr(m, "lora_A")}
                assert actual == expected_native.keys()
                if model is remap:
                    layers = get_peft_model_state_dict(model, adapter_name="inventory")
                    Loader.save_lora_weights(d, unet_lora_layers=layers, weight_name=t.name)
            inspect_outputs(target.eval(), dst, expected_native)
            result = compare(s, t, declared=True)
            assert result["verdict"] == "sampled_agreement" and len(result["layers"]) == 280
            cohorts.append(
                {
                    "parity": parity,
                    "source_sha256": result["files"]["kohya"]["sha256"],
                    "native_sha256": result["files"]["native"]["sha256"],
                    "active_per_stack": len(expected),
                    "untouched_per_stack": len(src) - len(expected),
                    "max_sampled_error": max(l["E"] for l in result["layers"].values()),
                    "signature_map_sha256": hashlib.sha256(
                        json.dumps(expected_native, sort_keys=True).encode()
                    ).hexdigest(),
                }
            )
    data = {
        "manifest_sha256": MANIFEST_HASH,
        "paths_tested": len(PROJECTIONS),
        "probe": "first input basis vector; expected first output = sorted source index + 1, or zero",
        "cohorts": cohorts,
    }
    (output / "inventory-results.json").write_text(json.dumps(data, indent=2) + "\n")
