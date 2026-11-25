import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from lora_witness.compare import compare, unique_swaps
from lora_witness.manifest import FIXTURE_PROJECTIONS, INDEX, MANIFEST_HASH, PROJECTIONS
from lora_witness.numeric import probes

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
S = EXAMPLES / "synthetic/kohya.safetensors"
T = EXAMPLES / "synthetic/native.safetensors"


def test_upstream_export_and_verified_manifest():
    target = EXAMPLES / "upstream/converted.safetensors"
    r = compare(S, target, declared=True, mode="exhaustive")
    assert r["verdict"] == "exhaustive_agreement_within_tolerance"
    assert len(PROJECTIONS) == 560 and len(INDEX["kohya"]) == 1680 and len(INDEX["native"]) == 1120
    evidence = EXAMPLES.parent / "docs/evidence"
    inventory = json.loads((evidence / "inventory-results.json").read_text())
    results = json.loads((evidence / "loader-results.json").read_text())
    assert inventory["manifest_sha256"] == MANIFEST_HASH
    assert results["file_sha256"]["converted"] == r["files"]["native"]["sha256"]


def test_unrelated_architecture(tmp_path):
    target = tmp_path / "foreign.safetensors"
    save_file({"transformer.fused_qkv.lora_B.weight": np.ones((1, 1))}, target)
    r = compare(S, target, declared=True)
    assert r["verdict"] == "inconclusive"
    assert len(r["files"]["kohya"]["unassessed_keys"]) == 12


def test_heterogeneous_rank(tmp_path):
    native = load_file(T)
    p = FIXTURE_PROJECTIONS[1]
    for role, widths in (("A", ((0, 1), (0, 0))), ("B", ((0, 0), (0, 1)))):
        k = f"{p.native}.lora_{role}.weight"
        native[k] = np.pad(native[k], widths)
    t = tmp_path / "mixed-rank.safetensors"
    save_file(native, t)
    r = compare(S, t, declared=True)
    assert (
        r["verdict"] == "inconclusive"
        and "heterogeneous ranks" in r["files"]["native"]["configuration_issues"]
    )


def test_bfloat16_nonfinite_rejected_on_unsupported_key(tmp_path):
    name = "unet.conv.bias"
    header = json.dumps({name: {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}).encode()
    header += b" " * (-len(header) % 8)
    target = tmp_path / "bf16.safetensors"
    target.write_bytes(len(header).to_bytes(8, "little") + header + bytes.fromhex("807f"))
    r = compare(S, target, declared=True)
    assert "invalid_input" in r["reasons"]
    assert r["files"]["native"]["errors"] == [name + ": nonfinite value"]


def test_alias_in_place_of_one_factor(tmp_path):
    p = FIXTURE_PROJECTIONS[0]
    data = load_file(T)
    data[p.native + ".lora_A.default.weight"] = data.pop(p.native + ".lora_A.weight")
    target = tmp_path / "alias.safetensors"
    save_file(data, target)
    r = compare(S, target, declared=True)
    assert r["verdict"] == "inconclusive" and len(r["layers"]) == 3


def test_empty_pair(tmp_path):
    t = tmp_path / "empty.safetensors"
    save_file({}, t)
    r = compare(t, t, declared=True)
    assert r["verdict"] == "inconclusive" and r["reasons"] == ["empty_comparison"]


@pytest.mark.parametrize(
    "args", [{"rows": 0}, {"rows": 257}, {"atol": float("nan")}, {"rtol": float("inf")}, {"rtol": -1}]
)
def test_invalid_settings(args):
    with pytest.raises(ValueError):
        compare(S, T, declared=True, **args)


def test_probe_hash():
    _, a = probes("a", 3, 11)
    assert a == probes("a", 3, 11)[1]
    assert a["sha256"] != probes("b", 3, 11)[1]["sha256"]
    assert a["sha256"] != probes("a", 3, 12)[1]["sha256"]


def test_duplicate_source_effect_makes_swap_ambiguous():
    layers = {
        k: {"shape": [2, 2], "effect_sha256": hashes, "status": "numerical_mismatch"}
        for k, hashes in zip("qkv", [("A", "B"), ("B", "A"), ("A", "C")])
    }
    assert unique_swaps(layers) == []


def test_rounded_dense_fp16_export(tmp_path):
    p = FIXTURE_PROJECTIONS[0]
    rng = np.random.default_rng(2026)
    a = rng.standard_normal((2, p.inputs)).astype("float32")
    b = rng.standard_normal((p.outputs, 2)).astype("float32")
    source, target = tmp_path / "source.safetensors", tmp_path / "target.safetensors"
    save_file(
        {
            p.kohya + ".lora_down.weight": a,
            p.kohya + ".lora_up.weight": b,
            p.kohya + ".alpha": np.array(1.0, dtype="float32"),
        },
        source,
    )
    save_file(
        {
            p.native + ".lora_A.weight": a.astype("float16"),
            p.native + ".lora_B.weight": (b / 2).astype("float16"),
        },
        target,
    )
    result = compare(source, target, declared=True, profile="fp16", mode="exhaustive")
    layer = result["layers"][p.native]
    assert 0 < layer["E"] < layer["threshold"]
    assert result["verdict"] == "exhaustive_agreement_within_tolerance"
    strict = compare(source, target, declared=True, profile="fp16", mode="exhaustive", atol=0, rtol=0)
    assert strict["reasons"] == ["numerical_mismatch"]


def test_fused_replacement_is_unassessed(tmp_path):
    p = FIXTURE_PROJECTIONS[1]
    data = load_file(T)
    for suffix in (".lora_A.weight", ".lora_B.weight"):
        data[p.native.removesuffix(".to_q") + ".to_qkv" + suffix] = data.pop(p.native + suffix)
    t = tmp_path / "fused.safetensors"
    save_file(data, t)
    r = compare(S, t, declared=True)
    assert r["verdict"] == "inconclusive" and len(r["files"]["native"]["ambiguous_keys"]) == 2
    assert len(r["layers"]) == 2
