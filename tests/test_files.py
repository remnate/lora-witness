import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from lora_witness.cli import main
from lora_witness.compare import compare
from lora_witness.fixture import generate
from lora_witness.manifest import FIXTURE_PROJECTIONS as PROJECTIONS

FIX = Path(__file__).resolve().parents[1] / "examples/synthetic"
S, N = (FIX / f"{name}.safetensors" for name in ("kohya", "native"))
Q, V = [p.native for p in PROJECTIONS[1:3]]


def run(s=S, t=N, **kwargs):
    return compare(s, t, declared=True, **kwargs)


def mutate(tmp_path, fun, original=N, metadata=None):
    data = load_file(original)
    fun(data)
    file = tmp_path / "test.safetensors"
    save_file(data, file, metadata=metadata)
    return file


@pytest.mark.parametrize(
    "name,mode,status",
    [
        ("native", "exhaustive", "exhaustive_agreement_within_tolerance"),
        ("native", "sampled", "sampled_agreement"),
        ("half", "exhaustive", "fail"),
        ("mixed", "exhaustive", "fail"),
    ],
)
def test_release_pairs(name, mode, status, capsys):
    target = FIX / f"{name}.safetensors"
    code = main(["compare", str(S), str(target), "--native-unit-scale", "--json", "--mode", mode])
    report = json.loads(capsys.readouterr().out)
    main(["schema"])
    jsonschema.validate(report, json.loads(capsys.readouterr().out))
    assert report["verdict"] == status and code == int(status == "fail")
    assert report["files"]["kohya"]["coverage"]["compared_keys"] == 12
    if code:
        assert report["layers"][Q]["E"] == 3.5 and report["layers"][Q]["M"] == 7
    if name == "mixed":
        assert report["files"]["native"]["unsupported_keys"] == ["text_encoder.extra.bias"]


def test_fixture_roundtrip(tmp_path):
    generate(tmp_path)
    for f in FIX.glob("*.safetensors"):
        assert (tmp_path / f.name).read_bytes() == f.read_bytes()
    with pytest.raises(ValueError):
        generate(tmp_path)


@pytest.mark.parametrize("side", ["source", "target"])
def test_missing_extra(tmp_path, side):
    p = PROJECTIONS[1]
    root = p.kohya if side == "source" else p.native

    def drop(data):
        for k in list(data):
            if k.startswith(root + "."):
                del data[k]

    file = mutate(tmp_path, drop, S if side == "source" else N)
    r = run(file, N) if side == "source" else run(S, file)
    assert r["reasons"] == ["projection_set_mismatch"]
    assert len(r["layers"]) == 3
    assert any(f["missing_keys"] for f in r["files"].values())
    assert any(f["extra_keys"] for f in r["files"].values())


def test_swap(tmp_path):
    def swap(data):
        a, b = [k + ".lora_B.weight" for k in (Q, V)]
        data[a], data[b] = data[b], data[a]

    report = run(t=mutate(tmp_path, swap), mode="exhaustive")
    assert report["likely_swaps"] == [[Q, V]]
    assert {p for p, l in report["layers"].items() if l["status"] == "numerical_mismatch"} == {Q, V}


def test_square_transpose(tmp_path):
    def transpose_effect(data):
        a, b = [Q + ".lora_" + role + ".weight" for role in "AB"]
        data[a], data[b] = data[b].T.copy(), data[a].T.copy()

    assert run(t=mutate(tmp_path, transpose_effect))["layers"][Q]["status"] == "numerical_mismatch"


def test_rank_padding(tmp_path):
    def pad(data):
        for key, t in data.items():
            padding = ((0, 1), (0, 0)) if ".lora_A." in key else ((0, 0), (0, 1))
            data[key] = np.pad(t, padding)

    assert run(t=mutate(tmp_path, pad))["verdict"] == "sampled_agreement"


def test_partial_unsupported_and_ambiguous(tmp_path):
    f = mutate(tmp_path, lambda d: d.update({Q + ".lora_A.default.weight": np.ones(1)}))
    r = run(t=f)
    assert r["verdict"] == "inconclusive" and len(r["layers"]) == 3
    assert r["files"]["native"]["ambiguous_keys"]
    f = mutate(tmp_path, lambda d: d.update({"unet.conv.bias": np.ones(1)}))
    assert run(t=f)["verdict"] == "inconclusive"


@pytest.mark.parametrize("change", ["delete_B", "rank", "dimension", "nan", "inf"])
def test_malformed_preserves_other_layers(tmp_path, change):
    def modify(data):
        key = Q + ".lora_B.weight"
        if change == "delete_B":
            del data[key]
        elif change == "rank":
            data[key] = data[key][:, :1].copy()
        elif change == "dimension":
            data[key] = data[key][:1].copy()
        else:
            data[key][0, 0] = float(change)

    r = run(t=mutate(tmp_path, modify))
    assert "invalid_input" in r["reasons"] and len(r["layers"]) == 3
    if change == "delete_B":
        assert len(r["files"]["native"]["unpaired_keys"]) == 1


@pytest.mark.parametrize("change", ["absent", "zero", "partial", "heterogeneous", "mixed"])
def test_scaling_profiles(tmp_path, change):
    def modify(data):
        key = PROJECTIONS[0].kohya + ".alpha"
        if change == "absent":
            for k in list(data):
                if k.endswith(".alpha"):
                    del data[k]
                elif k.endswith(".lora_up.weight"):
                    data[k] /= 2
        elif change == "partial":
            del data[key]
        elif change == "mixed":
            data["lora_unet_down_blocks_1_extra.lora_down.weight"] = np.ones(1)
        else:
            data[key] = np.array(0 if change == "zero" else 2, dtype="float32")

    assert run(s=mutate(tmp_path, modify, S))["verdict"] == (
        "sampled_agreement" if change == "absent" else "inconclusive"
    )


def test_report_refuses_unknown_scale(tmp_path):
    assert compare(S, N)["verdict"] == "inconclusive"
    t = mutate(tmp_path, lambda d: None, metadata={"scale": "2"})
    assert run(t=t)["verdict"] == "inconclusive"
    (tmp_path / "adapter_config.json").write_text("{}")
    assert run(t=mutate(tmp_path, lambda d: None))["verdict"] == "inconclusive"


def test_over_limit(tmp_path):
    f = tmp_path / "large.safetensors"
    with f.open("wb") as stream:
        stream.truncate(64 * 1024**2 + 1)
    r = run(t=f)
    assert r["verdict"] == "fail" and r["files"]["native"]["raw_keys"] is None


def test_duplicate_header(tmp_path):
    f = tmp_path / "duplicate.safetensors"
    obj = b'{"name":{},"name":{}}'
    f.write_bytes(len(obj).to_bytes(8, "little") + obj)
    assert "duplicate" in run(t=f)["files"]["native"]["errors"][0]


def test_precision_opt_in(tmp_path):
    def cast(data):
        for k in data:
            data[k] = data[k].astype("float16")

    t = mutate(tmp_path, cast)
    assert run(t=t)["verdict"] == "inconclusive"
    assert run(t=t, profile="fp16")["verdict"] == "sampled_agreement"


@pytest.mark.parametrize("mode", ["sampled", "exhaustive"])
def test_native_dora_blocks_all_numerical_claims(tmp_path, mode, capsys):
    source = load_file(S)
    for k, tensor in source.items():
        if k.endswith(".lora_up.weight"):
            tensor.fill(0)
    s, t = tmp_path / "source.safetensors", tmp_path / "target.safetensors"
    save_file(source, s)
    target = load_file(N)
    for tensor in target.values():
        tensor.fill(0)
    target[Q + ".lora_A.weight"][0, 0] = 1
    target[Q + ".lora_B.weight"][0, 0] = 1
    marker = V + ".lora_magnitude_vector"
    target[marker] = np.ones(1280, np.float32)
    save_file(target, t)
    assert main(["compare", str(s), str(t), "--mode", mode, "--native-unit-scale", "--json"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == "inconclusive" and "unsupported_configuration" in report["reasons"]
    assert "numerical_mismatch" not in report["reasons"] and report["layers"] == {}
    for side, data in (("kohya", source), ("native", target)):
        file = report["files"][side]
        assert file["compared_keys"] == file["verified_attachments"] == []
        assert set(file["unassessed_keys"]) == set(data)
        assert len(file["paired_projections"]) == 4
        assert set(file["matched_keys"]) == set(data) - {marker}
        assert file["coverage"] == {
            "compared_keys": 0,
            "total_keys": len(data),
            "compared_parameters": 0,
            "total_parameters": sum(a.size for a in data.values()),
        }
    assert report["files"]["native"]["unsupported_keys"] == [marker]


@pytest.mark.parametrize(
    "key, issue",
    [
        ("unet.off_manifest.lora_magnitude_vector.weight", "DoRA"),
        ("unet.off_manifest.lokr_w1", "LoKr"),
        ("unet.off_manifest.lora_B.bias", "B bias"),
        ("unet.off_manifest.processor.to_out_lora.up.bias", "B bias"),
        ("unet.lora_controlnet", "Control LoRA"),
    ],
)
def test_native_adapter_wide_markers(tmp_path, key, issue):
    t = mutate(tmp_path, lambda d: d.update({key: np.ones(1, np.float32)}), FIX / "half.safetensors")
    r = run(t=t)
    assert r["verdict"] == "inconclusive" and not r["layers"]
    assert any(issue in reason for reason in r["files"]["native"]["configuration_issues"])
    assert r["files"]["native"]["unsupported_keys"] == [key]


@pytest.mark.parametrize(
    "key",
    [
        "text_encoder.extra.bias",
        "text_encoder_2.other.lora_magnitude_vector",
        "text_encoder.layer.lokr_w1",
        "text_encoder.layer.lora_B.bias",
        "unet.off_manifest.dora_scale.lora_magnitude_vector",
    ],
)
def test_ordinary_half_scale_survives_component_filter(tmp_path, key):
    t = mutate(tmp_path, lambda d: d.update({key: np.ones(1, np.float32)}), FIX / "half.safetensors")
    r = run(t=t, mode="exhaustive")
    assert r["verdict"] == "fail" and "numerical_mismatch" in r["reasons"]
    assert r["files"]["native"]["configuration_issues"] == []
    assert r["layers"][Q]["E"] == 3.5 and len(r["layers"]) == 4
    assert r["files"]["native"]["coverage"]["compared_keys"] == 8


def test_global_configuration_preserves_structural_inventory(tmp_path):
    source = {k: v for k, v in load_file(S).items() if not k.startswith(PROJECTIONS[0].kohya)}
    s = tmp_path / "source.safetensors"
    save_file(source, s)
    t = mutate(tmp_path, lambda d: d.update({V + ".lokr_w1": np.ones(1, np.float32)}))
    r = run(s, t)
    assert r["verdict"] == "inconclusive" and not r["layers"]
    assert len(r["files"]["kohya"]["missing_keys"]) == len(r["files"]["native"]["extra_keys"]) == 2
    source[next(k for k in source if k.endswith(".lora_up.weight"))][0, 0] = np.nan
    save_file(source, s)
    r = run(s, t)
    assert r["verdict"] == "fail" and "invalid_input" in r["reasons"]
