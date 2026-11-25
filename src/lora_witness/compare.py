import math
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

from threadpoolctl import threadpool_limits

from .artifact import Artifact
from .manifest import LOADER_EVIDENCE, MANIFEST_HASH, PROJECTIONS, SUFFIXES
from .numeric import PROFILES, compare_layer


def unique_swaps(layers):
    """Reciprocal exact blockwise cross-matches, unique among comparable layers."""
    if not layers or any("effect_sha256" not in layer for layer in layers.values()):
        return []
    sources, targets = defaultdict(list), defaultdict(list)
    for path, layer in layers.items():
        source, target = layer["effect_sha256"]
        shape = tuple(layer["shape"])
        sources[shape, source].append(path)
        targets[shape, target].append(path)
    candidates = {}
    for identity, source_names in sources.items():
        target_names = targets.get(identity, [])
        if len(source_names) == len(target_names) == 1:
            path = source_names[0]
            if layers[path]["status"] == "numerical_mismatch":
                candidates[path] = target_names[0]
    return [[a, b] for a, b in candidates.items() if a < b and candidates.get(b) == a]


def compare(
    source,
    target,
    *,
    declared=False,
    mode="sampled",
    seed=0,
    rows=64,
    profile="fp32",
    atol=None,
    rtol=None,
):
    atol = PROFILES[profile][0] if atol is None else atol
    rtol = PROFILES[profile][1] if rtol is None else rtol
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (atol, rtol)):
        raise ValueError("tolerances must be finite and between 0 and 1")
    if not 1 <= rows <= 256 or mode not in ("exhaustive", "sampled"):
        raise ValueError("expected 1 to 256 rows and a supported mode")
    with ExitStack() as stack, threadpool_limits(limits=1, user_api="blas"):
        left = Artifact("kohya", Path(source), profile, stack)
        right = Artifact("native", Path(target), profile, stack)
        if not declared:
            right.global_unknown.append("unit-scale native export undeclared")
        reasons, layers = set(), {}
        report = {
            "schema_version": "1.0",
            "tool_version": "0.1.0",
            "mode": mode,
            "seed": seed,
            "block_rows": rows,
            "tolerance_profile": profile + "-v1",
            "atol": atol,
            "rtol": rtol,
            "manifest_sha256": MANIFEST_HASH,
            "component": "SDXL Base UNet",
            "loader_evidence": LOADER_EVIDENCE,
            "files": {"kohya": left.r, "native": right.r},
            "layers": layers,
        }
        for f in (left, right):
            if f.r["errors"]:
                reasons.add("invalid_input")
            if f.global_unknown:
                reasons.add("unsupported_configuration")
            if f.r["unsupported_keys"] or f.global_unknown:
                reasons.add("unsupported_input")
        for p in PROJECTIONS:
            present = [p in f.groups for f in (left, right)]
            if not any(present):
                continue
            if not all(present):
                missing, extra = (right, left) if present[0] else (left, right)
                if not missing.complete:
                    continue
                for suffix, role in SUFFIXES[missing.side].items():
                    if role != "alpha":
                        missing.r["missing_keys"].append(getattr(p, missing.side) + "." + suffix)
                extra.r["extra_keys"].extend(extra.groups[p].values())
                if not (
                    missing.global_unknown
                    or extra.global_unknown
                    or p in missing.unknown
                    or (not missing.groups and missing.r["unsupported_keys"])
                ):
                    reasons.add("projection_set_mismatch")
                continue
            if any(
                not f.complete or f.global_unknown or p in f.invalid or p in f.unknown
                for f in (left, right)
            ):
                continue
            s, t = left.factors(p), right.factors(p)
            layer = compare_layer(s, t, p.native, mode, seed, rows, atol, rtol)
            layer["shape"] = [p.outputs, p.inputs]
            for f, factors in ((left, s), (right, t)):
                roles = f.groups[p]
                layer[f.side] = {
                    "rank": len(factors[0]),
                    "scale": factors[2],
                    "dtypes": {r: f.r["keys"][key]["dtype"] for r, key in roles.items()},
                }
                f.r["compared_keys"].extend(roles.values())
                f.r["verified_attachments"].append(p.native)
            layers[p.native] = layer
            if layer["status"] in ("numerical_mismatch", "under_resolution"):
                reasons.add(layer["status"])
        if not layers:
            reasons.add("empty_comparison")
        if reasons & {"projection_set_mismatch", "invalid_input", "numerical_mismatch"}:
            verdict = "fail"
        elif reasons:
            verdict = "inconclusive"
        else:
            verdict = (
                "sampled_agreement"
                if any(v["mode"] == "sampled" for v in layers.values())
                else "exhaustive_agreement_within_tolerance"
            )
        report.update(
            verdict=verdict,
            reasons=sorted(reasons),
            likely_swaps=unique_swaps(layers) if reasons == {"numerical_mismatch"} else [],
        )
        for f in (left, right):
            f.r["unassessed_keys"] = sorted(f.r["keys"].keys() - set(f.r["compared_keys"]))
            f.r["coverage"] = {
                "compared_keys": len(f.r["compared_keys"]),
                "total_keys": f.r["raw_keys"],
                "compared_parameters": sum(
                    math.prod(f.r["keys"][k]["shape"]) for k in f.r["compared_keys"]
                ),
                "total_parameters": f.r["parameters"],
            }
        return report
