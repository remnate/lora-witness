"""Binary64 comparison of low-rank effects."""

import hashlib
import math

import numpy as np

PROFILES = {"fp32": (1e-10, 1e-5), "fp16": (1e-9, 2e-3)}
PROBES = 32


def probes(path, inputs, seed):
    digest = hashlib.sha256(f"{seed}/{path}".encode()).digest()
    derived = int.from_bytes(digest[:16], "little")
    rng = np.random.Generator(np.random.PCG64(derived))
    x = rng.standard_normal((inputs, PROBES))
    return x, {
        "rng": "PCG64/NumPy standard_normal float64",
        "derived_seed": str(derived),
        "shape": list(x.shape),
        "sha256": hashlib.sha256(x.astype("<f8").tobytes()).hexdigest(),
    }


def norm(matrix):
    largest = float(np.max(np.abs(matrix)))
    return largest * float(np.linalg.norm(matrix / largest)) if largest else 0.0


def compare_layer(source, target, path, mode, seed, rows, atol, rtol):
    a, b, scale = source
    c, d, strength = target
    xinfo = None
    if mode == "sampled":
        x, xinfo = probes(path, a.shape[1], seed)
        a, c = a @ x, c @ x

    peak = error = reference_norm = delta_norm = 0.0
    signatures = [hashlib.sha256(), hashlib.sha256()]
    worst = [0, 0]
    for start in range(0, b.shape[0], rows):
        reference = (b[start : start + rows] @ a) * scale
        target_block = (d[start : start + rows] @ c) * strength
        if mode == "exhaustive":
            for digest, values in zip(signatures, (reference, target_block)):
                digest.update(values.astype("<f8", copy=False).tobytes())
        delta = target_block - reference
        abs_delta = np.abs(delta)
        coord = np.unravel_index(abs_delta.argmax(), delta.shape)
        block_error = float(abs_delta[coord])
        if block_error > error:
            error = block_error
            worst = [start + int(coord[0]), int(coord[1])]
        peak = max(peak, float(np.max(np.abs(reference))))
        reference_norm = math.hypot(reference_norm, norm(reference))
        delta_norm = math.hypot(delta_norm, norm(abs_delta))
    floor = peak <= (atol / rtol if rtol else atol)
    result = {
        "mode": mode,
        "E": error,
        "M": peak,
        "threshold": atol + rtol * peak,
        "relative_frobenius": delta_norm / reference_norm if reference_norm else None,
        "worst_coordinate": worst,
        "under_resolution": floor,
        "probe": xinfo,
    }
    if mode == "exhaustive":
        result["effect_sha256"] = [digest.hexdigest() for digest in signatures]
    if floor and mode == "sampled":
        result = compare_layer(source, target, path, "exhaustive", seed, rows, atol, rtol)
        result["fallback_probe"] = xinfo
        return result
    if error > result["threshold"]:
        result["status"] = "numerical_mismatch"
    elif floor and (error or peak):
        result["status"] = "under_resolution"
    else:
        result["status"] = "zero_effect_agreement" if peak == 0 else "agreement"
    return result
