import numpy as np
import pytest

from lora_witness.numeric import compare_layer, probes


def measure(s, t, mode="exhaustive", atol=0, rtol=0):
    return compare_layer(s, t, "test", mode, 19, 1, atol, rtol)


def test_exact_nonsquare_scaled_product():
    # BA/2 = [[1,2,-1],[-0.5,0,3.5]].
    a = np.array([[1, 2, -1], [0, 1, 3]], dtype=float)
    b = np.array([[2, 0], [-1, 2]], dtype=float)
    expected = np.array([[1, 2, -1], [-0.5, 0, 3.5]])
    source = a, b, 0.5
    assert measure(source, (expected, np.eye(2), 1))["E"] == 0
    mismatch = measure(source, (expected, np.eye(2), 0.5))
    assert mismatch["E"] == 1.75 and mismatch["M"] == 3.5
    assert mismatch["relative_frobenius"] == 0.5 and mismatch["worst_coordinate"] == [1, 2]


def test_finite_probes_have_nullspace():
    x, _ = probes("test", 33, 19)
    null_vector = np.linalg.svd(x.T, full_matrices=True)[2][-1:, :]
    a = np.ones_like(null_vector)
    b = np.ones((2, 1))
    s, t = (a, b, 1), (a + null_vector, b, 1)
    assert measure(s, t, "sampled", 1e-10, 1e-8)["status"] == "agreement"
    assert measure(s, t, "exhaustive", 1e-10, 1e-8)["status"] == "numerical_mismatch"


@pytest.mark.parametrize(
    "strength,other,status",
    [
        (0, 0, "zero_effect_agreement"),
        (0, 1, "numerical_mismatch"),
        (0, 1e-12, "under_resolution"),
        (1e-12, 1e-12, "under_resolution"),
    ],
)
def test_near_zero(strength, other, status):
    a = np.eye(2)
    result = measure((a, a, strength), (a, a, other), "sampled", 1e-10, 1e-5)
    assert result["status"] == status and result["mode"] == "exhaustive"
    assert result["fallback_probe"]["shape"] == [2, 32]
