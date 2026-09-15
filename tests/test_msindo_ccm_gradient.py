"""MSINDO Cyclic Cluster Model (CCM) analytic nuclear gradient — Phase 5.

Two independent cross-checks of :func:`ccm_gradient_analytic`:

1. **Analytic vs finite difference** (the exact derivative of vibe-qc's own CCM
   total energy).  This is the primary, universal check — it holds for every
   cluster regardless of Wigner-Seitz symmetry, in 1-D / 2-D / 3-D and with the
   Madelung embedding on and off.

2. **Analytic vs the MSINDO oracle** (``CARTOPT ANALY GRADONLY``, parsed
   out-of-process by ``examples/regression/msindo/runner_msindo.py``; values
   frozen in ``ccm_reference.json`` ``gradient_clusters``).  The oracle's
   central-atom-only Madelung gradient assembly (``dedmadelsum.f``:
   ``EDX(IATOM) += DMADEL·q_I``) equals the exact energy derivative only for
   *mirror-symmetric* WS cells, so the oracle cells used here are symmetric ones
   (the distorted MgO bulk, the undistorted MgO(100) slab, the H-F NOEWALD
   chain).  For asymmetric ionic Madelung cells the oracle's own analytic
   gradient deviates from the finite difference of its own energy by up to
   ~2e-2 Ha/bohr, whereas vibe-qc's direct-assembly gradient stays exact — see
   docs/user_guide/msindo.md (CCM analytic gradient) and msindo_ccm_gradient_analytic.py.
   The frozen rows use the executable's default ``DELEN=1e-8`` SCF stopping
   threshold.  A separate tight-DELEN H-F regression below distinguishes that
   convergence envelope from an implementation error in the derivative.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from vibeqc.semiempirical.methods import msindo_ccm as ccm
from vibeqc.semiempirical.methods import (
    msindo_ccm_gradient_analytic as gradient_module,
)
from vibeqc.semiempirical.methods.msindo_ccm_gradient_analytic import (
    ccm_gradient_analytic,
)

_SYM2Z = {"H": 1, "He": 2, "C": 6, "N": 7, "O": 8, "F": 9, "Mg": 12,
          "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17}

_REF_PATH = (Path(__file__).parent.parent / "examples" / "regression" / "msindo"
             / "ccm_reference.json")
_REF = json.loads(_REF_PATH.read_text())
_CLUSTERS = {c["name"]: c for c in _REF["clusters"]}
_GRAD_CLUSTERS = {c["name"]: c for c in _REF["gradient_clusters"]}


def _Z_and_coords(cluster):
    Z = [_SYM2Z[s] for s, *_ in cluster["real_atoms"]]
    coords = np.array([[x, y, z] for _s, x, y, z in cluster["real_atoms"]], float)
    return Z, coords


def _distort(coords, amp=0.04):
    """Deterministic symmetry-breaking displacement (Angstrom) so the gradient
    is non-trivial (the reference geometries are near a stationary point)."""
    C = np.asarray(coords, float).copy()
    for i in range(len(C)):
        C[i] += amp * ((-1.0) ** i) * np.array([1.0, -0.7, 0.5]) * (1.0 + 0.1 * i)
    return C


# --------------------------------------------------------------------------- #
# 1. Analytic == finite difference  (exact energy derivative; all dims/madelung)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name,dim", [
    ("hfionic_a1.6_n2_madelung", 1),     # 1-D ionic chain
    ("hf_layer_a1.6_madelung", 2),       # 2-D ionic slab
    ("mgo_rocksalt_a4.21_madelung", 3),  # 3-D ionic bulk
])
@pytest.mark.parametrize("madelung", [False, True])
def test_ccm_analytic_matches_fd(name, dim, madelung):
    """The analytic CCM gradient reproduces the fixed-WS finite-difference
    gradient of the *same* total energy to ~1e-6 Ha/bohr — the rigorous proof
    that it is the exact derivative — for 1-D / 2-D / 3-D, NOEWALD and Madelung.
    The 2-D and 3-D Madelung cases exercise the Parry/Heyes and Ewald reciprocal
    + K=0 + direct lattice-sum derivatives; the 1-D Madelung case exercises the
    finite ±T, ±2T lattice-sum derivative."""
    cluster = _CLUSTERS[name]
    assert len(cluster["translations"]) == dim
    Z, coords = _Z_and_coords(cluster)
    C = _distort(coords)
    T = cluster["translations"]
    g_an = ccm_gradient_analytic(Z, C, T, madelung=madelung, conv_tol=1e-11)
    g_fd = ccm.ccm_gradient_fd(Z, C, T, madelung=madelung, conv_tol=1e-11,
                               step=1e-3)
    # non-trivial gradient (guard against a vacuous all-zero comparison)
    assert np.max(np.abs(g_fd)) > 1e-2
    assert np.max(np.abs(g_an - g_fd)) < 1e-6, (
        f"{name} madelung={madelung}: "
        f"max|analytic-FD|={np.max(np.abs(g_an - g_fd)):.2e}")


def test_ccm_fd_falls_back_when_cpp_returns_nonfinite(monkeypatch):
    cluster = _CLUSTERS["hfionic_a1.6_n2_madelung"]
    Z, coords = _Z_and_coords(cluster)
    C = _distort(coords)
    T = cluster["translations"]

    def fake_cpp(*_args, **_kwargs):
        return np.full((len(Z), 3), np.nan)

    monkeypatch.setattr(
        ccm,
        "_cpp_ccm_gradient_fd_kernel",
        lambda: (fake_cpp, object()),
    )

    g_fd = ccm.ccm_gradient_fd(Z, C, T, madelung=False, conv_tol=1e-11,
                               step=1e-3)

    assert np.all(np.isfinite(g_fd))
    assert np.max(np.abs(g_fd)) > 1e-2


@pytest.mark.parametrize("force_python", [False, True], ids=["native", "python"])
def test_ccm3d_madelung_gradient_is_strong_skew_basis_invariant(
    monkeypatch, force_python
):
    """Energy and derivative share one cutoff-complete 3-D Ewald inventory."""
    if force_python:
        monkeypatch.setattr(
            gradient_module, "_cpp_ccm_gradient_kernel", lambda: None
        )
        monkeypatch.setattr(ccm, "_cpp_ccm_gradient_fd_kernel", lambda: None)
    else:
        assert gradient_module._cpp_ccm_gradient_kernel() is not None
        assert ccm._cpp_ccm_gradient_fd_kernel() is not None

    atomic_numbers = [3, 1]
    coordinates = np.array([[0.0, 0.0, 0.0], [1.6, 0.1, -0.05]])
    cubic = np.eye(3) * 4.0
    skewed = cubic.copy()
    skewed[1] += 13.0 * skewed[0]

    reference = gradient_module.ccm_gradient_analytic(
        atomic_numbers,
        coordinates,
        cubic,
        madelung=True,
        conv_tol=1.0e-11,
    )
    transformed = gradient_module.ccm_gradient_analytic(
        atomic_numbers,
        coordinates,
        skewed,
        madelung=True,
        conv_tol=1.0e-11,
    )
    finite_difference = ccm.ccm_gradient_fd(
        atomic_numbers,
        coordinates,
        skewed,
        madelung=True,
        conv_tol=1.0e-11,
        step=1.0e-3,
    )

    np.testing.assert_allclose(transformed, reference, rtol=0.0, atol=1.0e-9)
    np.testing.assert_allclose(
        transformed, finite_difference, rtol=0.0, atol=1.0e-6
    )


# --------------------------------------------------------------------------- #
# 2. Analytic == MSINDO oracle  (symmetric WS cells; frozen oracle gradients)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", [
    "hfionic_chain_distorted",  # 1-D NOEWALD H-F chain
    "mgo_100_slab",             # 2-D Madelung MgO(100) slab (symmetric)
    "mgo_bulk_distorted",       # 3-D Madelung distorted MgO rocksalt
])
def test_ccm_analytic_matches_oracle(name):
    """The analytic gradient reproduces MSINDO's analytic CCM gradient
    (``CARTOPT ANALY GRADONLY``) to the oracle's print precision, for cells
    where the oracle's central-atom-only Madelung assembly is itself exact
    (mirror-symmetric WS).  Reference gradients are frozen in
    ccm_reference.json ``gradient_clusters`` (generated by
    examples/regression/msindo/runner_msindo.run_ccm_gradient)."""
    cluster = _GRAD_CLUSTERS[name]
    Z, coords = _Z_and_coords(cluster)
    ref = np.array(cluster["gradient_ha_per_bohr"], float)
    g = ccm_gradient_analytic(Z, coords, cluster["translations"],
                              madelung=cluster["ewald"], conv_tol=1e-11)
    assert g.shape == ref.shape
    assert np.max(np.abs(g - ref)) < 2e-5, (
        f"{name}: max|analytic-oracle|={np.max(np.abs(g - ref)):.2e}")


def test_hf_chain_analytic_matches_tight_oracle():
    """A tight-SCF oracle removes the apparent H-F gradient discrepancy.

    The retained campaign row used MSINDO's default ``DELEN=1e-8`` and differs
    by about 3.62e-6 Ha/bohr.  Re-running the same reference executable with
    ``DELEN=1e-14`` takes 28 cycles and converges to these x components, which
    agree with vibe-qc's exact fixed-WS energy derivative within 5e-9.
    """
    cluster = _GRAD_CLUSTERS["hfionic_chain_distorted"]
    Z, coords = _Z_and_coords(cluster)
    tight_oracle_x = np.array([
        -0.2074835915,
        0.2056009822,
        -0.2015090518,
        0.2033916611,
    ])
    g = ccm_gradient_analytic(
        Z,
        coords,
        cluster["translations"],
        madelung=cluster["ewald"],
        conv_tol=1e-11,
    )
    assert np.max(np.abs(g[:, 0] - tight_oracle_x)) < 5e-9
    assert np.max(np.abs(g[:, 1:])) < 1e-12


# --------------------------------------------------------------------------- #
# 3. Guards
# --------------------------------------------------------------------------- #

def test_ccm_gradient_closed_shell_only():
    """An odd-electron (open-shell) cluster raises NotImplementedError — the
    analytic CCM gradient is closed-shell (RHF) only."""
    # 5 He, +1 charge -> 9 valence electrons (odd).
    Z = [2, 2, 2, 2, 2]
    C = [[0, 0, 0], [2.5, 0, 0], [5, 0, 0], [7.5, 0, 0], [10, 0, 0]]
    T = [[12.5, 0, 0]]
    with pytest.raises(NotImplementedError, match="closed-shell"):
        ccm_gradient_analytic(Z, C, T, charge=1)


def test_ccm_gradient_rejects_invalid_cluster(monkeypatch):
    """A geometry that is not a valid cyclic cluster (round(Σ WS weight) !=
    NATOMS-1) raises ValueError rather than silently returning a wrong gradient.
    Forced here via the shared validity check used by run_ccm / ccm_gradient_fd
    (its geometric trigger is exercised by test_msindo_ccm.py)."""
    monkeypatch.setattr(ccm.WignerSeitzCells, "is_valid",
                        lambda self, n: False)
    with pytest.raises(ValueError, match="cyclic cluster"):
        ccm_gradient_analytic([2, 2, 2], [[0, 0, 0], [2.5, 0, 0], [5, 0, 0]],
                              [[7.5, 0, 0]])
