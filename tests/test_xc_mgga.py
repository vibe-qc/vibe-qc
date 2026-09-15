"""Meta-GGA XC functional support — F3 (v0.9.0).

Validates the τ-density-aware evaluation path landed in v0.9.0:

* Functional construction for τ-dependent mGGAs (TPSS, r²SCAN, SCAN,
  M06-L) — should report ``kind == XCKind.MGGA`` and HF fraction 0
  (pure mGGAs; HYB_MGGA variants are not yet bundled in this first
  cut and would route through ``xc_hyb_exx_coef``).
* ``Functional.eval_unpolarised_mgga`` returns finite exc / v_ρ /
  v_σ / v_τ on a small synthetic grid.
* ``Functional.eval_unpolarised`` raises an actionable error when
  called on an mGGA functional (caller must use the τ-aware
  signature).
* Full RKS / UKS SCF runs at TPSS and r²SCAN against H2, OH;
  energies land in the physically expected range (mGGAs are more
  binding than HF; bind below the HF reference at equilibrium
  geometry).
* Gradient + Hessian + periodic-SCF paths surface a clear
  "mGGA not yet supported" error at the call site instead of
  bubbling up a libxc-level exception.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc._vibeqc_core import Functional, XCKind


# ---------------------------------------------------------------------------
# Functional construction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["tpss", "r2scan", "scan", "m06-l", "m06l"])
def test_mgga_functional_constructs_and_reports_mgga_kind(name):
    f = Functional(name, 1)
    assert f.kind == XCKind.MGGA
    assert f.hf_exchange_fraction == 0.0     # all four are non-hybrid
    assert not f.is_hybrid
    assert not f.is_double_hybrid


def test_mgga_alias_lookup_is_case_insensitive():
    f1 = Functional("TPSS", 1)
    f2 = Functional("tpss", 1)
    assert f1.kind == f2.kind == XCKind.MGGA


# ---------------------------------------------------------------------------
# Evaluator surface
# ---------------------------------------------------------------------------

def test_eval_unpolarised_on_mgga_raises_actionable_error():
    """Calling the LDA/GGA-style evaluator on an mGGA should redirect
    the caller to the τ-aware path, not silently produce wrong
    numbers."""
    f = Functional("tpss", 1)
    n = 5
    rho = np.full(n, 0.5)
    sigma = np.full(n, 0.1)
    with pytest.raises(RuntimeError) as exc_info:
        f.eval_unpolarised(rho, sigma)
    msg = str(exc_info.value)
    assert "meta-GGA" in msg
    assert "eval_unpolarised_mgga" in msg


# ---------------------------------------------------------------------------
# Live SCF — RKS-TPSS on H2 against literature.
# ---------------------------------------------------------------------------

def _h2_molecule(R_bohr: float = 1.4) -> vq.Molecule:
    return vq.Molecule([
        vq.Atom(1, [0.0, 0.0, 0.0]),
        vq.Atom(1, [R_bohr, 0.0, 0.0]),
    ], 0, 1)


def test_rks_tpss_h2_converges_to_literature():
    """TPSS / def2-svp on H2 at R=1.4 bohr lands at ~-1.176 Ha
    (literature mGGA precision). The exact reference is sensitive to
    the integration grid; ±5 mHa around the published value is the
    acceptable window for a first τ-pipeline correctness check."""
    mol = _h2_molecule(1.4)
    res = vq.run_job(mol, basis="def2-svp", method="rks", functional="tpss",
                     output="/tmp/h2_tpss", progress=False)
    assert res.converged
    assert -1.18 < res.energy < -1.17, (
        f"RKS-TPSS/def2-svp/H2 energy {res.energy} outside expected "
        "literature window"
    )


def test_rks_r2scan_h2_converges():
    """r²SCAN / def2-svp on H2 — gates the v0.9.0 r²SCAN-3c headline
    composite. Energy ~-1.166 Ha at R=1.4 bohr."""
    mol = _h2_molecule(1.4)
    res = vq.run_job(mol, basis="def2-svp", method="rks", functional="r2scan",
                     output="/tmp/h2_r2scan", progress=False)
    assert res.converged
    assert -1.17 < res.energy < -1.16


def test_mgga_more_binding_than_hf_on_h2():
    """Physically: any reasonable XC approximation should bind H2 below
    the HF reference at equilibrium. This is the cheapest physical
    sanity-check that τ + v_τ are wired with the right sign convention.
    (A sign error in v_τ would make mGGAs *less* bound than HF.)"""
    mol = _h2_molecule(1.4)
    e_hf = vq.run_job(mol, basis="def2-svp", method="rhf",
                      output="/tmp/h2_hf", progress=False).energy
    e_tpss = vq.run_job(mol, basis="def2-svp", method="rks", functional="tpss",
                        output="/tmp/h2_tpss2", progress=False).energy
    e_r2 = vq.run_job(mol, basis="def2-svp", method="rks", functional="r2scan",
                      output="/tmp/h2_r2", progress=False).energy
    assert e_tpss < e_hf
    assert e_r2 < e_hf


# ---------------------------------------------------------------------------
# Gradient — upstream's F3 milestone shipped mGGA analytic gradients.
# ---------------------------------------------------------------------------

def test_mgga_analytic_gradient_runs():
    """mGGA analytic gradients landed with the upstream meta-GGA
    milestone (TPSS / TPSSh / M06-L / M06-2X). Computing the RKS-TPSS
    gradient should succeed and return an (n_atoms, 3) array — no
    longer the "not yet implemented" raise of the v0.9.0-prep cut."""
    mol = _h2_molecule(1.4)
    res = vq.run_job(mol, basis="def2-svp", method="rks", functional="tpss",
                     output="/tmp/h2_tpss_grad", progress=False)
    from vibeqc._vibeqc_core import compute_gradient_rks
    grad = compute_gradient_rks(mol, vq.BasisSet(mol, "def2-svp"), res)
    import numpy as np
    g = np.asarray(grad)
    assert g.shape == (2, 3)
    # H2 at R=1.4 bohr is near the TPSS equilibrium — the residual
    # gradient is small but the test only needs it to be finite.
    assert np.all(np.isfinite(g))
