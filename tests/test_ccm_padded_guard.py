"""Memory guard on the dense direct CCM four-center (`ccm_eri` / `ccm_eri_symmetric`).

The direct padded-cluster four-center is an O(n_pad⁴), screening-free *validation*
builder: `compute_eri(pad.basis)` allocates a dense tensor over the padded basis
(home cell + every ±2t image cell), so `n_pad = n_ref_ao × ERI-image-cells` and
the memory explodes for >2-atom / 3-D cells (e.g. c-diamond 2×2×2: ~80 reference
AOs → thousands of padded AOs → a TB-scale tensor). The guard turns that
out-of-memory crash into a clear, actionable error pointing first at the
scalable four-center path for the same Γ-CCM construction. It labels GDF as a
separately declared neutral fitted-torus control, not a Γ-CCM substitute.

Reported by the AICCM benchmark chat (c-diamond 2×2×2 OOM). Action per
`handovers/HANDOVER_AICCM_FOLLOWON.md`: error rather than OOM; route the same
Γ-CCM construction to the scalable four-center builder.
"""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.padded import (
    _check_padded_eri_size,
    _padded_eri_max_bytes,
    build_padded_cluster,
    ccm_eri,
    ccm_eri_symmetric,
    eri_cells,
)

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane


def _h2_chain(cell=6.0, vac=15.0):
    return PeriodicSystem(
        3,
        np.diag([cell, vac, vac]),
        [Atom(1, [0, 0, 0]), Atom(1, [1.4, 0, 0])],
        charge=0,
        multiplicity=1,
    )


def test_default_ceiling_allows_small_clusters():
    """A small 1-D cluster (n_pad ≈ 20) is far under the default ceiling — builds fine."""
    ccm = CCMSystem(_h2_chain(), (2, 1, 1), "sto-3g")
    eff = ccm_eri(ccm)
    assert eff.shape == (ccm.nbf, ccm.nbf, ccm.nbf, ccm.nbf)
    assert np.all(np.isfinite(eff))


@pytest.mark.parametrize("builder", [ccm_eri, ccm_eri_symmetric])
def test_low_ceiling_raises_actionable_error(monkeypatch, builder):
    """A tiny ceiling makes even a small cluster raise the actionable MemoryError."""
    monkeypatch.setenv("VIBEQC_CCM_PADDED_ERI_MAX_GB", "1e-7")
    ccm = CCMSystem(_h2_chain(), (2, 1, 1), "sto-3g")
    with pytest.raises(MemoryError) as exc:
        builder(ccm)
    msg = str(exc.value)
    assert msg.index("run_ccm_rhf_scalable") < msg.index("run_ccm_rhf_gdf")
    assert "separately declared neutral fitted-torus control" in msg
    assert "not a Γ-CCM substitute" in msg
    assert "VIBEQC_CCM_PADDED_ERI_MAX_GB" in msg
    assert "float64" in msg  # reports the dense tensor size


def test_env_override_is_read(monkeypatch):
    monkeypatch.setenv("VIBEQC_CCM_PADDED_ERI_MAX_GB", "2.5")
    assert _padded_eri_max_bytes() == int(2.5 * 1024**3)
    monkeypatch.delenv("VIBEQC_CCM_PADDED_ERI_MAX_GB", raising=False)
    assert _padded_eri_max_bytes() == 16 * 1024**3


def test_check_returns_padded_ao_count():
    """The guard returns n_pad = n_ref_ao × ERI-image-cells (the dense ERI dimension)."""
    ccm = CCMSystem(_h2_chain(), (2, 1, 1), "sto-3g")
    pad = build_padded_cluster(ccm, eri_cells(ccm))
    n_pad = _check_padded_eri_size(pad, "test")
    assert n_pad > ccm.nbf  # padded basis is larger than the home cell
    assert n_pad % ccm.nbf == 0  # whole-cell images


def test_zero_ao_element_raises_named_error():
    """A basis that omits an element (cc-pVDZ covers only H-Ar,
    so Cs gets zero AOs) must fail at CCMSystem construction naming the
    bare element -- not crash far downstream in build_padded_cluster with an
    opaque KeyError(0)."""
    cscl_like = PeriodicSystem(
        3,
        np.diag([4.2, 4.2, 4.2]),
        [Atom(55, [0, 0, 0]), Atom(17, [2.1, 2.1, 2.1])],  # Cs (uncovered) + Cl
        charge=0,
        multiplicity=1,
    )
    with pytest.raises(ValueError, match=r"Cs \(Z=55\)"):
        CCMSystem(cscl_like, (1, 1, 1), "cc-pvdz")
