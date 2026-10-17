"""Block-circulant cderi storage on the open-shell real-Gamma drivers (#519, RG-5).

``cderi_storage="block"`` on ``run_ccm_uhf_direct`` and ``run_ccm_uks_direct``:
J from the total density, per-spin exchange through the block object's
occupied-rank factor on each spin density, the exchange ``q = 0`` seam per k
as on the closed-shell drivers. Identity to the dense fold at 1e-10 Ha/cell
on the Li doublet cube for UHF, PBE and PBE0; a prebuilt block object is
accepted; the Wigner-Seitz partition and the exchange form reach the
open-shell drivers.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.block_cderi import is_block_cderi
from vibeqc.periodic.ccm.direct import run_ccm_uhf_direct, run_ccm_uks_direct
from vibeqc.periodic.ccm.neutral import ccm_neutral_cderi_blocks

pytestmark = [pytest.mark.experimental, pytest.mark.aiccm]

ENERGY_TOL_HA_PER_CELL = 1e-10
#: Both runs converge far below the default commutator bound (1e-6), under
#: which two SCF paths may stop at densities 1e-6 apart while their energies,
#: quadratic in that error, agree to 1e-10.
TIGHT = dict(conv_tol=1e-12, conv_tol_grad=1e-9, max_iter=300)


def _li_doublet(nrep=(2, 1, 1)):
    cell = PeriodicSystem(
        3, np.diag([7.0, 7.0, 7.0]), [Atom(3, [3.5, 3.5, 3.5])], 0, 2)
    return CCMSystem(cell, nrep, "sto-3g")


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


@pytest.fixture(scope="module")
def li():
    ccm = _li_doublet()
    blk = ccm_neutral_cderi_blocks(ccm)
    return ccm, blk, blk.to_dense()


def _assert_same_open_shell_state(ccm, a, b):
    assert a.converged and b.converged
    assert abs(a.energy - b.energy) / ccm.n_cells < ENERGY_TOL_HA_PER_CELL
    np.testing.assert_allclose(a.density_alpha, b.density_alpha, atol=1e-8)
    np.testing.assert_allclose(a.density_beta, b.density_beta, atol=1e-8)
    if hasattr(a, "s_squared"):  # the UHF result; the KS result carries none
        assert a.s_squared == pytest.approx(b.s_squared, abs=1e-8)


def test_uhf_block_matches_dense(li):
    ccm, blk, dense = li
    d = _quiet(run_ccm_uhf_direct, ccm, cderi=dense, **TIGHT)
    b = _quiet(run_ccm_uhf_direct, ccm, cderi=blk, **TIGHT)
    assert is_block_cderi(b.cderi) and b.cderi_info["storage"] == "block"
    assert d.cderi_info["storage"] == "dense"
    _assert_same_open_shell_state(ccm, d, b)
    assert b.exchange_q0 == d.exchange_q0


@pytest.mark.parametrize("functional", ["pbe", "pbe0"])
def test_uks_block_matches_dense(li, functional):
    ccm, blk, dense = li
    d = _quiet(run_ccm_uks_direct, ccm, functional, cderi=dense, **TIGHT)
    b = _quiet(run_ccm_uks_direct, ccm, functional, cderi=blk, **TIGHT)
    assert b.cderi_info["storage"] == "block"
    _assert_same_open_shell_state(ccm, d, b)
    assert b.e_xc == pytest.approx(d.e_xc, abs=1e-9)
    assert b.e_hf_exchange == pytest.approx(d.e_hf_exchange, abs=1e-9)


def test_open_shell_drivers_build_block_storage_themselves(li):
    ccm, _, dense = li
    d = _quiet(run_ccm_uhf_direct, ccm, cderi=dense, **TIGHT)
    b = _quiet(run_ccm_uhf_direct, ccm, cderi_storage="block",
               exchange_form="pair", **TIGHT)
    assert b.cderi_info["storage"] == "block"
    assert b.cderi_info["build"] == "fold"
    _assert_same_open_shell_state(ccm, d, b)


def test_wigner_seitz_partition_reaches_the_open_shell_drivers(li):
    ccm, _, _ = li
    b = _quiet(run_ccm_uhf_direct, ccm, cderi_storage="block",
               ws_cutoff_bohr=1e6)
    assert b.cderi_info["ws_cutoff_bohr"] == 1e6
    assert b.cderi_info["ws_n_omitted"] == 0
    with pytest.raises(ValueError, match="ws_cutoff_bohr"):
        _quiet(run_ccm_uks_direct, ccm, "pbe", ws_cutoff_bohr=4.0)
