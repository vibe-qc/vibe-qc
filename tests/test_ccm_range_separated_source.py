"""The SR/LR range-separated fit as a source of the real-Gamma fold (#519, RG-12).

``RangeSeparatedFitSource`` plans and builds the multi-k GDF driver's
default 3-D fit (``fit_source="sr"``) per kept channel, in the canonical
auxiliary frame the fold contracts across bras. With it the real-Gamma
producer and the neutral-Bloch control run on one finite SR source, so their
energies per cell must agree (Theorem 1 for the SR source). The SR fit needs
no RSGDF tail on core-heavy cells: its reciprocal sphere is planned from a
raw integral error, not from the most compact AO pair.

Rows: the fold on the SR source against ``run_ccm_rhf_gdf(fit_source="sr")``
on the H2 cube at (1,1,1) and (2,1,1); the block storage on the same source
against the dense fold; the star plan and star storage on the SR fits; the
receipt and the planned domains; the refusals.
The slow row is MgO pob-DZVP-rev2 (1,1,1), whose classic fit needs a
54,492 Ha tail.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.direct import (
    ccm_direct_oneelectron,
    ccm_exchange_q0_madelung,
    run_ccm_rhf_direct,
)
from vibeqc.periodic.ccm.fit_source import (
    RANGE_SEPARATED_ENGINE,
    RangeSeparatedFitSource,
)
from vibeqc.periodic.ccm.neutral import (
    ccm_neutral_cderi_blocks,
    ccm_neutral_cderi_fold,
)
from vibeqc.periodic.ccm.ri import run_ccm_rhf_gdf

pytestmark = [pytest.mark.experimental, pytest.mark.aiccm]

TIGHT = dict(conv_tol=1e-12, conv_tol_grad=1e-9, max_iter=300)


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


def _h2_cube(nrep):
    return CCMSystem(PeriodicSystem(
        3, np.diag([6.0, 6.0, 6.0]),
        [Atom(1, [3.0, 3.0, 2.3]), Atom(1, [3.0, 3.0, 3.7])], 0, 1),
        nrep, "sto-3g")


def test_shared_sr_sources_are_bitwise_the_per_channel_fold(monkeypatch):
    """The SR real-space sources do not depend on the channel's transfer, so
    the source builds them for chunks of channels in one native pass (#543,
    ``_build_sr_three_center_for_groups``). The fold on shared sources is
    bitwise the fold that builds each channel's source on its own."""
    from vibeqc.periodic.ccm import fit_source as fs

    ccm = _h2_cube((2, 2, 1))
    shared_channels = []
    original = fs._BoundRangeSeparated._shared_source

    def counting(self, qi, bras):
        out = original(self, qi, bras)
        shared_channels.append(out is not None)
        return out

    monkeypatch.setattr(fs._BoundRangeSeparated, "_shared_source", counting)
    shared = _quiet(ccm_neutral_cderi_fold, ccm, storage="block",
                    fit_source=RangeSeparatedFitSource())
    assert sum(shared_channels) >= 2
    monkeypatch.setattr(fs._BoundRangeSeparated, "prefetch", lambda self, req: None)
    alone = _quiet(ccm_neutral_cderi_fold, ccm, storage="block",
                   fit_source=RangeSeparatedFitSource())
    np.testing.assert_array_equal(shared.to_dense(), alone.to_dense())


def test_star_plan_reconstructs_sr_fits_exactly():
    """The SR fits come in the canonical auxiliary frame ``M^(-1/2) T``, which
    is covariant under the plan's operations (``M(gq) = A M(q) A^H`` and
    ``T(gk, gq) = A T(k, q) Om``, so ``L(gk, gq) = A L(k, q) Om``) and under
    time reversal (``L(-k, -q) = conj L(k, q)``). The fold's star plan and its
    time-reversal pairing therefore reconstruct them exactly: on the compact
    H2 (2,2,2) torus J and K on random nonsymmetric densities from the
    symmetry-reduced fold and from star storage equal the unreduced SR fold
    to rounding, and star storage keeps 42 of 64 fits."""
    ccm = _h2_cube((2, 2, 2))
    full = _quiet(ccm_neutral_cderi_fold, ccm, storage="block",
                  fit_source=RangeSeparatedFitSource())
    plan = _quiet(ccm_neutral_cderi_fold, ccm, storage="block", symmetry=True,
                  fit_source=RangeSeparatedFitSource())
    star = _quiet(ccm_neutral_cderi_fold, ccm, storage="block-star",
                  fit_source=RangeSeparatedFitSource())
    S, _, _ = ccm_direct_oneelectron(ccm)
    xi = ccm_exchange_q0_madelung(ccm)
    rng = np.random.default_rng(3)
    n = S.shape[0]
    for _ in range(2):
        x = rng.standard_normal((n, n))
        d = x @ x.T / n
        k_full = full.exchange(d, S, xi)
        scale = float(np.max(np.abs(k_full)))
        for other in (plan, star):
            assert np.max(np.abs(other.coulomb(d) - full.coulomb(d))) < 1e-13
            assert np.max(np.abs(other.exchange(d, S, xi) - k_full)) < 1e-13 * scale
    record = star.info["star_storage"]
    assert (record["stored_fits"], record["total_fits"]) == (42, 64)
    assert record["recipe_check_max_rel"] < 1e-12


@pytest.mark.parametrize("nrep", [(1, 1, 1), (2, 1, 1)],
                         ids=lambda m: "x".join(map(str, m)))
def test_the_fold_on_the_sr_source_is_the_sr_bloch_control(nrep):
    ccm = _h2_cube(nrep)
    bloch = _quiet(run_ccm_rhf_gdf, ccm, fit_source="sr")
    L = _quiet(ccm_neutral_cderi_fold, ccm, fit_source=RangeSeparatedFitSource())
    real = _quiet(run_ccm_rhf_direct, ccm, cderi=L, **TIGHT)
    assert real.converged and bloch.converged
    assert abs(real.energy / ccm.n_cells - bloch.energy) < 1e-10


def test_block_storage_on_the_sr_source():
    ccm = _h2_cube((2, 1, 1))
    source = RangeSeparatedFitSource()
    dense = _quiet(ccm_neutral_cderi_fold, ccm, fit_source=source)
    block = _quiet(ccm_neutral_cderi_blocks, ccm, fit_source=source)
    receipt = block.info["fit_source"]
    assert receipt["engine"] == RANGE_SEPARATED_ENGINE
    assert receipt["auxiliary_frame"] == "canonical"
    # The planned reciprocal sphere is far below the classic 200 Ha base.
    assert receipt["reciprocal_cutoff_ha"] < 50.0
    assert receipt["pair_cutoff_bohr"] >= 15.0
    S, _, _ = ccm_direct_oneelectron(ccm)
    xi = ccm_exchange_q0_madelung(ccm)
    from vibeqc.periodic.ccm.ri import ccm_ri_j_neutral, ccm_ri_k_neutral

    rng = np.random.default_rng(2)
    x = rng.standard_normal(S.shape)
    d = x @ x.T / S.shape[0]
    d = 0.5 * (d + d.T)
    from vibeqc.periodic.ccm.stability import circulant_projection

    d = circulant_projection(ccm.nrep, d)     # block storage sees the projection
    np.testing.assert_allclose(block.coulomb(d), ccm_ri_j_neutral(dense, d),
                               atol=1e-11, rtol=0)
    np.testing.assert_allclose(block.exchange(d, S, xi),
                               ccm_ri_k_neutral(dense, d) + xi * (S @ d @ S),
                               atol=1e-11, rtol=0)
    e_dense = _quiet(run_ccm_rhf_direct, ccm, cderi=dense, **TIGHT).energy
    e_block = _quiet(run_ccm_rhf_direct, ccm, cderi=block, **TIGHT).energy
    assert abs(e_block - e_dense) / ccm.n_cells < 1e-10


def test_the_drivers_and_the_runner_take_the_sr_source(tmp_path):
    """``fit_source=`` on the driver and ``fit_source="sr"`` on the runner's
    real-gamma variant: the SR identity of the Bloch control, its energy."""
    import tomllib

    from vibeqc import run_periodic_job
    from vibeqc._vibeqc_core import BasisSet

    ccm = _h2_cube((2, 1, 1))
    bloch = _quiet(run_ccm_rhf_gdf, ccm, fit_source="sr")
    real = _quiet(run_ccm_rhf_direct, ccm, fit_source=RangeSeparatedFitSource(),
                  **TIGHT)
    assert abs(real.energy / ccm.n_cells - bloch.energy) < 1e-10
    assert real.fit_source == "sr" and real.rsgdf_tail_ke_cutoff is None
    assert real.neutral_source.identity == bloch.neutral_source.identity
    fit = real.neutral_source.as_dict()["details"]["fit"]
    assert fit["engine"] == RANGE_SEPARATED_ENGINE and fit["source"] == "sr"

    cell = ccm.unit_system
    basis = BasisSet(cell.unit_cell_molecule(), "sto-3g")
    rg = _quiet(run_periodic_job, cell, basis, method="aiccm",
                variant="real-gamma", aiccm_lattice_extension=(2, 1, 1),
                fit_source="sr", output=str(tmp_path / "rg"), progress=False)
    nb = _quiet(run_periodic_job, cell, basis, method="aiccm",
                variant="neutral-bloch", aiccm_lattice_extension=(2, 1, 1),
                fit_source="sr", output=str(tmp_path / "nb"), progress=False)
    assert rg.energy == pytest.approx(nb.energy, abs=1e-9)
    manifests = [tomllib.loads((tmp_path / f"{n}.system").read_text())
                 for n in ("rg", "nb")]

    def field(m, key):
        for section in m.values():
            if isinstance(section, dict) and key in section:
                return section[key]
        raise KeyError(key)

    assert field(manifests[0], "neutral_source_identity") == \
        field(manifests[1], "neutral_source_identity")
    assert field(manifests[0], "rsgdf_fit_source_executed") == "sr"


def test_range_separated_source_refusals():
    with pytest.raises(ValueError, match="omega"):
        RangeSeparatedFitSource(omega=0.0)
    with pytest.raises(ValueError, match="raw_integral_error"):
        RangeSeparatedFitSource(raw_integral_error=-1.0)
    with pytest.raises(ValueError, match="ke_cutoff"):
        RangeSeparatedFitSource(ke_cutoff=0.0)
    ccm = _h2_cube((1, 1, 1))
    with pytest.raises(NotImplementedError, match="screened"):
        _quiet(ccm_neutral_cderi_fold, ccm, fit_source=RangeSeparatedFitSource(),
               omega_screen=0.11)
    L = _quiet(ccm_neutral_cderi_fold, ccm, ke_cutoff=60.0)
    with pytest.raises(ValueError, match="fit_source builds the cderi"):
        _quiet(run_ccm_rhf_direct, ccm, cderi=L, fit_source=RangeSeparatedFitSource())
    from vibeqc.periodic.ccm.real_gamma_runner import run_real_gamma_scf

    with pytest.raises(ValueError, match="fit_source must be 'classic' or 'sr'"):
        run_real_gamma_scf(ccm.unit_system, "sto-3g", "RHF", (1, 1, 1),
                           fit_source="mdf")


@pytest.mark.slow
def test_mgo_pob_dzvp_rev2_on_the_sr_source():
    """The #442 cell: the classic fit needs a 54,492 Ha tail here."""
    ang = 1.0 / 0.529177210903
    a = 4.2239 * ang
    lat = 0.5 * a * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    cell = PeriodicSystem(3, lat, [Atom(12, [0.0, 0.0, 0.0]),
                                   Atom(8, [0.5 * a] * 3)], 0, 1)
    ccm = CCMSystem(cell, (1, 1, 1), "pob-dzvp-rev2")
    bloch = _quiet(run_ccm_rhf_gdf, ccm, fit_source="sr")
    L = _quiet(ccm_neutral_cderi_fold, ccm, fit_source=RangeSeparatedFitSource())
    real = _quiet(run_ccm_rhf_direct, ccm, cderi=L, initial_guess="SAD",
                  conv_tol=1e-10, conv_tol_grad=1e-7, max_iter=300)
    assert abs(real.energy - bloch.energy) < 1e-8
    # The tailed classic Bloch energy of #442, -274.3905141646 Ha/cell.
    assert abs(bloch.energy - (-274.3905141646)) < 1e-4
