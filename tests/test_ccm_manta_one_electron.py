"""MANTA's whole Hamiltonian on the real-Gamma torus (#519, RG-8b).

``MANTAOneElectronSource`` supplies the overlap, core Hamiltonian and nuclear
repulsion of MANTA's fitted RI HF on the torus; ``MANTAFitSource`` supplies
its fits (RG-8a). With both, the real-Gamma HF drivers evaluate the same
finite Hamiltonian as ``manta.run_periodic_ri_rhf_from_cutoffs`` / the UHF
sibling in the real supercell representation, so their energies per cell must
agree (Theorem 1 for the MANTA source). Rows: the one-electron torus equals
MANTA's RI HF overlap and core Hamiltonian moved onto the torus, its nuclear
repulsion MANTA's receipt; RHF on the H2 cube STO-3G at (2,1,1) and (3,1,1)
and UHF on the Li doublet cube STO-3G at (1,1,1) equal MANTA's energies; the
one-electron record joins the neutral-source identity.

The finite source (pair cutoff 6 bohr, auxiliary images to 12 bohr, a
12 Ha reciprocal sphere, MANTA's default omega, BvK Madelung seam) keeps the
fitting metric positive definite; it is not an accuracy setting. The MANTA
budget is explicit and uncapped (the domain factory admits a whole run's
upper bound).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem

pytestmark = [pytest.mark.experimental, pytest.mark.aiccm]

manta = pytest.importorskip("vibeqc.manta")

from vibeqc._vibeqc_core import BasisSet  # noqa: E402
from vibeqc.aux_basis import default_aux_for, make_aux_basis_set  # noqa: E402
from vibeqc.manta_scf import IntegralEngineOptions  # noqa: E402
from vibeqc.periodic.ccm.direct import run_ccm_rhf_direct, run_ccm_uhf_direct  # noqa: E402
from vibeqc.periodic.ccm.fit_source import MANTAFitSource  # noqa: E402
from vibeqc.periodic.ccm.neutral import ccm_neutral_cderi_blocks  # noqa: E402
from vibeqc.periodic.ccm.one_electron_source import (  # noqa: E402
    MANTA_ONE_ELECTRON_ENGINE,
    MANTAOneElectronSource,
)

CUTOFF_HA = 12.0
ENERGY_TOL_HA_PER_CELL = 1e-10
TIGHT = dict(conv_tol=1e-12, conv_tol_grad=1e-9, max_iter=300)


def _h2_cube():
    return PeriodicSystem(
        3, np.diag([6.0, 6.0, 6.0]),
        [Atom(1, [3.0, 3.0, 2.3]), Atom(1, [3.0, 3.0, 3.7])], 0, 1)


def _li_doublet():
    return PeriodicSystem(
        3, np.diag([7.0, 7.0, 7.0]), [Atom(3, [3.5, 3.5, 3.5])], 0, 2)


def _options(open_shell=False, auxiliary_cutoff_bohr=12.0, max_iter=300):
    o = manta.PeriodicRIUHFOptions() if open_shell else manta.PeriodicRIRHFOptions()
    o.pair_cutoff_bohr = o.fitting.pair_cutoff_bohr = 6.0
    o.fitting.auxiliary_cutoff_bohr = auxiliary_cutoff_bohr
    o.exchange_q0 = "BvK-ewald"
    o.scf.energy_change_tolerance = 1e-12
    o.scf.commutator_tolerance = 1e-9
    # MANTA admits the whole run's upper bound, which scales with this cap.
    o.scf.maximum_iterations = max_iter
    return o


def _budget():
    b = manta.ResourceBudget()
    for name in ("memory_bytes", "shell_calls", "primitive_quartets",
                 "scalar_terms", "kernel_terms"):
        setattr(b, name, 2 ** 62)
    return b


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


def _manta_reference(cell, nrep, open_shell=False, *, cutoff=CUTOFF_HA,
                     auxiliary_cutoff_bohr=12.0, max_iter=300):
    mol = cell.unit_cell_molecule()
    orb = BasisSet(mol, "sto-3g")
    aux = make_aux_basis_set(mol, aux_name=default_aux_for("sto-3g"), drop_eta=0.0)
    mesh = manta.RegularKMesh(nrep, (0, 0, 0))
    engine = IntegralEngineOptions(backend="manta", fallback="error", budget=_budget())
    run = (manta.run_periodic_ri_uhf_from_cutoffs if open_shell
           else manta.run_periodic_ri_rhf_from_cutoffs)
    ref = _quiet(run, orb, aux, cell, mesh, cutoff,
                 options=_options(open_shell, auxiliary_cutoff_bohr, max_iter),
                 integral_engine=engine)
    assert ref.converged
    return ref, mesh, int(orb.nbasis)


def _real_gamma(ccm, open_shell=False, *, cutoff=CUTOFF_HA,
                auxiliary_cutoff_bohr=12.0, max_iter=300):
    # The sources build MANTA's domains with the same options as the control.
    o = _options(open_shell, auxiliary_cutoff_bohr, max_iter)
    cderi = ccm_neutral_cderi_blocks(
        ccm, fit_source=MANTAFitSource(o, cutoff, budget=_budget()))
    oneel = MANTAOneElectronSource(o, cutoff, budget=_budget())
    run = run_ccm_uhf_direct if open_shell else run_ccm_rhf_direct
    return _quiet(run, ccm, cderi=cderi, oneelectron=oneel, **TIGHT)


def test_the_one_electron_torus_is_mantas():
    cell, nrep = _h2_cube(), (2, 1, 1)
    ref, mesh, n = _manta_reference(cell, nrep)
    ccm = CCMSystem(cell, nrep, "sto-3g")
    S, h, e_nn, record = MANTAOneElectronSource(
        _options(), CUTOFF_HA, budget=_budget()).torus(ccm)
    block = ccm_neutral_cderi_blocks(
        ccm, fit_source=MANTAFitSource(_options(), CUTOFF_HA, budget=_budget()))
    nk = block.n_c
    nvec = np.asarray(nrep)
    frac = np.array([mesh.fractional(mesh.address(i)) for i in range(nk)])
    manta_labels = [tuple(int(x) for x in np.round(f * nvec).astype(int) % nvec)
                    for f in frac]
    to_manta = [manta_labels.index(tuple(int(x) for x in block.frac[i] % nvec))
                for i in range(nk)]

    def torus(values):
        per_k = np.asarray(values).reshape(nk, n, n)
        return block.embed(block.unbloch_real(
            np.stack([per_k[to_manta[i]] for i in range(nk)])))

    np.testing.assert_allclose(S, torus(ref.overlap), atol=1e-14, rtol=0)
    np.testing.assert_allclose(h, torus(ref.hcore), atol=1e-14, rtol=0)
    rc = ref.receipt
    assert e_nn / nk == pytest.approx(rc.nuclear_energy, abs=1e-13)
    parts = record["nuclear_repulsion_per_cell"]
    assert parts["real"] == pytest.approx(rc.nuclear_real, abs=1e-13)
    assert parts["reciprocal"] == pytest.approx(rc.nuclear_reciprocal, abs=1e-13)
    assert parts["self"] == pytest.approx(rc.nuclear_self, abs=1e-13)
    assert parts["background"] == pytest.approx(rc.nuclear_background, abs=1e-13)
    assert record["engine"] == MANTA_ONE_ELECTRON_ENGINE


@pytest.mark.parametrize("nrep", [(2, 1, 1), (3, 1, 1)],
                         ids=lambda m: "x".join(map(str, m)))
def test_real_gamma_rhf_is_mantas_ri_rhf(nrep):
    cell = _h2_cube()
    ref, _, _ = _manta_reference(cell, nrep)
    ccm = CCMSystem(cell, nrep, "sto-3g")
    res = _real_gamma(ccm)
    assert res.converged
    assert abs(res.energy / ccm.n_cells - ref.receipt.energy) < ENERGY_TOL_HA_PER_CELL
    details = res.neutral_source.as_dict()["details"]
    assert details["one_electron"]["engine"] == MANTA_ONE_ELECTRON_ENGINE
    assert details["fit"]["engine"] == "caller"  # the block cderi was prebuilt
    assert res.fock_build is not None


def test_real_gamma_uhf_is_mantas_ri_uhf():
    """Li doublet at (1,1,1), where both spin conventions name one state.

    At N_c > 1 they differ: ``CCMSystem`` pairs the unpaired electrons of a
    doublet cell on an even-cell torus (the pinned historical convention, a
    supercell singlet at (2,1,1)), while MANTA's whole-mesh UHF keeps one
    unpaired electron per cell (a supercell triplet); measured 42 mHa/cell
    apart on this cell. Li's def2-SVP-JK auxiliary set needs images to 20
    bohr and a 30 Ha sphere before its fitting metric is positive definite
    (smallest eigenvalue -0.88 at 12 bohr / 12 Ha, +8.7e-5 at 20 / 30)."""
    cell, nrep = _li_doublet(), (1, 1, 1)
    kw = dict(cutoff=30.0, auxiliary_cutoff_bohr=20.0, max_iter=60)
    ref, _, _ = _manta_reference(cell, nrep, open_shell=True, **kw)
    ccm = CCMSystem(cell, nrep, "sto-3g")
    res = _real_gamma(ccm, open_shell=True, **kw)
    assert res.converged
    assert (res.n_alpha, res.n_beta) == (2, 1)
    assert abs(res.energy / ccm.n_cells - ref.receipt.energy) < ENERGY_TOL_HA_PER_CELL


def test_one_electron_source_refusals():
    ccm = CCMSystem(_h2_cube(), (2, 1, 1), "sto-3g")
    with pytest.raises(ValueError, match="reciprocal_energy_cutoff"):
        MANTAOneElectronSource(_options(), -1.0)
    with pytest.raises(TypeError, match="MANTA periodic RI HF options"):
        MANTAOneElectronSource(object(), CUTOFF_HA)
    from vibeqc import LatticeSumOptions

    with pytest.raises(ValueError, match="lat_opts"):
        _quiet(run_ccm_rhf_direct, ccm,
               oneelectron=MANTAOneElectronSource(_options(), CUTOFF_HA,
                                                  budget=_budget()),
               lat_opts=LatticeSumOptions())
