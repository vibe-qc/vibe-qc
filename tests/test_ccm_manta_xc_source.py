"""MANTA's semilocal XC on the real-Gamma torus (#519, RG-8c).

``MANTAXCSource`` evaluates MANTA's semilocal XC quadrature on the torus; with
``MANTAFitSource`` (RG-8a) and ``MANTAOneElectronSource`` (RG-8b) the
real-Gamma KS drivers evaluate the same finite Hamiltonian as MANTA's fitted
KS (``manta.run_periodic_ri_ks_from_cutoffs``) in the real supercell
representation, so their energies per cell must agree (Theorem 1 for the
MANTA source). Rows: the torus XC of a translation-invariant density equals
MANTA's XC on its Bloch blocks; the torus potential is the derivative of the
torus energy; RKS PBE on the H2 cube at (1,1,1), (2,1,1) and (3,1,1) and UKS
PBE on the Li doublet at (1,1,1) equal MANTA's energies; the XC record joins
the neutral-source identity; the refusals.

The finite source (pair cutoff 6 bohr, auxiliary images to 12 bohr, a 12 Ha
reciprocal sphere, XC images to 12 bohr, a 7-point Gauss-Legendre product
quadrature over the cell) is the one MANTA's own RI KS tests use; it is not
an accuracy setting. The MANTA budget is explicit and uncapped.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pytest

from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem

pytestmark = [pytest.mark.experimental, pytest.mark.aiccm]

manta = pytest.importorskip("vibeqc.manta")

import vibeqc as vq  # noqa: E402
from vibeqc._vibeqc_core import BasisSet  # noqa: E402
from vibeqc.aux_basis import default_aux_for, make_aux_basis_set  # noqa: E402
from vibeqc.manta_scf import IntegralEngineOptions  # noqa: E402
from vibeqc.periodic.ccm.direct import (  # noqa: E402
    run_ccm_rks_direct,
    run_ccm_uks_direct,
)
from vibeqc.periodic.ccm.fit_source import MANTAFitSource  # noqa: E402
from vibeqc.periodic.ccm.neutral import ccm_neutral_cderi_blocks  # noqa: E402
from vibeqc.periodic.ccm.one_electron_source import MANTAOneElectronSource  # noqa: E402
from vibeqc.periodic.ccm.xc_source import MANTA_XC_ENGINE, MANTAXCSource  # noqa: E402

CUTOFF_HA = 12.0
XC_REACH = 12.0
ENERGY_TOL_HA_PER_CELL = 1e-10
TIGHT = dict(conv_tol=1e-12, conv_tol_grad=1e-9, max_iter=300)


def _h2_cube():
    return PeriodicSystem(
        3, np.diag([6.0, 6.0, 6.0]),
        [Atom(1, [3.0, 3.0, 2.3]), Atom(1, [3.0, 3.0, 3.7])], 0, 1)


def _li_doublet():
    return PeriodicSystem(
        3, np.diag([7.0, 7.0, 7.0]), [Atom(3, [3.5, 3.5, 3.5])], 0, 2)


def _quadrature(cell, order=7):
    lattice = np.asarray(cell.lattice, dtype=float)
    x, w = np.polynomial.legendre.leggauss(order)
    idx = np.array(list(itertools.product(range(order), repeat=3)))
    points = np.ascontiguousarray(((x[idx] + 1.0) / 2.0) @ lattice.T)
    weights = np.prod(w[idx] / 2.0, axis=1) * abs(np.linalg.det(lattice))
    return points, weights


def _budget():
    b = manta.ResourceBudget()
    for name in ("memory_bytes", "shell_calls", "primitive_quartets",
                 "scalar_terms", "kernel_terms"):
        setattr(b, name, 2 ** 62)
    return b


def _ks_options(*, auxiliary_cutoff_bohr=12.0, max_iter=100):
    o = manta.PeriodicRIKSOptions()
    o.fitting.pair_cutoff_bohr = 6.0
    o.ks.pair_cutoff_bohr = 6.0
    o.fitting.auxiliary_cutoff_bohr = auxiliary_cutoff_bohr
    o.ks.scf.energy_change_tolerance = 1e-12
    o.ks.scf.commutator_tolerance = 1e-9
    # MANTA admits the whole run's upper bound, which scales with this cap.
    o.ks.scf.maximum_iterations = max_iter
    return o


def _hf_twin(ks, open_shell=False):
    """The RI HF options whose domains and one-electron side equal the KS run's."""
    o = manta.PeriodicRIUHFOptions() if open_shell else manta.PeriodicRIRHFOptions()
    o.fitting = ks.fitting
    o.ri = ks.ri
    for name in ("alpha", "nuclear_cutoff_bohr", "pair_cutoff_bohr",
                 "reciprocal_batch"):
        setattr(o, name, getattr(ks.ks, name))
    o.scf.maximum_iterations = ks.ks.scf.maximum_iterations
    return o


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


def _sources(cell, open_shell=False, *, cutoff=CUTOFF_HA, auxiliary_cutoff_bohr=12.0,
             max_iter=100):
    ks = _ks_options(auxiliary_cutoff_bohr=auxiliary_cutoff_bohr, max_iter=max_iter)
    hf = _hf_twin(ks, open_shell)
    points, weights = _quadrature(cell)
    return (ks, MANTAFitSource(hf, cutoff, budget=_budget()),
            MANTAOneElectronSource(hf, cutoff, budget=_budget()),
            MANTAXCSource(ks, cutoff, XC_REACH, points, weights, budget=_budget()))


def _manta_reference(cell, nrep, functional, *, cutoff=CUTOFF_HA, ks=None):
    mol = cell.unit_cell_molecule()
    orb = BasisSet(mol, "sto-3g")
    aux = make_aux_basis_set(mol, aux_name=default_aux_for("sto-3g"), drop_eta=0.0)
    mesh = manta.RegularKMesh(nrep, (0, 0, 0))
    points, weights = _quadrature(cell)
    engine = IntegralEngineOptions(backend="manta", fallback="error", budget=_budget())
    ref = _quiet(manta.run_periodic_ri_ks_from_cutoffs, orb, aux, cell, mesh,
                 functional, cutoff, XC_REACH, points, weights,
                 options=ks if ks is not None else _ks_options(), integral_engine=engine)
    assert ref.converged
    return ref


@pytest.mark.parametrize("nrep", [(1, 1, 1), (2, 1, 1), (3, 1, 1)],
                         ids=lambda m: "x".join(map(str, m)))
def test_real_gamma_rks_is_mantas_ri_ks(nrep):
    cell = _h2_cube()
    ks, fit, oneel, xc = _sources(cell)
    ref = _manta_reference(cell, nrep, vq.Functional("pbe", 1), ks=ks)
    ccm = CCMSystem(cell, nrep, "sto-3g")
    cderi = ccm_neutral_cderi_blocks(ccm, fit_source=fit)
    res = _quiet(run_ccm_rks_direct, ccm, "pbe", cderi=cderi, oneelectron=oneel,
                 xc_source=xc, **TIGHT)
    assert res.converged
    assert abs(res.energy / ccm.n_cells - ref.receipt.energy) < ENERGY_TOL_HA_PER_CELL
    details = res.neutral_source.as_dict()["details"]
    assert details["xc"]["source"]["engine"] == MANTA_XC_ENGINE
    assert details["one_electron"]["engine"] == oneel.engine
    assert res.fock_build is not None


def test_the_torus_potential_is_the_derivative_of_the_torus_energy():
    cell = _h2_cube()
    ccm = CCMSystem(cell, (3, 1, 1), "sto-3g")
    _, fit, oneel, xc = _sources(cell)
    cderi = ccm_neutral_cderi_blocks(ccm, fit_source=fit)
    res = _quiet(run_ccm_rks_direct, ccm, "pbe", cderi=cderi, oneelectron=oneel,
                 xc_source=xc, **TIGHT)
    xc_of_D = xc.closed_shell(ccm, "pbe")
    D = res.density
    e0, V = xc_of_D(D)
    rng = np.random.default_rng(8)
    # A translation-invariant symmetric direction: the circulant projection.
    from vibeqc.periodic.ccm.stability import circulant_projection

    X = rng.standard_normal(D.shape)
    X = circulant_projection(ccm.nrep, 0.5 * (X + X.T))
    h = 1e-5
    ep, _ = xc_of_D(D + h * X)
    em, _ = xc_of_D(D - h * X)
    assert (ep - em) / (2.0 * h) == pytest.approx(float(np.sum(V * X)), rel=1e-6, abs=1e-9)


def test_real_gamma_uks_is_mantas_ri_ks():
    """Li doublet at (1,1,1), where both spin conventions name one state."""
    cell = _li_doublet()
    kw = dict(cutoff=30.0, auxiliary_cutoff_bohr=20.0, max_iter=60)
    ks, fit, oneel, xc = _sources(cell, open_shell=True, **kw)
    ref = _manta_reference(cell, (1, 1, 1), vq.Functional("pbe", 2),
                           cutoff=30.0, ks=ks)
    ccm = CCMSystem(cell, (1, 1, 1), "sto-3g")
    cderi = ccm_neutral_cderi_blocks(ccm, fit_source=fit)
    res = _quiet(run_ccm_uks_direct, ccm, "pbe", cderi=cderi, oneelectron=oneel,
                 xc_source=xc, **TIGHT)
    assert res.converged
    assert abs(res.energy / ccm.n_cells - ref.receipt.energy) < ENERGY_TOL_HA_PER_CELL


def test_xc_source_refusals():
    cell = _h2_cube()
    points, weights = _quadrature(cell)
    with pytest.raises(ValueError, match="reciprocal_energy_cutoff"):
        MANTAXCSource(_ks_options(), -1.0, XC_REACH, points, weights)
    with pytest.raises(ValueError, match="xc_image_cutoff_bohr"):
        MANTAXCSource(_ks_options(), CUTOFF_HA, 0.0, points, weights)
    with pytest.raises(TypeError, match="RI KS options"):
        MANTAXCSource(object(), CUTOFF_HA, XC_REACH, points, weights)
    with pytest.raises(ValueError, match="points"):
        MANTAXCSource(_ks_options(), CUTOFF_HA, XC_REACH, points[:, :2], weights)
    ccm = CCMSystem(cell, (1, 1, 1), "sto-3g")
    xc = MANTAXCSource(_ks_options(), CUTOFF_HA, XC_REACH, points, weights,
                       budget=_budget())
    with pytest.raises(NotImplementedError, match="semilocal"):
        xc.closed_shell(ccm, "pbe0")
    from vibeqc import LatticeSumOptions

    hf = _hf_twin(_ks_options())
    with pytest.raises(ValueError, match="lat_opts"):
        _quiet(run_ccm_rks_direct, ccm, "pbe",
               oneelectron=MANTAOneElectronSource(hf, CUTOFF_HA, budget=_budget()),
               lat_opts=LatticeSumOptions())
