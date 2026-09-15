"""EXPERIMENTAL: the union-and-weight four-centre Γ-CCM through the runner.

Gates for :mod:`vibeqc.periodic.ccm.four_center_runner` and the
``method="aiccm", variant="four-center"`` arm (milestone M3a of
``handovers/HANDOVER_AICCM_STANDARD_METHOD.md``). This is the construction the
line is named after, and ruling R1 keeps it distinct from the two neutral
producers, so the pins here are deliberately of a different kind from the
neutral-Bloch file's: there is NO cross-variant energy agreement to assert. A
difference between this variant and ``real-gamma`` is a construction
difference (D74, D89), and a test that asserted equality across that boundary
would be asserting the paper's conclusion rather than the code's contract.

What is pinned instead: the per-unit-cell normalisation (measured, because the
library result is per SUPERCELL and two of the three result dataclasses used to
claim otherwise), the weighting actually executed, the records, and the
fail-closed envelope.
"""

from __future__ import annotations

import tomllib
import warnings

import numpy as np
import pytest

from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.four_center_runner import (
    FOUR_CENTRE_WEIGHTING,
    CCMFourCentreConvention,
    run_four_center_scf,
)
from vibeqc.periodic.ccm.scf import run_ccm_rhf_scalable

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane


def _h2_cell():
    return PeriodicSystem(
        3, np.diag([6.0, 6.0, 6.0]),
        [Atom(1, [3.0, 3.0, 2.3]), Atom(1, [3.0, 3.0, 3.7])], 0, 1)


def _li_doublet():
    return PeriodicSystem(
        3, np.diag([7.0, 7.0, 7.0]), [Atom(3, [3.5, 3.5, 3.5])], 0, 2)


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


def _job(tmp_path, tag, **kw):
    from vibeqc import run_periodic_job
    from vibeqc._vibeqc_core import BasisSet

    cell = kw.pop("cell", None) or _h2_cell()
    basis = BasisSet(cell.unit_cell_molecule(), "sto-3g")
    kw.setdefault("method", "aiccm")
    kw.setdefault("variant", "four-center")
    kw.setdefault("initial_guess", "HCORE")
    return _quiet(
        run_periodic_job, cell, basis, output=str(tmp_path / tag),
        progress=False, **kw)


# --- the adapter ------------------------------------------------------------


def test_external_default_lattice_domain_is_shared_with_density_fold(monkeypatch):
    """Direct callers must not fold a narrower density than the XC domain."""
    from types import SimpleNamespace

    from vibeqc import define_external_functional
    from vibeqc.periodic.ccm import dft as ccm_dft
    from vibeqc.periodic.ccm import four_center_runner

    def unused_provider(_features):
        raise AssertionError("the patched CCM driver must not evaluate XC")

    name = "test-four-center-default-lattice-domain"
    define_external_functional(name, unused_provider)
    cell = _h2_cell()
    captured = {}
    density_sentinel = object()

    def fake_rks(ccm, functional, **kwargs):
        captured["xc_lattice_options"] = kwargs["xc_lattice_options"]
        nbf = int(ccm.basis.nbasis)
        zeros = np.zeros((nbf, nbf))
        return SimpleNamespace(
            converged=True,
            n_iter=1,
            energy=-1.0,
            density=zeros,
            mo_energies=np.zeros(nbf),
            mo_coeffs=np.eye(nbf),
            overlap=np.eye(nbf),
            fock=zeros,
            hcore=zeros,
            e_xc=-0.1,
            e_hf_exchange=0.0,
        )

    def fake_fold(_ccm, _density, lat_opts=None):
        captured.setdefault("fold_lattice_options", []).append(lat_opts)
        return density_sentinel

    monkeypatch.setattr(ccm_dft, "run_ccm_rks", fake_rks)
    monkeypatch.setattr(
        four_center_runner,
        "_fold_density_to_lattice_set",
        fake_fold,
    )

    result = four_center_runner.run_four_center_scf(
        cell,
        "sto-3g",
        "RKS",
        (1, 1, 1),
        functional=name,
    )

    xc_options = captured["xc_lattice_options"]
    assert xc_options is captured["fold_lattice_options"][0]
    assert float(xc_options.cutoff_bohr) == pytest.approx(25.0)
    assert result.density is density_sentinel


@pytest.mark.parametrize("mesh", [(1, 1, 1), (2, 1, 1)])
def test_adapter_divides_the_total_cyclic_energy_by_n_cells(mesh):
    """The library drivers return the TOTAL cyclic-cluster energy, not a
    per-cell one. Measured rather than read: at (2,1,1) the driver's energy is
    twice its (1,1,1) value, and the CCMKSResult / CCMUHFResult field comments
    claimed "per reference cell" until this milestone corrected them. Getting
    this wrong is an error of exactly N_c, which looks like a plausible
    binding energy rather than like a bug."""
    cell = _h2_cell()
    ccm = CCMSystem(cell, mesh, "sto-3g")
    adapted = _quiet(run_four_center_scf, cell, "sto-3g", "RHF", mesh)
    library = _quiet(run_ccm_rhf_scalable, ccm, method=FOUR_CENTRE_WEIGHTING)
    assert adapted.converged and library.converged
    assert adapted.energy == pytest.approx(
        library.energy / ccm.n_cells, abs=1e-12)
    if ccm.n_cells > 1:
        # The division is real, not a no-op the test cannot distinguish.
        assert abs(adapted.energy - library.energy) > 0.5


def test_adapter_folds_the_density_and_keeps_supercell_orbitals():
    """Same split as the real-Γ sibling: the density is folded to unit-cell
    lattice blocks so the artefacts are per unit cell, while the orbitals stay
    supercell-wide and are kept out of unit-cell artefacts by the #655 gate."""
    from vibeqc._vibeqc_core import BasisSet

    cell = _h2_cell()
    n_mu = int(BasisSet(cell.unit_cell_molecule(), "sto-3g").nbasis)
    result = _quiet(run_four_center_scf, cell, "sto-3g", "RHF", (2, 1, 1))
    assert np.asarray(result.mo_coeffs).shape == (2 * n_mu, 2 * n_mu)
    blocks = result.density.blocks
    assert np.asarray(blocks[0]).shape == (n_mu, n_mu)
    convention = result.four_center
    assert isinstance(convention, CCMFourCentreConvention)
    assert convention.route == "four-center"
    assert convention.weighting == FOUR_CENTRE_WEIGHTING != "union12"
    assert convention.n_cells == 2 and convention.nrep == (2, 1, 1)
    # The construction label must NOT be the neutral producers' string.
    assert "union-and-weight" in convention.ccm_construction
    assert "representation control" not in convention.ccm_construction
    assert convention.exchange_q0.startswith("not-applicable")


def test_adapter_boundary():
    cell = _h2_cell()
    with pytest.raises(ValueError, match="mesh"):
        _quiet(run_four_center_scf, cell, "sto-3g", "RHF", (0, 1, 1))
    with pytest.raises(NotImplementedError, match="four-center"):
        _quiet(run_four_center_scf, cell, "sto-3g", "MP2", (1, 1, 1))
    with pytest.raises(ValueError, match="functional"):
        _quiet(run_four_center_scf, cell, "sto-3g", "RKS", (1, 1, 1))


def test_adapter_runs_every_wired_reference():
    """RHF, RKS, UHF and UKS all reach a converged result. HF goes through the
    scalable builder (D-6b); UHF has no scalable entry, so it records the
    dense one rather than pretending otherwise."""
    cell, li = _h2_cell(), _li_doublet()
    rhf = _quiet(run_four_center_scf, cell, "sto-3g", "RHF", (1, 1, 1))
    rks = _quiet(run_four_center_scf, cell, "sto-3g", "RKS", (1, 1, 1),
                 functional="pbe")
    uhf = _quiet(run_four_center_scf, li, "sto-3g", "UHF", (1, 1, 1))
    uks = _quiet(run_four_center_scf, li, "sto-3g", "UKS", (1, 1, 1),
                 functional="pbe")
    for result in (rhf, rks, uhf, uks):
        assert result.converged and result.energy < 0.0
    assert rks.e_xc is not None and rhf.e_xc is None
    assert uhf.four_center.builder == "dense-python-padded"
    assert rhf.four_center.builder == "scalable-cxx-lattice-sum-jk"
    assert uhf.density_alpha is not None and uhf.density_beta is not None


# --- the runner arm ---------------------------------------------------------


def test_runner_four_center_matches_the_library_driver(tmp_path):
    result = _job(tmp_path, "fc", aiccm_lattice_extension=(2, 1, 1))
    library = _quiet(
        run_ccm_rhf_scalable,
        CCMSystem(_h2_cell(), (2, 1, 1), "sto-3g"),
        method=FOUR_CENTRE_WEIGHTING,
    )
    assert result.converged
    assert result.energy == pytest.approx(library.energy / 2.0, abs=1e-12)


def test_runner_four_center_records_what_it_built(tmp_path):
    """The manifest records the CONSTRUCTION, not a representation, and the
    weighting actually executed, because the library default (union12) is not
    what the front door runs."""
    _job(tmp_path, "fc", aiccm_lattice_extension=(2, 1, 1))
    run = tomllib.loads(
        (tmp_path / "fc.system").read_text(encoding="utf-8"))["run"]
    assert run["method_status"] == "experimental"
    assert run["aiccm_variant"] == "four-center"
    assert run["jk_method_executed"] == "aiccm2026dev-a"
    assert "union-and-weight" in run["ccm_construction"]
    assert "representation control" not in run["ccm_construction"]
    assert run["ccm_four_centre_weighting"] == FOUR_CENTRE_WEIGHTING
    assert run["ccm_four_centre_builder"] == "scalable-cxx-lattice-sum-jk"
    assert run["exchange_q0"].startswith("not-applicable")
    assert run["bvk_n_cells"] == 2
    assert run["experimental_route"] is True
    out = (tmp_path / "fc.out").read_text(encoding="utf-8")
    assert "cyclic cluster (nrep) = (2, 1, 1)" in out
    assert "route               = four-center" in out
    assert "union12 is not reachable here" in out


def test_runner_four_center_never_executes_union12(tmp_path):
    """GitLab #242: the union12 supermatrix carries a negative subspace on any
    basis with more than one function per centre. The library drivers still
    default to it; the front door must not, and there is no keyword through
    which a runner caller could ask for it."""
    import inspect

    from vibeqc.periodic_runner import run_periodic_job

    assert "union12" not in inspect.signature(run_periodic_job).parameters
    _job(tmp_path, "fc", aiccm_lattice_extension=(1, 1, 1))
    run = tomllib.loads(
        (tmp_path / "fc.system").read_text(encoding="utf-8"))["run"]
    assert run["ccm_four_centre_weighting"] != "union12"


def test_runner_four_center_fail_closed_surface(tmp_path):
    from vibeqc import run_periodic_job
    from vibeqc._vibeqc_core import BasisSet
    from vibeqc.dft_plus_u import HubbardSite

    cell = _h2_cell()
    basis = BasisSet(cell.unit_cell_molecule(), "sto-3g")

    def job(**kw):
        kw.setdefault("initial_guess", "HCORE")
        return _quiet(
            run_periodic_job, kw.pop("cell", cell), kw.pop("basis", basis),
            method="aiccm", variant="four-center",
            output=str(tmp_path / "x"), dry_run=True, progress=False, **kw)

    with pytest.raises(NotImplementedError, match="geometry optimization"):
        job(aiccm_lattice_extension=(1, 1, 1), optimize=True)
    with pytest.raises(NotImplementedError, match="force constants"):
        job(aiccm_lattice_extension=(1, 1, 1), hessian=True)
    with pytest.raises(NotImplementedError, match=r"DFT\+U"):
        job(aiccm_lattice_extension=(1, 1, 1), functional="pbe",
            dft_plus_u=[HubbardSite(0, 0, 1.0)])
    for knob in ({"damping": 0.3}, {"level_shift": 0.5},
                 {"fock_mixing": 0.3}, {"density_mixer": "broyden"}):
        with pytest.raises(NotImplementedError, match="silently dropped"):
            job(aiccm_lattice_extension=(1, 1, 1), **knob)
    chain = PeriodicSystem(
        1, np.diag([6.0, 15.0, 15.0]),
        [Atom(1, [0.0, 7.5, 7.5]), Atom(1, [1.4, 7.5, 7.5])], 0, 1)
    chain_basis = BasisSet(chain.unit_cell_molecule(), "sto-3g")
    with pytest.raises(NotImplementedError, match="dim"):
        job(cell=chain, basis=chain_basis, aiccm_lattice_extension=(2, 1, 1))


def test_runner_four_center_keeps_supercell_orbitals_out_of_artefacts(tmp_path):
    """The #655 invariant, re-pinned for the third producer that returns
    supercell-wide orbitals: no unit-cell artefact carries them."""
    import zipfile

    from vibeqc.periodic_runner import _sidecar_mo_ao_dimension

    result = _job(tmp_path, "fc", aiccm_lattice_extension=(2, 1, 1),
                  output_qvf=True)
    assert _sidecar_mo_ao_dimension(result) == 4  # 2 cells x 2 functions
    with zipfile.ZipFile(tmp_path / "fc.qvf") as archive:
        assert "wavefunction/mo_metadata.json" not in set(archive.namelist())
    assert not (tmp_path / "fc.molden").exists()
    assert "QVF wavefunction    = omitted" in (
        tmp_path / "fc.out").read_text(encoding="utf-8")
