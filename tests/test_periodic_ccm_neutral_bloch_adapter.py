"""EXPERIMENTAL: the neutral Γ-CCM Bloch producer through ``run_periodic_job``.

Gates for :mod:`vibeqc.periodic.ccm.neutral_bloch_runner` and for the
``method="aiccm", variant="neutral-bloch"`` runner arm (milestone M1b of
the original investigation). Ruling R1: this and
``variant="real-gamma"`` are the two admissible producers of ONE neutral
finite-BvK-torus Hamiltonian, so the pins below are of three kinds:

* the adapter returns the production multi-k GDF result with a per-unit-cell
  energy (no fold, no ``N_c`` division) and a convention record, which is what
  keeps the Molden / density / QVF artefacts alive;
* the arm executes the same Hamiltonian as ``jk_method="gdf"`` on the same
  Gamma-centred mesh, and the same Hamiltonian as ``variant="real-gamma"``
  in the other representation (Theorem 1);
* the fail-closed surface, which is deliberately narrower than plain GDF: the
  producer refuses a cell declared 3-D whose transverse directions measure as
  pure vacuum, and a positive converged energy on a neutral cell (both
  IID 291), and the arm forwards no mixing control.
"""

from __future__ import annotations

import tomllib
import warnings
import zipfile

import numpy as np
import pytest

from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.neutral_bloch_runner import (
    CCMNeutralBlochConvention,
    run_neutral_bloch_scf,
)
from vibeqc.periodic.ccm.ri import run_ccm_rhf_gdf

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
    kw.setdefault("variant", "neutral-bloch")
    return _quiet(
        run_periodic_job, cell, basis, output=str(tmp_path / tag),
        progress=False, **kw)


# --- the adapter ------------------------------------------------------------


def test_adapter_returns_the_production_result_per_unit_cell():
    """No fold and no N_c division: the producer's energy is already per unit
    cell, and the object handed back is the multi-k GDF result the output
    stage understands, not the CCM wrapper (which carries no orbitals, no
    density and no k-point metadata, so returning it would silently drop the
    Molden, density-grid and DOS artefacts)."""
    cell = _h2_cell()
    raw = _quiet(run_neutral_bloch_scf, cell, "sto-3g", "RHF", (2, 1, 1),
                 progress=False)
    lib = _quiet(run_ccm_rhf_gdf, CCMSystem(cell, (2, 1, 1), "sto-3g"),
                 symmetry=False, progress=False)
    assert raw.converged and lib.converged
    assert raw.energy == pytest.approx(lib.energy, abs=1e-13)
    assert type(raw).__name__ == "PeriodicKRHFGDFResult"
    for field in ("mo_coeffs", "mo_energies", "occupations", "density",
                  "kpoints_cart", "kpoint_weights"):
        assert hasattr(raw, field), field
    convention = raw.neutral_bloch
    assert isinstance(convention, CCMNeutralBlochConvention)
    assert convention.route == "neutral-bloch"
    assert convention.exchange_q0 == "BvK-ewald"
    assert convention.exchange_q0_applicability == "active"
    assert convention.nrep == (2, 1, 1) and convention.n_cells == 2
    assert convention.pair_symmetry is False
    assert convention.lattice_vector_convention == "columns"
    assert convention.experimental is True
    assert raw.ccm_result is not None


def test_adapter_returns_spin_lattice_density_without_replacing_per_k():
    """The same producer-owned fold is available for both spin channels."""
    raw = _quiet(
        run_neutral_bloch_scf,
        _li_doublet(),
        "sto-3g",
        "UHF",
        (1, 1, 1),
        return_lattice_density=True,
        progress=False,
    )

    assert raw.converged
    assert isinstance(raw.density_alpha, list)
    assert isinstance(raw.density_beta, list)
    assert len(raw.density_alpha) == len(raw.density_beta) == 1
    assert raw.density_alpha_lattice is not raw.density_alpha
    assert raw.density_beta_lattice is not raw.density_beta
    assert len(raw.density_alpha_lattice.cells) > 1
    assert len(raw.density_beta_lattice.cells) > 1


def test_adapter_boundary():
    """Bad mesh and unsupported references fail loudly, naming the variant."""
    cell = _h2_cell()
    with pytest.raises(ValueError, match="mesh"):
        _quiet(run_neutral_bloch_scf, cell, "sto-3g", "RHF", (0, 1, 1))
    with pytest.raises(NotImplementedError, match="neutral-bloch"):
        _quiet(run_neutral_bloch_scf, cell, "sto-3g", "ROHF", (1, 1, 1))
    with pytest.raises(NotImplementedError, match="neutral-bloch"):
        _quiet(run_neutral_bloch_scf, cell, "sto-3g", "MP2", (1, 1, 1))
    with pytest.raises(ValueError, match="functional"):
        _quiet(run_neutral_bloch_scf, cell, "sto-3g", "RKS", (1, 1, 1))


# --- the runner arm ---------------------------------------------------------


def test_runner_neutral_bloch_is_the_multik_gdf_hamiltonian(tmp_path):
    """With the CCM pair-star reduction off, the arm executes the production
    multi-k GDF Hamiltonian on the torus mesh, so at a matched fit source it
    must agree with jk_method='gdf' on that mesh to machine precision. This
    is the identity that makes the arm's difference from plain GDF one of
    GUARDS and RECORDS, not of operator.

    Since #378 the arm pins ``fit_source="classic"`` (the fit the real-Γ fold
    builds; its identity with the real-Γ arm is the next test), while
    jk_method='gdf' keeps the driver's own default, the SR/LR source on 3-D
    cells: the two differ by 4.3e-10 Ha/cell on this H2 cell (-1.1945309121965
    against -1.1945309126270), which is the source, not the arm. So the arm is
    run here with an explicit ``fit_source="sr"``, which it honours over its
    pin."""
    nb = _job(tmp_path, "nb", aiccm_lattice_extension=(2, 1, 1),
              fit_source="sr")
    gdf = _job(tmp_path, "gdf", method="RHF", variant=None,
               jk_method="gdf", kpoints=(2, 1, 1))
    assert nb.converged and gdf.converged
    assert nb.energy == pytest.approx(gdf.energy, abs=1e-10)


@pytest.mark.parametrize("mesh", [(1, 1, 1), (2, 1, 1), (2, 2, 1)])
def test_runner_neutral_bloch_and_real_gamma_are_one_hamiltonian(tmp_path, mesh):
    """Theorem 1 through the public front door: the two producers of the
    neutral torus agree. The tree pins the library pair at 1e-8 Ha/cell
    (tests/test_ccm_route.py::test_neutral_bloch_and_real_gamma_agree_via_route_keyword);
    measured here at 1e-13 on three torus sizes, pinned at 1e-9 to keep a
    margin over that measurement without asserting machine precision."""
    nb = _job(tmp_path, f"nb{mesh}", aiccm_lattice_extension=mesh)
    rg = _job(tmp_path, f"rg{mesh}", variant="real-gamma",
              aiccm_lattice_extension=mesh, initial_guess="HCORE")
    assert nb.converged and rg.converged
    assert nb.energy == pytest.approx(rg.energy, abs=1e-9)


def test_runner_neutral_bloch_default_path_pins_classic_and_records_it(tmp_path):
    """#378 through the public front door with DEFAULTS: no ``rsgdf_ke_cutoff``
    and no tail, so the runner forwards its unresolved request (``None``, #257)
    and the producer must resolve it before sizing the fold's tail (a
    ``float(None)`` crash on the default path was found by the post-SCF cost
    chat on a trial rebase). Both arms record the executed fit source, which
    must be the classic pin on both, and the executed tail; the two energies
    agree as the Theorem-1 test above pins them."""
    import tomllib

    nb = _job(tmp_path, "nb_default", aiccm_lattice_extension=(2, 1, 1))
    rg = _job(tmp_path, "rg_default", variant="real-gamma",
              aiccm_lattice_extension=(2, 1, 1), initial_guess="HCORE")
    assert nb.converged and rg.converged
    assert nb.energy == pytest.approx(rg.energy, abs=1e-9)
    for tag in ("nb_default", "rg_default"):
        run = tomllib.loads(
            (tmp_path / f"{tag}.system").read_text(encoding="utf-8"))["run"]
        assert run["rsgdf_fit_source_executed"] == "classic", (tag, run.get("rsgdf_fit_source_executed"))
        assert run["rsgdf_tail_ke_cutoff_executed"] == "none"
    out = (tmp_path / "nb_default.out").read_text(encoding="utf-8")
    assert "fit_source          = classic" in out


def test_runner_forwards_an_explicit_tail_to_both_neutral_arms(tmp_path):
    """``rsgdf_tail_ke_cutoff`` reaches both neutral producers through the front
    door (2026-09-30; it used to fail closed for ``method="aiccm"``). On the H2
    cell the classifier sizes no tail, so an explicit 500 Ha shell is the only
    tail in play: both arms must record it as executed, and the two energies
    must still agree (one Hamiltonian, Theorem 1)."""
    import tomllib

    nb = _job(tmp_path, "nb_tail", aiccm_lattice_extension=(2, 1, 1),
              rsgdf_tail_ke_cutoff=500.0)
    rg = _job(tmp_path, "rg_tail", variant="real-gamma",
              aiccm_lattice_extension=(2, 1, 1), initial_guess="HCORE",
              rsgdf_tail_ke_cutoff=500.0)
    assert nb.converged and rg.converged
    assert nb.energy == pytest.approx(rg.energy, abs=1e-9)
    for tag in ("nb_tail", "rg_tail"):
        run = tomllib.loads(
            (tmp_path / f"{tag}.system").read_text(encoding="utf-8"))["run"]
        assert run["rsgdf_fit_source_executed"] == "classic", tag
        assert float(run["rsgdf_tail_ke_cutoff_executed"]) == pytest.approx(500.0), (
            tag, run.get("rsgdf_tail_ke_cutoff_executed"))


@pytest.mark.slow
def test_runner_neutral_bloch_default_path_tail_on_a_dense_core_cell(tmp_path):
    """#378, the tailed case through the front door with DEFAULTS: on MgO/STO-3G
    at the Gamma mesh (the tree's tight-core reference cell, fcc a = 7.956 bohr)
    the fold's tail is auto-sized from the base mesh, so the tail the neutral
    Bloch producer sizes after resolving ``rsgdf_ke_cutoff=None`` must be the
    one the classic fit runs on, 3291.6114 Ha, and both arms must record it
    (a tail sized on another base mesh would read differently here; on LiH,
    which needs no tail, the mismatch would be invisible)."""
    import tomllib

    a = 7.956
    lat = 0.5 * a * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]]).T
    cell = PeriodicSystem(3, lat, [Atom(12, [0, 0, 0]), Atom(8, [0.5 * a, 0.5 * a, 0.5 * a])], 0, 1)
    nb = _job(tmp_path, "nb_mgo", cell=cell, aiccm_lattice_extension=(1, 1, 1))
    rg = _job(tmp_path, "rg_mgo", cell=cell, variant="real-gamma",
              aiccm_lattice_extension=(1, 1, 1), initial_guess="HCORE")
    assert nb.converged and rg.converged
    assert nb.energy == pytest.approx(rg.energy, abs=1e-9)
    tails = {}
    for tag in ("nb_mgo", "rg_mgo"):
        run = tomllib.loads(
            (tmp_path / f"{tag}.system").read_text(encoding="utf-8"))["run"]
        assert run["rsgdf_fit_source_executed"] == "classic"
        tails[tag] = float(run["rsgdf_tail_ke_cutoff_executed"])
    assert tails["nb_mgo"] == tails["rg_mgo"]
    assert tails["nb_mgo"] == pytest.approx(3291.6114, abs=1e-3)


def test_runner_neutral_bloch_manifest_and_out(tmp_path):
    """The .system carries the RESULT-derived convention record and the
    front-door stamps; the .out names the torus and the route."""
    _job(tmp_path, "nb", aiccm_lattice_extension=(2, 1, 1))
    run = tomllib.loads(
        (tmp_path / "nb.system").read_text(encoding="utf-8"))["run"]
    assert run["method_status"] == "experimental"
    assert run["aiccm_variant"] == "neutral-bloch"
    assert run["aiccm_selector"] == "front-door"
    for key in ("jk_method_requested", "jk_method_resolved",
                "jk_method_executed"):
        assert run[key] == "neutral-bloch", key
    assert run["exchange_q0"] == "BvK-ewald"
    assert run["exchange_q0_applicability"] == "active"
    # Byte-identical to the real-Γ producer's label on purpose: ruling R1
    # says the two evaluate the same construction.
    assert run["ccm_construction"] == (
        "none (neutral fitted-torus representation control)")
    assert "Bloch representation" in run["evaluation_representation"]
    assert run["lattice_vector_convention"] == "columns"
    assert run["bvk_n_cells"] == 2
    assert run["lpq_pair_symmetry"] is False
    assert run["experimental_route"] is True
    out = (tmp_path / "nb.out").read_text(encoding="utf-8")
    assert "BvK torus (nrep)    = (2, 1, 1)" in out
    assert "route               = neutral-bloch" in out
    assert "J/K method: aiccm (neutral-bloch)" in out


def test_runner_neutral_bloch_qvf_carries_the_convention(tmp_path):
    """The QVF vendor record mirrors the real-Γ one so a reader can tell the
    two producers apart inside one archive."""
    _job(tmp_path, "nb", aiccm_lattice_extension=(1, 1, 1), output_qvf=True)
    with zipfile.ZipFile(tmp_path / "nb.qvf") as zf:
        blob = "\n".join(
            zf.read(name).decode("utf-8", "replace")
            for name in zf.namelist()
            if name.endswith(".json")
        )
    assert "neutral_bloch_convention" in blob
    assert "neutral-bloch" in blob


def test_runner_neutral_bloch_true_multik_writes_requested_density(tmp_path):
    """The public Bloch producer returns the exact lattice density needed by
    the strict density-artifact path, without replacing its per-k SCF state.

    This is the independently reproduced IID 645 closure hole: the SCF and
    the other multi-k writers completed, then finalisation raised because the
    adapter exposed only ``D(k)`` while the exact XSF writer deliberately
    refuses to invent ``D(g)``.  A genuine ``(2,2,2)`` run exercises that
    boundary and keeps the already-working Molden/QVF/DOS outputs live.
    """
    result = _job(
        tmp_path,
        "nb",
        aiccm_lattice_extension=(2, 2, 2),
        write_molden_file=True,
        write_density=True,
        density_spacing_bohr=0.5,
        dos_kmesh=(2, 2, 2),
        output_qvf=True,
    )

    assert result.converged
    assert isinstance(result.density, list)
    assert len(result.density) == 8
    for suffix in (".molden", ".xsf", ".qvf"):
        path = tmp_path / f"nb{suffix}"
        assert path.is_file() and path.stat().st_size > 0, suffix
    with zipfile.ZipFile(tmp_path / "nb.qvf") as zf:
        members = "\n".join(zf.namelist())
        assert "density" in members.lower()
        assert "dos" in members.lower()


def test_runner_neutral_bloch_open_shell_references(tmp_path):
    """The front door infers UHF / UKS from the multiplicity, and the arm has
    producers for both. Spin bookkeeping is per unit cell here (the multi-k
    convention), which is why the docs warn that an open-shell comparison
    against the real-Γ producer is convention-matched only at (1,1,1)."""
    li = _li_doublet()
    uhf = _job(tmp_path, "uhf", cell=li, aiccm_lattice_extension=(1, 1, 1))
    uks = _job(tmp_path, "uks", cell=li, functional="pbe",
               aiccm_lattice_extension=(1, 1, 1))
    assert uhf.converged and uks.converged
    assert uhf.energy < 0.0 and uks.energy < 0.0
    plan = tomllib.loads(
        (tmp_path / "uhf.system").read_text(encoding="utf-8"))["plan"]
    assert plan["method"] == "UHF"


def test_runner_neutral_bloch_fail_closed_surface(tmp_path):
    """The arm's boundary: no gradients, no DFT+U, no restart, dim 3 only, the
    exchange-q0 convention fixed, and every mixing control it would silently
    drop refused before SCF (CLAUDE.md section 7)."""
    from vibeqc import run_periodic_job
    from vibeqc._vibeqc_core import BasisSet
    from vibeqc.dft_plus_u import HubbardSite

    cell = _h2_cell()
    basis = BasisSet(cell.unit_cell_molecule(), "sto-3g")

    def job(**kw):
        return _quiet(
            run_periodic_job, kw.pop("cell", cell), kw.pop("basis", basis),
            method="aiccm", variant="neutral-bloch",
            output=str(tmp_path / "x"), dry_run=True, progress=False, **kw)

    with pytest.raises(NotImplementedError, match="geometry optimization"):
        job(aiccm_lattice_extension=(1, 1, 1), optimize=True)
    with pytest.raises(NotImplementedError, match="force constants"):
        job(aiccm_lattice_extension=(1, 1, 1), hessian=True)
    with pytest.raises(NotImplementedError, match=r"DFT\+U"):
        job(aiccm_lattice_extension=(1, 1, 1), functional="pbe",
            dft_plus_u=[HubbardSite(0, 0, 1.0)])
    with pytest.raises(ValueError, match="exchange-q0"):
        job(aiccm_lattice_extension=(1, 1, 1), exchange_exxdiv="none")
    with pytest.raises(NotImplementedError, match="restart"):
        job(aiccm_lattice_extension=(1, 1, 1), initial_guess="READ")
    for knob in ({"damping": 0.3}, {"fock_mixing": 0.3},
                 {"fmixing_percent": 30.0}, {"level_shift": 0.5},
                 {"density_mixer": "broyden"}):
        with pytest.raises(NotImplementedError, match="silently dropped"):
            job(aiccm_lattice_extension=(1, 1, 1), **knob)
    chain = PeriodicSystem(
        1, np.diag([6.0, 15.0, 15.0]),
        [Atom(1, [0.0, 7.5, 7.5]), Atom(1, [1.4, 7.5, 7.5])], 0, 1)
    chain_basis = BasisSet(chain.unit_cell_molecule(), "sto-3g")
    with pytest.raises(NotImplementedError, match="dim"):
        job(cell=chain, basis=chain_basis, aiccm_lattice_extension=(2, 1, 1))


def test_runner_neutral_bloch_producer_guards_are_not_plain_gdf(tmp_path):
    """The producer's own refusals reach the runner arm, which is why the arm
    is not an alias of jk_method='gdf': a cell DECLARED 3-D whose transverse
    directions carry only vacuum (measured, IID 291) fails closed here, while
    plain GDF runs it and reports a vacuum-padding artifact. The measurement
    needs a gap above the 25-bohr floor AND above the ratio to the shortest
    lattice vector, so this is a 1-D chain padded to 40 bohr transversely."""
    from vibeqc import run_periodic_job
    from vibeqc._vibeqc_core import BasisSet

    chain = PeriodicSystem(
        3, np.diag([2.8, 40.0, 40.0]),
        [Atom(1, [0.0, 20.0, 20.0]), Atom(1, [1.4, 20.0, 20.0])], 0, 1)
    basis = BasisSet(chain.unit_cell_molecule(), "sto-3g")
    with pytest.raises(NotImplementedError, match="vacuum padding") as exc:
        _quiet(run_periodic_job, chain, basis, method="aiccm",
               variant="neutral-bloch", aiccm_lattice_extension=(2, 1, 1),
               output=str(tmp_path / "vac"), progress=False)
    assert "IID 291" in str(exc.value)
    # Same refusal from the real-Gamma producer: one construction, one guard.
    with pytest.raises(NotImplementedError, match="vacuum padding"):
        _quiet(run_periodic_job, chain, basis, method="aiccm",
               variant="real-gamma", aiccm_lattice_extension=(2, 1, 1),
               initial_guess="HCORE", output=str(tmp_path / "vac_rg"),
               progress=False)


# --- ruling R6: the vacuum-padding opt-in ------------------------------------


def _padded_chain_cell():
    """A 1-D H2 chain along a1, declared 3-D with 30 bohr of transverse
    vacuum: the IID 291 repro class (gap above 25 bohr and above four
    shortest lattice vectors on a2 and a3)."""
    return PeriodicSystem(
        3, np.diag([6.0, 30.0, 30.0]),
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [1.4, 0.0, 0.0])], 0, 1)


def test_runner_vacuum_opt_in_is_refused_off_the_neutral_variants(tmp_path):
    """The keyword names a guard only the two neutral producers carry, so it
    fails closed on four-center and chi and with any other method rather
    than being silently ignored."""
    with pytest.raises(ValueError, match="neutral producers only"):
        _job(tmp_path, "fc", variant="four-center",
             aiccm_lattice_extension=(1, 1, 1),
             aiccm_allow_vacuum_padding=True)
    with pytest.raises(ValueError, match="require method='aiccm'"):
        _job(tmp_path, "plain", method="RHF", variant=None,
             jk_method="gdf", kpoints=(1, 1, 1),
             aiccm_allow_vacuum_padding=True)


def test_runner_neutral_bloch_records_the_vacuum_opt_in_negatively(tmp_path):
    """On a measured-3-D cell the manifest records the opt-in as false with
    an empty axis list: absence of a flag is never a verdict."""
    _job(tmp_path, "nb", aiccm_lattice_extension=(1, 1, 1))
    run = tomllib.loads(
        (tmp_path / "nb.system").read_text(encoding="utf-8"))["run"]
    assert run["aiccm_allow_vacuum_padding"] is False
    # list-valued [run] fields are TOML arrays (system_info._toml_value)
    assert run["aiccm_vacuum_padded_axes"] == []
    # O29: the convergence actually reached, read from the multi-k trace.
    assert run["scf_cycles_recorded"] >= 2
    assert run["scf_final_residual_metric"] == "gradient_norm"
    assert isinstance(run["scf_final_delta_e_ha"], float)
    assert isinstance(run["scf_final_residual"], float)


def test_runner_neutral_bloch_vacuum_opt_in_warns_and_stamps(tmp_path):
    """With the opt-in the padded chain runs, warns, prints the switch in the
    .out header and stamps the padded directions (1-based) in the manifest;
    without it the same cell is still refused."""
    from vibeqc import run_periodic_job
    from vibeqc._vibeqc_core import BasisSet
    from vibeqc.periodic.ccm.experimental import AICCMVacuumPaddingWarning

    cell = _padded_chain_cell()
    basis = BasisSet(cell.unit_cell_molecule(), "sto-3g")
    with pytest.raises(NotImplementedError, match="allow_vacuum_padding"):
        _quiet(run_periodic_job, cell, basis, output=str(tmp_path / "nb_ref"),
               progress=False, method="aiccm", variant="neutral-bloch",
               aiccm_lattice_extension=(2, 1, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.simplefilter("error", AICCMVacuumPaddingWarning)
        with pytest.raises(AICCMVacuumPaddingWarning, match="2, 3"):
            run_periodic_job(cell, basis, output=str(tmp_path / "nb_w"),
                             progress=False, method="aiccm",
                             variant="neutral-bloch",
                             aiccm_lattice_extension=(2, 1, 1),
                             aiccm_allow_vacuum_padding=True)
    r = _job(tmp_path, "nb_pad", cell=cell, aiccm_lattice_extension=(2, 1, 1),
             aiccm_allow_vacuum_padding=True)
    assert r.converged
    assert r.neutral_bloch.vacuum_padding_allowed is True
    assert r.neutral_bloch.vacuum_padded_axes == (2, 3)
    run = tomllib.loads(
        (tmp_path / "nb_pad.system").read_text(encoding="utf-8"))["run"]
    assert run["aiccm_allow_vacuum_padding"] is True
    assert run["aiccm_vacuum_padded_axes"] == [2, 3]
    out = (tmp_path / "nb_pad.out").read_text(encoding="utf-8")
    assert "allow_vacuum_padding = True" in out
