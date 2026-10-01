"""One decision for the GDF engine, recorded in the output.

Row 3 of the 2026-09-29 cluster audit (#54, #297, #301, #395, #400): a bulk
GDF job has one Hamiltonian and several evaluators. ``resolve_gdf_route``
decides the evaluator once; the runner's convergence filter, its dispatch
and its manifest read that decision, and the finished result must report
the engine the decision chose. These tests pin the decision table, the
record, and the two cross-engine comparisons the audit found untested.
"""

from __future__ import annotations

import tomllib

import numpy as np
import pytest

import vibeqc as vq
from vibeqc.pbc_gdf import run_pbc_gdf_uhf, run_pbc_gdf_uks
from vibeqc.periodic.exchange_convention import BVK_EWALD, MOLECULAR_LIMIT
from vibeqc.periodic_convergence_auto import KnobResolution
from vibeqc.periodic_gdf_route import (
    EXACT_GAMMA,
    K_MESH,
    LEGACY_GAMMA,
    check_executed_engine,
    executed_engine,
    resolve_gdf_route,
)
from vibeqc.periodic_k_gdf import (
    run_krhf_periodic_gdf,
    run_krks_periodic_gdf,
    run_kuhf_periodic_gdf,
)

ROUTE_TOL = 1e-9


def _knobs(**kw):
    """``ConvergenceStrategy.knobs``-shaped mapping; ``name=(value, source)``."""
    out = {}
    for name in ("smearing_temperature", "fock_mixing", "level_shift"):
        value, source = kw.get(name, (0.0, "default"))
        out[name] = KnobResolution(float(value), source, "test")
    return out


def _decide(**overrides):
    base = dict(
        method="RHF", dim=3, n_electrons=20, multiplicity=1, nuclear_charge=20.0,
        ecp_active=False, kpoints=None, gdf_method=None, uses_external_xc=False,
        knobs=_knobs(),
    )
    base.update(overrides)
    return resolve_gdf_route(**base)


# --- the decision table -------------------------------------------------------


def test_plain_closed_shell_gamma_takes_the_exact_gamma_driver():
    d = _decide()
    assert (d.engine, d.driver, d.kmesh) == (EXACT_GAMMA, "run_pbc_gdf_rhf", None)
    assert d.reasons == () and d.exchange_q0 == BVK_EWALD and d.auto_zeroed == ()


def test_auto_fock_mixing_yields_to_the_exact_gamma_driver_and_says_why():
    d = _decide(knobs=_knobs(fock_mixing=(0.3, "auto")))
    assert d.engine == EXACT_GAMMA
    assert [k for k, _ in d.auto_zeroed] == ["fock_mixing"]
    assert "run_pbc_gdf_rhf" in d.auto_zeroed[0][1]
    assert d.auto_zeroed[0][1].startswith("capability-filtered to zero")


def test_explicit_knob_moves_the_job_and_keeps_the_auto_knob():
    d = _decide(knobs=_knobs(level_shift=(0.2, "explicit"), fock_mixing=(0.3, "auto")))
    assert (d.engine, d.driver, d.kmesh) == (K_MESH, "run_krhf_periodic_gdf", (1, 1, 1))
    assert d.reasons == ("level shift",) and d.auto_zeroed == ()
    assert {"fock_mixing", "level_shift"} <= set(d.honoured)


def test_auto_smearing_moves_the_engine_and_is_never_dropped():
    d = _decide(method="RKS", knobs=_knobs(smearing_temperature=(0.005, "auto")))
    assert (d.engine, d.driver) == (K_MESH, "run_krks_periodic_gdf")
    assert d.reasons == ("smearing",) and d.auto_zeroed == ()


def test_ecp_cell_runs_the_k_mesh_engine_with_its_auto_aids():
    d = _decide(ecp_active=True,
                knobs=_knobs(level_shift=(0.2, "auto"), fock_mixing=(0.5, "auto")))
    assert d.engine == K_MESH and d.reasons == ("effective-core potentials",)
    assert d.auto_zeroed == ()


def test_charged_cell_and_explicit_mesh_run_the_k_mesh_engine():
    assert _decide(nuclear_charge=19.0).reasons == ("charged cell",)
    d = _decide(kpoints=(2, 1, 1))
    assert d.engine == K_MESH and d.kmesh == (2, 1, 1)


def test_rks_keeps_fock_mixing_on_the_exact_gamma_uks_driver_only():
    d = _decide(method="RKS", knobs=_knobs(fock_mixing=(0.2, "explicit")))
    assert (d.engine, d.driver) == (EXACT_GAMMA, "run_pbc_gdf_uks")
    d = _decide(method="RKS", knobs=_knobs(level_shift=(0.2, "explicit")))
    assert (d.engine, d.driver) == (K_MESH, "run_krks_periodic_gdf")


def test_open_shell_auto_knobs_are_dropped_where_no_driver_implements_them():
    d = _decide(method="UHF", n_electrons=21, multiplicity=2, nuclear_charge=21.0,
                knobs=_knobs(level_shift=(0.2, "auto"), fock_mixing=(0.5, "auto")))
    assert (d.engine, d.driver) == (EXACT_GAMMA, "run_pbc_gdf_uhf")
    assert [k for k, _ in d.auto_zeroed] == ["level_shift"]
    d = _decide(method="UKS", kpoints=(2, 1, 1),
                knobs=_knobs(fock_mixing=(0.5, "auto")))
    assert d.driver == "run_kuks_periodic_gdf"
    assert [k for k, _ in d.auto_zeroed] == ["fock_mixing"]


def test_a_chain_takes_the_legacy_driver_and_declares_the_molecular_limit():
    d = _decide(dim=1)
    assert (d.engine, d.driver, d.exchange_q0) == (
        LEGACY_GAMMA, "run_rhf_periodic_gamma_gdf", MOLECULAR_LIMIT)


def test_executed_engine_is_read_from_the_backend_label():
    assert executed_engine("pbc-gdf-rsgdf") == EXACT_GAMMA
    assert executed_engine("pbc-gdf-mdf-uks+PARITY_HELD") == EXACT_GAMMA
    assert executed_engine("native-multi-k-gdf-gdf-rhf") == K_MESH
    assert executed_engine("native-gamma-gdf-via-k-gdf") == K_MESH
    assert executed_engine("ewald-jk-fallback") == LEGACY_GAMMA
    assert executed_engine("bipole") is None
    with pytest.raises(RuntimeError, match="route invariant"):
        check_executed_engine(_decide(), "native-multi-k-gdf-gdf-rhf", where="test")


# --- the record ----------------------------------------------------------------


def _h2_cube6():
    system = vq.PeriodicSystem(
        3, np.eye(3) * 6.0, [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [1.4, 0, 0])], 0, 1)
    return system, vq.BasisSet(system.unit_cell_molecule(), "sto-3g")


def _job(system, basis, tmp_path, tag, **kwargs):
    return vq.run_periodic_job(
        system, basis, jk_method="gdf", output=str(tmp_path / tag),
        output_qvf=False, write_molden_file=False, write_xyz_file=False,
        write_poscar_file=False, write_xsf_structure_file=False,
        write_cif_file=False, write_population_file=False, citations=False,
        **kwargs)


def _run_record(tmp_path, tag):
    with open(tmp_path / f"{tag}.system", "rb") as fh:
        return tomllib.load(fh)["run"]


@pytest.mark.parametrize("tag, kwargs, engine, driver, reason, gauge", [
    ("plain", {}, EXACT_GAMMA, "run_pbc_gdf_rhf", "", BVK_EWALD),
    ("shift", {"level_shift": 0.2}, K_MESH, "run_krhf_periodic_gdf", "level shift", BVK_EWALD),
    ("k111", {"kpoints": (1, 1, 1)}, K_MESH, "run_krhf_periodic_gdf", "explicit k mesh", BVK_EWALD),
])
def test_manifest_records_the_engine_the_gauge_and_the_reason(
    tmp_path, tag, kwargs, engine, driver, reason, gauge,
):
    system, basis = _h2_cube6()
    result = _job(system, basis, tmp_path, tag, method="RHF", **kwargs)
    assert result.converged
    run = _run_record(tmp_path, tag)
    assert run["gdf_engine"] == engine
    assert run["gdf_engine_planned"] == engine
    assert run["gdf_driver"] == driver
    assert reason in run["gdf_route_reason"]
    assert run["exchange_q0"] == gauge
    assert executed_engine(run["backend"]) == engine


def test_manifest_records_the_legacy_engine_on_a_chain(tmp_path):
    system = vq.PeriodicSystem(1, np.diag([3.0, 12.0, 12.0]),
                             [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [1.4, 0, 0])])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    _job(system, basis, tmp_path, "chain", method="RHF")
    run = _run_record(tmp_path, "chain")
    assert run["gdf_engine"] == LEGACY_GAMMA
    assert run["gdf_driver"] == "run_rhf_periodic_gamma_gdf"
    assert run["exchange_q0"] == MOLECULAR_LIMIT


def test_k_mesh_results_declare_their_gauge():
    system, basis = _h2_cube6()
    assert run_krhf_periodic_gdf(system, basis, (1, 1, 1), progress=False).exchange_q0 == BVK_EWALD
    assert run_kuhf_periodic_gdf(system, basis, (1, 1, 1), progress=False).exchange_q0 == BVK_EWALD


# --- the comparisons the audit found untested ------------------------------------


MDF_ROUTE_SPLIT = (
    "#416 (row 3 finding, 2026-09-29): with exact exchange the multi-k MDF "
    "loop at Nk=1 sits 7.8e-8 Ha from the exact-Gamma MDF on H2/STO-3G in a "
    "6-bohr cube (PySCF MDF: +9.5e-8 vs +1.7e-8); compcell agrees to 1e-12 "
    "on the same legs, and the pure-DFT leg agrees since 6cd4c3a. The MDF "
    "builders belong to the #172 lane; this strict xfail turns into a "
    "failure once the two evaluations agree."
)


@pytest.mark.parametrize("gdf_method", [
    "compcell",
    pytest.param("mdf", marks=pytest.mark.xfail(strict=True, reason=MDF_ROUTE_SPLIT)),
])
def test_rhf_and_uhf_singlet_agree_at_gamma_on_the_delegating_fits(gdf_method):
    """#54's shape on the non-default fits: run_krhf at (1,1,1) delegates
    mdf/compcell to the exact-Gamma driver, run_kuhf never delegates, so the
    two calls run different engines of the same fitted Hamiltonian."""
    system, basis = _h2_cube6()
    rhf = run_krhf_periodic_gdf(system, basis, (1, 1, 1), gdf_method=gdf_method, progress=False)
    uhf = run_kuhf_periodic_gdf(system, basis, (1, 1, 1), gdf_method=gdf_method, progress=False)
    assert rhf.converged and uhf.converged
    assert rhf.backend.startswith("native-gamma-gdf-via-k-gdf")
    assert uhf.backend.startswith("native-multi-k-gdf")
    assert uhf.energy == pytest.approx(rhf.energy, abs=ROUTE_TOL)


def _h2_cube12_lda_opts(fock_mixing):
    system = vq.PeriodicSystem(
        3, np.eye(3) * 12.0, [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])], 0, 1)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    o = vq.PeriodicKSOptions()
    o.fock_mixing = fock_mixing
    o.conv_tol_energy = 1e-11
    o.max_iter = 100
    return system, basis, o


def test_rks_fock_mixing_does_not_move_the_fixed_point_on_the_exact_gamma_driver():
    """#297 item 3 measured a 9.4e-8 "drift with Fock mixing" on Gamma RKS +
    mdf. The mixing is not what moved: on the exact-Gamma UKS singlet driver
    0.0 and 0.2 are bit-identical, and the k-driver's Gamma adapter
    reproduces it. The knob moved the run onto the Nk=1 MDF loop, which
    (before 6cd4c3a) did not build the fit an explicit gdf_method asked
    for on a pure-DFT mesh; that leg is pinned below."""
    system, basis, o0 = _h2_cube12_lda_opts(0.0)
    _, _, o2 = _h2_cube12_lda_opts(0.2)
    exact0 = run_pbc_gdf_uks(system, basis, o0, functional="lda", gdf_method="mdf", progress=False)
    exact2 = run_pbc_gdf_uks(system, basis, o2, functional="lda", gdf_method="mdf", progress=False)
    adapter = run_krks_periodic_gdf(system, basis, (1, 1, 1), o0, functional="lda",
                                    gdf_method="mdf", progress=False)
    for r in (exact0, exact2, adapter):
        assert r.converged
    assert exact2.energy == pytest.approx(exact0.energy, abs=ROUTE_TOL)
    assert adapter.energy == pytest.approx(exact0.energy, abs=ROUTE_TOL)


def test_rks_mdf_loop_at_gamma_matches_the_exact_gamma_driver():
    """The pure-DFT leg of #416: before 6cd4c3a the Nk=1 loop did not build
    the fit an explicit gdf_method asked for on a pure-DFT k-mesh, and this
    comparison was 1.35e-7 Ha off (what #297 item 3 measured as a drift
    with Fock mixing). It agrees exactly now; the exact-exchange leg above
    still splits."""
    system, basis, o2 = _h2_cube12_lda_opts(0.2)
    exact = run_pbc_gdf_uks(system, basis, o2, functional="lda", gdf_method="mdf", progress=False)
    loop = run_krks_periodic_gdf(system, basis, (1, 1, 1), o2, functional="lda",
                                 gdf_method="mdf", fock_mixing=0.2, progress=False)
    assert exact.converged and loop.converged
    assert loop.backend.startswith("native-multi-k-gdf")
    assert loop.energy == pytest.approx(exact.energy, abs=ROUTE_TOL)


# --- honour or refuse, at the driver boundary too ---------------------------------


@pytest.mark.parametrize("field, value, words", [
    ("level_shift", 0.2, "level shift"), ("fock_mixing", 0.3, "Fock mixing"),
])
def test_open_shell_k_mesh_driver_refuses_what_it_would_drop(field, value, words):
    system, basis = _h2_cube6()
    options = vq.PeriodicRHFOptions()
    setattr(options, field, value)
    with pytest.raises(NotImplementedError, match=words):
        run_kuhf_periodic_gdf(system, basis, (1, 1, 1), options, progress=False)


def test_exact_gamma_open_shell_refusal_names_no_implementing_driver():
    system, basis = _h2_cube6()
    options = vq.PeriodicRHFOptions()
    options.level_shift = 0.2
    with pytest.raises(NotImplementedError, match="nor by run_kuhf_periodic_gdf"):
        run_pbc_gdf_uhf(system, basis, options, progress=False)
