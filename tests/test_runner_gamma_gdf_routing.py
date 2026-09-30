"""Default-Γ closed-shell GDF routing under auto convergence knobs.

Regression for G-GDF-001 (HANDOVER_GATED_ITEMS.md): the ionic-insulator
auto profile resolves FMIXING 30% on MgO-class cells, and the runner's
default-Γ RHF/GDF gate requires ``fock_mixing == 0`` to route through the
PySCF-µHa-validated ``run_pbc_gdf_rhf``. Before the fix, the AUTO knob was
only capability-filtered for an *explicit* ``gdf_method``, so a plain
``run_periodic_job(..., jk_method="gdf")`` on an ionic cell silently fell
back to the legacy molecular-limit Γ driver, whose dense-core absolute
energies are PARITY_HELD. The filter now also covers the default route;
Explicit user knobs send the job to the k-mesh engine at kmesh=(1,1,1),
which implements them on the same Hamiltonian; the legacy molecular-limit
Gamma driver is reached only for dim < 3 (#400, #95).

These tests pin the DISPATCH decision only (sentinel drivers, no SCF).
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
import vibeqc.periodic_runner as periodic_runner


class _Routed(Exception):
    """Sentinel raised by the stub drivers to record which one was picked."""

    def __init__(self, driver: str):
        super().__init__(driver)
        self.driver = driver


def _mgo_primitive_physical() -> vq.PeriodicSystem:
    """MgO rocksalt FCC primitive at the physical lattice (a_conv = 7.958
    bohr): classifies ionic-insulator, so the auto profile resolves
    FMIXING 30% (the G-GDF-001 trigger)."""
    h = 7.958 / 2.0
    lat = np.array([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]])
    atoms = [vq.Atom(12, [0, 0, 0]), vq.Atom(8, [h, h, h])]
    return vq.PeriodicSystem(3, lat, atoms)


def _run(tmp_path, monkeypatch, *, system=None, general=None, **kwargs) -> str:
    def _sentinel(name):
        return lambda *a, **k: (_ for _ in ()).throw(_Routed(name))

    monkeypatch.setattr(periodic_runner, "run_pbc_gdf_rhf", _sentinel("pure"))
    monkeypatch.setattr(periodic_runner, "run_pbc_gdf_uks", _sentinel("pure-uks"))
    monkeypatch.setattr(
        periodic_runner, "run_rhf_periodic_gamma_gdf", _sentinel("legacy"))
    monkeypatch.setattr(
        periodic_runner, "run_krhf_periodic_gdf", general or _sentinel("general"))
    monkeypatch.setattr(
        periodic_runner, "run_krks_periodic_gdf", general or _sentinel("general"))
    if system is None:
        system = _mgo_primitive_physical()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    with pytest.raises(_Routed) as excinfo:
        periodic_runner.run_periodic_job(
            system,
            basis,
            method=kwargs.pop("method", "RHF"),
            jk_method="gdf",
            output=str(tmp_path / "route"),
            write_molden_file=False,
            write_xyz_file=False,
            write_poscar_file=False,
            write_xsf_structure_file=False,
            write_cif_file=False,
            write_population_file=False,
            citations=False,
            **kwargs,
        )
    return excinfo.value.driver


def test_default_gamma_ionic_auto_fmixing_routes_pure_gdf(tmp_path, monkeypatch):
    """The ionic-insulator AUTO FMIXING must not force the default-Γ RHF
    run onto the legacy PARITY_HELD fallback: the knob is capability-
    filtered to zero and the run stays on run_pbc_gdf_rhf."""
    assert _run(tmp_path, monkeypatch) == "pure"


_KNOBS = [
    {"fock_mixing": 0.2},
    {"fmixing_percent": 20.0},
    {"level_shift": 0.2},
    {"smearing_temperature": 0.01},
]


@pytest.mark.parametrize("controls", _KNOBS)
def test_explicit_controls_on_dense_core_run_the_k_mesh_engine(
    tmp_path, monkeypatch, controls,
):
    """A knob the exact-Gamma driver lacks moves the job to the k-mesh
    engine at (1,1,1), which implements it on the same Hamiltonian. Before,
    the dense-core cell was refused (#95) and a low-Z cell took the legacy
    driver, whose exchange has no divergence correction (#400)."""
    assert _run(tmp_path, monkeypatch, **controls) == "general"


@pytest.mark.parametrize("controls", _KNOBS)
def test_explicit_controls_on_low_z_cell_run_the_k_mesh_engine(
    tmp_path, monkeypatch, controls,
):
    """The same routing on a low-Z vacuum box: no knob selects the legacy
    driver any more (measured before: +75.65 mHa on H2/STO-3G in a 6-bohr
    cube with any of these knobs)."""
    system = vq.PeriodicSystem(3, np.eye(3) * 20.0,
                             [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])])
    assert _run(tmp_path, monkeypatch, system=system, **controls) == "general"


@pytest.mark.parametrize("controls", [
    {"level_shift": 0.2}, {"smearing_temperature": 0.01},
])
def test_rks_controls_run_the_k_mesh_engine(tmp_path, monkeypatch, controls):
    assert _run(tmp_path, monkeypatch, method="RKS", functional="lda",
                **controls) == "general"


def test_rks_fock_mixing_stays_on_the_exact_gamma_uks_driver(tmp_path, monkeypatch):
    """run_pbc_gdf_uks implements Fock mixing, so RKS keeps it."""
    assert _run(tmp_path, monkeypatch, method="RKS", functional="lda",
                fock_mixing=0.2) == "pure-uks"


def test_charged_bulk_cell_runs_the_k_mesh_engine(tmp_path, monkeypatch):
    """run_pbc_gdf_rhf refuses a charged cell; the k-mesh engine carries the
    neutralising background. Measured on HeH+/STO-3G in a 7-bohr cube: the
    k-mesh engine at (1,1,1) matches PySCF 2.14 (exxdiv='ewald') to 3e-12
    Ha, the former legacy fallback sat 18.4 mHa above it."""
    system = vq.PeriodicSystem(3, np.eye(3) * 7.0,
                             [vq.Atom(2, [0, 0, 0]), vq.Atom(1, [1.46, 0, 0])],
                             1, 1)
    assert _run(tmp_path, monkeypatch, system=system) == "general"


@pytest.mark.parametrize("controls", [
    {"symmetry_stabilize": True},
    {"symmetry_reduce_fock": True},
])
def test_fock_symmetry_is_refused_on_bulk_gdf(tmp_path, monkeypatch, controls):
    """No bulk GDF engine implements Fock symmetrisation; the Gamma job used
    to take the legacy driver for it and a k-mesh job dropped it."""
    with pytest.raises(NotImplementedError, match="symmetry_stabilize and"):
        _run(tmp_path, monkeypatch, **controls)
    with pytest.raises(NotImplementedError, match="symmetry_stabilize and"):
        _run(tmp_path, monkeypatch, kpoints=(2, 1, 1), **controls)


def test_one_dimensional_chain_keeps_the_legacy_gamma_driver(tmp_path, monkeypatch):
    """The k-mesh engine has no dim = 1 Gamma route, so a chain stays on the
    legacy driver, where an explicit kpoints=(1,1,1) also lands."""
    system = vq.PeriodicSystem(1, np.diag([3.0, 12.0, 12.0]),
                             [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [1.4, 0, 0])])
    assert _run(tmp_path, monkeypatch, system=system,
                level_shift=0.2) == "legacy"


def test_explicit_bulk_rsgdf_gamma_controls_reach_general_driver(tmp_path, monkeypatch):
    def general(*args, **kwargs):
        assert kwargs["gdf_method"] == "rsgdf"
        raise _Routed("general")

    assert _run(tmp_path, monkeypatch, kpoints=(1, 1, 1), general=general,
                gdf_method="rsgdf", fock_mixing=0.2) == "general"


@pytest.mark.parametrize("gdf_method", ["rsgdf", "mdf"])
def test_default_gamma_knob_keeps_the_requested_gdf_method(
    tmp_path, monkeypatch, gdf_method,
):
    """The k-mesh route for a default-Gamma knob runs (1,1,1) with the
    requested builder rather than refusing it (the former #283 guard)."""
    seen = {}

    def general(system, basis, kmesh, *args, **kwargs):
        seen.update(kmesh=tuple(kmesh), gdf_method=kwargs["gdf_method"])
        raise _Routed("general")

    assert _run(tmp_path, monkeypatch, general=general, gdf_method=gdf_method,
                fock_mixing=0.2) == "general"
    assert seen == {"kmesh": (1, 1, 1), "gdf_method": gdf_method}


def _capture_general(seen):
    """A k-mesh stub that records the resolved options it was handed."""

    def general(system, basis, kmesh, options, *args, **kwargs):
        seen.update(
            kmesh=tuple(int(n) for n in kmesh),
            fock_mixing=float(options.fock_mixing),
            level_shift=float(getattr(options, "level_shift", 0.0)),
            gdf_method=kwargs.get("gdf_method"),
        )
        raise _Routed("general")

    return general


def _w_bcc_pob(monkeypatch):
    """W bcc / pob-TZVP-rev2: an ECP-bearing, tight, all-metal cell. HF on a
    metallic candidate resolves an AUTO level shift (0.2) and AUTO Fock
    mixing (0.5). The bundled CRYSTAL ECP records resolve offline; the
    fetcher is disarmed so the test cannot reach the network."""
    import sys
    import vibeqc.basis_crystal as basis_crystal

    assert sys.modules["vibeqc.basis_crystal"] is basis_crystal

    def _no_fetch(*args, **kwargs):
        raise AssertionError("bundled POB ECP records should not fetch")

    monkeypatch.setattr(basis_crystal, "fetch_bredow_basis_sets", _no_fetch)
    a = 3.191 * 1.8897261246257702
    return vq.PeriodicSystem(3, np.eye(3) * a, [
        vq.Atom(74, [0.0, 0.0, 0.0]), vq.Atom(74, [a / 2, a / 2, a / 2])])


def test_ecp_cell_keeps_its_auto_convergence_aids_on_the_k_mesh_engine(
    tmp_path, monkeypatch,
):
    """An ECP-bearing cell always runs the k-mesh engine (#88), which
    implements the AUTO level shift and Fock mixing of the metallic-candidate
    HF profile. Before, the filter's private copy of the exact-Gamma domain
    gate knew nothing about ECPs: it zeroed both aids "to stay on
    run_pbc_gdf_rhf", a driver the job never reached."""
    system = _w_bcc_pob(monkeypatch)
    basis = vq.BasisSet(system.unit_cell_molecule(), "pob-tzvp-rev2")
    seen = {}

    def _sentinel(name):
        return lambda *a, **k: (_ for _ in ()).throw(_Routed(name))

    monkeypatch.setattr(periodic_runner, "run_pbc_gdf_rhf", _sentinel("pure"))
    monkeypatch.setattr(
        periodic_runner, "run_rhf_periodic_gamma_gdf", _sentinel("legacy"))
    monkeypatch.setattr(
        periodic_runner, "run_krhf_periodic_gdf", _capture_general(seen))
    with pytest.raises(_Routed) as excinfo:
        periodic_runner.run_periodic_job(
            system, basis, method="RHF", jk_method="gdf",
            output=str(tmp_path / "w"), write_molden_file=False,
            write_xyz_file=False, write_poscar_file=False,
            write_xsf_structure_file=False, write_cif_file=False,
            write_population_file=False, citations=False)
    assert excinfo.value.driver == "general"
    assert seen["kmesh"] == (1, 1, 1)
    assert seen["level_shift"] == 0.2 and seen["fock_mixing"] == 0.5


def test_parity_hold_summary_states_hold_in_out_text():
    """IID 344: a parity-held result states the hold in the .out next to the
    energy it qualifies, instead of only in the .err sidecar; an ordinary
    result emits nothing, so golden outputs are byte-unchanged."""
    from types import SimpleNamespace

    held = SimpleNamespace(backend="pbc-gdf-rsgdf+PARITY_HELD")
    text = periodic_runner._parity_hold_summary(held)
    assert "PARITY HELD" in text
    assert "pbc-gdf-rsgdf+PARITY_HELD" in text
    # aiccm2026dev-b results carry the executed label as runtime_backend.
    text_b = periodic_runner._parity_hold_summary(
        SimpleNamespace(
            runtime_backend="native-multi-k-gdf-gdf-rhf+PARITY_HELD",
            backend="aiccm2026dev-b-ri",
        )
    )
    assert "PARITY HELD" in text_b
    assert "native-multi-k-gdf-gdf-rhf+PARITY_HELD" in text_b
    # The common case emits nothing.
    assert periodic_runner._parity_hold_summary(
        SimpleNamespace(backend="pbc-gdf-rsgdf")
    ) == ""
    assert periodic_runner._parity_hold_summary(SimpleNamespace()) == ""
