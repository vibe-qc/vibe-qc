"""Default-Γ closed-shell GDF routing under auto convergence knobs.

Regression for G-GDF-001 (HANDOVER_GATED_ITEMS.md): the ionic-insulator
auto profile resolves FMIXING 30% on MgO-class cells, and the runner's
default-Γ RHF/GDF gate requires ``fock_mixing == 0`` to route through the
PySCF-µHa-validated ``run_pbc_gdf_rhf``. Before the fix, the AUTO knob was
only capability-filtered for an *explicit* ``gdf_method``, so a plain
``run_periodic_job(..., jk_method="gdf")`` on an ionic cell silently fell
back to the legacy molecular-limit Γ driver, whose dense-core absolute
energies are PARITY_HELD. The filter now also covers the default route;
explicit user knobs still fall back (fail-open would silently drop a knob
the user asked for).

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


def _run(tmp_path, monkeypatch, **kwargs) -> str:
    monkeypatch.setattr(
        periodic_runner,
        "run_pbc_gdf_rhf",
        lambda *a, **k: (_ for _ in ()).throw(_Routed("pure")),
    )
    monkeypatch.setattr(
        periodic_runner,
        "run_rhf_periodic_gamma_gdf",
        lambda *a, **k: (_ for _ in ()).throw(_Routed("legacy")),
    )
    system = _mgo_primitive_physical()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    with pytest.raises(_Routed) as excinfo:
        periodic_runner.run_periodic_job(
            system,
            basis,
            method="RHF",
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


def test_explicit_fock_mixing_still_uses_legacy_fallback(tmp_path, monkeypatch):
    """An EXPLICIT user Fock-mixing request is not silently dropped: the
    run falls back to the legacy Γ driver that implements the knob."""
    assert _run(tmp_path, monkeypatch, fock_mixing=0.2) == "legacy"


def test_explicit_fmixing_percent_still_uses_legacy_fallback(tmp_path, monkeypatch):
    """Same contract through the CRYSTAL-style fmixing_percent spelling."""
    assert _run(tmp_path, monkeypatch, fmixing_percent=20.0) == "legacy"


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
