"""Smoke tests for GAPW molecular dynamics driver."""

from __future__ import annotations

import ast
import inspect
import io

import vibeqc.periodic_gapw_md as gapw_md

from vibeqc.output import (
    DEFAULT_POLICY,
    OutputChannel,
    active_policy,
    set_active_policy,
)
from vibeqc.periodic_gapw_md import (
    _format_md_status,
    run_md,
    run_npt,
    run_nve,
    run_nvt,
)


def test_md_imports():
    assert callable(run_md)
    assert callable(run_nve)
    assert callable(run_nvt)
    assert callable(run_npt)


def test_md_status_uses_active_energy_policy():
    saved = active_policy()
    hartree_ev = gapw_md.units.Hartree
    try:
        set_active_policy(DEFAULT_POLICY)
        default = _format_md_status(3, hartree_ev, 2.0 * hartree_ev, 125.0)
        assert default == (
            "step    3:  E_pot = +1.000000 Ha  E_tot = +2.000000 Ha  T =  125.0 K"
        )

        set_active_policy(DEFAULT_POLICY.with_unit("energy", "eV"))
        ev = _format_md_status(3, hartree_ev, 2.0 * hartree_ev, 125.0)
        assert ev == (
            "step    3:  E_pot = +27.211386 eV  E_tot = +54.422772 eV  T =  125.0 K"
        )
    finally:
        set_active_policy(saved)


def test_run_md_emits_policy_formatted_live_status(monkeypatch, capsys):
    class DummyAtoms:
        calc = None

        def __len__(self):
            return 2

        def get_kinetic_energy(self):
            return 0.0

        def get_potential_energy(self, *, force_consistent):
            assert force_consistent is False
            return -27.211386245988

    class DummyDynamics:
        def __init__(self):
            self.callback = None

        def get_number_of_steps(self):
            return 0

        def attach(self, callback, *, interval):
            assert interval == 1
            self.callback = callback

        def run(self, n_steps):
            assert n_steps == 1
            self.callback()

    dyn = DummyDynamics()
    monkeypatch.setattr(gapw_md, "VibeqcGAPW", lambda **kwargs: object())
    monkeypatch.setattr(
        gapw_md, "MaxwellBoltzmannDistribution", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(gapw_md, "Stationary", lambda *args, **kwargs: None)
    monkeypatch.setattr(gapw_md, "VelocityVerlet", lambda *args, **kwargs: dyn)

    saved = active_policy()
    try:
        set_active_policy(DEFAULT_POLICY)
        atoms = DummyAtoms()
        assert (
            run_md(
                atoms,
                md_engine="nve",
                n_steps=1,
                log_interval=1,
            )
            is atoms
        )
    finally:
        set_active_policy(saved)

    assert capsys.readouterr().out == (
        "  step    0:  E_pot = -1.000000 Ha  E_tot = -1.000000 Ha  T =    0.0 K\n"
    )


def test_run_npt_emits_policy_formatted_persistent_status(monkeypatch, capsys):
    class DummyAtoms:
        calc = None

        def __len__(self):
            return 2

        def get_kinetic_energy(self):
            return 0.0

        def get_potential_energy(self, *, force_consistent):
            assert force_consistent is False
            return -gapw_md.units.Hartree

    class DummyDynamics:
        def __init__(self):
            self.callback = None

        def get_number_of_steps(self):
            return 4

        def attach(self, callback, *, interval):
            assert interval == 10
            self.callback = callback

        def run(self, n_steps):
            assert n_steps == 1
            self.callback()

    dyn = DummyDynamics()
    monkeypatch.setattr(gapw_md, "VibeqcGAPW", lambda **kwargs: object())
    monkeypatch.setattr(
        gapw_md, "MaxwellBoltzmannDistribution", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(gapw_md, "Stationary", lambda *args, **kwargs: None)
    monkeypatch.setattr(gapw_md, "NPTBerendsen", lambda *args, **kwargs: dyn)

    saved = active_policy()
    output = io.StringIO()
    try:
        set_active_policy(DEFAULT_POLICY)
        atoms = DummyAtoms()
        with OutputChannel.to_stream(output):
            assert run_npt(atoms, n_steps=1) is atoms
    finally:
        set_active_policy(saved)

    assert output.getvalue() == (
        "  step    4:  E_pot = -1.000000 Ha  E_tot = -1.000000 Ha  T =    0.0 K\n"
    )
    assert capsys.readouterr().out == ""


def test_gapw_md_has_no_direct_print_calls():
    tree = ast.parse(inspect.getsource(gapw_md))
    direct_prints = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]
    assert direct_prints == []
