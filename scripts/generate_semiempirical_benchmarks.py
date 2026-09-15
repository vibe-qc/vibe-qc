#!/usr/bin/env python3
"""Semiempirical benchmark data generator.

Runs all implemented semiempirical methods on a standard molecular
and periodic test set.  Exports structured JSON for article tables.

Usage:
    python scripts/generate_semiempirical_benchmarks.py [--output bench.json]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# ── Molecular test set ──
MOLECULES = {
    "H2": [
        ("H", [0.0, 0.0, 0.0]),
        ("H", [1.4, 0.0, 0.0]),
    ],
    "H2O": [
        ("O", [0.0000, 0.0000, 0.1173]),
        ("H", [0.0000, 1.4315, -0.9386]),
        ("H", [0.0000, -1.4315, -0.9386]),
    ],
    "CH4": [
        ("C", [0.0000, 0.0000, 0.0000]),
        ("H", [1.1869, 1.1869, 1.1869]),
        ("H", [-1.1869, -1.1869, 1.1869]),
        ("H", [1.1869, -1.1869, -1.1869]),
        ("H", [-1.1869, 1.1869, -1.1869]),
    ],
    "NH3": [
        ("N", [0.0000, 0.0000, 0.1147]),
        ("H", [0.0000, 1.7670, -0.4810]),
        ("H", [1.5303, -0.8835, -0.4810]),
        ("H", [-1.5303, -0.8835, -0.4810]),
    ],
    "CO2": [
        ("C", [0.0000, 0.0000, 0.0000]),
        ("O", [0.0000, 0.0000, 2.1960]),
        ("O", [0.0000, 0.0000, -2.1960]),
    ],
    "H2CO": [
        ("C", [0.0000, 0.0000, 0.0000]),
        ("O", [0.0000, 0.0000, 2.2740]),
        ("H", [0.0000, 1.7700, -1.0820]),
        ("H", [0.0000, -1.7700, -1.0820]),
    ],
}

# ── Periodic test set ──
PERIODIC = {
    "He_chain_1D": {
        "dim": 1,
        "cell": [3.0, 20.0, 20.0],
        "atoms": [("He", [0, 0, 0])],
    },
    "C_chain_1D": {
        "dim": 1,
        "cell": [2.5, 20.0, 20.0],
        "atoms": [("C", [0, 0, 0])],
    },
    "Si_chain_1D": {
        "dim": 1,
        "cell": [4.0, 20.0, 20.0],
        "atoms": [("Si", [0, 0, 0])],
    },
}


def _molecule(name):
    from vibeqc._vibeqc_core import Atom, Molecule

    atoms = [Atom(_Z(sym), xyz) for sym, xyz in MOLECULES[name]]
    return Molecule(atoms)


def _periodic(name):
    from vibeqc._vibeqc_core import Atom, PeriodicSystem

    spec = PERIODIC[name]
    cell = np.diag(spec["cell"])
    atoms = [Atom(_Z(sym), xyz) for sym, xyz in spec["atoms"]]
    return PeriodicSystem(spec["dim"], cell, atoms, 0, 1)


def _Z(symbol):
    _MAP = {
        "H": 1,
        "He": 2,
        "Li": 3,
        "Be": 4,
        "B": 5,
        "C": 6,
        "N": 7,
        "O": 8,
        "F": 9,
        "Ne": 10,
        "Na": 11,
        "Mg": 12,
        "Al": 13,
        "Si": 14,
        "P": 15,
        "S": 16,
        "Cl": 17,
        "Ar": 18,
    }
    return _MAP.get(symbol, 0)


def run_benchmarks(output_path: str | None = None):
    from vibeqc._vibeqc_core import semiempirical as _se
    from vibeqc._vibeqc_core.semiempirical import nddo
    from vibeqc.semiempirical import (
        DFTB0Model,
        SCCDFTBModel,
        UDFTB0Model,
        USCCDFTBModel,
    )
    from vibeqc.semiempirical.methods.gfn2 import GFN2Model
    from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params
    from vibeqc.semiempirical.methods.omx_params import (
        load_om1_params,
        load_om2_params,
        load_om3_params,
    )
    from vibeqc.semiempirical.methods.pm6_params import (
        load_pm6_mopac_params,
        load_pm6_params,
    )
    from vibeqc.semiempirical.parameters import default_parameters

    results: dict = {"molecular": {}, "periodic": {}, "meta": {}}

    start_total = time.time()

    # ──────────────────────────────────────────────────
    # Molecular methods
    # ──────────────────────────────────────────────────
    pm6_5 = load_pm6_params()
    pm6_full = load_pm6_mopac_params()
    gfn2_p = load_gfn2_params()
    dftb_p = default_parameters()
    om1_p = load_om1_params()
    om2_p = load_om2_params()
    om3_p = load_om3_params()

    molecular_configs = [
        ("DFTB0", lambda mol: DFTB0Model(mol, dftb_p)),
        ("SCC-DFTB", lambda mol: SCCDFTBModel(mol, dftb_p)),
        ("GFN2-xTB", lambda mol: GFN2Model(mol, gfn2_p)),
        ("PM6 (5 el)", lambda mol: _PM6Model(mol, pm6_5)),
        ("PM6 (75 el)", lambda mol: _PM6Model(mol, pm6_full)),
        ("OM1", lambda mol: _OMxModel(mol, om1_p)),
        ("OM2", lambda mol: _OMxModel(mol, om2_p)),
        ("OM3", lambda mol: _OMxModel(mol, om3_p)),
    ]

    for mol_name in MOLECULES:
        mol = _molecule(mol_name)
        results["molecular"][mol_name] = {}
        for method_name, factory in molecular_configs:
            t0 = time.time()
            try:
                model = factory(mol)
                e = model.energy()
                dt = time.time() - t0
                results["molecular"][mol_name][method_name] = {
                    "energy_ha": float(e),
                    "energy_ev": float(e * 27.2114),
                    "time_s": round(dt, 3),
                    "status": "ok",
                }
            except Exception as exc:
                dt = time.time() - t0
                results["molecular"][mol_name][method_name] = {
                    "energy_ha": None,
                    "time_s": round(dt, 3),
                    "status": f"error: {exc}",
                }

    # ──────────────────────────────────────────────────
    # Periodic methods
    # ──────────────────────────────────────────────────
    periodic_configs = [
        ("DFTB0", lambda sys: _se.run_dftb0_gamma(sys, dftb_p)),
        (
            "SCC-DFTB",
            lambda sys: _se.run_scc_dftb_gamma(sys, dftb_p, _se.PeriodicSCCOptions()),
        ),
        ("GFN2-xTB", lambda sys: _gf2_periodic(sys, gfn2_p)),
        ("PM6 (5 el)", lambda sys: _pm6_periodic(sys, pm6_5)),
        ("PM6 (75 el)", lambda sys: _pm6_periodic(sys, pm6_full)),
    ]

    for per_name in PERIODIC:
        system = _periodic(per_name)
        results["periodic"][per_name] = {}
        for method_name, runner in periodic_configs:
            t0 = time.time()
            try:
                r = runner(system)
                dt = time.time() - t0
                results["periodic"][per_name][method_name] = {
                    "energy_ha": float(r.energy),
                    "energy_ev": float(r.energy * 27.2114),
                    "time_s": round(dt, 3),
                    "converged": getattr(r, "converged", True),
                    "n_cells": getattr(r, "n_cells", 1),
                    "status": "ok",
                }
            except Exception as exc:
                dt = time.time() - t0
                results["periodic"][per_name][method_name] = {
                    "energy_ha": None,
                    "time_s": round(dt, 3),
                    "status": f"error: {exc}",
                }

    # ── Meta ──
    results["meta"] = {
        "total_time_s": round(time.time() - start_total, 1),
        "molecular_count": len(MOLECULES),
        "periodic_count": len(PERIODIC),
        "method_count_molecular": len(molecular_configs),
        "method_count_periodic": len(periodic_configs),
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {output_path}")
    else:
        print(json.dumps(results, indent=2))


# ── Helper model wrappers ──


class _PM6Model:
    def __init__(self, mol, params):
        self._mol = mol
        self._p = params

    def energy(self):
        from vibeqc._vibeqc_core.semiempirical.nddo import run_pm6

        return run_pm6(self._mol, self._p, max_iter=200).energy


class _OMxModel:
    def __init__(self, mol, params):
        self._mol = mol
        self._p = params

    def energy(self):
        from vibeqc._vibeqc_core.semiempirical.nddo import run_omx_v2

        return run_omx_v2(self._mol, self._p, max_iter=200).energy


def _gf2_periodic(system, params):
    from vibeqc._vibeqc_core.semiempirical import xtb as _xtb

    return _xtb.run_gfn2_xtb_gamma(system, params)


def _pm6_periodic(system, params):
    from vibeqc._vibeqc_core.semiempirical.nddo import (
        PeriodicPM6Options,
        run_pm6_gamma,
    )

    return run_pm6_gamma(system, params, PeriodicPM6Options())


if __name__ == "__main__":
    out = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--output" else None
    run_benchmarks(out)
