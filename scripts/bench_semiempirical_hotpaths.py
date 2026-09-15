#!/usr/bin/env python3
"""Semiempirical hot-path benchmark and C++ port inventory.

This is a developer tool, not a scientific benchmark. It records which
semiempirical routes are already C++ hot paths and gives a cheap way to time
representative energy/gradient/stress calls while isolating each case in its
own Python process.

Examples
--------

    python scripts/bench_semiempirical_hotpaths.py --list
    python scripts/bench_semiempirical_hotpaths.py --list-suites
    python scripts/bench_semiempirical_hotpaths.py --suite smoke
    python scripts/bench_semiempirical_hotpaths.py --only gfn2_nacl_energy --json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parent.parent

SUITE_GMTKN55 = "GMTKN55"
SUITE_GMTKN24 = "Korth-Thiel GMTKN24"
SUITE_DRAL_THIEL = "Dral-Thiel ground-state OMx/ODMx"
SUITE_THIEL_EXCITED = "Thiel excited-state family"
SUITE_DELTA_SOLIDS = "Delta-factor / SSSP solids"
SUITE_MATBENCH = "Matbench Discovery"
SUITE_MACE = "MACE literature validation suites"

ARTICLE_BENCHMARK_SCOPES: dict[str, str] = {
    SUITE_GMTKN55: "compact-runnable molecular representatives",
    SUITE_GMTKN24: "compact-runnable semiempirical representatives",
    SUITE_DRAL_THIEL: "compact-runnable ground-state OMx representatives",
    SUITE_DELTA_SOLIDS: "compact-runnable periodic/solid representatives",
    SUITE_MATBENCH: "future-driver solid-discovery expansion",
    SUITE_THIEL_EXCITED: "future-driver excited-state route",
    SUITE_MACE: "adjacent non-semiempirical wrapper scope",
}


@dataclass(frozen=True)
class Case:
    key: str
    suite: str
    label: str
    operation: str
    backend: str
    hot_path: str
    priority: str
    external_suites: tuple[str, ...] = ()


CASES: tuple[Case, ...] = (
    Case(
        "dftb0_h2o_energy", "smoke", "DFTB0 H2O energy",
        "molecular energy", "C++", "cpp/src/semiempirical/dftb0.cpp", "S4",
        external_suites=(SUITE_GMTKN55, SUITE_GMTKN24),
    ),
    Case(
        "dftb0_h2o_gradient", "inventory", "DFTB0 H2O gradient",
        "molecular analytic gradient", "C++",
        "cpp/src/semiempirical/gradient.cpp", "S4",
    ),
    Case(
        "dftb0_h2o_preoptimize", "inventory", "DFTB0 H2O preoptimization",
        "molecular geometry optimization",
        "Python ASE orchestration + C++ energy/gradient",
        "python/vibeqc/semiempirical/preoptimize.py; "
        "cpp/src/semiempirical/dftb0.cpp; cpp/src/semiempirical/gradient.cpp", "S4",
    ),
    Case(
        "scc_dftb_h2o_energy", "smoke", "SCC-DFTB H2O energy",
        "molecular SCC energy", "C++", "cpp/src/semiempirical/scc_dftb.cpp", "S4",
        external_suites=(SUITE_GMTKN55, SUITE_GMTKN24),
    ),
    Case(
        "gfn2_h2o_energy", "smoke", "GFN2-xTB H2O energy",
        "molecular SCC + post-SCF D4", "C++ core + Python D4 wrapper",
        "cpp/src/semiempirical/methods/xtb/gfn2_driver.cpp", "S3",
        external_suites=(SUITE_GMTKN55, SUITE_GMTKN24),
    ),
    Case(
        "gfn2_h2o_gradient", "inventory", "GFN2-xTB H2O gradient",
        "molecular analytic gradient", "C++",
        "cpp/src/semiempirical/gradient.cpp", "S3",
    ),
    Case(
        "gfn2_nh3_energy", "inventory", "GFN2-xTB NH3 energy",
        "molecular SCC + post-SCF D4", "C++ core + Python D4 wrapper",
        "cpp/src/semiempirical/methods/xtb/gfn2_driver.cpp", "S3",
        external_suites=(SUITE_GMTKN55, SUITE_GMTKN24),
    ),
    Case(
        "gfn2_benzene_energy", "inventory", "GFN2-xTB benzene energy",
        "molecular SCC + post-SCF D4", "C++ core + Python D4 wrapper",
        "cpp/src/semiempirical/methods/xtb/gfn2_driver.cpp", "S3",
        external_suites=(SUITE_GMTKN55, SUITE_GMTKN24),
    ),
    Case(
        "gfn2_nacl_energy", "inventory", "GFN2-xTB NaCl ionic diatomic",
        "molecular SCC stabilization", "C++ SCC core",
        "cpp/src/semiempirical/methods/xtb/gfn2_driver.cpp", "S3",
        external_suites=(SUITE_GMTKN24,),
    ),
    Case(
        "gfn2_lif_energy", "inventory", "GFN2-xTB LiF ionic diatomic",
        "molecular SCC stabilization", "C++ SCC core",
        "cpp/src/semiempirical/methods/xtb/gfn2_driver.cpp", "S3",
        external_suites=(SUITE_GMTKN24,),
    ),
    Case(
        "gfn2_lih_energy", "inventory", "GFN2-xTB LiH ionic diatomic",
        "molecular SCC stabilization", "C++ SCC core",
        "cpp/src/semiempirical/methods/xtb/gfn2_driver.cpp", "S3",
        external_suites=(SUITE_GMTKN24,),
    ),
    Case(
        "pm6_h2o_energy", "smoke", "PM6 H2O energy",
        "molecular NDDO energy", "C++",
        "cpp/src/semiempirical/methods/nddo/pm6_fock.cpp", "S5",
        external_suites=(SUITE_GMTKN55, SUITE_GMTKN24),
    ),
    Case(
        "pm6_h2o_gradient_fd", "inventory", "PM6 H2O finite-difference gradient",
        "molecular finite-difference gradient", "C++ finite-difference wrapper",
        "cpp/src/semiempirical/methods/nddo/pm6_fock.cpp", "S5",
    ),
    Case(
        "pm6_ch4_energy", "inventory", "PM6 CH4 energy",
        "molecular NDDO energy", "C++",
        "cpp/src/semiempirical/methods/nddo/pm6_fock.cpp", "S5",
        external_suites=(SUITE_GMTKN55, SUITE_GMTKN24),
    ),
    Case(
        "pm6_nh3_energy", "inventory", "PM6 NH3 energy",
        "molecular NDDO energy", "C++",
        "cpp/src/semiempirical/methods/nddo/pm6_fock.cpp", "S5",
        external_suites=(SUITE_GMTKN55, SUITE_GMTKN24),
    ),
    Case(
        "om2_h2o_energy", "smoke", "OM2 H2O energy",
        "molecular OMx energy", "C++",
        "cpp/src/semiempirical/methods/nddo/omx_fock.cpp", "S5",
        external_suites=(SUITE_GMTKN55, SUITE_GMTKN24, SUITE_DRAL_THIEL),
    ),
    Case(
        "om2_ch4_energy", "inventory", "OM2 CH4 energy",
        "molecular OMx energy", "C++",
        "cpp/src/semiempirical/methods/nddo/omx_fock.cpp", "S5",
        external_suites=(SUITE_GMTKN55, SUITE_GMTKN24, SUITE_DRAL_THIEL),
    ),
    Case(
        "om2_nh3_energy", "inventory", "OM2 NH3 energy",
        "molecular OMx energy", "C++",
        "cpp/src/semiempirical/methods/nddo/omx_fock.cpp", "S5",
        external_suites=(SUITE_GMTKN55, SUITE_GMTKN24, SUITE_DRAL_THIEL),
    ),
    Case(
        "msindo_h2o_cpp_energy", "smoke", "MSINDO H2O energy via run_job",
        "molecular INDO energy", "C++ through run_job",
        "cpp/include/vibeqc/semiempirical/methods/indo/indo_engine.hpp", "S2",
        external_suites=(SUITE_GMTKN55, SUITE_GMTKN24),
    ),
    Case(
        "msindo_h2o_direct_energy", "inventory", "MSINDO H2O direct energy",
        "molecular INDO energy", "C++ through public run_msindo",
        "cpp/include/vibeqc/semiempirical/methods/indo/indo_engine.hpp", "S2",
        external_suites=(SUITE_GMTKN55, SUITE_GMTKN24),
    ),
    Case(
        "msindo_nddo_hf_direct_energy", "inventory", "MSINDO NDDO HF direct energy",
        "molecular NDDO energy", "C++ NDDO through public run_msindo",
        "cpp/include/vibeqc/semiempirical/methods/indo/indo_engine.hpp", "S2",
        external_suites=(SUITE_GMTKN24,),
    ),
    Case(
        "msindo_sbf3_direct_energy", "inventory", "MSINDO SbF3 direct energy",
        "heavy closed-shell INDO energy", "C++ through public run_msindo",
        "cpp/include/vibeqc/semiempirical/methods/indo/indo_engine.hpp", "S2",
    ),
    Case(
        "msindo_xef2_direct_energy", "inventory", "MSINDO XeF2 direct energy",
        "heavy closed-shell INDO energy", "C++ through public run_msindo",
        "cpp/include/vibeqc/semiempirical/methods/indo/indo_engine.hpp", "S2",
    ),
    Case(
        "periodic_dftb0_he_chain", "inventory", "Periodic DFTB0 He chain",
        "periodic Gamma energy", "C++",
        "cpp/src/semiempirical/periodic_dftb0.cpp", "S4",
    ),
    Case(
        "periodic_dftb0_he_gradient", "inventory",
        "Periodic DFTB0 He-dimer gradient",
        "periodic Gamma analytic gradient", "C++",
        "cpp/src/semiempirical/gradient.cpp", "S4",
    ),
    Case(
        "periodic_dftb0_he_stress", "inventory",
        "Periodic DFTB0 He-dimer stress",
        "periodic Gamma analytic stress", "C++",
        "cpp/src/semiempirical/gradient.cpp", "S4",
    ),
    Case(
        "periodic_dftb0_quartz_sio2", "inventory",
        "Periodic DFTB0 simplified alpha-quartz SiO2",
        "periodic Gamma energy", "C++",
        "cpp/src/semiempirical/periodic_dftb0.cpp", "S4",
        external_suites=(SUITE_DELTA_SOLIDS,),
    ),
    Case(
        "dftb0_benzene_energy", "inventory", "DFTB0 benzene energy",
        "molecular energy", "C++", "cpp/src/semiempirical/dftb0.cpp", "S4",
        external_suites=(SUITE_GMTKN55, SUITE_GMTKN24),
    ),
    Case(
        "periodic_gfn2_h2o_box", "inventory", "Periodic GFN2 H2O molecular limit",
        "periodic Gamma energy", "C++ experimental",
        "cpp/src/semiempirical/methods/xtb/periodic_gfn2.cpp", "S3",
    ),
    Case(
        "periodic_pm6_he_chain", "inventory", "Periodic PM6 He chain",
        "periodic Gamma energy", "C++",
        "cpp/src/semiempirical/methods/nddo/periodic_pm6.cpp", "S5",
    ),
    Case(
        "periodic_pm6_he_dimer_gradient_fd", "inventory",
        "Periodic PM6 He-dimer finite-difference gradient",
        "periodic finite-difference gradient",
        "C++ native FD batch",
        "cpp/src/semiempirical/methods/nddo/periodic_fd_batch.cpp; "
        "cpp/src/semiempirical/methods/nddo/periodic_pm6.cpp", "S5",
    ),
    Case(
        "periodic_pm6_he_dimer_stress_fd", "inventory",
        "Periodic PM6 He-dimer finite-difference stress",
        "periodic finite-difference stress",
        "C++ native FD batch",
        "cpp/src/semiempirical/methods/nddo/periodic_fd_batch.cpp; "
        "cpp/src/semiempirical/methods/nddo/periodic_pm6.cpp", "S5",
    ),
    Case(
        "msindo_cosmo_h2o", "inventory", "MSINDO COSMO H2O",
        "solvation single point", "C++ B-matrix/vector ops + Python COSMO SCF",
        "cpp/include/vibeqc/semiempirical/methods/indo/indo_engine.hpp; "
        "python/vibeqc/semiempirical/methods/msindo_cosmo.py", "S2",
    ),
    Case(
        "msindo_cis_h2o_singlet", "inventory", "MSINDO CIS H2O singlets",
        "CIS/TDA excited states", "Python-reference CIS/TDA matrix build",
        "python/vibeqc/semiempirical/methods/msindo.py; "
        "python/vibeqc/excited.py", "S2",
    ),
    Case(
        "ccm_he7_1d_noewald", "inventory", "MSINDO CCM He7 1D no-Ewald",
        "cyclic-cluster energy", "C++ through public run_ccm",
        "cpp/include/vibeqc/semiempirical/methods/indo/indo_engine.hpp", "S2",
    ),
    Case(
        "ccm_hf6_1d_madelung", "inventory", "MSINDO CCM HF6 1D Madelung",
        "cyclic-cluster energy", "C++ through public run_ccm",
        "cpp/include/vibeqc/semiempirical/methods/indo/indo_engine.hpp", "S2",
    ),
    Case(
        "ccm_mgo_2d_noewald", "inventory", "MSINDO CCM MgO 2D no-Ewald",
        "cyclic-cluster energy", "C++ through public run_ccm",
        "cpp/include/vibeqc/semiempirical/methods/indo/indo_engine.hpp", "S2",
        external_suites=(SUITE_DELTA_SOLIDS,),
    ),
    Case(
        "ccm_mgo_3d_noewald", "inventory", "MSINDO CCM MgO 3D no-Ewald",
        "cyclic-cluster energy", "C++ through public run_ccm",
        "cpp/include/vibeqc/semiempirical/methods/indo/indo_engine.hpp", "S2",
        external_suites=(SUITE_DELTA_SOLIDS,),
    ),
)

CASE_BY_KEY = {case.key: case for case in CASES}


def _case_keys_for_external_suite(suite: str) -> list[str]:
    return [case.key for case in CASES if suite in case.external_suites]


def _rss_mb() -> float | None:
    try:
        import resource
    except ImportError:
        return None
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def _mol_bohr(atoms: list[tuple[int, list[float]]], charge: int = 0, multiplicity: int = 1):
    import vibeqc as vq

    return vq.Molecule(
        [vq.Atom(z, xyz) for z, xyz in atoms],
        charge=charge,
        multiplicity=multiplicity,
    )


def _h2o_bohr():
    return _mol_bohr(
        [
            (8, [0.0, 0.0, 0.0]),
            (1, [1.430522676, 1.107379485, 0.0]),
            (1, [-1.430522676, 1.107379485, 0.0]),
        ]
    )


def _ch4_bohr():
    a = 2.0 / (3.0 ** 0.5)
    return _mol_bohr(
        [
            (6, [0.0, 0.0, 0.0]),
            (1, [a, a, a]),
            (1, [-a, -a, a]),
            (1, [-a, a, -a]),
            (1, [a, -a, -a]),
        ]
    )


def _nh3_bohr():
    return _mol_bohr(
        [
            (7, [0.0, 0.0, 0.1147]),
            (1, [0.0, 1.7670, -0.4810]),
            (1, [1.5303, -0.8835, -0.4810]),
            (1, [-1.5303, -0.8835, -0.4810]),
        ]
    )


def _benzene_bohr():
    ang_to_bohr = 1.8897261254578281
    r_c = 1.3915 * ang_to_bohr
    r_h = 2.4715 * ang_to_bohr
    atoms = []
    for i in range(6):
        theta = math.pi / 3.0 * i
        atoms.append((6, [r_c * math.cos(theta), r_c * math.sin(theta), 0.0]))
        atoms.append((1, [r_h * math.cos(theta), r_h * math.sin(theta), 0.0]))
    return _mol_bohr(atoms)


def _periodic_chain(z: int):
    import numpy as np
    import vibeqc as vq

    return vq.PeriodicSystem(
        1,
        np.diag([6.0, 24.0, 24.0]),
        [vq.Atom(z, [0.0, 0.0, 0.0])],
        0,
        1,
    )


def _periodic_he_dimer_cell():
    import numpy as np
    import vibeqc as vq

    return vq.PeriodicSystem(
        3,
        np.eye(3) * 10.0,
        [vq.Atom(2, [0.0, 0.0, 0.0]), vq.Atom(2, [2.0, 0.0, 0.0])],
        0,
        1,
    )


def _periodic_h2o_box():
    import numpy as np
    import vibeqc as vq

    mol = _h2o_bohr()
    return vq.PeriodicSystem(
        3,
        np.eye(3) * 20.0,
        list(mol.atoms),
        0,
        1,
    )


def _periodic_quartz_sio2():
    import numpy as np
    import vibeqc as vq

    ang_to_bohr = 1.8897261254578281
    a = 4.913 * ang_to_bohr
    c = 5.405 * ang_to_bohr
    lattice = np.array(
        [
            [a, 0.0, 0.0],
            [-a / 2.0, a * np.sqrt(3.0) / 2.0, 0.0],
            [0.0, 0.0, c],
        ]
    )
    atoms = [
        vq.Atom(14, [0.47 * a, 0.0, 0.0]),
        vq.Atom(14, [-0.47 * a / 2.0, 0.47 * a * np.sqrt(3.0) / 2.0, c / 3.0]),
        vq.Atom(14, [-0.47 * a / 2.0, -0.47 * a * np.sqrt(3.0) / 2.0, 2.0 * c / 3.0]),
        vq.Atom(8, [0.41 * a, 0.27 * a, 0.12 * c]),
        vq.Atom(8, [0.27 * a, 0.41 * a, -0.12 * c]),
        vq.Atom(
            8,
            [
                -0.41 * a / 2.0 - 0.27 * a,
                0.41 * a * np.sqrt(3.0) / 2.0 - 0.27 * a * np.sqrt(3.0) / 2.0,
                c / 3.0,
            ],
        ),
        vq.Atom(
            8,
            [
                -0.27 * a / 2.0 - 0.41 * a,
                0.27 * a * np.sqrt(3.0) / 2.0 - 0.41 * a * np.sqrt(3.0) / 2.0,
                c / 3.0,
            ],
        ),
        vq.Atom(
            8,
            [
                -0.41 * a / 2.0 + 0.16 * a,
                0.41 * a * np.sqrt(3.0) / 2.0 + 0.16 * a * np.sqrt(3.0) / 2.0,
                2.0 * c / 3.0,
            ],
        ),
        vq.Atom(
            8,
            [
                -0.27 * a / 2.0 + 0.16 * a,
                0.27 * a * np.sqrt(3.0) / 2.0 + 0.16 * a * np.sqrt(3.0) / 2.0,
                2.0 * c / 3.0,
            ],
        ),
    ]
    return vq.PeriodicSystem(3, lattice.T, atoms, 0, 1)


def _case_dftb0_h2o_energy() -> dict[str, Any]:
    from vibeqc.semiempirical import DFTB0Model

    model = DFTB0Model(_h2o_bohr())
    return {"energy_ha": float(model.energy())}


def _case_dftb0_h2o_gradient() -> dict[str, Any]:
    import numpy as np
    from vibeqc.semiempirical import DFTB0Model
    from vibeqc.semiempirical.parameters import default_parameters

    grad = np.asarray(DFTB0Model(_h2o_bohr(), params=default_parameters()).gradient())
    return {
        "gradient_norm": float(np.linalg.norm(grad)),
        "shape": list(grad.shape),
    }


def _case_dftb0_h2o_preoptimize() -> dict[str, Any]:
    import numpy as np
    from vibeqc.semiempirical import DFTB0Model
    from vibeqc.semiempirical.parameters import default_parameters
    from vibeqc.semiempirical.preoptimize import preoptimize_molecule

    mol = _h2o_bohr()
    params = default_parameters()
    e_initial = float(DFTB0Model(mol, params=params).energy())
    mol_opt = preoptimize_molecule(mol, method="dftb0", fmax=0.1, max_steps=30)
    model_opt = DFTB0Model(mol_opt, params=params)
    e_final = float(model_opt.energy())
    grad = np.asarray(model_opt.gradient())
    return {
        "energy_initial_ha": e_initial,
        "energy_final_ha": e_final,
        "energy_delta_ha": e_final - e_initial,
        "gradient_rms": float(np.sqrt(np.mean(grad**2))),
        "n_atoms": len(mol_opt.atoms),
    }


def _case_dftb0_benzene_energy() -> dict[str, Any]:
    from vibeqc.semiempirical import DFTB0Model

    model = DFTB0Model(_benzene_bohr())
    return {"energy_ha": float(model.energy())}


def _case_scc_dftb_h2o_energy() -> dict[str, Any]:
    from vibeqc.semiempirical import SCCDFTBModel

    model = SCCDFTBModel(_h2o_bohr())
    energy = float(model.energy())
    return {"energy_ha": energy}


def _case_gfn2_h2o_energy() -> dict[str, Any]:
    from vibeqc.semiempirical.methods.gfn2 import GFN2Model
    from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

    model = GFN2Model(_h2o_bohr(), load_gfn2_params(), warn=False)
    return {"energy_ha": float(model.energy())}


def _case_gfn2_h2o_gradient() -> dict[str, Any]:
    import numpy as np
    from vibeqc._vibeqc_core import semiempirical as _se
    from vibeqc._vibeqc_core.semiempirical import xtb as _xtb
    from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

    mol = _h2o_bohr()
    params = load_gfn2_params()
    result = _xtb.run_gfn2_xtb(mol, params)
    grad = np.asarray(_se.compute_gfn2_gradient(mol, result, params))
    return {
        "gradient_norm": float(np.linalg.norm(grad)),
        "shape": list(grad.shape),
        "converged": bool(result.converged),
        "n_iter": int(result.n_iter),
    }


def _case_gfn2_nh3_energy() -> dict[str, Any]:
    from vibeqc.semiempirical.methods.gfn2 import GFN2Model
    from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

    model = GFN2Model(_nh3_bohr(), load_gfn2_params(), warn=False)
    return {"energy_ha": float(model.energy())}


def _case_gfn2_benzene_energy() -> dict[str, Any]:
    from vibeqc.semiempirical.methods.gfn2 import GFN2Model
    from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

    model = GFN2Model(_benzene_bohr(), load_gfn2_params(), warn=False)
    return {"energy_ha": float(model.energy())}


def _case_gfn2_ionic_diatomic(z1: int, z2: int, r_bohr: float) -> dict[str, Any]:
    from vibeqc._vibeqc_core.semiempirical import xtb as _xtb
    from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

    mol = _mol_bohr([(z1, [0.0, 0.0, 0.0]), (z2, [r_bohr, 0.0, 0.0])])
    result = _xtb.run_gfn2_xtb(mol, load_gfn2_params())
    return {
        "energy_ha": float(result.energy),
        "converged": bool(result.converged),
        "n_iter": int(result.n_iter),
        "charges": [float(q) for q in result.charges],
    }


def _case_gfn2_nacl_energy() -> dict[str, Any]:
    return _case_gfn2_ionic_diatomic(11, 17, 12.0)


def _case_gfn2_lif_energy() -> dict[str, Any]:
    return _case_gfn2_ionic_diatomic(3, 9, 12.0)


def _case_gfn2_lih_energy() -> dict[str, Any]:
    return _case_gfn2_ionic_diatomic(3, 1, 8.0)


def _case_pm6_h2o_energy() -> dict[str, Any]:
    from vibeqc.semiempirical.methods.pm6 import PM6Model

    model = PM6Model(_h2o_bohr())
    return {"energy_ha": float(model.energy())}


def _case_pm6_h2o_gradient_fd() -> dict[str, Any]:
    import numpy as np
    from vibeqc._vibeqc_core.semiempirical import nddo as _nddo
    from vibeqc.semiempirical.methods.pm6_params import load_pm6_params

    grad = np.asarray(_nddo.compute_pm6_gradient_fd(_h2o_bohr(), load_pm6_params(), 0.001))
    return {
        "gradient_norm": float(np.linalg.norm(grad)),
        "shape": list(grad.shape),
    }


def _case_pm6_ch4_energy() -> dict[str, Any]:
    from vibeqc.semiempirical.methods.pm6 import PM6Model

    model = PM6Model(_ch4_bohr())
    return {"energy_ha": float(model.energy())}


def _case_pm6_nh3_energy() -> dict[str, Any]:
    from vibeqc.semiempirical.methods.pm6 import PM6Model

    model = PM6Model(_nh3_bohr())
    return {"energy_ha": float(model.energy())}


def _case_om2_h2o_energy() -> dict[str, Any]:
    from vibeqc.semiempirical.methods.omx import OMxModel

    model = OMxModel(_h2o_bohr(), variant="om2")
    return {"energy_ha": float(model.energy())}


def _case_om2_ch4_energy() -> dict[str, Any]:
    from vibeqc.semiempirical.methods.omx import OMxModel

    model = OMxModel(_ch4_bohr(), variant="om2")
    return {"energy_ha": float(model.energy())}


def _case_om2_nh3_energy() -> dict[str, Any]:
    from vibeqc.semiempirical.methods.omx import OMxModel

    model = OMxModel(_nh3_bohr(), variant="om2")
    return {"energy_ha": float(model.energy())}


def _case_msindo_h2o_cpp_energy() -> dict[str, Any]:
    import tempfile
    import vibeqc as vq

    mol = _h2o_bohr()
    out = str(Path(tempfile.gettempdir()) / "vibeqc_msindo_hotpath")
    result = vq.run_job(mol, method="msindo", output=out, verbose=0)
    return {"energy_ha": float(result.energy), "converged": bool(result.converged)}


def _case_msindo_h2o_direct_energy() -> dict[str, Any]:
    from vibeqc.semiempirical.methods.msindo import run_msindo

    result = run_msindo(
        [8, 1, 1],
        [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]],
    )
    return {"energy_ha": float(result.total_energy), "converged": bool(result.converged)}


def _case_msindo_nddo_hf_direct_energy() -> dict[str, Any]:
    from vibeqc.semiempirical.methods.msindo import run_msindo

    result = run_msindo([1, 9], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.917]], nddo=True)
    return {"energy_ha": float(result.total_energy), "converged": bool(result.converged)}


def _case_msindo_sbf3_direct_energy() -> dict[str, Any]:
    from vibeqc.semiempirical.methods.msindo import run_msindo

    result = run_msindo(
        [51, 9, 9, 9],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.88], [1.78, 0.0, -0.6], [-1.78, 0.0, -0.6]],
    )
    return {"energy_ha": float(result.total_energy), "converged": bool(result.converged)}


def _case_msindo_xef2_direct_energy() -> dict[str, Any]:
    from vibeqc.semiempirical.methods.msindo import run_msindo

    result = run_msindo([54, 9, 9], [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, -2.0]])
    return {"energy_ha": float(result.total_energy), "converged": bool(result.converged)}


def _case_periodic_dftb0_he_chain() -> dict[str, Any]:
    from vibeqc._vibeqc_core import semiempirical as _se
    from vibeqc.semiempirical.parameters import default_parameters

    result = _se.run_dftb0_gamma(_periodic_chain(2), default_parameters())
    return {"energy_ha": float(result.energy), "n_cells": int(result.n_cells)}


def _case_periodic_dftb0_quartz_sio2() -> dict[str, Any]:
    from vibeqc._vibeqc_core import semiempirical as _se
    from vibeqc.semiempirical.parameters import default_parameters

    result = _se.run_dftb0_gamma(_periodic_quartz_sio2(), default_parameters())
    return {"energy_ha": float(result.energy), "n_cells": int(result.n_cells)}


def _case_periodic_dftb0_he_gradient() -> dict[str, Any]:
    import numpy as np
    from vibeqc._vibeqc_core import semiempirical as _se
    from vibeqc.semiempirical.parameters import default_parameters

    system = _periodic_he_dimer_cell()
    params = default_parameters()
    result = _se.run_dftb0_gamma(system, params)
    grad = np.asarray(_se.compute_periodic_dftb0_gradient(system, result, params))
    return {
        "gradient_norm": float(np.linalg.norm(grad)),
        "shape": list(grad.shape),
        "n_cells": int(result.n_cells),
    }


def _case_periodic_dftb0_he_stress() -> dict[str, Any]:
    import numpy as np
    from vibeqc._vibeqc_core import semiempirical as _se
    from vibeqc.semiempirical.parameters import default_parameters

    system = _periodic_he_dimer_cell()
    params = default_parameters()
    result = _se.run_dftb0_gamma(system, params)
    stress = np.asarray(_se.compute_periodic_dftb0_stress(system, result, params))
    return {
        "stress_norm": float(np.linalg.norm(stress)),
        "shape": list(stress.shape),
        "n_cells": int(result.n_cells),
    }


def _case_periodic_gfn2_h2o_box() -> dict[str, Any]:
    from vibeqc._vibeqc_core.semiempirical import xtb as _xtb
    from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

    result = _xtb.run_gfn2_xtb_gamma(_periodic_h2o_box(), load_gfn2_params())
    return {"energy_ha": float(result.energy), "converged": bool(result.converged)}


def _case_periodic_pm6_he_chain() -> dict[str, Any]:
    from vibeqc.semiempirical.methods.periodic_pm6 import run_pm6_gamma
    from vibeqc.semiempirical.methods.pm6_params import load_pm6_mopac_params

    result = run_pm6_gamma(_periodic_chain(2), load_pm6_mopac_params())
    return {"energy_ha": float(result.energy), "converged": bool(result.converged)}


def _case_periodic_pm6_he_dimer_gradient_fd() -> dict[str, Any]:
    import numpy as np
    from vibeqc.semiempirical.methods.periodic_pm6 import (
        compute_pm6_gamma_gradient_fd,
    )
    from vibeqc.semiempirical.methods.pm6_params import load_pm6_mopac_params

    grad = np.asarray(
        compute_pm6_gamma_gradient_fd(
            _periodic_he_dimer_cell(),
            load_pm6_mopac_params(),
        )
    )
    return {
        "gradient_norm": float(np.linalg.norm(grad)),
        "shape": list(grad.shape),
    }


def _case_periodic_pm6_he_dimer_stress_fd() -> dict[str, Any]:
    import numpy as np
    from vibeqc.semiempirical.methods.periodic_pm6 import compute_pm6_gamma_stress_fd
    from vibeqc.semiempirical.methods.pm6_params import load_pm6_mopac_params

    stress = np.asarray(
        compute_pm6_gamma_stress_fd(
            _periodic_he_dimer_cell(),
            load_pm6_mopac_params(),
        )
    )
    return {
        "stress_norm": float(np.linalg.norm(stress)),
        "shape": list(stress.shape),
    }


def _case_msindo_cosmo_h2o() -> dict[str, Any]:
    from vibeqc.semiempirical.methods.msindo_cosmo import msindo_cosmo

    result = msindo_cosmo(
        [8, 1, 1],
        [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]],
        epsilon=78.39,
    )
    cavity = getattr(result, "cavity", None)
    return {
        "energy_ha": float(result.total_energy),
        "e_gas_ha": float(result.e_gas),
        "e_solv_ha": float(result.e_solv),
        "e_pol_ha": float(result.e_pol),
        "converged": bool(result.converged),
        "n_iter": int(result.n_iter),
        "cavity_points": int(getattr(cavity, "n_points", 0) or 0),
        "cavity_kind": type(cavity).__name__ if cavity is not None else "",
    }


def _case_msindo_cis_h2o_singlet() -> dict[str, Any]:
    import numpy as np
    from vibeqc.semiempirical.methods.msindo import msindo_cis

    result = msindo_cis(
        [8, 1, 1],
        [[0.0, 0.0, 0.0], [0.0, 0.757, 0.587], [0.0, -0.757, 0.587]],
        spin="singlet",
        n_states=4,
    )
    occ, vir, weight = result.dominant_transition(0)
    amplitudes = np.asarray(result.amplitudes)
    return {
        "excitation_energies_ev": [
            float(x) for x in np.asarray(result.excitation_energies_ev)
        ],
        "amplitude_shape": list(amplitudes.shape),
        "n_occ": int(result.n_occ),
        "n_vir": int(result.n_vir),
        "dominant_transition": [int(occ), int(vir), float(weight)],
        "spin": result.spin,
    }


def _case_ccm_he7_1d_noewald() -> dict[str, Any]:
    from vibeqc.semiempirical.methods.msindo_ccm import run_ccm

    result = run_ccm(
        [2, 2, 2, 2, 2, 2, 2],
        [[i * 2.0, 0.0, 0.0] for i in range(7)],
        [[14.0, 0.0, 0.0]],
        madelung=False,
    )
    return {
        "energy_ha": float(result.total_energy),
        "converged": bool(result.converged),
        "n_iter": int(result.n_iter),
    }


def _case_ccm_hf6_1d_madelung() -> dict[str, Any]:
    from vibeqc.semiempirical.methods.msindo_ccm import run_ccm

    result = run_ccm(
        [1, 9, 1, 9, 1, 9],
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [3.5, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
        ],
        [[7.5, 0.0, 0.0]],
        madelung=True,
    )
    return {
        "energy_ha": float(result.total_energy),
        "converged": bool(result.converged),
        "n_iter": int(result.n_iter),
    }


def _mgo_ccm_fixture():
    a_mgo = 2.105
    z = [12, 8, 12, 8, 8, 12, 8, 12]
    coords = [
        [0.0, 0.0, 0.0],
        [a_mgo, 0.0, 0.0],
        [0.0, a_mgo, 0.0],
        [a_mgo, a_mgo, 0.0],
        [0.0, 0.0, a_mgo],
        [a_mgo, 0.0, a_mgo],
        [0.0, a_mgo, a_mgo],
        [a_mgo, a_mgo, a_mgo],
    ]
    return z, coords, a_mgo


def _case_ccm_mgo_2d_noewald() -> dict[str, Any]:
    from vibeqc.semiempirical.methods.msindo_ccm import run_ccm

    z, coords, a_mgo = _mgo_ccm_fixture()
    result = run_ccm(
        z,
        coords,
        [[2.0 * a_mgo, 0.0, 0.0], [0.0, 2.0 * a_mgo, 0.0]],
        madelung=False,
    )
    return {
        "energy_ha": float(result.total_energy),
        "converged": bool(result.converged),
        "n_iter": int(result.n_iter),
    }


def _case_ccm_mgo_3d_noewald() -> dict[str, Any]:
    from vibeqc.semiempirical.methods.msindo_ccm import run_ccm

    z, coords, a_mgo = _mgo_ccm_fixture()
    result = run_ccm(
        z,
        coords,
        [
            [2.0 * a_mgo, 0.0, 0.0],
            [0.0, 2.0 * a_mgo, 0.0],
            [0.0, 0.0, 2.0 * a_mgo],
        ],
        madelung=False,
    )
    return {
        "energy_ha": float(result.total_energy),
        "converged": bool(result.converged),
        "n_iter": int(result.n_iter),
    }


RUNNERS: dict[str, Callable[[], dict[str, Any]]] = {
    "dftb0_h2o_energy": _case_dftb0_h2o_energy,
    "dftb0_h2o_gradient": _case_dftb0_h2o_gradient,
    "dftb0_h2o_preoptimize": _case_dftb0_h2o_preoptimize,
    "dftb0_benzene_energy": _case_dftb0_benzene_energy,
    "scc_dftb_h2o_energy": _case_scc_dftb_h2o_energy,
    "gfn2_h2o_energy": _case_gfn2_h2o_energy,
    "gfn2_h2o_gradient": _case_gfn2_h2o_gradient,
    "gfn2_nh3_energy": _case_gfn2_nh3_energy,
    "gfn2_benzene_energy": _case_gfn2_benzene_energy,
    "gfn2_nacl_energy": _case_gfn2_nacl_energy,
    "gfn2_lif_energy": _case_gfn2_lif_energy,
    "gfn2_lih_energy": _case_gfn2_lih_energy,
    "pm6_h2o_energy": _case_pm6_h2o_energy,
    "pm6_h2o_gradient_fd": _case_pm6_h2o_gradient_fd,
    "pm6_ch4_energy": _case_pm6_ch4_energy,
    "pm6_nh3_energy": _case_pm6_nh3_energy,
    "om2_h2o_energy": _case_om2_h2o_energy,
    "om2_ch4_energy": _case_om2_ch4_energy,
    "om2_nh3_energy": _case_om2_nh3_energy,
    "msindo_h2o_cpp_energy": _case_msindo_h2o_cpp_energy,
    "msindo_h2o_direct_energy": _case_msindo_h2o_direct_energy,
    "msindo_nddo_hf_direct_energy": _case_msindo_nddo_hf_direct_energy,
    "msindo_sbf3_direct_energy": _case_msindo_sbf3_direct_energy,
    "msindo_xef2_direct_energy": _case_msindo_xef2_direct_energy,
    "periodic_dftb0_he_chain": _case_periodic_dftb0_he_chain,
    "periodic_dftb0_he_gradient": _case_periodic_dftb0_he_gradient,
    "periodic_dftb0_he_stress": _case_periodic_dftb0_he_stress,
    "periodic_dftb0_quartz_sio2": _case_periodic_dftb0_quartz_sio2,
    "periodic_gfn2_h2o_box": _case_periodic_gfn2_h2o_box,
    "periodic_pm6_he_chain": _case_periodic_pm6_he_chain,
    "periodic_pm6_he_dimer_gradient_fd": _case_periodic_pm6_he_dimer_gradient_fd,
    "periodic_pm6_he_dimer_stress_fd": _case_periodic_pm6_he_dimer_stress_fd,
    "msindo_cosmo_h2o": _case_msindo_cosmo_h2o,
    "msindo_cis_h2o_singlet": _case_msindo_cis_h2o_singlet,
    "ccm_he7_1d_noewald": _case_ccm_he7_1d_noewald,
    "ccm_hf6_1d_madelung": _case_ccm_hf6_1d_madelung,
    "ccm_mgo_2d_noewald": _case_ccm_mgo_2d_noewald,
    "ccm_mgo_3d_noewald": _case_ccm_mgo_3d_noewald,
}


def _child(case_key: str, repeat: int) -> int:
    warnings.filterwarnings("ignore")
    case = CASE_BY_KEY[case_key]
    runner = RUNNERS[case_key]
    payload: dict[str, Any] = {"case": asdict(case), "status": "ok"}
    try:
        last: dict[str, Any] = {}
        t0 = time.perf_counter()
        for _ in range(repeat):
            last = runner()
        payload["wall_time_s"] = (time.perf_counter() - t0) / max(repeat, 1)
        payload["result"] = last
        payload["rss_max_mb"] = _rss_mb()
    except Exception as exc:  # noqa: BLE001
        payload.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "rss_max_mb": _rss_mb(),
            }
        )
    print(json.dumps(payload, sort_keys=True))
    return 0


def _select_cases(suite: str, only: str | None) -> list[Case]:
    if only:
        keys = [item.strip() for item in only.split(",") if item.strip()]
        unknown = [key for key in keys if key not in CASE_BY_KEY]
        if unknown:
            raise SystemExit(f"unknown case(s): {', '.join(unknown)}")
        return [CASE_BY_KEY[key] for key in keys]
    if suite == "all":
        return list(CASES)
    selected = [case for case in CASES if case.suite == suite]
    if not selected:
        raise SystemExit(f"unknown suite {suite!r}")
    return selected


def _run_parent(cases: list[Case], repeat: int) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        proc = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--_child",
                case.key,
                "--repeat",
                str(repeat),
            ],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            rows.append(
                {
                    "case": asdict(case),
                    "status": "driver-error",
                    "error": proc.stderr.strip() or proc.stdout.strip(),
                }
            )
            continue
        try:
            rows.append(json.loads(proc.stdout.strip().splitlines()[-1]))
        except (IndexError, json.JSONDecodeError) as exc:
            rows.append(
                {
                    "case": asdict(case),
                    "status": "parse-error",
                    "error": f"{type(exc).__name__}: {exc}; stdout={proc.stdout!r}",
                }
            )
    return rows


def _print_list() -> None:
    print("| key | suite | backend | priority | external suites | hot path |")
    print("|---|---|---|---|---|---|")
    for case in CASES:
        external = ", ".join(case.external_suites) if case.external_suites else "-"
        print(
            f"| `{case.key}` | {case.suite} | {case.backend} | "
            f"{case.priority} | {external} | `{case.hot_path}` |"
        )


def _print_suite_list() -> None:
    print("| external suite | scope in this harness | compact runnable cases |")
    print("|---|---|---|")
    for suite, scope in ARTICLE_BENCHMARK_SCOPES.items():
        keys = _case_keys_for_external_suite(suite)
        case_text = ", ".join(f"`{key}`" for key in keys) if keys else "-"
        print(f"| {suite} | {scope} | {case_text} |")


def _print_markdown(rows: list[dict[str, Any]]) -> None:
    print("| case | status | wall s | max RSS MB | backend | result |")
    print("|---|---:|---:|---:|---|---|")
    for row in rows:
        case = row["case"]
        wall = row.get("wall_time_s")
        rss = row.get("rss_max_mb")
        result = row.get("result") or row.get("error", "")
        wall_s = f"{wall:.6f}" if isinstance(wall, float) else "n/a"
        rss_s = f"{rss:.1f}" if isinstance(rss, float) else "n/a"
        print(
            f"| `{case['key']}` | {row['status']} | {wall_s} | {rss_s} | "
            f"{case['backend']} | `{json.dumps(result, sort_keys=True)}` |"
        )


def _row_failed(row: dict[str, Any]) -> bool:
    """Return whether a parsed benchmark row should fail the harness."""
    if row.get("status") != "ok":
        return True
    result = row.get("result")
    if isinstance(result, dict) and result.get("converged") is False:
        return True
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="list available cases and exit")
    parser.add_argument(
        "--list-suites",
        action="store_true",
        help="list article external-suite mappings and exit",
    )
    parser.add_argument("--suite", default="smoke", help="suite to run: smoke, inventory, all")
    parser.add_argument("--only", help="comma-separated case keys to run")
    parser.add_argument("--repeat", type=int, default=1, help="repeat each case in the child process")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of markdown")
    parser.add_argument("--_child", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args._child:
        if args._child not in CASE_BY_KEY:
            raise SystemExit(f"unknown child case {args._child!r}")
        return _child(args._child, max(args.repeat, 1))

    if args.list:
        _print_list()
        return 0
    if args.list_suites:
        _print_suite_list()
        return 0

    cases = _select_cases(args.suite, args.only)
    rows = _run_parent(cases, max(args.repeat, 1))
    if args.json:
        print(json.dumps({"cases": rows}, indent=2, sort_keys=True))
    else:
        _print_markdown(rows)
    return 1 if any(_row_failed(row) for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
