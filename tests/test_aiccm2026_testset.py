"""CI coverage for the AICCM-2026 benchmark geometry registry (studies/aiccm-2026/).

The paper test set lives in ``studies/aiccm-2026/testset.py`` (not a package). These
tests pin that every system still builds and that its cyclic-cluster overlap is
positive definite at the benchmark cluster size (the Fig-6 guard) — so an edit
to the registry can't silently ship a broken or ill-conditioned entry (cf. the
old lih-chain a=2.0 Å, which was non-PD).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane

_TESTSET = Path(__file__).resolve().parent.parent / "studies" / "aiccm-2026" / "testset.py"
_spec = importlib.util.spec_from_file_location("aiccm2026_testset", _TESTSET)
testset = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(testset)


_HAS_ASE = importlib.util.find_spec("ase") is not None


def test_all_geometries_build():
    """Every registry system builds a PeriodicSystem with the expected atom count.

    The Bravais-coverage crystals are built from grounded space-group specs via
    ASE (the ``[ase]`` extra); skipped if ASE is absent so the core suite still
    runs.
    """
    built = 0
    for name, s in testset.SYSTEMS.items():
        try:
            sysobj = testset.build(name)
        except ImportError:                 # ASE-based crystal builder, [ase] extra
            continue
        n = len(list(sysobj.unit_cell_molecule().atoms))
        assert n == s["atoms"], f"{name}: built {n} atoms, registry says {s['atoms']}"
        assert sysobj.dim == s["dim"]
        built += 1
    assert built >= 18                       # the ASE-free core always builds


@pytest.mark.parametrize("name", list(testset.SYSTEMS))
@pytest.mark.slow
def test_overlap_positive_definite(name):
    """The cyclic-cluster overlap S^CCM is well-conditioned at the benchmark nrep."""
    from vibeqc.periodic.ccm import CCMSystem, ccm_overlap
    s = testset.SYSTEMS[name]
    try:
        ccm = CCMSystem(testset.build(name), tuple(s["nrep_4c"]), s["basis"])
    except ImportError:
        pytest.skip("Bravais-coverage crystal needs the [ase] extra")
    emin = float(np.linalg.eigvalsh(ccm_overlap(ccm))[0])
    assert emin > 1e-3, f"{name}: S^CCM min eig {emin:.2e} (ill-conditioned)"


def test_bravais_lattice_coverage():
    """The expanded set spans the crystal systems / Bravais lattices."""
    bravais = {s["bravais"] for s in testset.SYSTEMS.values() if "bravais" in s}
    # the explicitly-tagged expansion: triclinic, monoclinic-C, orthorhombic-C,
    # tetragonal-I, rhombohedral, hexagonal, cubic P/I/F
    assert {"aP", "mS", "oS", "tI", "hR", "hP", "cP", "cI", "cF"} <= bravais
