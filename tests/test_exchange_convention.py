"""Exchange-``q=0`` convention provenance (§0.5 item 1 reproducibility gate).

On a finite BvK torus the Coulomb/``J`` kernel is forced, but the exchange-``q=0``
seam is an additional finite-size convention (``BvK-ewald`` vs ``strict-zero-mode``)
that shifts the HF gap — and hence every MP2/KMP2/CCSD/DLPNO denominator. So an
``A``/``B``/``KMP2`` comparison is only meaningful at a matched convention. This
pins: the convention maps correctly, lands in the ``.system`` manifest of a real
periodic run, and the comparison assertion fires on a deliberate mismatch.

Ref: docs/manuscripts/aiccm_a_position.md §0.5 item 1.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc.periodic.exchange_convention import (
    BVK_EWALD,
    STRICT_ZERO,
    assert_matched_exchange_q0,
    exchange_q0_label,
    read_exchange_q0,
)


# --------------------------------------------------------------------------- #
# The exxdiv -> convention map.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("exxdiv,expected", [
    ("ewald", BVK_EWALD),
    ("Ewald", BVK_EWALD),
    ("bvk-ewald", BVK_EWALD),
    (None, STRICT_ZERO),
    ("none", STRICT_ZERO),
    ("", STRICT_ZERO),
    ("strict-zero-mode", STRICT_ZERO),
])
def test_exchange_q0_label_map(exxdiv, expected):
    assert exchange_q0_label(exxdiv) == expected


def test_unknown_exxdiv_recorded_verbatim():
    """An unrecognised exxdiv is captured verbatim (never silently lost)."""
    assert exchange_q0_label("vcut-3d") == "exxdiv:vcut-3d"


# --------------------------------------------------------------------------- #
# The comparison assertion.
# --------------------------------------------------------------------------- #
def test_assert_matched_passes_and_returns_label():
    assert assert_matched_exchange_q0([BVK_EWALD, BVK_EWALD, BVK_EWALD]) == BVK_EWALD


def test_assert_matched_raises_on_mismatch():
    with pytest.raises(ValueError, match="exchange-q=0 convention mismatch"):
        assert_matched_exchange_q0([BVK_EWALD, STRICT_ZERO])


def test_assert_matched_empty_raises():
    with pytest.raises(ValueError):
        assert_matched_exchange_q0([])


# --------------------------------------------------------------------------- #
# Integration: a real periodic SCF lands [run].exchange_q0 in .system.
# --------------------------------------------------------------------------- #
def test_periodic_run_lands_exchange_q0(tmp_path):
    import vibeqc as vq
    from vibeqc import Atom, PeriodicSystem

    sysp = PeriodicSystem(
        3, np.diag([6.0, 6.0, 6.0]),
        [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])], charge=0, multiplicity=1)
    b = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    stem = str(tmp_path / "h2box")
    vq.run_periodic_job(
        sysp, b, method="RHF", kpoints=(1, 1, 1), output=stem, max_iter=50,
        conv_tol_energy=1e-6, write_molden_file=False, write_density=False,
        write_cif_file=False, write_xsf_structure_file=False,
        write_poscar_file=False, write_xyz_file=False, write_population_file=False)

    sysfile = tmp_path / "h2box.system"
    assert sysfile.is_file()
    # Production default exxdiv='ewald' -> BvK-ewald.
    assert read_exchange_q0(sysfile) == BVK_EWALD
    # The comparison helper consumes the manifest path directly.
    assert assert_matched_exchange_q0([sysfile, BVK_EWALD]) == BVK_EWALD
    with pytest.raises(ValueError):
        assert_matched_exchange_q0([sysfile, STRICT_ZERO])


def test_read_exchange_q0_missing(tmp_path):
    assert read_exchange_q0(tmp_path / "nope.system") is None
