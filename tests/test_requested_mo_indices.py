"""``requested_mo_indices`` MO-label resolution + occupation fallback.

The function resolves user-facing MO labels (``"homo"`` / ``"lumo"`` /
``"homo-2"`` / int indices) against an SCF result. Two code paths:

* ``result`` carries ``mo_occupations`` or ``mo_occ`` -> read them
  directly.
* Neither attribute present (the current RHF / RKS result type) ->
  derive a closed-shell occupation pattern from
  the physical electron count, the result's replaced-core provenance,
  and ``result.mo_coeffs.shape[1]``.

The 2-arg form (no ``molecule``) must raise a clear ``ValueError``
when occupations cannot be inferred from ``result`` alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from vibeqc.output.formats.cube import requested_mo_indices


class _DummyMolecule:
    """Minimal molecule shim exposing ``n_electrons()`` for the
    derived-occupation fallback path. Avoids spinning up a real
    vibe-qc ``Molecule`` for what is fundamentally a unit test on
    the label-resolution helper."""

    def __init__(self, n_electrons: int) -> None:
        self._n = int(n_electrons)

    def n_electrons(self) -> int:
        return self._n


class _DummyResultWithOcc:
    def __init__(self, occ: np.ndarray, n_mo: int | None = None) -> None:
        self.mo_occupations = np.asarray(occ, dtype=float)
        n = n_mo if n_mo is not None else len(occ)
        # ``mo_coeffs`` shape: (n_basis, n_mo). The function only
        # reads ``.shape[1]``, so n_basis is arbitrary.
        self.mo_coeffs = np.zeros((n, n))


class _DummyResultNoOcc:
    """Shape mimicking the live RHF / RKS results: ``mo_coeffs`` is
    present, ``mo_occupations`` / ``mo_occ`` are not."""

    def __init__(self, n_mo: int, *, ecp_total_ncore: int = 0) -> None:
        self.mo_coeffs = np.zeros((n_mo, n_mo))
        self.ecp_total_ncore = ecp_total_ncore


# --------------------------------------------------------------------- #
# Occupation-attribute path                                              #
# --------------------------------------------------------------------- #


def test_resolves_homo_lumo_from_mo_occupations() -> None:
    # Closed-shell H2O / sto-3g: 10 electrons -> 5 occupied MOs out
    # of 7 total. HOMO at index 4, LUMO at index 5.
    occ = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0])
    res = _DummyResultWithOcc(occ)
    pairs = requested_mo_indices(["homo", "lumo"], res)
    assert pairs == [(4, "homo"), (5, "lumo")]


def test_resolves_homo_minus_n_and_lumo_plus_n() -> None:
    occ = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0])
    res = _DummyResultWithOcc(occ)
    pairs = requested_mo_indices(["homo-1", "lumo+1"], res)
    assert pairs == [(3, "homo-1"), (6, "lumo+1")]


def test_resolves_int_label_as_zero_based_index() -> None:
    occ = np.array([2.0, 2.0, 0.0, 0.0])
    res = _DummyResultWithOcc(occ)
    pairs = requested_mo_indices([0, 2], res)
    assert pairs == [(0, "mo_0"), (2, "mo_2")]


def test_raises_on_unknown_label() -> None:
    res = _DummyResultWithOcc(np.array([2.0, 0.0]))
    with pytest.raises(ValueError, match="unknown MO label"):
        requested_mo_indices(["bogo"], res)


def test_raises_on_out_of_range_int() -> None:
    res = _DummyResultWithOcc(np.array([2.0, 0.0]))
    with pytest.raises(ValueError, match="out of range"):
        requested_mo_indices([42], res)


def test_lumo_without_virtual_orbital_raises() -> None:
    # All MOs doubly occupied: no LUMO.
    occ = np.array([2.0, 2.0])
    res = _DummyResultWithOcc(occ)
    with pytest.raises(ValueError, match="no virtual MO"):
        requested_mo_indices(["lumo"], res)


# --------------------------------------------------------------------- #
# n_electrons fallback path (no mo_occupations / mo_occ on result)       #
# --------------------------------------------------------------------- #


def test_fallback_derives_homo_lumo_from_molecule_n_electrons() -> None:
    # 10 electrons / 7 MOs (H2O / sto-3g shape).
    mol = _DummyMolecule(n_electrons=10)
    res = _DummyResultNoOcc(n_mo=7)
    pairs = requested_mo_indices(["homo", "lumo"], res, mol)
    assert pairs == [(4, "homo"), (5, "lumo")]


def test_fallback_subtracts_result_ecp_core_provenance() -> None:
    # Ten physical electrons with a two-electron ECP leave four doubly
    # occupied variational orbitals, so the frontier is 3/4 rather than 4/5.
    mol = _DummyMolecule(n_electrons=10)
    res = _DummyResultNoOcc(n_mo=7, ecp_total_ncore=2)
    pairs = requested_mo_indices(["homo", "lumo"], res, mol)
    assert pairs == [(3, "homo"), (4, "lumo")]


def test_fallback_two_arg_form_raises_clear_error() -> None:
    res = _DummyResultNoOcc(n_mo=7)
    with pytest.raises(ValueError, match="no molecule was supplied"):
        requested_mo_indices(["homo"], res)


def test_fallback_indices_match_closed_shell_expectation() -> None:
    # Sanity sweep: H2 (n_elec=2, HOMO=0, LUMO=1) and CH4
    # (n_elec=10, HOMO=4, LUMO=5).
    for n_elec, n_mo, homo, lumo in (
        (2, 5, 0, 1),
        (10, 9, 4, 5),
    ):
        mol = _DummyMolecule(n_electrons=n_elec)
        res = _DummyResultNoOcc(n_mo=n_mo)
        pairs = requested_mo_indices(["homo", "lumo"], res, mol)
        assert pairs == [(homo, "homo"), (lumo, "lumo")]


# --------------------------------------------------------------------- #
# Integration against a live H2O / sto-3g RHF result                     #
# --------------------------------------------------------------------- #


@pytest.mark.slow
def test_live_h2o_sto3g_rhf_homo_lumo_indices() -> None:
    """Run the same H2O / sto-3g RHF the rest of the suite uses, then
    check the fallback path resolves HOMO / LUMO to the expected
    closed-shell indices.

    Pins the convention against a real result so a future change to
    the result-type API (e.g. exposing ``mo_occupations``) does not
    silently drift the returned indices.
    """
    import vibeqc as vq

    mol = vq.Molecule.from_xyz(
        str(Path(__file__).parent.parent / "examples" / "h2o.xyz")
    )
    basis = vq.BasisSet(mol, "sto-3g")
    result = vq.run_rhf(mol, basis)
    n_elec = mol.n_electrons()
    homo_expected = n_elec // 2 - 1
    lumo_expected = n_elec // 2
    pairs = requested_mo_indices(["homo", "lumo"], result, mol)
    assert pairs == [(homo_expected, "homo"), (lumo_expected, "lumo")]
