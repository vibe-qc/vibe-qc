"""MRCI (multireference CI) tests — correctness and sanity checks."""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc import Atom, BasisSet, Molecule
from vibeqc.solvers import (
    build_hamiltonian_mo,
    casci,
    generate_cisd_determinants,
    get_hf_orbital_provider,
)
from vibeqc.solvers._mrci import (
    _generate_doubles_unrestricted,
    _generate_singles_unrestricted,
    mrci,
)

H2 = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])
H2O = Molecule(
    [
        Atom(8, [0.0, 0.0, 0.0]),
        Atom(1, [0.0, 1.43, -0.93]),
        Atom(1, [0.0, -1.43, -0.93]),
    ]
)


def _ham(mol, basis_name):
    b = BasisSet(mol, basis_name)
    c = get_hf_orbital_provider(mol, b, method="rhf")
    return build_hamiltonian_mo(mol, b, c)


class TestMRCIBasic:
    """Basic MRCI correctness checks."""

    def test_h2o_mrci_lower_than_casci(self):
        """MRCI must be strictly lower than CASCI (dynamic correlation)."""
        H = _ham(H2O, "sto-3g")
        e_casci = casci(
            H.h1e,
            H.h2e,
            n_active_elec=2,
            n_active_orb=2,
            n_core=4,
            nuclear_repulsion=H.nuclear_repulsion,
        ).e_total
        res = mrci(H.h1e, H.h2e, 2, 2, 4, H.nuclear_repulsion)
        assert res.e_total < e_casci - 1e-6  # strictly lower

    def test_mrci_variational(self):
        """MRCI must be at or above FCI (variational bound)."""
        H = _ham(H2O, "sto-3g")
        e_fci = casci(
            H.h1e,
            H.h2e,
            H.nelec,
            H.norb,
            0,
            nuclear_repulsion=H.nuclear_repulsion,
        ).e_total
        res = mrci(H.h1e, H.h2e, 2, 2, 4, H.nuclear_repulsion)
        assert res.e_total >= e_fci - 1e-9

    def test_full_space_mrci_is_fci(self):
        """When CAS = full space, MRCI = FCI (no virtuals, no excitations)."""
        H = _ham(H2, "sto-3g")
        e_fci = casci(
            H.h1e,
            H.h2e,
            H.nelec,
            H.norb,
            0,
            nuclear_repulsion=H.nuclear_repulsion,
        ).e_total
        res = mrci(H.h1e, H.h2e, H.nelec, H.norb, 0, H.nuclear_repulsion)
        assert abs(res.e_total - e_fci) < 1e-12
        assert res.n_det == res.n_ref  # no new determinants

    def test_single_reference_mrci_h2_equals_fci(self):
        """H₂/STO-3G MRCISD from a *single-reference* CAS(2,1) reaches FCI.

        With two electrons CISD is exact, and a CAS(2,1) reference is a single
        closed-shell determinant — so unlike :meth:`test_full_space_mrci_is_fci`
        (where the whole FCI space is already the reference set) the leading
        αβ HOMO²→LUMO² double must be *generated* by the doubles enumerator for
        MRCISD to reach FCI.  Regression guard: the earlier ``a == b`` omission
        dropped exactly that determinant, leaving 3 dets and MRCISD above FCI.
        """
        H = _ham(H2, "sto-3g")
        e_fci = casci(
            H.h1e, H.h2e, H.nelec, H.norb, 0,
            nuclear_repulsion=H.nuclear_repulsion,
        ).e_total
        res = mrci(H.h1e, H.h2e, 2, 1, 0, H.nuclear_repulsion)
        assert res.n_ref == 1  # genuinely single-reference
        assert res.n_det == 4  # full 2e/2o FCI space (was 3 with the bug)
        assert abs(res.e_total - e_fci) < 1e-12  # exact for two electrons

    def test_q_correction(self):
        """Davidson +Q correction is applied and reasonable."""
        H = _ham(H2O, "sto-3g")
        res = mrci(H.h1e, H.h2e, 2, 2, 4, H.nuclear_repulsion, do_q_correction=True)
        assert res.e_total_q is not None
        # +Q should be below MRCI (correcting for size-extensivity)
        assert res.e_total_q <= res.e_total + 1e-9

    def test_no_q_correction(self):
        """do_q_correction=False skips the +Q term."""
        H = _ham(H2O, "sto-3g")
        res = mrci(H.h1e, H.h2e, 2, 2, 4, H.nuclear_repulsion, do_q_correction=False)
        assert res.e_total_q is None

    def test_ref_weight_between_0_and_1(self):
        """Reference weight must be between 0 and 1."""
        H = _ham(H2O, "sto-3g")
        res = mrci(H.h1e, H.h2e, 2, 2, 4, H.nuclear_repulsion)
        assert 0.0 < res.ref_weight <= 1.0

    def test_max_det_limit(self):
        """max_det limit raises ValueError when exceeded."""
        H = _ham(H2O, "sto-3g")
        with pytest.raises(ValueError, match="MRCI space has"):
            mrci(H.h1e, H.h2e, 2, 2, 4, H.nuclear_repulsion, max_det=10)

    def test_mrci_cas44(self):
        """MRCI with CAS(4,4) — larger reference space."""
        H = _ham(H2O, "sto-3g")
        e_casci = casci(
            H.h1e,
            H.h2e,
            n_active_elec=4,
            n_active_orb=4,
            n_core=3,
            nuclear_repulsion=H.nuclear_repulsion,
        ).e_total
        res = mrci(H.h1e, H.h2e, 4, 4, 3, H.nuclear_repulsion)
        assert res.e_total < e_casci  # MRCI adds correlation
        assert res.n_det > res.n_ref  # more dets than ref


def _single_reference_space(norb, nalpha, nbeta):
    """Reference + all singles + all doubles from the aufbau reference, built
    with the unrestricted MRCI generators (the single-reference special case of
    the MRCI space)."""
    a_ref = tuple(range(nalpha))
    b_ref = tuple(range(nbeta))
    space = {(a_ref, b_ref)}
    space.update(_generate_singles_unrestricted(a_ref, b_ref, norb))
    space.update(_generate_doubles_unrestricted(a_ref, b_ref, norb))
    return space


class TestMRCIDeterminantSpace:
    """The unrestricted single/double generators must enumerate the full CISD
    space, matching the validated
    :func:`~vibeqc.solvers.generate_cisd_determinants` det-for-det.

    Regression guards for two coupled enumeration bugs:

    * the opposite-spin (αβ) double placing the promoted α and β electrons in
      the **same** spatial virtual (the HOMO²→LUMO²-type configuration, usually
      the leading correlating determinant) was dropped by a ``qa == pa`` guard;
    * virtuals were taken from the **combined** ``range(norb) − α − β`` set,
      which for an open-shell reference wrongly forbade an electron from
      entering an orbital singly occupied by the *opposite* spin.

    Both silently undercounted the CI space and raised the MRCI energy.
    """

    @pytest.mark.parametrize(
        "norb, nalpha, nbeta",
        [(2, 1, 1), (4, 2, 2), (6, 4, 4), (5, 3, 2), (7, 4, 3)],
    )
    def test_generators_match_validated_cisd_space(self, norb, nalpha, nbeta):
        """Single-reference space equals ``generate_cisd_determinants`` for
        closed (nα == nβ) *and* open (nα ≠ nβ) shells, det-for-det."""
        got = _single_reference_space(norb, nalpha, nbeta)
        want = set(generate_cisd_determinants(norb, nalpha, nbeta))
        assert got == want

    def test_opposite_spin_double_into_same_virtual_present(self):
        """αβ double promoting both electrons into the same spatial virtual is
        generated (2 orbitals, 1α+1β ⇒ the determinant ``((1,), (1,))``)."""
        space = _single_reference_space(2, 1, 1)
        assert ((1,), (1,)) in space
        assert len(space) == 4  # full 2e/2o FCI; the old guard gave only 3

    def test_open_shell_excite_into_opposite_spin_singly_occupied(self):
        """Open-shell guard: a β electron may single-excite into an orbital
        singly occupied by an α electron (combined-virtual set forbade it).

        Doublet reference (3α, 2β) in 5 orbitals — orbital 2 is α-occupied and
        β-empty, so the β single ``1β→2β`` (doubly occupying orbital 2) exists.
        """
        singles = _generate_singles_unrestricted((0, 1, 2), (0, 1), 5)
        assert ((0, 1, 2), (0, 2)) in singles
