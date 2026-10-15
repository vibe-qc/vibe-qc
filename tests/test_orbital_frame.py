"""The orbital frame inside degenerate clusters is a contract, not eigensolver noise.

GitLab #322. N2/cc-pVDZ has exactly degenerate 1pi_u and 1pi_g pairs. The
eigenvectors LAPACK returns inside such a pair are whatever rounding noise
selects, and they differ between hosts and even between two geometries on
one host. Full CI in the active space does not care; seniority-zero (DOCI)
CI does: its energy sweeps about 16 mHa with the relative pi_u/pi_g frame
angle, and the two DOCI pins of ``tests/test_solvers_selected_ci.py`` were
four samples of that noise (the pin, the literal, a second fleet host at
v0.17.4 and the laptop that filed the audit). ``get_hf_orbital_provider``
now fixes every
degenerate cluster by successive projected second moments about the nuclear
centre of charge, so the frame -- and every frame-dependent number computed
on it -- is a function of the deck.

References: Helgaker, Jorgensen and Olsen, *Molecular Electronic-Structure
Theory* (Wiley, 2000), Sec. 11.8 (CI in a complete active space is invariant
under active-active rotations; canonical HF orbitals diagonalise the Fock
matrix); Bytautas, Henderson, Jimenez-Hoyos, Ellis and Scuseria, J. Chem.
Phys. 135, 044119 (2011), doi:10.1063/1.3613706 (seniority-zero CI depends
on the orbital set).
"""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc import Atom, BasisSet, Molecule
from vibeqc._vibeqc_core import compute_overlap
from vibeqc.solvers import (
    build_hamiltonian_matrix_unrestricted,
    build_hamiltonian_mo,
    casci,
    generate_closed_shell_determinants,
    get_hf_orbital_provider,
)

ANGSTROM_TO_BOHR = 1.8897259886

# Seniority-zero energies of the two N2/cc-pVDZ CAS(6e,6o) decks in the
# canonical frame (both pi frames axis-aligned, so pi_u and pi_g are
# aligned with each other: the minimum of the 16 mHa frame surface).
# Measured at the fix commit on Apple Accelerate; the same numbers must come
# out on every host, which is the point.
N2_2P0787_CANONICAL_DOCI = -108.9994509922
N2_2P074_CANONICAL_DOCI = -108.9994363827
# Rotation-invariant guards for the same two decks (CASCI(6e,6o), full
# active space), matched by PySCF 2.14 to 1e-12 Ha in the #322 audit.
N2_2P0787_CASCI = -109.0219049828
N2_2P074_CASCI = -109.0217750033


def _n2(half_bond_bohr: float, axis: int = 2) -> Molecule:
    lo = [0.0, 0.0, 0.0]
    hi = [0.0, 0.0, 0.0]
    lo[axis] = -half_bond_bohr
    hi[axis] = half_bond_bohr
    return Molecule([Atom(7, lo), Atom(7, hi)], charge=0, multiplicity=1)


def _cas66(mol: Molecule, C: np.ndarray):
    basis = BasisSet(mol, "cc-pvdz")
    return build_hamiltonian_mo(mol, basis, C).active_space(6, 6)


def _doci(H) -> float:
    cs = generate_closed_shell_determinants(H.norb, H.nelec // 2)
    M = build_hamiltonian_matrix_unrestricted([(d, d) for d in cs], H.h1e, H.h2e)
    return float(np.linalg.eigvalsh(M)[0]) + H.nuclear_repulsion


def _casci(H) -> float:
    r = casci(H.h1e, H.h2e, H.nelec, H.norb, nuclear_repulsion=H.nuclear_repulsion)
    return float(r.e_total)


def _rot2(n: int, i: int, j: int, theta: float) -> np.ndarray:
    U = np.eye(n)
    c, s = np.cos(theta), np.sin(theta)
    U[i, i] = c
    U[j, j] = c
    U[i, j] = -s
    U[j, i] = s
    return U


def _rotated(H, U):
    h1 = U.T @ H.h1e @ U
    h2 = np.einsum("pqrs,pi,qj,rk,sl->ijkl", H.h2e, U, U, U, U, optimize=True)
    return type(H)(
        h1e=h1, h2e=h2, nuclear_repulsion=H.nuclear_repulsion,
        norb=H.norb, nelec=H.nelec, ms2=H.ms2,
    )


DECKS = [
    pytest.param(0.55 * ANGSTROM_TO_BOHR, N2_2P0787_CANONICAL_DOCI, N2_2P0787_CASCI, id="R=2.0787"),
    pytest.param(2.074 / 2, N2_2P074_CANONICAL_DOCI, N2_2P074_CASCI, id="R=2.074"),
]


# ---------------------------------------------------------------------
# The control: what is frame-dependent and what is not
# ---------------------------------------------------------------------


@pytest.mark.parametrize("half_bond, doci_literal, casci_literal", DECKS)
def test_seniority_zero_depends_on_the_pi_frame_but_casci_does_not(
    half_bond, doci_literal, casci_literal
):
    """Rotating one degenerate pair moves DOCI by mHa and CASCI by nothing.

    This is the non-vacuity control for the pins below: on a fixed active
    Hamiltonian, a 45 degree rotation inside the 1pi_g pair (active orbitals
    3 and 4) changes the seniority-zero energy by more than 1 mHa while the
    full CASCI(6e,6o) energy is unchanged to 1e-9 Ha. A pin on DOCI is
    therefore a pin on the frame, and only a defined frame makes it a pin.
    """
    mol = _n2(half_bond)
    H = _cas66(mol, get_hf_orbital_provider(mol, BasisSet(mol, "cc-pvdz")))
    Hr = _rotated(H, _rot2(6, 3, 4, np.radians(45.0)))
    assert abs(_doci(Hr) - _doci(H)) > 1e-3
    assert _casci(Hr) == pytest.approx(_casci(H), abs=1e-9)
    assert _casci(H) == pytest.approx(casci_literal, abs=1e-9)


# ---------------------------------------------------------------------
# The provider contract
# ---------------------------------------------------------------------


@pytest.mark.parametrize("half_bond, doci_literal, casci_literal", DECKS)
def test_provider_frame_reproduces_the_canonical_doci_literal(
    half_bond, doci_literal, casci_literal
):
    """The seniority-zero energy of the provider's orbitals is the deck's.

    Before #322 this host returned -108.9850687702 Ha for the first deck (the
    old pin, 14.4 mHa above) and -108.9896767157 Ha for the second (neither
    its old literal nor the second host's value): three hosts, three frames.
    """
    mol = _n2(half_bond)
    H = _cas66(mol, get_hf_orbital_provider(mol, BasisSet(mol, "cc-pvdz")))
    assert _doci(H) == pytest.approx(doci_literal, abs=1e-9)
    assert _casci(H) == pytest.approx(casci_literal, abs=1e-9)


def test_canonical_frame_is_axis_aligned_and_survives_a_molecule_rotation():
    """N2 along x gives the same DOCI as N2 along z, and pure pi_x / pi_y.

    A physical rotation of the deck must not change any energy. In the raw
    eigensolver frame the two orientations land on unrelated pi mixings; in
    the canonical frame each degenerate pair is aligned with the Cartesian
    axes, so the two members of a pair have disjoint AO supports.
    """
    half = 0.55 * ANGSTROM_TO_BOHR
    energies = []
    for axis in (2, 0):
        mol = _n2(half, axis=axis)
        basis = BasisSet(mol, "cc-pvdz")
        C = get_hf_orbital_provider(mol, basis)
        energies.append(_doci(_cas66(mol, C)))
        for a, b in ((5, 6), (7, 8)):  # 1pi_u and 1pi_g in HF order
            sup_a = set(np.flatnonzero(np.abs(C[:, a]) > 1e-8).tolist())
            sup_b = set(np.flatnonzero(np.abs(C[:, b]) > 1e-8).tolist())
            assert sup_a and sup_b and not (sup_a & sup_b), (axis, a, b)
    assert energies[0] == pytest.approx(energies[1], abs=1e-9)
    assert energies[0] == pytest.approx(N2_2P0787_CANONICAL_DOCI, abs=1e-9)


def test_opt_out_returns_the_raw_scf_frame_and_canonicalisation_is_idempotent():
    """``canonical_frame=False`` hands back the SCF's own eigenvectors."""
    from vibeqc import run_rhf
    from vibeqc._vibeqc_core import RHFOptions
    from vibeqc.solvers import canonicalize_degenerate_orbitals

    mol = _n2(0.55 * ANGSTROM_TO_BOHR)
    basis = BasisSet(mol, "cc-pvdz")
    raw, result = get_hf_orbital_provider(
        mol, basis, canonical_frame=False, _with_result=True
    )
    opts = RHFOptions()
    opts.max_iter = 200
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-10
    assert np.array_equal(raw, np.asarray(run_rhf(mol, basis, opts).mo_coeffs))

    canonical = get_hf_orbital_provider(mol, basis)
    eps = np.asarray(result.mo_energies)
    again, report = canonicalize_degenerate_orbitals(mol, basis, canonical, eps)
    assert np.max(np.abs(again - canonical)) < 1e-12
    assert report.clusters  # N2/cc-pVDZ has degenerate pairs to resolve


# ---------------------------------------------------------------------
# The canonicaliser itself
# ---------------------------------------------------------------------


def test_canonicalize_degenerate_orbitals_returns_one_frame_from_any_start():
    """Any rotation inside a degenerate pair is undone: one frame, same Fock.

    Simulates other hosts' eigensolvers by rotating the 1pi_u and 1pi_g pairs
    of the raw SCF orbitals by random angles before canonicalising. The
    result must not depend on the start (DOCI spread below 1e-9 Ha), must
    keep the Fock matrix diagonal and the orbital energies and orthonormality
    intact, and must leave non-degenerate columns untouched.
    """
    from vibeqc.solvers import canonicalize_degenerate_orbitals

    mol = _n2(0.55 * ANGSTROM_TO_BOHR)
    basis = BasisSet(mol, "cc-pvdz")
    raw, result = get_hf_orbital_provider(
        mol, basis, canonical_frame=False, _with_result=True
    )
    eps = np.asarray(result.mo_energies)
    S = np.asarray(compute_overlap(basis))
    F_ao = S @ raw @ np.diag(eps) @ raw.T @ S

    rng = np.random.default_rng(322)
    energies = []
    for _ in range(5):
        start = np.array(raw)
        for a, b in ((5, 6), (7, 8)):
            th = rng.uniform(0.0, np.pi)
            start[:, [a, b]] = start[:, [a, b]] @ np.array(
                [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]
            )
        C, report = canonicalize_degenerate_orbitals(mol, basis, start, eps)
        F_mo = C.T @ F_ao @ C
        assert np.max(np.abs(F_mo - np.diag(np.diag(F_mo)))) < 1e-10
        assert np.max(np.abs(np.diag(F_mo) - eps)) < 1e-10
        assert np.max(np.abs(C.T @ S @ C - np.eye(C.shape[1]))) < 1e-12
        clustered = {k for start_, stop in report.clusters for k in range(start_, stop)}
        for k in range(C.shape[1]):
            if k not in clustered:
                assert np.array_equal(C[:, k], start[:, k]), k
        energies.append(_doci(_cas66(mol, C)))
    assert max(energies) - min(energies) < 1e-9
    assert energies[0] == pytest.approx(N2_2P0787_CANONICAL_DOCI, abs=1e-9)
    assert "second moments" in report.description


def test_uno_provider_is_a_fixed_point_of_the_canonicaliser():
    """Natural orbitals with degenerate occupations get the same treatment.

    O2 triplet UNOs carry two exactly degenerate pi occupations; the
    provider's output must already be canonical (idempotent under a second
    pass keyed on the occupations) and stay orthonormal.
    """
    from vibeqc._vibeqc_core import UHFOptions, run_uhf
    from vibeqc.solvers import canonicalize_degenerate_orbitals

    mol = Molecule([Atom(8, [0.0, 0.0, 0.0]), Atom(8, [0.0, 0.0, 2.28])], multiplicity=3)
    basis = BasisSet(mol, "sto-3g")
    C = get_hf_orbital_provider(mol, basis, method="uno")
    S = np.asarray(compute_overlap(basis))
    opts = UHFOptions()
    opts.max_iter = 200
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-10
    r = run_uhf(mol, basis, opts)
    P = np.asarray(r.density_alpha) + np.asarray(r.density_beta)
    occ = np.diag(C.T @ S @ P @ S @ C)
    again, report = canonicalize_degenerate_orbitals(mol, basis, C, occ)
    assert report.clusters
    assert np.max(np.abs(again - C)) < 1e-12
    assert np.max(np.abs(C.T @ S @ C - np.eye(C.shape[1]))) < 1e-12


# ---------------------------------------------------------------------
# The window guard
# ---------------------------------------------------------------------


def test_active_space_refuses_a_window_that_cuts_a_degenerate_cluster():
    """CAS(2e,2o) on N2 would put one pi_u and one pi_g in the window.

    Both boundaries then fall inside a degenerate pair and the CI energy
    depends on the frame, not on the deck: refused with the cluster named.
    CAS(4e,4o) and CAS(6e,6o) take whole pairs and still work.
    """
    mol = _n2(0.55 * ANGSTROM_TO_BOHR)
    basis = BasisSet(mol, "cc-pvdz")
    H = build_hamiltonian_mo(mol, basis, get_hf_orbital_provider(mol, basis))
    with pytest.raises(ValueError, match=r"degenerate"):
        H.active_space(2, 2)
    assert H.active_space(4, 4).norb == 4
    assert H.active_space(6, 6).norb == 6
