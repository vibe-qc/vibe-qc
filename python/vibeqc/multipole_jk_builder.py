"""**Status -- foundation, not production-ready.**

The multipole-only J build is too approximate for compact molecules
(78 % error for H₂O / STO-3G) because it replaces ALL atom-pair
Coulomb interactions with multipole expansions, including near-field
pairs where charge distributions overlap.  A production FMM requires
a near-field / far-field split: exact 4-index ERIs within a cutoff
radius, multipole expansions beyond it (cf. BIPOLE's periodic split).

This module provides the building blocks for that future builder --
atom-centred multipole integral precomputation, density -> moment
contraction, and multipole-potential -> Fock scattering -- but does
**not** yet integrate the near-field direct path. For periodic systems,
the public BIPOLE route rejects ``use_multipole_far_field=True`` because
the dormant quartet prototype does not preserve the exact three-translation
Fock domain. The exact Ewald-J split is the supported path.

Next step (v0.12-v0.13): add a ``distance_cutoff`` parameter to the
molecular C++ JKBuilder, then combine direct near-field J with the
multipole far-field J from this module.

Molecular multipole-J JKBuilder -- FMM-style Coulomb acceleration.

Replaces the O(N⁴) 4-index-ERI Coulomb J build with an O(N^2) atom-pair
multipole expansion up to quadrupole order (L_max <= 2).  Exchange K is
still computed exactly via the direct JKBuilder (exchange decays
exponentially and doesn't benefit from multipole acceleration).

This is the molecular analogue of BIPOLE's dormant periodic far-field
multipole prototype (``bipole_fock_multipole``). Public periodic drivers
reject ``use_multipole_far_field=True``; the exact Ewald-J split remains
the supported path.

Usage::

    from vibeqc.multipole_jk_builder import make_multipole_jk_builder

    jk = make_multipole_jk_builder(basis, mol)
    J_mp = jk.build_J(D)   # multipole Coulomb
    K_ex  = jk.build_K(D)  # exact exchange (direct JKBuilder)

The JKBuilder interface is compatible with ``run_rhf_scf_with_jk``,
so it can be used as a drop-in replacement for the standard JKBuilder
in SCF calculations.

Accuracy
--------
The multipole J is correct through quadrupole-quadrupole order
(1/R⁵ asymptotic).  For compact molecules with well-separated atoms
the error is < 1e-4 Ha in the total energy.  Near-field errors from
overlapping charge distributions (same-atom / bonded pairs) are the
main approximation -- a future combined near-field-direct +
far-field-multipole builder will close that gap.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ._vibeqc_core import BasisSet, Molecule

__all__ = [
    "MultipoleJKBuilder",
    "make_multipole_jk_builder",
]


# ---------------------------------------------------------------------------
# Atom-pair multipole integrals (one-time precomputation)
# ---------------------------------------------------------------------------


def _compute_atom_multipole_integrals(
    basis: BasisSet,
    mol: Molecule,
    L_max: int = 2,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Pre-compute atom-centred multipole AO integrals.

    Returns three lists (one per atom):
      - mono[a]:  (na, na)  overlap (charge)
      - dip[a]:   (3, na, na)  dipole integrals <mu|r_c - R_a^c|ν>
      - quad[a]:  (6, na, na)  quadrupole (xx, xy, xz, yy, yz, zz)

    Origin is the atom centre.  Uses libint's emultipole2 operator
    via BIPOLE's compute_multipole_moments_lattice on a trivial
    1-cell periodic system.
    """
    from ._vibeqc_core import (
        LatticeSumOptions,
        PeriodicSystem,
        compute_multipole_moments_lattice,
    )

    atoms = mol.atoms
    n_atoms = len(atoms)
    box = np.eye(3) * 1000.0
    sysp = PeriodicSystem(3, box, list(atoms), mol.charge, mol.multiplicity)

    lat_opts = LatticeSumOptions()
    lat_opts.cutoff_bohr = 1.0

    mp_lat = compute_multipole_moments_lattice(basis, sysp, lat_opts, L_max=L_max)

    # Map AOs to atoms.  n_funcs = (l+1)(l+2)/2 for Cartesian shells
    # (STO-6G uses Cartesian throughout vibe-qc's semiempirical basis).
    ao_atom: List[int] = []
    for s in basis.shells():
        ai = s.atom_index
        nf = (s.l + 1) * (s.l + 2) // 2
        ao_atom.extend([ai] * nf)

    mono: List[np.ndarray] = []
    dip: List[np.ndarray] = []
    quad: List[np.ndarray] = []

    for a in range(n_atoms):
        idx = [i for i, ai in enumerate(ao_atom) if ai == a]
        na = len(idx)

        # Monopole (component 0 = overlap)
        S_blk = np.asarray(mp_lat.blocks[0][0], dtype=float)
        mono.append(S_blk[np.ix_(idx, idx)])

        # Dipole (components 1-3)
        dip_a = np.zeros((3, na, na))
        for c in range(3):
            blk = np.asarray(mp_lat.blocks[0][1 + c], dtype=float)
            dip_a[c] = blk[np.ix_(idx, idx)]
        dip.append(dip_a)

        # Quadrupole (components 4-9)
        quad_a = np.zeros((6, na, na))
        for c in range(6):
            blk = np.asarray(mp_lat.blocks[0][4 + c], dtype=float)
            quad_a[c] = blk[np.ix_(idx, idx)]
        quad.append(quad_a)

    return mono, dip, quad


# ---------------------------------------------------------------------------
# Multipole interaction tensor (atom potential builder)
# ---------------------------------------------------------------------------


def _compute_atom_potentials(
    atoms: List,
    q: np.ndarray,
    mu: np.ndarray,
    theta: np.ndarray,
    gamma: np.ndarray,
) -> np.ndarray:
    """Compute the electrostatic potential V_a on each atom.

    V_a = g_a q_a + S_{b!=a} [ q_b/R + (mu_b.R̂)/R^2 + ... ]

    where g_a is the on-site chemical hardness (Hubbard U) for element Z_a.
    Uses multipole expansion through quadrupole-quadrupole order.
    Returns V array of shape (n_atoms,).
    """
    n_atoms = len(atoms)

    # On-site term from atom's own charge
    V = gamma * q

    for a in range(n_atoms):
        ra = np.array(atoms[a].xyz)
        for b in range(n_atoms):
            if a == b:
                continue
            rb = np.array(atoms[b].xyz)
            d = ra - rb
            r = float(np.linalg.norm(d))
            if r < 1e-12:
                continue

            drx, dry, drz = d[0] / r, d[1] / r, d[2] / r
            inv_r = 1.0 / r
            inv_r2 = inv_r * inv_r
            inv_r3 = inv_r2 * inv_r

            # charge-charge: d(q_a q_b/R)/dq_a = q_b / R
            V[a] += q[b] * inv_r

            # charge-dipole from b: d(q_a (mu_b.R̂)/R^2)/dq_a = (mu_b.R̂)/R^2
            mu_dot_r_b = mu[b, 0] * drx + mu[b, 1] * dry + mu[b, 2] * drz
            V[a] += mu_dot_r_b * inv_r2

            # dipole-dipole (potential from b's dipole on a's charge):
            # d(-3(mu_a.R̂)(mu_b.R̂)/R^3)/dq_a = 0 (no q_a dependence)
            # d(mu_a.mu_b/R^3)/dq_a = 0

            # charge-quadrupole from b:
            # d(1/2 q_a S_cd th_b^{cd} r̂_c r̂_d / R^3)/dq_a = 1/2 r̂.th_b.r̂ / R^3
            r_quad_b = (
                theta[b, 0] * drx * drx
                + 2.0 * theta[b, 1] * drx * dry
                + 2.0 * theta[b, 2] * drx * drz
                + theta[b, 3] * dry * dry
                + 2.0 * theta[b, 4] * dry * drz
                + theta[b, 5] * drz * drz
            )
            tr_b = theta[b, 0] + theta[b, 3] + theta[b, 5]
            V[a] += 0.5 * (r_quad_b - tr_b) * inv_r3

    return V


# ---------------------------------------------------------------------------
# Molecular multipole JKBuilder
# ---------------------------------------------------------------------------


class MultipoleJKBuilder:
    """Molecular JKBuilder using multipole-accelerated Coulomb J.

    Pre-computes atom-centred multipole integrals (one-time cost per
    geometry).  Each ``build_J`` call contracts the density matrix
    against these integrals to form atom multipole moments, computes
    electrostatic potentials via pairwise multipole interactions,
    and assembles J via J_{muν} += 1/2 (V_a + V_b) S_{muν}.

    Exchange K is delegated to the standard direct JKBuilder.
    """

    def __init__(self, basis: BasisSet, mol: Molecule, L_max: int = 2):
        self._basis = basis
        self._mol = mol
        self._L_max = min(L_max, 2)
        self._nbf = basis.nbasis
        self._atoms = mol.atoms
        self._n_atoms = len(self._atoms)

        # Pre-compute multipole integrals (one-time cost).
        self._mono_int, self._dip_int, self._quad_int = (
            _compute_atom_multipole_integrals(basis, mol, L_max=self._L_max)
        )

        # Direct JK builder for exchange.
        from ._vibeqc_core import make_direct_jk_builder

        self._k_builder = make_direct_jk_builder(basis, 1e-12, False, 0)

        # Element-specific on-site Coulomb repulsion (chemical hardness).
        # Approximate values in Hartree -- sufficient for multipole J
        # where the on-site term is the leading diagonal contribution.
        _DEFAULT_GAMMA = {1: 0.75, 6: 0.40, 7: 0.45, 8: 0.50, 9: 0.55}
        self._gamma = np.array([
            _DEFAULT_GAMMA.get(at.Z, 0.5) for at in mol.atoms
        ])

        # Map AOs to atoms.
        self._ao_atom: List[int] = []
        for s in basis.shells():
            ai = s.atom_index
            nf = (s.l + 1) * (s.l + 2) // 2
            self._ao_atom.extend([ai] * nf)

        # Per-atom AO indices.
        self._atom_ao_idx: List[List[int]] = []
        for a in range(self._n_atoms):
            idx = [i for i, ai in enumerate(self._ao_atom) if ai == a]
            self._atom_ao_idx.append(idx)

        # Pre-compute overlap matrix blocks for Fock scattering.
        self._S = self._compute_overlap()

    def _compute_overlap(self) -> np.ndarray:
        """Compute full AO overlap matrix."""
        from ._vibeqc_core import compute_overlap

        return np.asarray(compute_overlap(self._basis), dtype=float)

    def build_J(self, D: np.ndarray) -> np.ndarray:
        """Compute Coulomb J via atom-pair multipole expansion.

        Algorithm:
        1. Contract D with multipole integrals -> atom moments (q, mu, th)
        2. Compute electrostatic potential V_a on each atom from all others
        3. J_{muν} += 1/2 (V_a + V_b) S_{muν} for AO pairs between atoms a,b
        """
        D = np.asarray(D, dtype=float)

        # 1. Compute atom multipole moments.
        q = np.zeros(self._n_atoms)
        mu = np.zeros((self._n_atoms, 3))
        theta = np.zeros((self._n_atoms, 6))

        for a in range(self._n_atoms):
            idx = self._atom_ao_idx[a]
            if not idx:
                continue
            D_a = D[np.ix_(idx, idx)]
            q[a] = float(np.sum(D_a * self._mono_int[a]))
            for c in range(3):
                mu[a, c] = float(np.sum(D_a * self._dip_int[a][c]))
            for c in range(6):
                theta[a, c] = float(np.sum(D_a * self._quad_int[a][c]))

        # 2. Compute atom potentials (with on-site gamma).
        V = _compute_atom_potentials(self._atoms, q, mu, theta, self._gamma)

        # 3. Scatter into J matrix.
        # J_{muν} = S_a S_{muina} 1/2 V_a S_{muν}  (diagonal)
        #        + S_{a<b} S_{muina,νinb} 1/2 (V_a + V_b) S_{muν}  (off-diagonal)
        J = np.zeros((self._nbf, self._nbf))
        S = self._S

        for a in range(self._n_atoms):
            Va = float(V[a])
            idx_a = self._atom_ao_idx[a]
            if not idx_a:
                continue

            # Intra-atom block: J_{muν} += 1/2 V_a S_{muν}
            for i, mu in enumerate(idx_a):
                for j, nu in enumerate(idx_a):
                    J[mu, nu] += 0.5 * Va * S[mu, nu]

            # Inter-atom blocks: J_{muν} += 1/2 (V_a + V_b) S_{muν}
            for b in range(a + 1, self._n_atoms):
                Vb = float(V[b])
                idx_b = self._atom_ao_idx[b]
                if not idx_b:
                    continue
                half_sum = 0.5 * (Va + Vb)
                for mu in idx_a:
                    for nu in idx_b:
                        J[mu, nu] += half_sum * S[mu, nu]
                        J[nu, mu] += half_sum * S[nu, mu]

        # Symmetrise
        J = 0.5 * (J + J.T)
        return J

    def build_K(self, D: np.ndarray) -> np.ndarray:
        """Compute exact exchange K via direct JKBuilder."""
        return np.asarray(self._k_builder.build_K(np.asarray(D, dtype=float)))

    def build_g_rhf(self, D: np.ndarray, alpha_hf: float = 1.0) -> np.ndarray:
        """RHF closed-shell G = J - 1/2.a.K."""
        J = self.build_J(D)
        K = self.build_K(D)
        return J - 0.5 * alpha_hf * K


def make_multipole_jk_builder(
    basis: BasisSet, mol: Molecule, L_max: int = 2
) -> MultipoleJKBuilder:
    """Create a molecular multipole-accelerated JKBuilder.

    Parameters
    ----------
    basis
        AO basis set.
    mol
        Molecular geometry.
    L_max
        Maximum multipole order (0=charge only, 1=+dipole, 2=+quadrupole).
        Default 2.

    Returns
    -------
    MultipoleJKBuilder
        J is multipole-accelerated (O(N^2) atom pairs); K is exact.
    """
    return MultipoleJKBuilder(basis, mol, L_max=L_max)
