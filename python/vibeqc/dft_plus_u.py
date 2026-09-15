"""DFT+U (Dudarev rotationally-invariant) -- occupation-matrix machinery.

Increment 1 of the DFT+U implementation: the AO-projected occupation
matrix

    n^A_l_{mm'} = sum_i f_i <phi^A_lm | psi_i><psi_i | phi^A_lm'>
                = (S P S)_{(A,l),(A,l)}

and the Dudarev (1998) rotationally-invariant +U energy

    E_U = sum_A (U_eff/2) (tr(n^A_l) - tr(n^A_l n^A_l))

with ``U_eff = U - J``. This module is the configuration + math surface;
the SCF Fock-build integration is complete -- see ``docs/user_guide/dft_plus_u.md``).

Spin convention is the caller's responsibility: pass a per-spin P for
the per-spin formula, or the closed-shell total P for the closed-shell
formula. The functions here treat P as an opaque AO matrix.

References
----------
* Dudarev, Botton, Savrasov, Humphreys, Sutton, *Electron-energy-loss
  spectra and the structural stability of nickel oxide: An LSDA+U
  study*, Phys. Rev. B 57, 1505 (1998).

(The citation database entry + ``routes.methods.dft_plus_u`` route
land in Increment 2 when the user-callable surface fires the route;
see CLAUDE.md Sec. 8.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ._vibeqc_core import BasisSet, _HubbardSiteCxx

__all__ = [
    "HubbardSite",
    "ao_group_indices",
    "compute_occupation_matrices",
    "compute_dudarev_energy",
]


_HARTREE_TO_EV = 27.211386245988
_EV_TO_HARTREE = 1.0 / _HARTREE_TO_EV


@dataclass(frozen=True)
class HubbardSite:
    """A Hubbard-U-active centre x angular-shell channel.

    Parameters
    ----------
    atom_index
        Zero-based index into the molecule's atom list.
    l
        Angular momentum of the +U-active shell (0=s, 1=p, 2=d, 3=f).
        Typical use is l=2 for transition metals, l=3 for lanthanides
        and actinides.
    U_ev
        Hubbard U in eV (user surface). Stored as given; converted to
        Hartree internally via :attr:`U_eff_hartree`.
    J_ev
        Hund J in eV. The Dudarev energy uses only the combination
        ``U_eff = U - J``, so only the difference is physically
        meaningful. Defaults to 0.
    """

    atom_index: int
    l: int
    U_ev: float
    J_ev: float = 0.0

    def __post_init__(self) -> None:
        if self.atom_index < 0:
            raise ValueError(f"atom_index must be >= 0, got {self.atom_index}")
        if self.l < 0:
            raise ValueError(f"l must be >= 0, got {self.l}")

    @property
    def U_eff_ev(self) -> float:
        return self.U_ev - self.J_ev

    @property
    def U_eff_hartree(self) -> float:
        return self.U_eff_ev * _EV_TO_HARTREE


def ao_group_indices(basis: BasisSet) -> Dict[Tuple[int, int], List[int]]:
    """AO indices grouped by ``(atom_index, l)``.

    Walks ``basis.shells()`` in order, accumulating an AO offset, and
    bins the resulting AO indices into the ``(atom_index, l)`` channel
    corresponding to each shell. An atom carrying multiple shells of
    the same l (e.g. cc-pVDZ has two d shells on a transition metal)
    has all of those AOs in one bin.

    All shells in ``basis`` must be spherical (``shell.pure``); the
    Cartesian d/f Gaussian basis mixes lower-l character (the trace of
    the Cartesian d component is s-like) which breaks the rotational
    invariance of the Dudarev energy. Encountering a Cartesian shell
    raises ``NotImplementedError``.
    """
    groups: Dict[Tuple[int, int], List[int]] = {}
    offset = 0
    for shell in basis.shells():
        l = int(shell.l)
        if not bool(shell.pure):
            raise NotImplementedError(
                f"DFT+U requires spherical AOs, but a shell on atom "
                f"{int(shell.atom_index)} with l={l} is Cartesian. "
                f"Rebuild the basis with pure spherical harmonics (the "
                f"default for most vibe-qc workflows)."
            )
        n_ao = 2 * l + 1
        key = (int(shell.atom_index), l)
        groups.setdefault(key, []).extend(range(offset, offset + n_ao))
        offset += n_ao
    return groups


def compute_occupation_matrices(
    sites: Sequence[HubbardSite],
    P: np.ndarray,
    S: np.ndarray,
    ao_groups: Dict[Tuple[int, int], List[int]],
) -> Dict[Tuple[int, int], np.ndarray]:
    """AO-projected occupation matrix per Hubbard site.

    Computes ``n^A_l = (S P S)_{(A,l),(A,l)}`` -- the block of the
    transformed density on the AO indices that belong to atom A's
    angular-momentum channel l.

    Parameters
    ----------
    sites
        The Hubbard sites the caller wants occupation matrices for.
        ``ao_groups`` must contain a ``(site.atom_index, site.l)`` key
        for each one.
    P
        AO density matrix, shape ``(nbf, nbf)``. The spin convention is
        the caller's choice (see module docstring).
    S
        AO overlap matrix, shape ``(nbf, nbf)``.
    ao_groups
        Output of :func:`ao_group_indices` for the same basis ``P``
        and ``S`` live in. Passed in (rather than recomputed) so a
        caller projecting many sites pays the shell walk once.

    Returns
    -------
    dict
        Keyed by ``(atom_index, l)``; values are real square NumPy
        arrays of side ``len(ao_groups[(atom_index, l)])`` -- i.e.
        ``2l+1`` for a single-shell channel.

    Raises
    ------
    ValueError
        If ``P`` and ``S`` are not square and equal-shape.
    KeyError
        If a requested site has no corresponding channel in
        ``ao_groups`` (e.g. asking for d-AOs on an H/STO-3G atom, or
        pointing at an atom_index outside the molecule).
    """
    P_arr = np.asarray(P)
    S_arr = np.asarray(S)
    if (
        P_arr.ndim != 2
        or P_arr.shape[0] != P_arr.shape[1]
        or P_arr.shape != S_arr.shape
    ):
        raise ValueError(
            f"P and S must be square nbf x nbf matrices of equal "
            f"shape, got P.shape={P_arr.shape}, S.shape={S_arr.shape}"
        )
    SPS = S_arr @ P_arr @ S_arr
    out: Dict[Tuple[int, int], np.ndarray] = {}
    for site in sites:
        key = (site.atom_index, site.l)
        if key not in ao_groups:
            raise KeyError(
                f"No AOs for HubbardSite(atom_index={site.atom_index}, "
                f"l={site.l}) in the supplied basis. Available "
                f"(atom_index, l) channels: {sorted(ao_groups.keys())}"
            )
        idx = np.asarray(ao_groups[key], dtype=np.int64)
        out[key] = np.asarray(SPS[np.ix_(idx, idx)])
    return out


def compute_dudarev_energy(
    sites: Sequence[HubbardSite],
    occupation_matrices: Dict[Tuple[int, int], np.ndarray],
) -> float:
    """Dudarev (1998) rotationally-invariant +U energy.

    ::

        E_U = sum_A (U_eff / 2) * (tr(n^A_l) - tr(n^A_l @ n^A_l))
            = sum_A (U_eff / 2) * sum_m (l_m - l_m^2)

    where ``{l_m}`` are the eigenvalues of the Hermitian per-site
    occupation matrix. Contributions from fully-occupied (l = 1) and
    fully-empty (l = 0) shells are exactly zero; partial occupations
    give a positive penalty that drives the SCF toward integer
    occupations -- the physical purpose of +U.

    The trace form is used directly (no eigendecomposition) for
    robustness on near-degenerate occupation matrices where small
    eigenvalues amplify rounding.

    Returns
    -------
    float
        E_U in Hartree.
    """
    E_U = 0.0
    for site in sites:
        key = (site.atom_index, site.l)
        n = occupation_matrices[key]
        tr_n = float(np.trace(n))
        tr_n_squared = float(np.trace(n @ n))
        E_U += 0.5 * site.U_eff_hartree * (tr_n - tr_n_squared)
    return E_U


def _apply_dft_plus_u_to_options(options, basis, dft_plus_u) -> None:
    """Translate a sequence of user-facing :class:`HubbardSite` into
    the parallel C++ ``Options`` fields on ``options`` -- the same
    ``dft_plus_u_sites`` / ``dft_plus_u_ao_groups`` field names are
    used by ``RHFOptions`` / ``RKSOptions`` / (later) ``UHFOptions``
    / ``UKSOptions``.

    Lives here (not in ``vibeqc/__init__.py``) so both the public
    ``vibeqc.run_rhf`` / ``vibeqc.run_rks`` wrappers and
    ``vibeqc.runner._run_single_point`` can call it without the
    circular import a package-level helper would create. Mutates
    ``options`` in place; raises ``ValueError`` at the Python
    boundary if a requested ``(atom_index, l)`` channel has no AOs
    in the supplied basis.
    """
    sites = list(dft_plus_u or ())
    if not sites:
        return
    ao_groups_map = ao_group_indices(basis)
    sites_cxx = []
    ao_groups_list = []
    for site in sites:
        key = (site.atom_index, site.l)
        if key not in ao_groups_map:
            raise ValueError(
                f"HubbardSite(atom_index={site.atom_index}, "
                f"l={site.l}) has no AOs in the basis. Available "
                f"(atom_index, l) channels: "
                f"{sorted(ao_groups_map.keys())}"
            )
        sites_cxx.append(_HubbardSiteCxx(site.atom_index, site.l, site.U_eff_hartree))
        ao_groups_list.append(ao_groups_map[key])
    options.dft_plus_u_sites = sites_cxx
    options.dft_plus_u_ao_groups = ao_groups_list


def _v_ao_per_spin(
    sites: Sequence[HubbardSite],
    P_sigma: np.ndarray,
    S: np.ndarray,
    ao_groups_map: Dict[Tuple[int, int], List[int]],
) -> np.ndarray:
    """Compute the unsandwiched Dudarev V_AO for a per-spin density.

    ``V_AO^A_{mm'} = U_eff (1/2 d_{mm'} - n^A_l_{mm'})`` scattered into
    the full AO basis with zeros outside the (A,l) blocks. This is
    the standard "Dudarev V" before the variational S-sandwich. Used
    by :func:`_compute_dft_plus_u_gradient` -- the explicit gradient
    ``2 tr(V_AO_s S P_s dS/dR)`` is contracted with libint's
    overlap-derivative buffers, so V_AO (without sandwich) is the
    right matrix here.
    """
    nbf = int(P_sigma.shape[0])
    SPS = S @ P_sigma @ S
    V_AO = np.zeros((nbf, nbf))
    for site in sites:
        idx = np.asarray(
            ao_groups_map[(site.atom_index, site.l)],
            dtype=np.int64,
        )
        n = SPS[np.ix_(idx, idx)]
        block = -site.U_eff_hartree * n
        block[np.diag_indices_from(block)] += 0.5 * site.U_eff_hartree
        V_AO[np.ix_(idx, idx)] += block
    return 0.5 * (V_AO + V_AO.T)


def _compute_dft_plus_u_gradient(
    basis,
    molecule,
    S: np.ndarray,
    sites: Sequence[HubbardSite],
    *,
    P_total: "np.ndarray | None" = None,
    P_alpha: "np.ndarray | None" = None,
    P_beta: "np.ndarray | None" = None,
) -> np.ndarray:
    """Analytic dE_U/dR contribution from the explicit overlap-derivative.

    For a converged +U SCF (with the variational ``V_U_fock = S V_AO S``
    Fock contribution from :mod:`cpp/src/dft_plus_u.cpp`), the
    orbital-response (Pulay) piece is captured automatically by the
    standard ``W = C e C^T`` energy-weighted-density gradient term in
    ``compute_gradient(result)`` -- the converged ``e`` already
    includes the +U shift. What remains is the explicit derivative
    of ``E_U`` at fixed orbital coefficients:

    .. math::

        \\left. \\frac{\\partial E_U}{\\partial R}\\right|_C
            = 2 \\,\\mathrm{tr}\\!\\bigl(V_{AO,s} \\, S \\, P_s \\,
              \\partial S/\\partial R \\bigr) \\qquad
              \\text{(per spin)}

    where ``V_{AO,s}`` is the unsandwiched Dudarev V on the (A,l)
    blocks. Closed-shell sums to
    ``2 \\, \\mathrm{tr}(V_{AO,s} \\, S \\, P_{\\mathrm{total}} \\,
    \\partial S/\\partial R)``; open-shell sums per spin.

    Verified against FD of ``E_total`` at SCF-reconverged displaced
    geometries to ``6.9 x 10⁻¹¹`` Ha/bohr on H2O/STO-3G with
    ``U=4 eV`` on O's p-channel -- see docs/user_guide/dft_plus_u.md

    Parameters
    ----------
    basis, molecule
        AO basis + Molecule (passed to libint for ``dS/dR`` buffers).
    S
        Converged AO overlap matrix.
    sites
        Hubbard sites the SCF was run with.
    P_total
        Total (closed-shell) density. Pass for RHF / RKS results.
        The function internally halves it for the per-spin V_AO.
    P_alpha, P_beta
        Per-spin densities (UHF / UKS). Pass both for open-shell.

    Returns
    -------
    ndarray
        ``(n_atoms, 3)`` gradient contribution in Hartree/bohr that
        adds to the total ``dE/dR`` from
        :func:`vibeqc.compute_gradient` (or its UHF / RKS / UKS
        siblings).
    """
    sites = list(sites or ())
    if not sites:
        n_atoms = len(molecule.atoms()) if hasattr(molecule, "atoms") else 0
        return np.zeros((n_atoms, 3))
    closed_shell = P_total is not None
    open_shell = P_alpha is not None and P_beta is not None
    if closed_shell == open_shell:
        raise ValueError(
            "Pass exactly one of P_total (closed-shell) or "
            "(P_alpha, P_beta) (open-shell) to "
            "_compute_dft_plus_u_gradient."
        )
    from ._vibeqc_core import overlap_gradient_contribution

    ao_groups_map = ao_group_indices(basis)
    # Validate every site has AOs in the basis (mirror the run_rhf
    # boundary check for a clean error path).
    for site in sites:
        if (site.atom_index, site.l) not in ao_groups_map:
            raise ValueError(
                f"HubbardSite(atom_index={site.atom_index}, "
                f"l={site.l}) has no AOs in the basis."
            )

    if closed_shell:
        P = np.asarray(P_total)
        V_sigma = _v_ao_per_spin(sites, 0.5 * P, S, ao_groups_map)
        # dE_U_total/dR_explicit = 2 tr(V_AO_s S D dS/dR) per atom.
        M = 2.0 * (V_sigma @ S @ P)
    else:
        Pa = np.asarray(P_alpha)
        Pb = np.asarray(P_beta)
        V_a = _v_ao_per_spin(sites, Pa, S, ao_groups_map)
        V_b = _v_ao_per_spin(sites, Pb, S, ao_groups_map)
        # dE_U_total/dR_explicit = S_s 2 tr(V_AO_s S P_s dS/dR) per atom.
        M = 2.0 * ((V_a @ S @ Pa) + (V_b @ S @ Pb))
    # overlap_gradient_contribution returns -S_A tr(M dS/dR_A);
    # the +U explicit gradient is +S_A tr(M_sym dS/dR_A), so negate.
    return -np.asarray(overlap_gradient_contribution(basis, molecule, M))


def _compute_dft_plus_u_gradient_periodic_gamma(
    basis,
    system,
    sites: Sequence[HubbardSite],
    *,
    S_gamma: np.ndarray,
    P_total_gamma: "np.ndarray | None" = None,
    P_alpha_gamma: "np.ndarray | None" = None,
    P_beta_gamma: "np.ndarray | None" = None,
    lattice_opts=None,
) -> np.ndarray:
    """Γ-only periodic dE_U/dR contribution from the overlap derivative.

    The Γ-only periodic analogue of :func:`_compute_dft_plus_u_gradient`.
    For a Γ-only periodic SCF (Monkhorst-Pack ``[1,1,1]`` mesh, or the
    explicit ``run_rhf_periodic_gamma`` / ``run_rks_periodic_gamma``
    drivers), the occupation matrix is built from the home-cell density
    and overlap -- same math as the molecular path. The Pulay
    overlap-gradient contribution is

    .. math::

        \\left. \\frac{\\partial E_U}{\\partial R}\\right|_C
            = 2 \\,\\mathrm{tr}\\!\\bigl(V_{AO,s} \\, S \\, P_s \\,
              \\partial S/\\partial R \\bigr) \\quad\\text{(per spin)}

    where the matrices are Γ-block ``(nbf, nbf)`` and the
    ``dS/dR`` lattice sum is carried out by
    :func:`overlap_lattice_gradient_contribution` with the home-cell
    block set to ``M = 2 V_AO_s S P_s`` (summed over s for open-shell)
    and all other cells zero.

    Parameters
    ----------
    basis, system
        AO basis + periodic system (passed to libint for ``dS/dR``
        lattice buffers).
    sites
        Hubbard sites the SCF was run with.
    S_gamma
        Converged Γ-point AO overlap matrix.
    P_total_gamma
        Total (closed-shell) Γ-point density. Pass for periodic RHF /
        RKS Γ-only results. Internally halved to per-spin for V_AO_s.
    P_alpha_gamma, P_beta_gamma
        Per-spin Γ-point densities (periodic UHF / UKS Γ-only). Pass
        both for open-shell.
    lattice_opts
        :class:`LatticeSumOptions` -- must match the SCF.

    Returns
    -------
    ndarray
        ``(n_atoms, 3)`` gradient contribution in Hartree/bohr.

    Notes
    -----
    For multi-k SCFs the proper expression Bloch-folds per k:
    ``M(g) = S_k w_k Re[exp(-i k.g) V_{AO,s} S(k) P_s(k)]``. That
    multi-k version is queued for a follow-up -- this helper covers
    the Γ-only kmesh case only.
    """
    from ._vibeqc_core import (
        LatticeSumOptions,
        compute_overlap_lattice,
        overlap_lattice_gradient_contribution,
    )

    sites = list(sites or ())
    n_atoms = len(system.unit_cell)
    if not sites:
        return np.zeros((n_atoms, 3))
    if lattice_opts is None:
        lattice_opts = LatticeSumOptions()

    closed_shell = P_total_gamma is not None
    open_shell = P_alpha_gamma is not None and P_beta_gamma is not None
    if closed_shell == open_shell:
        raise ValueError(
            "Pass exactly one of P_total_gamma (closed-shell) or "
            "(P_alpha_gamma, P_beta_gamma) (open-shell) to "
            "_compute_dft_plus_u_gradient_periodic_gamma."
        )

    ao_groups_map = ao_group_indices(basis)
    for site in sites:
        if (site.atom_index, site.l) not in ao_groups_map:
            raise ValueError(
                f"HubbardSite(atom_index={site.atom_index}, "
                f"l={site.l}) has no AOs in the basis."
            )

    S = np.asarray(S_gamma, dtype=np.float64)
    if closed_shell:
        P = np.asarray(P_total_gamma, dtype=np.float64)
        V_sigma = _v_ao_per_spin(sites, 0.5 * P, S, ao_groups_map)
        # dE_U_total/dR_explicit = 2 tr(V_AO_s S P dS/dR) per atom.
        M_gamma = 2.0 * (V_sigma @ S @ P)
    else:
        Pa = np.asarray(P_alpha_gamma, dtype=np.float64)
        Pb = np.asarray(P_beta_gamma, dtype=np.float64)
        V_a = _v_ao_per_spin(sites, Pa, S, ao_groups_map)
        V_b = _v_ao_per_spin(sites, Pb, S, ao_groups_map)
        # dE_U_total/dR_explicit = S_s 2 tr(V_AO_s S P_s dS/dR) per atom.
        M_gamma = 2.0 * ((V_a @ S @ Pa) + (V_b @ S @ Pb))
    M_gamma = 0.5 * (M_gamma + M_gamma.T)

    # Build a LatticeMatrixSet with M at the home cell, zero elsewhere
    # (Γ-only: only the home-cell density contributes).
    M_set = compute_overlap_lattice(basis, system, lattice_opts)
    zero = np.zeros_like(M_gamma)
    for c, cell in enumerate(M_set.cells):
        idx = tuple(int(v) for v in np.asarray(cell.index).reshape(3))
        M_set.set_block(c, M_gamma if idx == (0, 0, 0) else zero)

    # overlap_lattice_gradient_contribution returns -S_g tr(W(g) dS(g)/dR);
    # the +U explicit gradient is +S_A tr(M_gamma dS(0)/dR_A), so negate.
    return -np.asarray(
        overlap_lattice_gradient_contribution(
            basis,
            system,
            M_set,
            lattice_opts,
        )
    )


def _bloch_fold_m_per_k(
    M_k_list: Sequence[np.ndarray],
    kmesh,
    template,
):
    """Bloch-fold a per-k (complex) matrix ``M(k)`` into the real-space
    LatticeMatrixSet using the same ``exp(-i k.g)`` convention as
    :func:`vibeqc.periodic_gradient_multi_k._bloch_fold_w_per_k` (which
    in turn mirrors ``real_space_density_from_kpoints``):

        M(g) = S_k w_k Re[ exp(-i k.g) . M(k) ]

    The lattice cell list comes from ``template`` (which the caller
    builds via :func:`compute_overlap_lattice` with the *same*
    :class:`LatticeSumOptions` the SCF used). The ``template`` is
    consumed -- its blocks are overwritten.
    """
    n_cells = len(template.cells)
    nbf = int(template.nbf)
    blocks = [np.zeros((nbf, nbf), dtype=np.float64) for _ in range(n_cells)]
    weights = list(kmesh.weights)
    kpts = list(kmesh.kpoints)
    for ik, M_k in enumerate(M_k_list):
        Mk = np.asarray(M_k, dtype=np.complex128)
        Mk_re = Mk.real
        Mk_im = Mk.imag
        w_k = float(weights[ik])
        k_cart = np.asarray(kpts[ik], dtype=np.float64)
        for c, cell in enumerate(template.cells):
            r = np.asarray(cell.r_cart, dtype=np.float64)
            phase = float(np.dot(k_cart, r))
            # exp(-i k.g) . M(k) -> Re part = cos.Re(M) + sin.Im(M)
            blocks[c] += w_k * (np.cos(phase) * Mk_re + np.sin(phase) * Mk_im)
    for c in range(n_cells):
        template.set_block(c, blocks[c])
    return template


def _compute_dft_plus_u_gradient_periodic_multi_k(
    basis,
    system,
    sites: Sequence[HubbardSite],
    *,
    kmesh,
    S_k_list: Sequence[np.ndarray],
    P_total_k_list: "Sequence[np.ndarray] | None" = None,
    P_alpha_k_list: "Sequence[np.ndarray] | None" = None,
    P_beta_k_list: "Sequence[np.ndarray] | None" = None,
    lattice_opts=None,
) -> np.ndarray:
    """Multi-k periodic dE_U/dR contribution from the overlap derivative.

    Generalises :func:`_compute_dft_plus_u_gradient_periodic_gamma`
    to a multi-k :class:`BlochKMesh`. For a converged multi-k +U SCF
    the gradient contribution at fixed orbital coefficients is

    .. math::

        \\left.\\frac{\\partial E_U}{\\partial R}\\right|_C
            = \\sum_k w_k \\, 2 \\,\\mathrm{tr}\\!\\bigl(V_{AO,s} \\, S(k) \\, P_s(k)
              \\, \\partial S(k)/\\partial R\\bigr)
            = \\sum_g \\mathrm{tr}\\bigl(M(g) \\, \\partial S(g)/\\partial R\\bigr)

    with

    .. math::

        M(g) = \\sum_k w_k \\,\\mathrm{Re}\\!\\bigl[ \\exp(-i k\\!\\cdot\\! g)
              \\, \\sum_\\sigma 2 V_{AO,s} S(k) P_s(k)\\bigr]

    (per spin, summed). ``V_{AO,s}`` is built from the k-averaged
    occupation matrix ``n_s = S_k w_k Re[(S(k) P_s(k) S(k))_{(A,l)}]``
    -- k-independent -- and ``M(k) = V_{AO,s} S(k) P_s(k)`` is
    Bloch-folded to real space via :func:`_bloch_fold_m_per_k`.

    Parameters
    ----------
    basis, system
        AO basis + periodic system.
    sites
        Hubbard sites the SCF was run with.
    kmesh
        :class:`BlochKMesh` -- same one passed to the SCF.
    S_k_list
        Per-k Bloch-summed AO overlap matrices ``S(k)`` -- same list
        the SCF used. ``len == n_k``.
    P_total_k_list
        Per-k closed-shell density (already Hermitised). Pass for
        RHF / RKS multi-k results. The kernel internally halves
        these to per-spin densities for ``V_{AO,s}``.
    P_alpha_k_list, P_beta_k_list
        Per-k per-spin densities. Pass for UHF / UKS multi-k results.
    lattice_opts
        :class:`LatticeSumOptions` -- must match the SCF (used to
        build the lattice cell list).

    Returns
    -------
    ndarray
        ``(n_atoms, 3)`` gradient contribution in Hartree/bohr.
    """
    from ._vibeqc_core import (
        LatticeSumOptions,
        _compute_dft_plus_u_multi_k_per_spin_cxx,
        compute_overlap_lattice,
        overlap_lattice_gradient_contribution,
    )

    sites = list(sites or ())
    n_atoms = len(system.unit_cell)
    if not sites:
        return np.zeros((n_atoms, 3))
    if lattice_opts is None:
        lattice_opts = LatticeSumOptions()

    closed_shell = P_total_k_list is not None
    open_shell = P_alpha_k_list is not None and P_beta_k_list is not None
    if closed_shell == open_shell:
        raise ValueError(
            "Pass exactly one of P_total_k_list (closed-shell) or "
            "(P_alpha_k_list, P_beta_k_list) (open-shell) to "
            "_compute_dft_plus_u_gradient_periodic_multi_k."
        )

    ao_groups_map = ao_group_indices(basis)
    for site in sites:
        if (site.atom_index, site.l) not in ao_groups_map:
            raise ValueError(
                f"HubbardSite(atom_index={site.atom_index}, "
                f"l={site.l}) has no AOs in the basis."
            )

    # Translate sites to the C++ struct + parallel ao_groups list.
    from ._vibeqc_core import _HubbardSiteCxx

    sites_cxx = [_HubbardSiteCxx(s.atom_index, s.l, s.U_eff_hartree) for s in sites]
    ao_groups = [ao_groups_map[(s.atom_index, s.l)] for s in sites]
    weights = list(kmesh.weights)
    S_k_list = [np.asarray(S, dtype=np.complex128) for S in S_k_list]

    if closed_shell:
        # Per-spin densities = P_total / 2.
        P_sigma_k = [0.5 * np.asarray(P, dtype=np.complex128) for P in P_total_k_list]
        # V_AO_s (k-independent, real) from the multi-k per-spin kernel.
        _, V_AO = _compute_dft_plus_u_multi_k_per_spin_cxx(
            sites_cxx,
            ao_groups,
            S_k_list,
            P_sigma_k,
            weights,
        )
        V_AO_c = np.asarray(V_AO, dtype=np.complex128)
        # M(k) = 2 * S_s V_AO_s S(k) P_s(k) -- closed-shell sums to
        # `2 V_AO_s S(k) P_total(k)` (since P_s = P/2 and s sum gives x2).
        M_k_list = [
            2.0 * (V_AO_c @ S @ np.asarray(P, dtype=np.complex128))
            for S, P in zip(S_k_list, P_total_k_list)
        ]
    else:
        Pa_k = [np.asarray(P, dtype=np.complex128) for P in P_alpha_k_list]
        Pb_k = [np.asarray(P, dtype=np.complex128) for P in P_beta_k_list]
        _, V_AO_a = _compute_dft_plus_u_multi_k_per_spin_cxx(
            sites_cxx,
            ao_groups,
            S_k_list,
            Pa_k,
            weights,
        )
        _, V_AO_b = _compute_dft_plus_u_multi_k_per_spin_cxx(
            sites_cxx,
            ao_groups,
            S_k_list,
            Pb_k,
            weights,
        )
        Va = np.asarray(V_AO_a, dtype=np.complex128)
        Vb = np.asarray(V_AO_b, dtype=np.complex128)
        M_k_list = [
            2.0 * (Va @ S @ Pa + Vb @ S @ Pb) for S, Pa, Pb in zip(S_k_list, Pa_k, Pb_k)
        ]

    # Bloch-fold M(k) -> M(g) into a LatticeMatrixSet.
    M_set = compute_overlap_lattice(basis, system, lattice_opts)
    M_set = _bloch_fold_m_per_k(M_k_list, kmesh, M_set)

    # overlap_lattice_gradient_contribution returns -S_g tr(M(g) dS(g)/dR);
    # the +U explicit gradient is +S_g tr(M_sym dS/dR), so negate.
    return -np.asarray(
        overlap_lattice_gradient_contribution(
            basis,
            system,
            M_set,
            lattice_opts,
        )
    )
