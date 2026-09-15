"""Phase 17a-1 -- molecular finite-difference Hessian + harmonic frequencies.

Two-sided finite differences on the analytic atomic gradient give the
3N x 3N nuclear Hessian in Hartree/bohr^2. Diagonalizing the mass-
weighted Hessian (with translation + rotation modes projected out)
returns the 3N - 6 (or 3N - 5 for linear molecules) harmonic
vibrational frequencies in cm⁻¹.

API
---
::

    from vibeqc import compute_hessian_fd, HessianFDOptions

    result = compute_hessian_fd(
        mol, basis_name="sto-3g", method="RHF",
    )
    print(result.frequencies_cm1)

For DFT methods, pass ``scf_options`` carrying the functional plus
``grid_options``::

    opts = vibeqc.RKSOptions()
    opts.functional = "PBE"
    result = compute_hessian_fd(
        mol, "6-31g*", method="RKS", scf_options=opts,
    )

Cost
----
6N gradient evaluations (one ± displacement per Cartesian DOF). Each
evaluation = SCF + analytic gradient -- typically a few seconds for
small molecules and order N^3 in basis size for the SCF itself. Phase
17b (analytic CPHF) replaces this with a single linear-response solve
and an order-of-magnitude speed-up.

Convention notes
----------------
* Imaginary modes (saddle points / non-stationary geometries) are
  reported as **negative** wavenumbers -- this matches the convention
  used by Gaussian, NWChem, ORCA, and PySCF. Mathematically they're
  ``i * |ν|``; the negative-real encoding is just convenient for
  sorting and plotting.

* Trans / rot projection is "exact zero" -- when
  ``project_trans_rot=True`` (default) the 6 (or 5 for linear) lowest
  modes are replaced with literal zeros so the user can rely on
  ``frequencies_cm1[6:]`` (or ``[5:]``) being the vibrational subset.

* Mass weighting uses standard atomic masses (the same table used by
  :func:`vibeqc.center_of_mass`). For isotope effects, edit
  ``HessianFDOptions.atomic_masses_amu`` to override the table.

* Step size 0.005 bohr (~0.0026 Å) is the default. Smaller steps risk
  round-off at SCF tolerance; larger steps include cubic anharmonicity.
  Run a step-size convergence check if frequencies look suspicious.

* **Tighten SCF tolerances.** The default ``RHFOptions`` use
  ``conv_tol_grad = 1e-6``; that is fine for energies but contributes
  ~1e-6 / step ≈ 2e-4 Ha/bohr^2 of noise to each Hessian element at
  the default step. For µHa-level frequency reproducibility, pass
  ``scf_options`` with ``conv_tol_energy = 1e-12`` and
  ``conv_tol_grad = 1e-9`` (the PySCF default for Hessian work).

* **ECPs and the JK backend follow the SCF options.** The gradient that
  is differentiated is built from ``scf_options`` unless an explicit
  ``gradient_options`` is passed: density fitting / COSX and the ECP
  fields of both routes (libecpint XML-library ``ecp_centers`` and
  inline ``ecp_primitive_blocks``) are mirrored, and the ECP centres --
  absolute coordinates on the options -- are moved with each displaced
  atom (#576). Pre-fix the default gradient was the bare-Z all-electron
  one and the centres stayed behind, so an ECP-bearing atom's SCF
  diverged at the first displacement while the all-electron atoms'
  columns were silently wrong. The analytic Hessian kernels
  (``compute_hessian_*_analytic``) still take no ECP options; use this
  FD path for ECP systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ._vibeqc_core import (
    Atom,
    BasisSet,
    GridOptions,
    Molecule,
    RHFOptions,
    RKSOptions,
    UHFOptions,
    UKSOptions,
    compute_dipole,
    compute_gradient,
    compute_gradient_rks,
    compute_gradient_uhf,
    compute_gradient_uks,
    run_uks,
)
from .ecp_metadata import (
    attach_inline_ecp_options_from_basis_sidecar,
    ecp_centre_atom_indices,
    ecp_centres_follow_atoms,
    effective_nuclear_charges,
)
from .gradient_options import gradient_options_from_scf
from .properties import _ATOMIC_MASSES, _total_density, center_of_mass

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

# CODATA 2018: 1 atomic-unit angular frequency in cm⁻¹.
# Derivation: w in a.u. corresponds to E = ℏw with ℏ = 1, so w[a.u.]
# numerically equals an energy in Hartree, and 1 Ha = 219474.6313632 cm⁻¹.
_HARTREE_TO_CM_INV: float = 219474.6313632

# 1 atomic mass unit in electron-mass units (so Hessian [Ha/bohr^2]
# divided by mass [m_e] gives w^2 [a.u.^2]). CODATA 2018: m_u/m_e.
_AMU_TO_ELECTRON_MASS: float = 1822.888486209

# IR intensity conversion (Wilson-Decius-Cross formula
#   I = (N_A pi / (3 c^2)) . |dmu/dQ|^2
# in SI, then rescaled to km/mol).
#
# The widely-quoted 974.864 km/mol per |dmu/dQ|^2[a.u.] factor (Gaussian,
# Halls-Schlegel JCC 1998) assumes Q is the mass-weighted normal
# coordinate with masses in *amu* (i.e. Q in √amu.bohr). Our
# mass-weighted Hessian uses *electron-mass* units throughout, so Q
# is in √m_e.bohr and |dmu/dQ|^2 differs by a factor m_u/m_e ≈ 1823.
# Multiply through to get the right factor for our convention:
_AU_TO_KM_PER_MOL: float = 974.864 * _AMU_TO_ELECTRON_MASS


# ----------------------------------------------------------------------
# Public dataclasses
# ----------------------------------------------------------------------

@dataclass
class HessianFDOptions:
    """Knobs for :func:`compute_hessian_fd`.

    step_bohr
        Centered-finite-difference displacement (default 0.005 bohr).
        Smaller -> more sensitive to SCF round-off; larger -> more
        anharmonicity contamination. 0.001-0.01 bohr is the practical
        sweet spot.

    sym_strict
        Symmetrize the FD Hessian as ``H = (H + H.T) / 2`` before
        mass-weighting (default True). Set False to keep the raw FD
        matrix -- useful for diagnosing translational invariance of
        the gradient.

    project_trans_rot
        Project the 3 (linear: 2) translational + 3 (linear: 2)
        rotational zero modes out of the mass-weighted Hessian
        before diagonalisation (default True). When True, the
        corresponding entries of ``frequencies_cm1`` are exactly
        zero so callers can slice ``[6:]`` (or ``[5:]`` for linear
        molecules) for the vibrational subset.

    atomic_masses_amu
        Optional per-atom mass override (length n_atoms, in amu).
        Defaults to the standard atomic masses for the molecule's
        Z values. Use this for isotope substitution (D for H, ¹^3C,
        etc.).
    """
    step_bohr: float = 0.005
    sym_strict: bool = True
    project_trans_rot: bool = True
    atomic_masses_amu: Optional[Sequence[float]] = None
    include_dipole_derivatives: bool = False
    """When True, also evaluate the dipole moment at every displaced
    geometry and assemble the dipole-derivative tensor
    ``dmu_b/dR_ia`` (in atomic units, e). Cheap because the SCF is
    already done -- adds only one ``compute_dipole`` call per
    displacement. Required by :func:`ir_intensities`. Default False."""
    frozen_indices: Optional[Sequence[int]] = None
    """Atom indices to hold fixed (not displaced). When set, the result
    is the **partial-Hessian** sub-block over the *unfrozen* atoms only --
    the ``(3M, 3M)`` block of the full Hessian for ``M`` unfrozen atoms --
    and its ``3M`` frequencies are all vibrational: the frozen atoms
    anchor the system, so there are no translational/rotational zero
    modes and ``project_trans_rot`` is forced off. This is the
    partial-Hessian vibrational analysis (PHVA; Li & Jensen, *Theor.
    Chem. Acc.* 2002) used to characterise a transition state on a slab
    while holding the bulk-like substrate layers fixed -- it cuts the
    displacement count from ``6N`` to ``6M``. ``None`` => every atom is
    displaced (the full Hessian). Downstream
    :func:`vibeqc.compute_thermochemistry` treats a partial-Hessian
    result as a vibrational-only (anchored) system: no gas-phase
    translational / rotational partition functions."""


@dataclass
class HessianResult:
    """Output of :func:`compute_hessian_fd`.

    hessian
        (3N, 3N) matrix in Hartree/bohr^2. Indexing convention:
        row/col ``3*i + a`` is atom ``i``, Cartesian direction
        ``a in {0, 1, 2}`` for ``{x, y, z}``.

    hessian_mw
        Mass-weighted Hessian: ``M^(-1/2) H M^(-1/2)`` where ``M``
        is the diagonal mass matrix in electron-mass units, with
        each atom's mass duplicated 3x (one per Cartesian DOF).
        Eigenvalues of this matrix are ``w^2`` in atomic units.

    frequencies_cm1
        (3N,) array of harmonic frequencies in cm⁻¹, sorted
        ascending. Imaginary modes appear as negative numbers.
        Trans/rot zero modes appear as exact 0.0 when
        ``project_trans_rot=True``.

    normal_modes
        (3N, 3N) orthonormal mass-weighted normal-mode matrix
        (each column is a mode). Cartesian displacement pattern
        for atom ``i`` in mode ``k`` is
        ``normal_modes[3i:3i+3, k] / sqrt(M_i)``.

    imaginary_count
        Number of modes with w^2 < 0 after trans/rot projection.

    n_displacements
        ``6 * n_atoms`` for centered FD.

    is_linear
        True when the molecule's inertia tensor is rank-deficient
        (collinear atoms) -- 5 zero modes instead of 6.
    """
    hessian: np.ndarray
    hessian_mw: np.ndarray
    frequencies_cm1: np.ndarray
    normal_modes: np.ndarray
    imaginary_count: int
    n_displacements: int
    is_linear: bool
    dipole_derivatives: Optional[np.ndarray] = None
    """(3N, 3) tensor of ``dmu_b/dR_ia`` in atomic units (e), populated
    when ``HessianFDOptions.include_dipole_derivatives=True`` (else
    None). Row index ``3*i + a``; column ``b in {0,1,2}`` for ``{x,y,z}``.
    Used by :func:`ir_intensities` to compute IR band intensities."""
    dipole_origin: Optional[np.ndarray] = None
    """(3,) array, the dipole reference origin in bohr -- typically the
    center of mass of the *reference* (undisplaced) geometry, which
    keeps the dipole-derivative tensor origin-consistent across
    displacements. None when dipole derivatives weren't requested."""
    masses_amu: Optional[np.ndarray] = None
    """(n_atoms,) atomic masses in amu used during mass-weighting -- for
    a partial Hessian (frozen atoms) this is the ``M`` unfrozen masses,
    matching the ``3M`` rows/columns. Stored so :func:`ir_intensities`
    and downstream thermochemistry can stay self-contained without
    re-deriving masses from the molecule."""
    unfrozen_indices: Optional[np.ndarray] = None
    """``None`` for a full Hessian. When ``HessianFDOptions.frozen_indices``
    was set, the ``(M,)`` array of *unfrozen* atom indices this partial
    Hessian's ``3M`` rows/columns correspond to (row/col ``3*k + a`` is
    unfrozen atom ``unfrozen_indices[k]``, Cartesian ``a``). Lets callers
    map normal modes back to atoms and signals
    :func:`vibeqc.compute_thermochemistry` to use the vibrational-only
    (anchored-system) partition function."""


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

def _atom_positions_bohr(mol: Molecule) -> np.ndarray:
    """(n_atoms, 3) array of atom coordinates in bohr."""
    return np.asarray(
        [[a.xyz[0], a.xyz[1], a.xyz[2]] for a in mol.atoms],
        dtype=np.float64,
    )


def _atom_z(mol: Molecule) -> np.ndarray:
    """(n_atoms,) int array of atomic numbers."""
    return np.asarray([int(a.Z) for a in mol.atoms], dtype=np.int64)


def _displaced_molecule(mol: Molecule, atom_index: int,
                        cart: int, delta: float) -> Molecule:
    """Return a copy of ``mol`` with one atom shifted by ``delta`` along
    Cartesian axis ``cart``. ``delta`` is in bohr and may be negative."""
    atoms = []
    for j, a in enumerate(mol.atoms):
        xyz = [a.xyz[0], a.xyz[1], a.xyz[2]]
        if j == atom_index:
            xyz[cart] += delta
        atoms.append(Atom(int(a.Z), xyz))
    return Molecule(atoms, charge=mol.charge, multiplicity=mol.multiplicity)


def _dipole_at(mol: Molecule, basis: BasisSet,
               result, origin: np.ndarray,
               nuclear_charges: np.ndarray) -> np.ndarray:
    """Total dipole moment (electronic + nuclear, in atomic units) of
    a converged SCF result, evaluated about ``origin`` (bohr).

    Mirrors :func:`vibeqc.dipole_moment` but returns a plain ``ndarray``
    instead of the ``DipoleMoment`` dataclass -- used inside the FD
    Hessian loop where we just need the three numbers.

    ``nuclear_charges`` is the per-atom charge the valence-only Hamiltonian
    sees (``Z - n_core`` on ECP atoms, bare ``Z`` otherwise;
    :func:`vibeqc.ecp_metadata.effective_nuclear_charges`). It is the same
    vector for every displaced geometry, so the caller resolves it once on
    the reference molecule. With bare ``Z`` on an ECP system each ECP
    atom's dipole-derivative row picked up a spurious ``n_core`` per
    Cartesian direction and leaked into every IR intensity of a mode that
    moves that atom (GitLab #642).
    """
    dip = compute_dipole(basis, [float(x) for x in origin])
    Mx = np.asarray(dip.x)
    My = np.asarray(dip.y)
    Mz = np.asarray(dip.z)
    P = _total_density(result)
    mu_e = -np.array([
        np.einsum("ij,ji->", P, Mx),
        np.einsum("ij,ji->", P, My),
        np.einsum("ij,ji->", P, Mz),
    ])
    mu_n = np.zeros(3)
    for z, atom in zip(nuclear_charges, mol.atoms):
        z = float(z)
        mu_n[0] += z * (atom.xyz[0] - origin[0])
        mu_n[1] += z * (atom.xyz[1] - origin[1])
        mu_n[2] += z * (atom.xyz[2] - origin[2])
    return mu_e + mu_n


def _scf_and_gradient_at(
    mol: Molecule,
    basis_name: str,
    method: str,
    scf_options,
    grid_options,
    gradient_options=None,
    *,
    reference_mol: Optional[Molecule] = None,
):
    """Run SCF + analytic gradient at ``mol``. Returns
    ``(basis, result, gradient)`` so the caller can reuse the
    converged result (e.g. to compute the dipole moment) without a
    redundant SCF.

    The gradient differentiates the Hamiltonian the SCF at ``mol``
    solves (#576):

    * ``gradient_options=None`` derives the ``GradientOptions`` from
      ``scf_options`` -- JK backend (density_fit / cosx / aux_basis) and
      ECP fields, both routes -- through
      :func:`vibeqc.gradient_options.gradient_options_from_scf`. An
      explicit object is used as given, except that its ECP fields are
      synchronised to the SCF options for the call.
    * When ``reference_mol`` is given, ``scf_options`` describes that
      geometry and ``mol`` is a displaced copy of it: the ECP centres
      (``ecp_centers`` / ``ecp_primitive_centers``, which are absolute
      coordinates) are moved onto the displaced atoms for the duration
      of the call and restored afterwards
      (:func:`vibeqc.ecp_metadata.ecp_centres_follow_atoms`). Without
      this the displaced atom silently reverts to bare Z while V_ECP
      keeps acting and the SCF diverges.
    """
    with ecp_centres_follow_atoms(
        scf_options, reference_mol, mol, gradient_options
    ):
        if gradient_options is None:
            gradient_options = gradient_options_from_scf(scf_options)
        # Preloaded READ matrices belong to the reference geometry. Project
        # from that fixed basis at every displacement and restore them after
        # the call, so projections do not accumulate along the FD stencil.
        restored = {}
        if reference_mol is not None and scf_options is not None:
            from ._initial_guess import coerce_initial_guess
            from ._vibeqc_core import InitialGuess
            if coerce_initial_guess(scf_options.initial_guess) == InitialGuess.READ:
                from .guess_read import _to_current_basis
                source_basis = BasisSet(reference_mol, basis_name)
                target_basis = BasisSet(mol, basis_name)
                for name in ("read_density", "read_density_alpha", "read_density_beta"):
                    value = getattr(scf_options, name, None)
                    if value is not None and np.asarray(value).size:
                        restored[name] = np.asarray(value).copy()
                        setattr(scf_options, name, _to_current_basis(
                            value, source_basis, target_basis,
                        ))
        try:
            return _scf_and_gradient_at_impl(
                mol, basis_name, method, scf_options, grid_options,
                gradient_options,
            )
        finally:
            for name, value in restored.items():
                setattr(scf_options, name, value)



def _scf_and_gradient_at_impl(
    mol: Molecule,
    basis_name: str,
    method: str,
    scf_options,
    grid_options,
    gradient_options,
):
    """Body of :func:`_scf_and_gradient_at`; ``gradient_options`` is a
    concrete ``GradientOptions`` already matched to ``scf_options``."""
    # Public wrappers apply selector validation and project file-backed READ
    # sources onto each displaced geometry before entering the native driver.
    from . import run_rhf, run_uhf, run_rks, run_uks
    basis = BasisSet(mol, basis_name)
    if method == "RHF":
        result = run_rhf(mol, basis, scf_options)
        if not result.converged:
            raise RuntimeError(
                "FD Hessian: SCF failed to converge at a displaced geometry "
                "(method=RHF). Try a tighter scf_options or a smaller step."
            )
        g = compute_gradient(mol, basis, result, gradient_options)
    elif method == "UHF":
        result = run_uhf(mol, basis, scf_options)
        if not result.converged:
            raise RuntimeError(
                "FD Hessian: UHF SCF failed to converge at a displaced "
                "geometry."
            )
        g = compute_gradient_uhf(mol, basis, result, gradient_options)
    elif method == "RKS":
        result = run_rks(mol, basis, scf_options)
        if not result.converged:
            raise RuntimeError(
                "FD Hessian: RKS SCF failed to converge at a displaced "
                "geometry."
            )
        g = compute_gradient_rks(mol, basis, result, grid_options,
                                 gradient_options)
    elif method == "UKS":
        result = run_uks(mol, basis, scf_options)
        if not result.converged:
            raise RuntimeError(
                "FD Hessian: UKS SCF failed to converge at a displaced "
                "geometry."
            )
        g = compute_gradient_uks(mol, basis, result, grid_options,
                                 gradient_options)
    elif method == "ROHF":
        # Restricted open-shell HF: analytic gradient with the ROHF
        # energy-weighted density (vibeqc.rohf.compute_rohf_gradient).
        from .rohf import ROHFOptions, compute_rohf_gradient, run_rohf

        result = run_rohf(
            mol, basis, scf_options if scf_options is not None else ROHFOptions()
        )
        if not result.converged:
            raise RuntimeError(
                "FD Hessian: ROHF SCF failed to converge at a displaced "
                "geometry."
            )
        g = compute_rohf_gradient(
            mol, basis, result, gradient_options=gradient_options
        )
    else:
        raise ValueError(
            f"FD Hessian: unknown method {method!r}. "
            "Expected one of {'RHF', 'UHF', 'RKS', 'UKS', 'ROHF'}."
        )
    return basis, result, np.asarray(g, dtype=np.float64)


def _resolve_masses(mol: Molecule,
                    override: Optional[Sequence[float]]) -> np.ndarray:
    """Look up atomic masses (in amu) from the standard table, applying
    a user-supplied override if given. Returns an (n_atoms,) array."""
    n_atoms = len(mol.atoms)
    if override is not None:
        m = np.asarray(override, dtype=np.float64)
        if m.shape != (n_atoms,):
            raise ValueError(
                f"FD Hessian: atomic_masses_amu has shape {m.shape}, "
                f"expected ({n_atoms},)"
            )
        if np.any(m <= 0):
            raise ValueError("FD Hessian: atomic masses must be positive")
        return m
    masses = np.empty(n_atoms, dtype=np.float64)
    for i, atom in enumerate(mol.atoms):
        z = int(atom.Z)
        if z < len(_ATOMIC_MASSES) and _ATOMIC_MASSES[z] > 0:
            masses[i] = _ATOMIC_MASSES[z]
        else:
            raise ValueError(
                f"FD Hessian: no built-in mass for Z={z}. Pass an explicit "
                "atomic_masses_amu override."
            )
    return masses


def _build_trans_rot_modes(positions_bohr: np.ndarray,
                           masses_e: np.ndarray) -> tuple[np.ndarray, bool]:
    """Build orthonormal mass-weighted translation + rotation vectors,
    one column per zero mode. Detects linear molecules (rank-deficient
    rotational subspace) and emits 5 modes instead of 6.

    Convention: vectors live in mass-weighted Cartesian space, i.e.
    if ``v`` is a column then ``v[3i:3i+3]`` is atom ``i``'s
    contribution scaled by ``sqrt(m_i)``. Translation/rotation in
    *real* coordinates corresponds to no change in potential energy
    by Galilean / SO(3) invariance, so these are exact zero modes
    of the mass-weighted Hessian.

    Returns (modes, is_linear) where modes has shape (3N, n_zero) and
    columns are orthonormal.
    """
    n_atoms = positions_bohr.shape[0]
    n_dof = 3 * n_atoms
    sqrt_m = np.sqrt(masses_e)  # (n_atoms,)

    # --- Translations ---------------------------------------------------
    trans = np.zeros((n_dof, 3), dtype=np.float64)
    for alpha in range(3):
        for i in range(n_atoms):
            trans[3 * i + alpha, alpha] = sqrt_m[i]

    # --- Rotations (about the center of mass) --------------------------
    total_mass = masses_e.sum()
    com = (masses_e[:, None] * positions_bohr).sum(axis=0) / total_mass
    R = positions_bohr - com  # (n_atoms, 3) coords relative to COM

    rot = np.zeros((n_dof, 3), dtype=np.float64)
    # Generator R_a x: column for rotation about a has components
    #   (R_i x ê_a) at atom i, scaled by sqrt(m_i).
    e = np.eye(3)
    for alpha in range(3):
        cross = np.cross(R, e[alpha])  # (n_atoms, 3)
        for i in range(n_atoms):
            rot[3 * i:3 * i + 3, alpha] = sqrt_m[i] * cross[i]

    raw = np.concatenate([trans, rot], axis=1)  # (n_dof, 6)

    # Orthonormalise via QR; drop columns whose norm collapsed below a
    # tolerance (linear-molecule case: one rotational mode is null).
    Q, R_qr = np.linalg.qr(raw)
    diag = np.abs(np.diag(R_qr))
    # Tolerance scaled to the largest column so it works for any size
    keep = diag > 1e-8 * diag.max()
    modes = Q[:, keep]
    is_linear = modes.shape[1] == 5
    if modes.shape[1] not in (5, 6):
        # Single-atom or weirdly degenerate case -- fall through with
        # whatever survived. Hard to reach in practice for n_atoms >= 2.
        pass
    return modes, is_linear


def _project_out(H_mw: np.ndarray, zero_modes: np.ndarray) -> np.ndarray:
    """Return P . H_mw . P with P = I - S_k v_k v_k^T (zero modes
    orthonormal)."""
    n = H_mw.shape[0]
    P = np.eye(n) - zero_modes @ zero_modes.T
    return P @ H_mw @ P


def _enforce_projected_zero(eigenvalues: np.ndarray,
                            eigenvectors: np.ndarray,
                            n_zero: int) -> tuple[np.ndarray, np.ndarray]:
    """After projection, the lowest ``n_zero`` eigenvalues should be
    near-zero (numerical noise of the projector). Replace them with
    exact zeros so callers can slice them off cleanly. Eigenvalues
    are returned in ascending order -- pull the n_zero with smallest
    |value| (not most-negative; small noise can be of either sign)."""
    abs_idx = np.argsort(np.abs(eigenvalues))
    zero_idx = abs_idx[:n_zero]
    eigenvalues = eigenvalues.copy()
    eigenvalues[zero_idx] = 0.0
    return eigenvalues, eigenvectors


def _omega2_to_cm_inv(omega2: np.ndarray) -> np.ndarray:
    """Convert eigenvalues of the mass-weighted Hessian (in atomic units
    with masses in m_e) to wavenumbers in cm⁻¹. Imaginary modes
    (negative eigenvalues) come out negative."""
    out = np.empty_like(omega2)
    pos = omega2 >= 0
    out[pos] = np.sqrt(omega2[pos]) * _HARTREE_TO_CM_INV
    out[~pos] = -np.sqrt(-omega2[~pos]) * _HARTREE_TO_CM_INV
    return out


# ----------------------------------------------------------------------
# Public driver
# ----------------------------------------------------------------------

def compute_hessian_fd(
    mol: Molecule,
    basis_name: str,
    method: str = "RHF",
    *,
    scf_options=None,
    grid_options: Optional[GridOptions] = None,
    grid_level: str = "orca-defgrid3",
    gradient_options=None,
    hessian_options: Optional[HessianFDOptions] = None,
) -> HessianResult:
    """Build the 3N x 3N nuclear Hessian by central differences on the
    analytic atomic gradient and return mass-weighted normal-mode
    frequencies.

    See module docstring for API examples and convention notes.

    Parameters
    ----------
    mol :
        Molecule at the geometry where the Hessian is wanted.
        Should be a stationary point for the resulting frequencies
        to be physical, but the function doesn't enforce this so
        non-equilibrium force constants can be evaluated too.
    basis_name :
        Basis-set name. Re-built per displacement so basis-center
        derivative pieces are absorbed into the FD.
    method :
        One of ``"RHF"``, ``"UHF"``, ``"ROHF"``, ``"RKS"``, ``"UKS"``
        (case-insensitive).
    scf_options :
        Per-method options struct. Default-initialises a fresh
        ``RHFOptions`` / ``UHFOptions`` / ``ROHFOptions`` /
        ``RKSOptions`` / ``UKSOptions`` if None. For DFT methods the functional
        comes from this struct's ``functional`` field.
    grid_options :
        DFT integration grid. Forwarded to
        :func:`compute_gradient_rks` / ``_uks``. Defaults to
        the resolved SCF grid for DFT methods, unused for HF. An explicit
        grid is used by both the displaced SCFs and their gradients.
    grid_level :
        Preset for an untouched SCF grid (default ``"orca-defgrid3"``).
        Pass ``"legacy"`` for construction defaults; custom grid fields win.
    gradient_options :
        ``GradientOptions`` for the analytic gradient that is
        differentiated. ``None`` (the default) derives it from
        ``scf_options`` -- JK backend (density_fit / aux_basis / cosx)
        and ECP fields, both the XML-library and the inline-primitive
        route -- so the differentiated gradient describes the same
        Hamiltonian as the SCF energy (#576). Pass an explicit object
        only to override the JK backend; its ECP fields are always
        re-synchronised to ``scf_options`` at every displaced geometry.
        On ECP systems the ECP centres carry absolute coordinates and
        are moved with the displaced atom; a centre that coincides with
        no atom of ``mol`` is refused up front (``ValueError``).
    hessian_options :
        Step size + projection knobs.
    """
    if hessian_options is None:
        hessian_options = HessianFDOptions()

    method_norm = method.upper()
    if method_norm not in {"RHF", "UHF", "ROHF", "RKS", "UKS"}:
        raise ValueError(
            f"FD Hessian: unknown method {method!r}. "
            "Expected one of 'RHF', 'UHF', 'ROHF', 'RKS', 'UKS'."
        )

    if scf_options is None:
        if method_norm == "ROHF":
            from .rohf import ROHFOptions

            scf_options = ROHFOptions()
        else:
            scf_options = {
                "RHF": RHFOptions(),
                "UHF": UHFOptions(),
                "RKS": RKSOptions(),
                "UKS": UKSOptions(),
            }[method_norm]
    if method_norm in {"RKS", "UKS"}:
        from .runner import apply_ks_grid_default
        if grid_options is not None:
            scf_options.grid = grid_options
        else:
            apply_ks_grid_default(scf_options, grid_level)
        grid_options = scf_options.grid
    elif grid_options is None:
        grid_options = GridOptions()

    # A direct caller commonly supplies only an ECP-paired basis name. Attach
    # that sidecar before deriving GradientOptions or entering the displacement
    # loop; otherwise the first displaced SCF would discover the ECP too late
    # and its gradient would still describe the all-electron defaults.
    attach_inline_ecp_options_from_basis_sidecar(
        scf_options, mol, BasisSet(mol, basis_name)
    )

    # ECP centres are absolute coordinates on the SCF options. Refuse a
    # set that does not sit on the atoms of ``mol`` before spending any
    # SCF on it: such an SCF is already inconsistent (bare Z on the atom,
    # V_ECP still acting), and the displaced-geometry loop below relies on
    # this mapping to move each centre with its atom (#576).
    ecp_centre_atom_indices(scf_options, mol)
    # Nuclear charges for the dipole-derivative rows: Z - n_core on ECP
    # atoms (both routes), bare Z otherwise. Resolved once here -- the
    # displaced geometries carry the same ECP assignment (#642).
    _dipole_nuclear_charges = effective_nuclear_charges(mol, scf_options)

    # ---- Build the Hessian via centered FD ---------------------------------
    n_atoms = len(mol.atoms)
    if n_atoms < 1:
        raise ValueError("FD Hessian: empty molecule")
    n_dof = 3 * n_atoms
    H = np.zeros((n_dof, n_dof), dtype=np.float64)
    delta = float(hessian_options.step_bohr)
    if delta <= 0:
        raise ValueError("FD Hessian: step_bohr must be positive")

    # Partial-Hessian (frozen atoms): displace only the unfrozen atoms;
    # the result is the (3M, 3M) sub-block over them (see HessianFDOptions
    # .frozen_indices). Frozen columns are never filled and frozen rows
    # are dropped at the end.
    frozen_set = {int(i) for i in (hessian_options.frozen_indices or ())}
    bad = sorted(i for i in frozen_set if i < 0 or i >= n_atoms)
    if bad:
        raise ValueError(
            f"FD Hessian: frozen_indices {bad} out of range [0, {n_atoms})"
        )
    is_partial = len(frozen_set) > 0
    unfrozen = [i for i in range(n_atoms) if i not in frozen_set]
    if not unfrozen:
        raise ValueError(
            "FD Hessian: every atom is frozen -- nothing to displace"
        )
    unfrozen_dof = [3 * i + c for i in unfrozen for c in range(3)]

    # Optional dipole-derivative tensor. Reference origin = COM of the
    # undisplaced geometry; using the same origin for all dipole
    # evaluations keeps the dipole-derivative tensor free of spurious
    # origin-shift contributions (which matter for charged systems and
    # leak into IR intensities even for neutral ones if you mix origins).
    include_dipole = bool(hessian_options.include_dipole_derivatives)
    if include_dipole:
        dipole_origin = center_of_mass(mol)
        dipole_derivs = np.zeros((n_dof, 3), dtype=np.float64)
    else:
        dipole_origin = None
        dipole_derivs = None

    n_disp = 0
    for i in range(n_atoms):
        if i in frozen_set:
            continue  # frozen atom: not displaced (partial Hessian)
        for cart in range(3):
            mol_plus = _displaced_molecule(mol, i, cart, +delta)
            mol_minus = _displaced_molecule(mol, i, cart, -delta)
            basis_p, res_p, g_plus = _scf_and_gradient_at(
                mol_plus, basis_name, method_norm,
                scf_options, grid_options, gradient_options,
                reference_mol=mol,
            )
            basis_m, res_m, g_minus = _scf_and_gradient_at(
                mol_minus, basis_name, method_norm,
                scf_options, grid_options, gradient_options,
                reference_mol=mol,
            )
            n_disp += 2
            # Column 3i+cart of H = (g(+d) - g(-d)) / (2d), flattened.
            col = (g_plus - g_minus).reshape(-1) / (2.0 * delta)
            H[:, 3 * i + cart] = col

            if include_dipole:
                mu_p = _dipole_at(
                    mol_plus, basis_p, res_p, dipole_origin, _dipole_nuclear_charges
                )
                mu_m = _dipole_at(
                    mol_minus, basis_m, res_m, dipole_origin, _dipole_nuclear_charges
                )
                dipole_derivs[3 * i + cart, :] = (mu_p - mu_m) / (2.0 * delta)

    # Partial Hessian: keep only the unfrozen x unfrozen sub-block (frozen
    # columns were never filled; the frozen rows are dropped here too).
    if is_partial:
        H = H[np.ix_(unfrozen_dof, unfrozen_dof)]
        if include_dipole:
            dipole_derivs = dipole_derivs[unfrozen_dof, :]

    if hessian_options.sym_strict:
        H = 0.5 * (H + H.T)

    # ---- Mass weight ------------------------------------------------------
    # For a partial Hessian, mass-weight with the unfrozen masses only so
    # the (3M, 3M) block matches its 3M rows/columns.
    masses_amu_all = _resolve_masses(mol, hessian_options.atomic_masses_amu)
    masses_amu = masses_amu_all[unfrozen] if is_partial else masses_amu_all
    masses_e = masses_amu * _AMU_TO_ELECTRON_MASS
    inv_sqrt_m_per_dof = np.repeat(1.0 / np.sqrt(masses_e), 3)
    M = inv_sqrt_m_per_dof  # diagonal of M^(-1/2)
    H_mw = (M[:, None] * H) * M[None, :]
    H_mw = 0.5 * (H_mw + H_mw.T)  # squash residual asymmetry from FD noise

    # ---- Project out trans/rot, diagonalize -------------------------------
    # A partial Hessian is anchored by its frozen atoms -- it has no
    # translational/rotational zero modes, so projection is skipped and all
    # 3M modes are vibrational.
    is_linear = False
    if hessian_options.project_trans_rot and not is_partial:
        positions = _atom_positions_bohr(mol)
        zero_modes, is_linear = _build_trans_rot_modes(positions, masses_e)
        H_proj = _project_out(H_mw, zero_modes)
        omega2, modes = np.linalg.eigh(0.5 * (H_proj + H_proj.T))
        n_zero = zero_modes.shape[1]
        omega2, modes = _enforce_projected_zero(omega2, modes, n_zero)
    else:
        omega2, modes = np.linalg.eigh(H_mw)

    # eigh returns ascending eigenvalues already, but the
    # "force smallest |value| to zero" step may have reordered them
    # via index reuse. Re-sort to keep the contract clean.
    order = np.argsort(omega2)
    omega2 = omega2[order]
    modes = modes[:, order]

    freqs = _omega2_to_cm_inv(omega2)
    imag_count = int(np.sum(omega2 < -1e-10))

    return HessianResult(
        hessian=H,
        hessian_mw=H_mw,
        frequencies_cm1=freqs,
        normal_modes=modes,
        imaginary_count=imag_count,
        n_displacements=n_disp,
        is_linear=is_linear,
        dipole_derivatives=dipole_derivs,
        dipole_origin=(np.asarray(dipole_origin)
                       if dipole_origin is not None else None),
        masses_amu=masses_amu,
        unfrozen_indices=(np.asarray(unfrozen, dtype=int)
                          if is_partial else None),
    )


def ir_intensities(result: HessianResult) -> np.ndarray:
    """Convert a Hessian result's dipole derivatives + normal modes
    into IR band intensities in km/mol.

    Phase 17a-2.

    The harmonic-oscillator IR intensity for the *p*-th normal mode
    is the Wilson-Decius-Cross formula

    .. math::
        I_p = \\frac{N_A \\pi}{3 c^2} \\left| \\frac{\\partial \\mu}{\\partial Q_p} \\right|^2,

    where :math:`Q_p` is the mass-weighted normal coordinate of mode
    *p* with eigenvector :math:`\\mathbf{L}_p` of the mass-weighted
    Hessian. Going from Cartesian to normal-mode dipole derivatives
    uses the chain rule

    .. math::
        \\frac{\\partial \\mu_\\beta}{\\partial Q_p} = \\sum_k
            \\frac{\\partial \\mu_\\beta}{\\partial R_k}
            \\cdot \\frac{L_{kp}}{\\sqrt{M_k}},

    with :math:`M_k` the mass (in electron masses) of the atom
    corresponding to flat index *k*. The conversion from atomic units
    of :math:`|\\partial \\mu / \\partial Q|^2` to km/mol uses the
    standard Gaussian / NWChem / ORCA / PySCF factor of 974.864
    (CODATA-derived).

    Parameters
    ----------
    result :
        :class:`HessianResult` produced with
        ``HessianFDOptions(include_dipole_derivatives=True)``.

    Returns
    -------
    np.ndarray
        ``(3N,)`` array of IR intensities in km/mol, indexed by the
        same ordering as ``result.frequencies_cm1`` (ascending
        frequency). Trans/rot zero modes have intensities ≈ 0 by
        construction (the corresponding eigenvector has no overlap
        with the dipole-gradient field for translation-invariant
        observables).

    Raises
    ------
    ValueError
        If ``result.dipole_derivatives`` is None (the Hessian was
        built without ``include_dipole_derivatives=True``).
    """
    if result.dipole_derivatives is None:
        raise ValueError(
            "ir_intensities: dipole derivatives missing on the "
            "HessianResult. Re-run compute_hessian_fd with "
            "HessianFDOptions(include_dipole_derivatives=True)."
        )
    if result.masses_amu is None:
        raise ValueError(
            "ir_intensities: masses_amu missing on the HessianResult "
            "(should be populated by compute_hessian_fd)."
        )

    B = np.asarray(result.dipole_derivatives, dtype=np.float64)  # (3N, 3)
    L = np.asarray(result.normal_modes, dtype=np.float64)         # (3N, 3N)
    masses_e = np.asarray(result.masses_amu, dtype=np.float64) * _AMU_TO_ELECTRON_MASS
    inv_sqrt_m_per_dof = np.repeat(1.0 / np.sqrt(masses_e), 3)     # (3N,)

    # Mass-weighted Cartesian dipole derivative tensor: B_mw[k, b] =
    # (dmu_b/dR_k) / sqrt(M_k). Then transform to normal-mode basis:
    # dmu/dQ[b, p] = S_k B_mw[k, b] . L[k, p].
    B_mw = B * inv_sqrt_m_per_dof[:, None]
    dmu_dQ = B_mw.T @ L                                            # (3, 3N)

    intensities = _AU_TO_KM_PER_MOL * np.sum(dmu_dQ ** 2, axis=0)   # (3N,)

    # Mask out zero-frequency (trans / rot) modes. These modes don't
    # oscillate, so the harmonic IR intensity formula doesn't apply to
    # them -- they're not "vibrations". Numerically, the eigh
    # decomposition of the trans/rot-projected Hessian returns an
    # arbitrary orthonormal basis of the kernel; that basis can mix
    # translations (zero dmu/dQ by sum rule) with rotations (nonzero
    # dmu/dQ because rotating a polar molecule changes the dipole
    # direction), giving a spurious "intensity" for each kernel
    # eigenvector. Standard convention (Gaussian, NWChem, ORCA): drop
    # them. Threshold at 1 cm⁻¹ -- well below any real vibration but
    # well above floating-point noise on the projected eigenvalues.
    freqs = np.asarray(result.frequencies_cm1, dtype=np.float64)
    intensities = np.where(np.abs(freqs) < 1.0, 0.0, intensities)
    return intensities


__all__ = [
    "HessianFDOptions",
    "HessianResult",
    "compute_hessian_fd",
    "ir_intensities",
]
