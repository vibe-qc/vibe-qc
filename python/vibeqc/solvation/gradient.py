"""CPCM solvation gradient (S1c) -- analytic at SCF+CPCM convergence.

Total in-solvent energy (Cossi-Scalmani 2003 convention):

    E_tot = E_HF^{gas}[D^{solv}] + E_pol
    E_pol = (1/2) q^T V_tot
    A q   = - f(e) V_tot

with the screening factor set by the solvent model's variant --
``f = (e - 1)/e`` for CPCM (Cossi 2003) and ``f = (e - 1)/(e + 1/2)``
for COSMO (Klamt-Schuurmann 1993). ``q`` is stored scaled, ``q = f q0``
with ``q0 = -A^-1 V_tot``, so every ``f`` below is the one that solve
used -- see :func:`_screening_factor`.

Apply the envelope theorem at full SCF + CPCM convergence (both ``D``
and ``q`` are stationary):

    dE_tot/dR = dE_HF^{gas}/dR + dE_pol/dR
              = dE_HF^{gas}[D^{solv}]/dR
                + (1/(2f)) q^T (dA/dR) q
                + q^T (dV/dR)|_D

where ``f`` is the *same* factor that built ``q``. It is not a free
choice: Klamt-Schuurmann 1993 p. 801 introduces ``f(e)`` into the
screening energy and its gradient together (#546).

Each piece is evaluated as follows:

* **dE_HF^{gas}[D^{solv}]/dR** -- the standard analytic SCF gradient
  routine (``compute_gradient`` / ``compute_gradient_rks`` /
  ``compute_gradient_uhf`` / ``compute_gradient_uks``) applied to the
  in-solvent SCF result. The energy-weighted density built from the
  *in-solvent* orbital energies is exactly the right one for the
  total in-solvent functional: the orthonormality-constraint (Pulay)
  term takes the eigenvalues of the operator the density actually
  diagonalises -- the solvated Fock matrix ``F_gas + V_q`` -- not the
  gas-phase ``F``. (At full SCF + CPCM convergence the assembled
  analytic gradient agrees with full-energy finite differences to
  FD-truncation level, ~6e-9 Ha/bohr at h = 2e-4 on water/6-31G.)

* **(1/(2f)) q^T (dA/dR) q** -- closed-form. The Scalmani-Frisch
  off-diagonal ``1/|s_i - s_j|`` and the diagonal ``a/√(w_i)`` both
  have analytic derivatives wrt nuclear positions through the
  cavity-point and switching-function dependencies.

* **q^T (dV^{nuc}/dR)** -- closed-form. ``V^{nuc}_i = S_A Z_A/|R_A - s_i|``
  with both ``R_A`` and (via parent-atom motion) ``s_i`` differentiable.

* **q^T (dV^{elec}/dR)** -- closed-form. Built by one libint
  nuclear-attraction *gradient* pass over the cavity points, via the
  C++ entry point ``compute_external_charge_density_gradient``. The
  engine emits the bra/ket basis-center derivatives (the AO-position
  motion) and the per-point-charge derivatives (the cavity-point
  motion) together; :func:`_electronic_esp_gradient_analytic` routes
  each point-charge derivative onto its parent atom. No geometry
  rebuilds, no finite-difference step error.

A finite-difference fallback for the last piece --
:func:`_electronic_esp_gradient_via_fd`, which rebuilds the basis +
cavity + operator stack at ``6 . n_atoms`` displaced geometries -- is
retained and reachable via ``cpcm_gradient(..., use_fd_electronic=
True)`` for cross-checking the closed-form kernel. The standalone
full-SCF FD reference :func:`cpcm_gradient_fd` is the outermost
oracle.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .cavity import (
    atom_radii_bohr as cavity_atom_radii_bohr,
    CavityTessellation,
)
from .cavity_derivative import cavity_A_adjoints, cavity_derivative
from .screening import ScreeningModel
from .driver import (
    SolventResult,
    _build_point_charge_operators,
    _density_of,
)

# =====================================================================
# Component 1: gas-phase analytic gradient at in-solvent density
# =====================================================================


def _gas_phase_gradient(
    method: str,
    molecule,
    basis,
    scf_result,
    options: Any = None,
    grid_options: Any = None,
) -> np.ndarray:
    """Dispatcher for the standard SCF analytic gradient.

    Routes to the appropriate ``vibeqc.compute_gradient*`` based on
    method. Returns an ``(n_atoms, 3)`` Hartree/bohr matrix.
    """
    from vibeqc import (
        compute_gradient,
        compute_gradient_rks,
        compute_gradient_uhf,
        compute_gradient_uks,
    )

    m = method.lower()
    if m == "rhf":
        return np.asarray(compute_gradient(molecule, basis, scf_result, options))
    if m == "uhf":
        return np.asarray(compute_gradient_uhf(molecule, basis, scf_result, options))
    if m == "rks":
        return np.asarray(
            compute_gradient_rks(
                molecule,
                basis,
                scf_result,
                grid_options,
                options,
            )
        )
    if m == "uks":
        return np.asarray(
            compute_gradient_uks(
                molecule,
                basis,
                scf_result,
                grid_options,
                options,
            )
        )
    raise ValueError(
        f"cpcm_gradient: unsupported method {method!r} "
        f"(expected one of rhf, uhf, rks, uks)."
    )


# =====================================================================
# Component 2: closed-form A-matrix derivative contribution
#   contrib = (1/(2f)) q^T (dA/dR_A) q
# =====================================================================


def _atom_radii_bohr(
    atom_numbers: np.ndarray,
    radii_override: Optional[dict[int, float]],
    radii_scale: float,
    solvent_probe_radius_ang: float,
) -> np.ndarray:
    """The scaled cavity radii, from the one shared derivation.

    Delegates to :func:`vibeqc.solvation.cavity.atom_radii_bohr` so the
    gradient cannot see radii that differ from the ones the energy's cavity was
    built with. It used to re-implement the formula, which is the same
    two-copies-of-one-quantity hazard as #546 and #548 -- and a gradient built
    on different radii is wrong in exactly that silent way.
    """
    return cavity_atom_radii_bohr(
        atom_numbers, radii_override, radii_scale, solvent_probe_radius_ang
    )


def _cavity_derivative(cavity, deriv=None):
    """The cavity's own chain rule, built once and shared by all three terms.

    Reading ``switching_sigma_bohr`` off the cavity rather than defaulting it
    here is not defensive tidiness. ``SolventModel.switching_sigma_bohr`` is
    user-settable and reaches ``build_cavity``, so a hard-coded 0.5 in the
    gradient differentiates a cavity the run never built whenever the user
    moves it -- the same two-copies-of-one-quantity defect as #546, and just as
    silent: measured against a converged finite difference, 12 percent of the
    largest gradient component at ``sigma = 0.8``.

    Until #745 the read was a ``getattr`` with a 0.5 fallback, against a
    tessellation that never carried the field -- so it always took the
    fallback, and the fix looked like it was already in. A cavity that switches
    and cannot say how widely is now refused instead.
    """
    if deriv is not None:
        return deriv
    from .fine_cavity import FineCavity

    if isinstance(cavity, FineCavity):
        # The CFC has no switching function -- its ``switching`` is identically
        # one -- so there is no width to carry and none to pass on.
        return cavity_derivative(cavity)
    try:
        sigma = cavity.switching_sigma_bohr
    except AttributeError as exc:
        raise ValueError(
            "cpcm_gradient: this cavity carries a switching function but does "
            "not record the width it was built with, so the gradient cannot "
            "know which cavity to differentiate. Build it with "
            "vibeqc.solvation.cavity.build_cavity, which records it. Assuming "
            "a default here is what #745 was, and it is not detectable from "
            "inside: the resulting gradient is self-consistent, so it passes "
            "translational invariance and every other exact check."
        ) from exc
    return cavity_derivative(cavity, switching_sigma_bohr=float(sigma))


def _A_matrix_gradient_contribution(
    cavity,
    q: np.ndarray,
    f_dielectric: float,
    n_atoms: int,
    switching_sigma_bohr: float = 0.5,
    deriv=None,
) -> np.ndarray:
    """Per-atom Cartesian contribution from ``(1/(2f)) q^T (dA/dR) q``.

    Two halves, and the split is what makes this construction-agnostic:

    * :func:`~vibeqc.solvation.cavity_derivative.cavity_A_adjoints` gives
      ``dG/dp_S`` and ``dG/dw_S`` -- the derivative of the *term*, identical
      for every cavity, since ``A`` sees a cavity only as positions and areas.
    * the cavity's own :class:`CavityDerivative` turns those into
      ``d/dR_A`` -- the derivative of the *construction*, which is a scatter
      for a Lebedev cavity and a full chain through the pseudo-density
      iso-surface for a FINE one.

    Before this split the two were fused into one routine that assumed rigid
    parent-atom motion plus an erf switch, so a FINE cavity had to be refused
    outright rather than merely lacking a branch.

    Returns ``(n_atoms, 3)`` Hartree/bohr.
    """
    d = _cavity_derivative(cavity, deriv)
    # The representation is read off the cavity, never chosen here: the
    # adjoints must differentiate the kernel the energy actually used
    # (#744). Passing the wrong one is silent and costs a factor of ~4 in the
    # analytic-vs-FD residual, which still looks like a plausible gradient.
    from .cpcm import GAUSSIAN_CHARGE, POINT_CHARGE

    representation = getattr(cavity, "charge_representation", POINT_CHARGE)
    switching = getattr(cavity, "switching", None)
    adj_position, adj_area = cavity_A_adjoints(
        cavity.points,
        cavity.weights,
        q,
        q,
        1.0 / (2.0 * f_dielectric),
        representation=representation,
        switching=switching if representation == GAUSSIAN_CHARGE else None,
        n_points_per_sphere=getattr(cavity, "n_points_per_sphere", None),
    )
    return d.contract(adj_position, adj_area)


# =====================================================================
# Component 3: closed-form nuclear-ESP derivative
#   contrib = q^T (dV^{nuc}/dR)
# =====================================================================


def _nuclear_esp_gradient_contribution(
    cavity,
    q: np.ndarray,
    R_atoms: np.ndarray,
    Z_atoms: np.ndarray,
    deriv=None,
) -> np.ndarray:
    """``q^T (dV^{nuc}/dR_A)`` per atom -- closed form.

    ``V^{nuc}_i = S_A Z_A / |R_A - s_i|``. Two contributions per atom:

    * Direct: ``d/dR_A V^{nuc}_i = -Z_A (R_A - s_i)/|R_A - s_i|^3``
      whenever the differentiated atom is in the sum.
    * Indirect (segment motion): ``s_i`` moves when the atoms move,
      contributing the adjoint ``q_i S_B Z_B (R_B - s_i)/|R_B - s_i|^3``
      contracted through the cavity's own chain rule. For a Lebedev cavity
      that contraction *is* the old scatter onto the parent atom; for a FINE
      cavity it is not, because a segment has no single parent it rides on.

    Returns ``(n_atoms, 3)`` Hartree/bohr matrix.
    """
    pts = cavity.points  # (n_pts, 3)
    n_atoms = R_atoms.shape[0]

    # Pairwise (R_A - s_i) tensor and norms.
    diff = R_atoms[:, None, :] - pts[None, :, :]  # (n_atoms, n_pts, 3)
    r2 = np.einsum("ijk,ijk->ij", diff, diff)
    r2[r2 == 0] = 1.0
    inv_r3 = 1.0 / (r2 * np.sqrt(r2))

    # dV^{nuc}_i / dR_A = -Z_A . diff[A, i, :] . inv_r3[A, i]
    # (sign because |R_A - s_i| decreases when R_A moves toward s_i).
    dV_dRA = (-Z_atoms[:, None, None]) * diff * inv_r3[:, :, None]
    # Direct contribution: q_i . dV^{nuc}_i/dR_A summed over i.
    grad = np.einsum("i,aik->ak", q, dV_dRA)  # (n_atoms, 3)

    # Segment-motion channel: dV^nuc_i/ds_i = S_B Z_B (R_B - s_i)/|R_B - s_i|^3.
    # That is an adjoint on the segment *position*; the cavity supplies
    # ds_i/dR_A.
    adj_position = q[:, None] * np.einsum(
        "aik,ai->ik", diff, (Z_atoms[:, None] * inv_r3)
    )  # (n_pts, 3)
    grad += _cavity_derivative(cavity, deriv).contract(adj_position)
    return grad


# =====================================================================
# Component 4: q^T dV^{elec}/dR via fast FD on the point-charge operator
# =====================================================================


def _electronic_esp_gradient_via_fd(
    cavity: CavityTessellation,
    q: np.ndarray,
    density: np.ndarray,
    molecule,
    basis_name: str,
    *,
    step_bohr: float = 1e-3,
) -> np.ndarray:
    """``q^T (dV^{elec}/dR_A)`` per atom via finite difference on the
    ``M_stack = <mu|1/|r - s_i||ν>`` operator stack.

    For each atom A and Cartesian direction d, build a perturbed
    molecule with atom A shifted by ±h, regenerate the cavity
    (cavity points move with their parent atoms -- captures both the
    AO-position derivative AND the cavity-point-motion derivative in
    one shot), recompute the M_stack at the perturbed geometry, and
    contract.

    Cost is ``6.n_atoms.n_pts.n_bf^2`` flops + ``6.n_atoms.n_pts``
    libint nuclear-attraction calls -- typically two orders of
    magnitude cheaper than full-SCF FD on the total energy.

    The full analytic alternative needs the AO-derivative of
    ``<mu|1/|r - s|||ν>`` from libint's gradient engine with
    fractional point charges -- not yet wired in vibe-qc's C++
    surface. Ships in v0.9.1 once the C++ entry point lands.
    """
    from vibeqc import Atom, BasisSet, Molecule

    from .cavity import build_cavity
    from .driver import _density_potential_at_cavity

    atoms = list(molecule.atoms)
    n_atoms = len(atoms)
    R = np.array([list(a.xyz) for a in atoms], dtype=np.float64)
    Zs = np.array([int(a.Z) for a in atoms], dtype=int)
    charge = molecule.charge
    mult = molecule.multiplicity

    # Cavity recipe parameters -- must match how the SCF cavity was
    # built. We don't have direct access to the SolventModel here, so
    # re-derive from the CavityTessellation (atom_radii encodes the
    # scaled Bondi radius; we read n_points_per_sphere from the
    # cavity dataclass).
    n_pts_per_sphere = cavity.n_points_per_sphere

    def _build_M_stack_at(positions: np.ndarray) -> np.ndarray:
        """Rebuild the ``<mu|1/|r - s_i||ν>`` stack at a displaced geometry.

        Both the basis (atoms shifted) AND the cavity (parent-atomic
        spheres shifted with the atoms) move. This captures the full
        ``dM_i/dR_A`` -- AO-position-motion *and* cavity-point-motion
        in one FD step. The reference ``q`` is held fixed (envelope
        theorem at SCF + CPCM convergence) and contracted against the
        displaced operator stack.

        Requires the displaced cavity to keep the same point count as
        the reference cavity (true for any sub-bohr displacement of
        well-conditioned cavities; an assertion below guards against
        the rare switching-window crossing).
        """
        new_atoms = [Atom(int(Zs[i]), tuple(positions[i])) for i in range(n_atoms)]
        new_mol = Molecule(new_atoms, charge, mult)
        new_basis = BasisSet(new_mol, basis_name)
        new_cav = build_cavity(
            atom_positions_bohr=positions,
            atom_numbers=Zs.tolist(),
            n_points_per_sphere=n_pts_per_sphere,
            switching_sigma_bohr=cavity.switching_sigma_bohr,
        )
        # The rest of the recipe cannot be read back off a tessellation --
        # ``radii``, ``radii_scale`` and the probe offset survive only as the
        # scaled radii they produced -- so check the rebuild rather than
        # assuming it. Silently rebuilding with different spheres is #745 one
        # level down, and this path is where it would land next.
        if not np.allclose(new_cav.atom_radii, cavity.atom_radii, rtol=0, atol=1e-12):
            raise ValueError(
                "_electronic_esp_gradient_via_fd: the reference cavity was "
                "built with radii this function cannot reconstruct from the "
                "tessellation (a radii override, a non-default radii_scale, or "
                "a solvent probe offset). Displacing it would differentiate a "
                "different cavity. Pass the SolventModel through rather than "
                "re-deriving the recipe here."
            )
        # Switching-window guard: tiny displacements (1e-3 bohr) keep
        # every point in/out of the cavity on the same side of the
        # Scalmani-Frisch erf cutoff, so the point count is invariant.
        # A change in n_points here would mean the cavity reconnected
        # -- which would make the q-vector indices meaningless.
        if new_cav.n_points != cavity.n_points:
            raise RuntimeError(
                f"CPCM gradient FD step crossed a switching-window "
                f"boundary: reference cavity has {cavity.n_points} "
                f"points; displaced cavity has {new_cav.n_points}. "
                f"Reduce ``fd_step_bohr`` below {step_bohr} or widen "
                f"``switching_sigma_bohr`` to keep the boundary smooth."
            )
        return _build_point_charge_operators_at_fixed_points(
            new_basis,
            new_cav.points,
        )

    h = float(step_bohr)
    grad = np.zeros((n_atoms, 3), dtype=np.float64)
    for ia in range(n_atoms):
        for ic in range(3):
            R_plus = R.copy()
            R_minus = R.copy()
            R_plus[ia, ic] += h
            R_minus[ia, ic] -= h
            M_plus = _build_M_stack_at(R_plus)
            M_minus = _build_M_stack_at(R_minus)
            # q . V^{elec}(R) = -S_i q_i tr(D . M_i)
            # -> d/dR (q . V^{elec}) = -S_i q_i tr(D . dM_i/dR).
            # D is held fixed via the envelope theorem at SCF
            # convergence; q is held fixed via the CPCM convergence.
            dM = (M_plus - M_minus) / (2.0 * h)
            grad[ia, ic] = -np.einsum("i,ab,iab->", q, density, dM)
    return grad


def _build_point_charge_operators_at_fixed_points(
    basis,
    cavity_points: np.ndarray,
) -> np.ndarray:
    """Mirror of :func:`driver._build_point_charge_operators` but
    parameterised so the caller can supply a basis built from a
    perturbed molecule while keeping the cavity points at their
    reference positions.

    Used inside the FD electronic-ESP gradient where we want to
    isolate the AO-position-motion derivative of ``M_i,muν`` (the
    cavity-point-motion piece is folded in separately when atom A
    is the parent of point i -- see comment in
    :func:`_electronic_esp_gradient_via_fd`).
    """
    from vibeqc import Atom, Molecule, compute_nuclear

    n_pts = int(cavity_points.shape[0])
    n_bf = int(basis.nbasis)
    stack = np.empty((n_pts, n_bf, n_bf), dtype=np.float64)
    for i in range(n_pts):
        fake = Molecule([Atom(1, tuple(cavity_points[i]))], 1, 1)
        M_i = compute_nuclear(basis, fake)
        stack[i] = -np.asarray(M_i)
    return stack


# =====================================================================
# Top-level entry point
# =====================================================================


def _electronic_esp_gradient_analytic(
    cavity,
    q: np.ndarray,
    density: np.ndarray,
    molecule,
    basis,
    deriv=None,
) -> np.ndarray:
    """``q . dV^elec/dR`` per atom -- closed form via libint (v0.9.1).

    Calls the C++ ``compute_external_charge_density_gradient`` once: it
    drives libint's nuclear-attraction *gradient* engine with the
    cavity points as fractional external charges and returns

      * ``atom_grad``  (n_atoms, 3) -- the AO-basis-center derivatives,
      * ``point_grad`` (n_pts, 3)   -- the per-cavity-point derivatives,

    of ``E_ext = S_i q_i Tr(D.M_i)`` with ``M_i,muν = <mu|1/|r-s_i||ν>``
    the positive Coulomb kernel. ``point_grad`` is an adjoint on the segment
    *positions*, so the cavity's own chain rule converts it:

      dE_ext/dR_A = atom_grad[A] + sum_i point_grad[i] . ds_i/dR_A.

    Scattering ``point_grad`` onto parent atoms, as this did before, is that
    contraction specialized to ``ds_i/dR_A = delta_{A,a(i)} I``. True for a
    Lebedev cavity, false for every other construction.

    CPCM needs ``q . dV^elec/dR`` with ``V^elec_i = -Tr(D.M_i)``, i.e.
    ``S_i q_i V^elec_i = -E_ext`` => the contribution is ``-dE_ext/dR``.

    One libint pass -- no geometry rebuilds, no FD step error. The exact
    closed-form replacement for :func:`_electronic_esp_gradient_via_fd`.
    """
    from vibeqc import compute_external_charge_density_gradient

    ext = compute_external_charge_density_gradient(
        basis,
        molecule,
        np.ascontiguousarray(density, dtype=np.float64),
        [float(x) for x in np.asarray(q).ravel()],
        [tuple(float(c) for c in p) for p in cavity.points],
    )
    grad_ext = np.array(ext.atom_grad, dtype=np.float64)  # (n_atoms, 3)
    point_grad = np.asarray(ext.point_grad, dtype=np.float64)  # (n_pts, 3)
    grad_ext += _cavity_derivative(cavity, deriv).contract(point_grad)
    # q . dV^elec/dR = -dE_ext/dR (V^elec carries the electron sign).
    return -grad_ext


def _screening_model(solvent_result: SolventResult) -> Optional[ScreeningModel]:
    """The screening model that actually built ``q``, or ``None`` in gas phase.

    Read off the result, never re-derived from a fixed variant. ``q`` is
    stored in the scaled convention ``q = f q0`` with ``q0 = -A^-1 V``
    (:func:`vibeqc.solvation.driver.run_cpcm_scf` solves with
    ``variant=SolventModel.variant``), so the ``f`` in the cavity term
    ``(1/(2f)) q^T (dA) q`` is fixed by that solve -- it is not a free
    choice. Klamt & Schuurmann, *J. Chem. Soc. Perkin Trans. 2* 799
    (1993), doi:10.1039/P29930000799, p. 801 introduces the correction
    factor ``f(e) = (e - 1)/(e + 1/2)`` "into the expressions for the
    screening energy *and its gradient*" -- one factor in both. Their
    eq. (13) gives the unscaled gradient as
    ``-q*^T (grad B) Q + (1/2) q*^T (grad A) q*`` with
    ``q* = -A^-1 B Q`` (eq. 10); substituting ``q = f q*`` reproduces the
    scaled form vibe-qc assembles below.

    Hard-coding ``variant="cpcm"`` here made a ``variant="cosmo"`` run
    differentiate a *different* energy than it reported: smooth, plausible
    and wrong, by 18% of the cavity term at benzene's dielectric. The
    analytic-vs-FD residual then plateaus under step refinement instead of
    converging (#546; pinned by
    ``tests/test_solvation_variant_gradient.py``).

    Older results predate the carried model; fall back to rebuilding it from
    the variant they do carry, which is still the run's own variant and never
    a hard-coded one.
    """
    screening = getattr(solvent_result, "screening", None)
    if screening is not None:
        return screening
    if solvent_result.epsilon <= 1.0:
        return None
    return ScreeningModel.from_variant(
        solvent_result.epsilon, solvent_result.solvent_variant
    )


def _screening_factor(solvent_result: SolventResult) -> float:
    """``f`` for a solvated result. Raises in gas phase, where there is none."""
    screening = _screening_model(solvent_result)
    if screening is None:
        raise ValueError(
            "solvation gradient: this result carries no reaction field "
            "(gas phase); there is no screening factor to read."
        )
    return screening.f


def cpcm_gradient(
    scf_result,
    molecule,
    basis,
    solvent_result: SolventResult,
    *,
    method: str = "rhf",
    options: Any = None,
    grid_options: Any = None,
    use_fd_electronic: bool = False,
    fd_step_bohr: float = 1e-3,
) -> np.ndarray:
    """Analytic CPCM solvation gradient -- total in-solvent energy.

    Returns ``(n_atoms, 3)`` ``dE_tot/dR`` in Hartree/bohr, where
    ``E_tot = E_HF^{gas}[D^{solv}] + (1/2) q^T V_tot`` (the standard
    Cossi-Scalmani CPCM convention).

    Parameters
    ----------
    scf_result
        Converged in-solvent SCF result (the ``.scf`` field of
        :class:`SolventResult`).
    molecule, basis
        The :class:`vibeqc.Molecule` + :class:`vibeqc.BasisSet` the
        SCF was run on. Used for the standard analytic gradient call.
    solvent_result
        :class:`SolventResult` carrying the converged cavity,
        apparent charges ``q``, and total ESP ``V`` at cavity points.
    method
        ``"rhf"``, ``"uhf"``, ``"rks"``, or ``"uks"`` -- must match
        the SCF method that produced ``scf_result``.
    options
        Method-specific options struct forwarded to the underlying
        analytic gradient call (typically the same one used for the
        SCF run).
    grid_options
        :class:`GridOptions` for KS methods (only used for ``rks`` /
        ``uks``).
    use_fd_electronic
        When ``False`` (default) the electronic-ESP-derivative piece
        ``q.dV^elec/dR`` is computed in closed form via the C++
        ``compute_external_charge_density_gradient`` (one libint
        nuclear-attraction-gradient pass). When ``True`` it falls back
        to the finite-difference path
        :func:`_electronic_esp_gradient_via_fd` -- kept reachable as a
        cross-check oracle.
    fd_step_bohr
        Central-difference step for the FD electronic-ESP path. Only
        used when ``use_fd_electronic=True``. Default 1e-3 bohr.

    Notes
    -----
    Cost (analytic path, the default): one full gas-phase analytic-
    gradient call + a single libint nuclear-attraction-gradient pass
    over the cavity points. No geometry rebuilds, no FD step error.
    The ``dA/dR`` and ``dV^nuc/dR`` pieces are closed-form. The
    optional ``use_fd_electronic=True`` path costs ``6.n_atoms``
    cavity + basis rebuilds instead -- slower, kept only for
    verification.

    Raises
    ------
    NotImplementedError
        For a Direct COSMO-RS result (``variant="dcosmo-rs"``), whose energy is
        not differentiable in the geometry; see :func:`_refuse_direct_cosmors`.
    """
    from vibeqc.ecp_metadata import refuse_molecular_ecp_derivative_route

    # First, before any work: every term below is the conductor gradient's.
    _refuse_direct_cosmors(solvent_result)
    if method.lower() in ("rks", "uks"):
        # The gas-phase piece below is the ordinary analytic RKS/UKS gradient,
        # so a functional whose analytic gradient omits terms (#571) would get
        # a reaction field added to a wrong surface. Refuse rather than return
        # it; the optimizers route such functionals to full-energy finite
        # differences, which differentiate the solvated energy.
        from vibeqc.gradient_terms import require_complete_analytic_gradient

        require_complete_analytic_gradient(
            scf_result,
            route="cpcm_gradient",
            spin=1 if method.lower() == "rks" else 2,
        )
    refuse_molecular_ecp_derivative_route(
        molecule,
        basis,
        options=options,
        result=scf_result,
        route="cpcm_gradient",
    )
    if solvent_result.z_eff is not None:
        bare_z = np.asarray(
            [float(atom.Z) for atom in molecule.atoms], dtype=np.float64
        )
        if not np.allclose(
            np.asarray(solvent_result.z_eff, dtype=np.float64),
            bare_z,
            atol=1e-12,
            rtol=0.0,
        ):
            raise NotImplementedError(
                "cpcm_gradient: ECP nuclear derivatives are not implemented; "
                "refusing to combine an ECP-aware CPCM ESP with a bare-Z "
                "gas-phase gradient"
            )
    cav = solvent_result.cavity
    cpcm = solvent_result.cpcm
    q = np.asarray(cpcm.q, dtype=np.float64)

    screening = _screening_model(solvent_result)
    if screening is None:
        # Gas phase. ``_gas_phase_solvent_result`` hands downstream code a
        # uniform return type with ``q`` identically zero, so the reaction
        # field contributes exactly nothing and the answer is the gas-phase
        # gradient itself. Rejecting that shape with "epsilon must be > 1"
        # defeated the point of having one type (#549).
        return _gas_phase_gradient(
            method,
            molecule,
            basis,
            scf_result,
            options=options,
            grid_options=grid_options,
        )
    f = screening.f

    atoms = list(molecule.atoms)
    R = np.array([list(a.xyz) for a in atoms], dtype=np.float64)
    Zs = np.array([int(a.Z) for a in atoms], dtype=int)
    # Use effective Z for the nuclear-ESP gradient when ECPs are in play
    # (same correction as the CPCM driver's V_nuc_cav).
    if solvent_result.z_eff is not None:
        Zs_grad = np.asarray(solvent_result.z_eff, dtype=np.float64)
    else:
        Zs_grad = Zs.astype(float)
    n_atoms = len(atoms)

    # Component 1: gas-phase analytic gradient at D^{solv}.
    grad = _gas_phase_gradient(
        method, molecule, basis, scf_result, options=options, grid_options=grid_options
    )

    # The cavity's chain rule, built once. All three cavity terms contract
    # against the same object, which for a FINE cavity means the iso-surface
    # Jacobians are assembled once instead of three times.
    deriv = _cavity_derivative(cav)

    # Component 2: (1/(2f)) q^T (dA/dR) q -- closed-form.
    grad += _A_matrix_gradient_contribution(cav, q, f, n_atoms, deriv=deriv)

    # Component 3: q^T (dV^{nuc}/dR) -- closed-form.
    grad += _nuclear_esp_gradient_contribution(cav, q, R, Zs_grad, deriv=deriv)

    # Component 4: q^T (dV^{elec}/dR).
    open_shell = method.lower() in ("uhf", "uks")
    density = _density_of(scf_result, open_shell=open_shell)
    if use_fd_electronic:
        # FD cross-check path -- rebuilds basis + cavity at ±h.
        basis_name = _resolve_basis_name(basis)
        grad += _electronic_esp_gradient_via_fd(
            cav,
            q,
            density,
            molecule,
            basis_name,
            step_bohr=fd_step_bohr,
        )
    else:
        # Closed-form path (default) -- one libint gradient pass.
        grad += _electronic_esp_gradient_analytic(
            cav,
            q,
            density,
            molecule,
            basis,
            deriv=deriv,
        )

    return grad


def _refuse_direct_cosmors(solvent_result) -> None:
    """Refuse a Direct COSMO-RS result: there is no gradient to return for it.

    Nothing in :func:`cpcm_gradient` knows the variant. Handed a
    ``variant="dcosmo-rs"`` result it assembled the conductor-COSMO gradient --
    no ``sum_t a_t mu_S(sigma_t)`` energy term and no ``q^dRS`` operator -- and
    returned it. Nothing about that number looks wrong: on asymmetric water
    (RHF/STO-3G, fine cavity at 0.40 A) its translation residual was 2.1e-15,
    yet it sat 1.4e-03 Ha/bohr off a central difference of the energy the run
    reported, with one component of the wrong sign, where the conductor
    gradient at the same settings agrees with its own to 1.0e-06.

    Adding the two missing terms would not make a gradient. Klamt's
    hydrogen-bond term makes ``mu_S'`` jump by about 40 against values near 70
    at ``sigma = +-sigma_hb``; every solvent partner steps at the same sigma, so
    the Boltzmann average does not damp it, and about a quarter of a real
    solute's segments sit within a bin of the corner. The energy is not
    differentiable wherever a segment crosses it, and a smooth one means
    changing a published functional form -- a maintainer decision.

    Both markers are read, so a result that carries a feedback is refused even
    if its variant label is lost.
    """
    variant = str(getattr(solvent_result, "solvent_variant", "") or "")
    if (
        variant.strip().lower() != "dcosmo-rs"
        and getattr(solvent_result, "direct_feedback", None) is None
    ):
        return
    raise NotImplementedError(
        "cpcm_gradient: there is no nuclear gradient for a Direct COSMO-RS "
        "result (variant='dcosmo-rs'). Klamt's hydrogen-bond term makes the "
        "sigma potential's derivative mu_S' jump at sigma = +-sigma_hb, and "
        "segments cross that corner as the atoms move, so the Direct COSMO-RS "
        "energy is not differentiable in the geometry. A gradient needs a "
        "smoothed hydrogen-bond term, which departs from the published "
        "functional form and is a maintainer decision. Returning the "
        "conductor-COSMO gradient instead would silently omit the "
        "sum_t a_t mu_S(sigma_t) energy term and the q^dRS operator. For an "
        "analytic gradient use variant='cosmo', which is a different energy; "
        "cpcm_gradient_fd rebuilds the Direct COSMO-RS energy, but it steps "
        "wherever a segment crosses the corner."
    )


def _resolve_basis_name(basis) -> str:
    """Best-effort extraction of the libint basis-name string.

    pybind11's BasisSet doesn't always expose ``.name`` cleanly; we
    poke at the standard attributes and fall back to a clear error if
    none is available. Callers needing custom basis sets can pass
    ``basis_name`` explicitly via a future kwarg.
    """
    for attr in ("basis_name", "name"):
        if hasattr(basis, attr):
            v = getattr(basis, attr)
            if isinstance(v, str) and v:
                return v
    raise AttributeError(
        "cpcm_gradient: could not infer the basis-set name from the "
        "BasisSet object. Pass it explicitly or upgrade vibeqc to a "
        "build that exposes BasisSet.name."
    )


# =====================================================================
# FD reference (kept for v0.9.1 cross-checks)
# =====================================================================


def cpcm_gradient_fd(
    molecule,  # vibeqc.Molecule
    basis_name: str,
    *,
    method: str = "rhf",
    solvent: Any = "water",
    options: Any = None,
    step_bohr: float = 1e-3,
) -> np.ndarray:
    """Finite-difference reference gradient -- slow but provably correct.

    Runs ``6.n_atoms`` full :func:`run_cpcm_scf` calls. Kept as the
    cross-check oracle for the production :func:`cpcm_gradient`
    routine. For routine geometry optimisation prefer
    :func:`cpcm_gradient`, which is ~100x faster.
    """
    from vibeqc import Atom, BasisSet, Molecule

    from .driver import run_cpcm_scf

    atoms_in = list(molecule.atoms)
    n_atoms = len(atoms_in)
    charge = molecule.charge
    mult = molecule.multiplicity

    def _energy_at(positions: np.ndarray) -> float:
        new_atoms = [
            Atom(int(atoms_in[i].Z), tuple(positions[i])) for i in range(n_atoms)
        ]
        new_mol = Molecule(new_atoms, charge, mult)
        new_basis = BasisSet(new_mol, basis_name)
        result = run_cpcm_scf(
            new_mol,
            new_basis,
            method=method,
            solvent=solvent,
            options=options,
        )
        return float(result.energy)

    positions = np.array([list(a.xyz) for a in atoms_in], dtype=np.float64)
    grad = np.zeros_like(positions)

    h = float(step_bohr)
    for ia in range(n_atoms):
        for ic in range(3):
            xp = positions.copy()
            xm = positions.copy()
            xp[ia, ic] += h
            xm[ia, ic] -= h
            e_plus = _energy_at(xp)
            e_minus = _energy_at(xm)
            grad[ia, ic] = (e_plus - e_minus) / (2.0 * h)

    return grad
