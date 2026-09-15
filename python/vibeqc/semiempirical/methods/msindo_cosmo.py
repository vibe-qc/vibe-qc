"""MSINDO implicit solvation (COSMO) via the ``SolutePotentialProvider`` seam.

This is the MSINDO arm of vibe-qc's reference-pluggable CPCM/COSMO engine
(:mod:`vibeqc.solvation`).  Everything in an apparent-surface-charge model -- the
cavity tessellation, the segment interaction matrix ``A``, the dielectric
screening ``f(e)``, the ASC solve ``A q = -f V``, and the polarisation energy
``1/2 q.V`` -- is reference-independent and reused verbatim from
:mod:`vibeqc.solvation.cavity` / :mod:`vibeqc.solvation.cpcm`.  Only the two
solute<->cavity arrows are method-specific, and those are what this module
supplies for an INDO (MSINDO) reference:

1. **density -> surface potential** ``V_elec(s_i)`` and
2. **surface charges -> Fock operator** ``V_q``.

MSINDO couples the solute to the cavity not through an ESP-on-grid (as the
Gaussian/HF/DFT provider does) but through a **distributed-multipole** B-matrix
built from two-centre Slater penetration integrals
``<mu(A)|1/|r - s_i||ν(A)|>`` (``V2INT``).  The solute "charge" vector is

    Q = { core charges CZ(A) } ∪ { diagonal AO populations P_mumu }
        ∪ { 2.P_muν for same-atom p-p and d-d pairs }

(``cosmo_cbi.f``).  With ``B`` the (component x segment) coupling matrix and
``A`` the segment matrix, the COSMO interaction is the standard conductor
energy ``E = -1/2 f Qᵀ (B A⁻¹ Bᵀ) Q`` and the Fock term is
``F_muν -= f (B A⁻¹ Bᵀ Q)_(muν)`` (``cosmo_fock.f`` / ``cosmo_scf.f``;
``cosmo_integrals.f``).  These are Klamt-Schüürmann 1993 eqns (10) and (12):
surface charges ``q* = -A⁻¹ B Q`` and screening energy
``ΔE = -1/2 Q (B A⁻¹ B) Q`` (with the f(e) factor folded in).  The B-matrix here
is built from *basis-function* penetration integrals (Klamt 1993 Appendix C,
eq. C2) -- the formulation Klamt called "preferable" but left unimplemented in
the original MOPAC code, which instead used the point-charge monopole+dipole
representation of eqns (17)-(18); MSINDO realises the basis-function B.
Algebraically this is *exactly* vibe-qc's CPCM energy ``1/2 q.V`` with the
surface potential ``V = Bᵀ Q`` -- penetration integrals for the electronic
charges, bare ``1/r`` point charges for the cores -- so the same
:func:`~vibeqc.solvation.cpcm.solve_apparent_charges` engine drives it.

The screening factor is the Klamt-Schüürmann conductor form
``f(e) = (e - 1)/(e + 0.5)`` (``cosmo_mat.f``;
:func:`vibeqc.solvation.cpcm.dielectric_factor` ``variant="cosmo"``).  The
``x = 1/2`` value is Klamt 1993 p. 801: it extends the conductor (e->inf) result to
finite e with a relative error below ``1/2e⁻¹``.

Algorithm sources (read for the algorithm only; MSINDO 2025e is licensed --
no source text is reproduced and no source path is recorded here): the COSMO
B/A assembly ``cosmo_mat.f``, the charge-vector / component map
``cosmo_cbi.f``, the Fock contribution ``cosmo_fock.f`` / ``cosmo_scf.f``, the
effective one-/two-electron matrices ``cosmo_integrals.f``, and the penetration
integral ``v2int.f`` (already ported to
:func:`vibeqc.semiempirical.methods.msindo_integrals.v2int`).

References
----------
* Klamt, A. & Schüürmann, G. *J. Chem. Soc. Perkin Trans. 2* 799 (1993)
  -- COSMO conductor model and the ``(e-1)/(e+0.5)`` factor.
* Silla, E., Tuñón, I. & Pascual-Ahuir, J. L. *J. Comput. Chem.* 12, 1077
  (1991) -- GEPOL cavity construction (MSINDO's native tessellation; vibe-qc
  reuses its own Lebedev-on-Bondi cavity -- see ``msindo_cosmo`` Notes).

Cavity (``cavity=`` argument of :func:`msindo_cosmo`)
----------------------------------------------------
* ``cavity="gepol"`` (**default**) -- MSINDO's own GEPOL/SAS tessellation
  (:mod:`vibeqc.semiempirical.methods.msindo_gepol`): pentakisdodecahedron
  segments at the COSMOR radius + the capped A-matrix.  Reproduces the
  reference-MSINDO oracle to ``~2e-9 Ha`` (H₂O/HF/CH₄/H₂S vs the oracle's
  ``NOSYM`` run -- vibe-qc builds each atom's cavity independently, matching
  MSINDO's no-symmetry path; the reference's default symmetry path transforms
  symmetry-equivalent atoms' segments and can differ by up to ~10 µHa).
* ``cavity="lebedev"`` -- vibe-qc's Lebedev-on-Bondi cavity + Scalmani-Frisch
  A-matrix (the one HF/DFT CPCM uses).  Its absolute E_solv carries that cavity
  convention (≈ -1.4 mHa from the oracle on H₂O: -6.83 vs -8.26 mHa); the
  multipole coupling is identical, so on the *same* cavity it agrees with the
  Gaussian ESP provider to ~0.1 mHa.  Notably the A-matrix self-term already
  matches MSINDO: vibe-qc's ``1.0694.√(4pi)/√w = 3.79/√w`` ≈ Klamt 1993 eq. (7b)
  ``a_mumu ≈ 3.8.|S|^-1/2``.

See ``docs/user_guide/msindo.md`` Sec. Implicit solvation (COSMO).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from ...solvation.cavity import build_cavity
from ...solvation.cpcm import build_A_matrix, solve_apparent_charges
from ...solvation.engine import ReactionField, reaction_field_step
from ...solvation.screening import ScreeningModel
from . import msindo as _m
from . import msindo_integrals as _ki

# Local-frame orbital order used by HARMTR / the diagonal penetration matrix:
#   0 = s, 1 = ps, 2,3 = ppi, 4 = ds, 5,6 = dpi, 7,8 = dd
# (matches msindo._pair_blocks and msindo_integrals.harmtr).


@lru_cache(maxsize=1)
def _cpp_cosmo_b_kernel():
    try:
        from vibeqc._vibeqc_core.semiempirical import indo as _indo
    except ImportError:
        return None
    try:
        build_b = _indo.cosmo_b_matrix
        load_params_from_json = _indo.load_params_from_json
    except AttributeError:
        return None
    params = load_params_from_json(
        Path(__file__).with_name("msindo_params.json").read_text()
    )
    return build_b, params


@lru_cache(maxsize=1)
def _cpp_cosmo_ops_kernel():
    try:
        from vibeqc._vibeqc_core.semiempirical import indo as _indo
    except ImportError:
        return None
    try:
        charge_vector = _indo.cosmo_charge_vector
        esp_at_cavity = _indo.cosmo_esp_at_cavity
        fock_contribution = _indo.cosmo_fock_contribution
        core_potential = _indo.cosmo_core_potential_at_cavity
    except AttributeError:
        return None
    return charge_vector, esp_at_cavity, fock_contribution, core_potential


@lru_cache(maxsize=1)
def _cpp_cosmo_reaction_kernel():
    try:
        from vibeqc._vibeqc_core.semiempirical import indo as _indo
    except ImportError:
        return None
    return getattr(_indo, "cosmo_reaction_field", None)


def _local_penetration_diag(z: int, dist: float, nb: int) -> np.ndarray:
    """Local-frame diagonal <mu|1/r_seg|mu> vector (length 9) for atom ``z``.

    The s/pi/d resolved nuclear-attraction (penetration) integrals of an atomic
    STO shell against a point at distance ``dist`` (bohr).  Mirrors the
    ``V2INT(n, l1, l2, mu, R, 'R')`` calls of ``cosmo_mat.f`` -- the local matrix
    is **diagonal** there (the s-p / s-d off-diagonal local terms ``V2SP`` etc.
    are passed as zero), so only the per-(l,|m|) diagonal entries are needed; the
    global-frame off-diagonal couplings appear after the ``T.M.Tᵀ`` rotation.

    Uses the two-centre valence exponents ``MUS/MUP/MUD`` and the per-shell
    principal quantum numbers (``cosmo_mat.f`` BSFN = IN(N) with H->p n=2 and the
    transition-metal (n-1)d shift, here via ``n_principal``/``n_p_principal``/
    ``n_d_principal``).
    """
    M = np.zeros(9)
    ns = _m.n_principal(z)
    M[0] = _ki.v2int(ns, 0, 0, _m.MUS[z], dist)
    if nb >= 4:
        npr = _m.n_p_principal(z)
        mup = _m.MUP[z]
        M[1] = _ki.v2int(npr, 1, 0, mup, dist)  # ps  (m = 0)
        M[2] = M[3] = _ki.v2int(npr, 1, 1, mup, dist)  # ppi  (m = ±1)
    if nb >= 9:
        nd = _m.n_d_principal(z)
        mud = _m.MUD[z]
        M[4] = _ki.v2int(nd, 2, 0, mud, dist)  # ds  (m = 0)
        M[5] = M[6] = _ki.v2int(nd, 2, 1, mud, dist)  # dpi  (m = ±1)
        M[7] = M[8] = _ki.v2int(nd, 2, 2, mud, dist)  # dd  (m = ±2)
    return M


def _v2sas_global(z: int, dist: float, E: np.ndarray, nb: int) -> np.ndarray:
    """Global-frame 9x9 penetration matrix <mu(A)|1/r_seg|ν(A)> for one segment.

    ``T.M.Tᵀ`` with ``T = HARMTR`` and ``M`` the local-frame diagonal -- the same
    local->global rotation MSINDO does in ``cosmo_mat.f`` via ``HARMTR`` + ``TINT``
    (the ``T.M.Tᵀ`` form is identical to TINT's sparse unroll and is what the
    MSINDO molecular engine uses for its two-centre integrals too -- see
    ``msindo._pair_blocks``).  ``E`` is the unit vector segment->atom; for a
    diagonal ``M`` the result is invariant under ``E -> -E``.
    """
    M = _local_penetration_diag(z, dist, nb)
    k = 1 if nb == 1 else (2 if nb == 4 else 3)
    T = np.asarray(_ki.harmtr(k, E))
    return T @ np.diag(M) @ T.T


def _same_atom_pairs(lo: int, nb: int):
    """Same-atom off-diagonal charge components (``cosmo_cbi.f``).

    Yields ``(mu, nu)`` global AO index pairs (``mu > nu``) for the p-p and d-d
    blocks of one atom -- the only off-diagonal density elements MSINDO COSMO
    carries in its charge vector (no s-p / s-d / p-d cross terms).  Local atom
    layout: ``lo`` = s, ``lo+1..lo+3`` = pₓ,p_y,p_z, ``lo+4..lo+8`` = d₁..d₅.
    """
    if nb >= 4:  # p-p: (p_y,p_x), (p_z,p_x), (p_z,p_y)
        for mu in (lo + 2, lo + 3):
            for nu in range(lo + 1, mu):
                yield mu, nu
    if nb >= 9:  # d-d: all 10 (d_j, d_i) with j > i
        for mu in range(lo + 5, lo + 9):
            for nu in range(lo + 4, mu):
                yield mu, nu


class MSINDOMultipoleProvider:
    """``SolutePotentialProvider`` for an MSINDO (INDO) reference.

    Distributed-multipole solute<->cavity coupling: a B-matrix of two-centre
    Slater penetration integrals couples the solute charge components (diagonal
    AO populations + same-atom p-p / d-d off-diagonal pairs) to the cavity
    segments.  Built once per geometry + cavity; both seam arrows are then cheap
    matrix-vector products.

    Sign convention matches :class:`vibeqc.solvation.driver.GaussianESPProvider`:
    :meth:`esp_at_cavity` returns the *electronic* potential at the segments
    (negative for a normal density) and :meth:`fock_contribution` returns the
    operator to **add** to the core Hamiltonian for a given ASC vector.  The
    nuclear/core point-charge potential is supplied separately by the driver
    (``S_A CZ_A/|R_A - s_i|``), exactly as MSINDO splits ``cosmo_mat.f``'s
    ``COSMOB`` into a core block (``-1/r``) and an electronic block (``V2INT``).
    """

    __slots__ = ("nsto", "n_comp", "_B", "_pairs", "_diag_idx", "_native_ops")

    def __init__(self, Z, coords_bohr, blocks, cavity_points):
        Z = list(Z)
        C = np.asarray(coords_bohr, dtype=np.float64)
        S = np.asarray(cavity_points, dtype=np.float64)
        nsto = int(blocks[-1][1]) if blocks else 0
        n_pts = int(S.shape[0])

        kernel = _cpp_cosmo_b_kernel()
        if kernel is not None:
            build_b, params = kernel
            B, pairs, nsto_cpp = build_b(Z, C, S, params)
            self.nsto = int(nsto_cpp)
            self.n_comp = int(np.asarray(B).shape[0])
            self._B = np.asarray(B, dtype=np.float64)
            self._pairs = [tuple(int(v) for v in pair) for pair in pairs]
            self._diag_idx = np.arange(self.nsto)
            self._native_ops = _cpp_cosmo_ops_kernel()
            return

        # Charge-component layout (cosmo_cbi.f, OFFSET=0): the first ``nsto``
        # components are the diagonal AO populations (component index = AO index);
        # the same-atom p-p / d-d off-diagonal pairs are appended.
        pairs: list[tuple[int, int, int]] = []  # (component, mu, nu)
        comp = nsto
        for lo, hi in blocks:
            for mu, nu in _same_atom_pairs(lo, hi - lo):
                pairs.append((comp, mu, nu))
                comp += 1
        n_comp = comp

        # Group each off-diagonal pair under its owning atom, as local AO indices
        # for the V2SAS lookup (mu, nu are always on the same atom).
        pair_lookup: dict[int, list[tuple[int, int, int]]] = {}
        for cmp, mu, nu in pairs:
            for ia, (lo, hi) in enumerate(blocks):
                if lo <= mu < hi:
                    pair_lookup.setdefault(ia, []).append((cmp, mu - lo, nu - lo))
                    break

        # B-matrix (component x segment).  Stored with MSINDO's sign:
        # ``cosmo_mat.f`` writes ``-V2SAS`` for the electronic block, so the
        # electronic surface potential ``Bᵀ Q_e`` is negative for a normal
        # (positive-population) density.
        B = np.zeros((n_comp, n_pts), dtype=np.float64)
        for ia, ((lo, hi), z) in enumerate(zip(blocks, Z)):
            nb = hi - lo
            R_a = C[ia]
            d = R_a[None, :] - S  # (n_pts, 3) atom - segment
            dist = np.sqrt((d * d).sum(axis=1))
            for i in range(n_pts):
                Di = float(dist[i])
                Ei = d[i] / Di
                V = _v2sas_global(z, Di, Ei, nb)
                for k in range(nb):
                    B[lo + k, i] = -V[k, k]
                for cmp, a_loc, b_loc in pair_lookup.get(ia, ()):
                    B[cmp, i] = -V[a_loc, b_loc]

        self.nsto = nsto
        self.n_comp = n_comp
        self._B = B
        self._pairs = pairs
        self._diag_idx = np.arange(nsto)
        self._native_ops = _cpp_cosmo_ops_kernel()

    # -- charge vector ------------------------------------------------------
    def charge_vector(self, density: np.ndarray) -> np.ndarray:
        """Electronic charge components ``Q_e`` from the density (``cosmo_cbi.f``).

        Diagonal components are ``P_mumu``; same-atom off-diagonal pair components
        are ``2.P_muν`` (the factor 2 folds the symmetric ``(mu,ν)``/``(ν,mu)``
        density pair into one component).  ``density`` is the closed-shell RHF
        density ``P`` (= PA, with ``S P_mumu = N_elec``).
        """
        P = np.asarray(density, dtype=np.float64)
        if self._native_ops is not None:
            charge_vector, _esp_at_cavity, _fock_contribution, _core = self._native_ops
            return np.asarray(
                charge_vector(P, self.nsto, self._pairs),
                dtype=np.float64,
            )
        Q = np.empty(self.n_comp, dtype=np.float64)
        Q[: self.nsto] = np.diag(P)
        for cmp, mu, nu in self._pairs:
            Q[cmp] = 2.0 * P[mu, nu]
        return Q

    # -- SolutePotentialProvider seam --------------------------------------
    def esp_at_cavity(self, density: np.ndarray) -> np.ndarray:
        """Electronic potential ``V_elec(s_i) = S_comp B(comp,i) Q_e(comp)``."""
        if self._native_ops is not None:
            _charge_vector, esp_at_cavity, _fock_contribution, _core = self._native_ops
            return np.asarray(
                esp_at_cavity(
                    self._B,
                    np.asarray(density, dtype=np.float64),
                    self.nsto,
                    self._pairs,
                ),
                dtype=np.float64,
            )
        return self._B.T @ self.charge_vector(density)

    def fock_contribution(self, charges: np.ndarray) -> np.ndarray:
        """Operator to add to ``H_core`` for ASC vector ``q``.

        ``(V_q)_muν = (B q)_(component(muν))`` (``cosmo_fock.f``): the reaction
        field each segment charge induces on the molecular charge components,
        scattered back onto the AO pairs (diagonal + same-atom p-p / d-d).
        """
        q = np.asarray(charges, dtype=np.float64)
        if self._native_ops is not None:
            _charge_vector, _esp_at_cavity, fock_contribution, _core = self._native_ops
            return np.asarray(
                fock_contribution(self._B, q, self.nsto, self._pairs),
                dtype=np.float64,
            )
        psi = self._B @ q  # (n_comp,)
        Vq = np.zeros((self.nsto, self.nsto), dtype=np.float64)
        Vq[self._diag_idx, self._diag_idx] = psi[: self.nsto]
        for cmp, mu, nu in self._pairs:
            Vq[mu, nu] = Vq[nu, mu] = psi[cmp]
        return Vq


@dataclass
class MsindoCosmoResult:
    """Outcome of an MSINDO COSMO SCF.

    ``total_energy`` is the in-solvent total energy (the parity quantity).
    ``e_solv`` is the solvation energy = ``total_energy - e_gas`` (the
    energy-difference convention MSINDO reports); ``e_pol`` is the polarisation
    energy ``1/2 q.V`` at the converged density (the conductor energy
    ``-1/2 f Qᵀ B A⁻¹ Bᵀ Q``).
    """

    total_energy: float = 0.0
    e_gas: float = 0.0
    e_solv: float = 0.0
    e_pol: float = 0.0
    electronic_energy: float = 0.0
    binding_energy: float = 0.0
    epsilon: float = 1.0
    density: np.ndarray = field(default=None, repr=False)
    mo_energies: np.ndarray = field(default=None, repr=False)
    n_iter: int = 0
    converged: bool = False
    # CavityTessellation (cavity="lebedev") or GepolCavity (cavity="gepol").
    cavity: object = field(default=None, repr=False)


def _emit_msindo_cosmo_citations(output, cavity_kind: str = "gepol") -> None:
    """Best-effort ``.bibtex`` / ``.references`` for an MSINDO COSMO run.

    Fires the MSINDO method route (Ahlswede-Jug 1999) plus the COSMO solvation
    bundle.  The bundle depends on the cavity: ``cavity="gepol"`` fires
    ``routes.solvation['cosmo_gepol']`` (Klamt 1993 + CPCM layout + the GEPOL
    cavity paper Silla 1991), while ``cavity="lebedev"`` fires
    ``routes.solvation['cosmo']`` (Klamt + CPCM + the Lebedev/SWIG smooth-cavity
    papers).  INDO uses no Gaussian integrals, so libint is suppressed
    (``uses_integrals=False``).  Non-fatal -- a citation-writer failure never
    tanks a finished calculation (mirrors ``msindo._emit_msindo_cis_citations``
    / the runner)."""
    variant = "cosmo_gepol" if cavity_kind.lower() == "gepol" else "cosmo"
    try:
        from vibeqc.output.citations import emit_citations

        emit_citations(
            output,
            method="msindo",
            uses_integrals=False,
            uses_cpcm=True,
            solvent_variant=variant,
        )
    except Exception:
        pass


def _core_potential_at_cavity(C: np.ndarray, cz, S: np.ndarray) -> np.ndarray:
    """Core point-charge potential ``V_core(s_i) = S_A CZ_A/|R_A - s_i|``.

    MSINDO's ``cosmo_mat.f`` couples the core charges ``CZ(A)`` to each segment
    as a bare point charge ``-1/r`` (stored, then sign-flipped) -- i.e. the same
    nuclear-potential-at-cavity the Gaussian driver uses, but with the INDO
    effective core charge ``CZ = Z - n_core`` rather than the full nuclear ``Z``.
    """
    C = np.asarray(C, dtype=np.float64)
    cz = np.asarray(cz, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    ops = _cpp_cosmo_ops_kernel()
    if ops is not None:
        _charge_vector, _esp_at_cavity, _fock_contribution, core_potential = ops
        return np.asarray(core_potential(C, cz, S), dtype=np.float64)
    diff = C[:, None, :] - S[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=2))  # (n_atoms, n_pts)
    return (cz[:, None] / dist).sum(axis=0)


def _cosmo_reaction_field(provider, A, V_core, density, *, screening):
    """Return the :class:`~vibeqc.solvation.engine.ReactionField` for one density.

    This is the per-SCF reaction-field hot path.  The native route performs the
    electronic surface potential, apparent-surface-charge solve, Fock scatter,
    and energy bookkeeping in one C++ call; the Python route stays as the
    reference/fallback implementation.

    Takes the :class:`~vibeqc.solvation.screening.ScreeningModel` that governs
    the run rather than a bare ``epsilon`` plus a restated ``variant="cosmo"``
    literal. Both routes then screen with one factor from one table, and the
    native and reference paths cannot drift apart (#548).
    """
    P = np.asarray(density, dtype=np.float64)
    A_arr = np.asarray(A, dtype=np.float64)
    Vc = np.asarray(V_core, dtype=np.float64)
    reaction = _cpp_cosmo_reaction_kernel()
    if reaction is not None:
        # Fused native path: one C++ call does ESP, solve, Fock scatter and
        # the energy split. Projected onto the shared ReactionField record so
        # both routes hand back one shape and one convention.
        res = reaction(
            provider._B, A_arr, Vc, P, provider.nsto, provider._pairs,
            screening.epsilon, screening.variant,
        )
        V_total = np.asarray(res.V_total, dtype=np.float64)
        return ReactionField(
            q=np.asarray(res.q, dtype=np.float64),
            V_elec=V_total - Vc,
            V_core=Vc,
            V_total=V_total,
            fock=np.asarray(res.fock, dtype=np.float64),
            e_pol=float(res.e_pol),
            e_core_share=float(res.e_add),
            screening=screening,
        )

    # Reference path: the generic step, shared with the Gaussian driver.
    return reaction_field_step(
        provider, lambda rhs: np.linalg.solve(A_arr, rhs), Vc, P, screening
    )


def msindo_cosmo(
    atomic_numbers,
    coords_angstrom,
    *,
    epsilon: float,
    variant: str = "cosmo",
    charge: int = 0,
    multiplicity: int = 1,
    cavity: str = "gepol",
    radii: Optional[dict] = None,
    radii_scale: float = 1.20,
    solvent_probe_radius_ang: float = 0.0,
    n_points_per_sphere: int = 302,
    max_iter: int = 200,
    conv_tol: float = 1e-9,
    output=None,
) -> MsindoCosmoResult:
    """MSINDO closed-shell (RHF) SCF in implicit solvent (COSMO).

    The reaction field is folded into the SCF through the MSINDO
    :func:`~vibeqc.semiempirical.methods.msindo._scf_rhf` ``fock_extra`` hook --
    each Fock build solves ``A q = -f (V_elec(P) + V_core)`` on the cavity and
    adds the resulting one-electron operator, so the density and the apparent
    surface charge converge together in a single SCF (exactly as MSINDO's
    ``cosmo_scf.f`` does; typically one extra cycle over the gas-phase SCF).

    Parameters
    ----------
    atomic_numbers, coords_angstrom
        The solute (Å), as for :func:`vibeqc.semiempirical.methods.msindo.run_msindo`.
    epsilon
        Solvent dielectric constant (must be > 1).
    charge
        Net charge (shifts the valence electron count).
    cavity : {"gepol", "lebedev"}, default "gepol"
        ``"gepol"`` -- MSINDO's GEPOL/SAS cavity for exact oracle parity (the
        default).  ``"lebedev"`` -- vibe-qc's Lebedev-on-Bondi cavity (shared
        with HF/DFT CPCM); see the module Cavity section.
    radii, radii_scale, solvent_probe_radius_ang, n_points_per_sphere
        Cavity controls forwarded to :func:`vibeqc.solvation.cavity.build_cavity`
        -- **only used for** ``cavity="lebedev"`` (the GEPOL cavity uses MSINDO's
        COSMOR radii).
    output
        Optional path stem; when given, writes the ``{stem}.bibtex`` /
        ``{stem}.references`` citation siblings (MSINDO + the COSMO solvation
        bundle -- see :func:`_emit_msindo_cosmo_citations`).

    Returns
    -------
    MsindoCosmoResult
    """
    if epsilon <= 1.0:
        raise ValueError(
            f"msindo_cosmo: epsilon must be > 1; got {epsilon} "
            "(use msindo.run_msindo for gas phase)."
        )
    Z = list(atomic_numbers)
    missing = sorted({z for z in Z if z not in _m._SUPPORTED})
    if missing:
        raise NotImplementedError(
            f"MSINDO engine supports {sorted(_m._SUPPORTED)}; got Z={missing}."
        )

    C = np.asarray(coords_angstrom, dtype=np.float64) * _m.ANGSTROM_TO_BOHR
    blocks, nsto = _m._atom_blocks(Z)
    cz = [_m.eff_core_charge(z) for z in Z]
    nelec = sum(cz) - charge
    if nelec < 0:
        raise ValueError(f"charge={charge} exceeds the {sum(cz)} valence electrons")

    H, G = _m._build_core_and_gamma(Z, C, blocks, nsto)
    e_core = _m._core_repulsion(C, cz)

    open_shell = multiplicity != 1 or nelec % 2 != 0

    # ---- gas-phase reference (same engine, no reaction field) -------------
    if open_shell:
        nalpha, nbeta = _m._uhf_occupation(nelec, multiplicity)
        PA_gas, PB_gas, _CA, _CB, _epsA, _epsB, e_elec_gas, conv_gas, _ = _m._scf_uhf(
            H, G, blocks, Z, nalpha, nbeta, max_iter=max_iter, conv_tol=conv_tol
        )
        P_gas = PA_gas + PB_gas
    else:
        nocc = nelec // 2
        P_gas, _F, e_elec_gas, eps_gas, conv_gas, _ = _m._scf_rhf(
            H, G, blocks, Z, nocc, max_iter=max_iter, conv_tol=conv_tol
        )
    e_gas = e_elec_gas + e_core

    # ---- cavity + reaction-field engine (reference-independent) -----------
    # ``cavity="gepol"`` (default) is MSINDO's own GEPOL/SAS tessellation +
    # capped A-matrix + COSMOR radii -> exact oracle parity. ``cavity="lebedev"``
    # reuses vibe-qc's Lebedev-on-Bondi cavity (shared with HF/DFT CPCM); its
    # absolute E_solv carries that cavity convention (≈ -1.4 mHa from the oracle
    # on H₂O) -- see the module Cavity-convention note.
    cav_kind = cavity.lower()
    if cav_kind == "gepol":
        from .msindo_gepol import build_gepol_A_matrix, build_gepol_cavity

        cavity_obj = build_gepol_cavity(Z, C)
        cav_points = cavity_obj.points
        A = build_gepol_A_matrix(cavity_obj)
    elif cav_kind == "lebedev":
        Zs = np.array([int(z) for z in Z], dtype=int)
        cavity_obj = build_cavity(
            atom_positions_bohr=C,
            atom_numbers=Zs,
            radii=radii,
            radii_scale=radii_scale,
            solvent_probe_radius_ang=solvent_probe_radius_ang,
            n_points_per_sphere=n_points_per_sphere,
        )
        cav_points = cavity_obj.points
        A = build_A_matrix(cavity_obj.points, cavity_obj.weights)
    else:
        raise ValueError(
            f"msindo_cosmo: unknown cavity {cavity!r} (use 'gepol' or 'lebedev')."
        )
    provider = MSINDOMultipoleProvider(Z, C, blocks, cav_points)
    V_core = _core_potential_at_cavity(C, cz, cav_points)
    # One screening model for the run. MSINDO's default is the Klamt-
    # Schuurmann COSMO form, but the screening variant is a property of the
    # generic reaction-field layer, not of the Hamiltonian, so it is a
    # parameter here rather than a constant baked into the method (#548).
    screening = ScreeningModel.from_variant(epsilon, variant)

    # ---- COSMO SCF: reaction field folded in via fock_extra ---------------
    # fock_extra(P) -> (F_add, e_add):
    #   q       = -f A⁻¹ (V_elec(P) + V_core)            [solve_apparent_charges]
    #   F_add   = provider.fock_contribution(q)          [the full reaction field]
    #   e_add   = 1/2 q.V_core
    # _scf_rhf reports 1/2Tr[P(H+F)] + e_add = E_elec_gas[P] + 1/2q.V_elec + 1/2q.V_core
    #         = E_elec_gas[P] + 1/2q.V_tot = E_elec_gas[P] + E_solv(P),
    # so the total in-solvent electronic energy is exact in one SCF (the 1/2q.V_core
    # term is the core's share of the conductor energy; cosmo_scf.f bookkeeping).
    # The screening factor comes from the run's ScreeningModel; see
    # vibeqc.solvation.screening for why it is carried, not restated.
    def fock_extra(P):
        field = _cosmo_reaction_field(
            provider, A, V_core, P, screening=screening
        )
        return field.fock, field.e_core_share

    # ---- COSMO SCF: reaction field folded in via fock_extra ---------------
    if open_shell:
        PA, PB, CA, CB, epsA, epsB, e_elec, converged, n_iter = _m._scf_uhf(
            H,
            G,
            blocks,
            Z,
            nalpha,
            nbeta,
            max_iter=max_iter,
            conv_tol=conv_tol,
            fock_extra=fock_extra,
        )
        P = PA + PB
        eps_mo = epsA  # a MO energies for the result
    else:
        P, _Fc, e_elec, eps_mo, converged, n_iter = _m._scf_rhf(
            H,
            G,
            blocks,
            Z,
            nocc,
            max_iter=max_iter,
            conv_tol=conv_tol,
            fock_extra=fock_extra,
        )
    total = e_elec + e_core

    # Polarisation energy 1/2 q.V at the converged density (diagnostic).
    _final_field = _cosmo_reaction_field(
        provider, A, V_core, P, screening=screening
    )
    e_pol_final = _final_field.e_pol

    if output is not None:
        _emit_msindo_cosmo_citations(output, cav_kind)

    binding = total - sum(_m.ateng(z) for z in Z)
    return MsindoCosmoResult(
        total_energy=total,
        e_gas=e_gas,
        e_solv=total - e_gas,
        e_pol=e_pol_final,
        electronic_energy=e_elec,
        binding_energy=binding,
        epsilon=float(epsilon),
        density=P,
        mo_energies=eps_mo,
        n_iter=n_iter,
        converged=converged and conv_gas,
        cavity=cavity_obj,
    )
