"""Excited-state (CIS / TDA) nuclear gradients by finite difference.

The fourth capability on the :class:`vibeqc.correlation.ERIProvider` seam (after
MP2, GF2/OVGF, and the CIS energies in :mod:`vibeqc.excited`).  A CIS
excited-state *total* energy is ``E_state(R) = E_SCF(R) + w_state(R)`` -- the
ground-state SCF energy plus the tracked excitation energy -- so its nuclear
gradient is obtained by central-differencing that total energy over displaced
geometries.  Because the underlying SCF + CIS path is reference-agnostic, the
same driver gives excited-state gradients for HF, DFT (TDA), and the
semiempirical engines (MSINDO).

The one subtlety a finite-difference excited-state gradient must handle is
**root flipping**: as the geometry is displaced the CIS roots can reorder, so
naively differencing "the k-th eigenvalue" mixes states across the step and
gives a meaningless gradient.  The fix is to *track state character* -- at each
displaced geometry, follow the root whose CIS amplitude vector has maximal
overlap with the reference state's amplitude vector (captured once at the
center geometry).  For the small displacements used here the MO basis barely
rotates, so the overlap taken directly in the frozen ``(occ, vir)`` amplitude
index space is an excellent tracker; see :func:`track_state`.

Algorithm references (read for structure only):

* CIS excited-state gradient -- the analytic CPHF/Z-vector route is MSINDO
  ``cisgrad.f`` (``CPHF_SOLVER``, line ~308); the finite-difference route here
  re-evaluates the energy instead, which is the cheap validation baseline the
  analytic version is later checked against.  CIS itself: Foresman,
  Head-Gordon, Pople & Frisch, *J. Phys. Chem.* **96**, 135 (1992).

State indexing convention used throughout this module: ``state = 0`` is the
ground state (``w = 0``); ``state = k >= 1`` selects the ``k``-th CIS excited
root (1-based -- spectroscopic S0, S1, S2, ...).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .excited import CISResult

# Bohr per Angstrom (CODATA).  Used both to build Cartesian geometries for the
# HF adapter and to convert the per-Angstrom finite-difference slope to the
# Hartree/bohr gradient convention used across vibe-qc.
ANGSTROM_TO_BOHR = 1.8897259886

__all__ = [
    "CISStateSet",
    "track_state",
    "cis_state_gradients_fd",
    "cis_states_energy_and_gradients_fd",
    "make_hf_cis_energy_fn",
]


# A CIS energy function maps a geometry (Angstrom, shape (natom, 3)) to the full
# excited-state picture at that geometry.  HF and MSINDO each provide one.
CISEnergyFn = Callable[[np.ndarray], "CISStateSet"]


@dataclass(frozen=True)
class CISStateSet:
    """Ground-state energy + CIS excited states at one geometry.

    ``e_ground`` is the ground-state SCF *total* energy (Hartree).  ``cis`` is
    the :class:`vibeqc.excited.CISResult` for the same geometry (vertical
    excitation energies + CIS amplitude eigenvectors).  Together they give the
    total energy of any state via :meth:`total_energy`.
    """

    e_ground: float
    cis: CISResult

    @property
    def excitation_energies(self) -> np.ndarray:
        return self.cis.excitation_energies

    @property
    def amplitudes(self) -> np.ndarray:
        """CIS eigenvectors, shape ``(n_occ*n_vir, n_states)``."""
        return self.cis.amplitudes

    @property
    def n_excited(self) -> int:
        return int(self.cis.excitation_energies.shape[0])

    def total_energy(self, state: int) -> float:
        """Total energy (Hartree) of ``state`` (0 = ground, k >= 1 = k-th root)."""
        s = int(state)
        if s == 0:
            return float(self.e_ground)
        if not 1 <= s <= self.n_excited:
            raise IndexError(
                f"state {s} out of range; have ground + {self.n_excited} excited "
                f"roots (valid: 0..{self.n_excited})")
        return float(self.e_ground) + float(self.cis.excitation_energies[s - 1])


def track_state(ref_amplitude: np.ndarray, amplitudes: np.ndarray) -> Tuple[int, float]:
    """Follow a CIS state by amplitude-vector overlap (root-flip guard).

    Returns ``(root_index, overlap)`` where ``root_index`` is the column of
    ``amplitudes`` (shape ``(n_pair, n_states)``) whose unit eigenvector has the
    largest absolute overlap with ``ref_amplitude``.  The absolute value makes
    the tracker robust to the arbitrary global sign of an eigenvector.

    The overlap is taken in the frozen ``(occ, vir)`` amplitude index space.
    This neglects the (small) rotation of the MO basis between the two
    geometries, which is appropriate for the tight finite-difference
    displacements used for gradients; it is *not* a rigorous wavefunction
    overlap across arbitrarily different geometries.
    """
    ref = np.asarray(ref_amplitude, float).ravel()
    A = np.asarray(amplitudes, float)
    ov = np.abs(ref @ A)                       # (n_states,)
    j = int(np.argmax(ov))
    return j, float(ov[j])


def _tracked_total_energy(state_set: "CISStateSet", state: int,
                          ref_amplitude: Optional[np.ndarray]) -> float:
    """Total energy of ``state`` at ``state_set``, following ``ref_amplitude``.

    Ground state (``state == 0``) needs no tracking.  An excited state is
    followed to the best-overlap root so a displaced geometry returns the energy
    of the *same* state rather than of whatever happens to be the k-th root.
    """
    if int(state) == 0:
        return float(state_set.e_ground)
    j, _ = track_state(ref_amplitude, state_set.amplitudes)
    return float(state_set.e_ground) + float(state_set.excitation_energies[j])


def _validate_states(states: Sequence[int], n_excited: int) -> List[int]:
    out: List[int] = []
    for s in states:
        si = int(s)
        if not 0 <= si <= n_excited:
            raise ValueError(
                f"state {si} out of range; the energy function returns ground + "
                f"{n_excited} excited roots (valid states: 0..{n_excited}). "
                f"Increase n_states so the requested root is computed.")
        out.append(si)
    return out


def _fd_state_gradients(energy_fn: CISEnergyFn, coords_angstrom: np.ndarray,
                        states: Sequence[int], *, step: float,
                        atoms: Optional[Iterable[int]]
                        ) -> Tuple["CISStateSet", Dict[int, np.ndarray]]:
    """Shared central-difference core: returns the center state-set and a
    ``{state: gradient (natom, 3) in Ha/bohr}`` map, with per-state tracking."""
    C0 = np.asarray(coords_angstrom, float)
    if C0.ndim != 2 or C0.shape[1] != 3:
        raise ValueError("coords_angstrom must have shape (natom, 3)")
    natom = C0.shape[0]
    center = energy_fn(C0)
    states = _validate_states(states, center.n_excited)

    # Lock the reference amplitude vectors once, at the center geometry, so the
    # + and - displacements are both followed back to the same reference state.
    ref_vec: Dict[int, np.ndarray] = {
        s: np.asarray(center.amplitudes[:, s - 1], float).copy()
        for s in states if s >= 1
    }

    which = range(natom) if atoms is None else list(atoms)
    grads: Dict[int, np.ndarray] = {s: np.zeros((natom, 3)) for s in states}
    h = float(step)
    for i in which:
        for d in range(3):
            cp = C0.copy(); cp[i, d] += h
            cm = C0.copy(); cm[i, d] -= h
            sp = energy_fn(cp)
            sm = energy_fn(cm)
            for s in states:
                ep = _tracked_total_energy(sp, s, ref_vec.get(s))
                em = _tracked_total_energy(sm, s, ref_vec.get(s))
                # Central difference of the per-Angstrom slope -> Ha/bohr.
                grads[s][i, d] = (ep - em) / (2.0 * h) / ANGSTROM_TO_BOHR
    return center, grads


def cis_state_gradients_fd(energy_fn: CISEnergyFn, coords_angstrom,
                           states: Union[int, Sequence[int]], *,
                           step: float = 1e-3,
                           atoms: Optional[Iterable[int]] = None
                           ) -> Dict[int, np.ndarray]:
    """Finite-difference nuclear gradient(s) (Ha/bohr) of CIS state energies.

    ``energy_fn(coords_angstrom)`` -> :class:`CISStateSet` is the reference-agnostic
    seam: :func:`make_hf_cis_energy_fn` builds it for HF, and
    ``vibeqc.semiempirical.methods.msindo.make_msindo_cis_energy_fn`` for MSINDO.
    ``states`` is a state index or a list of them (0 = ground, k >= 1 = k-th CIS
    root).  Each excited state is tracked across the displacements by CIS
    amplitude overlap (:func:`track_state`) so root flips do not corrupt the
    gradient.  ``step`` is the central-difference displacement in Angstrom;
    ``atoms`` restricts differentiation to a subset of atom indices.

    Returns ``{state: gradient}`` with each gradient shaped ``(natom, 3)``.
    """
    if isinstance(states, (int, np.integer)):
        states = [int(states)]
    _, grads = _fd_state_gradients(energy_fn, coords_angstrom, list(states),
                                   step=step, atoms=atoms)
    return grads


def cis_states_energy_and_gradients_fd(energy_fn: CISEnergyFn, coords_angstrom,
                                       states: Sequence[int], *,
                                       step: float = 1e-3,
                                       atoms: Optional[Iterable[int]] = None
                                       ) -> Tuple[Dict[int, float],
                                                  Dict[int, np.ndarray]]:
    """State total energies (Ha) *and* their FD gradients (Ha/bohr) at one
    geometry, sharing the displaced SCF+CIS evaluations.

    This is the form the conical-intersection optimizer consumes: it needs
    ``(E, gradE)`` for two tracked states at the current geometry, and computing
    both from one set of displacements avoids re-running the SCF twice.
    Returns ``(energies, gradients)`` each keyed by state index.
    """
    states = list(states)
    center, grads = _fd_state_gradients(energy_fn, coords_angstrom, states,
                                        step=step, atoms=atoms)
    energies = {s: center.total_energy(s) for s in states}
    return energies, grads


def make_hf_cis_energy_fn(atomic_numbers: Sequence[int], basis_name: str, *,
                          charge: int = 0, spin: str = "singlet",
                          n_states: int = 5, rhf_options=None) -> CISEnergyFn:
    """Build a CIS energy function for a closed-shell Hartree-Fock reference.

    Returns ``energy_fn(coords_angstrom) -> CISStateSet`` that, at each geometry,
    runs the RHF SCF, transforms the libint ERIs to the ``(ia|jb)`` / ``(ij|ab)``
    MO blocks, and evaluates the general :func:`vibeqc.excited.cis_excitations`
    kernel -- the libint backend of the ERIProvider seam.  Feed the result to
    :func:`cis_state_gradients_fd` for HF excited-state gradients or to the
    conical-intersection optimizer.

    ``spin`` is ``"singlet"`` or ``"triplet"``; ``n_states`` is how many CIS
    roots to compute (must cover the highest state index you differentiate).
    """
    Z = [int(z) for z in atomic_numbers]
    nelec = sum(Z) - int(charge)
    if nelec % 2 != 0:
        raise NotImplementedError(
            "make_hf_cis_energy_fn is closed-shell (RHF reference) only; the "
            "electron count is odd.")
    n_occ = nelec // 2

    def energy_fn(coords_angstrom) -> CISStateSet:
        from ._vibeqc_core import (Atom, BasisSet, Molecule, RHFOptions,
                                   compute_eri, run_rhf)
        from .ecp_metadata import (
            basis_sidecar_has_ecp_operator,
            molecular_options_request_ecp_operator,
        )
        from .excited import cis_excitations
        from .tddft import make_eri_provider

        C = np.asarray(coords_angstrom, float)
        atoms = [Atom(Z[i], (C[i] * ANGSTROM_TO_BOHR).tolist())
                 for i in range(len(Z))]
        mol = Molecule(atoms, int(charge), 1)
        if basis_sidecar_has_ecp_operator(
            mol, basis_name
        ) or molecular_options_request_ecp_operator(rhf_options):
            raise NotImplementedError(
                "make_hf_cis_energy_fn: molecular ECP excited-state "
                "gradients and optimizations are not implemented. The "
                "finite-difference reference does not preserve the exact "
                "ECP Hamiltonian or effective occupied-space count at every "
                "geometry. Use an all-electron basis."
            )
        bas = BasisSet(mol, basis_name)
        rhf = run_rhf(mol, bas, rhf_options or RHFOptions())
        eps = np.asarray(rhf.mo_energies)
        cmo = np.asarray(rhf.mo_coeffs)
        prov = make_eri_provider(np.asarray(compute_eri(bas)), cmo, n_occ)
        cis = cis_excitations(eps, n_occ, prov.ovov(), prov.oovv(),
                              spin=spin, n_states=n_states)
        return CISStateSet(e_ground=float(rhf.energy), cis=cis)

    return energy_fn
