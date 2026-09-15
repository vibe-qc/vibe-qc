"""General CIS / Tamm-Dancoff excited states (reference-agnostic).

The third method built on the :class:`vibeqc.correlation.ERIProvider` seam
(after MP2 in :mod:`vibeqc.correlation` and GF2/OVGF in
:mod:`vibeqc.propagator`): configuration-interaction singles / the
Tamm-Dancoff approximation depends on the SCF reference only through its MO
energies and the ``(ia|jb)`` / ``(ij|ab)`` integral blocks, so one kernel gives
excited states for HF, DFT (TDA), and the semiempirical engines (MSINDO).

The spin-adapted TDA / Casida-A matrix (chemist's notation ``(pq|rs)``):

    singlet:  A_{ia,jb} = d_ij d_ab (e_a - e_i) + 2 (ia|jb) - (ij|ab)
    triplet:  A_{ia,jb} = d_ij d_ab (e_a - e_i)              - (ij|ab)

Diagonalizing ``A`` gives the vertical excitation energies (eigenvalues) and
the CIS amplitudes (eigenvectors).  This matches vibe-qc's libint-coupled
:func:`vibeqc.tddft.run_tddft_tda` with an HF reference (``c_x = 1``); the value
added here is that the *same* kernel runs over any reference's integral blocks
(e.g. the INDO set behind MSINDO).

Method reference: J. B. Foresman, M. Head-Gordon, J. A. Pople & M. J. Frisch,
"Toward a systematic molecular orbital theory for excited states",
J. Phys. Chem. 96, 135 (1992) (CIS); the TDA / Casida-A convention follows
S. Hirata & M. Head-Gordon, Chem. Phys. Lett. 314, 291 (1999).  (These are
named here as the inline algorithmic record; vibe-qc's excited-state runs are
not yet wired to a Sec.8 citation route -- a pre-existing gap shared with
``vibeqc.tddft``.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_HARTREE_TO_EV = 27.211386245988


@dataclass(frozen=True)
class CISResult:
    """CIS / TDA excited states (energies in Hartree, ascending)."""

    excitation_energies: np.ndarray   # (n_states,)
    amplitudes: np.ndarray            # (n_occ*n_vir, n_states) CIS eigenvectors
    spin: str                         # "singlet" | "triplet"
    n_occ: int
    n_vir: int

    @property
    def excitation_energies_ev(self) -> np.ndarray:
        return self.excitation_energies * _HARTREE_TO_EV

    def dominant_transition(self, state: int) -> tuple:
        """(occ, vir, weight) of the leading amplitude of ``state`` (0-based).

        ``occ`` is a 0-based occupied index, ``vir`` a 0-based virtual index
        (0 = LUMO), ``weight`` the squared amplitude.
        """
        vec = self.amplitudes[:, state].reshape(self.n_occ, self.n_vir)
        i, a = np.unravel_index(int(np.argmax(vec ** 2)), vec.shape)
        return int(i), int(a), float(vec[i, a] ** 2)


def cis_matrix(eps, n_occ, ovov, oovv, *, spin: str = "singlet") -> np.ndarray:
    """Build the spin-adapted CIS / TDA A-matrix (shape ``(n_occ*n_vir,)*2``).

    ``ovov[i,a,j,b] = (ia|jb)`` ``(n_occ,n_vir,n_occ,n_vir)``;
    ``oovv[i,j,a,b] = (ij|ab)`` ``(n_occ,n_occ,n_vir,n_vir)``.  ``spin`` is
    ``"singlet"`` (Coulomb + exchange) or ``"triplet"`` (exchange only).
    """
    if spin not in ("singlet", "triplet"):
        raise ValueError(f"spin must be 'singlet' or 'triplet', got {spin!r}")
    eps = np.asarray(eps, float)
    no = int(n_occ)
    nv = eps.shape[0] - no
    eo, ev = eps[:no], eps[no:]
    ovov = np.asarray(ovov, float)
    oovv = np.asarray(oovv, float)
    # (ij|ab) reindexed to the [i,a,j,b] layout of the A-matrix.
    exch = np.transpose(oovv, (0, 2, 1, 3))          # exch[i,a,j,b] = (ij|ab)
    if spin == "singlet":
        A = 2.0 * ovov - exch
    else:
        A = -exch
    # Add the orbital-energy differences on the diagonal block (ia|ia).
    de = ev[None, :] - eo[:, None]                    # (n_occ, n_vir)
    ii = np.arange(no)
    aa = np.arange(nv)
    A[ii[:, None], aa[None, :], ii[:, None], aa[None, :]] += de
    return A.reshape(no * nv, no * nv)


def cis_excitations(eps, n_occ, ovov, oovv, *, spin: str = "singlet",
                    n_states: int = 5) -> CISResult:
    """Lowest ``n_states`` CIS / TDA excitations (see :func:`cis_matrix`).

    Reference-agnostic: feed MO energies + the ``(ia|jb)`` / ``(ij|ab)`` blocks
    from any :class:`~vibeqc.correlation.ERIProvider`.  IP/EA-free vertical
    singlet or triplet excitation spectrum.
    """
    A = cis_matrix(eps, n_occ, ovov, oovv, spin=spin)
    w, V = np.linalg.eigh(A)                          # A is symmetric
    no = int(n_occ)
    nv = np.asarray(eps).shape[0] - no
    k = min(int(n_states), w.shape[0])
    return CISResult(excitation_energies=w[:k], amplitudes=V[:, :k],
                     spin=spin, n_occ=no, n_vir=nv)


__all__ = ["CISResult", "cis_matrix", "cis_excitations"]
