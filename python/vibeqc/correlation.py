"""General post-SCF correlation / response layer (reference-agnostic).

The methods here (MP2 now; OVGF / CIS to follow) are *textbook* -- they depend on
the SCF reference only through its molecular-orbital energies and a source of
**MO-basis two-electron integrals**.  That source is abstracted as
:class:`ERIProvider` so the same kernels run on any reference:

* **libint** (canonical or density-fitted Gaussian ERIs) for HF / DFT, and
* the **INDO / NDDO approximated integral set** for the semiempirical engines
  (MSINDO), exposed as MO-basis blocks (the vibe-qc analogue of MSINDO's
  ``pqrs.f`` / ``rstu.f``).

This is the "keystone" seam: build it once, and MP2 / SCS-MP2 / OVGF / CIS work
for *every* method (HF, DFT, MSINDO) instead of being re-implemented per engine.
See ``docs/user_guide/msindo.md`` for the program plan.

Integral convention: chemist's notation ``(pq|rs)``.  The occupied-virtual block
``ovov[i, a, j, b] = (ia|jb)`` (shape ``(n_occ, n_vir, n_occ, n_vir)``) drives
closed-shell MP2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ERIProvider(Protocol):
    """Abstract MO-basis two-electron integral source (chemist's notation).

    A backend (libint or semiempirical) implements the integral blocks the
    correlated/response methods need.  Closed-shell spatial-orbital blocks:

    * :meth:`ovov` -- ``(ia|jb)``  (occ, vir, occ, vir) -- MP2, CIS, OVGF
    * :meth:`oovv` -- ``(ij|ab)``  (occ, occ, vir, vir) -- CIS exchange, OVGF

    Backends may compute these lazily / on demand.  Only :meth:`ovov` is needed
    for the MP2 energy; the others are declared for the methods that follow.
    """

    def ovov(self) -> np.ndarray:  # (ia|jb)
        ...

    def oovv(self) -> np.ndarray:  # (ij|ab)
        ...


# Spin-component scaling factors (opposite-spin, same-spin).
#   MP2     -- plain (Moller-Plesset).
#   SCS-MP2 -- Grimme, J. Chem. Phys. 118, 9095 (2003): 6/5 OS + 1/3 SS.
#   SOS-MP2 -- Jung et al., J. Chem. Phys. 121, 9793 (2004): 1.3 OS, no SS.
_MP2_SCALES = {
    "mp2": (1.0, 1.0),
    "scs-mp2": (6.0 / 5.0, 1.0 / 3.0),
    "sos-mp2": (1.3, 0.0),
}


@dataclass(frozen=True)
class MP2Result:
    """Closed-shell MP2 correlation energy and its spin components (Hartree)."""

    e_corr: float          # total correlation (after any spin-component scaling)
    e_os: float            # opposite-spin component (unscaled)
    e_ss: float            # same-spin component (unscaled)
    variant: str           # "mp2" | "scs-mp2" | "sos-mp2"

    @property
    def e_os_scaled(self) -> float:
        return _MP2_SCALES[self.variant][0] * self.e_os

    @property
    def e_ss_scaled(self) -> float:
        return _MP2_SCALES[self.variant][1] * self.e_ss


def mp2_energy(eps_occ, eps_vir, ovov, *, variant: str = "mp2") -> MP2Result:
    """Closed-shell (RHF/RKS-reference) MP2 correlation energy.

    Reference-agnostic: feed occupied/virtual MO energies and the ``(ia|jb)``
    integral block from any :class:`ERIProvider` (Gaussian or semiempirical).

    Spin decomposition (Grimme 2003), with ``D_{iajb}=e_i+e_j-e_a-e_b``:

        E_OS = S (ia|jb)^2 / D
        E_SS = S (ia|jb)[(ia|jb) - (ib|ja)] / D
        E_MP2 = E_OS + E_SS                 (= S (ia|jb)[2(ia|jb)-(ib|ja)]/D)

    ``variant`` scales the components: ``"scs-mp2"`` (6/5, 1/3), ``"sos-mp2"``
    (1.3, 0).  Returns an :class:`MP2Result` carrying both components so callers
    can report all variants from one evaluation.
    """
    if variant not in _MP2_SCALES:
        raise ValueError(f"unknown MP2 variant {variant!r}; "
                         f"choose from {sorted(_MP2_SCALES)}")
    eps_occ = np.asarray(eps_occ, float)
    eps_vir = np.asarray(eps_vir, float)
    ovov = np.asarray(ovov, float)            # (i, a, j, b) = (ia|jb)
    # Orbital-energy denominator D[i,a,j,b] = e_i + e_j - e_a - e_b.
    d = (eps_occ[:, None, None, None] + eps_occ[None, None, :, None]
         - eps_vir[None, :, None, None] - eps_vir[None, None, None, :])
    t = ovov / d                              # MP2 amplitudes x D⁻¹ carrier
    ibja = np.transpose(ovov, (0, 3, 2, 1))   # (ib|ja): map [i,a,j,b] <- [i,b,j,a]
    e_os = float(np.sum(ovov * t))            # S (ia|jb)^2 / D
    e_ss = float(np.sum((ovov - ibja) * t))   # S (ia|jb)[(ia|jb)-(ib|ja)] / D
    os_s, ss_s = _MP2_SCALES[variant]
    return MP2Result(e_corr=os_s * e_os + ss_s * e_ss,
                     e_os=e_os, e_ss=e_ss, variant=variant)


@dataclass(frozen=True)
class UMP2Result:
    """Unrestricted (UHF-reference) MP2 correlation energy, spin-channel resolved.

    ``e_corr`` is after any spin-component scaling; ``e_aa``/``e_bb``/``e_ab`` are
    the unscaled aa / bb / ab channels.
    """

    e_corr: float          # total (after any spin-component scaling)
    e_aa: float            # aa same-spin (unscaled)
    e_bb: float            # bb same-spin (unscaled)
    e_ab: float            # ab opposite-spin (unscaled)
    variant: str

    @property
    def e_ss(self) -> float:
        """Same-spin total aa + bb (unscaled)."""
        return self.e_aa + self.e_bb

    @property
    def e_os(self) -> float:
        """Opposite-spin ab (unscaled)."""
        return self.e_ab


def _ump2_same_spin(eps_occ, eps_vir, ovov) -> float:
    """1/4 S_{ijab} [(ia|jb)-(ib|ja)]^2 / D over one spin's occ/vir blocks.

    The 1/4 x full (unrestricted) sum equals the restricted S_{i<j,a<b}: the i=j
    and a=b diagonal terms vanish identically (``(ia|ib)=(ib|ia)``), and every
    distinct (i<j,a<b) quartet appears four times.
    """
    ovov = np.asarray(ovov, float)
    if ovov.size == 0:
        return 0.0
    eps_occ = np.asarray(eps_occ, float)
    eps_vir = np.asarray(eps_vir, float)
    d = (eps_occ[:, None, None, None] + eps_occ[None, None, :, None]
         - eps_vir[None, :, None, None] - eps_vir[None, None, None, :])
    anti = ovov - np.transpose(ovov, (0, 3, 2, 1))   # (ia|jb) - (ib|ja)
    return 0.25 * float(np.sum(anti * anti / d))


def _ump2_opp_spin(eps_occ_a, eps_vir_a, eps_occ_b, eps_vir_b, ovov) -> float:
    """S (ia|jb)^2 / D, bra = a(ia), ket = b(jb).  No antisymmetrisation."""
    ovov = np.asarray(ovov, float)
    if ovov.size == 0:
        return 0.0
    eoa = np.asarray(eps_occ_a, float)
    eva = np.asarray(eps_vir_a, float)
    eob = np.asarray(eps_occ_b, float)
    evb = np.asarray(eps_vir_b, float)
    d = (eoa[:, None, None, None] + eob[None, None, :, None]
         - eva[None, :, None, None] - evb[None, None, None, :])
    return float(np.sum(ovov * ovov / d))


def ump2_energy(eps_occ_a, eps_vir_a, eps_occ_b, eps_vir_b,
                ovov_aa, ovov_bb, ovov_ab, *, variant: str = "mp2") -> UMP2Result:
    """Unrestricted (UHF-reference) MP2 correlation energy, spin-channel resolved.

    Reference-agnostic spin-orbital UMP2 grouped by spin channel (the same
    decomposition as ``cpp/src/ump2.cpp``; Szabo-Ostlund Sec.6.7):

        E_aa = 1/4 S_{ijabina} [(ia|jb)-(ib|ja)]^2 / (e_i+e_j-e_a-e_b)
        E_bb = 1/4 S_{ijabinb} [(ia|jb)-(ib|ja)]^2 / (e_i+e_j-e_a-e_b)
        E_ab =   S_{iaina, jbinb} (ia|jb)^2 / (e_i^a+e_j^b-e_a^a-e_b^b)
        E_corr = c_os.E_ab + c_ss.(E_aa + E_bb)

    ``ovov_aa[i,a,j,b] = (ia|jb)`` over the a occ/vir MOs, ``ovov_bb`` over b, and
    ``ovov_ab[i,a,j,b]`` pairs a(ia) with b(jb).  ``variant`` scales the
    components with the same factors as the closed-shell :func:`mp2_energy`
    ("scs-mp2" = 6/5, 1/3; "sos-mp2" = 1.3, 0).  Reduces to RMP2 when the a and b
    references coincide.
    """
    if variant not in _MP2_SCALES:
        raise ValueError(f"unknown MP2 variant {variant!r}; "
                         f"choose from {sorted(_MP2_SCALES)}")
    e_aa = _ump2_same_spin(eps_occ_a, eps_vir_a, ovov_aa)
    e_bb = _ump2_same_spin(eps_occ_b, eps_vir_b, ovov_bb)
    e_ab = _ump2_opp_spin(eps_occ_a, eps_vir_a, eps_occ_b, eps_vir_b, ovov_ab)
    os_s, ss_s = _MP2_SCALES[variant]
    return UMP2Result(e_corr=os_s * e_ab + ss_s * (e_aa + e_bb),
                      e_aa=e_aa, e_bb=e_bb, e_ab=e_ab, variant=variant)


__all__ = ["ERIProvider", "MP2Result", "mp2_energy", "UMP2Result", "ump2_energy"]
