"""Transport the executed HF exchange constant into canonical CC integrals.

Xing and Lin, Phys. Rev. X 14, 011059 (2024), Sec. II B, Eqs. (7)-(11),
doi:10.1103/PhysRevX.14.011059. Their signed xi is minus our Madelung.
The correction belongs to both orbital energies and general ERI blocks;
occupied-virtual MP2 numerators alone have no diagonal zero-mode term.
"""
import numpy as np


CONVENTION = 'bvk-ewald-diagonal-eri/v1'
ENERGY_ONLY = 'bvk-ewald-hf-energy-only/v1'


def reference_madelung(hf, channels):
    """Recover and verify the actual scalar seam, without an integral rebuild.

    HF retains G=0-omitted factors and F_sigma=h+J-K_sigma-xi S D_sigma S.
    All matrix elements in both spin channels must support the same xi. This
    also supports a restored reference without choosing new Ewald cutoffs.
    The sealed reference binds every input; no fitted factor is changed.
    """
    s = np.asarray(hf.overlap)
    h = np.asarray(hf.hcore)
    densities = tuple(c[:, :, :n] @ c[:, :, :n].conj().transpose(0, 2, 1)
                      for c, _, n in channels)
    op = hf.fitted_operator
    j = np.asarray(op.coulomb(densities[0]+densities[1]))
    seams = tuple(s @ d @ s for d in densities)
    # RHF has one spin-independent Fock, UHF has two actual physical Focks.
    focks = ((hf.fock, hf.fock) if hasattr(hf, 'fock') and
             not hasattr(hf, 'fock_alpha') else (hf.fock_alpha, hf.fock_beta))
    differences = tuple(h+j-op.exchange(d)-f for d, f in zip(densities, focks))
    norm = sum(np.vdot(x, x).real for x in seams)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('CC reference has no finite occupied zero-mode projector')
    xi = float(sum(np.vdot(x, y).real for x, y in zip(seams, differences))/norm)
    if (not np.isfinite(xi) or any(not np.isfinite(y).all() or
            np.max(abs(y-xi*x)) > 1e-9 for x, y in zip(seams, differences))):
        raise ValueError('CC reference Focks do not share the retained-source scalar zero mode')
    return xi
