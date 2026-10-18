"""Representative finite-torus pair contractions for restricted local MP2.

These are residual and energy primitives, not an iterative solver or an
automatic production route. Pair coefficients and translated overlaps come
from :class:`~vibeqc.dlpno.translation.TranslationPairSpaces`. The residual
keeps each coupled amplitude in its own pair space until the two overlap
projections, as in Nejad et al., J. Chem. Phys. 163, 214107 (2025),
doi:10.1063/5.0290816, Eqs. (42)-(44). No pair screening, discarded-PNO
correction, residual projection or convergence decision is applied here.
"""

from __future__ import annotations

import math

import numpy as np

from .translation import TranslationPairSpaces


def _validated_tensors(spaces, integrals, amplitudes):
    if not isinstance(spaces, TranslationPairSpaces):
        raise TypeError("spaces must be a TranslationPairSpaces instance")
    return (
        spaces.validate_tensors(integrals, name="integrals"),
        spaces.validate_tensors(amplitudes, name="amplitudes"),
    )


def translation_mp2_residual(spaces, integrals, amplitudes):
    """Return one raw LMP2 residual in each representative pair's basis.

    ``integrals`` contains ``K_ij[a,b] = (ia|jb)``; ``amplitudes`` contains
    the corresponding ``T_ij``. Both sequences follow the representative
    order of ``spaces``. Every placed occupied ``k`` contributes to each
    residual, including the two occupied-Fock diagonal terms. No pair
    multiplicity belongs in this equation; those factors enter the energy.

    The virtual Fock blocks need not be diagonal. Transposing a reversed
    source pair acts on its amplitude axes, never on its target/source
    overlap. Input arrays are read-only to this function, including on a
    validation or arithmetic failure.
    """
    integrals, amplitudes = _validated_tensors(spaces, integrals, amplitudes)
    occupied_fock = spaces.occupied_fock
    residuals = []
    with np.errstate(over="raise", invalid="raise"):
        for row, (i, j) in enumerate(spaces.representative_pairs):
            target = (i, j)
            virtual_fock = spaces.fock_between(target, target)
            amplitude = amplitudes[row]
            residual = (
                integrals[row]
                + virtual_fock @ amplitude
                + amplitude @ virtual_fock
            )
            for k in range(spaces.plan.n_occupied):
                overlap = spaces.overlap_between(target, (k, j))
                source = spaces.orient_tensor(k, j, amplitudes)
                residual -= occupied_fock[i, k] * (overlap @ source @ overlap.T)
                overlap = spaces.overlap_between(target, (i, k))
                source = spaces.orient_tensor(i, k, amplitudes)
                residual -= occupied_fock[k, j] * (overlap @ source @ overlap.T)
            if not np.isfinite(residual).all():
                raise FloatingPointError("non-finite translation-pair MP2 residual")
            residuals.append(residual)
    return tuple(residuals)


def translation_pair_energy_per_cell(spaces, integrals, amplitudes):
    """Contract the restricted pair energy once per translation orbit.

    ``spaces.energy_weights`` includes both the unordered-pair spin factor
    and the orbit multiplicity divided by the number of primitive cells.
    In particular, a same-orbital nonzero self-inverse translation has
    weight one, not two. The result is already per primitive cell.
    """
    integrals, amplitudes = _validated_tensors(spaces, integrals, amplitudes)
    with np.errstate(over="raise", invalid="raise"):
        terms = [
            float(weight) * float(np.sum(exchange * (2.0 * amplitude - amplitude.T)))
            for weight, exchange, amplitude in zip(
                spaces.energy_weights, integrals, amplitudes
            )
        ]
        energy = math.fsum(terms)
    if not math.isfinite(energy):
        raise FloatingPointError("non-finite translation-pair correlation energy")
    return energy
