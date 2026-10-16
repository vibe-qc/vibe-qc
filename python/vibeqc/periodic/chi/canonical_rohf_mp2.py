"""Experimental same-source Knowles RMP2 on a stationary chi ROHF state.

Knowles, Andrews, Amos, Handy and Pople, Chem. Phys. Lett. 186, 130 (1991),
doi:10.1016/S0009-2614(91)85118-G, Sec. 2, Eqs. (6)-(16). This is RMP2,
not the different AAHK ROMP partition. The spin-dependent semicanonical
rotations preserve the ROHF determinant and its spin densities; they are not
an unrestricted SCF optimization. Retained chi factors are only transported.
"""
from dataclasses import dataclass
import json
import numbers

import numpy as np

from .canonical_mp2 import (CanonicalFactorAdapter, CanonicalMP2Result, _contract,
                           _double_work_counts, _persistent_contract, _positive_integer)


ROHF_RMP2_METHOD = 'ROHF-RMP2'
RMP2_CANONICALIZATION = 'knowles-1991-spin-semicanonical/v1'
RMP2_PARTITION = 'H0=sum(spin-semicanonical epsilon_p a_p^dagger a_p); V=H-H0'
RMP2_FROZEN_CORE = 'shared-lowest-closed-average-fock-projector/active-spin-semicanonical/v1'


@dataclass(frozen=True)
class CanonicalROHFMP2Result(CanonicalMP2Result):
    e_singles_alpha_per_cell: float = 0.
    e_singles_beta_per_cell: float = 0.
    reference_method: str = 'ROHF'
    canonicalization: str = RMP2_CANONICALIZATION
    perturbation_partition: str = RMP2_PARTITION
    frozen_core_convention: str = RMP2_FROZEN_CORE
    canonicalization_identity: str = ''
    normalization: str = 'energy per primitive cell; uniform characters; singles Nk^-1, doubles Nk^-3'

    @property
    def e_singles_per_cell(self):
        return self.e_singles_alpha_per_cell + self.e_singles_beta_per_cell

    @property
    def e_doubles_per_cell(self):
        return self.e_aa_per_cell + self.e_bb_per_cell + self.e_ab_per_cell

    @property
    def e_corr_per_cell(self):
        return self.e_singles_per_cell + self.e_doubles_per_cell


def _max_abs(array):
    return float(np.max(abs(array), initial=0.))


def _rohf_state(hf, tolerance=1e-9):
    """Independently enforce shared orbitals and ROHF, not UHF, stationarity.

    Knowles Eqs. (6)-(8): (Fa+Fb)_cv=0, Fb_co=0, Fa_ov=0. Individual
    spin occupied-virtual Fock blocks need not vanish and must not be erased.
    """
    c, s, occ, fa, fb, da, db = (np.asarray(getattr(hf, key)) for key in
        ('mo_coeffs', 'overlap', 'occupations', 'fock_alpha', 'fock_beta',
         'density_alpha', 'density_beta'))
    if (c.ndim != 3 or occ.shape != (len(c), c.shape[-1])
            or s.shape != (len(c), c.shape[1], c.shape[1])
            or any(x.shape != s.shape for x in (fa, fb, da, db))
            or any(not np.isfinite(x).all() for x in (c, s, occ, fa, fb, da, db))):
        raise ValueError('invalid shared ROHF state dimensions or values')
    nc, no = getattr(hf, 'n_closed', None), getattr(hf, 'n_open', None)
    if any(isinstance(n, bool) or not isinstance(n, numbers.Integral) or n < 0 for n in (nc, no)):
        raise ValueError('invalid ROHF closed/open subspace ranks')
    na, nb, nmo = int(nc+no), int(nc), c.shape[-1]
    if na > nmo or getattr(hf, 'n_alpha', None) != na or getattr(hf, 'n_beta', None) != nb:
        raise ValueError('inconsistent ROHF spin/subspace ranks')
    expected = np.zeros_like(occ); expected[:, :nb] = 2; expected[:, nb:na] = 1
    if not np.allclose(occ, expected, atol=1e-12, rtol=0):
        raise ValueError('ROHF needs integer closed/open/virtual occupations')
    subspaces = dict(closed=tuple(range(nb)), open=tuple(range(nb, na)), virtual=tuple(range(na, nmo)))
    if getattr(hf, 'rohf_subspaces', None) != subspaces:
        raise ValueError('ROHF subspace identity mismatch')
    if (getattr(hf, 'rohf_convention', None) != 'coulson-roothaan-average-diagonal/v1'
            or getattr(hf, 'rohf_canonicalization', None) != 'closed-open-virtual-average-fock/v1'):
        raise ValueError('unsupported ROHF effective-Fock/canonicalization convention')
    errors = [_max_abs(c.conj().transpose(0, 2, 1) @ s @ c - np.eye(nmo))]
    fmo = []
    for f, d, n in ((fa, da, na), (fb, db, nb)):
        errors += [_max_abs(f-f.conj().transpose(0, 2, 1)),
                   _max_abs(c[:, :, :n] @ c[:, :, :n].conj().transpose(0, 2, 1)-d)]
        fmo.append(c.conj().transpose(0, 2, 1) @ f @ c)
    a, b = fmo
    errors += [_max_abs((a+b)[:, :nb, na:]), _max_abs(b[:, :nb, nb:na]),
               _max_abs(a[:, nb:na, na:])]
    if max(errors) > tolerance:
        raise ValueError('reference fails shared-spatial ROHF density or generalized stationarity')
    return c, (fa, fb), (na, nb)


def _semicanonical_channels(c, focks, counts, frozen, denominator_tolerance, *, with_singles=True):
    """Knowles Sec. 2 steps (1)-(5), with a fixed shared frozen projector.

    With no frozen core this is exactly Fa:(c+o),v and Fb:c,(o+v).
    With a core, first select the lowest eigenvectors of (Fa+Fb)/2 in c,
    discard that same projector for both spins, and diagonalize each remaining
    spin occupied/virtual block. This explicit projected-space extension does
    not freeze different alpha/beta semicanonical orbitals. It is invariant to
    rotations inside c/o/v, including rotations of the supplied closed frame;
    a degenerate core boundary is refused because its projector is not unique.
    """
    nc = counts[1]
    active = np.array(c, dtype=complex, copy=True)
    if frozen:
        for k in range(len(c)):
            closed = c[k, :, :nc]
            fc = closed.conj().T @ ((focks[0][k]+focks[1][k])*.5) @ closed
            e, u = np.linalg.eigh(fc)
            if frozen < nc and e[frozen]-e[frozen-1] <= denominator_tolerance:
                raise ValueError('frozen-core boundary cuts a degenerate closed subspace')
            active[k, :, :nc] = closed @ u
        active = active[:, :, frozen:]
    channels, singles = [], []
    for f, count in zip(focks, counts):
        n = count-frozen
        transformed = np.empty_like(active)
        energies = np.empty((len(c), active.shape[-1]))
        for k in range(len(c)):
            for block in (slice(0, n), slice(n, active.shape[-1])):
                x = active[k, :, block]
                e, u = np.linalg.eigh(x.conj().T @ f[k] @ x)
                transformed[k, :, block] = x @ u
                energies[k, block] = e
        # Global spin gaps preclude unsupported inter-character zero gaps as
        # well as all single and double intruder denominators. No level shifts.
        if n and n < active.shape[-1] and np.min(energies[:, n:])-np.max(energies[:, :n]) <= denominator_tolerance:
            raise ValueError('RMP2 spin occupied/virtual gap is nonpositive or too small')
        channels.append((transformed, energies, n))
        if with_singles:
            singles.append(_single_term(channels[-1], f, denominator_tolerance))
    return tuple(channels), tuple(singles)


def _single_term(channel, fock, denominator_tolerance):
    """Eq. (16): diagonal in momentum, hence one character sum / Nk."""
    transformed, energies, n = channel
    fov = transformed[:, :, :n].conj().transpose(0, 2, 1) @ fock @ transformed[:, :, n:]
    denominator = energies[:, :n, None]-energies[:, None, n:]
    if not np.isfinite(denominator).all() or np.any(denominator >= -denominator_tolerance):
        raise ValueError('RMP2 singles denominator is nonnegative or too small')
    return float(np.sum(abs(fov)**2/denominator).real/len(transformed))


def evaluate_same_source_rohf_mp2(reference, *, frozen_core=0, auxiliary_batch_size=64,
                                 denominator_tolerance=1e-12, checkpoint_path=None,
                                 resume_from=None, checkpoint_interval=1):
    """Evaluate explicitly named Knowles ROHF-RMP2, retaining its singles.

    All-electron or scalar-ECP uncorrected full-Coulomb classic chi RI is admitted.
    ``frozen_core`` is a closed spatial count (or two identical spin counts),
    with the projector/active-space convention recorded in the result.
    Optional persistent work includes separate alpha/beta singles and
    deterministic aa/bb/ab doubles, with no replay of completed contributions.
    """
    from .reference import ChiHFReference, _digest
    if not isinstance(reference, ChiHFReference):
        raise TypeError('same-source ROHF RMP2 requires a sealed ChiHFReference')
    if reference.method != 'ROHF':
        raise ValueError('Knowles ROHF-RMP2 requires a genuine ROHF reference')
    hf = reference.validate(reference.request_identity, 'ROHF')
    assembly = getattr(hf, 'exact_exchange_assembly', None)
    if (getattr(assembly, 'c_full', None) != 1. or getattr(assembly, 'c_sr', None) != 0.
            or getattr(hf.aiccm2026dev_b, 'backend', None) != 'ri'
            or getattr(hf, 'returned_operator', 'iterated') != 'iterated'):
        raise NotImplementedError('ROHF RMP2 requires uncorrected full-exchange RI HF')
    if getattr(hf, 'smearing_temperature', 0) != 0:
        raise ValueError('ROHF RMP2 does not support smearing')
    from .ecp import validate_ecp_reference
    validate_ecp_reference(hf)
    if getattr(hf, 'executed_chi_xc', None) is not None:
        raise NotImplementedError('ROHF RMP2 does not support KS/hybrid references')
    report = hf.chi_scf_acceptance
    for key, limit in (('energy_residual', 1e-11), ('energy_consistency_error', 1e-11),
                       ('generalized_rohf_gradient_norm', 1e-9), ('density_refill_residual', 1e-9)):
        value = report.get(key, np.inf)
        if not np.isfinite(value) or not 0 <= value <= limit:
            raise ValueError('RMP2 needs achieved tight generalized ROHF stationarity')
    adapter = CanonicalFactorAdapter(hf.fitted_operator)
    source = hf.executed_fit_source
    if source.source != 'classic' or adapter.operator.descriptor.source != 'classic':
        raise NotImplementedError('ROHF RMP2 supports classic only; SR remains gated')
    if json.loads(source.executed_domains_json).get('fitted_operator_identity') != adapter.source_identity:
        raise ValueError('ROHF and RMP2 source identities differ')
    if (not np.allclose(hf.kpoints_cart, adapter.operator.descriptor.kpoints_cart, atol=1e-14, rtol=0)
            or not np.allclose(hf.kpoint_weights, np.full(adapter.nk, 1/adapter.nk), atol=1e-14, rtol=0)):
        raise ValueError('ROHF and RMP2 characters or normalization differ')
    batch = _positive_integer(auxiliary_batch_size, 'auxiliary_batch_size')
    if (isinstance(denominator_tolerance, bool) or not isinstance(denominator_tolerance, numbers.Real)
            or not np.isfinite(denominator_tolerance) or denominator_tolerance <= 0):
        raise ValueError('denominator_tolerance must be finite and positive')
    c, focks, counts = _rohf_state(hf)
    if isinstance(frozen_core, numbers.Integral):
        frozen = (frozen_core, frozen_core)
    else:
        try:
            frozen = tuple(frozen_core)
        except TypeError as exc:
            raise ValueError('frozen core needs an integer closed spatial count') from exc
    if (len(frozen) != 2 or any(isinstance(n, bool) or not isinstance(n, numbers.Integral)
                              or n < 0 or n > counts[1] for n in frozen)
            or frozen[0] != frozen[1]):
        raise ValueError('ROHF frozen cores must agree and fit in the closed subspace')
    core = int(frozen[0])
    persistent = checkpoint_path is not None or resume_from is not None
    channels, singles = _semicanonical_channels(c, focks, counts, core, denominator_tolerance,
                                               with_singles=not persistent)
    # Eq. (16) spin sum gives 1/4 for ordered aa/bb and 1 for ab; retained
    # primitive-cell factors supply two 1/Nk factors plus energy per cell 1/Nk.
    restart = {}
    if persistent:
        doubles, singles, restart = _persistent_contract(reference, adapter, channels, (0, 0),
            batch, denominator_tolerance, checkpoint_path=checkpoint_path, resume_from=resume_from,
            checkpoint_interval=checkpoint_interval, method=ROHF_RMP2_METHOD,
            conventions=(RMP2_CANONICALIZATION, RMP2_PARTITION, RMP2_FROZEN_CORE, core),
            singles_term=lambda spin: _single_term(channels[spin], focks[spin], denominator_tolerance))
    else:
        doubles = _contract(adapter, channels, (0, 0), batch, denominator_tolerance)
        total = 2+sum(_double_work_counts(adapter, channels, (0, 0)))
        restart = dict(completed_work_units=total, total_work_units=total)
    # This is a portable frame identity, separate from the live process seal.
    frame_identity = _digest((adapter.source_identity,
        RMP2_CANONICALIZATION, RMP2_PARTITION, RMP2_FROZEN_CORE, core, channels))
    adapter.validate()
    reference.validate(reference.request_identity, 'ROHF')
    from .ecp import electron_accounting
    accounting = electron_accounting(hf, (core, core), counts)
    return CanonicalROHFMP2Result(
        e_hf_per_cell=float(hf.energy), e_aa_per_cell=doubles[0], e_bb_per_cell=doubles[1],
        e_ab_per_cell=doubles[2], active_spaces=tuple((core, n, x.shape[-1]-n) for x, e, n in channels),
        source_identity=adapter.source_identity, reference_identity=reference.state_identity,
        retained_metric_ranks=adapter.retained_ranks, method=ROHF_RMP2_METHOD,
        source_descriptor_json=json.dumps(source.as_dict(), sort_keys=True),
        scf_acceptance_json=json.dumps(dict(report), sort_keys=True),
        e_singles_alpha_per_cell=singles[0], e_singles_beta_per_cell=singles[1],
        canonicalization_identity=frame_identity,
        ecp_json=json.dumps(getattr(hf, 'chi_ecp', None) or {}, sort_keys=True),
        electron_counts_json=json.dumps(accounting, sort_keys=True), **restart)
