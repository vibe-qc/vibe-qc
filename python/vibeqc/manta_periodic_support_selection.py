"""Selection bookkeeping; scientific support decisions use native producers."""
from . import _vibeqc_core as core


def support_request(domains, options, source_plan, *, ks, hybrid, unrestricted):
    request = core._manta.PeriodicSupportRequest()
    physical = options.ks if ks else options
    for name in ('alpha', 'nuclear_cutoff_bohr', 'pair_cutoff_bohr', 'scf'):
        setattr(request, name, getattr(physical, name))
    request.reciprocal_energy_cutoff = domains.reciprocal_energy_cutoff
    request.one_electron_accuracy = options.one_electron_accuracy
    request.nuclear_accuracy = options.nuclear_accuracy
    request.exchange_accuracy = options.exchange_accuracy
    if options.exchange_accuracy.matrix_error > 0:
        coefficient = abs(source_plan.exact_exchange_fraction) if ks else 1.
        request.exchange_scale = coefficient / (1 if unrestricted else 2)
        for name in ('exchange_q0', 'madelung_alpha', 'madelung_tolerance'):
            setattr(request, name, getattr(options, name))
    else:
        request.exchange_scale = 0.
        request.exchange_q0 = 'strict-zero-mode'
    return request


def support_selection_provenance(result):
    evidence = {name: getattr(result, name) for name in (
        'qualified', 'termination', 'certificate_identity', 'retained_bytes', 'modeled_peak_bytes',
        'preparation_scalar_terms', 'preparation_kernel_terms',
        'shell_calls', 'primitive_pairs', 'scalar_terms', 'kernel_terms')}
    evidence['qualification'] = 'selected-physical-supports-for-requested-component-targets-before-scf'
    evidence['work_accounting'] = 'sum-of-admitted-trial-upper-bounds-including-discarded-trials-and-preparation'
    evidence['excludes'] = ['finite-ri-fitting-and-metric', 'integral-roundoff', 'overlap-rank-stability',
                            'density-relaxation', 'basis-and-k-convergence', 'xc-quadrature']
    evidence['policy'] = {name: getattr(result.policy, name) for name in (
        'maximum_attempts', 'maximum_completion_cube_radius', 'growth_factor',
        'maximum_pair_cutoff_bohr', 'maximum_nuclear_cutoff_bohr', 'maximum_reciprocal_energy_cutoff')}
    evidence['trials'] = [{name: getattr(t, name) for name in (
        'qualified', 'reciprocal_energy_cutoff', 'pair_cutoff_bohr', 'nuclear_cutoff_bohr',
        'ao_cube_radius', 'real_cube_radius', 'reciprocal_cube_radius',
        'overlap_error', 'kinetic_error', 'kinetic_energy_error', 'nuclear_matrix_error',
        'attraction_energy_error', 'nuclear_energy_error', 'exchange_matrix_error', 'exchange_energy_error',
        'grow_pair', 'grow_nuclear', 'grow_reciprocal', 'overlap_identity', 'nuclear_identity',
        'one_electron_certificate', 'nuclear_certificate', 'exchange_certificate', 'density_envelope_identity',
        'modeled_peak_bytes', 'shell_calls', 'primitive_pairs', 'scalar_terms', 'kernel_terms')}
        for t in result.trials]
    evidence['exchange_certified'] = bool(result.trials and result.trials[-1].exchange_certificate)
    if not evidence['exchange_certified']:
        evidence['excludes'].append('exchange')
    return evidence


def verify_selected_supports(native, selection):
    final = selection.trials[-1]
    receipt = native.receipt
    checks = ((receipt.overlap_identity, final.overlap_identity),
              (receipt.nuclear_identity, final.nuclear_identity),
              (receipt.one_electron_accuracy.certificate_identity, final.one_electron_certificate),
              (receipt.nuclear_accuracy.certificate_identity, final.nuclear_certificate))
    if any(actual != expected for actual, expected in checks):
        raise RuntimeError('MANTA final component certificates differ from selected physical supports')
    if final.exchange_certificate and (receipt.exchange_accuracy is None or
            receipt.exchange_accuracy.certificate_identity != final.exchange_certificate):
        raise RuntimeError('MANTA final exchange certificate differs from selected physical supports')
