"""Selected matched BUPO electrostatics in bounded periodic RHF/UHF."""
from . import _vibeqc_core as _core
from .manta_periodic import _periodic_hf_configuration


def periodic_bupo_hf_provenance(result, mesh, options, python_bytes):
    r,p,s=result.receipt,result.plan,result.solver_snapshot
    fields=('provider','algorithm','boundary','source_identity','bupo_source_identity','overlap_identity',
            'kinetic_identity','nuclear_identity','nuclear_energy_identity','final_density_identity','final_fock_identity',
            'confirmed','electrostatic_accuracy_passed','accuracy_qualification','one_electron_energy','coulomb_energy',
            'exchange_energy','uncorrected_exchange_energy','exchange_correction_energy','nuclear_energy','energy',
            'exchange_q0','exchange_correction_identity','madelung','confirmation_fock_error','confirmation_energy_error',
            'maximum_electrostatic_energy_error','maximum_electrostatic_matrix_error',
            'maximum_observed_electrostatic_energy_error','maximum_observed_electrostatic_matrix_error','shell_calls','primitive_pairs','fock_calls')
    evidence={name:getattr(r,name) for name in fields}
    evidence.update({name:getattr(r,name) for name in ('maximum_overlap_image_error','maximum_kinetic_image_error',
        'maximum_kinetic_energy_error','kinetic_energy_error_bound','maximum_observed_kinetic_energy_error','one_electron_accuracy_passed')})
    evidence['one_electron_images']={name:getattr(result.one_electron_images.receipt,name) for name in (
        'qualification','source_identity','pair_source_identity','cube_radius','cube_cells','real_metric_lower',
        'maximum_overlap_error','maximum_kinetic_error','roundoff_included')}
    evidence.update({name:getattr(r,name) for name in ('maximum_exchange_matrix_error','maximum_exchange_energy_error',
        'exchange_matrix_error_bound','exchange_energy_error_bound','maximum_observed_exchange_matrix_error',
        'maximum_observed_exchange_energy_error','exchange_correction_matrix_error','exchange_correction_energy_error',
        'exchange_accuracy_passed')})
    evidence['bound_exchange']=options.bound_exchange
    evidence['exchange_bound_cube_radius']=r.exchange_bound_cube_radius
    evidence.update({name:getattr(r,name) for name in ('domain_selection','domain_identity','reciprocal_energy_cutoff',
        'domain_translations','domain_exchange_vectors','domain_conjugacy_vectors')})
    evidence.update({name:getattr(p,name) for name in ('domain_live_bytes','domain_scalar_terms','domain_kernel_terms')})
    evidence.update({name:getattr(p,name) for name in ('source_live_bytes','source_shell_calls','source_primitive_pairs',
        'source_scalar_terms','source_kernel_terms')})
    evidence['preparation'] = None if r.preparation is None else {name:getattr(r.preparation,name) for name in (
        'identity','pair_source_identity','tree_identity','kernel_identity','electronic_source_identity',
        'pair_cube_radius','real_cube_radius','reciprocal_cube_radius','near_cube_radius','pairs','nodes','leaf_capacity',
        'order','branching','maximum_depth','eta','opening','integral_tolerance','pair_cutoff_bohr',
        'pointwise_tolerance','tail_mass_tolerance','minimum_radius','maximum_radius')}
    evidence['exchange_bounds']=None if r.exchange_bounds is None else {name:getattr(r.exchange_bounds,name) for name in (
        'qualification','source_identity','pair_source_identity','density_identity','fourier_decay','maximum_error','exchange_scale','energy_error_bounds','roundoff_included')}
    es=r.electrostatic
    evidence['electrostatic']={name:getattr(es,name) for name in (
        'source_identity','electronic_source_identity','pair_source_identity','kernel_identity','point_source_identity','density_identity',
        'qualification','boundary','charge_policy','charge_tolerance','nuclear_charge','electron_charge','retained_net_charge',
        'charge_uncertainty_bound','maximum_net_charge_bound','background_charge','nuclear_real','nuclear_reciprocal',
        'nuclear_self','nuclear_gauge','nuclear_energy','attraction_real','attraction_reciprocal','attraction_gauge',
        'attraction_energy','electronic_energy','electronic_gauge','total_gauge','energy','trace_energy',
        'nuclear_domain_error_bound','attraction_domain_error_bound','attraction_image_error_bound',
        'energy_error_bound','maximum_j_error_bound','roundoff_included')}
    evidence['partition']={name:getattr(r.coulomb.partition,name) for name in ('identity','direct_blocks','approximate_blocks','direct_pairs','approximate_pairs')}
    evidence['exchange']={name:getattr(r.exchange,name) for name in (
        'algorithm','boundary','convention','kernel','omega','source_identity','density_identity','transfer_vectors','zero_vectors')}
    methods=['manta_periodic_bupo_hf','manta_periodic_one_electron_images']
    if options.bound_exchange:methods.append('manta_periodic_exchange_bounds')
    if r.exchange_q0=='BvK-ewald':methods.append('manta_periodic_madelung')
    evidence.update(schema_version=1,citation_methods=methods,converged=result.converged,status=s.status.name,
        iterations=s.evaluated_iteration,operator_kind='fock',operator_generation=s.evaluated_iteration,
        basis_frame_identity=r.overlap_identity,operator_layout='spin,k,ao,ao',
        spin_labels=['restricted-total'] if p.spins==1 else ['alpha','beta'],spin_electrons_per_cell=[p.n_alpha,p.n_beta],
        occupation_policy='fixed-spin-integer-uniform-full-mesh',symmetry_reduced=False,
        coulomb_algorithm='bupo',exchange_algorithm='exact-reciprocal',
        energy_convention='nuclear + one_electron + coulomb + exchange; exchange includes the declared zero-mode correction',
        mesh=list(mesh.mesh),is_shift=list(mesh.is_shift),retained_rank=result.solver_memory.retained_rank,
        commutator_frobenius_rms=s.commutator_frobenius_rms,
        maximum_density_closure_frobenius=s.maximum_density_closure_frobenius,
        maximum_electron_count_error=s.maximum_electron_count_error,global_band_gap=s.global_band_gap,
        python_reserved_bytes=python_bytes,modeled_peak_bytes=p.modeled_peak_bytes,
        scalar_terms=p.scalar_terms,kernel_terms=p.kernel_terms,scf_history_steps=s.evaluated_iteration,
        scf_history_capacity=p.history_capacity,scf_history_reserved_bytes=p.history_live_bytes)
    if p.spins==2:evidence['spin_band_gaps']=[gap if n else None for gap,n in zip(s.spin_band_gap,(p.n_alpha,p.n_beta))]
    return evidence


def run_periodic_bupo_hf(electronic, system, mesh, exchange_vectors, transfer_offsets, *, options=None, integral_engine):
    """Run native RHF/UHF with matched BUPO electrostatics and explicit exact K.

    Every evaluated density must meet electrostatic and S/T image targets.
    Optional exchange certification includes full AO and reciprocal images
    at fixed mesh/Madelung coefficient, excluding basis, SCF-state and roundoff
    errors. The electronic source's original AO images, lattice and mesh are
    retained; neither a new partition nor symmetry reduction is inferred.
    """
    from .output import record
    options,budget,python_bytes=_periodic_hf_configuration(options,integral_engine,'BupoHF')
    # Nested source ledgers and scalar receipts can coexist with output copies.
    extra=32768
    if budget.other_live_bytes+extra>budget.memory_bytes:raise ValueError('MANTA BUPO HF Python provenance memory budget exceeded')
    budget.other_live_bytes+=extra;python_bytes+=extra
    result=_core._manta.run_periodic_bupo_hf(electronic,system,mesh,exchange_vectors,transfer_offsets,options,budget)
    evidence=periodic_bupo_hf_provenance(result,mesh,options,python_bytes)
    accuracy='electrostatic, S/T image and exchange targets passed' if options.bound_exchange else 'electrostatic and S/T image targets passed'
    record('manta_periodic_bupo_hf',f'  Periodic BUPO {"RHF" if result.plan.spins==1 else "UHF"}: manta; '
        f'converged={result.converged}; E={result.receipt.energy:.12f}; {accuracy}.\n',integral_provenance=evidence)
    return result


def run_periodic_bupo_hf_from_cutoff(electronic, system, mesh, reciprocal_energy_cutoff, *, options=None, integral_engine):
    """Select certified reciprocal spheres for an existing BUPO RHF/UHF source.

    The cutoff is a reciprocal kinetic energy in hartree. Every transfer uses
    the shared certified sphere predicate and an audited opposite support.
    AO images, Ewald domains and BUPO partition remain source-owned. Full
    exchange bounds are required at every evaluated density; insufficient
    cutoffs fail the declared targets rather than silently enlarging support.

    Supplied options require bound_exchange=True and a zero input
    exchange_bound_cube_radius; the factory derives that completion cube.
    None selects these settings with the ordinary HF defaults. No automatic
    whole-method accuracy profile or symmetry admission is implied.
    """
    from .output import record
    if options is None:
        options = _core._manta.PeriodicBupoHFOptions()
        options.bound_exchange = True
    options, budget, python_bytes = _periodic_hf_configuration(options, integral_engine, 'BupoHF')
    extra = 32768
    if budget.other_live_bytes+extra > budget.memory_bytes:
        raise ValueError('MANTA BUPO HF Python provenance memory budget exceeded')
    budget.other_live_bytes += extra
    python_bytes += extra
    result = _core._manta.run_periodic_bupo_hf_from_cutoff(
        electronic, system, mesh, reciprocal_energy_cutoff, options, budget)
    evidence = periodic_bupo_hf_provenance(result, mesh, options, python_bytes)
    record('manta_periodic_bupo_hf', f'  Periodic BUPO {"RHF" if result.plan.spins == 1 else "UHF"}: manta; '
           f'converged={result.converged}; E={result.receipt.energy:.12f}; '
           'electrostatic, S/T image and cutoff-selected exchange targets passed.\n', integral_provenance=evidence)
    return result


def run_periodic_bupo_hf_from_cubes(basis, system, mesh, reciprocal_energy_cutoff, *, source_options=None,
                                    options=None, integral_engine):
    """Construct the complete native BUPO source and run bounded RHF/UHF.

    Source options declare finite integer AO/Ewald/near cubes, the physical
    AO pair cutoff, extents and BUPO partition policy. Preparation charges
    each stage against the cumulative budget before it executes, then admits
    the exchange sphere and complete SCF. All source/exchange bounds are
    mandatory; no source domain is adapted during SCF. This is explicit
    source construction, not automatic whole-method accuracy selection.
    """
    from .output import record
    from .periodic_runner import _resolve_ecp_data
    if options is None:
        options = _core._manta.PeriodicBupoHFOptions()
        options.bound_exchange = True
    options, budget, python_bytes = _periodic_hf_configuration(options, integral_engine, 'BupoHF')
    if source_options is None:
        source_options = _core._manta.PeriodicBupoHFSourceOptions()
    if not isinstance(source_options, _core._manta.PeriodicBupoHFSourceOptions):
        raise TypeError('source_options must be manta.PeriodicBupoHFSourceOptions')
    if not isinstance(basis, _core.BasisSet) or not isinstance(system, _core.PeriodicSystem):
        raise TypeError('BUPO HF preparation requires BasisSet and PeriodicSystem objects')
    if _resolve_ecp_data(system, basis)[3]:
        raise NotImplementedError('BUPO HF preparation currently requires an all-electron basis')
    extra = 49152
    if budget.other_live_bytes+extra > budget.memory_bytes:
        raise ValueError('MANTA BUPO HF preparation provenance memory budget exceeded')
    budget.other_live_bytes += extra
    python_bytes += extra
    prepared = _core._manta.prepare_periodic_bupo_hf(basis, system, mesh, reciprocal_energy_cutoff,
                                                   source_options, options, budget)
    result = prepared.run()
    evidence = periodic_bupo_hf_provenance(result, mesh, prepared.options, python_bytes)
    record('manta_periodic_bupo_hf', f'  Periodic BUPO {"RHF" if result.plan.spins == 1 else "UHF"}: manta; '
           f'converged={result.converged}; E={result.receipt.energy:.12f}; '
           'native source construction and all evaluated-density integral targets passed.\n', integral_provenance=evidence)
    return result
