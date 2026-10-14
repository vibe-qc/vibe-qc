"""Output-layer record for experimental same-source chi correlation."""
import json


def _restart_record(result):
    """Portable execution history; checkpoint paths are never provenance."""
    fields = ('restart_mode', 'resumed_work_units', 'completed_work_units',
              'total_work_units', 'restart_identity', 'resumed_cc_iterations',
              'resumed_triples_work', 'triples_work_completed',
              'triples_work_total', 'diis_restart_policy')
    return {key: getattr(result, key) for key in fields if hasattr(result, key)}


def _ecp_citations():
    from .citations import load_default_database
    entries = load_default_database().entries()
    return [dict(key=key, title=entries[key].title, doi=entries[key].doi)
            for key in ('shaw_gilbert_libecpint', 'dolg_cao_pseudopotentials_2011')]


def _ecp_correlation_fields(result):
    """Carry the solver's ECP and explicit active-space accounting verbatim."""
    fields = {}
    ecp = json.loads(getattr(result, 'ecp_json', '{}'))
    counts = json.loads(getattr(result, 'electron_counts_json', '{}'))
    if ecp:
        fields['ecp'] = ecp
        fields['ecp_citations'] = _ecp_citations()
    if counts:
        fields['electron_counts'] = counts
    return fields


def chi_ecp_record(result):
    """Portable executed scalar Hamiltonian; no reconstruction from a keyword."""
    record = getattr(result, 'chi_ecp', None)
    if record is None:
        return None
    if (not isinstance(record, dict)
            or record.get('schema') != 'vibeqc.chi.scalar-ecp/v1'
            or not record.get('identity')
            or 'executed_support' not in record or 'energy_components' not in record):
        raise ValueError('chi ECP output requires the complete executed scalar Hamiltonian')
    # A plain JSON copy keeps output records detached from the mutable result.
    return json.loads(json.dumps(record, allow_nan=False))


def write_chi_ecp(write, result):
    """Report actual ionic and projector energies through the output channel."""
    from .channel import section_header
    record = chi_ecp_record(result)
    if record is None:
        return
    write(section_header('Chi scalar ECP Hamiltonian', width=56))
    write('    method maturity     = experimental; not-qualified\n')
    write(f"    ECP identity        = {record['identity']}\n")
    write(f"    operator            = {record['operator']}\n")
    for citation in _ecp_citations():
        write(f"    ECP citation        = {citation['key']}; doi:{citation['doi']}\n")
    write(f"    physical electrons  = {record['physical_electrons']}\n")
    write(f"    ECP core electrons  = {record['ecp_core_electrons']}\n")
    write(f"    explicit electrons  = {record['explicit_electrons']}\n")
    for i, atom in enumerate(record['atoms']):
        write(f"    ionic atom {i:<7d} = Z={atom['Z']}, Z_eff={atom['effective_charge']}, "
              f"ECP core={atom['core_electrons']}\n")
    for projector in record['projectors']:
        write(f"    projector atom      = {projector['atom_index']}; local l={projector['local_l']}; "
              f"Gaussian terms={len(projector['primitives'])}\n")
    for key, value in record['energy_components'].items():
        write(f"    E({key}) / cell = {value:.12f} Ha\n")
    write('    ECP parameters/support = complete in chi_ecp output record\n\n')


def chi_correlation_record(result):
    """Return the complete result payload for an OutputWriter consumer.

    Developer acceptance callers can include this in their output record.
    Public results carry their actual admission policy separately from qualification.
    """
    from ..periodic.chi.canonical_mp2 import CanonicalMP2Result
    from ..periodic.chi.canonical_cc import CanonicalCCResult
    if isinstance(result, CanonicalCCResult):
        return dict(
            schema='vibeqc.chi.same-source-cc/v1', method=result.method,
            **_ecp_correlation_fields(result),
            restart=_restart_record(result),
            reference_method=result.reference_method,
            method_maturity=result.method_maturity, quantitative_status=result.quantitative_status,
            public_admission=result.public_admission,
            energy_hf_per_cell=result.e_hf_per_cell,
            energy_nuclear_per_cell=result.e_nuclear_per_cell,
            energy_total_electronic_per_cell=result.e_total_electronic_per_cell,
            energy_ccsd_per_cell=result.e_ccsd_per_cell,
            energy_triples_per_cell=result.e_triples_per_cell,
            energy_correlation_per_cell=result.e_corr_per_cell,
            energy_total_per_cell=result.energy,
            energy_spin_components={},
            triples_variant=result.triples_variant if result.method.endswith('(T)') else 'not computed',
            converged=result.converged, iterations=result.iterations,
            residual_norm=result.residual_norm, energy_change=result.energy_change,
            max_t1=result.max_t1,max_t2=result.max_t2,workspace_bytes=result.workspace_bytes,
            active_spaces=[dict(spin=spin,frozen_per_character=n[0],occupied_per_character=n[1],virtual_per_character=n[2])
                           for spin,n in zip(('alpha','beta'),result.active_spaces)],
            canonical_residuals=[dict(spin=spin, **dict(zip(
                ('metric_orthogonality', 'fock_eigenpair', 'density_refill'), values)))
                for spin,values in zip(('alpha','beta'), result.canonical_residuals)],
            source_identity=result.source_identity,reference_identity=result.reference_identity,
            source=json.loads(result.source_descriptor_json),
            retained_metric_ranks=result.retained_metric_ranks,
            scf_acceptance=json.loads(result.scf_acceptance_json),
            normalization=result.normalization,citation_route=result.backend)
    if not isinstance(result, CanonicalMP2Result):
        raise TypeError('expected same-source CanonicalMP2Result or CanonicalCCResult')
    record = dict(
        schema='vibeqc.chi.same-source-mp2/v1', method=result.method,
        **_ecp_correlation_fields(result),
        restart=_restart_record(result),
        method_maturity=result.method_maturity,
        quantitative_status=result.quantitative_status,
        public_admission=result.public_admission,
        energy_hf_per_cell=result.e_hf_per_cell,
        energy_correlation_per_cell=result.e_corr_per_cell,
        energy_total_per_cell=result.energy,
        energy_spin_components=dict(aa=result.e_aa_per_cell,bb=result.e_bb_per_cell,ab=result.e_ab_per_cell),
        active_spaces=[dict(spin=spin,frozen_per_character=n[0],occupied_per_character=n[1],virtual_per_character=n[2])
                       for spin,n in zip(('alpha','beta'),result.active_spaces)],
        source_identity=result.source_identity, reference_identity=result.reference_identity,
        source=json.loads(result.source_descriptor_json),
        retained_metric_ranks=result.retained_metric_ranks,
        scf_acceptance=json.loads(result.scf_acceptance_json),
        normalization=result.normalization,
        citation_route=result.backend)
    from ..periodic.chi.canonical_rohf_mp2 import CanonicalROHFMP2Result
    if isinstance(result, CanonicalROHFMP2Result):
        record.update(
            schema='vibeqc.chi.same-source-rohf-rmp2/v1',
            reference_method=result.reference_method,
            canonicalization=result.canonicalization,
            canonicalization_identity=result.canonicalization_identity,
            perturbation_partition=result.perturbation_partition,
            frozen_core_convention=result.frozen_core_convention,
            energy_singles_alpha_per_cell=result.e_singles_alpha_per_cell,
            energy_singles_beta_per_cell=result.e_singles_beta_per_cell,
            energy_singles_per_cell=result.e_singles_per_cell,
            energy_doubles_per_cell=result.e_doubles_per_cell)
    return record


def write_chi_correlation(write, result):
    """Render real matched energies and provenance through the output layer."""
    from .channel import section_header
    record = chi_correlation_record(result)
    write(section_header('Chi same-source correlation', width=56))
    write(f"    treatment           = {record['method']} (EXPERIMENTAL)\n")
    write("    lineage             = chi finite-character retained canonical RI\n")
    write(f"    qualification       = {record['quantitative_status']}\n")
    write(f"    public admission    = {record['public_admission']}\n")
    write(f"    source identity     = {record['source_identity']}\n")
    write(f"    reference identity  = {record['reference_identity']}\n")
    write(f"    citation route      = {record['citation_route']}\n")
    write(f"    normalization       = {record['normalization']}\n")
    if 'ecp' in record:
        write(f"    ECP identity        = {record['ecp']['identity']}\n")
        for citation in record['ecp_citations']:
            write(f"    ECP citation        = {citation['key']}; doi:{citation['doi']}\n")
    counts = record.get('electron_counts', {})
    if counts:
        for key in ('physical_electrons', 'ecp_core_electrons', 'explicit_electrons',
                    'frozen_explicit_electrons', 'correlated_electrons'):
            write(f"    {key:26s}= {counts[key]} / cell\n")
        write(f"    frozen convention   = {counts['frozen_convention']}\n")
    for key, value in record['restart'].items():
        write(f"    {key:20s}= {value}\n")
    if 'energy_singles_per_cell' in record:
        write(f"    HF reference        = {record['reference_method']}\n")
        write(f"    canonicalization    = {record['canonicalization']}\n")
        write(f"    canonical identity  = {record['canonicalization_identity']}\n")
        write(f"    partition           = {record['perturbation_partition']}\n")
        write(f"    frozen convention   = {record['frozen_core_convention']}\n")
        write(f"    E(singles alpha)    = {record['energy_singles_alpha_per_cell']:.12f} Ha/cell\n")
        write(f"    E(singles beta)     = {record['energy_singles_beta_per_cell']:.12f} Ha/cell\n")
        write(f"    E(singles) / cell   = {record['energy_singles_per_cell']:.12f} Ha\n")
        write(f"    E(doubles) / cell   = {record['energy_doubles_per_cell']:.12f} Ha\n")
        for space in record['active_spaces']:
            write(f"    active space        = {space}\n")
    if 'energy_ccsd_per_cell' in record:
        write(f"    HF reference        = {record['reference_method']}\n")
        write(f"    E(electronic) / cell = {record['energy_total_electronic_per_cell']:.12f} Ha\n")
        write(f"    E(CCSD) / cell      = {record['energy_ccsd_per_cell']:.12f} Ha\n")
        write(f"    E(triples) / cell   = {record['energy_triples_per_cell']:.12f} Ha\n")
        write(f"    triples variant     = {record['triples_variant']}\n")
        write(f"    convergence         = {record['converged']}; iterations={record['iterations']}\n")
        write(f"    residual max        = {record['residual_norm']:.6e}\n")
        write(f"    energy change       = {record['energy_change']:.6e} Ha\n")
        write(f"    amplitude max T1/T2 = {record['max_t1']:.6e} / {record['max_t2']:.6e}\n")
        write(f"    workspace estimate  = {record['workspace_bytes']} bytes\n")
        for space in record['active_spaces']:
            write(f"    active space        = {space}\n")
        for residuals in record['canonical_residuals']:
            write(f"    canonical residuals = {residuals}\n")
    for spin, energy in record['energy_spin_components'].items():
        write(f"    E({spin}) / cell        = {energy:.10f} Ha\n")
    write(f"    E(SCF) / cell       = {record['energy_hf_per_cell']:.10f} Ha\n")
    write(f"    E(corr) / cell      = {record['energy_correlation_per_cell']:.10f} Ha\n")
    write(f"    E(SCF+corr) / cell  = {record['energy_total_per_cell']:.10f} Ha\n\n")
