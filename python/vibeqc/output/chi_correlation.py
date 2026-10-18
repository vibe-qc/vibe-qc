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
    from ..periodic.chi.local_mp2 import ChiLocalMP2Result
    if isinstance(result, ChiLocalMP2Result):
        return chi_local_mp2_record(result)
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
            coulomb_zero_mode=result.coulomb_zero_mode,
            madelung_hartree=result.madelung_hartree,
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
    if record['schema'] == 'vibeqc.chi.translation-local-mp2/v1':
        return write_chi_local_mp2(write, record)
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
        write(f"    Coulomb zero mode   = {record['coulomb_zero_mode']}\n")
        write(f"    BvK Madelung        = {record['madelung_hartree']:.12f} Ha\n")
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


def chi_local_mp2_record(result):
    """Consumed chi source, approximation controls and raw solver diagnostics."""
    from ..dlpno.translation_builder import _record
    source = json.loads(result.source_record_json)
    calculation = result.translation
    solved = calculation.solution
    record = dict(
        schema='vibeqc.chi.translation-local-mp2/v1', method=result.method,
        method_maturity=result.method_maturity, quantitative_status=result.quantitative_status,
        source_identity=source['source_identity'], reference_identity=source['reference_identity'],
        adapter_identity=result.adapter_identity, consumed_identity=calculation.identity,
        same_source=True, source=source, settings=json.loads(result.settings_json),
        electron_counts=source['electron_counts'],
        energy_hf_per_cell=solved.e_hf_per_cell,
        energy_raw_correlation_per_cell=result.raw_e_corr_per_cell,
        energy_discarded_pno_correction_per_cell=result.pno_correction_per_cell,
        domain_correction='not computed', pair_screening='none',
        energy_correlation_per_cell=result.e_corr_per_cell,
        energy_total_per_cell=result.e_total_per_cell,
        method_convention=source['method_convention'],
        normalization='energy per primitive cell; exact translation orbit multiplicities',
        converged=solved.converged, iterations=solved.n_iter, residual_norm=solved.max_residual,
        energy_change=solved.trace[-1].energy_change,
        pao_ranks=list(calculation.pair_build.pao_ranks),
        pno_ranks=list(calculation.pair_build.pno_ranks),
        domains=_record(calculation.domain_selection.provenance),
        occupied_selection=_record(calculation.occupied_selection.provenance),
        coefficient_storage=_record(calculation.pair_build.provenance['coefficient_storage']),
        solver=_record(solved.provenance), citation_route=result.backend,
        factor_provider=source.get('factor_provider', 'dense'),
        factor_provider_record=source.get('factor_provider_execution', source.get('factor_provider_record', {})),
        memory_budget_scope='separate occupied-preparation, factor-provider and native transport caps; '
                            'dense oracle has no factor cap; not total process memory')
    if source['ecp']:
        record.update(ecp=source['ecp'], ecp_citations=_ecp_citations())
    return record


def write_chi_local_mp2(write, record):
    """Render raw and corrected quantities without concealing nonconvergence."""
    from .channel import section_header
    write(section_header('Chi translation local MP2 (EXPERIMENTAL)', width=56))
    for label, key in (('qualification', 'quantitative_status'),
                       ('source identity', 'source_identity'),
                       ('reference identity', 'reference_identity'),
                       ('consumed identity', 'consumed_identity'),
                       ('convention', 'method_convention'),
                       ('normalization', 'normalization'),
                       ('citation route', 'citation_route')):
        write(f'    {label:20s}= {record[key]}\n')
    write(f"    convergence         = {record['converged']}; iterations={record['iterations']}\n")
    write(f"    raw residual max    = {record['residual_norm']:.6e}\n")
    write(f"    approximation/solve = {record['settings']}\n")
    preparation = record['occupied_selection'].get('preparation_workspace')
    if preparation:
        write(f"    occupied prep bytes = accounted {preparation['accounted_peak_bytes']}; "
              f"cap {preparation['max_selection_bytes']}; home output {preparation['home_output_bytes']}\n")
    storage = record['coefficient_storage']
    write(f"    coefficient storage = {storage['kind']}; selected occupied "
          f"{storage['selected_occupied_bytes']}; pair AO {storage['pair_ao_bytes']}; "
          f"pair recipes {storage['pair_recipe_bytes']} bytes\n")
    write(f"    pair recovery error = {storage['maximum_pair_reconstruction_error']}\n")
    provider = record['factor_provider_record']
    write(f"    factor provider     = {record['factor_provider']}\n")
    if record['factor_provider'] == 'streamed':
        write(f"    provider identity   = {provider['identity']}\n")
        write(f"    factor bytes        = accounted peak {provider['peak_accounted_bytes']}; "
              f"cap {provider['max_factor_bytes']}; cache {provider['cache_bytes']}; "
              f"largest returned buffer {provider['max_output_bytes']}\n")
        write(f"    factor audit        = {provider['audits']}\n")
    write(f"    electrons per cell  = {record['electron_counts']}\n")
    for label, key in (('E(SCF)', 'energy_hf_per_cell'),
                       ('E(raw local MP2)', 'energy_raw_correlation_per_cell'),
                       ('discarded-PNO estimate', 'energy_discarded_pno_correction_per_cell'),
                       ('E(corr+PNO estimate)', 'energy_correlation_per_cell'),
                       ('E(total+PNO estimate)', 'energy_total_per_cell')):
        write(f'    {label:24s}= {record[key]:.12f} Ha/cell\n')
    write('    domain correction   = not computed\n')
    write(f"    memory budget scope = {record['memory_budget_scope']}\n\n")


def emit_chi_local_mp2(result, *, output=None):
    """Publish one experimental library result through the shared output layer."""
    from .channel import record as emit_record, write
    record = chi_local_mp2_record(result)
    emit_record('chi_translation_local_mp2', **record)
    if output is None:
        write_chi_local_mp2(write, record)
        return
    from pathlib import Path
    from .plan import OutputPlan, PlannedFile
    from .writer import OutputWriter
    from .citations import emit_citations, citation_manifest_rows
    stem = Path(output)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = {ext: Path(str(stem)+ext) for ext in ('.out', '.system', '.bibtex', '.references')}
    plan = OutputPlan(stem=stem, job_kind='post_scf', method=result.backend,
        basis=record['source']['source']['orbital_basis'], functional=None,
        files=tuple(PlannedFile(role, paths[ext], fmt, True, description)
            for ext,role,fmt,description in (
                ('.out','log','text','Experimental chi local MP2 result'),
                ('.system','manifest','toml','Consumed source and approximation settings'),
                ('.bibtex','citations','bibtex','Method citations'),
                ('.references','citations','text','Method references'))))
    writer = OutputWriter(plan, record_hostname=False,
                          extra_run_fields={'chi_correlation': record})
    with writer.context():
        citations, bib, refs, block = emit_citations(stem, method=result.backend,
            basis=plan.basis, periodic=True, uses_ecp=bool(record.get('ecp')), pno_norm='mp2')
        writer.set_citations(citation_manifest_rows(citations))
        for path in (bib, refs):
            writer.record(path)
        with paths['.out'].open('w', encoding='utf-8') as stream:
            write_chi_local_mp2(stream.write, record)
            stream.write(block)
        writer.record(paths['.out'])
        writer.set_result_status('converged' if result.converged else 'not_converged')
