"""Output-layer record for experimental same-source chi correlation."""
import json


def chi_correlation_record(result):
    """Return the complete result payload for an OutputWriter consumer.

    Developer acceptance callers can include this in their output record.
    Public results carry their actual admission policy separately from qualification.
    """
    from ..periodic.chi.canonical_mp2 import CanonicalMP2Result
    if not isinstance(result, CanonicalMP2Result):
        raise TypeError('expected same-source CanonicalMP2Result')
    return dict(
        schema='vibeqc.chi.same-source-mp2/v1', method=result.method,
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
    for spin, energy in record['energy_spin_components'].items():
        write(f"    E({spin}) / cell        = {energy:.10f} Ha\n")
    write(f"    E(SCF) / cell       = {record['energy_hf_per_cell']:.10f} Ha\n")
    write(f"    E(corr) / cell      = {record['energy_correlation_per_cell']:.10f} Ha\n")
    write(f"    E(SCF+corr) / cell  = {record['energy_total_per_cell']:.10f} Ha\n\n")
