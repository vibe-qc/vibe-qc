"""Experimental automatic finite-torus restricted local MP2 entry point.

This bounded route connects projector-frame and Mulliken-domain selection
to representative PAO/PNO construction and compiled Jacobi iteration. It
does not alter any molecular or periodic method's default dispatch.
"""

from dataclasses import dataclass

from .source import source_identity
from .translation_builder import build_translation_pair_pnos, _freeze, _record
from .translation_domains import derive_translation_atom_groups, select_translation_pair_domains
from .translation_selection import select_translation_occupied
from .translation_solver import solve_translation_mp2
from .translation_audit import occupied_audit_context, audit_execution


@dataclass(frozen=True)
class AutomaticTranslationMP2Result:
    """The selections, pair build and solved state used by one calculation.

    Energies and convergence status live on ``solution``. Returning all
    preparation records makes domain truncation and the admitted occupied
    gauge inspectable even when the compiled iteration does not converge.
    """

    occupied_selection: object
    domain_selection: object
    pair_build: object
    solution: object
    identity: str
    provenance: object


def run_translation_dlpno_mp2(
    source, mesh, *, n_frozen=0, home_atom_aos=None, tcut_mkn=0.0,
    tcut_pno=0.0, pno_norm="mp2", lindep=1e-8, seed_lindep=1e-8,
    tolerance=1e-10, max_iter=100, conv_tol_energy=1e-9,
    conv_tol_residual=1e-7, damping=1.0,
    factor_audit="auto", transport_backend="prepared", max_transport_bytes=256 * 1024**2,
    compact_storage=False, max_selection_bytes=64*1024**2,
):
    """Automatically prepare and solve admitted real restricted sources.

    ``n_frozen`` is the total finite-torus frozen orbital count and must
    comprise complete bands. The source's first ``n_frozen`` occupieds
    define that projector. Home atom AO groups come from repeated source
    shell metadata unless explicitly supplied for a model Hamiltonian.

    Zero domain and PNO cutoffs retain the full virtual space. Nonzero
    domain cutoffs select the historical absolute Mulliken population
    criterion, not DOI or OSV domains. Selection and pair construction
    retain their explicit admission guards. No caller arrays or options
    are changed; no process-wide overlap replacement is used.
    ``compact_storage=True`` shares immutable selected occupied home columns
    and retains projected pair recipes instead of a dense AO basis pool.
    Complete source and pair audits still run. Bloch projector sources use
    the cyclic selector under ``max_selection_bytes``; generic dense sources
    keep their established selector. Downstream projected pair matrices are
    outside that separately scoped occupied-preparation reservation.
    """
    with occupied_audit_context():
        selection = select_translation_occupied(
            source, mesh, n_frozen=n_frozen, tolerance=tolerance, seed_lindep=seed_lindep,
            compact_storage=compact_storage, max_selection_bytes=max_selection_bytes,
        )
        groups = (derive_translation_atom_groups(source, mesh)
                  if home_atom_aos is None else home_atom_aos)
        domains = select_translation_pair_domains(
            mesh, source.overlap, selection.occupied, groups,
            tcut_mkn=tcut_mkn, tolerance=tolerance,
        )
        built = build_translation_pair_pnos(
            source, mesh, selection.occupied, domains.pair_ao_domains,
            frozen_occupied=selection.frozen_occupied, tcut_pno=tcut_pno,
            pno_norm=pno_norm, lindep=lindep, tolerance=tolerance, factor_audit=factor_audit,
            compact_storage=compact_storage,
        )
        occupied_audits = audit_execution()
    solved = solve_translation_mp2(
        built, max_iter=max_iter, conv_tol_energy=conv_tol_energy,
        conv_tol_residual=conv_tol_residual, damping=damping,
        transport_backend=transport_backend, max_transport_bytes=max_transport_bytes,
    )
    record = {
        "kind": "automatic-translation-mp2-v1",
        "source_identity": built.source_identity, "same_source": built.same_source,
        "occupied_selection_identity": selection.identity,
        "domain_selection_identity": domains.identity,
        "pair_build_identity": built.identity,
        "solver": _record(solved.provenance),
        "occupied_audit_execution": occupied_audits,
    }
    return AutomaticTranslationMP2Result(selection, domains, built, solved,
                                         source_identity(record), _freeze(record))
