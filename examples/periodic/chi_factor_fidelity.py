"""Read-only conjugate-character diagnosis of an accepted classic chi archive.

Run through vq with an identified Python/native runtime. This measures every
retained factor entry and every within-transfer Coulomb block, then exercises
both local factor providers for frozen_core=0/1. Refusals are results, not a
nonzero process status. No SCF, refit, projection or source substitution occurs.
See docs/chi_mp2_convergence.md for the retained-eigenspace investigation.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-core-sha256', required=True)
    args = parser.parse_args()
    import numpy as np
    from vibeqc import _vibeqc_core
    from vibeqc.periodic.chi.checkpoint import load_chi_hf_reference
    from vibeqc.periodic.chi.local_mp2 import build_chi_translation_source
    from vibeqc.periodic.chi.operator_diagnostics import factor_conjugation_measurements
    from vibeqc.output import OutputChannel, write

    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if sha(_vibeqc_core.__file__) != args.expected_core_sha256:
        raise RuntimeError('native runtime pin mismatch')
    output = args.output.resolve()
    repo = next((p for p in Path(__file__).resolve().parents
                 if (p/'pyproject.toml').exists()), None)
    if repo and (output == repo or repo in output.parents):
        raise ValueError('output must be outside checkout')
    reference = load_chi_hf_reference(args.reference, expected_method='RHF', mode='accepted')
    hf = reference.hf_result
    physical = hf._chi_restart_physical
    operator = hf.fitted_operator
    if operator.descriptor.source != 'classic':
        raise ValueError('this diagnostic requires the classic retained source')
    lattice = np.asarray(physical['geometry']['lattice_bohr'])
    k = np.asarray(operator.descriptor.kpoints_cart) @ lattice / (2*np.pi)
    negative = []
    for point in k:
        candidates = np.flatnonzero(np.max(np.abs(k+point-np.rint(k+point)), axis=1) < 1e-12)
        if len(candidates) != 1:
            raise ValueError('unique conjugate character required')
        negative.append(int(candidates[0]))
    result = dict(archive_sha256=sha(args.reference), factor_identity=operator.identity,
                  native_sha256=args.expected_core_sha256, physical=physical,
                  executed_domains=json.loads(operator.descriptor.executed_domains_json),
                  negative_character_indices=negative, groups=[], local_admission=[],
                  qualification='diagnostic only; chi experimental; D93 not-qualified')
    for q, pairs in operator.descriptor.q_grouping:
        a = np.concatenate([operator[i, j].reshape(operator[i, j].shape[0], -1)
                            for i, j in pairs], axis=1)
        b = np.concatenate([operator[negative[i], negative[j]].reshape(a.shape[0], -1)
                            for i, j in pairs], axis=1)
        result['groups'].append(dict(q=q, pairs=pairs,
                                     **factor_conjugation_measurements(a, b)))
    for frozen in (0, 1):
        for provider in ('dense', 'streamed'):
            row = dict(frozen_core=frozen, provider=provider)
            try:
                build_chi_translation_source(reference, frozen_core=frozen,
                                             factor_provider=provider, compact_occupied=True)
            except (ValueError, MemoryError, NotImplementedError) as exc:
                row.update(status='REFUSED', exception=type(exc).__name__, message=str(exc))
            else:
                row['status'] = 'ACCEPTED'
            result['local_admission'].append(row)
    if sha(_vibeqc_core.__file__) != args.expected_core_sha256:
        raise RuntimeError('native runtime changed during diagnosis')
    output.parent.mkdir(parents=True, exist_ok=True)
    with OutputChannel.to_file(output):
        write(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
