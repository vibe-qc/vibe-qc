# Matched periodic performance reports

The #574 reporting contract compares complete calculations on a declared
matrix. It does not establish the full #593 production goal or change the
completed 0.18 release. Existing regression summaries retain their
accuracy-filtered timing tables; those tables alone do not qualify performance
parity.

Run the native-free reader from the source checkout:

```sh
python -m examples.regression.performance_report protocol.json samples.json > report.json
```

Exit status 0 means every declared case passed this protocol. Status 1 means
at least one case is missing, blocked, slow or unresolved by timing uncertainty.
Invalid documents return status 2. The report includes every required case,
all admitted pair receipts, observed and uncertainty-adjusted ratios, repeated
measurement ranges, reasons for exclusion and the worst ratio. Min/max ranges
are observations, not confidence intervals. A median cannot hide a slow repeat.

## Freeze the protocol before measurement

A protocol has this shape; the case below is illustrative, not accepted data:

```json
{
  "schema": "vibeqc-performance-v1",
  "repetitions": 3,
  "cache_mode": "cold",
  "target_ratio": 1.0,
  "cases": [{
    "id": "declared-periodic-control",
    "reference_code": "crystal",
    "accuracy_evidence": "qualified-accuracy-record-identity",
    "expected": {
      "system_id": "declared-periodic-control",
      "basis": "explicit-basis-identity",
      "method_id": "rhf",
      "kmesh": [1, 1, 1],
      "tolerance_ha": 0.000001,
      "tolerance_rationale": "Predeclared matched accuracy criterion"
    }
  }]
}
```

Use `examples.regression.core.performance.protocol_digest(protocol)` to compute
and add `sha256`, then preserve that document with the campaign's evidence before
running. The digest covers all fields except `sha256`; every sample records it.
A digest detects content differences, but does not prove when a protocol was
frozen. Never adjust the tolerance or remove a required slow case after seeing
results. Freeze separate protocols for cold and warm measurements. Warm-cache
construction time, storage and ownership must remain in the campaign evidence;
a warm pass does not qualify cold performance.

Ordinary cases require CRYSTAL23. A SKALA case instead names its actual
same-functional reference and `checkpoint_identity`; both receipts must carry
that checkpoint identity. Its report is labelled `same-functional checkpoint`,
not CRYSTAL23 parity. A declared identity is not proof that a real checkpoint
was loaded; independent producer evidence must establish that fact.

## Supply paired receipts

`samples.json` is a list. Each entry contains `case_id`, one-based `repeat`,
`protocol_sha256`, `candidate` and `reference`. Each side has:

- `row`: an existing `CodeRow` object serialized as a JSON object, including
  case labels, code/version, campaign `run_id`, convergence, energy, iterations,
  `wall_s` and physical-input qualification. Candidate qualification must be
  `qualified`, reference qualification `reference`. The reader recomputes the
  energy verdict with the frozen tolerance; a saved `pass` label is insufficient.
- `receipt`: the fields below, measured by the campaign producer.

| Receipt field | Meaning |
|---|---|
| `hardware_id`, `cpus`, `memory_limit_mb` | Same hardware and resource allocation for both sides; disclose model/topology in the linked evidence |
| `ranks`, `threads_per_rank`, `accelerator_identity` | Matched rank/thread layout and accelerator allocation (`none` for CPU-only); ranks times threads must not exceed the CPU allocation |
| `execution_identity` | Unique execution/job identity within each code; the `CodeRow.run_id` may be shared by the campaign |
| `runtime_identity` | Actual executable/native plus source identity, including any source overlay |
| `input_identity`, `output_identity` | Content identities of the executed input and retained terminal output |
| `qualification_evidence` | Record establishing matched geometry, basis/ECP, method/spin, k coordinates/weights, numerical settings, physical supports and zero-mode convention |
| `measurement_evidence` | Raw resource/timing record, including cost provenance and available work counts |
| `timing_scope` | Exactly `setup-through-final-confirmation`; excludes queue latency, includes setup, fitting/grid and terminal physical-state rebuild |
| `cache_mode` | `cold` or `warm`, matching the protocol |
| `phase_wall_s` | Positive disjoint `setup`, `scf`, `final` durations whose sum agrees with `wall_s` within its uncertainty |
| `wall_uncertainty_s` | Positive measured timing uncertainty smaller than the wall time, including timer resolution |
| `peak_rss_mb` | Positive measured peak memory; document single-process versus whole-job aggregation in measurement evidence |

Input/runtime identities and allocation must remain unchanged across repeats.
Execution identities cannot be reused. Missing/duplicate repetitions, unknown cases,
unqualified inputs, retries without their own accounting protocol, invalid
measurements and failed energy/convergence checks cannot pass. Producers must
not invent phase timings or memory to satisfy the schema; unavailable quantities
remain blockers for this report.

Each admitted pair reports its measured wall ratio and the conservative upper
ratio `(candidate_wall + candidate_uncertainty) /
(reference_wall - reference_uncertainty)`. A case passes only when every upper
ratio is at most one. A measured ratio above one is `slow`; measurements that
meet one but whose uncertainty crosses it are `uncertain`.

This reader checks declared receipts and their consistency. It does not fetch
artifacts, verify executable bytes, attest hardware, requalify physical inputs,
or substitute for independent numerical verification. Retain the containing
source/native identity and original reference outputs with the numerical
campaign. A report pass proves only the declared matrix and supplied evidence
contract; full production coverage and CRYSTAL23 parity remain separate claims.
