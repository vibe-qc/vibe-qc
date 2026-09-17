# Protocol 1 fixtures

These files belong to the standalone vibe-qc worker API 1.0.0. They contain no
viewer dependency. JSON requests are indented for review: compact each to one
JSON line before sending it. From the repository root:

```sh
.venv/bin/python -c 'import json,sys; print(json.dumps(json.load(sys.stdin)))' \
  < tests/data/relocalize/h2.request.json | .venv/bin/python -m vibeqc_relocalize
.venv/bin/python -m vibeqc_relocalize --probe
```

- `h2`: H2/STO-3G, bond length 1.4 bohr, neutral restricted singlet. The input
  occupied coefficients were generated once with native RHF. Relocalization
  itself consumes those coefficients and must not run SCF. The result is IBO
  with MINI IAO populations and zero net charge.
- `periodic_gamma`: a one-dimensional H2 Gaussian lattice with primitive
  vector length 4 bohr. Its overlap comes from native Gaussian lattice blocks
  with explicit lattice translations at distance <= 12 bohr and no additional
  pair cutoff, folded at Gamma. The input occupied vector is normalized in
  that metric. The overlap differs appreciably from molecular overlap.
- `periodic_complex`: the same finite-image Gaussian model with a two-point
  cyclic mesh. The second occupied vector has a genuine relative complex
  phase (`[1, 0.4+0.8i]` before metric normalization), so dropping imaginary
  parts changes the projector. The localized output keeps complex encoding.
- Both periodic fixtures are **synthetic, isolated occupied manifolds**, with
  declared band energies -1 and 0.5 hartree. They exercise the experimental
  finite-torus localization contract and native integral/Fourier paths; they
  are not periodic SCF ground-state or material-accuracy references. Their
  orbital centers use the reported projected circular approximation.
- `missing_orbitals` and `periodic_ibo`: deliberate invalid inputs paired with
  expected stable error code/field subsets. Error messages and versions need
  not compare byte-for-byte.

`*.result.json` contains the terminal event's `result` payload, without the
transport envelope. Results are reproducible up to floating-point tolerance,
orbital phase/order and symmetry-equivalent localization gauges. Tests compare
projectors, populations and charges with tolerance, not byte-identical output.
For more general degenerate manifolds, populations may also need permutation
matching. Regenerate the fixtures from this checkout with:

```sh
.venv/bin/python tests/data/relocalize/generate.py
```

The generator explicitly does one molecular RHF solve to prepare input. It
constructs the periodic occupied subspaces directly without periodic SCF.
Generated files stay under this test-data directory.

Validation commands:

```sh
OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_relocalization_worker.py -q
OMP_NUM_THREADS=1 .venv/bin/python -m pytest \
  tests/test_binding_sanity.py tests/test_test_gate_lanes.py -q
OMP_NUM_THREADS=1 .venv/bin/python -m pytest \
  tests/test_iao_ibo.py tests/test_localise.py \
  tests/test_ao_convention_invariants.py \
  tests/test_periodic_aiccm2026dev_b_localization.py \
  tests/test_periodic_gaussian_localization.py \
  -k 'not public_wrapper_localizes_a_vacuum_padded_b_stream_reference and not real_chain_spread_stabilizes_as_the_cyclic_cluster_grows' -q
.venv/bin/ruff check python/vibeqc/relocalize.py python/vibeqc_relocalize \
  tests/test_relocalization_worker.py tests/data/relocalize/generate.py
```

The regression command excludes two pre-existing full periodic SCF tests; the
worker consumes supplied periodic matrices without running either SCF wrapper.

The worker tests independently compare `C C^H` and `C^H S C`, reconstruct the
periodic Fourier transform, forbid SCF/molecular dispatch on supplied periodic
inputs, rebuild mixed spherical/Cartesian archived shells, and consume the
existing water QVF. They also verify the isolated standard-library launcher
returns a structured failed capability probe when a fake backend emits Python
and native stdout diagnostics and then fails to import.

The complete input/output contract and exact viewer changes are in
[the integration handoff](../../../docs/relocalization_worker.md).
