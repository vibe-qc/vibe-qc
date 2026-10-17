# Bounded selected and pair coefficient storage

Run `.venv/bin/python -m benchmarks.chi_pair_storage` from the source checkout
with development test dependencies. The ladder reuses the nonorthogonal,
occupied-coupled frozen-core finite model from the source tests: four primitive
AOs, one active and one frozen occupied band, full domains and zero PNO cutoff.
Model/source construction precedes measurement. No physical QC acceptance is
claimed. The script emits actual source/native runtime identities through
`vibeqc.output`.

One local run (seconds are observations, not performance promises):

| Cells | Selected bytes dense / compact | Pair AO bytes dense / compact | Recipe bytes both | Solve/preparation seconds dense / compact | Traced peak bytes dense / compact |
|---|---|---|---|---|---|
| 1 | 128 / 64 | 64 / 0 | 128 | 0.0105 / 0.0091 | 211582 / 64123 |
| 2 | 512 / 128 | 512 / 0 | 896 | 0.0113 / 0.0146 | 62596 / 67279 |
| 4 | 2048 / 256 | 3072 / 0 | 4992 | 0.0214 / 0.0282 | 101368 / 92032 |
| 8 | 8192 / 512 | 20480 / 0 | 32000 | 0.0592 / 0.0839 | 302306 / 220086 |

Selected bytes count unique buffers in the returned automatic result: dense
selection and pair spaces have separate snapshots; compact storage shares
home columns. Recipe bytes include indices, local A and occupied-tail B.
Recipes are shared by pair storage and build metadata. At eight cells this
scope falls from 60672 to 32512 bytes. The full-domain recipe itself is larger
than a single AO pool; the improvement removes its duplicate dense pool.
Native pair-AO input bytes are zero in compact mode. Recipe-only transport
accounts an extra one-pair reconstruction rectangle: required native bytes
at eight cells are 210024 versus 205928 for dense coefficient inputs. The
pre-existing native cap excludes binding copies of other inputs.

Largest raw-energy difference is below `3e-17` Ha/cell. Every solve converges;
maximum final residual is below `2.6e-13`. PNO corrections are reported
separately by the script. Complete pair reconstruction defects in this model
are below `1.1e-31`; other nonorthogonal/truncated and ill-conditioned cases
are covered by source tests at the unchanged `1e-10` admission tolerance.

Tracemalloc includes Python overhead and cold imports and does not measure
all native allocations or process RSS. No repeated-run timing statistics,
total-memory bound, linear scaling or production accuracy follows from this
small ladder. Selector eigensystems, transient full selected frames, source
storage, full-domain Gram matrices, amplitudes and transport links remain.
Scientific acceptance through a containing vq runtime remains separate.
