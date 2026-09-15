# Stage 0 Implementation Plan, vibe-basis

**Date:** 2026-05-14

## Phase 1: Create references module

* `vibe_basis/io/references.py`, migrate PT2013 SI Table 2 reference energies
  from `examples/basisset_dev/_generator.py`. The dict `PT2013_T2_HF` maps
  compound name (str) → HF energy per unit cell in Hartree (float, or None
  for compounds where SCF didn't converge in the paper).
* Expands the dict with utility accessors: `get_ref(name)`, `has_ref(name)`,
  `compounds_with_ref()`.

## Phase 2: Create Stage 0 recipe driver

`vibe_basis/recipes/pob_parity.py`, wires the existing building blocks:

  structures list → emit_input → transport.run → parse_output → compare to reference

The driver accepts an injectable Transport so tests can mock it.

```python
def run_pob_parity(
    structures: Iterable[Structure],
    transport: Transport,
    *,
    basis: str = "pob-tzvp",
    method: str = "rhf",
    workdir_root: Path = Path("pob_parity_runs"),
    cpus: int = 14,
    wall_time_s: int = 1800,
) -> ParityReport:
```

## Phase 3: Tests

`vibe_basis/tests/test_pob_parity_recipe.py`, all mocked, no CRYSTAL/compute-reference needed.
Tests verify the pipeline logic: emit, submit, parse, compare, report.

## Phase 4: Documentation

* Update `GOAL4_DESIGN.md`, mark as superseded by vibe-basis
* Update `vibe_basis/recipes/__init__.py`, reflect that pob_parity is implemented
* Update `vibe-basis/README.md` status section
* Add `vibe_basis/RECIPE_PATTERNS.md`

## Acceptance criteria

* `cd vibe-basis && ../.venv/bin/python -m pytest tests/test_pob_parity_recipe.py -v` → all pass
* `cd vibe-basis && ../.venv/bin/python -m pytest -q`, existing 61 tests still pass
* Stage 0 driver produces the correct `.d12` decks and can run end-to-end
  on compute-reference (real run gated on compute-reference access, not CI-blocking)
