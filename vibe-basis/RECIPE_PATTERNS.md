# Recipe patterns — how to write a vibe-basis recipe

A *recipe* is a module in `vibe_basis/recipes/` that orchestrates
an end-to-end basis-set optimization workflow.  Recipes sit at the
top of the vibe-basis stack and are the entry points that users
and CLIs call.

## Anatomy of a recipe

Every recipe follows the same six-step pattern:

```
1.  SELECT   — choose which structures / compounds to include
2.  EMIT     — produce `.d12` (or `.inp` / `.gjf`) decks
3.  SUBMIT   — ship through a transport (local / vq / slurm)
4.  PARSE    — extract energy + convergence from output
5.  COMPARE  — compute delta to a reference (or objective for fitting)
6.  REPORT   — collate per-compound results into a summary dataclass
```

## Conventions

1. **Transport injection.**  Every recipe function takes a
   `transport: Transport` parameter.  Tests inject a mock transport
   that writes synthetic output back; production injects
   `LocalTransport` or `VqTransport`.

2. **Structure DB as input.**  Recipes accept `structures: Iterable[Structure]`,
   not hard-coded compound lists.  The caller filters with
   `in_table("PT2013-T4")` or `all_structures()`.

3. **Report dataclass.**  Each recipe has a dedicated report dataclass
   (e.g. `ParityReport`) with a `.summary()` method that prints
   human-readable results and a `.passes_acceptance` property for
   CI / automated gating.

4. **Mock-friendly.**  Tests never require CRYSTAL14, vq, or
   compute-host-d.  Use `MockTransport` (or a derived class) that takes
   a dict of `{compound_name: synthetic_output_text}`.

5. **No vibe-qc imports.**  Recipes import from `vibe_basis.*` only.
   vibe-basis is standalone — it does not depend on vibe-qc.

## Example: Stage 0 parity check

```python
from vibe_basis.recipes.pob_parity import run_pob_parity
from vibe_basis.io.structures import in_table
from vibe_basis.transports.local import LocalTransport

# Local smoke test — single compound
from vibe_basis.transports.local import LocalTransport
report = run_pob_parity(
    [STRUCTURES["MgO"]],
    LocalTransport(),
    basis="pob-tzvp",
    crystal_wrapper="crystal",   # or /path/to/run-crystal.sh
)
print(report.summary())

# Production on compute-host-d — full test set
from vibe_basis.transports.vq import VqTransport
report = run_pob_parity(
    in_table("PT2013-T4"),
    VqTransport(),
    basis="pob-tzvp",
    crystal_wrapper="/home/USER/crystal/run-crystal.sh",
)
print(report.summary())
```

## Adding a new recipe

1. Create `vibe_basis/recipes/<name>.py`.
2. Define a report dataclass (dataclass, frozen, with `.summary()`).
3. Implement the driver function with `transport: Transport` injection.
4. Add tests in `tests/test_<name>_recipe.py` using a mock transport.
5. Update `vibe_basis/recipes/__init__.py` with the new module.
