# Basis-set toolkit

vibe-qc ships two basis-set toolkits that cover the full lifecycle:
import from the Basis Set Exchange (BSE) and other QC-program formats,
validation against the canonical model, conversion to legacy formats,
and export to the QVF-Basis standalone container.

```{admonition} Feature status
:class: note

``vibeqc.basis_toolkit`` (under `python/vibeqc/basis_toolkit/`) is a
production-quality prototype for modern basis-set storage, validation,
conversion, and export. ``vibe-basis/`` is a co-located sub-project
that optimises basis-set parameters against reference calculations
(via CRYSTAL14 backends and `vq` transport). Both are active
development areas; the APIs are stable but the feature surface is
growing.
```

## `vibeqc.basis_toolkit` -- Python API

Import from BSE, convert between formats, and validate:

```python
from vibeqc.basis_toolkit import (
    BasisSetData, from_bse_file, to_g94, to_orca, to_nwchem,
    to_qvf_json, save_qvf, load_qvf, validate_basis_file,
)

# Import from BSE JSON
bse = from_bse_file("cc-pvdz.json")
print(bse.name, bse.basis_family, bse.elements.keys())

# Export to legacy formats
g94_text = to_g94(bse)
orca_text = to_orca(bse)
nwchem_text = to_nwchem(bse)

# Save as QVF-Basis (text mode)
save_qvf_json(bse, "cc-pvdz.qvf.json")

# Save as QVF-Basis (packaged, binary-safe)
save_qvf(bse, "cc-pvdz.qvf")

# Round-trip
bse2 = load_qvf("cc-pvdz.qvf")
assert bse.name == bse2.name
```

### Import formats

| Format | Function | Notes |
|---|---|---|
| BSE JSON | `from_bse_file()`, `from_bse_json()` | Basis Set Exchange format |
| CRYSTAL atom | `from_crystal_atom()`, `from_crystal_file()` | CRYSTAL basis-set input |
| Gaussian 94 | `from_g94()`, `from_g94_file()` | Legacy `.g94` format |
| ORCA | `from_orca()`, `from_orca_file()` | ORCA basis-block format |
| NWChem | `from_nwchem()`, `from_nwchem_file()` | NWChem basis format |
| QVF (JSON / packed) | `load_qvf()`, `load_any()` | QVF-Basis container |

### Export formats

| Format | Function | Notes |
|---|---|---|
| Gaussian 94 | `to_g94()`, `to_g94_file()` | With loss report via `loss_report_g94()` |
| ORCA | `to_orca()`, `to_orca_file()` | With loss report via `loss_report_orca()` |
| NWChem | `to_nwchem()`, `to_nwchem_file()` | With loss report via `loss_report_nwchem()` |
| QVF JSON | `to_qvf_json()`, `save_qvf_json()` | Human-readable text mode |
| QVF packed | `to_qvf_bytes()`, `save_qvf()` | Binary-safe package for distribution |
| libint bridge | `to_libint_basis()` | Converts to vibe-qc's native C++ `BasisSet` |

```{admonition} Cartesian imports on v0.15.0 to v0.15.114
:class: warning

`to_libint_basis()` honors the imported basis's `harmonic_type`, so a
basis imported as Cartesian builds genuine Cartesian (`pure=False`)
shells. On releases v0.15.0 through v0.15.114 the AO evaluators applied a
wrong per-component normalization to Cartesian shells with l ≥ 2, making
DFT (through v0.15.113) and COSX (through v0.15.114) results on such
imports wrong (e.g. ~0.35 Ha on a Be test case). Fixed on `main`
2026-08-05; the DFT half shipped in v0.15.114, the COSX half lands in the
next release. See
[troubleshooting](../troubleshooting.md#dft--cosx-wrong-on-a-basis-imported-as-cartesian-via-basis_toolkit-v0150-to-v015114)
for affected surfaces and workarounds. Named bases were never affected.
```

### Canonical model

The `BasisSetData` object carries a format-neutral representation:

- `name`, `basis_family`, `description` -- metadata
- `elements: dict[str, ElementBasis]` -- per-element basis functions
- `Shell` / `Primitive` / `HarmonicType` -- angular-momentum encoding
- `ECPEntry` / `ECPPotential` / `ECPTerm` -- effective core potentials
- `Provenance` / `Reference` -- source and citation tracking
- `BasisRole` -- `orbital`, `jkfit`, `mp2fit`, `ecp`

### Validation

```python
from vibeqc.basis_toolkit import validate_basis_file, validate_basis_json

errors = validate_basis_file("cc-pvdz.qvf.json")
if errors:
    for e in errors:
        print(e)
```

The validator checks structural completeness, angular-momentum
consistency, primitive ordering, ECP coherency, and metadata
completeness.

## `vibeqc.basis_toolkit` -- CLI

```bash
# Validate a QVF-Basis file
python -m vibeqc.basis_toolkit.cli validate sto-3g.qvf.json

# Convert BSE JSON to QVF-Basis (text mode)
python -m vibeqc.basis_toolkit.cli convert sto-3g.bse.json -o sto-3g.qvf.json

# Convert to legacy format
python -m vibeqc.basis_toolkit.cli convert sto-3g.bse.json -o sto-3g.g94 -f g94

# Show loss report for a conversion
python -m vibeqc.basis_toolkit.cli loss -f g94 sto-3g.bse.json

# Inspect a basis set (summary)
python -m vibeqc.basis_toolkit.cli inspect sto-3g.qvf.json
```

## `vibe-basis` -- basis-set optimisation

`vibe-basis/` is a co-located sub-project (separate `pyproject.toml`)
for optimising basis-set parameters against reference calculations.
It exposes the `vb` CLI:

```bash
./vibe-basis/scripts/install.sh
vibe-basis/.venv/bin/vb --version
vibe-basis/.venv/bin/vb parse crystal output.out
```

The standalone installer supports macOS and Linux and has matching
`update.sh`, `reinstall.sh`, and marker-guarded `uninstall.sh` commands in
`vibe-basis/scripts/`. The basis toolkit stays in the core repository.
Install the queue separately from [vibe-queue](https://github.com/vibe-qc/vibe-queue);
the old `--with-vq` recipe expects a co-located source directory and no longer applies.

Planned verbs: `vb emit-test-set`, `vb fit`, `vb parity` -- the full
basis-optimisation workflow (generate test sets, fit exponents against
reference data, verify against published parity tables).

### Transport backends

`vibe-basis` runs calculations through `vq` (the vibe-queue tool):

```python
from vibe_basis.transports.vq import VqTransport
# Submit basis-parameter sweeps to a compute fleet
```

See the [queue guide](queue.md) for the `vq` infrastructure.

## Where to look in the code

| Layer | Path |
|---|---|
| Python API (toolkit) | `python/vibeqc/basis_toolkit/` |
| Canonical model | `python/vibeqc/basis_toolkit/model.py` |
| CLI | `python/vibeqc/basis_toolkit/cli.py` |
| QVF-Basis format | `python/vibeqc/basis_toolkit/qvf_basis.py` |
| Library management | `python/vibeqc/basis_toolkit/library_writer.py` |
| Validator | `python/vibeqc/basis_toolkit/validator.py` |
| vibe-basis sub-project | `vibe-basis/` |
| `vb` CLI | `vibe-basis/src/vibe_basis/cli.py` |
| CRYSTAL23-compatible backend | `vibe-basis/src/vibe_basis/backends/crystal.py` |
| Tests (toolkit) | `python/vibeqc/basis_toolkit/tests/` |
| Tests (vibe-basis) | `vibe-basis/tests/` |

## See also

- [Basis sets](basis_sets.md) -- using basis sets in calculations.
- [Design: QVF Basis format](../design_qvf_basis.md) -- the QVF-Basis
  container specification.
- [QVF Basis developer readme](../qvf_basis_developer_readme.md) --
  developer-oriented format details.
- [Queue (vq)](queue.md) -- the compute-transport layer vibe-basis uses.
