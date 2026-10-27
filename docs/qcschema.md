# QCSchema molecular interchange

QCSchema provides a common JSON representation for molecular calculations.
vibe-qc can read and write molecule, atomic input, and atomic result documents,
and can run a supported atomic input with `run_qcschema`. See the
[worked tutorial](tutorial/qcschema_interchange.md) and the
[runnable examples](../examples/qcschema/compare_h2.py).

```python
import vibeqc as vq

atomic_input = {
    "schema_name": "qcschema_input",
    "schema_version": 1,
    "molecule": {
        "schema_name": "qcschema_molecule",
        "schema_version": 2,
        "symbols": ["H", "H"],
        "geometry": [0.0, 0.0, 0.0, 0.0, 0.0, 1.4011],
    },
    "driver": "energy",
    "model": {"method": "HF", "basis": "sto-3g"},
    "keywords": {},
}
vq.write_qcschema("h2-input.json", atomic_input)
result = vq.run_qcschema("h2-input.json", output_path="h2-result.json")
print(result["return_result"])
```

## Public functions

| Function | Purpose |
|---|---|
| `read_qcschema(path)` | Read a JSON object from a file; reject malformed JSON and non-object roots |
| `write_qcschema(path, document)` | Write a schema-tagged molecule, atomic input, or atomic result to a `.json` file |
| `molecule_from_qcschema(data)` | Convert a supported `qcschema_molecule` version 2 object to `Molecule` |
| `molecule_to_qcschema(molecule)` | Convert a native `Molecule` to a molecule object with its atom order, charge, multiplicity, and geometry |
| `run_qcschema(input_data, output_path=None, output_stem=None)` | Run an atomic input dictionary or JSON file and return an atomic result dictionary |

`read_qcschema` and `write_qcschema` transport JSON; they do not validate
every field against the complete QCSchema specification. Use the official
schema or QCElemental for independent validation. `run_qcschema` validates
the subset that it executes and raises an error when it cannot represent a
requested input. The input spelling `qc_schema_input` is also accepted. The
result spelling is `qcschema_output`, schema version 1.

## Supported calculations

| Input | Supported values |
|---|---|
| `driver` | `energy` for supported molecular methods; `gradient` and `hessian` for HF |
| `model.method` | `HF` or `SCF`, explicit `RHF` or `UHF`, `MP2`, `CCSD`, `CCSD(T)`, `CISD`, `FCI`, or a DFT functional name such as `B3LYP` |
| `model.basis` | A named basis available to the installed vibe-qc basis library |
| `keywords` | Empty object only |
| Molecule | Real atoms with supported element symbols, Cartesian geometry, integer charge and multiplicity; optional matching atomic numbers and `fix_com`/`fix_orientation` flags |

For `HF`/`SCF`, multiplicity 1 selects RHF and higher multiplicity selects
UHF. A functional name selects restricted or unrestricted Kohn-Sham in the
same way. The `gradient` result is a flat array of length `3N` in atom order.
The `hessian` result is a flat, row-major array of length `(3N)^2`.

Molecule geometry uses **bohr**, energy uses **Hartree**, gradients use
Hartree/bohr, and Hessians use Hartree/bohr squared. `return_result` contains
the requested driver result. The `properties.return_energy` field reports the
total energy, including correlation energy for correlated methods. The result
also contains the input molecule, model, driver, empty keywords, atom and
electron counts, provenance, and `success: true`.

`run_qcschema` writes normal vibe-qc sidecars in a temporary directory and
removes them after the run by default. Pass `output_stem=` to retain them;
pass `output_path=` to write the QCSchema atomic result JSON. Paths supplied
to `write_qcschema` or `output_path` must end in `.json`.

## Scope and errors

This adapter is for **molecules**, not periodic systems. It rejects unsupported
molecule fields such as fragments, ghost atoms, nonempty `keywords`, invalid
or non-finite coordinates, unsupported drivers, and derivative requests for
methods other than HF. It raises an exception if a calculation fails to
converge; it does not fabricate a successful atomic result. An arbitrary
unknown method string is treated as a DFT functional name, so a misspelled
method will fail during calculation rather than during input parsing.

The native `Molecule` conversion represents atoms, geometry, charge, and
multiplicity. Other schema metadata is not preserved through that conversion;
if it matters, keep the original JSON dictionary. `run_qcschema` echoes the
input molecule dictionary in its result.

## QCElemental compatibility

QCElemental is optional and is not needed for file I/O or calculation. This
adapter emits QCSchema atomic input/output **version 1** and molecule
**version 2** documents. QCElemental's newer `models.v2` classes use a
different schema, so these dictionaries cannot be passed directly to those
classes. Use a QCElemental model that supports the emitted schema version, or
exchange the JSON dictionaries without constructing QCElemental models.

As a concrete environment limitation, QCElemental 0.51.2's legacy
`models.AtomicInput` and `models.AtomicResult` cannot be instantiated on
Python 3.14 because their Pydantic v1 backend is unavailable there. The
official QCSchema JSON schemas still validate the input and output files in
that environment.
