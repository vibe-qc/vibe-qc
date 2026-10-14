# Exchange an H2 calculation with QCSchema

This lesson uses QCSchema as a portable input and result format. You will
run the same H2 molecule with three electronic-structure methods, inspect
their atomic result JSON files, and compare the energies with a published
Born-Oppenheimer reference.

QCSchema does not specify which approximation must produce an energy. Its
JSON fields describe the molecule, the requested driver, the method and basis,
and the returned values. An interchange check therefore has two parts:
the files must match the schema, and the calculated quantity must make
physical sense. The [QCSchema reference](../qcschema.md) lists the subset
vibe-qc executes.

## 1. Inspect the input

The complete input is
[`examples/qcschema/h2_input.json`](../../examples/qcschema/h2_input.json):

```json
{
  "schema_name": "qcschema_input",
  "schema_version": 1,
  "molecule": {
    "schema_name": "qcschema_molecule",
    "schema_version": 2,
    "symbols": ["H", "H"],
    "geometry": [0.0, 0.0, 0.0, 0.0, 0.0, 1.4011],
    "molecular_charge": 0,
    "molecular_multiplicity": 1
  },
  "driver": "energy",
  "model": {"method": "HF", "basis": "sto-3g"},
  "keywords": {}
}
```

The geometry is Cartesian and measured in **bohr**. The H-H separation is
1.4011 bohr, approximately 0.7414 Angstrom. The molecule has two electrons
and is a singlet. STO-3G is deliberately small so this exercise runs quickly;
it is not a basis-converged prediction.

## 2. Run the energy examples

From the repository root, with vibe-qc installed in `.venv`:

```sh
mkdir -p /tmp/vibeqc-qcschema-example
.venv/bin/python examples/qcschema/compare_h2.py \
    --output-dir /tmp/vibeqc-qcschema-example
```

The script reads the input JSON, changes only `model.method`, and calls
`run_qcschema` for HF, MP2, and FCI. It writes
`h2_hf_result.json`, `h2_mp2_result.json`, and `h2_fci_result.json`
to the output directory. To use the API without the script:

```python
import vibeqc as vq

data = vq.read_qcschema("examples/qcschema/h2_input.json")
result = vq.run_qcschema(data, output_path="/tmp/h2-result.json")
assert result["success"]
print(result["return_result"], result["properties"]["return_energy"])
```

The two printed energies should match for an energy driver. The JSON result
includes the method, basis, original molecule, units implied by QCSchema,
provenance, and electron counts. Native vibe-qc logs and other sidecars are
temporary unless you pass `output_stem=`.

## 3. Compare with the literature

[Pachucki, *Phys. Rev. A* **82**, 032509 (2010),
doi:10.1103/PhysRevA.82.032509](https://doi.org/10.1103/PhysRevA.82.032509)
reports the high-precision nonrelativistic Born-Oppenheimer ground-state
energy of H2 at exactly 1.4011 bohr as
**-1.1744759314002167(3) Hartree**. The paper's Table II and abstract
give this value. Its uncertainty is far smaller than the STO-3G basis
error in this exercise.

The script prints each energy and `E - BO reference`. HF neglects electron
correlation. MP2 adds a perturbative estimate of it. FCI treats correlation
exactly **within STO-3G**, but a finite basis still prevents it from reaching
the published near-complete-basis energy. A sensible run has HF above FCI
and all three STO-3G energies above the reference. The numerical comparison
tests the method and coordinate conventions, not QCSchema compliance by itself.

| Method | vibe-qc energy (Hartree) | Above BO reference (Hartree) |
|---|---:|---:|
| HF/STO-3G | -1.116682734332 | 0.057793197068 |
| MP2/STO-3G | -1.129854206572 | 0.044621724828 |
| FCI/STO-3G | -1.137269844715 | 0.037206086686 |

These values were calculated from the accompanying input with vibe-qc
`v0.17.4` at the fixed geometry. The FCI gap is primarily a small-basis
effect; it must not be read as JSON interchange error.

For a stricter interchange check, validate the input and output JSON against
the [official QCSchema repository](https://github.com/MolSSI/QCSchema).
vibe-qc's `read_qcschema` and `write_qcschema` perform JSON transport,
while `run_qcschema` validates the supported execution subset.

## 4. Run the derivative drivers

```sh
.venv/bin/python examples/qcschema/hf_derivatives.py \
    --output-dir /tmp/vibeqc-qcschema-example
```

The gradient result has 6 numbers, one x/y/z triple per atom in Hartree/bohr.
The Hessian result has 36 numbers, a flattened 6 by 6 matrix in
Hartree/bohr squared. These drivers currently support HF only. For a
gradient, the two H atoms should exert equal and opposite forces along the
bond; the x and y components should be near zero by symmetry.

## 5. Move between native molecules and JSON

```python
import vibeqc as vq

data = vq.read_qcschema("examples/qcschema/h2_input.json")
native_molecule = vq.molecule_from_qcschema(data["molecule"])
portable_molecule = vq.molecule_to_qcschema(native_molecule)
vq.write_qcschema("/tmp/h2-molecule.json", portable_molecule)
```

The converter preserves the native atom order, geometry, charge, and
multiplicity. If your source dictionary contains additional QCSchema metadata,
keep that dictionary as well: native `Molecule` does not carry every schema
field. QCElemental is optional. Its models must match the emitted QCSchema
versions; see the [compatibility note](../qcschema.md#qcelemental-compatibility)
before constructing a QCElemental model from these dictionaries.
