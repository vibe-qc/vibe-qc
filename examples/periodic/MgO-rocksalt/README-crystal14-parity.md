# MgO CRYSTAL14 Parity Inputs

This directory contains paired MgO rocksalt inputs for CRYSTAL14 and
vibe-qc. They use the same physical setup:

- conventional lattice constant `a = 4.21 A`;
- CRYSTAL space group `225` (`Fm-3m`) with Mg at `4a` and O at `4b`;
- `POB-TZVP`;
- `SHRINK 8 8`;
- `TOLDEE 8`;
- `MAXCYCLE 50`;
- CRYSTAL-style SCF stabilization: `FMIXING 30` and `LEVSHIFT 6 0`.

The CRYSTAL inputs list the asymmetric unit and let CRYSTAL reduce to
the 2-atom FCC primitive cell. Each vibe-qc script is a standalone
input file: it builds that primitive cell explicitly, sets the SCF
controls, and runs the 8x8x8 Gamma-centred mesh (IBZ-reduced through
spglib) end to end on the native multi-k GDF backend,
`vq.run_krhf_periodic_gdf` for RHF and `vq.run_krks_periodic_gdf` for
RKS-PBE and RKS-B3LYP. PySCF is never imported; CRYSTAL and PySCF
references stay out of process and are compared as parsed outputs.

For the RHF case the CRYSTAL input intentionally has no `DFT` block;
closed-shell Hartree-Fock is CRYSTAL's default Hamiltonian for this
closed-shell insulating input.

| Method | CRYSTAL14 input | vibe-qc input |
|---|---|---|
| RHF | `MgO-rocksalt-RHF-pobtzvp-mp8-crystal14.d12` | `MgO-rocksalt-RHF-pobtzvp-mp8-gdf.py` |
| RKS-PBE | `MgO-rocksalt-RKS-PBE-pobtzvp-mp8-crystal14.d12` | `MgO-rocksalt-RKS-PBE-pobtzvp-mp8-gdf.py` |
| RKS-B3LYP | `MgO-rocksalt-RKS-B3LYP-pobtzvp-mp8-crystal14.d12` | `MgO-rocksalt-RKS-B3LYP-pobtzvp-mp8-gdf.py` |

Run CRYSTAL14 from this directory, for example:

```sh
crystal < MgO-rocksalt-RKS-PBE-pobtzvp-mp8-crystal14.d12 > MgO-rocksalt-RKS-PBE-pobtzvp-mp8-crystal14.out
```

Run a vibe-qc input from the repository root (pob-TZVP on the two-atom
primitive cell with 8x8x8 sampling takes minutes, not seconds, so start
with the RHF deck):

```sh
.venv/bin/python examples/periodic/MgO-rocksalt/MgO-rocksalt-RKS-PBE-pobtzvp-mp8-gdf.py
```

Or copy one `.py` file to a calculation directory and run it there:

```sh
/path/to/vibe-qc/.venv/bin/python MgO-rocksalt-RKS-PBE-pobtzvp-mp8-gdf.py
```

Convergence-control mapping:

| CRYSTAL14 | vibe-qc |
|---|---|
| `FMIXING 30` | `opts.fock_mixing = 0.30` |
| `LEVSHIFT 6 0` | `opts.level_shift = 0.60` Ha with `level_shift_warmup_cycles=5` |
| `TOLDEE 8` | `opts.conv_tol_energy = 1e-8` |
| `MAXCYCLE 50` | `opts.max_iter = 50` |

Each script writes `<stem>.out` and `<stem>.system` beside itself and
prints the per-cell energy, the electronic and nuclear pieces, the SCF
iteration count and the IBZ expansion. The retired PySCF-backed spike
is gone; the multi-k GDF backend these inputs targeted is the one that
now runs them.

## B3LYP flavor note

CRYSTAL14's `B3LYP` keyword computes the **VWN5** flavor of B3LYP
(verified against libxc `HYB_GGA_XC_B3LYP5` to 1e-9 Ha on
H2/STO-3G), exactly vibe-qc's bare `b3lyp` (the ORCA definition),
so the paired vibe-qc input uses the bare name and the comparison
is flavor-correct. Do NOT use the Gaussian-flavor spellings
(`b3lyp/g` / `b3lypg`, libxc `HYB_GGA_XC_B3LYP`) against a CRYSTAL
reference: CRYSTAL14 cannot express that flavor, and the comparison
would inherit the ~10-15 mHa/heavy-atom flavor gap. PySCF is the
opposite: its bare `b3lyp` IS the Gaussian flavor, so PySCF
references for these decks must spell `b3lyp5`.
See `docs/user_guide/functionals.md` § B3LYP.
