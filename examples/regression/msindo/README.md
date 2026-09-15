# MSINDO reference oracle (parity validation)

This directory hosts the **out-of-process MSINDO reference harness** used to
validate vibe-qc's MSINDO implementation. Per CLAUDE.md §10, MSINDO is **not** a
vibe-qc runtime dependency — it is executed as an external subprocess and its
text output parsed independently, exactly like `runner_pyscf.py`. vibe-qc never
imports MSINDO.

MSINDO (Bredow, Geudtner & Jug) is © Mulliken Center for Theoretical Chemistry,
University of Bonn. It is used here only as a comparison oracle; the vibe-qc
MSINDO port is an independent re-implementation (see `docs/license.md`).

## Files

| File | Purpose |
|------|---------|
| `build_oracle.sh` | Reproducibly build a local MSINDO binary from a pristine MSINDO 2025 source tree. |
| `runner_msindo.py` | Build an MSINDO input, run the oracle, parse energies. Importable (`run_msindo`) + a CLI parity check. |
| `molecular_reference.json` | Canonical H–F geometries + MSINDO reference energies (the parity anchor). |

## Build

```sh
MSINDO_SRC=/path/to/msindo/2025e ./build_oracle.sh /tmp/msindo_oracle
```

The script copies the pristine source to a temp dir and applies three minimal,
documented patches **to the copy** (the reference checkout is never modified):

1. **`delimiter.h` `MV=5000 → 200`** — MSINDO statically allocates COMMON arrays
   dimensioned `O(MV²)` (e.g. `MADKONST(MV,2*MV)`). At `MV=5000` the `__common`
   segment is ~2.85 GB and collides with the macOS arm64 dyld shared-cache region
   (`map cache into shared region failed`). Reducing `MV` only caps the maximum
   system size (200 atoms ≫ the validation set); the physics is unchanged.
2. **`einles.f` add `LOGICAL :: EWALDMESH`** — a set-only keyword flag that is
   undeclared upstream; declaring it locally is inert.
3. **`vderf_shim.f`** — supplies MKL-VML `vderf`/`vderfc` (vectorized erf/erfc)
   via the standard `ERF`/`ERFC` intrinsics, since we link reference LAPACK/BLAS
   rather than MKL.

Toolchain: `gfortran -fdefault-real-8 -std=legacy -fallow-argument-mismatch
-fno-align-commons`, linked against homebrew `lapack`/`blas`.

## Run / verify

```sh
MSINDO_ORACLE=/tmp/msindo_oracle python3 runner_msindo.py   # parity table vs the JSON
```

Reference energies (RHF, Hartree), MSINDO 2025 (Jul. 2024 build):

| Molecule | Total energy |
|----------|--------------|
| H₂  | −1.1732166872 |
| HF  | −24.3089965036 |
| N₂  | −19.2858118276 |
| CO  | −21.6961833869 |
| CH₄ | −8.2956064178 |
| H₂O | −17.0182087674 |

> macOS note: the freshly-built binary maps the dyld shared cache; if you run it
> under a restrictive sandbox you may see `libSystem.B.dylib not loaded` — run it
> in a normal shell.
