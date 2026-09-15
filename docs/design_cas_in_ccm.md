# Design: CAS-in-CCM integration (embed → CCM bridge)

**Status**: DRAFT, gated behind v2.0 HF-CCM + v2.3 CCM orbital
localisation.  This document defines the integration contract so the
CCM chat can consume the embed module without re-deriving it.

**Date**: 2026-06-19
**Branch**: `embedded-cluster-cas`

---

## 1. The two embedding strategies

vibe-qc will ship two routes to periodic multireference:

| Route | Embedding | Status |
|---|---|---|
| **Embedded-cluster** (MR6-MR10) | External Madelung array + AIMP shell, hand to molecular CAS solvers | ✅ Production |
| **CAS-in-CCM** (v2.x) | CCM itself is the embedding; run CAS on the cyclic cluster | 🔮 Design |

The embedded-cluster route reaches OpenMolcas parity.  CAS-in-CCM is
the vibe-qc-native route, same molecular CAS solvers, but the
embedding is the CCM framework instead of an external AIMP library.

---

## 2. What the embed module provides

The `vibeqc.embed` package exposes these integration points for CCM:

```python
from vibeqc.embed import (
    carve_cluster,         # PeriodicSystem → finite cluster
    embedding_target,      # V(infinite lattice) − V(QM formal charges)
    evjen_array,           # Evjen boundary-corrected Madelung array
    fitted_array,          # Derenzo-Klintenberg-Weber fitted array
    embedded_pieces,       # Build S, Hcore, E_nuc, jk from charges
    embedded_mo_hamiltonian,  # AO→MO transform with V_ext in h1e
    offsite_probe_ball,    # Random off-site probe grid
    parse_aimp_file,       # OpenMolcas AIMP data parser
    write_aimp_ecp_library, # AIMP→libecpint XML converter
)
```

Each function is self-contained and can be called independently.
The `embed_cluster` driver composes them; CCM can compose differently.

---

## 3. Integration point 1: carve the CCM cluster

The CCM framework produces a finite cyclic cluster (PeriodicSystem with
cyclic boundary conditions).  The `carve_cluster` function is the
geometry-only step, it takes a PeriodicSystem and returns a finite
Molecule.  CCM may use this as-is, or may have its own carve logic.

```python
mol, atom_records = carve_cluster(ccm_system, center, radius)
# mol: vq.Molecule with the carved atoms
# atom_records: [(Z, xyz_bohr), ...] for building embedding arrays
```

---

## 4. Integration point 2: build the embedding Hamiltonian

The CCM Hamiltonian is H_CCM = H_cluster + H_embed.  The embed module
provides H_embed via the Madelung array (or AIMP shell):

```python
# Build the Madelung array from CCM's periodic density.
from vibeqc.embed._madelung import (
    _crystal_sites_within, embedding_target,
    fitted_array, offsite_probe_ball,
)
from vibeqc.embed._external_field import embedded_pieces

# 1. Define the embedding target from the CCM density-derived charges.
probes = offsite_probe_ball(center, all_sites, probe_radius, n_probes)
v_target = embedding_target(ccm_system, atom_records, formal_charges, probes)

# 2. Build a fitted array (or use Evjen for parameter-free convergence).
pos, q_array, info = fitted_array(
    ccm_system, atom_records, center, formal_charges,
    probes, v_target, r_exact=..., r_fit=..., ridge=1e-4,
)

# 3. Add AIMP shell (optional, for frontier-ion exchange/orthogonality).
#    Parse AIMP data from user's OpenMolcas install.
from vibeqc.embed import parse_aimp_file, resolve_aimp_library
aimp_entries = parse_aimp_file(resolve_aimp_library())
# Select bare entries for the crystal's elements.
# (See embed_cluster source for the full AIMP wiring.)

# 4. Build the embedded Hamiltonian.
S, Hcore, E_nuc, jk = embedded_pieces(
    mol, basis, ext_pos=pos, ext_q=q_array,
    ecp_centers=aimp_centers,           # if AIMP
    ecp_library=aimp_lib_name,
    ecp_share_dir=aimp_share_dir,
)
# Hcore now carries T + V_nuc(cluster) + V_madelung + V_aimp
```

---

## 5. Integration point 3: active-space construction

The molecular CAS solvers (`casci`, `casscf`, `caspt2`) take an
MO-basis Hamiltonian.  CCM must:

1. Solve the CCM SCF (produces MO coefficients C).
2. Transform Hcore to the MO basis via `embedded_mo_hamiltonian`.
3. Select an active space (CASCI/CASSCF) or all MOs (CASPT2 on-top).
4. Feed to the solvers.

```python
from vibeqc.embed._external_field import embedded_mo_hamiltonian
from vibeqc.solvers import casci, casscf, caspt2

# CCM SCF → MO coefficients C
H_mo = embedded_mo_hamiltonian(mol, basis, C, Hcore, E_nuc)
# H_mo.h1e, H_mo.h2e, H_mo.nuclear_repulsion

# Active space selection (CCM localisation → local MOs → active)
ci = casci(H_mo.h1e, H_mo.h2e, nae, nao,
           n_core=n_core, nuclear_repulsion=H_mo.nuclear_repulsion)
sc = casscf(H_mo.h1e, H_mo.h2e, nae, nao,
            n_core=n_core, nuclear_repulsion=H_mo.nuclear_repulsion)
```

---

## 6. Differences from the embedded-cluster route

| Aspect | Embedded-cluster | CAS-in-CCM |
|---|---|---|
| Cluster definition | `carve_cluster` (sphere) | CCM cyclic cluster |
| Embedding | External Madelung array | CCM periodic embedding |
| AIMP | External AIMP library (LGPL) | Not needed (CCM is the embedding) |
| Active space | User-specified | CCM-localised MOs |
| Parity target | OpenMolcas | CCM itself (v2.0 HF-CCM as reference) |

---

## 7. What the CCM chat needs from the embed module

1. **Madelung array builders**: `evjen_array`, `fitted_array` work on
   any PeriodicSystem; CCM provides the system + formal charges.
2. **External-field Hamiltonian**: `embedded_pieces` builds Hcore with
   arbitrary external charges; CCM passes its own array.
3. **AIMP parser**: if CCM wants AIMP on the frontier (optional).
4. **Convergence protocol**: MR9's cluster-size scan methodology
   applies equally to CCM cluster-size convergence.
5. **Parity test pattern**: MR10's parametrised test suite is the
   template for CCM parity tests.

The key design principle: **the molecular CAS solvers are shared**.
Both routes produce an MO-basis Hamiltonian and feed it to the same
`casci`/`casscf`/`caspt2` functions.  The only difference is how the
Hamiltonian is built.
