# QVF-Basis -- Conversion Fidelity Matrix

Which fields survive a round-trip from the QVF-Basis canonical model to
each target format and back?

| Field | QVF-Basis | BSE JSON | G94 | ORCA | NWChem |
|---|---|---|---|---|---|
| **schema_version** | ✅ | ✅ (set to "1.0.0") | ❌ | ❌ | ❌ |
| **name** | ✅ | ✅ | ❌ (comment only) | ❌ | ✅ (header) |
| **description** | ✅ | ✅ | ❌ | ❌ | ❌ |
| **role** | ✅ | ✅ | ❌ | ❌ | ❌ |
| **basis_family** | ✅ | ✅ | ❌ | ❌ | ❌ |
| **elements** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **element symbol** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **angular_momentum** | ✅ | ✅ | ✅ (via labels) | ✅ | ✅ |
| **SP shell** | ✅ (explicit [0,1]) | ✅ | ✅ (SP label) | ✅ (S+P blocks) | ✅ (S+P blocks) |
| **harmonic_type** | ✅ | ✅ (default: spherical) | ❌ | ❌ | ✅ (global header) |
| **exponents** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **coefficients** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **primitive ordering** | ✅ (descending) | ✅ | ✅ | ✅ | ✅ |
| **ecp_electrons** | ✅ | ✅ | ❌ | ❌ | ❌ |
| **ECP potentials** | ✅ | ✅ | ❌ (separate file) | ❌ | ❌ (separate block) |
| **references / DOI** | ✅ | ✅ | ❌ | ❌ | ❌ |
| **provenance** | ✅ | ✅ (version only) | ❌ | ❌ | ❌ |
| **parent basis link** | ✅ | ❌ (not in BSE) | ❌ | ❌ | ❌ |
| **segmented contraction** | ✅ (implicit) | ✅ | ✅ | ✅ | ✅ |
| **general contraction** | ✅ (implicit) | ✅ | ⚠️ (expanded) | ⚠️ (expanded) | ⚠️ (expanded) |

**Legend:**
- ✅ = round-trips faithfully
- ⚠️ = survives but with information loss (general → segmented expansion)
- ❌ = lost entirely

## Loss summaries by format

### G94
**7 fields dropped**: name, description, role, basis_family, references, provenance, harmonic_type.
ECPs must go in a separate file. General contractions are expanded to segmented form.

### ORCA
**7 fields dropped**: same as G94. The `%basis` block is an input directive, not a storage format.
ECPs are separate. General contractions are expanded.

### NWChem
**7 fields dropped** (same set). ECPs require a separate `ecp` directive (not emitted).
`harmonic_type` is partially preserved as global `spherical`/`cartesian` keyword.
