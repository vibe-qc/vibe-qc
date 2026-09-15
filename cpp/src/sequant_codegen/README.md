# SeQuant → vibe-qc Code Generation

Prototype: derive CCSD equations from second-quantized operators using
SeQuant's MBPT framework, then emit C++ Eigen code matching vibe-qc's
conventions.

## Architecture

```
SeQuant (symbolic)
  │
  ├─ 1. Register index spaces (occupied i,j,k,l; virtual a,b,c,d)
  ├─ 2. Define Ĥ = Σ h_pq a†_p a_q + ½ Σ (pq|rs) a†_p a†_r a_s a_q
  ├─ 3. Define T̂ = T̂₁ + T̂₂
  ├─ 4. BCH: H̄ = e^(-T̂) Ĥ e^(T̂)
  ├─ 5. Project: <Φ|a†_i a_a H̄|Φ> = 0, <Φ|a†_i a†_j a_b a_a H̄|Φ> = 0
  ├─ 6. Spin-sum for closed-shell
  │
  ▼
SeQuant Expression DAG
  │
  ├─ Tensor canonicalization (exploit ERI 8-fold symmetry)
  ├─ Common subexpression identification
  │
  ▼
VibeQCCodegen (this library)
  │
  ├─ ExprVisitor: walk SeQuant expression tree
  ├─ IntegralBlockMapper: map SeQuant tensors to vibe-qc IntegralBlocks
  ├─ Emitter: generate C++ Eigen code with #pragma omp
  │
  ▼
Generated C++ (output)
  │
  └─ compute_residuals(T1, T2_flat, V, f_oo, f_vv, f_ov, R1, R2_flat)
     └─ Identical to cpp/src/ccsd.cpp (the validation gate)
```

## Pipeline Steps (the SeQuant side)

### Step 1: Set up the MBPT context

```cpp
using namespace sequant;
using namespace sequant::mbpt;

// Create index spaces
IndexSpace occ_space = IndexSpace::occupied(L"i");
IndexSpace vir_space = IndexSpace::unoccupied(L"a");

// Register them
auto isr = std::make_shared<IndexSpaceRegistry>();
isr->register_base_space(occ_space);
isr->register_base_space(vir_space);

// Set up context
Context ctx;
ctx.set_index_space_registry(isr);
set_default_context(ctx);

// Set up MBPT context
mbpt::Context mbpt_ctx({.op_registry_ptr = mbpt::make_minimal_registry()});
set_default_mbpt_context(mbpt_ctx);
```

### Step 2: Define operators

```cpp
// Hamiltonian (already built-in)
auto H_expr = H(2);  // up to 2-body

// Cluster operator
auto T_expr = T(2);  // T1 + T2
```

### Step 3: Similarity-transformed Hamiltonian

```cpp
// H̄ = e^(-T̂) Ĥ e^(T̂) via BCH expansion
// SeQuant provides this through its operator algebra
auto Hbar = bch_series(H_expr, T_expr, /*order=*/4);
```

### Step 4: Project onto excitation manifolds

```cpp
// <Φ_i^a| H̄ |Φ>   → T1 residual
// <Φ_ij^ab| H̄ |Φ>  → T2 residual

// Project with deexcitation operators
auto P1 = A(-1);  // rank-1 deexcitation: a†_i a_a
auto P2 = A(-2);  // rank-2 deexcitation: a†_i a†_j a_b a_a

auto R1_expr = vac_av(P1 * Hbar);
auto R2_expr = vac_av(P2 * Hbar);
```

### Step 5: Simplify and canonicalize

```cpp
R1_expr = simplify(R1_expr);
R2_expr = simplify(R2_expr);
```

The resulting `R1_expr` and `R2_expr` are symbolic tensor expressions
in SeQuant's DAG — sums of products of tensors (ERIs, amplitudes).

## Codegen Steps (this library)

### Emitter: walk the expression tree

The `VibeQCCodegen` visitor traverses the simplified SeQuant
expression and emits C++ Eigen code. For each `Product` node:

1. Identify the tensor factors (e.g., `T2_flat`, `V.ov_ov`)
2. Determine contraction indices
3. Emit appropriate loops or `Eigen::MatrixXd::noalias() += A * B`
   for GEMM-able contractions

### Integral block mapping

SeQuant's tensor names are mapped to vibe-qc's IntegralBlocks:

| SeQuant tensor | vibe-qc IntegralBlock | Layout |
|---|---|---|
| `g(i,a,j,b)` | `V.ov_ov` | `(i*nv+a, j*nv+b)` |
| `g(m,i,n,j)` | `V.oo_oo` | `(m*no+i, n*no+j)` |
| `g(m,i,n,e)` | `V.oo_ov` | `(m*no+i, n*nv+e)` |
| `g(m,i,a,b)` | `V.oo_vv` | `(m*no+i, a*nv+b)` |
| `g(m,e,a,f)` | `V.ov_vv` | `(m*nv+e, a*nv+f)` |
| `g(a,e,b,f)` | `V.vv_vv` | `(a*nv+e, b*nv+f)` |

## Validation Gate

The generated C++ `compute_residuals` must produce bit-identical results
to the existing hand-coded version in `cpp/src/ccsd.cpp`. The test:

```cpp
// tests/test_sequant_codegen_ccsd.cpp

// 1. Build IntegralBlocks from DF
// 2. Random T1, T2 amplitudes
// 3. Compute R1, R2 via existing hand-coded compute_residuals
// 4. Compute R1, R2 via generated compute_residuals_seq
// 5. Assert |R1 - R1_seq| < 1e-14, |R2 - R2_seq| < 1e-14
```

## File Layout

```
cpp/src/sequant_codegen/
  README.md                 ← this file
  vibe_qc_codegen.hpp       ← public header
  vibe_qc_codegen.cpp       ← visitor + emitter implementation
  integral_map.hpp          ← SeQuant tensor → IntegralBlock mapping
  integral_map.cpp
  codegen_ccsd.hpp          ← CCSD-specific codegen entry point
  codegen_ccsd.cpp
  CMakeLists.txt            ← build (conditionally includes SeQuant)

tests/
  test_sequant_codegen_ccsd.cpp  ← validation against hand-coded CCSD
```

## Dependencies

- **SeQuant** (vendored via FetchContent or system install)
- **vibe-qc core** (Eigen, IntegralBlocks, Mat/RowMat typedefs)
- **C++17** (matching both vibe-qc and SeQuant)

## Status

Phase 1 (prototype): CCSD T1/T2 residual derivation + C++ emission.
Gate: bit-identical to `cpp/src/ccsd.cpp::compute_residuals`.
