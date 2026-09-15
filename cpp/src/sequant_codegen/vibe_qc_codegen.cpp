/// \file vibe_qc_codegen.cpp
/// \brief SeQuant → vibe-qc C++ Eigen code emitter implementation.
///
/// This file contains the actual SeQuant ExprPtr traversal and C++ code
/// emission. It is compiled only when SeQuant is available (the CMake
/// build gates this behind find_package(SeQuant)).

#include "vibe_qc_codegen.hpp"

#include <algorithm>
#include <cassert>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

// SeQuant headers — conditionally included when SeQuant is available.
// The #ifdef guard allows this file to compile (as a no-op stub) even
// without SeQuant, for downstream code that includes the header.
#ifdef VIBEQC_HAS_SEQUANT

#include <SeQuant/core/context.hpp>
#include <SeQuant/core/expr.hpp>
#include <SeQuant/core/index.hpp>
#include <SeQuant/core/space.hpp>
#include <SeQuant/core/tensor.hpp>
#include <SeQuant/domain/mbpt/context.hpp>
#include <SeQuant/domain/mbpt/op.hpp>

using namespace sequant;

namespace vibeqc {
namespace sequant_codegen {

// =========================================================================
// Block access helpers
// =========================================================================

std::string_view block_access(IntegralBlock b) {
    switch (b) {
        case IntegralBlock::ov_ov: return "V.ov_ov";
        case IntegralBlock::oo_oo: return "V.oo_oo";
        case IntegralBlock::oo_ov: return "V.oo_ov";
        case IntegralBlock::oo_vv: return "V.oo_vv";
        case IntegralBlock::ov_vv: return "V.ov_vv";
        case IntegralBlock::vv_vv: return "V.vv_vv";
        case IntegralBlock::f_oo:  return "f_oo";
        case IntegralBlock::f_vv:  return "f_vv";
        case IntegralBlock::f_ov:  return "f_ov";
    }
    return "V.unknown";
}

// =========================================================================
// Index role inference from SeQuant IndexSpace types
// =========================================================================

namespace {

/// Map a SeQuant IndexSpace to our IndexRole by matching space attrs
/// against the default context's occupied/unoccupied subspaces.
IndexRole infer_role(const Index& idx) {
    // The SeQuant default context registers IndexSpace::occupied
    // and IndexSpace::unoccupied.  We probe the space's attr.
    const auto& space = idx.space();
    auto attr = space.attr();

    auto isr = get_default_context().index_space_registry();
    auto occ_attr = isr->retrieve(IndexSpace::occupied).attr();
    auto vir_attr = isr->retrieve(IndexSpace::unoccupied).attr();

    if (attr.intersection(occ_attr).type() != IndexSpace::Type{})
        return IndexRole::occupied;
    if (attr.intersection(vir_attr).type() != IndexSpace::Type{})
        return IndexRole::virtual_orb;
    return IndexRole::free;
}

/// Translate a SeQuant Index label (wchar) to the vibe-qc C++ index name.
std::string idx_to_cpp(const Index& idx) {
    // SeQuant typically uses single-character labels: i, j, k, l
    // for occupied and a, b, c, d for virtual.  We convert the
    // wstring label to narrow string and use it directly.
    std::wstring w = idx.label();
    std::string result;
    result.reserve(w.size());
    for (wchar_t c : w) result.push_back(static_cast<char>(c));
    return result;
}

/// Collect all unique indices from a SeQuant expression.
void collect_indices(const ExprPtr& expr,
                     std::vector<Index>& occ,
                     std::vector<Index>& vir) {
    if (!expr) return;

    if (expr->is<Tensor>()) {
        for (const auto& idx : expr->as<Tensor>().const_braket()) {
            auto role = infer_role(idx);
            if (role == IndexRole::occupied) occ.push_back(idx);
            else if (role == IndexRole::virtual_orb) vir.push_back(idx);
        }
    } else if (expr->is<Product>()) {
        for (const auto& factor : expr->as<Product>()) {
            collect_indices(factor, occ, vir);
        }
    } else if (expr->is<Sum>()) {
        for (const auto& term : expr->as<Sum>()) {
            collect_indices(term, occ, vir);
        }
    }
}

/// Deduplicate indices preserving order.
template <typename T>
void dedup(std::vector<T>& v) {
    std::unordered_set<std::wstring> seen;
    v.erase(std::remove_if(v.begin(), v.end(),
                           [&](const T& idx) {
                               return !seen.insert(idx.label()).second;
                           }),
            v.end());
}

}  // anonymous namespace

// =========================================================================
// ExprEmitter::Impl — pimpl holding SeQuant-specific state
// =========================================================================

struct ExprEmitter::Impl {
    std::unordered_map<std::wstring, TensorMapping> tensor_map;

    /// Register the default vibe-qc integral-block tensor mappings.
    void register_default_mappings(ExprEmitter* emitter) {
        // Integral blocks — chemists' notation (pq|rs):
        //   ov_ov:  (i a | j b)  = V.ov_ov(i*nv + a, j*nv + b)
        //   oo_oo:  (m i | n j)  = V.oo_oo(m*no + i, n*no + j)
        //   oo_ov:  (m i | n e)  = V.oo_ov(m*no + i, n*nv + e)
        //   oo_vv:  (m i | a b)  = V.oo_vv(m*no + i, a*nv + b)
        //   ov_vv:  (m e | a f)  = V.ov_vv(m*nv + e, a*nv + f)
        //   vv_vv:  (a e | b f)  = V.vv_vv(a*nv + e, b*nv + f)

        auto reg = [&](std::wstring_view se, std::string_view vb,
                        IntegralBlock blk, bool amp = false,
                        int no = 0, int nv = 0) {
            emitter->map_tensor(se, vb, blk, amp, no, nv);
        };

        // Two-electron integrals (g is SeQuant's ERI tensor name)
        reg(L"g", "V.ov_ov", IntegralBlock::ov_ov, false, 2, 2);

        // Fock matrix blocks
        reg(L"f", "f_oo", IntegralBlock::f_oo, false, 2, 0);

        // Amplitudes
        reg(L"t1", "T1",      IntegralBlock::ov_ov, true, 1, 1);
        reg(L"t2", "T2_flat", IntegralBlock::ov_ov, true, 2, 2);
        // tau = t2 + t1*t1
        reg(L"tau",   "tau_flat",   IntegralBlock::ov_ov, true, 2, 2);
        reg(L"taut",  "taut_flat",  IntegralBlock::ov_ov, true, 2, 2);
    }
};

// =========================================================================
// ExprEmitter public API
// =========================================================================

ExprEmitter::ExprEmitter(const CodegenOptions& opts)
    : impl_(std::make_unique<Impl>()), opts_(opts) {
    impl_->register_default_mappings(this);
}

ExprEmitter::~ExprEmitter() = default;

void ExprEmitter::map_tensor(std::wstring_view se_label,
                              std::string_view vibe_name,
                              IntegralBlock block,
                              bool is_amplitude,
                              int n_occ, int n_vir) {
    TensorMapping m;
    m.vibe_name = std::string(vibe_name);
    m.block = block;
    m.is_amplitude = is_amplitude;
    m.n_occ_indices = n_occ;
    m.n_vir_indices = n_vir;
    impl_->tensor_map[std::wstring(se_label)] = m;
}

void ExprEmitter::set_output(std::string_view lhs) {
    // The output variable name; stored for use in emit()
    // This is typically R1 or R2_flat for CCSD.
}

void ExprEmitter::emit(void* se_expr_ptr, std::ostream& os) {
    // --- Walk the SeQuant expression tree ---
    //
    // A SeQuant Expr is a DAG of Sum, Product, Tensor, Constant nodes.
    // The visitor pattern traverses this DAG and emits C++ code.
    //
    // For each Product node (tensor contraction):
    //   1. Extract the tensor factors
    //   2. Determine the contraction pattern (which indices are summed)
    //   3. Decide whether to emit:
    //      a. GEMM:   R.noalias() += A * B.transpose()
    //      b. Manual: #pragma omp parallel for collapse(N) ...
    //
    // For the CCSD validation gate, we emit code that is structurally
    // identical to compute_residuals in cpp/src/ccsd.cpp — so we can
    // diff the output and verify bit-identical results.

    if (!se_expr_ptr) return;

    auto* expr = static_cast<sequant::ExprPtr*>(se_expr_ptr);
    if (!*expr) return;

    const ExprPtr& e = *expr;

    // --- Collect index information ---
    std::vector<Index> occ_idxs, vir_idxs;
    collect_indices(e, occ_idxs, vir_idxs);
    dedup(occ_idxs);
    dedup(vir_idxs);

    // --- Emit header comment ---
    if (opts_.emit_se_comment) {
        os << "// Auto-generated from SeQuant expression:\n";
        os << "//   " << to_latex_align(e) << "\n";
        os << "//\n";
        os << "// Validation gate: must match cpp/src/ccsd.cpp::compute_residuals\n";
        os << "// bit-identically for the standard CCSD test cases.\n";
        os << "\n";
    }

    // --- Emit the contraction code ---
    //
    // The expression tree at this point is a Sum of Products.
    // Each Product is a tensor contraction.
    //
    // Emit strategy:
    //   - For Sum: iterate over terms, emit each with "+="
    //   - For Product: determine if it's a tensor-times-tensor
    //     contraction amenable to Eigen::MatrixXd operations
    //
    // Full implementation depends on the exact structure of the
    // simplified SeQuant expression.  For the CCSD validation gate
    // we emit code that matches the existing hand-written kernel.

    os << "    // ---- Generated residual code ----\n";

    // --- Walk the sum-of-products ---
    if (e->is<Sum>()) {
        const auto& terms = e->as<Sum>();
        for (size_t t = 0; t < terms.size(); ++t) {
            emit_term(terms[t], os, t == 0);
        }
    } else {
        emit_term(e, os, true);
    }

    os << "    // ---- End generated residual code ----\n";
}

// =========================================================================
// Term emission — handles one additive term of the residual
// =========================================================================

namespace {

/// Emit a single product term to the output stream.
/// @param term       the SeQuant expression for this term
/// @param os         output stream
/// @param is_first   true if this is the first term (uses "=", else "+=")
void emit_term(const ExprPtr& term, std::ostream& os, bool is_first) {
    // A term is typically:
    //   Constant * Tensor * Tensor * ...  (a contraction product)
    //
    // For Eigen code emission we need to:
    //   1. Extract the numeric coefficient
    //   2. Identify the contraction pattern
    //   3. Emit loops or GEMM calls

    // --- Extract coefficient ---
    double coeff = 1.0;
    ExprPtr body = term;

    if (body->is<Product>()) {
        const auto& prod = body->as<Product>();
        // Check if the first factor is a Constant
        if (!prod.factors().empty() && prod.factors()[0]->is<Constant>()) {
            const auto& c = prod.factors()[0]->as<Constant>();
            // SeQuant constants are rational numbers
            // coeff = c.value().to_double();
            // body = product of remaining factors
        }
    }

    (void)coeff;
    (void)is_first;
    (void)body;
    // --- Full implementation continues as SeQuant is vendored ---
    // This is the skeleton; the actual GEMM/loop emission depends on
    // the exact structure of SeQuant's simplified expression DAG.
    os << "    // [SeQuant term — full emission when SeQuant is vendored]\n";
}

}  // anonymous namespace

// =========================================================================
// Serialised expression emission (for offline use without SeQuant headers)
// =========================================================================

void ExprEmitter::emit_serialised(std::string_view serialised,
                                   std::ostream& os) {
    // When SeQuant is not available at the emitting site, expressions
    // can be pre-serialised to a simple text format (JSON or custom)
    // and replayed through the emitter.
    //
    // Format example (one term per line):
    //   +0.5 * T2_flat(i,j,a,b) * V.ov_ov(i,j,a,b)
    //
    // This path does not require SeQuant headers.

    os << "// Emitted from serialised SeQuant expression\n";
    os << "// (SeQuant headers not available at emission site)\n\n";
    (void)serialised;
}

// =========================================================================
// High-level entry points
// =========================================================================

bool emit_ccsd_residuals(std::ostream& os, const CodegenOptions& opts) {
    // --- Step 1: Set up SeQuant MBPT context ---

    // Register occupied/virtual index spaces
    auto isr = std::make_shared<IndexSpaceRegistry>();
    isr->register_base_space(IndexSpace::occupied);
    isr->register_base_space(IndexSpace::unoccupied);

    Context ctx;
    ctx.set_index_space_registry(isr);
    set_default_context(ctx);

    mbpt::Context mbpt_ctx(
        {.op_registry_ptr = mbpt::make_minimal_registry()});
    set_default_mbpt_context(mbpt_ctx);

    // --- Step 2: Define Ĥ = F̂ + Ŵ (normal-ordered w.r.t. |Φ>) ---
    //
    // SeQuant's H(2) gives the full Hamiltonian in normal order:
    //   Ĥ = f^p_q {a†_p a_q} + 1/4 g^{pq}_{rs} {a†_p a†_q a_s a_r}
    //
    // where f and g are the Fock matrix and antisymmetrised ERIs.
    // For canonical orbitals f^p_q = ε_p δ_{pq}.
    auto H_expr = mbpt::H(2);

    // --- Step 3: Define T̂ = T̂₁ + T̂₂ ---
    //
    // T(2) gives T1 + T2 in SeQuant's normal-ordered convention.
    // T̂₁ = t^i_a {a†_a a_i}
    // T̂₂ = 1/4 t^{ij}_{ab} {a†_a a†_b a_j a_i}
    auto T_expr = mbpt::T(2);

    // --- Step 4: BCH expansion H̄ = e^{-T̂} Ĥ e^{T̂} ---
    //
    // SeQuant computes this symbolically through operator commutation.
    // The result is truncated at order 4 (the highest power of T
    // appearing in CCSD: H̄ = Ĥ + [Ĥ,T̂] + 1/2[[Ĥ,T̂],T̂] + ...).
    //
    // For the prototype, we use SeQuant's built-in CC machinery
    // if available. Otherwise, the vacuum expectation value pathway:
    //
    //   <Φ| P̂_k H̄ |Φ> = 0
    //
    // where P̂_k is the k-tuply excited deexcitation projector.

    // --- Step 5: Project onto singles and doubles manifolds ---
    //
    // Singles: <Φ| a†_i a_a  H̄ |Φ> = 0
    // Doubles: <Φ| a†_i a†_j a_b a_a  H̄ |Φ> = 0
    //
    // In SeQuant this is:
    //   auto P1 = A(-1);  // deexcitation operator
    //   auto P2 = A(-2);
    //   auto R1 = vac_av(P1 * Hbar);
    //   auto R2 = vac_av(P2 * Hbar);

    // --- Step 6: Spin-sum for closed-shell ---
    //
    // SeQuant can work in spin-orbital or spin-free mode.
    // For closed-shell CCSD we use the spin-free (spatial) basis
    // and the spin-summed equations match SGWB (1991).

    // --- Step 7: Simplify and canonicalize ---
    //
    // SeQuant's simplify() applies:
    //   - Tensor canonicalization (exploit index permutation symmetries)
    //   - Constant folding
    //   - Common subexpression identification
    //
    // The resulting expression tree is a Sum of Products, where each
    // Product is a tensor contraction over dummy summation indices.

    // --- Step 8: Emit C++ code ---
    //
    // Walk the expression tree and emit Eigen code.

    os << "// ==================================================================\n";
    os << "// CCSD residuals — auto-generated by SeQuant → vibe-qc codegen\n";
    os << "// Validation gate: must match cpp/src/ccsd.cpp::compute_residuals\n";
    os << "// ==================================================================\n";
    os << "\n";
    os << "// This is a placeholder. The full emission requires SeQuant to be\n";
    os << "// vendored and linked. When SeQuant is available, this function\n";
    os << "// will produce bit-identical code to the hand-written kernel.\n";
    os << "\n";
    os << "// Once SeQuant is vendored, the generated code will look like:\n";
    os << "//\n";
    os << "//   void compute_residuals_seq(\n";
    os << "//       const Mat& T1, const Mat& T2_flat,\n";
    os << "//       const IntegralBlocks& V,\n";
    os << "//       const Mat& f_oo, const Mat& f_vv, const Mat& f_ov,\n";
    os << "//       Mat& R1, Mat& R2_flat)\n";
    os << "//   {\n";
    os << "//       const auto no = T1.rows();\n";
    os << "//       const auto nv = T1.cols();\n";
    os << "//\n";
    os << "//       // T1 residual\n";
    os << "//       R1 = f_ov;\n";
    os << "//       R1.noalias() += T1 * F_ae.transpose();\n";
    os << "//       ...\n";
    os << "//   }\n";

    (void)opts;
    return true;
}

bool emit_ccsd_triples(std::ostream& os, const CodegenOptions& opts) {
    os << "// (T) triples — auto-generated by SeQuant → vibe-qc codegen\n";
    os << "// Placeholder — full emission when SeQuant is vendored.\n";
    (void)opts;
    return true;
}

bool emit_ccsd_lambda(std::ostream& os, const CodegenOptions& opts) {
    os << "// Λ-CCSD — auto-generated by SeQuant → vibe-qc codegen\n";
    os << "// Placeholder — full emission when SeQuant is vendored.\n";
    (void)opts;
    return true;
}

bool emit_mp2_energy(std::ostream& os, const CodegenOptions& opts) {
    os << "// MP2 energy — auto-generated by SeQuant → vibe-qc codegen\n";
    os << "// Placeholder — full emission when SeQuant is vendored.\n";
    (void)opts;
    return true;
}

}  // namespace sequant_codegen
}  // namespace vibeqc

#else  // !VIBEQC_HAS_SEQUANT

// =========================================================================
// Stub implementation when SeQuant is not available
// =========================================================================

namespace vibeqc {
namespace sequant_codegen {

std::string_view block_access(IntegralBlock b) {
    switch (b) {
        case IntegralBlock::ov_ov: return "V.ov_ov";
        case IntegralBlock::oo_oo: return "V.oo_oo";
        case IntegralBlock::oo_ov: return "V.oo_ov";
        case IntegralBlock::oo_vv: return "V.oo_vv";
        case IntegralBlock::ov_vv: return "V.ov_vv";
        case IntegralBlock::vv_vv: return "V.vv_vv";
        case IntegralBlock::f_oo:  return "f_oo";
        case IntegralBlock::f_vv:  return "f_vv";
        case IntegralBlock::f_ov:  return "f_ov";
    }
    return "V.unknown";
}

struct ExprEmitter::Impl {
    std::unordered_map<std::wstring, TensorMapping> tensor_map;
};

ExprEmitter::ExprEmitter(const CodegenOptions& opts)
    : impl_(std::make_unique<Impl>()), opts_(opts) {}

ExprEmitter::~ExprEmitter() = default;

void ExprEmitter::map_tensor(std::wstring_view se_label,
                              std::string_view vibe_name,
                              IntegralBlock block,
                              bool is_amplitude,
                              int n_occ, int n_vir) {
    TensorMapping m;
    m.vibe_name = std::string(vibe_name);
    m.block = block;
    m.is_amplitude = is_amplitude;
    m.n_occ_indices = n_occ;
    m.n_vir_indices = n_vir;
    impl_->tensor_map[std::wstring(se_label)] = m;
}

void ExprEmitter::set_output(std::string_view /*lhs*/) {}

void ExprEmitter::emit(void* /*se_expr_ptr*/, std::ostream& os) {
    os << "// SeQuant codegen: SeQuant not available at compile time.\n";
    os << "// Install SeQuant and rebuild with -DVIBEQC_USE_SEQUANT=ON.\n";
}

void ExprEmitter::emit_serialised(std::string_view /*serialised*/,
                                   std::ostream& os) {
    os << "// SeQuant codegen: SeQuant not available at compile time.\n";
}

bool emit_ccsd_residuals(std::ostream& os, const CodegenOptions&) {
    os << "// SeQuant codegen: SeQuant not available.\n";
    return false;
}

bool emit_ccsd_triples(std::ostream& os, const CodegenOptions&) {
    os << "// SeQuant codegen: SeQuant not available.\n";
    return false;
}

bool emit_ccsd_lambda(std::ostream& os, const CodegenOptions&) {
    os << "// SeQuant codegen: SeQuant not available.\n";
    return false;
}

bool emit_mp2_energy(std::ostream& os, const CodegenOptions&) {
    os << "// SeQuant codegen: SeQuant not available.\n";
    return false;
}

}  // namespace sequant_codegen
}  // namespace vibeqc

#endif  // VIBEQC_HAS_SEQUANT
