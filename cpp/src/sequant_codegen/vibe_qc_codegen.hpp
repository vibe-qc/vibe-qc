#ifndef VIBEQC_SEQUANT_CODEGEN_HPP
#define VIBEQC_SEQUANT_CODEGEN_HPP

/// \file vibe_qc_codegen.hpp
/// \brief SeQuant expression → vibe-qc C++ Eigen code emitter.
///
/// Walks a SeQuant tensor expression DAG (typically the output of
/// mbpt::vac_av after projecting the BCH-expanded similarity-transformed
/// Hamiltonian) and emits C++17 Eigen code that matches vibe-qc's
/// conventions: Mat for dense tensors, RowMat for row-major, integral
/// block naming (V.ov_ov, V.oo_vv, ...), and OpenMP parallelism.
///
/// The emitted code is designed to be bit-identical to the existing
/// hand-coded kernels in cpp/src/ccsd.cpp for the CCSD validation gate.

#include <cstddef>
#include <ostream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace vibeqc {
namespace sequant_codegen {

// =========================================================================
// Integral block descriptor — maps SeQuant tensor labels to vibe-qc naming
// =========================================================================

/// Identifies one of vibe-qc's canonical integral-block tensors.
enum class IntegralBlock {
    ov_ov,   // (ia|jb), stored (i*nv+a, j*nv+b)
    oo_oo,   // (mi|nj), stored (m*no+i, n*no+j)
    oo_ov,   // (mi|ne), stored (m*no+i, n*nv+e)
    oo_vv,   // (mi|ab), stored (m*no+i, a*nv+b)
    ov_vv,   // (me|af), stored (m*nv+e, a*nv+f)
    vv_vv,   // (ae|bf), stored (a*nv+e, b*nv+f)
    f_oo,    // diagonal Fock: occupied block (canonical: ε_i δ_mi)
    f_vv,    // diagonal Fock: virtual block
    f_ov,    // off-diagonal Fock (zero for canonical reference)
};

/// Human-readable C++ access expression for an integral block.
std::string_view block_access(IntegralBlock b);

/// Describes an index's role in the current contraction context.
enum class IndexRole {
    occupied,       // i, j, k, l, m, n
    virtual_orb,    // a, b, c, d, e, f
    auxiliary,      // P, Q (density-fitting auxiliary)
    free,           // unrestricted
};

// =========================================================================
// Tensor descriptor — maps a SeQuant tensor to vibe-qc naming
// =========================================================================

/// Maps a SeQuant tensor label (wstring) to the vibe-qc tensor name
/// and its integral block category.
struct TensorMapping {
    std::string vibe_name;          // e.g. "T2_flat", "V.ov_ov"
    IntegralBlock block{IntegralBlock::ov_ov};
    bool is_amplitude{false};       // T1/T2 — stored separately from V
    int n_occ_indices{0};           // for determining loop structure
    int n_vir_indices{0};
};

// =========================================================================
// Code generation options
// =========================================================================

struct CodegenOptions {
    /// Number of occupied orbitals (variable name in emitted code).
    std::string no_var = "no";
    /// Number of virtual orbitals.
    std::string nv_var = "nv";
    /// OpenMP schedule clause (e.g., "static", "dynamic").
    std::string omp_schedule = "static";
    /// Use Eigen noalias() for GEMM-able contractions.
    bool use_noalias{true};
    /// Maximum tensor-contraction depth to unroll as explicit loops
    /// before falling back to GEMM (0 = always GEMM when possible).
    int max_unroll_rank{2};
    /// Print the SeQuant expression as a comment above the generated code.
    bool emit_se_comment{true};
};

// =========================================================================
// Code line — a single line of emitted C++
// =========================================================================

struct CodeLine {
    std::string text;
    int indent{0};
    bool is_omp_directive{false};
    bool is_comment{false};
};

// =========================================================================
// Expression visitor — the core codegen engine
// =========================================================================

/// Walks a SeQuant Expr (or its simplified textual/serialised form)
/// and emits vibe-qc-style C++ Eigen tensor-contraction code.
///
/// Because SeQuant is vendored as a build dependency, the visitor
/// operates on a serialised expression representation to avoid a
/// hard compile-time dependency in the public header. The actual
/// SeQuant ExprPtr traversal happens in the .cpp file behind a
/// pimpl.
class ExprEmitter {
public:
    explicit ExprEmitter(const CodegenOptions& opts = {});

    ~ExprEmitter();

    // ---- Configuration ----

    /// Register a tensor name mapping.
    void map_tensor(std::wstring_view se_label,
                    std::string_view vibe_name,
                    IntegralBlock block,
                    bool is_amplitude = false,
                    int n_occ = 0, int n_vir = 0);

    /// Set the name of the output variable.
    void set_output(std::string_view lhs);

    // ---- Emission ----

    /// Emit C++ code for a SeQuant expression pointer.
    /// @param expr   the SeQuant expression (ExprPtr)
    /// @param os     output stream
    void emit(void* se_expr_ptr, std::ostream& os);

    /// Emit C++ code from a pre-serialised expression.
    void emit_serialised(std::string_view serialised, std::ostream& os);

    /// Get the emitted code as a vector of lines (for post-processing).
    const std::vector<CodeLine>& lines() const { return lines_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::vector<CodeLine> lines_;
    CodegenOptions opts_;
};

// =========================================================================
// High-level entry points for specific methods
// =========================================================================

/// Derive and emit the closed-shell CCSD T1 + T2 residual kernel.
///
/// Uses SeQuant to:
///   1.  Define Ĥ(2), T̂(2)
///   2.  Compute H̄ via BCH(Ĥ, T̂, order=4)
///   3.  Project <Φ|a†_i a_a H̄|Φ> and <Φ|a†_i a†_j a_b a_a H̄|Φ>
///   4.  Spin-sum for closed-shell (spatial) equations
///   5.  Simplify and canonicalize
///   6.  Emit C++ Eigen code through ExprEmitter
///
/// @param os   output stream for the generated C++ function body
/// @param opts codegen options
/// @return true if emission succeeded, false on SeQuant unavailability
bool emit_ccsd_residuals(std::ostream& os, const CodegenOptions& opts = {});

/// Derive and emit the closed-shell (T) triples correction.
bool emit_ccsd_triples(std::ostream& os, const CodegenOptions& opts = {});

/// Derive and emit the CCSD Λ-equations (for properties/gradients).
bool emit_ccsd_lambda(std::ostream& os, const CodegenOptions& opts = {});

/// Derive and emit the MP2 energy expression.
bool emit_mp2_energy(std::ostream& os, const CodegenOptions& opts = {});

}  // namespace sequant_codegen
}  // namespace vibeqc

#endif  // VIBEQC_SEQUANT_CODEGEN_HPP
