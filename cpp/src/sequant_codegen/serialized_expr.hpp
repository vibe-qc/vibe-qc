/// \file serialized_expr.hpp
/// \brief Serialized SeQuant expression format for offline code generation.
///
/// SeQuant derives the CC equations, simplifies them, and serializes the
/// resulting expression DAG to a compact JSON/text format.  This header
/// reads that format and drives the C++ Eigen code emission — without
/// needing SeQuant at build time.  The pipeline is:
///
///   dev machine:  SeQuant derive → serialize → <method>.seq.json
///   CI / build:   read .seq.json → ExprEmitter → C++ Eigen code
///
/// This is analogous to how vibe-qc bundles basis sets (.g94 files):
/// the derivation is done once and the result is a data file checked
/// into the repo.  No runtime dependency on a CAS.

#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace vibeqc {
namespace sequant_codegen {

// =========================================================================
// Serialised expression DAG nodes
// =========================================================================

/// Identifies the kind of a serialised tensor factor.
enum class TensorKind : uint8_t {
    eri_ov_ov = 0,   // (ia|jb)  → V.ov_ov
    eri_oo_oo = 1,   // (mi|nj)  → V.oo_oo
    eri_oo_ov = 2,   // (mi|ne)  → V.oo_ov
    eri_oo_vv = 3,   // (mi|ab)  → V.oo_vv
    eri_ov_vv = 4,   // (me|af)  → V.ov_vv
    eri_vv_vv = 5,   // (ae|bf)  → V.vv_vv
    fock_oo   = 6,   // f_mi      → f_oo / F_mi / Fh_mj
    fock_vv   = 7,   // f_ae      → f_vv / F_ae / Fh_be
    fock_ov   = 8,   // f_ia      → f_ov / F_me
    ampl_t1   = 9,   // t_i^a     → T1
    ampl_t2   = 10,  // t_ij^ab   → T2_flat
    ampl_tau  = 11,  // τ_ij^ab   → tau_flat
    ampl_taut = 12,  // τ̃_ij^ab  → taut_flat
};

/// One tensor reference in a contraction product.
struct TensorRef {
    TensorKind kind;
    /// Index labels (single characters: i, j, a, b, …).
    /// Order: bra indices first, then ket indices.
    std::string indices;
};

/// A single term in the residual:  coeff * Product(tensors...)
struct SerializedTerm {
    double coefficient = 1.0;
    std::vector<TensorRef> factors;
};

/// A full residual expression: sum of terms.
struct SerializedExpr {
    /// Human-readable label (e.g. "CCSD T1 residual").
    std::string label;
    /// The additive terms.
    std::vector<SerializedTerm> terms;
    /// Output variable name in emitted code.
    std::string output_var = "R1";
    /// Whether the output starts from zero (=) or accumulates (+=).
    bool accumulate = false;
};

// =========================================================================
// Serialised-expression emitter — no SeQuant needed
// =========================================================================

/// Emits C++ Eigen code from a pre-serialised expression.
///
/// Unlike ExprEmitter (which walks a live SeQuant ExprPtr), this class
/// operates on the simple SerializedExpr DAG.  It is always available
/// regardless of whether SeQuant is linked.
class SerializedEmitter {
public:
    struct Options {
        std::string no_var = "no";
        std::string nv_var = "nv";
        std::string omp_schedule = "static";
        bool use_noalias = true;
        int max_unroll_rank = 2;
    };

    SerializedEmitter() : SerializedEmitter(Options{}) {}
    explicit SerializedEmitter(const Options& opts);

    /// Emit C++ code for one serialised expression.
    void emit(const SerializedExpr& expr, std::ostream& os);

    /// Emit a complete function wrapping the expression.
    /// Produces:
    ///   void <func_name>(const Mat& T1, const Mat& T2_flat,
    ///                    const IntegralBlocks& V,
    ///                    const Mat& f_oo, const Mat& f_vv,
    ///                    const Mat& f_ov,
    ///                    Mat& R1, Mat& R2_flat) { ... }
    void emit_function(std::string_view func_name,
                       const std::vector<SerializedExpr>& residuals,
                       std::ostream& os);

private:
    Options opts_;

    /// Emit one term. Returns the C++ expression string for the RHS.
    std::string emit_term(const SerializedTerm& term);

    /// Map tensor kind + indices to a C++ access expression.
    /// e.g., TensorKind::eri_ov_ov + "iajb" → "V.ov_ov(i*nv + a, j*nv + b)"
    std::string tensor_access(const TensorRef& ref);

    /// Determine whether two tensors form a GEMM-able contraction.
    /// Returns the GEMM expression if true, empty string otherwise.
    std::string try_gemm(const SerializedTerm& term);
};

// =========================================================================
// Serialisation format (JSON) — reading and writing
// =========================================================================

/// Parse a JSON-serialised expression.
/// Format:
///   {
///     "label": "CCSD T1 residual",
///     "output_var": "R1",
///     "accumulate": false,
///     "terms": [
///       {
///         "coeff": 1.0,
///         "factors": [
///           {"kind": "fock_ov", "indices": "ia"},
///           ...
///         ]
///       }
///     ]
///   }
SerializedExpr parse_serialized_json(std::string_view json);

/// Serialise an expression to JSON (for the offline derivation tool).
std::string serialize_to_json(const SerializedExpr& expr);

// =========================================================================
// Bundled expression data — CCSD residuals (pre-derived)
// =========================================================================

/// Returns the pre-derived CCSD T1 residual in serialised form.
/// This is computed once offline with SeQuant and checked into the repo.
/// When SeQuant is not available at build time, this bundled data drives
/// the codegen instead.
const SerializedExpr& ccsd_t1_residual_serialized();

/// Returns the pre-derived CCSD T2 residual in serialised form.
const SerializedExpr& ccsd_t2_residual_serialized();

/// Returns the pre-derived (T) triples expression.
const SerializedExpr& ccsd_triples_serialized();

}  // namespace sequant_codegen
}  // namespace vibeqc
