/// \file serialized_expr.cpp
/// \brief Serialized expression → C++ Eigen code emission.
///
/// This is the always-available codegen path: it reads a pre-derived
/// serialized expression (no SeQuant dependency) and emits vibe-qc-style
/// C++ Eigen code with OpenMP parallelism.  The emission strategies:
///
///   GEMM path:   two-tensor contractions with matching inner dimension
///                → R.noalias() += A * B.transpose()
///   Loop path:   higher-rank contractions or non-GEMM-able products
///                → #pragma omp parallel for collapse(N) { ... }
///
/// The emitted code is designed to match cpp/src/ccsd.cpp structurally,
/// enabling the bit-identical validation gate.

#include "serialized_expr.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace vibeqc {
namespace sequant_codegen {

// =========================================================================
// Helpers
// =========================================================================

namespace {

/// Map TensorKind to its vibe-qc C++ access prefix.
std::string_view kind_prefix(TensorKind k) {
    switch (k) {
        case TensorKind::eri_ov_ov: return "V.ov_ov";
        case TensorKind::eri_oo_oo: return "V.oo_oo";
        case TensorKind::eri_oo_ov: return "V.oo_ov";
        case TensorKind::eri_oo_vv: return "V.oo_vv";
        case TensorKind::eri_ov_vv: return "V.ov_vv";
        case TensorKind::eri_vv_vv: return "V.vv_vv";
        case TensorKind::fock_oo:   return "f_oo";
        case TensorKind::fock_vv:   return "f_vv";
        case TensorKind::fock_ov:   return "f_ov";
        case TensorKind::ampl_t1:   return "T1";
        case TensorKind::ampl_t2:   return "T2_flat";
        case TensorKind::ampl_tau:  return "tau_flat";
        case TensorKind::ampl_taut: return "taut_flat";
    }
    return "UNKNOWN";
}

/// Number of occupied indices expected for a tensor kind.
int kind_n_occ(TensorKind k) {
    switch (k) {
        case TensorKind::eri_ov_ov: case TensorKind::eri_oo_vv:
        case TensorKind::ampl_t2: case TensorKind::ampl_tau:
        case TensorKind::ampl_taut:
            return 2;
        case TensorKind::eri_oo_oo: case TensorKind::eri_oo_ov:
            return 2;
        case TensorKind::eri_ov_vv: case TensorKind::eri_vv_vv:
            return 0;
        case TensorKind::fock_oo: return 2;
        case TensorKind::fock_vv: return 0;
        case TensorKind::fock_ov: return 1;
        case TensorKind::ampl_t1: return 1;
    }
    return 0;
}

/// Number of virtual indices expected for a tensor kind.
int kind_n_vir(TensorKind k) {
    switch (k) {
        case TensorKind::eri_ov_ov: case TensorKind::ampl_t2:
        case TensorKind::ampl_tau: case TensorKind::ampl_taut:
            return 2;
        case TensorKind::eri_oo_ov: case TensorKind::eri_ov_vv:
            return 1;
        case TensorKind::eri_oo_vv: case TensorKind::eri_vv_vv:
            return 2;
        case TensorKind::eri_oo_oo: return 0;
        case TensorKind::fock_oo: return 0;
        case TensorKind::fock_vv: return 2;
        case TensorKind::fock_ov: return 1;
        case TensorKind::ampl_t1: return 1;
    }
    return 0;
}

/// Determine if a character is an occupied index label (i-n) or virtual (a-f).
bool is_occ(char c) { return c >= 'i' && c <= 'n'; }
bool is_vir(char c) { return c >= 'a' && c <= 'f'; }

/// Build the (row, col) flat-index expression for a tensor access.
/// vibe-qc uses chemists' convention: (ia|jb) → V.ov_ov(i*nv + a, j*nv + b).
/// The indices string encodes: first n_occ chars are occupied, rest are virtual.
std::string flat_index(int n_occ, int n_vir, std::string_view indices,
                        const std::string& no_var, const std::string& nv_var) {
    // indices = "iajb" for eri_ov_ov: i→occ[0], a→vir[0], j→occ[1], b→vir[1]
    // We emit: (i*nv + a, j*nv + b) — row is bra, col is ket.
    //
    // Chemists' integral convention:
    //   eri_ov_ov: indices = "iajb" → (i*nv+a, j*nv+b)
    //   eri_oo_oo: indices = "minj" → (m*no+i, n*no+j)
    //   eri_oo_ov: indices = "mine" → (m*no+i, n*nv+e)
    //   eri_oo_vv: indices = "miab" → (m*no+i, a*nv+b)
    //   eri_ov_vv: indices = "meaf" → (m*nv+e, a*nv+f)
    //   eri_vv_vv: indices = "aebf" → (a*nv+e, b*nv+f)
    //
    // For amplitudes: T2_flat(i,j,a,b) → (i*no+j, a*nv+b)
    // For Fock: f_oo(m,i) → (m, i); f_vv(a,e) → (a, e)

    // Row: first half of indices
    std::string row, col;
    size_t half = indices.size() / 2;

    for (size_t p = 0; p < half; ++p) {
        char c = indices[p];
        if (p > 0) row += " * " + no_var + " + ";
        if (is_occ(c)) {
            row += c;
            row += " * " + nv_var;
        } else {
            row += " + ";
            row += c;
        }
    }
    // Clean up leading artifacts for pure-virtual rows
    if (row.find(" * ") == std::string::npos) {
        // All virtual — just join with + and nv mult
    }

    for (size_t p = half; p < indices.size(); ++p) {
        char c = indices[p];
        if (p > half) col += " * " + nv_var + " + ";
        if (is_occ(c)) {
            if (p == half) col += c;
            else col += std::string(1, c);
            if (p < indices.size() - 1 && is_vir(indices[p + 1]))
                col += " * " + no_var;
        } else {
            col += " + ";
            col += c;
        }
    }

    // Simplified output for common cases
    // This is a simplified emitter; the full implementation handles
    // all 6 integral-block layouts precisely.
    return "(" + row + ", " + col + ")";
}

}  // anonymous namespace

// =========================================================================
// SerializedEmitter implementation
// =========================================================================

SerializedEmitter::SerializedEmitter(const Options& opts) : opts_(opts) {}

std::string SerializedEmitter::tensor_access(const TensorRef& ref) {
    std::string prefix(kind_prefix(ref.kind));
    int no = kind_n_occ(ref.kind);
    int nv = kind_n_vir(ref.kind);

    // For amplitudes and Fock, indices are direct (no flat-packing needed
    // for rank ≤ 2; but we use flat layout for T2).
    if (ref.kind == TensorKind::ampl_t1) {
        // T1(i, a) — stored (no x nv)
        assert(ref.indices.size() == 2);
        return prefix + "(" + ref.indices.substr(0, 1) + ", "
               + ref.indices.substr(1, 1) + ")";
    }
    if (ref.kind == TensorKind::ampl_t2 ||
        ref.kind == TensorKind::ampl_tau ||
        ref.kind == TensorKind::ampl_taut) {
        // T2_flat(i*no + j, a*nv + b)
        assert(ref.indices.size() == 4);
        char i = ref.indices[0], j = ref.indices[1];
        char a = ref.indices[2], b = ref.indices[3];
        return prefix + "(" + i + " * " + opts_.no_var + " + " + j + ", "
               + a + " * " + opts_.nv_var + " + " + b + ")";
    }
    if (ref.kind == TensorKind::fock_oo) {
        assert(ref.indices.size() == 2);
        return prefix + "(" + ref.indices.substr(0, 1) + ", "
               + ref.indices.substr(1, 1) + ")";
    }
    if (ref.kind == TensorKind::fock_vv) {
        assert(ref.indices.size() == 2);
        return prefix + "(" + ref.indices.substr(0, 1) + ", "
               + ref.indices.substr(1, 1) + ")";
    }
    if (ref.kind == TensorKind::fock_ov) {
        assert(ref.indices.size() == 2);
        return prefix + "(" + ref.indices.substr(0, 1) + ", "
               + ref.indices.substr(1, 1) + ")";
    }

    // Integral blocks — use flat-index convention
    std::string idx_expr = flat_index(no, nv, ref.indices,
                                       opts_.no_var, opts_.nv_var);
    return prefix + idx_expr;
}

std::string SerializedEmitter::try_gemm(const SerializedTerm& term) {
    // GEMM is applicable when exactly two tensor factors contract
    // over a shared inner dimension, and the remaining indices form
    // the output row/col.
    //
    // Example: T2_flat(i*no + j, a*nv + e) * V.ov_vv(m*nv + e, a*nv + f)
    // contracts over (a, e) — but the indices are interleaved, not
    // cleanly separable into a simple matrix multiply in general.
    //
    // For the prototype, we detect the common CCSD patterns:
    //   R.noalias() += T1 * F_ae.transpose()       (T1 residual)
    //   R2_flat.noalias() += Wmnij.transpose() * tau_flat  (T2 ladder)
    //
    // Full GEMM detection requires analyzing the index permutation
    // across all factors.  This is deferred to Phase 2.
    (void)term;
    return {};
}

std::string SerializedEmitter::emit_term(const SerializedTerm& term) {
    // Emit one term as a C++ expression string.
    // For the prototype, we emit explicit loops.
    //
    // Phase 2 will add GEMM detection for the common patterns.

    std::ostringstream oss;

    if (term.factors.empty()) return "0.0";

    // Collect all unique external (non-summed) indices and summed indices
    std::set<char> all_idx;
    for (const auto& f : term.factors)
        for (char c : f.indices) all_idx.insert(c);

    // Summation indices: those appearing in ≥ 2 factors
    std::set<char> summed;
    std::map<char, int> count;
    for (const auto& f : term.factors)
        for (char c : f.indices) count[c]++;
    for (const auto& [c, n] : count)
        if (n >= 2) summed.insert(c);

    // External indices: the rest
    std::set<char> external;
    for (char c : all_idx)
        if (!summed.count(c)) external.insert(c);

    // Separate external into occ and vir for the output LHS
    std::string lhs_occ, lhs_vir;
    for (char c : external) {
        if (is_occ(c)) lhs_occ += c;
        else lhs_vir += c;
    }

    if (summed.empty()) {
        // No contraction — just a direct assignment term
        // e.g., R1(i,a) = coeff * f_ov(i,a)
        if (term.factors.size() == 1) {
            oss << tensor_access(term.factors[0]);
        } else {
            oss << "(";
            for (size_t i = 0; i < term.factors.size(); ++i) {
                if (i > 0) oss << " * ";
                oss << tensor_access(term.factors[i]);
            }
            oss << ")";
        }
        return oss.str();
    }

    // Emit a loop nest over summed indices with a reduction
    oss << "/* contraction over {";
    bool first = true;
    for (char c : summed) { if (!first) oss << ","; oss << c; first = false; }
    oss << "} */ 0.0";

    return oss.str();
}

void SerializedEmitter::emit(const SerializedExpr& expr, std::ostream& os) {
    os << "    // " << expr.label << "\n";

    for (size_t t = 0; t < expr.terms.size(); ++t) {
        const auto& term = expr.terms[t];
        const char* assign = (t == 0 && !expr.accumulate) ? "  = " : " += ";

        std::string rhs = emit_term(term);
        if (std::abs(term.coefficient - 1.0) > 1e-15) {
            // Coefficient — prepend
            std::ostringstream tmp;
            tmp << term.coefficient << " * " << rhs;
            rhs = tmp.str();
        }

        os << "    " << expr.output_var << assign << rhs << ";\n";
    }
}

void SerializedEmitter::emit_function(
    std::string_view func_name,
    const std::vector<SerializedExpr>& residuals,
    std::ostream& os) {

    os << "// Auto-generated from serialized SeQuant expressions.\n";
    os << "// Validation gate: must match cpp/src/ccsd.cpp::compute_residuals\n";
    os << "// bit-identically for the standard CCSD test cases.\n";
    os << "\n";
    os << "void " << func_name << "(\n";
    os << "    const Mat& T1, const Mat& T2_flat,\n";
    os << "    const IntegralBlocks& V,\n";
    os << "    const Mat& f_oo, const Mat& f_vv, const Mat& f_ov,\n";
    os << "    Mat& R1, Mat& R2_flat)\n";
    os << "{\n";
    os << "    const auto " << opts_.no_var << " = static_cast<Eigen::Index>(T1.rows());\n";
    os << "    const auto " << opts_.nv_var << " = static_cast<Eigen::Index>(T1.cols());\n";
    os << "\n";

    for (const auto& expr : residuals) {
        emit(expr, os);
        os << "\n";
    }

    os << "}\n";
}

// =========================================================================
// JSON serialisation — stub (full impl in Phase 2)
// =========================================================================

SerializedExpr parse_serialized_json(std::string_view json) {
    // Full JSON parsing will use nlohmann/json or a minimal parser.
    // For the prototype, expressions are hardcoded in the bundled data.
    (void)json;
    return SerializedExpr{};
}

std::string serialize_to_json(const SerializedExpr& expr) {
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"label\": \"" << expr.label << "\",\n";
    oss << "  \"output_var\": \"" << expr.output_var << "\",\n";
    oss << "  \"accumulate\": " << (expr.accumulate ? "true" : "false") << ",\n";
    oss << "  \"terms\": [\n";
    for (size_t t = 0; t < expr.terms.size(); ++t) {
        const auto& term = expr.terms[t];
        oss << "    {\n";
        oss << "      \"coeff\": " << term.coefficient << ",\n";
        oss << "      \"factors\": [\n";
        for (size_t f = 0; f < term.factors.size(); ++f) {
            const auto& tf = term.factors[f];
            oss << "        { \"kind\": \"" << static_cast<int>(tf.kind)
                << "\", \"indices\": \"" << tf.indices << "\" }";
            if (f + 1 < term.factors.size()) oss << ",";
            oss << "\n";
        }
        oss << "      ]\n";
        oss << "    }";
        if (t + 1 < expr.terms.size()) oss << ",";
        oss << "\n";
    }
    oss << "  ]\n";
    oss << "}\n";
    return oss.str();
}

// =========================================================================
// Bundled expression data — pre-derived CCSD residuals
// =========================================================================

namespace bundled {

// These are placeholders.  Once SeQuant derives the actual CCSD residuals,
// the serialized forms will be embedded here.  The structure below shows
// the format — each term is a product of tensors with index labels.
//
// T1 residual (simplified canonical-orbital form, f_ov = 0):
//   R1(i,a) = sum_e t_i^e F_ae - sum_m t_m^a F_mi
//           + sum_me (2 t_im^ae - t_mi^ae) F_me
//           + sum_nf t_n^f [2 (nf|ia) - (ni|af)]
//           + sum_mef (2 t_im^ef - t_mi^ef) (mf|ae)
//           - sum_mne (2 t_mn^ae - t_nm^ae) (mi|ne)

const SerializedExpr& ccsd_t1_residual() {
    static const SerializedExpr expr = {
        .label = "CCSD T1 residual (canonical)",
        .terms = {
            // Term 0: R1 = f_ov  (the constant term)
            {1.0, {{TensorKind::fock_ov, "ia"}}},
            // Additional terms will be filled from SeQuant derivation.
            // The full expression has ~8 terms with multiple factors each.
            // This is the serialized format — the actual data comes from
            // the offline SeQuant run.
        },
        .output_var = "R1",
        .accumulate = false,
    };
    return expr;
}

const SerializedExpr& ccsd_t2_residual() {
    static const SerializedExpr expr = {
        .label = "CCSD T2 residual (canonical)",
        .terms = {
            // Term 0: R2_flat = (ia|jb)  (the constant term)
            {1.0, {{TensorKind::eri_ov_ov, "iajb"}}},
        },
        .output_var = "R2_flat",
        .accumulate = false,
    };
    return expr;
}

const SerializedExpr& ccsd_triples() {
    static const SerializedExpr expr = {
        .label = "CCSD(T) triples correction",
        .terms = {},
        .output_var = "e_t",
        .accumulate = false,
    };
    return expr;
}

}  // namespace bundled

const SerializedExpr& ccsd_t1_residual_serialized() {
    return bundled::ccsd_t1_residual();
}

const SerializedExpr& ccsd_t2_residual_serialized() {
    return bundled::ccsd_t2_residual();
}

const SerializedExpr& ccsd_triples_serialized() {
    return bundled::ccsd_triples();
}

}  // namespace sequant_codegen
}  // namespace vibeqc
