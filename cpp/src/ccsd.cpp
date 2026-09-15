// Closed-shell CCSD and perturbative (T) on an RHF reference.  The MO
// two-electron integrals are density-fitted by default
// (CCSDOptions.density_fit = true) or exact four-index on the canonical
// conventional route (density_fit = false); everything downstream of the
// integral-block assembly is shared between the two.
//
// Equation provenance (see also the header block in vibeqc/ccsd.hpp):
//
//   The CCSD residuals below are the closed-shell spin integration of the
//   spin-orbital working equations of J. F. Stanton, J. Gauss, J. D. Watts,
//   R. J. Bartlett, J. Chem. Phys. 94, 4334 (1991), doi:10.1063/1.460620
//   (SGWB), Eqs. (1)-(13).  The spin-orbital form is implemented verbatim in
//   python/vibeqc/dlpno/_ccsd_ref.py (the FCI-anchored in-repo reference);
//   every spatial equation in this file was validated against that kernel
//   to machine precision (|dr| < 1e-11 at random amplitudes) before being
//   transcribed here, and the converged correlation energies agree with the
//   reference to < 1e-9 Ha on H2O/STO-3G and H2O/def2-SVP
//   (tests/test_ccsd_anchor.py).
//
//   Density fitting of all two-electron integrals follows
//   A. E. DePrince III, C. D. Sherrill, J. Chem. Theory Comput. 9, 2687
//   (2013), doi:10.1021/ct400250u (DF-CCSD with three-index intermediates).
//   Original closed-shell CCSD: G. D. Purvis III, R. J. Bartlett,
//   J. Chem. Phys. 76, 1910 (1982), doi:10.1063/1.443164.
//
//   The (T) correction is K. Raghavachari, G. W. Trucks, J. A. Pople,
//   M. Head-Gordon, Chem. Phys. Lett. 157, 479 (1989),
//   doi:10.1016/0009-2614(89)87395-6, in the standard formulation
//     D t3c = P(i/jk) P(a/bc) [ sum_e t_jk^ae <ei||bc>
//                             - sum_m t_im^bc <ma||jk> ]
//     D t3d = P(i/jk) P(a/bc) [ t_i^a <jk||bc> + f_i^a t_jk^bc ]
//     E(T)  = (1/36) sum_so t3c * D * (t3c + t3d)
//   spin-integrated CLASSWISE for a closed-shell reference (see
//   compute_triples below), again validated to machine precision against
//   the spin-orbital evaluation of the same formulas.
//
// Notation (spatial orbitals): i, j, k, l, m, n = occupied;
// a, b, c, d, e, f = virtual; P = auxiliary.  Chemists' integrals
//   (pq|rs) = sum_P B^P_pq B^P_rs                  (DF factorisation)
// with the MO-basis three-index tensor B^P_pq; on the canonical route the
// same (pq|rs) blocks come from the exact AO tensor via
// eri_mo_pair_transform (build_integral_blocks_canonical below).  Amplitudes:
//   t_i^a               T1, stored (no x nv)
//   t_ij^ab             T2 alpha-beta convention, stored flat
//                       T2_flat(i*no+j, a*nv+b); t_ij^ab = t_ji^ba
//   tau_ij^ab  = t_ij^ab + t_i^a t_j^b
//   taut_ij^ab = t_ij^ab + (1/2) t_i^a t_j^b
//
// Integral block layout (flat Eigen matrices, built once per run):
//   V_ov_ov(i*nv+a, j*nv+b) = (ia|jb)
//   V_oo_oo(m*no+i, n*no+j) = (mi|nj)
//   V_oo_ov(m*no+i, n*nv+e) = (mi|ne)
//   V_oo_vv(m*no+i, a*nv+b) = (mi|ab)
//   V_ov_vv(m*nv+e, a*nv+f) = (me|af)
//   V_vv_vv(a*nv+e, b*nv+f) = (ae|bf)
//
// The Fock diagonal is kept inside the F intermediates, so the residuals
// returned by compute_residuals are true residuals: they vanish at the
// CCSD fixed point (equivalent to SGWB's explicit "- t * D" form).

#include "vibeqc/ccsd.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vibeqc/correlation_conventions.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "vibeqc/df.hpp"
#include "vibeqc/integrals.hpp"

namespace vibeqc {

namespace {

using Mat = Eigen::MatrixXd;
using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                             Eigen::RowMajor>;

// Disk-backed triples stores the two DF factor blocks as interleaved auxiliary
// rows.  A worker therefore needs only one bounded P-row block in memory and a
// sequential read for each term-A contraction.  The file is deliberately an
// internal, process-local format: it is deleted by the store destructor on
// normal return and on every exception-unwind path.
class DiskDFFactorStore {
public:
    static constexpr std::uint64_t kMagic = 0x5649514354334446ULL;
    static constexpr std::size_t kHeaderBytes = 4 * sizeof(std::uint64_t);

    static std::shared_ptr<DiskDFFactorStore> spill(
        const RowMat& B_ov, const RowMat& B_vv,
        const std::string& scratch_directory) {
        if (B_ov.rows() < 1 || B_ov.rows() != B_vv.rows())
            throw std::invalid_argument(
                "CCSD(T) disk triples requires compatible non-empty "
                "B_ov/B_vv factors");

        namespace fs = std::filesystem;
        std::error_code ec;
        fs::path directory = scratch_directory.empty()
            ? fs::temp_directory_path(ec) : fs::path(scratch_directory);
        if (ec || !fs::is_directory(directory, ec) || ec)
            throw std::runtime_error(
                "CCSD(T) triples scratch directory does not exist or is "
                "not a directory");

        static std::atomic<std::uint64_t> serial{0};
        const auto clock_seed = static_cast<std::uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
        fs::path path;
        for (int attempt = 0; attempt < 1024; ++attempt) {
            const auto id = clock_seed + serial.fetch_add(1);
            const fs::path candidate = directory /
                ("vibeqc-triples-" + std::to_string(id) + ".bin");
            if (!fs::exists(candidate, ec) && !ec) {
                path = candidate;
                break;
            }
            ec.clear();
        }
        if (path.empty())
            throw std::runtime_error(
                "CCSD(T) could not reserve a unique triples scratch file");

        try {
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            if (!out)
                throw std::runtime_error(
                    "CCSD(T) could not create the triples scratch file");
            const std::uint64_t header[4] = {
                kMagic,
                static_cast<std::uint64_t>(B_ov.rows()),
                static_cast<std::uint64_t>(B_ov.cols()),
                static_cast<std::uint64_t>(B_vv.cols())};
            out.write(reinterpret_cast<const char*>(header), sizeof(header));
            for (Eigen::Index P = 0; P < B_ov.rows(); ++P) {
                out.write(reinterpret_cast<const char*>(B_ov.row(P).data()),
                          static_cast<std::streamsize>(
                              B_ov.cols() * sizeof(double)));
                out.write(reinterpret_cast<const char*>(B_vv.row(P).data()),
                          static_cast<std::streamsize>(
                              B_vv.cols() * sizeof(double)));
            }
            out.close();
            if (!out)
                throw std::runtime_error(
                    "CCSD(T) failed while writing the triples scratch file");
        } catch (...) {
            fs::remove(path, ec);
            throw;
        }

        return std::shared_ptr<DiskDFFactorStore>(new DiskDFFactorStore(
            std::move(path), B_ov.rows(), B_ov.cols(), B_vv.cols()));
    }

    ~DiskDFFactorStore() noexcept {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }

    DiskDFFactorStore(const DiskDFFactorStore&) = delete;
    DiskDFFactorStore& operator=(const DiskDFFactorStore&) = delete;

    const std::filesystem::path& path() const noexcept { return path_; }
    Eigen::Index n_aux() const noexcept { return n_aux_; }
    Eigen::Index ov_cols() const noexcept { return ov_cols_; }
    Eigen::Index vv_cols() const noexcept { return vv_cols_; }
    Eigen::Index row_width() const noexcept { return ov_cols_ + vv_cols_; }
    std::size_t disk_bytes() const noexcept { return disk_bytes_; }

private:
    DiskDFFactorStore(std::filesystem::path path, Eigen::Index n_aux,
                      Eigen::Index ov_cols, Eigen::Index vv_cols)
        : path_(std::move(path)), n_aux_(n_aux), ov_cols_(ov_cols),
          vv_cols_(vv_cols),
          disk_bytes_(kHeaderBytes
                      + static_cast<std::size_t>(n_aux)
                            * static_cast<std::size_t>(ov_cols + vv_cols)
                            * sizeof(double)) {}

    std::filesystem::path path_;
    Eigen::Index n_aux_ = 0;
    Eigen::Index ov_cols_ = 0;
    Eigen::Index vv_cols_ = 0;
    std::size_t disk_bytes_ = 0;
};

class DiskDFFactorReader {
public:
    DiskDFFactorReader(const DiskDFFactorStore& store,
                       Eigen::Index block_rows)
        : store_(store), block_rows_(std::max<Eigen::Index>(1, block_rows)),
          buffer_(static_cast<std::size_t>(block_rows_)
                  * static_cast<std::size_t>(store.row_width())),
          stream_(store.path(), std::ios::binary) {
        // Regression-only hook for the post-spill read-error cleanup path.
        // Public execution never sets this environment variable.
        const char* inject = std::getenv(
            "VIBEQC_TEST_TRIPLES_DISK_READ_FAILURE");
        inject_read_failure_ = inject != nullptr && std::string(inject) == "1";
        if (!stream_)
            throw std::runtime_error(
                "CCSD(T) could not open the triples scratch file for reading");
    }

    void rewind() {
        stream_.clear();
        stream_.seekg(static_cast<std::streamoff>(
            DiskDFFactorStore::kHeaderBytes));
        if (!stream_)
            throw std::runtime_error(
                "CCSD(T) could not seek in the triples scratch file");
        next_p_ = 0;
        rows_loaded_ = 0;
    }

    bool load_next() {
        if (next_p_ >= store_.n_aux()) return false;
        if (inject_read_failure_ && !read_failure_injected_) {
            read_failure_injected_ = true;
            throw std::runtime_error(
                "CCSD(T) injected triples scratch read failure");
        }
        rows_loaded_ = std::min(block_rows_, store_.n_aux() - next_p_);
        const std::size_t count = static_cast<std::size_t>(rows_loaded_)
            * static_cast<std::size_t>(store_.row_width());
        stream_.read(reinterpret_cast<char*>(buffer_.data()),
                     static_cast<std::streamsize>(count * sizeof(double)));
        if (!stream_)
            throw std::runtime_error(
                "CCSD(T) failed while reading the triples scratch file");
        next_p_ += rows_loaded_;
        return true;
    }

    Eigen::Index rows_loaded() const noexcept { return rows_loaded_; }
    double ov(Eigen::Index local_p, Eigen::Index column) const noexcept {
        return buffer_[static_cast<std::size_t>(local_p)
                           * static_cast<std::size_t>(store_.row_width())
                       + static_cast<std::size_t>(column)];
    }
    double vv(Eigen::Index local_p, Eigen::Index column) const noexcept {
        return buffer_[static_cast<std::size_t>(local_p)
                           * static_cast<std::size_t>(store_.row_width())
                       + static_cast<std::size_t>(store_.ov_cols() + column)];
    }

private:
    const DiskDFFactorStore& store_;
    Eigen::Index block_rows_ = 1;
    std::vector<double> buffer_;
    std::ifstream stream_;
    Eigen::Index next_p_ = 0;
    Eigen::Index rows_loaded_ = 0;
    bool inject_read_failure_ = false;
    bool read_failure_injected_ = false;
};

// =========================================================================
// DF block set: MO-basis three-index tensors over the correlated window
// =========================================================================
struct DFBlocks {
    RowMat B_ov;  // (n_aux, no * nv):  B^P_{ia}
    RowMat B_vv;  // (n_aux, nv * nv):  B^P_{ab}
    RowMat B_oo;  // (n_aux, no * no):  B^P_{ij}
    Eigen::Index n_aux;
    Eigen::Index no;
    Eigen::Index nv;
};

DFBlocks build_df_blocks(const DensityFitting& df,
                         const Mat& C_occ, const Mat& C_vir) {
    DFBlocks b;
    b.no = static_cast<Eigen::Index>(C_occ.cols());
    b.nv = static_cast<Eigen::Index>(C_vir.cols());
    b.n_aux = static_cast<Eigen::Index>(df.n_aux());
    b.B_ov = df.mo_transform(C_occ, C_vir);
    b.B_vv = df.mo_transform(C_vir, C_vir);
    b.B_oo = df.mo_transform(C_occ, C_occ);
    return b;
}

// =========================================================================
// Four-index integral blocks assembled once from the DF tensors.
// =========================================================================
struct IntegralBlocks {
    Mat ov_ov;  // (ia|jb)
    Mat oo_oo;  // (mi|nj)
    Mat oo_ov;  // (mi|ne)
    Mat oo_vv;  // (mi|ab)
    Mat ov_vv;  // (me|af)
    Mat vv_vv;  // (ae|bf) — materialised only when it fits the in-core budget;
                // empty otherwise, in which case the W_abef ladder regenerates
                // (ae|bf) panels from B_vv, tiled over the bra-virtual index a.
    RowMat B_ov;  // (n_aux, no*nv) — retained for factor-direct triples.
    RowMat B_vv;  // (n_aux, nv*nv) — retained for the blocked CCSD ladder and
                  // factor-direct triples without holding ov_vv/vv_vv cubes.
    std::shared_ptr<DiskDFFactorStore> disk_factors;
};

// In-core (ae|bf) is O(nv^4); above this many bytes we skip materialising it
// and regenerate it blocked from B_vv inside the W_abef ladder instead. The
// W_abef intermediate is itself O(nv^4), so the in-core peak is ~2*nv^4; the
// ~2 GiB default keeps the fast path for nv up to ~125 (the usual def2-SVP /
// TZVP small-molecule CCSD) and only tiles beyond it. Override at runtime with
// VIBEQC_CCSD_VVVV_INCORE_BYTES (0 forces the blocked path — used by the
// in-core/blocked parity test).
constexpr std::size_t kIncoreVvvvBudgetBytes = 2ULL << 30;  // 2 GiB

std::size_t incore_vvvv_budget_bytes() {
    if (const char* e = std::getenv("VIBEQC_CCSD_VVVV_INCORE_BYTES")) {
        try {
            return static_cast<std::size_t>(std::stoull(e));
        } catch (...) {
            // ignore a malformed override; fall through to the default
        }
    }
    return kIncoreVvvvBudgetBytes;
}

// Row budget for one blocked (ae|bf) panel (acur*nv rows). Default 2048;
// VIBEQC_CCSD_VVVV_TILE_ROWS overrides it (the parity test sets it to 1 to
// force one bra-virtual per tile and exercise the multi-tile offset logic).
Eigen::Index vvvv_tile_rows() {
    if (const char* e = std::getenv("VIBEQC_CCSD_VVVV_TILE_ROWS")) {
        try {
            const long v = std::stol(e);
            if (v > 0) return static_cast<Eigen::Index>(v);
        } catch (...) {
            // ignore a malformed override; fall through to the default
        }
    }
    return 2048;
}

// `need_vvvv=false` skips the O(nv^4) (ae|bf) build entirely. Only the
// W_abef ladder consumes it, so a caller that contracts that term elsewhere
// (the DLPNO pair-space ladder, issue #700) must not pay for it here.
IntegralBlocks build_integral_blocks(DFBlocks&& df, bool need_vvvv = true) {
    IntegralBlocks V;
    V.ov_ov = df.B_ov.transpose() * df.B_ov;
    V.oo_oo = df.B_oo.transpose() * df.B_oo;
    V.oo_ov = df.B_oo.transpose() * df.B_ov;
    V.oo_vv = df.B_oo.transpose() * df.B_vv;
    V.ov_vv = df.B_ov.transpose() * df.B_vv;
    // Keep B_vv (n_aux*nv^2, tiny vs nv^4) for the blocked ladder; only
    // materialise the full (ae|bf) tensor when it fits the in-core budget.
    const std::size_t nv = static_cast<std::size_t>(df.nv);
    const std::size_t vvvv_bytes = nv * nv * nv * nv * sizeof(double);
    if (need_vvvv && vvvv_bytes <= incore_vvvv_budget_bytes()) {
        V.vv_vv = df.B_vv.transpose() * df.B_vv;
    }
    // Transfer the factors after every four-index block has been formed.  A
    // move matters here: copying B_ov/B_vv would create a second factor set at
    // the integral-build peak before the short-lived DFBlocks object dies.
    V.B_ov = std::move(df.B_ov);
    V.B_vv = std::move(df.B_vv);
    return V;
}

// Canonical (non-DF) counterpart of build_integral_blocks: the same MO
// integral blocks, assembled from the exact four-index AO ERI tensor via
// eri_mo_pair_transform over the correlated window [C_occ | C_vir].  This
// is the conventional coupled-cluster route (CCSDOptions.density_fit =
// false), bit-exact in the integrals rather than RI-fitted -- the reference
// configuration for parity against conventional CCSD(T) in other programs.
// Memory: holds the AO tensor (nao^4) and the correlated-window MO tensor
// ((no+nv)^4) transiently, so it is a small-molecule route by design; the
// DF path remains the default.  (ae|bf) is always materialised in-core --
// there is no B_vv here for the blocked ladder to regenerate panels from.
IntegralBlocks build_integral_blocks_canonical(const BasisSet& basis,
                                               const Mat& C_occ,
                                               const Mat& C_vir) {
    const Eigen::Index no = C_occ.cols();
    const Eigen::Index nv = C_vir.cols();
    const Eigen::Index nm = no + nv;
    Mat C_corr(C_occ.rows(), nm);
    C_corr << C_occ, C_vir;

    // W(p*nm + q, r*nm + s) = (pq|rs), correlated window, occupied first.
    const Mat W = [&] {
        const Eri4D eri = compute_eri(basis);
        return eri_mo_pair_transform(eri, C_corr, C_corr);
    }();

    IntegralBlocks V;
    V.ov_ov.resize(no * nv, no * nv);
    V.oo_oo.resize(no * no, no * no);
    V.oo_ov.resize(no * no, no * nv);
    V.oo_vv.resize(no * no, nv * nv);
    V.ov_vv.resize(no * nv, nv * nv);
    V.vv_vv.resize(nv * nv, nv * nv);
    #pragma omp parallel for collapse(2) schedule(static)
    for (Eigen::Index i = 0; i < no; ++i)
        for (Eigen::Index a = 0; a < nv; ++a)
            for (Eigen::Index j = 0; j < no; ++j)
                for (Eigen::Index b = 0; b < nv; ++b)
                    V.ov_ov(i * nv + a, j * nv + b) =
                        W(i * nm + no + a, j * nm + no + b);
    #pragma omp parallel for collapse(2) schedule(static)
    for (Eigen::Index m = 0; m < no; ++m)
        for (Eigen::Index i = 0; i < no; ++i) {
            for (Eigen::Index n = 0; n < no; ++n)
                for (Eigen::Index j = 0; j < no; ++j)
                    V.oo_oo(m * no + i, n * no + j) =
                        W(m * nm + i, n * nm + j);
            for (Eigen::Index n = 0; n < no; ++n)
                for (Eigen::Index e = 0; e < nv; ++e)
                    V.oo_ov(m * no + i, n * nv + e) =
                        W(m * nm + i, n * nm + no + e);
            for (Eigen::Index a = 0; a < nv; ++a)
                for (Eigen::Index b = 0; b < nv; ++b)
                    V.oo_vv(m * no + i, a * nv + b) =
                        W(m * nm + i, (no + a) * nm + no + b);
        }
    #pragma omp parallel for collapse(2) schedule(static)
    for (Eigen::Index m = 0; m < no; ++m)
        for (Eigen::Index e = 0; e < nv; ++e)
            for (Eigen::Index a = 0; a < nv; ++a)
                for (Eigen::Index f = 0; f < nv; ++f)
                    V.ov_vv(m * nv + e, a * nv + f) =
                        W(m * nm + no + e, (no + a) * nm + no + f);
    #pragma omp parallel for collapse(2) schedule(static)
    for (Eigen::Index a = 0; a < nv; ++a)
        for (Eigen::Index e = 0; e < nv; ++e)
            for (Eigen::Index b = 0; b < nv; ++b)
                for (Eigen::Index f = 0; f < nv; ++f)
                    V.vv_vv(a * nv + e, b * nv + f) =
                        W((no + a) * nm + no + e, (no + b) * nm + no + f);
    return V;
}

Mat build_mo_fock(const Mat& C, const Mat& F_ao) {
    return C.transpose() * F_ao * C;
}

// =========================================================================
// Coupled-pair / QCI variant selector (CCSDOptions::cc_variant).  The
// linear variants (LCCD / LCCSD=CEPA(0) / CEPA(n)) reuse the canonical
// compute_residuals below through an algebraically exact linear-part
// extraction -- see compute_linear_residuals -- so there is exactly one
// implementation of the amplitude equations in the codebase.  QCISD uses
// the same philosophy with a bivariate (T1,T2) monomial projector.
// =========================================================================
enum class CCVariant {
    CCSD,
    CC2,
    CCD,
    BCCD,
    LCCD,
    CEPA0,
    CEPA1,
    CEPA2,
    CEPA3,
    QCISD
};

CCVariant parse_cc_variant(const std::string& name) {
    std::string k;
    k.reserve(name.size());
    for (const char c : name) {
        if (c == ' ' || c == '_' || c == '-') continue;
        k.push_back(static_cast<char>(
            std::tolower(static_cast<unsigned char>(c))));
    }
    if (k.empty() || k == "ccsd") return CCVariant::CCSD;
    if (k == "cc2") return CCVariant::CC2;
    if (k == "ccd") return CCVariant::CCD;
    if (k == "bccd") return CCVariant::BCCD;
    if (k == "lccd") return CCVariant::LCCD;
    if (k == "lccsd" || k == "cepa(0)" || k == "cepa0") return CCVariant::CEPA0;
    if (k == "cepa(1)" || k == "cepa1") return CCVariant::CEPA1;
    if (k == "cepa(2)" || k == "cepa2") return CCVariant::CEPA2;
    if (k == "cepa(3)" || k == "cepa3") return CCVariant::CEPA3;
    if (k == "qcisd") return CCVariant::QCISD;
    throw std::invalid_argument(
        "run_ccsd: unknown cc_variant '" + name +
        "'; supported: ccsd, cc2, ccd, bccd, lccd, lccsd (= cepa(0)), cepa(1), "
        "cepa(2), cepa(3), qcisd");
}

// T1 is identically zero for the doubles-only variants.
bool variant_has_singles(CCVariant v) {
    return v == CCVariant::CCSD || v == CCVariant::CC2 ||
           v == CCVariant::CEPA0 ||
           v == CCVariant::CEPA1 || v == CCVariant::CEPA2 ||
           v == CCVariant::CEPA3 || v == CCVariant::QCISD;
}

// Linearized amplitude equations (everything except CCSD / CCD).
bool variant_is_linear(CCVariant v) {
    return v == CCVariant::LCCD || v == CCVariant::CEPA0 ||
           v == CCVariant::CEPA1 || v == CCVariant::CEPA2 ||
           v == CCVariant::CEPA3;
}

// =========================================================================
// CCSD correlation energy (closed-shell):
//   E_corr = 2 sum_ia f_ia t_i^a
//          + sum_ijab [2 (ia|jb) - (ib|ja)] tau_ij^ab
// (spin integration of SGWB Eq. (25); tau = t2 + t1 t1)
//
// The linear variants (LCCD / LCCSD / CEPA) use the linear energy
// functional: same expression with tau -> t2 (the quadratic t1 t1 term
// belongs to the exponential CC ansatz, not to a CI-like linear wave
// operator).  tau_energy=false selects that form.
// =========================================================================
double compute_energy(const Mat& T1, const Mat& T2_flat,
                      const Mat& f_ov, const Mat& V_ov_ov,
                      Eigen::Index no, Eigen::Index nv,
                      bool tau_energy = true) {
    double e = 2.0 * (f_ov.array() * T1.array()).sum();

    #pragma omp parallel for reduction(+:e) schedule(static)
    for (Eigen::Index i = 0; i < no; ++i)
        for (Eigen::Index j = 0; j < no; ++j)
            for (Eigen::Index a = 0; a < nv; ++a)
                for (Eigen::Index b = 0; b < nv; ++b) {
                    const double tau = T2_flat(i * no + j, a * nv + b)
                                     + (tau_energy
                                            ? T1(i, a) * T1(j, b)
                                            : 0.0);
                    const double ia_jb = V_ov_ov(i * nv + a, j * nv + b);
                    const double ib_ja = V_ov_ov(i * nv + b, j * nv + a);
                    e += tau * (2.0 * ia_jb - ib_ja);
                }
    return e;
}

// =========================================================================
// CCSD residuals: closed-shell spin integration of SGWB (1991).
//
// One-body intermediates (SGWB Eqs. (3)-(5), spin-summed; Fock diagonal
// kept inside F_ae / F_mi so the returned residuals are true residuals):
//
//   F_me(m,e) = f_me + sum_nf t_n^f [2(me|nf) - (mf|ne)]
//   F_ae(a,e) = f_ae - 1/2 sum_m f_me t_m^a
//             + sum_mf t_m^f [2(mf|ae) - (me|af)]
//             - sum_mnf taut_mn^af [2(me|nf) - (mf|ne)]
//   F_mi(m,i) = f_mi + 1/2 sum_e t_i^e f_me
//             + sum_ne t_n^e [2(mi|ne) - (me|ni)]
//             + sum_nef taut_in^ef [2(me|nf) - (mf|ne)]
//
// T1 residual (SGWB Eq. (1), spin-summed):
//
//   r_i^a = f_ia + sum_e t_i^e F_ae - sum_m t_m^a F_mi
//         + sum_me [2 t_im^ae - t_mi^ae] F_me
//         + sum_nf t_n^f [2(nf|ai) - (ni|af)]
//         + sum_mef [2 t_im^ef - t_mi^ef] (mf|ae)
//         - sum_mne [2 t_mn^ae - t_nm^ae] (mi|ne)
//
// Two-body intermediates (SGWB Eqs. (6)-(8), alpha-beta external blocks):
//
//   W_mnij = (mi|nj) + sum_e t_j^e (mi|ne) + sum_e t_i^e (nj|me)
//          + 1/2 sum_ef tau_ij^ef (me|nf)
//   W_abef = (ae|bf) - sum_m t_m^b (mf|ae) - sum_m t_m^a (me|bf)
//          + 1/2 sum_mn tau_mn^ab (me|nf)
//
//   Ring intermediates, stored W(m*nv+e, j*nv+b); three spin flavours:
//   W1 (alpha-beta), W2 (same-spin), WX (cross "mbje" flavour):
//
//   W1 = (me|jb) + sum_f t_j^f (me|bf) - sum_n t_n^b (nj|me)
//      + sum_nf [t_nj^fb - 1/2 t_jn^fb - t_j^f t_n^b] (me|nf)
//      - 1/2 sum_nf t_nj^fb (mf|ne)
//   W2 = (me|jb) - (mj|be)
//      + sum_f t_j^f [(me|bf) - (mf|be)]
//      - sum_n t_n^b [(nj|me) - (mj|ne)]
//      - sum_nf [1/2 (t_jn^fb - t_jn^bf) + t_j^f t_n^b]
//                                  [(me|nf) - (mf|ne)]
//      + 1/2 sum_nf t_nj^fb (me|nf)
//   WX = -(mj|be) - sum_f t_j^f (mf|be) + sum_n t_n^b (mj|ne)
//      + sum_nf [1/2 t_jn^fb + t_j^f t_n^b] (mf|ne)
//
// T2 residual (SGWB Eq. (2), alpha-beta external block; with
// Ph f(i,j,a,b) = f(i,j,a,b) + f(j,i,b,a) the pair-exchange symmetriser):
//
//   r_ij^ab = (ia|jb)
//           + sum_mn tau_mn^ab W_mnij
//           + sum_ef tau_ij^ef W_abef
//           + Ph[ sum_e t_ij^ae Fh_be - sum_m t_im^ab Fh_mj
//               + sum_me (t_im^ae - t_mi^ae) W1(me,jb)
//               + sum_me t_im^ae W2(me,jb)
//               - sum_me t_i^e t_m^a (me|jb)
//               + sum_me t_mj^ae WX(me,ib)
//               - sum_me t_j^e t_m^a (mi|be)
//               + sum_e t_i^e (ae|jb)
//               - sum_m t_m^a (mi|jb) ]
//   with Fh_be = F_ae(b,e) - 1/2 sum_m t_m^b F_me(m,e)
//        Fh_mj = F_mi(m,j) + 1/2 sum_e t_j^e F_me(m,e)
// =========================================================================
void compute_residuals(const Mat& T1, const Mat& T2_flat,
                       const IntegralBlocks& V,
                       const Mat& f_oo, const Mat& f_vv, const Mat& f_ov,
                       Mat& R1, Mat& R2_flat,
                       Eigen::Index target_i = -1,
                       Eigen::Index target_j = -1,
                       bool include_ladder = true,
                       bool include_ring = true,
                       Mat* W1_out = nullptr, Mat* W2_out = nullptr,
                       Mat* WX_out = nullptr) {
    const auto no = static_cast<Eigen::Index>(T1.rows());
    const auto nv = static_cast<Eigen::Index>(T1.cols());
    const bool target_only = target_i >= 0 || target_j >= 0;
    if (target_only
        && (target_i < 0 || target_i >= no || target_j < 0 || target_j >= no)) {
        throw std::invalid_argument(
            "compute_residuals: target occupied indices are out of range");
    }
    const Eigen::Index target_ij =
        target_only ? target_i * no + target_j : -1;

    // ---- tau amplitudes ----
    Mat tau_flat = T2_flat;
    Mat taut_flat = T2_flat;
    #pragma omp parallel for collapse(2) schedule(static)
    for (Eigen::Index i = 0; i < no; ++i)
        for (Eigen::Index j = 0; j < no; ++j) {
            const Eigen::Index ij = i * no + j;
            for (Eigen::Index a = 0; a < nv; ++a)
                for (Eigen::Index b = 0; b < nv; ++b) {
                    const double t1t1 = T1(i, a) * T1(j, b);
                    tau_flat(ij, a * nv + b) += t1t1;
                    taut_flat(ij, a * nv + b) += 0.5 * t1t1;
                }
        }

    // ---- one-body intermediates ----
    // F_me(m,e) = f_me + sum_nf t_n^f [2(me|nf) - (mf|ne)]
    Mat F_me = f_ov;
    #pragma omp parallel for collapse(2) schedule(static)
    for (Eigen::Index m = 0; m < no; ++m)
        for (Eigen::Index e = 0; e < nv; ++e) {
            double s = 0.0;
            for (Eigen::Index n = 0; n < no; ++n)
                for (Eigen::Index f = 0; f < nv; ++f)
                    s += T1(n, f) * (2.0 * V.ov_ov(m * nv + e, n * nv + f)
                                     - V.ov_ov(m * nv + f, n * nv + e));
            F_me(m, e) += s;
        }

    // F_ae(a,e) = f_ae - 1/2 sum_m f_me t_m^a
    //           + sum_mf t_m^f [2(mf|ae) - (me|af)]
    //           - sum_mnf taut_mn^af [2(me|nf) - (mf|ne)]
    Mat F_ae = f_vv;
    #pragma omp parallel for collapse(2) schedule(static)
    for (Eigen::Index a = 0; a < nv; ++a)
        for (Eigen::Index e = 0; e < nv; ++e) {
            double s = 0.0;
            for (Eigen::Index m = 0; m < no; ++m) {
                s -= 0.5 * f_ov(m, e) * T1(m, a);
                for (Eigen::Index f = 0; f < nv; ++f)
                    s += T1(m, f) * (2.0 * V.ov_vv(m * nv + f, a * nv + e)
                                     - V.ov_vv(m * nv + e, a * nv + f));
            }
            for (Eigen::Index m = 0; m < no; ++m)
                for (Eigen::Index n = 0; n < no; ++n)
                    for (Eigen::Index f = 0; f < nv; ++f)
                        s -= taut_flat(m * no + n, a * nv + f)
                           * (2.0 * V.ov_ov(m * nv + e, n * nv + f)
                              - V.ov_ov(m * nv + f, n * nv + e));
            F_ae(a, e) += s;
        }

    // F_mi(m,i) = f_mi + 1/2 sum_e t_i^e f_me
    //           + sum_ne t_n^e [2(mi|ne) - (ni|me)]
    //           + sum_nef taut_in^ef [2(me|nf) - (mf|ne)]
    Mat F_mi = f_oo;
    #pragma omp parallel for collapse(2) schedule(static)
    for (Eigen::Index m = 0; m < no; ++m)
        for (Eigen::Index i = 0; i < no; ++i) {
            double s = 0.0;
            for (Eigen::Index e = 0; e < nv; ++e)
                s += 0.5 * T1(i, e) * f_ov(m, e);
            for (Eigen::Index n = 0; n < no; ++n)
                for (Eigen::Index e = 0; e < nv; ++e)
                    s += T1(n, e) * (2.0 * V.oo_ov(m * no + i, n * nv + e)
                                     - V.oo_ov(n * no + i, m * nv + e));
            for (Eigen::Index n = 0; n < no; ++n)
                for (Eigen::Index e = 0; e < nv; ++e)
                    for (Eigen::Index f = 0; f < nv; ++f)
                        s += taut_flat(i * no + n, e * nv + f)
                           * (2.0 * V.ov_ov(m * nv + e, n * nv + f)
                              - V.ov_ov(m * nv + f, n * nv + e));
            F_mi(m, i) += s;
        }

    // ---- T1 residual ----
    R1 = f_ov;
    R1.noalias() += T1 * F_ae.transpose();
    R1.noalias() -= F_mi.transpose() * T1;
    #pragma omp parallel for collapse(2) schedule(static)
    for (Eigen::Index i = 0; i < no; ++i)
        for (Eigen::Index a = 0; a < nv; ++a) {
            if (target_only && i != target_i) continue;
            double s = 0.0;
            // + sum_me [2 t_im^ae - t_mi^ae] F_me
            for (Eigen::Index m = 0; m < no; ++m)
                for (Eigen::Index e = 0; e < nv; ++e)
                    s += (2.0 * T2_flat(i * no + m, a * nv + e)
                          - T2_flat(m * no + i, a * nv + e)) * F_me(m, e);
            // + sum_nf t_n^f [2(nf|ia) - (ni|af)]
            for (Eigen::Index n = 0; n < no; ++n)
                for (Eigen::Index f = 0; f < nv; ++f)
                    s += T1(n, f) * (2.0 * V.ov_ov(n * nv + f, i * nv + a)
                                     - V.oo_vv(n * no + i, a * nv + f));
            // + sum_mef [2 t_im^ef - t_mi^ef] (mf|ae)
            for (Eigen::Index m = 0; m < no; ++m)
                for (Eigen::Index e = 0; e < nv; ++e)
                    for (Eigen::Index f = 0; f < nv; ++f)
                        s += (2.0 * T2_flat(i * no + m, e * nv + f)
                              - T2_flat(m * no + i, e * nv + f))
                           * V.ov_vv(m * nv + f, a * nv + e);
            // - sum_mne [2 t_mn^ae - t_nm^ae] (mi|ne)
            for (Eigen::Index m = 0; m < no; ++m)
                for (Eigen::Index n = 0; n < no; ++n)
                    for (Eigen::Index e = 0; e < nv; ++e)
                        s -= (2.0 * T2_flat(m * no + n, a * nv + e)
                              - T2_flat(n * no + m, a * nv + e))
                           * V.oo_ov(m * no + i, n * nv + e);
            R1(i, a) += s;
        }

    // ---- two-body intermediates ----
    // W_mnij(m*no+n, i*no+j)
    Mat Wmnij = Mat::Zero(no * no, no * no);
    #pragma omp parallel for collapse(2) schedule(static)
    for (Eigen::Index m = 0; m < no; ++m)
        for (Eigen::Index n = 0; n < no; ++n) {
            const Eigen::Index mn = m * no + n;
            for (Eigen::Index i = 0; i < no; ++i)
                for (Eigen::Index j = 0; j < no; ++j) {
                    if (target_only && (i != target_i || j != target_j)) {
                        continue;
                    }
                    double s = V.oo_oo(m * no + i, n * no + j);
                    for (Eigen::Index e = 0; e < nv; ++e) {
                        s += T1(j, e) * V.oo_ov(m * no + i, n * nv + e);
                        s += T1(i, e) * V.oo_ov(n * no + j, m * nv + e);
                    }
                    for (Eigen::Index e = 0; e < nv; ++e)
                        for (Eigen::Index f = 0; f < nv; ++f)
                            s += 0.5 * tau_flat(i * no + j, e * nv + f)
                               * V.ov_ov(m * nv + e, n * nv + f);
                    Wmnij(mn, i * no + j) = s;
                }
        }

    // The particle-particle ladder (W_abef and its R2 contraction) is built
    // fused below, after the R2 base + W_mnij ladder, tiled over the
    // bra-virtual index a so the O(nv^4) (ae|bf) panel need not be resident.

    // Ring intermediates W1, W2, WX stored (m*nv+e, j*nv+b). They have no
    // other consumer, so a caller that neither contracts them here nor wants
    // them back skips their build and their 3 * O(o^2 nv^2) storage (#700).
    const bool need_ring = include_ring || W1_out || W2_out || WX_out;
    Mat W1, W2, WX;
    if (need_ring) {
    W1 = Mat::Zero(no * nv, no * nv);
    W2 = Mat::Zero(no * nv, no * nv);
    WX = Mat::Zero(no * nv, no * nv);
    #pragma omp parallel for collapse(2) schedule(static)
    for (Eigen::Index m = 0; m < no; ++m)
        for (Eigen::Index e = 0; e < nv; ++e) {
            const Eigen::Index me = m * nv + e;
            for (Eigen::Index j = 0; j < no; ++j)
                for (Eigen::Index b = 0; b < nv; ++b) {
                    if (target_only && j != target_i && j != target_j) {
                        continue;
                    }
                    double s1 = V.ov_ov(me, j * nv + b);
                    double s2 = V.ov_ov(me, j * nv + b)
                              - V.oo_vv(m * no + j, b * nv + e);
                    double sx = -V.oo_vv(m * no + j, b * nv + e);
                    for (Eigen::Index f = 0; f < nv; ++f) {
                        const double tjf = T1(j, f);
                        s1 += tjf * V.ov_vv(me, b * nv + f);
                        s2 += tjf * (V.ov_vv(me, b * nv + f)
                                     - V.ov_vv(m * nv + f, b * nv + e));
                        sx -= tjf * V.ov_vv(m * nv + f, b * nv + e);
                    }
                    for (Eigen::Index n = 0; n < no; ++n) {
                        const double tnb = T1(n, b);
                        s1 -= tnb * V.oo_ov(n * no + j, me);
                        s2 -= tnb * (V.oo_ov(n * no + j, me)
                                     - V.oo_ov(m * no + j, n * nv + e));
                        sx += tnb * V.oo_ov(m * no + j, n * nv + e);
                    }
                    for (Eigen::Index n = 0; n < no; ++n)
                        for (Eigen::Index f = 0; f < nv; ++f) {
                            const double t_njfb =
                                T2_flat(n * no + j, f * nv + b);
                            const double t_jnfb =
                                T2_flat(j * no + n, f * nv + b);
                            const double t_jnbf =
                                T2_flat(j * no + n, b * nv + f);
                            const double t1t1 = T1(j, f) * T1(n, b);
                            const double v_d = V.ov_ov(me, n * nv + f);
                            const double v_x =
                                V.ov_ov(m * nv + f, n * nv + e);
                            s1 += (t_njfb - 0.5 * t_jnfb - t1t1) * v_d
                                - 0.5 * t_njfb * v_x;
                            const double tss =
                                0.5 * (t_jnfb - t_jnbf) + t1t1;
                            s2 += -tss * (v_d - v_x) + 0.5 * t_njfb * v_d;
                            sx += (0.5 * t_jnfb + t1t1) * v_x;
                        }
                    W1(me, j * nv + b) = s1;
                    W2(me, j * nv + b) = s2;
                    WX(me, j * nv + b) = sx;
                }
        }
    }
    if (W1_out) *W1_out = W1;
    if (W2_out) *W2_out = W2;
    if (WX_out) *WX_out = WX;

    // ---- T2 residual ----
    // Fh chains
    Mat Fh_be = F_ae;
    Fh_be.noalias() -= 0.5 * T1.transpose() * F_me;  // (b,e) -= t_m^b F_me
    Mat Fh_mj = F_mi;
    Fh_mj.noalias() += 0.5 * F_me * T1.transpose();  // (m,j) += t_j^e F_me

    // base integral + ladder GEMMs
    if (target_only) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (Eigen::Index a = 0; a < nv; ++a)
            for (Eigen::Index b = 0; b < nv; ++b)
                R2_flat(target_ij, a * nv + b) =
                    V.ov_ov(target_i * nv + a, target_j * nv + b);
        R2_flat.row(target_ij).noalias() +=
            Wmnij.col(target_ij).transpose() * tau_flat;
    } else {
        #pragma omp parallel for collapse(2) schedule(static)
        for (Eigen::Index i = 0; i < no; ++i)
            for (Eigen::Index j = 0; j < no; ++j)
                for (Eigen::Index a = 0; a < nv; ++a)
                    for (Eigen::Index b = 0; b < nv; ++b)
                        R2_flat(i * no + j, a * nv + b) =
                            V.ov_ov(i * nv + a, j * nv + b);
        R2_flat.noalias() += Wmnij.transpose() * tau_flat;
    }

    // ---- Particle-particle ladder: R2(ij,ab) += sum_ef tau_ij^ef W_abef ----
    // W_abef(ab, ef) = (ae|bf) - sum_m [t_m^b (mf|ae) + t_m^a (me|bf)]
    //                + 1/2 sum_mn tau_mn^ab (me|nf).
    // (ae|bf) is V.vv_vv when it was materialised in-core; otherwise it is
    // regenerated from B_vv in tiles over the bra-virtual index a, so the
    // nv^4 panel is never fully resident (blocked-vvvv low-memory path). The
    // in-core branch is bit-identical to the pre-blocking code.
    //
    // `include_ladder=false` skips the whole term: the DLPNO extended-domain
    // path (#700) contracts it in each pair's own PNO space instead, where it
    // costs n_pno^4 rather than n_ext^4 and is exact, because tau_ij -- which
    // carries the contracted indices e,f -- lives in that space.
    if (include_ladder) {
        Mat M(no * no, nv * nv);  // M(mn, ef) = (me|nf); tile-independent
        #pragma omp parallel for collapse(2) schedule(static)
        for (Eigen::Index m = 0; m < no; ++m)
            for (Eigen::Index n = 0; n < no; ++n)
                for (Eigen::Index e = 0; e < nv; ++e)
                    for (Eigen::Index f = 0; f < nv; ++f)
                        M(m * no + n, e * nv + f) =
                            V.ov_ov(m * nv + e, n * nv + f);

        // Fill W(local_ab, ef) for a in [a0, a0+acur) from a (ae|bf) panel.
        const auto fill_wabef = [&](Eigen::Index a0, Eigen::Index acur,
                                    const Mat& vv_panel, Mat& W) {
            #pragma omp parallel for collapse(2) schedule(static)
            for (Eigen::Index al = 0; al < acur; ++al)
                for (Eigen::Index b = 0; b < nv; ++b) {
                    const Eigen::Index a = a0 + al;
                    const Eigen::Index ab = al * nv + b;
                    for (Eigen::Index e = 0; e < nv; ++e)
                        for (Eigen::Index f = 0; f < nv; ++f) {
                            double s = vv_panel(al * nv + e, b * nv + f);
                            for (Eigen::Index m = 0; m < no; ++m) {
                                s -= T1(m, b) * V.ov_vv(m * nv + f, a * nv + e);
                                s -= T1(m, a) * V.ov_vv(m * nv + e, b * nv + f);
                            }
                            W(ab, e * nv + f) = s;
                        }
                }
            W.noalias() +=
                0.5 * tau_flat.middleCols(a0 * nv, acur * nv).transpose() * M;
        };

        if (V.vv_vv.size() != 0) {
            // In-core: single tile over all a; panel aliases V.vv_vv.
            Mat Wabef(nv * nv, nv * nv);
            fill_wabef(0, nv, V.vv_vv, Wabef);
            if (target_only) {
                R2_flat.row(target_ij).noalias() +=
                    tau_flat.row(target_ij) * Wabef.transpose();
            } else {
                R2_flat.noalias() += tau_flat * Wabef.transpose();
            }
        } else {
            // Blocked: regenerate (ae|bf) tiles from B_vv and fuse the ladder.
            const Eigen::Index max_tile_rows = vvvv_tile_rows();
            const Eigen::Index ablk = std::max<Eigen::Index>(
                1, std::min<Eigen::Index>(
                       nv, max_tile_rows / std::max<Eigen::Index>(1, nv)));
            for (Eigen::Index a0 = 0; a0 < nv; a0 += ablk) {
                const Eigen::Index acur = std::min(ablk, nv - a0);
                const Mat vv_panel =
                    V.B_vv.middleCols(a0 * nv, acur * nv).transpose() * V.B_vv;
                Mat Wtile(acur * nv, nv * nv);
                fill_wabef(a0, acur, vv_panel, Wtile);
                if (target_only) {
                    R2_flat.block(target_ij, a0 * nv, 1, acur * nv).noalias() +=
                        tau_flat.row(target_ij) * Wtile.transpose();
                } else {
                    R2_flat.middleCols(a0 * nv, acur * nv).noalias() +=
                        tau_flat * Wtile.transpose();
                }
            }
        }
    }

    // Ph-symmetrised half
    Mat half = Mat::Zero(no * no, nv * nv);
    #pragma omp parallel for collapse(2) schedule(static)
    for (Eigen::Index i = 0; i < no; ++i)
        for (Eigen::Index j = 0; j < no; ++j) {
            if (target_only
                && !((i == target_i && j == target_j)
                     || (i == target_j && j == target_i))) {
                continue;
            }
            const Eigen::Index ij = i * no + j;
            for (Eigen::Index a = 0; a < nv; ++a)
                for (Eigen::Index b = 0; b < nv; ++b) {
                    double s = 0.0;
                    for (Eigen::Index e = 0; e < nv; ++e)
                        s += T2_flat(ij, a * nv + e) * Fh_be(b, e);
                    for (Eigen::Index m = 0; m < no; ++m)
                        s -= T2_flat(i * no + m, a * nv + b) * Fh_mj(m, j);
                    if (include_ring) {
                    for (Eigen::Index m = 0; m < no; ++m)
                        for (Eigen::Index e = 0; e < nv; ++e) {
                            const double t_im_ae =
                                T2_flat(i * no + m, a * nv + e);
                            const double t_mi_ae =
                                T2_flat(m * no + i, a * nv + e);
                            const double t_mj_ae =
                                T2_flat(m * no + j, a * nv + e);
                            s += (t_im_ae - t_mi_ae) * W1(m * nv + e, j * nv + b);
                            s += t_im_ae * W2(m * nv + e, j * nv + b);
                            s += t_mj_ae * WX(m * nv + e, i * nv + b);
                            s -= T1(i, e) * T1(m, a)
                               * V.ov_ov(m * nv + e, j * nv + b);
                            s -= T1(j, e) * T1(m, a)
                               * V.oo_vv(m * no + i, b * nv + e);
                        }
                    } else {
                    // #700: the three ring terms are contracted in the source
                    // pairs' PNO spaces by the caller. The two t1-quadratic
                    // terms here are not ring terms and stay.
                    for (Eigen::Index m = 0; m < no; ++m)
                        for (Eigen::Index e = 0; e < nv; ++e) {
                            s -= T1(i, e) * T1(m, a)
                               * V.ov_ov(m * nv + e, j * nv + b);
                            s -= T1(j, e) * T1(m, a)
                               * V.oo_vv(m * no + i, b * nv + e);
                        }
                    }
                    // + sum_e t_i^e (ae|jb); (ae|jb) = V_ov_vv(j*nv+b, a*nv+e)
                    for (Eigen::Index e = 0; e < nv; ++e)
                        s += T1(i, e) * V.ov_vv(j * nv + b, a * nv + e);
                    // - sum_m t_m^a (mi|jb)
                    for (Eigen::Index m = 0; m < no; ++m)
                        s -= T1(m, a) * V.oo_ov(m * no + i, j * nv + b);
                    half(ij, a * nv + b) = s;
                }
        }
    if (target_only) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (Eigen::Index a = 0; a < nv; ++a)
            for (Eigen::Index b = 0; b < nv; ++b)
                R2_flat(target_ij, a * nv + b) +=
                    half(target_ij, a * nv + b)
                    + half(target_j * no + target_i, b * nv + a);
    } else {
        #pragma omp parallel for collapse(2) schedule(static)
        for (Eigen::Index i = 0; i < no; ++i)
            for (Eigen::Index j = 0; j < no; ++j)
                for (Eigen::Index a = 0; a < nv; ++a)
                    for (Eigen::Index b = 0; b < nv; ++b)
                        R2_flat(i * no + j, a * nv + b) +=
                            half(i * no + j, a * nv + b)
                            + half(j * no + i, b * nv + a);
    }
}

// =========================================================================
// CC2 residuals (Christiansen, Koch & Jorgensen, CPL 243, 409 (1995)).
//
// With H_tilde = exp(-T1) H exp(T1), ground-state CC2 is defined by
//
//   <mu1| H_tilde + [H_tilde, T2] |HF> = 0
//   <mu2| H_tilde + [F,       T2] |HF> = 0.
//
// The CCSD singles projection is already at most linear in T2 and therefore
// equals the first equation.  The T2-independent part of the canonical CCSD
// doubles residual is exactly <mu2|H_tilde|HF>.  In a semicanonical basis the
// remaining Fock commutator is diagonal and contributes -D_ij^ab t_ij^ab in
// this residual convention, where D = eps_i + eps_j - eps_a - eps_b.
//
// Reusing compute_residuals keeps the T1 similarity transformation on the
// same validated SGWB equation set as CCSD and avoids a second transcription.
// =========================================================================
void compute_cc2_residuals(const Mat& T1, const Mat& T2_flat,
                           const IntegralBlocks& V,
                           const Mat& f_oo, const Mat& f_vv,
                           const Mat& f_ov,
                           const Eigen::VectorXd& eps_occ,
                           const Eigen::VectorXd& eps_vir,
                           Mat& R1, Mat& R2_flat) {
    const Eigen::Index no = T1.rows();
    const Eigen::Index nv = T1.cols();

    Mat discarded_r2(no * no, nv * nv);
    compute_residuals(T1, T2_flat, V, f_oo, f_vv, f_ov,
                      R1, discarded_r2);

    const Mat zero_t2 = Mat::Zero(no * no, nv * nv);
    Mat discarded_r1(no, nv);
    compute_residuals(T1, zero_t2, V, f_oo, f_vv, f_ov,
                      discarded_r1, R2_flat);

    #pragma omp parallel for collapse(2) schedule(static)
    for (Eigen::Index i = 0; i < no; ++i)
        for (Eigen::Index j = 0; j < no; ++j)
            for (Eigen::Index a = 0; a < nv; ++a)
                for (Eigen::Index b = 0; b < nv; ++b) {
                    const double denom = eps_occ(i) + eps_occ(j)
                                       - eps_vir(a) - eps_vir(b);
                    R2_flat(i * no + j, a * nv + b) -=
                        denom * T2_flat(i * no + j, a * nv + b);
                }
}

// =========================================================================
// Linearized residuals (LCCD / LCCSD = CEPA(0) / CEPA(n) backbone).
//
// The linear coupled-pair variants solve  R0 + A.T = 0  where A is the
// CI-like singles-doubles matrix, i.e. the CCSD residual truncated to the
// terms of total degree 1 in the joint amplitude vector T = (T1, T2).
// Rather than transcribing a second (linearized) equation set -- a
// duplicate-implementation risk the 2026-06-10 rewrite history warns
// about -- the linear part is extracted algebraically exactly from the
// canonical compute_residuals above:
//
//   R(lambda T) is a polynomial in lambda of degree <= 4 (the highest
//   joint power is tau x tau = t1^4 in the W_abef ladder), so the
//   five-point first-derivative stencil
//
//     L(T) = d/dlambda R(lambda T)|_0
//          = [ 8 (R(T) - R(-T)) - (R(2T) - R(-2T)) ] / 12
//
//   is EXACT (error term ~ d^5/dlambda^5 = 0), and
//
//     R_lin(T) = R(0) + L(T)
//
//   with the constant term R1(0) = f_ov, R2(0) = (ia|jb) evaluated once
//   by calling compute_residuals at T = 0.  Cost: four residual
//   evaluations per iteration -- irrelevant for the small systems these
//   sub-CCSD variants target, in exchange for a linearization that is
//   correct by construction.
// =========================================================================
void compute_linear_residuals(const Mat& T1, const Mat& T2_flat,
                              const IntegralBlocks& V,
                              const Mat& f_oo, const Mat& f_vv,
                              const Mat& f_ov,
                              const Mat& R1_0, const Mat& R2_0,
                              Mat& R1, Mat& R2_flat) {
    Mat r1p(R1_0.rows(), R1_0.cols()), r1m(R1_0.rows(), R1_0.cols());
    Mat r2p(R2_0.rows(), R2_0.cols()), r2m(R2_0.rows(), R2_0.cols());
    compute_residuals(T1, T2_flat, V, f_oo, f_vv, f_ov, r1p, r2p);
    compute_residuals(-T1, -T2_flat, V, f_oo, f_vv, f_ov, r1m, r2m);
    if (T1.isZero(0.0)) {
        // Doubles-only variants: the residual is a degree-2 polynomial in
        // T2, so the two-point odd part is already exact.
        R1 = R1_0 + 0.5 * (r1p - r1m);
        R2_flat = R2_0 + 0.5 * (r2p - r2m);
        return;
    }
    Mat r1p2(R1_0.rows(), R1_0.cols()), r1m2(R1_0.rows(), R1_0.cols());
    Mat r2p2(R2_0.rows(), R2_0.cols()), r2m2(R2_0.rows(), R2_0.cols());
    compute_residuals(2.0 * T1, 2.0 * T2_flat, V, f_oo, f_vv, f_ov,
                      r1p2, r2p2);
    compute_residuals(-2.0 * T1, -2.0 * T2_flat, V, f_oo, f_vv, f_ov,
                      r1m2, r2m2);
    R1 = R1_0 + (8.0 * (r1p - r1m) - (r1p2 - r1m2)) / 12.0;
    R2_flat = R2_0 + (8.0 * (r2p - r2m) - (r2p2 - r2m2)) / 12.0;
}

template <std::size_t N>
std::array<double, N>
coefficient_weights(const std::array<double, N>& nodes, int degree) {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> powers(N, N);
    for (std::size_t col = 0; col < N; ++col) {
        double x_pow = 1.0;
        for (std::size_t row = 0; row < N; ++row) {
            powers(static_cast<Eigen::Index>(row),
                   static_cast<Eigen::Index>(col)) = x_pow;
            x_pow *= nodes[col];
        }
    }
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(N));
    rhs(static_cast<Eigen::Index>(degree)) = 1.0;
    const Eigen::VectorXd solved = powers.fullPivLu().solve(rhs);
    std::array<double, N> w{};
    for (std::size_t i = 0; i < N; ++i) {
        w[i] = solved(static_cast<Eigen::Index>(i));
    }
    return w;
}

struct MonomialDegree {
    int t1;
    int t2;
};

template <std::size_t NX, std::size_t NY, std::size_t NKEEP>
double monomial_weight(
    const std::array<std::array<double, NX>, 5>& wx,
    const std::array<std::array<double, NY>, 3>& wy,
    const std::array<MonomialDegree, NKEEP>& keep,
    std::size_t ix, std::size_t iy) {
    double w = 0.0;
    for (const auto& deg : keep) {
        w += wx[static_cast<std::size_t>(deg.t1)][ix] *
             wy[static_cast<std::size_t>(deg.t2)][iy];
    }
    return w;
}

// =========================================================================
// QCISD residuals by exact bivariate monomial extraction.
//
// Pople, Head-Gordon & Raghavachari's QCISD equations can be viewed as a
// term-selected approximation to CCSD.  In the CCSD projection equations,
// QCISD uses the CI-like energy (no T1*T1 tau contribution), keeps the
// T2-quadratic disconnected quadruple term, and drops the disconnected
// triple T1*T2 terms from the doubles projection (JCP 87, 5968 (1987);
// compare also He & Cremer, IJQC Symp. 25, 43 (1991), Eqs. 24-26).
//
// The selected monomial degrees in the canonical residual polynomial are:
//   R1: (0,0), (1,0), (0,1), (2,0), (1,1)
//   R2: (0,0), (1,0), (0,1), (2,0), (0,2)
// where the first degree scales T1 and the second scales T2.  The residual
// degree is <=4 in T1 and <=2 in T2, so interpolation on a 5x3 grid extracts
// these coefficients exactly without maintaining a duplicate equation set.
// =========================================================================
void compute_qcisd_residuals(const Mat& T1, const Mat& T2_flat,
                             const IntegralBlocks& V,
                             const Mat& f_oo, const Mat& f_vv,
                             const Mat& f_ov,
                             Mat& R1, Mat& R2_flat) {
    static constexpr std::array<double, 5> x_nodes = {-2.0, -1.0, 0.0,
                                                       1.0, 2.0};
    static constexpr std::array<double, 3> y_nodes = {-1.0, 0.0, 1.0};
    static const std::array<std::array<double, 5>, 5> wx = {
        coefficient_weights(x_nodes, 0),
        coefficient_weights(x_nodes, 1),
        coefficient_weights(x_nodes, 2),
        coefficient_weights(x_nodes, 3),
        coefficient_weights(x_nodes, 4),
    };
    static const std::array<std::array<double, 3>, 3> wy = {
        coefficient_weights(y_nodes, 0),
        coefficient_weights(y_nodes, 1),
        coefficient_weights(y_nodes, 2),
    };
    static constexpr std::array<MonomialDegree, 5> keep_r1 = {{
        {0, 0}, {1, 0}, {0, 1}, {2, 0}, {1, 1},
    }};
    static constexpr std::array<MonomialDegree, 5> keep_r2 = {{
        {0, 0}, {1, 0}, {0, 1}, {2, 0}, {0, 2},
    }};

    R1 = Mat::Zero(T1.rows(), T1.cols());
    R2_flat = Mat::Zero(T2_flat.rows(), T2_flat.cols());
    for (std::size_t ix = 0; ix < x_nodes.size(); ++ix) {
        for (std::size_t iy = 0; iy < y_nodes.size(); ++iy) {
            Mat r1(T1.rows(), T1.cols());
            Mat r2(T2_flat.rows(), T2_flat.cols());
            compute_residuals(x_nodes[ix] * T1, y_nodes[iy] * T2_flat,
                              V, f_oo, f_vv, f_ov, r1, r2);
            const double w1 = monomial_weight(wx, wy, keep_r1, ix, iy);
            const double w2 = monomial_weight(wx, wy, keep_r2, ix, iy);
            if (w1 != 0.0) R1 += w1 * r1;
            if (w2 != 0.0) R2_flat += w2 * r2;
        }
    }
}

// =========================================================================
// CEPA pair energies and EPV shifts.
//
// Ordered-pair correlation energies from the current doubles amplitudes
// (closed-shell; sums over ALL ordered (i,j) to the linear correlation
// energy for canonical orbitals):
//
//   m_ij = sum_ab [2 (ia|jb) - (ib|ja)] t_ij^ab
//
// The physical (unordered) interorbital pair energy is 2 m_ij for
// i != j, and the diagonal (i-alpha, i-beta) pair energy is m_ii.
//
// CEPA(n) subtracts a pair-specific shift from the linearized residual,
// R2(ij,ab) -= Delta_ij t_ij^ab and R1(i,a) -= Delta_i t_i^a,
// approximating the EPV (exclusion-principle-violating) part of the
// missing quadratic terms (W. Meyer, J. Chem. Phys. 58, 1017 (1973),
// doi:10.1063/1.1679283).  The variant table below is the closed-shell
// shift table of the ORCA MDCI module (F. Wennmohs, F. Neese, Chem.
// Phys. 343, 217 (2008), doi:10.1016/j.chemphys.2007.07.001; ORCA 6
// manual, "diagonal shifts used in various singles- and doubles
// methods"), rewritten over the ordered-pair energies m_ij with row
// sums S_i = sum_k m_ik.  The manual's table is written over
// interorbital pair energies eps_ij = 2 m_ij (i != j) and diagonal
// pair energies eps_i = m_ii; collapsing it gives
//
//              doubles Delta_ij (i!=j)   Delta_ii        singles Delta_i
//   CEPA(1):   1/2 (S_i + S_j)           S_i             S_i
//   CEPA(2):   2 m_ij                    m_ii            m_ii
//   CEPA(3):   S_i + S_j - 2 m_ij        2 S_i - m_ii    2 S_i - m_ii
//
// CEPA(0) is Delta = 0 (plain LCCSD).  Convention adjudicated
// empirically against out-of-process ORCA 6.1 RI-CEPA/1..3 with the
// identical def2-SVP/C auxiliary basis and %mdci Localize false on
// H2O/def2-SVP: agreement 0.02 / 0.03 / 0.00 microHa -- exact working-
// equation parity (2026-07-02; pinned in tests/test_cc_variants.py).
// NOTE: ORCA localizes the internal valence orbitals (Foster-Boys) for
// CEPA by default and CEPA(n>=1) is not invariant under occupied
// rotations; vibe-qc's CEPA uses canonical MOs, matching ORCA's
// "Localize false".
// =========================================================================
Mat compute_pair_energies(const Mat& T2_flat, const Mat& V_ov_ov,
                          Eigen::Index no, Eigen::Index nv) {
    Mat eps = Mat::Zero(no, no);
    #pragma omp parallel for collapse(2) schedule(static)
    for (Eigen::Index i = 0; i < no; ++i)
        for (Eigen::Index j = 0; j < no; ++j) {
            double s = 0.0;
            for (Eigen::Index a = 0; a < nv; ++a)
                for (Eigen::Index b = 0; b < nv; ++b)
                    s += T2_flat(i * no + j, a * nv + b)
                       * (2.0 * V_ov_ov(i * nv + a, j * nv + b)
                          - V_ov_ov(i * nv + b, j * nv + a));
            eps(i, j) = s;
        }
    return eps;
}

struct CEPAShifts {
    Mat doubles;              // Delta_ij, (no x no)
    Eigen::VectorXd singles;  // Delta_i, (no)
};

CEPAShifts cepa_shifts(CCVariant variant, const Mat& m) {
    const Eigen::Index no = m.rows();
    CEPAShifts sh;
    sh.doubles = Mat::Zero(no, no);
    sh.singles = Eigen::VectorXd::Zero(no);
    if (variant == CCVariant::CEPA0) return sh;
    const Eigen::VectorXd S = m.rowwise().sum();
    for (Eigen::Index i = 0; i < no; ++i) {
        for (Eigen::Index j = 0; j < no; ++j) {
            switch (variant) {
                case CCVariant::CEPA1:
                    sh.doubles(i, j) =
                        (i == j) ? S(i) : 0.5 * (S(i) + S(j));
                    break;
                case CCVariant::CEPA2:
                    sh.doubles(i, j) = (i == j) ? m(i, i) : 2.0 * m(i, j);
                    break;
                case CCVariant::CEPA3:
                    sh.doubles(i, j) = (i == j)
                        ? 2.0 * S(i) - m(i, i)
                        : S(i) + S(j) - 2.0 * m(i, j);
                    break;
                default:
                    break;
            }
        }
        switch (variant) {
            case CCVariant::CEPA1:
                sh.singles(i) = S(i);
                break;
            case CCVariant::CEPA2:
                sh.singles(i) = m(i, i);
                break;
            case CCVariant::CEPA3:
                sh.singles(i) = 2.0 * S(i) - m(i, i);
                break;
            default:
                break;
        }
    }
    return sh;
}

void apply_cepa_shifts(const CEPAShifts& sh, const Mat& T1,
                       const Mat& T2_flat, Mat& R1, Mat& R2_flat) {
    const auto no = static_cast<Eigen::Index>(T1.rows());
    const auto nv = static_cast<Eigen::Index>(T1.cols());
    #pragma omp parallel for collapse(2) schedule(static)
    for (Eigen::Index i = 0; i < no; ++i)
        for (Eigen::Index j = 0; j < no; ++j) {
            const double d = sh.doubles(i, j);
            if (d == 0.0) continue;
            for (Eigen::Index a = 0; a < nv; ++a)
                for (Eigen::Index b = 0; b < nv; ++b)
                    R2_flat(i * no + j, a * nv + b) -=
                        d * T2_flat(i * no + j, a * nv + b);
        }
    for (Eigen::Index i = 0; i < no; ++i)
        R1.row(i) -= sh.singles(i) * T1.row(i);
}

// =========================================================================
// Jacobi amplitude update: T += omega * R / D (true residuals)
// =========================================================================
void update_amplitudes(Mat& T1, Mat& T2_flat,
                       const Mat& R1, const Mat& R2_flat,
                       const Eigen::VectorXd& eps_occ,
                       const Eigen::VectorXd& eps_vir,
                       double omega) {
    const auto no = static_cast<Eigen::Index>(T1.rows());
    const auto nv = static_cast<Eigen::Index>(T1.cols());

    #pragma omp parallel for collapse(2) schedule(static)
    for (Eigen::Index i = 0; i < no; ++i)
        for (Eigen::Index a = 0; a < nv; ++a)
            T1(i, a) += omega * R1(i, a) / (eps_occ(i) - eps_vir(a));

    #pragma omp parallel for collapse(2) schedule(static)
    for (Eigen::Index i = 0; i < no; ++i)
        for (Eigen::Index j = 0; j < no; ++j)
            for (Eigen::Index a = 0; a < nv; ++a)
                for (Eigen::Index b = 0; b < nv; ++b)
                    T2_flat(i * no + j, a * nv + b) +=
                        omega * R2_flat(i * no + j, a * nv + b)
                        / (eps_occ(i) + eps_occ(j) - eps_vir(a) - eps_vir(b));
}

// =========================================================================
// DIIS extrapolation over (T1, T2) joint vectors (Pulay, CPL 73, 393
// (1980)).  Stores the post-Jacobi amplitudes, mirroring the validated
// reference kernel in python/vibeqc/dlpno/_ccsd_ref.py.
// =========================================================================
class CCSD_DIIS {
public:
    explicit CCSD_DIIS(std::size_t max_subspace = 6)
        : max_subspace_(max_subspace) {}

    void extrapolate(const Mat& T1_new, const Mat& T2_new,
                     const Mat& R1, const Mat& R2_flat,
                     Mat& T1_out, Mat& T2_out) {
        const auto no = static_cast<Eigen::Index>(T1_new.rows());
        const auto nv = static_cast<Eigen::Index>(T1_new.cols());
        const auto n1 = no * nv;
        const auto n2 = T2_new.rows() * T2_new.cols();

        Eigen::VectorXd ampl(n1 + n2), err(n1 + n2);
        ampl.head(n1) = Eigen::Map<const Eigen::VectorXd>(T1_new.data(), n1);
        ampl.tail(n2) = Eigen::Map<const Eigen::VectorXd>(T2_new.data(), n2);
        err.head(n1) = Eigen::Map<const Eigen::VectorXd>(R1.data(), n1);
        err.tail(n2) = Eigen::Map<const Eigen::VectorXd>(R2_flat.data(), n2);

        ampl_history_.push_back(ampl);
        err_history_.push_back(err);
        if (ampl_history_.size() > max_subspace_) {
            ampl_history_.pop_front();
            err_history_.pop_front();
        }
        n_used_ = ampl_history_.size();

        T1_out = T1_new;
        T2_out = T2_new;
        if (n_used_ < 2) return;

        const auto N = static_cast<Eigen::Index>(n_used_);
        Eigen::MatrixXd B = Eigen::MatrixXd::Zero(N + 1, N + 1);
        for (Eigen::Index k = 0; k < N; ++k)
            for (Eigen::Index l = 0; l < N; ++l)
                B(k, l) = err_history_[static_cast<std::size_t>(k)].dot(
                    err_history_[static_cast<std::size_t>(l)]);
        B.row(N).setConstant(-1.0);
        B.col(N).setConstant(-1.0);
        B(N, N) = 0.0;

        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(N + 1);
        rhs(N) = -1.0;
        Eigen::VectorXd c = B.fullPivLu().solve(rhs);
        if (!c.head(N).allFinite()) return;

        Eigen::VectorXd mix = Eigen::VectorXd::Zero(n1 + n2);
        for (Eigen::Index k = 0; k < N; ++k)
            mix += c(k) * ampl_history_[static_cast<std::size_t>(k)];
        T1_out = Eigen::Map<const Mat>(mix.head(n1).data(), no, nv);
        T2_out = Eigen::Map<const Mat>(mix.tail(n2).data(),
                                       T2_new.rows(), T2_new.cols());
    }

    std::size_t subspace_size() const noexcept { return n_used_; }

private:
    std::size_t max_subspace_;
    std::size_t n_used_ = 0;
    std::deque<Eigen::VectorXd> ampl_history_;
    std::deque<Eigen::VectorXd> err_history_;
};

// =========================================================================
// (T) triples correction (Raghavachari 1989), closed-shell classwise
// spin integration.
//
// Spin-orbital formulas (validated implementation: the (T) reference in
// the M3 prototype harness, itself checked against CCSD(T)=FCI limits):
//
//   G(p1 p2 p3; q1 q2 q3) = sum_e t_{p2 p3}^{q1 e} <e p1||q2 q3>
//                         - sum_m t_{p1 m}^{q2 q3} <m q1||p2 p3>
//   W  = P(i/jk) P(a/bc) G        (connected, equals D * t3c)
//   Wd = P(i/jk) P(a/bc) [ t_{p1}^{q1} <p2 p3||q2 q3>
//                           + f_{p1}^{q1} t_{p2 p3}^{q2 q3} ]
//                                                        (disconnected)
//   P(i/jk) f = f(i,j,k) - f(j,i,k) - f(k,j,i)
//
//   E(T) = (1/36) sum_{spin patterns} sum_{ijkabc} W (W + Wd) / D
//
// For a closed-shell reference the pattern sum collapses to two classes
// (verified numerically against the full spin-orbital sum):
//
//   E(T) = (1/18) S_aaa + (1/2) S_aab
//
// where S_aaa uses external spins (aaa;aaa) and S_aab uses (aab;aab);
// the three single-beta virtual patterns contribute identically by the
// slot-swap antisymmetry of W, giving the factor 18/36 = 1/2.
//
// Each spin-orbital tensor block reduces to spatial quantities:
//   t-block:  t[(i s1)(j s2)]^[(a s3)(b s4)] =
//       t_ij^ab [s1==s3][s2==s4] - t_ij^ba [s1==s4][s2==s3]
//   <e p1||q2 q3> = (e q2|p1 q3)[se==sq2][sp1==sq3]
//                 - (e q3|p1 q2)[se==sq3][sp1==sq2]
//   <m q1||p2 p3> = (m p2|q1 p3)[sm==sp2][sq1==sp3]
//                 - (m p3|q1 p2)[sm==sp3][sq1==sp2]
//
// The 3 x 3 permutation images are accumulated per occupied triple
// (i,j,k) into rank-3 (a,b,c) buffers; the spin conditionals depend only
// on the permutation, not on the indices, so they are hoisted out of the
// contraction loops.
//
// Canonical RHF has f_ov = 0.  The explicit-MO and Brueckner routes are only
// semicanonical (diagonal f_oo/f_vv), so the f_ov*T2 part of Wd must be kept.
// =========================================================================
namespace triples {

struct Pattern {
    std::array<int, 3> so;  // occupied external spins
    std::array<int, 3> sv;  // virtual external spins
};

// One P(i/jk) x P(a/bc) image: G at permuted indices/spins, accumulated
// into w[a*nv*nv + b*nv + c] with the image sign.
void accumulate_g_image(
    const Mat& T2_flat, const IntegralBlocks& V,
    Eigen::Index no, Eigen::Index nv,
    const std::array<Eigen::Index, 3>& occ,   // external (i,j,k)
    const Pattern& pat,
    const std::array<int, 3>& po,             // occupied permutation
    const std::array<int, 3>& pv,             // virtual permutation
    double sign, std::vector<double>& w) {
    const Eigen::Index p1 = occ[static_cast<std::size_t>(po[0])];
    const Eigen::Index p2 = occ[static_cast<std::size_t>(po[1])];
    const Eigen::Index p3 = occ[static_cast<std::size_t>(po[2])];
    const int s1 = pat.so[static_cast<std::size_t>(po[0])];
    const int s2 = pat.so[static_cast<std::size_t>(po[1])];
    const int s3 = pat.so[static_cast<std::size_t>(po[2])];
    const int z1 = pat.sv[static_cast<std::size_t>(pv[0])];
    const int z2 = pat.sv[static_cast<std::size_t>(pv[1])];
    const int z3 = pat.sv[static_cast<std::size_t>(pv[2])];

    // External (a,b,c) receive G's (q1,q2,q3) axes at positions
    // (pv[0], pv[1], pv[2]).  Strides of (a,b,c) in the flat buffer:
    const Eigen::Index stride_ext[3] = {nv * nv, nv, 1};
    const Eigen::Index sq1 = stride_ext[pv[0]];
    const Eigen::Index sq2 = stride_ext[pv[1]];
    const Eigen::Index sq3 = stride_ext[pv[2]];

    // ---- term A: sum_e t_{p2 p3}^{q1 e} <e p1||q2 q3>, e spin summed ----
    for (int se = 0; se < 2; ++se) {
        const bool t_dir = (s2 == z1 && s3 == se);   // + t_{p2p3}^{q1 e}
        const bool t_swp = (s2 == se && s3 == z1);   // - t_{p2p3}^{e q1}
        const bool v_cou = (se == z2 && s1 == z3);   // + (e q2|p1 q3)
        const bool v_exc = (se == z3 && s1 == z2);   // - (e q3|p1 q2)
        if (!(t_dir || t_swp) || !(v_cou || v_exc)) continue;
        Mat t_work(nv, nv);
        Mat v_work(nv, nv * nv);
        for (Eigen::Index x = 0; x < nv; ++x)
            for (Eigen::Index e = 0; e < nv; ++e) {
                double value = 0.0;
                if (t_dir)
                    value += T2_flat(p2 * no + p3, x * nv + e);
                if (t_swp)
                    value -= T2_flat(p2 * no + p3, e * nv + x);
                t_work(x, e) = value;
            }
        for (Eigen::Index e = 0; e < nv; ++e)
            for (Eigen::Index y = 0; y < nv; ++y)
                for (Eigen::Index z = 0; z < nv; ++z) {
                    double value = 0.0;
                    // (e q2|p1 q3) = V_ov_vv(p1*nv+q3, e*nv+q2)
                    if (v_cou)
                        value += V.ov_vv(p1 * nv + z, e * nv + y);
                    if (v_exc)
                        value -= V.ov_vv(p1 * nv + y, e * nv + z);
                    v_work(e, y * nv + z) = value;
                }

        Mat contracted(nv, nv * nv);
        contracted.noalias() = t_work * v_work;
        for (Eigen::Index x = 0; x < nv; ++x)
            for (Eigen::Index y = 0; y < nv; ++y)
                for (Eigen::Index z = 0; z < nv; ++z) {
                    const double value = contracted(x, y * nv + z);
                    if (value == 0.0) continue;
                    w[static_cast<std::size_t>(
                        x * sq1 + y * sq2 + z * sq3)] += sign * value;
                }
    }

    // ---- term B: - sum_m t_{p1 m}^{q2 q3} <m q1||p2 p3> ----
    for (int sm = 0; sm < 2; ++sm) {
        const bool t_dir = (s1 == z2 && sm == z3);   // + t_{p1 m}^{q2 q3}
        const bool t_swp = (s1 == z3 && sm == z2);   // - t_{p1 m}^{q3 q2}
        const bool v_cou = (sm == s2 && z1 == s3);   // + (m p2|q1 p3)
        const bool v_exc = (sm == s3 && z1 == s2);   // - (m p3|q1 p2)
        if (!(t_dir || t_swp) || !(v_cou || v_exc)) continue;
        for (Eigen::Index m = 0; m < no; ++m)
            for (Eigen::Index x = 0; x < nv; ++x) {  // q1
                double vv = 0.0;
                // (m p2|q1 p3) = V_oo_ov(m*no+p2, p3*nv+q1)
                if (v_cou) vv += V.oo_ov(m * no + p2, p3 * nv + x);
                if (v_exc) vv -= V.oo_ov(m * no + p3, p2 * nv + x);
                if (vv == 0.0) continue;
                for (Eigen::Index y = 0; y < nv; ++y)      // q2
                    for (Eigen::Index z = 0; z < nv; ++z) {  // q3
                        double tv = 0.0;
                        if (t_dir) tv += T2_flat(p1 * no + m, y * nv + z);
                        if (t_swp) tv -= T2_flat(p1 * no + m, z * nv + y);
                        if (tv == 0.0) continue;
                        w[static_cast<std::size_t>(
                            x * sq1 + y * sq2 + z * sq3)] -= sign * tv * vv;
                    }
            }
    }
}

// Disconnected image:
//   t_{p1}^{q1} <p2 p3||q2 q3> + f_{p1}^{q1} t_{p2 p3}^{q2 q3}.
void accumulate_d_image(
    const Mat& T1, const Mat& T2_flat, const Mat& f_ov,
    const IntegralBlocks& V, Eigen::Index no, Eigen::Index nv,
    const std::array<Eigen::Index, 3>& occ,
    const Pattern& pat,
    const std::array<int, 3>& po,
    const std::array<int, 3>& pv,
    double sign, std::vector<double>& w) {
    const Eigen::Index p1 = occ[static_cast<std::size_t>(po[0])];
    const Eigen::Index p2 = occ[static_cast<std::size_t>(po[1])];
    const Eigen::Index p3 = occ[static_cast<std::size_t>(po[2])];
    const int s1 = pat.so[static_cast<std::size_t>(po[0])];
    const int s2 = pat.so[static_cast<std::size_t>(po[1])];
    const int s3 = pat.so[static_cast<std::size_t>(po[2])];
    const int z1 = pat.sv[static_cast<std::size_t>(pv[0])];
    const int z2 = pat.sv[static_cast<std::size_t>(pv[1])];
    const int z3 = pat.sv[static_cast<std::size_t>(pv[2])];
    if (s1 != z1) return;

    const Eigen::Index stride_ext[3] = {nv * nv, nv, 1};
    const Eigen::Index sq1 = stride_ext[pv[0]];
    const Eigen::Index sq2 = stride_ext[pv[1]];
    const Eigen::Index sq3 = stride_ext[pv[2]];

    const bool v_cou = (s2 == z2 && s3 == z3);  // + (p2 q2|p3 q3)
    const bool v_exc = (s2 == z3 && s3 == z2);  // - (p2 q3|p3 q2)
    const bool t_dir = (s2 == z2 && s3 == z3);  // + t_{p2p3}^{q2 q3}
    const bool t_swp = (s2 == z3 && s3 == z2);  // - t_{p2p3}^{q3 q2}
    for (Eigen::Index x = 0; x < nv; ++x) {
        const double t1v = T1(p1, x);
        const double f1v = f_ov(p1, x);
        if (t1v == 0.0 && f1v == 0.0) continue;
        for (Eigen::Index y = 0; y < nv; ++y)
            for (Eigen::Index z = 0; z < nv; ++z) {
                double vv = 0.0;
                if (v_cou) vv += V.ov_ov(p2 * nv + y, p3 * nv + z);
                if (v_exc) vv -= V.ov_ov(p2 * nv + z, p3 * nv + y);
                double tv = 0.0;
                if (t_dir)
                    tv += T2_flat(p2 * no + p3, y * nv + z);
                if (t_swp)
                    tv -= T2_flat(p2 * no + p3, z * nv + y);
                const double value = t1v * vv + f1v * tv;
                if (value == 0.0) continue;
                w[static_cast<std::size_t>(
                    x * sq1 + y * sq2 + z * sq3)] += sign * value;
            }
    }
}

// Tiled accumulate: restricts whichever loop variable maps to
// external virtual index a_batch_pos (0='a', 1='b', 2='c') to [a0, a1).
// Tile buffer w has shape (tile_n, nv, nv) with batched index first.
void accumulate_g_image_batched(
    const Mat& T2_flat, const IntegralBlocks& V,
    Eigen::Index no, Eigen::Index nv,
    const std::array<Eigen::Index, 3>& occ,
    const Pattern& pat,
    const std::array<int, 3>& po,
    const std::array<int, 3>& pv,
    double sign, std::vector<double>& w,
    int a_batch_pos, Eigen::Index a0, Eigen::Index a1,
    DiskDFFactorReader* disk_reader = nullptr) {

    const Eigen::Index p1 = occ[static_cast<std::size_t>(po[0])];
    const Eigen::Index p2 = occ[static_cast<std::size_t>(po[1])];
    const Eigen::Index p3 = occ[static_cast<std::size_t>(po[2])];
    const int s1 = pat.so[static_cast<std::size_t>(po[0])];
    const int s2 = pat.so[static_cast<std::size_t>(po[1])];
    const int s3 = pat.so[static_cast<std::size_t>(po[2])];
    const int z1 = pat.sv[static_cast<std::size_t>(pv[0])];
    const int z2 = pat.sv[static_cast<std::size_t>(pv[1])];
    const int z3 = pat.sv[static_cast<std::size_t>(pv[2])];

    int batch_axis = -1;
    if (pv[0] == a_batch_pos) batch_axis = 0;
    else if (pv[1] == a_batch_pos) batch_axis = 1;
    else if (pv[2] == a_batch_pos) batch_axis = 2;
    if (batch_axis < 0) return;

    const Eigen::Index x0 = (batch_axis == 0) ? a0 : 0;
    const Eigen::Index x1 = (batch_axis == 0) ? a1 : nv;
    const Eigen::Index y0 = (batch_axis == 1) ? a0 : 0;
    const Eigen::Index y1 = (batch_axis == 1) ? a1 : nv;
    const Eigen::Index z_lo = (batch_axis == 2) ? a0 : 0;
    const Eigen::Index z_hi = (batch_axis == 2) ? a1 : nv;
    // Stride of first index in tile buffer of shape (tile_n, nv, nv)
    const std::size_t t_stride0 = static_cast<std::size_t>(nv) * static_cast<std::size_t>(nv);
    const std::size_t t_stride1 = static_cast<std::size_t>(nv);

    auto tile_idx = [&](Eigen::Index xi, Eigen::Index yi, Eigen::Index zi) -> std::size_t {
        if (batch_axis == 0)
            return static_cast<std::size_t>(xi - x0) * t_stride0
                 + static_cast<std::size_t>(yi - y0) * t_stride1
                 + static_cast<std::size_t>(zi - z_lo);
        else if (batch_axis == 1)
            return static_cast<std::size_t>(yi - y0) * t_stride0
                 + static_cast<std::size_t>(xi - x0) * t_stride1
                 + static_cast<std::size_t>(zi - z_lo);
        else
            return static_cast<std::size_t>(zi - z_lo) * t_stride0
                 + static_cast<std::size_t>(yi - y0) * t_stride1
                 + static_cast<std::size_t>(xi - x0);
    };

    // Term A.  This is deliberately an element/tile contraction: the old
    // implementation allocated v_work(nv^3) and contracted(nv^3) inside this
    // supposedly tiled helper, so its peak was still cubic even for tile=1.
    //
    // When ov_vv is resident (canonical integrals), contract the virtual e
    // index directly for each requested tile element.  On the DF path the
    // blocked/direct planner releases ov_vv before entering triples and this
    // branch contracts B_ov/B_vv in factor order.  For the Coulomb image,
    //
    //   sum_e t(x,e) (p1 z|e y)
    //     = sum_P [sum_e t(x,e) B_vv(P,e,y)] B_ov(P,p1,z),
    //
    // and analogously for exchange.  No temporary depends on nv^3; the only
    // virtual-rank-three storage is the caller's bounded output tile.
    for (int se = 0; se < 2; ++se) {
        const bool t_dir = (s2 == z1 && s3 == se);
        const bool t_swp = (s2 == se && s3 == z1);
        const bool v_cou = (se == z2 && s1 == z3);
        const bool v_exc = (se == z3 && s1 == z2);
        if (!(t_dir || t_swp) || !(v_cou || v_exc)) continue;
        const Eigen::Index ij = p2 * no + p3;
        if (V.ov_vv.size() != 0) {
            for (Eigen::Index x = x0; x < x1; ++x)
                for (Eigen::Index y = y0; y < y1; ++y)
                    for (Eigen::Index z = z_lo; z < z_hi; ++z) {
                        double value = 0.0;
                        for (Eigen::Index e = 0; e < nv; ++e) {
                            double tv = 0.0;
                            if (t_dir) tv += T2_flat(ij, x * nv + e);
                            if (t_swp) tv -= T2_flat(ij, e * nv + x);
                            if (v_cou)
                                value += tv * V.ov_vv(p1 * nv + z, e * nv + y);
                            if (v_exc)
                                value -= tv * V.ov_vv(p1 * nv + y, e * nv + z);
                        }
                        if (value != 0.0)
                            w[tile_idx(x, y, z)] += sign * value;
                    }
        } else {
            const auto accumulate_factor_row = [&](const auto& bov,
                                                    const auto& bvv) {
                if (v_cou) {
                    for (Eigen::Index x = x0; x < x1; ++x)
                        for (Eigen::Index y = y0; y < y1; ++y) {
                            double contracted = 0.0;
                            for (Eigen::Index e = 0; e < nv; ++e) {
                                double tv = 0.0;
                                if (t_dir) tv += T2_flat(ij, x * nv + e);
                                if (t_swp) tv -= T2_flat(ij, e * nv + x);
                                contracted += tv * bvv(e * nv + y);
                            }
                            if (contracted == 0.0) continue;
                            for (Eigen::Index z = z_lo; z < z_hi; ++z)
                                w[tile_idx(x, y, z)] +=
                                    sign * contracted * bov(p1 * nv + z);
                        }
                }
                if (v_exc) {
                    for (Eigen::Index x = x0; x < x1; ++x)
                        for (Eigen::Index z = z_lo; z < z_hi; ++z) {
                            double contracted = 0.0;
                            for (Eigen::Index e = 0; e < nv; ++e) {
                                double tv = 0.0;
                                if (t_dir) tv += T2_flat(ij, x * nv + e);
                                if (t_swp) tv -= T2_flat(ij, e * nv + x);
                                contracted += tv * bvv(e * nv + z);
                            }
                            if (contracted == 0.0) continue;
                            for (Eigen::Index y = y0; y < y1; ++y)
                                w[tile_idx(x, y, z)] -=
                                    sign * contracted * bov(p1 * nv + y);
                        }
                }
            };

            if (V.disk_factors) {
                if (disk_reader == nullptr)
                    throw std::runtime_error(
                        "CCSD(T) disk triples is missing its bounded factor "
                        "reader");
                disk_reader->rewind();
                while (disk_reader->load_next()) {
                    for (Eigen::Index P = 0;
                         P < disk_reader->rows_loaded(); ++P) {
                        accumulate_factor_row(
                            [&](Eigen::Index column) {
                                return disk_reader->ov(P, column);
                            },
                            [&](Eigen::Index column) {
                                return disk_reader->vv(P, column);
                            });
                    }
                }
            } else {
                if (V.B_ov.size() == 0 || V.B_vv.size() == 0
                    || V.B_ov.rows() != V.B_vv.rows()) {
                    throw std::runtime_error(
                        "CCSD(T) blocked triples requires retained or "
                        "disk-backed B_ov/B_vv density-fitting factors");
                }
                for (Eigen::Index P = 0; P < V.B_ov.rows(); ++P) {
                    accumulate_factor_row(
                        [&](Eigen::Index column) {
                            return V.B_ov(P, column);
                        },
                        [&](Eigen::Index column) {
                            return V.B_vv(P, column);
                        });
                }
            }
        }
    }

    // term B
    for (int sm = 0; sm < 2; ++sm) {
        const bool t_dir = (s1 == z2 && sm == z3);
        const bool t_swp = (s1 == z3 && sm == z2);
        const bool v_cou = (sm == s2 && z1 == s3);
        const bool v_exc = (sm == s3 && z1 == s2);
        if (!(t_dir || t_swp) || !(v_cou || v_exc)) continue;
        for (Eigen::Index m = 0; m < no; ++m)
            for (Eigen::Index x = x0; x < x1; ++x) {
                double vv = 0.0;
                if (v_cou) vv += V.oo_ov(m*no+p2, p3*nv+x);
                if (v_exc) vv -= V.oo_ov(m*no+p3, p2*nv+x);
                if (vv == 0.0) continue;
                for (Eigen::Index y = y0; y < y1; ++y)
                    for (Eigen::Index z = z_lo; z < z_hi; ++z) {
                        double tv = 0.0;
                        if (t_dir) tv += T2_flat(p1*no+m, y*nv+z);
                        if (t_swp) tv -= T2_flat(p1*no+m, z*nv+y);
                        if (tv == 0.0) continue;
                        w[tile_idx(x, y, z)] -= sign*tv*vv;
                    }
            }
    }
}

void accumulate_d_image_batched(
    const Mat& T1, const Mat& T2_flat, const Mat& f_ov,
    const IntegralBlocks& V, Eigen::Index no, Eigen::Index nv,
    const std::array<Eigen::Index, 3>& occ,
    const Pattern& pat,
    const std::array<int, 3>& po,
    const std::array<int, 3>& pv,
    double sign, std::vector<double>& w,
    int a_batch_pos, Eigen::Index a0, Eigen::Index a1) {

    const Eigen::Index p1 = occ[static_cast<std::size_t>(po[0])];
    const Eigen::Index p2 = occ[static_cast<std::size_t>(po[1])];
    const Eigen::Index p3 = occ[static_cast<std::size_t>(po[2])];
    const int s1 = pat.so[static_cast<std::size_t>(po[0])];
    const int s2 = pat.so[static_cast<std::size_t>(po[1])];
    const int s3 = pat.so[static_cast<std::size_t>(po[2])];
    const int z1 = pat.sv[static_cast<std::size_t>(pv[0])];
    const int z2 = pat.sv[static_cast<std::size_t>(pv[1])];
    const int z3 = pat.sv[static_cast<std::size_t>(pv[2])];
    if (s1 != z1) return;

    int batch_axis = -1;
    if (pv[0] == a_batch_pos) batch_axis = 0;
    else if (pv[1] == a_batch_pos) batch_axis = 1;
    else if (pv[2] == a_batch_pos) batch_axis = 2;
    if (batch_axis < 0) return;

    const Eigen::Index x0 = (batch_axis == 0) ? a0 : 0;
    const Eigen::Index x1 = (batch_axis == 0) ? a1 : nv;
    const Eigen::Index y0 = (batch_axis == 1) ? a0 : 0;
    const Eigen::Index y1 = (batch_axis == 1) ? a1 : nv;
    const Eigen::Index z_lo = (batch_axis == 2) ? a0 : 0;
    const Eigen::Index z_hi = (batch_axis == 2) ? a1 : nv;
    const std::size_t t_stride0 = static_cast<std::size_t>(nv) * static_cast<std::size_t>(nv);
    const std::size_t t_stride1 = static_cast<std::size_t>(nv);

    auto tile_idx = [&](Eigen::Index xi, Eigen::Index yi, Eigen::Index zi) -> std::size_t {
        if (batch_axis == 0)
            return static_cast<std::size_t>(xi - x0) * t_stride0
                 + static_cast<std::size_t>(yi - y0) * t_stride1
                 + static_cast<std::size_t>(zi - z_lo);
        else if (batch_axis == 1)
            return static_cast<std::size_t>(yi - y0) * t_stride0
                 + static_cast<std::size_t>(xi - x0) * t_stride1
                 + static_cast<std::size_t>(zi - z_lo);
        else
            return static_cast<std::size_t>(zi - z_lo) * t_stride0
                 + static_cast<std::size_t>(yi - y0) * t_stride1
                 + static_cast<std::size_t>(xi - x0);
    };

    const bool v_cou = (s2 == z2 && s3 == z3);
    const bool v_exc = (s2 == z3 && s3 == z2);
    const bool t_dir = (s2 == z2 && s3 == z3);
    const bool t_swp = (s2 == z3 && s3 == z2);
    for (Eigen::Index x = x0; x < x1; ++x) {
        const double t1v = T1(p1, x);
        const double f1v = f_ov(p1, x);
        if (t1v == 0.0 && f1v == 0.0) continue;
        for (Eigen::Index y = y0; y < y1; ++y)
            for (Eigen::Index z = z_lo; z < z_hi; ++z) {
                double vv = 0.0;
                if (v_cou) vv += V.ov_ov(p2*nv+y, p3*nv+z);
                if (v_exc) vv -= V.ov_ov(p2*nv+z, p3*nv+y);
                double tv = 0.0;
                if (t_dir) tv += T2_flat(p2*no+p3, y*nv+z);
                if (t_swp) tv -= T2_flat(p2*no+p3, z*nv+y);
                const double value = t1v*vv + f1v*tv;
                if (value == 0.0) continue;
                w[tile_idx(x, y, z)] += sign * value;
            }
    }
}

// Scalar images used by the minimum-workspace direct strategy.  They are the
// same G/Wd equations as the dense and blocked builders above, but return one
// external (a,b,c) element immediately so no virtual-rank-three buffer exists.
double ov_vv_element(const IntegralBlocks& V, Eigen::Index nv,
                     Eigen::Index m, Eigen::Index e,
                     Eigen::Index a, Eigen::Index f) {
    if (V.ov_vv.size() != 0)
        return V.ov_vv(m * nv + e, a * nv + f);
    if (V.B_ov.size() == 0 || V.B_vv.size() == 0
        || V.B_ov.rows() != V.B_vv.rows()) {
        throw std::runtime_error(
            "CCSD(T) direct triples requires ov_vv or retained B_ov/B_vv factors");
    }
    double value = 0.0;
    for (Eigen::Index P = 0; P < V.B_ov.rows(); ++P)
        value += V.B_ov(P, m * nv + e) * V.B_vv(P, a * nv + f);
    return value;
}

double g_image_value(
    const Mat& T2_flat, const IntegralBlocks& V,
    Eigen::Index no, Eigen::Index nv,
    const std::array<Eigen::Index, 3>& occ, const Pattern& pat,
    const std::array<int, 3>& po, const std::array<int, 3>& pv,
    Eigen::Index x, Eigen::Index y, Eigen::Index z) {
    const Eigen::Index p1 = occ[static_cast<std::size_t>(po[0])];
    const Eigen::Index p2 = occ[static_cast<std::size_t>(po[1])];
    const Eigen::Index p3 = occ[static_cast<std::size_t>(po[2])];
    const int s1 = pat.so[static_cast<std::size_t>(po[0])];
    const int s2 = pat.so[static_cast<std::size_t>(po[1])];
    const int s3 = pat.so[static_cast<std::size_t>(po[2])];
    const int z1 = pat.sv[static_cast<std::size_t>(pv[0])];
    const int z2 = pat.sv[static_cast<std::size_t>(pv[1])];
    const int z3 = pat.sv[static_cast<std::size_t>(pv[2])];

    double value = 0.0;
    for (int se = 0; se < 2; ++se) {
        const bool t_dir = (s2 == z1 && s3 == se);
        const bool t_swp = (s2 == se && s3 == z1);
        const bool v_cou = (se == z2 && s1 == z3);
        const bool v_exc = (se == z3 && s1 == z2);
        if (!(t_dir || t_swp) || !(v_cou || v_exc)) continue;
        for (Eigen::Index e = 0; e < nv; ++e) {
            double tv = 0.0;
            if (t_dir) tv += T2_flat(p2 * no + p3, x * nv + e);
            if (t_swp) tv -= T2_flat(p2 * no + p3, e * nv + x);
            if (v_cou)
                value += tv * ov_vv_element(V, nv, p1, z, e, y);
            if (v_exc)
                value -= tv * ov_vv_element(V, nv, p1, y, e, z);
        }
    }
    for (int sm = 0; sm < 2; ++sm) {
        const bool t_dir = (s1 == z2 && sm == z3);
        const bool t_swp = (s1 == z3 && sm == z2);
        const bool v_cou = (sm == s2 && z1 == s3);
        const bool v_exc = (sm == s3 && z1 == s2);
        if (!(t_dir || t_swp) || !(v_cou || v_exc)) continue;
        for (Eigen::Index m = 0; m < no; ++m) {
            double vv = 0.0;
            if (v_cou) vv += V.oo_ov(m * no + p2, p3 * nv + x);
            if (v_exc) vv -= V.oo_ov(m * no + p3, p2 * nv + x);
            double tv = 0.0;
            if (t_dir) tv += T2_flat(p1 * no + m, y * nv + z);
            if (t_swp) tv -= T2_flat(p1 * no + m, z * nv + y);
            value -= tv * vv;
        }
    }
    return value;
}

double d_image_value(
    const Mat& T1, const Mat& T2_flat, const Mat& f_ov,
    const IntegralBlocks& V, Eigen::Index no, Eigen::Index nv,
    const std::array<Eigen::Index, 3>& occ, const Pattern& pat,
    const std::array<int, 3>& po, const std::array<int, 3>& pv,
    Eigen::Index x, Eigen::Index y, Eigen::Index z) {
    const Eigen::Index p1 = occ[static_cast<std::size_t>(po[0])];
    const Eigen::Index p2 = occ[static_cast<std::size_t>(po[1])];
    const Eigen::Index p3 = occ[static_cast<std::size_t>(po[2])];
    const int s1 = pat.so[static_cast<std::size_t>(po[0])];
    const int s2 = pat.so[static_cast<std::size_t>(po[1])];
    const int s3 = pat.so[static_cast<std::size_t>(po[2])];
    const int z1 = pat.sv[static_cast<std::size_t>(pv[0])];
    const int z2 = pat.sv[static_cast<std::size_t>(pv[1])];
    const int z3 = pat.sv[static_cast<std::size_t>(pv[2])];
    if (s1 != z1) return 0.0;

    const bool v_cou = (s2 == z2 && s3 == z3);
    const bool v_exc = (s2 == z3 && s3 == z2);
    const bool t_dir = (s2 == z2 && s3 == z3);
    const bool t_swp = (s2 == z3 && s3 == z2);
    double vv = 0.0;
    if (v_cou) vv += V.ov_ov(p2 * nv + y, p3 * nv + z);
    if (v_exc) vv -= V.ov_ov(p2 * nv + z, p3 * nv + y);
    double tv = 0.0;
    if (t_dir) tv += T2_flat(p2 * no + p3, y * nv + z);
    if (t_swp) tv -= T2_flat(p2 * no + p3, z * nv + y);
    return T1(p1, x) * vv + f_ov(p1, x) * tv;
}

}  // namespace triples

// Perturbative triples, split into its two standard pieces:
//   first  = sum W (W)      / D   -- the fourth-order connected-triples
//                                    energy: CCSD[T], identically Urban's
//                                    CCSD+T(CCSD) (Urban, Noga, Cole,
//                                    Bartlett, J. Chem. Phys. 83, 4041
//                                    (1985), doi:10.1063/1.449067)
//   second = sum W (Wd)     / D   -- the fifth-order disconnected coupling
//                                    Raghavachari added; Wd contains T1*ERI
//                                    and, for semicanonical orbitals, Fov*T2.
//                                    Their sum is the standard CCSD(T).
struct TriplesEnergies {
    double fourth_order = 0.0;       // E[T]
    double disconnected_fifth = 0.0; // E_ST; E(T) = E[T] + E_ST
};

int triples_runtime_max_threads() {
#ifdef _OPENMP
    return std::max(1, omp_get_max_threads());
#else
    return 1;
#endif
}

TriplesEnergies compute_triples(const Mat& T1, const Mat& T2_flat,
                                const IntegralBlocks& V,
                                const Mat& f_ov,
                                const Eigen::VectorXd& eps_occ,
                                const Eigen::VectorXd& eps_vir,
                                int requested_threads = 0) {
    const auto no = static_cast<Eigen::Index>(T1.rows());
    const auto nv = static_cast<Eigen::Index>(T1.cols());
    if (no < 1 || nv < 1) return {};

    // P(i/jk) images and signs (applied to both occupied and virtual sides)
    static const std::array<std::pair<std::array<int, 3>, double>, 3> perms =
        {{{{0, 1, 2}, +1.0}, {{1, 0, 2}, -1.0}, {{2, 1, 0}, -1.0}}};

    // The two closed-shell spin classes, evaluated per occupied triple below:
    //   aaa = (aaa;aaa) external spins, weight 1/18 (same-spin block);
    //   aab = (aab;aab) external spins, weight 1/2  (opposite-spin block).
    // The weights are the spin-orbital pattern multiplicities of E(T) =
    // (1/36) sum_so W (W + Wd) / D collapsed for a closed-shell reference;
    // the three single-beta *virtual* patterns are already folded into the
    // 1/2 by the slot-swap antisymmetry of W (see the section header).
    static const triples::Pattern PAT_AAA = {{0, 0, 0}, {0, 0, 0}};
    static const triples::Pattern PAT_AAB = {{0, 0, 1}, {0, 0, 1}};  // beta=slot2
    constexpr double W_AAA = 1.0 / 18.0;
    constexpr double W_AAB = 0.5;

    // Restricted occupied loop (i <= j <= k).  The per-ordered-triple density
    // is *not* permutation-symmetric here: PAT_AAB pins the beta spin to the
    // third slot, so an ordered (i,j,k) only carries the "beta on k" occupied
    // configuration; the "beta on i / j" configurations are supplied by the
    // other orderings in the full loop.  Two sub-symmetries survive and let us
    // fold the orderings analytically (both follow from the antisymmetry of a
    // same-spin cluster amplitude under exchange of two same-spin indices,
    // J. Chem. Phys. 94, 4334 (1991)):
    //   H1  S_aaa is fully symmetric in (i,j,k)  -> weight by the multiset's
    //       ordering count (1 / 3 / 6).
    //   H2  S_aab is symmetric under exchanging its two alpha slots -> each
    //       distinct beta position needs one build, not two.
    // So per unordered triple we do 1 (aaa) + up to 3 (aab, one per beta
    // position) builds instead of 2 per ordered triple: ~3x fewer builds for
    // generic triples.  The pinned E(T) anchors in tests/test_ccsd_anchor.py
    // (H2O carries triples of every multiplicity class) verify exact
    // reproduction of the ordered-loop value.
    struct OccTriple {
        Eigen::Index i, j, k;
    };
    std::vector<OccTriple> work;
    work.reserve(static_cast<std::size_t>(no) * (no + 1) * (no + 2) / 6);
    for (Eigen::Index i = 0; i < no; ++i)
        for (Eigen::Index j = i; j < no; ++j)
            for (Eigen::Index k = j; k < no; ++k)
                work.push_back({i, j, k});
    const auto n_work = static_cast<Eigen::Index>(work.size());

    double e4 = 0.0, e5 = 0.0;
    const auto nv3 = static_cast<std::size_t>(nv * nv * nv);

    const int n_threads = requested_threads > 0
        ? requested_threads : triples_runtime_max_threads();
    #pragma omp parallel num_threads(n_threads) reduction(+:e4,e5)
    {
        std::vector<double> wc(nv3), wd(nv3);
        double s4 = 0.0, s5 = 0.0;  // per-contract outputs

        // Build W / Wd for one (spin class, occupied ordering) and contract
        // the (a,b,c) energy sums into (s4, s5).  occ[2] is the beta slot
        // for PAT_AAB.
        auto contract = [&](const triples::Pattern& pat,
                            const std::array<Eigen::Index, 3>& occ) {
            const double d_occ =
                eps_occ(occ[0]) + eps_occ(occ[1]) + eps_occ(occ[2]);
            std::fill(wc.begin(), wc.end(), 0.0);
            std::fill(wd.begin(), wd.end(), 0.0);
            for (const auto& [po, sgn_o] : perms)
                for (const auto& [pv, sgn_v] : perms) {
                    const double sign = sgn_o * sgn_v;
                    triples::accumulate_g_image(
                        T2_flat, V, no, nv, occ, pat, po, pv, sign, wc);
                    triples::accumulate_d_image(
                        T1, T2_flat, f_ov, V, no, nv,
                        occ, pat, po, pv, sign, wd);
                }
            s4 = 0.0;
            s5 = 0.0;
            std::size_t idx = 0;
            for (Eigen::Index a = 0; a < nv; ++a)
                for (Eigen::Index b = 0; b < nv; ++b)
                    for (Eigen::Index c = 0; c < nv; ++c, ++idx) {
                        const double d = d_occ - eps_vir(a)
                                       - eps_vir(b) - eps_vir(c);
                        s4 += wc[idx] * wc[idx] / d;
                        s5 += wc[idx] * wd[idx] / d;
                    }
        };

        #pragma omp for schedule(dynamic)
        for (Eigen::Index t = 0; t < n_work; ++t) {
            const Eigen::Index i = work[static_cast<std::size_t>(t)].i;
            const Eigen::Index j = work[static_cast<std::size_t>(t)].j;
            const Eigen::Index k = work[static_cast<std::size_t>(t)].k;

            // aaa: fully symmetric -> weight by the number of orderings.
            const double mult = (i == j && j == k) ? 1.0
                              : (i == j || j == k) ? 3.0
                                                   : 6.0;
            contract(PAT_AAA, {{i, j, k}});
            e4 += W_AAA * mult * s4;
            e5 += W_AAA * mult * s5;

            // aab: sum the distinct beta positions; alpha-slot pairs that are
            // equivalent under H2 are folded into the integer coefficient.
            double s4_aab = 0.0, s5_aab = 0.0;
            auto add_aab = [&](double coeff,
                               const std::array<Eigen::Index, 3>& occ) {
                contract(PAT_AAB, occ);
                s4_aab += coeff * s4;
                s5_aab += coeff * s5;
            };
            if (i == j && j == k) {                 // {i,i,i}
                add_aab(1.0, {{i, i, i}});
            } else if (i == j) {                    // {i,i,k}, i<k
                add_aab(1.0, {{i, i, k}});          // beta=k
                add_aab(2.0, {{i, k, i}});          // beta=i
            } else if (j == k) {                    // {i,j,j}, i<j
                add_aab(2.0, {{i, j, j}});          // beta=j
                add_aab(1.0, {{j, j, i}});          // beta=i
            } else {                                // {i,j,k} all distinct
                add_aab(2.0, {{i, j, k}});          // beta=k
                add_aab(2.0, {{i, k, j}});          // beta=j
                add_aab(2.0, {{j, k, i}});          // beta=i
            }
            e4 += W_AAB * s4_aab;
            e5 += W_AAB * s5_aab;
        }
    }
    return {e4, e5};
}
// Tiled variant: batches over external virtual 'a' to keep per-thread
// buffers at O(tile_size * nv^2) instead of O(nv^3).  Energy is bitwise
// identical to the untiled path.
TriplesEnergies compute_triples_tiled(
    const Mat& T1, const Mat& T2_flat,
    const IntegralBlocks& V,
    const Mat& f_ov,
    const Eigen::VectorXd& eps_occ,
    const Eigen::VectorXd& eps_vir,
    int tile_size, int requested_threads = 0,
    int factor_block_rows = 1) {

    const auto no = static_cast<Eigen::Index>(T1.rows());
    const auto nv = static_cast<Eigen::Index>(T1.cols());
    if (no < 1 || nv < 1 || tile_size < 1) return {};

    static const std::array<std::pair<std::array<int, 3>, double>, 3> perms =
        {{{{0, 1, 2}, +1.0}, {{1, 0, 2}, -1.0}, {{2, 1, 0}, -1.0}}};
    static const triples::Pattern PAT_AAA = {{0, 0, 0}, {0, 0, 0}};
    static const triples::Pattern PAT_AAB = {{0, 0, 1}, {0, 0, 1}};
    constexpr double W_AAA = 1.0 / 18.0;
    constexpr double W_AAB = 0.5;

    struct OccTriple { Eigen::Index i, j, k; };
    std::vector<OccTriple> work;
    work.reserve(static_cast<std::size_t>(no)*(no+1)*(no+2)/6);
    for (Eigen::Index i = 0; i < no; ++i)
        for (Eigen::Index j = i; j < no; ++j)
            for (Eigen::Index k = j; k < no; ++k)
                work.push_back({i, j, k});
    const auto n_work = static_cast<Eigen::Index>(work.size());

    double e4 = 0.0, e5 = 0.0;
    const Eigen::Index max_tile = static_cast<Eigen::Index>(tile_size);

    const int n_threads = requested_threads > 0
        ? requested_threads : triples_runtime_max_threads();
    std::atomic<bool> parallel_failed{false};
    std::exception_ptr parallel_error;
    #pragma omp parallel num_threads(n_threads) reduction(+:e4,e5)
    {
        const auto tile_nv2 = static_cast<std::size_t>(max_tile) *
                              static_cast<std::size_t>(nv) * static_cast<std::size_t>(nv);
        std::vector<double> wc_tile, wd_tile;
        std::unique_ptr<DiskDFFactorReader> disk_reader;
        bool worker_ready = true;
        const auto record_failure = [&]() {
            const std::exception_ptr error = std::current_exception();
            #pragma omp critical(vibeqc_ccsd_triples_failure)
            {
                if (!parallel_error) parallel_error = error;
            }
            parallel_failed.store(true, std::memory_order_release);
        };
        try {
            wc_tile.resize(tile_nv2);
            wd_tile.resize(tile_nv2);
            if (V.disk_factors) {
                disk_reader = std::make_unique<DiskDFFactorReader>(
                    *V.disk_factors,
                    static_cast<Eigen::Index>(factor_block_rows));
            }
        } catch (...) {
            worker_ready = false;
            record_failure();
        }
        double s4 = 0.0, s5 = 0.0;

        auto contract_batched = [&](const triples::Pattern& pat,
                                    const std::array<Eigen::Index, 3>& occ) {
            const double d_occ = eps_occ(occ[0])+eps_occ(occ[1])+eps_occ(occ[2]);
            s4 = 0.0; s5 = 0.0;
            for (Eigen::Index a0 = 0; a0 < nv; a0 += max_tile) {
                const Eigen::Index a1 = std::min(nv, a0 + max_tile);
                const Eigen::Index actual_n = a1 - a0;
                const auto buf_size = static_cast<std::size_t>(actual_n) *
                                      static_cast<std::size_t>(nv) *
                                      static_cast<std::size_t>(nv);
                std::fill(wc_tile.begin(), wc_tile.begin() + static_cast<std::ptrdiff_t>(buf_size), 0.0);
                std::fill(wd_tile.begin(), wd_tile.begin() + static_cast<std::ptrdiff_t>(buf_size), 0.0);
                for (const auto& [po, sgn_o] : perms) {
                    for (const auto& [pv, sgn_v] : perms) {
                        const double sign = sgn_o * sgn_v;
                        triples::accumulate_g_image_batched(
                            T2_flat, V, no, nv, occ, pat, po, pv,
                            sign, wc_tile, 0, a0, a1, disk_reader.get());
                        triples::accumulate_d_image_batched(
                            T1, T2_flat, f_ov, V, no, nv, occ, pat, po, pv,
                            sign, wd_tile, 0, a0, a1);
                    }
                }
                const auto t_stride0 = static_cast<std::size_t>(nv) *
                                       static_cast<std::size_t>(nv);
                for (Eigen::Index a = 0; a < actual_n; ++a) {
                    const double ea = eps_vir(a0 + a);
                    for (Eigen::Index b = 0; b < nv; ++b) {
                        const double eb = eps_vir(b);
                        for (Eigen::Index c = 0; c < nv; ++c) {
                            const double d = d_occ - ea - eb - eps_vir(c);
                            const auto idx = static_cast<std::size_t>(a) * t_stride0
                                           + static_cast<std::size_t>(b) * static_cast<std::size_t>(nv)
                                           + static_cast<std::size_t>(c);
                            s4 += wc_tile[idx] * wc_tile[idx] / d;
                            s5 += wc_tile[idx] * wd_tile[idx] / d;
                        }
                    }
                }
            }
        };

        #pragma omp for schedule(dynamic)
        for (Eigen::Index t = 0; t < n_work; ++t) {
            if (!worker_ready
                || parallel_failed.load(std::memory_order_acquire)) {
                continue;
            }
            try {
                const Eigen::Index i = work[static_cast<std::size_t>(t)].i;
                const Eigen::Index j = work[static_cast<std::size_t>(t)].j;
                const Eigen::Index k = work[static_cast<std::size_t>(t)].k;
                const double mult = (i==j&&j==k)?1.0:(i==j||j==k)?3.0:6.0;

                contract_batched(PAT_AAA, {{i, j, k}});
                e4 += W_AAA * mult * s4;
                e5 += W_AAA * mult * s5;

                double s4_aab = 0.0, s5_aab = 0.0;
                auto add_aab = [&](double coeff,
                                   const std::array<Eigen::Index,3>& occ) {
                    contract_batched(PAT_AAB, occ);
                    s4_aab += coeff * s4;
                    s5_aab += coeff * s5;
                };
                if (i==j&&j==k) { add_aab(1.0,{{i,i,i}}); }
                else if (i==j)  { add_aab(1.0,{{i,i,k}});add_aab(2.0,{{i,k,i}});}
                else if (j==k)  { add_aab(2.0,{{i,j,j}});add_aab(1.0,{{j,j,i}});}
                else            { add_aab(2.0,{{i,j,k}});add_aab(2.0,{{i,k,j}});
                                  add_aab(2.0,{{j,k,i}}); }
                e4 += W_AAB * s4_aab;
                e5 += W_AAB * s5_aab;
            } catch (...) {
                worker_ready = false;
                record_failure();
            }
        }
    }
    if (parallel_error) std::rethrow_exception(parallel_error);
    return {e4, e5};
}

// Minimum-workspace triples.  W and Wd are formed for one (a,b,c) element,
// contracted into the energy, and discarded immediately.  This is slower than
// the dense/tiled routes but gives the planner a final in-core fallback whose
// heap workspace is only the shared occupied-triple work list.
TriplesEnergies compute_triples_direct(
    const Mat& T1, const Mat& T2_flat, const IntegralBlocks& V,
    const Mat& f_ov, const Eigen::VectorXd& eps_occ,
    const Eigen::VectorXd& eps_vir, int requested_threads = 0) {
    const auto no = static_cast<Eigen::Index>(T1.rows());
    const auto nv = static_cast<Eigen::Index>(T1.cols());
    if (no < 1 || nv < 1) return {};

    static const std::array<std::pair<std::array<int, 3>, double>, 3> perms =
        {{{{0, 1, 2}, +1.0}, {{1, 0, 2}, -1.0}, {{2, 1, 0}, -1.0}}};
    static const triples::Pattern PAT_AAA = {{0, 0, 0}, {0, 0, 0}};
    static const triples::Pattern PAT_AAB = {{0, 0, 1}, {0, 0, 1}};
    constexpr double W_AAA = 1.0 / 18.0;
    constexpr double W_AAB = 0.5;

    struct OccTriple { Eigen::Index i, j, k; };
    std::vector<OccTriple> work;
    work.reserve(static_cast<std::size_t>(no) * (no + 1) * (no + 2) / 6);
    for (Eigen::Index i = 0; i < no; ++i)
        for (Eigen::Index j = i; j < no; ++j)
            for (Eigen::Index k = j; k < no; ++k)
                work.push_back({i, j, k});

    const int n_threads = requested_threads > 0
        ? requested_threads : triples_runtime_max_threads();
    double e4 = 0.0, e5 = 0.0;
    #pragma omp parallel for num_threads(n_threads) reduction(+:e4,e5) schedule(dynamic)
    for (Eigen::Index t = 0; t < static_cast<Eigen::Index>(work.size()); ++t) {
        const Eigen::Index i = work[static_cast<std::size_t>(t)].i;
        const Eigen::Index j = work[static_cast<std::size_t>(t)].j;
        const Eigen::Index k = work[static_cast<std::size_t>(t)].k;

        const auto contract = [&](const triples::Pattern& pat,
                                  const std::array<Eigen::Index, 3>& occ) {
            double s4 = 0.0, s5 = 0.0;
            const double d_occ =
                eps_occ(occ[0]) + eps_occ(occ[1]) + eps_occ(occ[2]);
            for (Eigen::Index a = 0; a < nv; ++a)
                for (Eigen::Index b = 0; b < nv; ++b)
                    for (Eigen::Index c = 0; c < nv; ++c) {
                        const std::array<Eigen::Index, 3> ext = {{a, b, c}};
                        double wc = 0.0, wd = 0.0;
                        for (const auto& [po, sgn_o] : perms)
                            for (const auto& [pv, sgn_v] : perms) {
                                const double sign = sgn_o * sgn_v;
                                const Eigen::Index x =
                                    ext[static_cast<std::size_t>(pv[0])];
                                const Eigen::Index y =
                                    ext[static_cast<std::size_t>(pv[1])];
                                const Eigen::Index z =
                                    ext[static_cast<std::size_t>(pv[2])];
                                wc += sign * triples::g_image_value(
                                    T2_flat, V, no, nv, occ, pat, po, pv,
                                    x, y, z);
                                wd += sign * triples::d_image_value(
                                    T1, T2_flat, f_ov, V, no, nv, occ, pat,
                                    po, pv, x, y, z);
                            }
                        const double d = d_occ - eps_vir(a)
                                       - eps_vir(b) - eps_vir(c);
                        s4 += wc * wc / d;
                        s5 += wc * wd / d;
                    }
            return std::pair<double, double>{s4, s5};
        };

        const double mult = (i == j && j == k) ? 1.0
                          : (i == j || j == k) ? 3.0 : 6.0;
        const auto aaa = contract(PAT_AAA, {{i, j, k}});
        e4 += W_AAA * mult * aaa.first;
        e5 += W_AAA * mult * aaa.second;

        double s4_aab = 0.0, s5_aab = 0.0;
        const auto add_aab = [&](double coeff,
                                 const std::array<Eigen::Index, 3>& occ) {
            const auto part = contract(PAT_AAB, occ);
            s4_aab += coeff * part.first;
            s5_aab += coeff * part.second;
        };
        if (i == j && j == k) {
            add_aab(1.0, {{i, i, i}});
        } else if (i == j) {
            add_aab(1.0, {{i, i, k}});
            add_aab(2.0, {{i, k, i}});
        } else if (j == k) {
            add_aab(2.0, {{i, j, j}});
            add_aab(1.0, {{j, j, i}});
        } else {
            add_aab(2.0, {{i, j, k}});
            add_aab(2.0, {{i, k, j}});
            add_aab(2.0, {{j, k, i}});
        }
        e4 += W_AAB * s4_aab;
        e5 += W_AAB * s5_aab;
    }
    return {e4, e5};
}

enum class TriplesMemoryStrategy { Fast, Blocked, Direct, Disk };

struct TriplesMemoryPlan {
    TriplesMemoryStrategy strategy = TriplesMemoryStrategy::Fast;
    std::string name = "fast";
    int tile_size = 0;
    int threads = 1;
    // Full modeled triples peak: retained numerical inputs plus aggregate
    // scratch of every active worker.  This is directly comparable to the
    // requested-memory cap and is the value exported as result telemetry.
    std::size_t workspace_bytes = 0;
    std::size_t retained_bytes = 0;
    std::size_t disk_bytes = 0;
    int factor_block_rows = 1;
    bool factor_direct = false;
    std::string scratch_directory;
};

std::size_t saturating_add(std::size_t a, std::size_t b) {
    const auto max = std::numeric_limits<std::size_t>::max();
    return b > max - a ? max : a + b;
}

std::size_t saturating_mul(std::size_t a, std::size_t b) {
    if (a == 0 || b == 0) return 0;
    const auto max = std::numeric_limits<std::size_t>::max();
    return a > max / b ? max : a * b;
}

template <typename Derived>
std::size_t matrix_storage_bytes(const Eigen::MatrixBase<Derived>& m) {
    return saturating_mul(static_cast<std::size_t>(m.size()), sizeof(double));
}

std::string normalise_triples_memory_mode(const std::string& input) {
    std::string mode;
    for (const char c : input) {
        if (std::isspace(static_cast<unsigned char>(c))) continue;
        mode.push_back(static_cast<char>(
            std::tolower(static_cast<unsigned char>(c))));
    }
    if (mode.empty()) mode = "auto";
    if (mode == "low") mode = "blocked";
    if (mode != "auto" && mode != "fast" && mode != "blocked"
        && mode != "direct" && mode != "disk") {
        throw std::invalid_argument(
            "run_ccsd: unknown triples_memory_mode '" + input
            + "'; supported: auto, fast, blocked (legacy: low), direct, disk");
    }
    return mode;
}

std::size_t triples_work_list_bytes(Eigen::Index no) {
    const std::size_t n = static_cast<std::size_t>(no);
    const std::size_t entries =
        saturating_mul(saturating_mul(n, n + 1), n + 2) / 6;
    return saturating_mul(entries, 3 * sizeof(Eigen::Index));
}

std::size_t retained_triples_bytes(
    const Mat& T1, const Mat& T2_flat, const IntegralBlocks& V,
    const Mat& f_ov, const Eigen::VectorXd& eps_occ,
    const Eigen::VectorXd& eps_vir, bool factor_direct,
    bool disk_backed = false) {
    std::size_t bytes = 0;
    const auto add = [&](std::size_t value) { bytes = saturating_add(bytes, value); };
    add(matrix_storage_bytes(T1));
    add(matrix_storage_bytes(T2_flat));
    add(matrix_storage_bytes(f_ov));
    add(matrix_storage_bytes(eps_occ));
    add(matrix_storage_bytes(eps_vir));
    add(matrix_storage_bytes(V.ov_ov));
    add(matrix_storage_bytes(V.oo_ov));
    if (disk_backed) {
        // Only a small RAII file descriptor/path object remains resident;
        // B_ov/B_vv themselves are released after the spill.
    } else if (factor_direct) {
        add(matrix_storage_bytes(V.B_ov));
        add(matrix_storage_bytes(V.B_vv));
    } else {
        add(matrix_storage_bytes(V.ov_vv));
    }
    return bytes;
}

TriplesMemoryPlan plan_triples_memory(
    const Mat& T1, const Mat& T2_flat, const IntegralBlocks& V,
    const Mat& f_ov, const Eigen::VectorXd& eps_occ,
    const Eigen::VectorXd& eps_vir, const CCSDOptions& opts) {
    if (opts.triples_tile_size < 0)
        throw std::invalid_argument("run_ccsd: triples_tile_size must be >= 0");
    if (opts.triples_max_threads < 0)
        throw std::invalid_argument("run_ccsd: triples_max_threads must be >= 0");

    const std::string mode = normalise_triples_memory_mode(opts.triples_memory_mode);
    const Eigen::Index no = T1.rows();
    const Eigen::Index nv = T1.cols();
    const std::size_t nv_sz = static_cast<std::size_t>(nv);
    const std::size_t nv2 = saturating_mul(nv_sz, nv_sz);
    const std::size_t nv3 = saturating_mul(nv2, nv_sz);
    const std::size_t work_bytes = triples_work_list_bytes(no);
    const std::size_t work_entries = static_cast<std::size_t>(no)
        * static_cast<std::size_t>(no + 1)
        * static_cast<std::size_t>(no + 2) / 6;
    int max_threads = triples_runtime_max_threads();
    if (opts.triples_max_threads > 0)
        max_threads = std::min(max_threads, opts.triples_max_threads);
    if (work_entries > 0)
        max_threads = std::min(max_threads, static_cast<int>(std::min<std::size_t>(
            work_entries, static_cast<std::size_t>(std::numeric_limits<int>::max()))));
    max_threads = std::max(1, max_threads);

    const bool has_df_factors = V.B_ov.size() != 0 && V.B_vv.size() != 0
        && V.B_ov.rows() == V.B_vv.rows();
    const std::size_t fast_retained = retained_triples_bytes(
        T1, T2_flat, V, f_ov, eps_occ, eps_vir, false);
    const std::size_t low_retained = retained_triples_bytes(
        T1, T2_flat, V, f_ov, eps_occ, eps_vir, has_df_factors);
    const std::size_t disk_retained = retained_triples_bytes(
        T1, T2_flat, V, f_ov, eps_occ, eps_vir, false, true);
    const std::size_t fast_per_thread = saturating_mul(
        saturating_add(saturating_mul(4, nv3), nv2), sizeof(double));
    const std::size_t blocked_tile_per_thread =
        saturating_mul(saturating_mul(2, nv2), sizeof(double));
    const std::size_t factor_row_bytes = has_df_factors
        ? saturating_mul(
              saturating_add(static_cast<std::size_t>(V.B_ov.cols()),
                             static_cast<std::size_t>(V.B_vv.cols())),
              sizeof(double))
        : 0;
    const std::size_t disk_bytes = has_df_factors
        ? saturating_add(
              DiskDFFactorStore::kHeaderBytes,
              saturating_mul(static_cast<std::size_t>(V.B_ov.rows()),
                             factor_row_bytes))
        : 0;
    const bool bounded = opts.requested_memory_bytes > 0;
    const std::size_t budget = opts.requested_memory_bytes;

    const auto fits = [&](std::size_t modeled_peak) {
        return !bounded || modeled_peak <= budget;
    };
    const auto make_fast = [&](int threads) {
        TriplesMemoryPlan plan;
        plan.strategy = TriplesMemoryStrategy::Fast;
        plan.name = "fast";
        plan.tile_size = static_cast<int>(nv);
        plan.threads = threads;
        const std::size_t scratch = saturating_add(
            work_bytes, saturating_mul(static_cast<std::size_t>(threads),
                                       fast_per_thread));
        plan.retained_bytes = fast_retained;
        plan.workspace_bytes = saturating_add(plan.retained_bytes, scratch);
        return plan;
    };
    const auto make_blocked = [&](int threads, int tile) {
        TriplesMemoryPlan plan;
        plan.strategy = TriplesMemoryStrategy::Blocked;
        plan.name = "blocked";
        plan.tile_size = tile;
        plan.threads = threads;
        const std::size_t scratch = saturating_add(
            work_bytes,
            saturating_mul(
                saturating_mul(static_cast<std::size_t>(threads),
                               static_cast<std::size_t>(tile)),
                blocked_tile_per_thread));
        plan.retained_bytes = low_retained;
        plan.workspace_bytes = saturating_add(plan.retained_bytes, scratch);
        plan.factor_direct = has_df_factors;
        return plan;
    };
    const auto make_direct = [&](int threads) {
        TriplesMemoryPlan plan;
        plan.strategy = TriplesMemoryStrategy::Direct;
        plan.name = "direct";
        plan.tile_size = 0;
        plan.threads = threads;
        plan.retained_bytes = low_retained;
        plan.workspace_bytes = saturating_add(plan.retained_bytes, work_bytes);
        plan.factor_direct = has_df_factors;
        return plan;
    };
    const auto make_disk = [&](int threads, int tile) {
        TriplesMemoryPlan plan;
        plan.strategy = TriplesMemoryStrategy::Disk;
        plan.name = "disk";
        plan.tile_size = tile;
        plan.threads = threads;
        plan.factor_block_rows = 1;
        plan.retained_bytes = disk_retained;
        const std::size_t per_thread = saturating_add(
            saturating_mul(static_cast<std::size_t>(tile),
                           blocked_tile_per_thread),
            factor_row_bytes);
        const std::size_t scratch = saturating_add(
            work_bytes,
            saturating_mul(static_cast<std::size_t>(threads), per_thread));
        const std::size_t streamed_peak = saturating_add(
            plan.retained_bytes, scratch);
        // Spilling must briefly read the already-resident source factors.
        // Count that transition peak too; disk telemetry therefore never
        // claims a cap below the memory physically required to create its
        // scratch representation.
        plan.workspace_bytes = std::max(low_retained, streamed_peak);
        plan.disk_bytes = disk_bytes;
        plan.scratch_directory = opts.triples_scratch_directory;
        return plan;
    };

    const auto find_fast = [&](bool allow_thread_reduction,
                               TriplesMemoryPlan& result) {
        const int min_threads = allow_thread_reduction ? 1 : max_threads;
        for (int threads = max_threads; threads >= min_threads; --threads) {
            const auto candidate = make_fast(threads);
            if (fits(candidate.workspace_bytes)) {
                result = candidate;
                return true;
            }
        }
        return false;
    };
    const auto find_blocked = [&](TriplesMemoryPlan& result) {
        const int requested_tile = opts.triples_tile_size > 0
            ? std::min<int>(opts.triples_tile_size, static_cast<int>(nv)) : 0;
        for (int threads = max_threads; threads >= 1; --threads) {
            int tile = requested_tile > 0 ? requested_tile : static_cast<int>(nv);
            if (bounded) {
                const std::size_t fixed = saturating_add(low_retained, work_bytes);
                if (fixed > budget) continue;
                const std::size_t denom = saturating_mul(
                    static_cast<std::size_t>(threads), blocked_tile_per_thread);
                const std::size_t max_tile = denom == 0 ? nv_sz
                    : (budget - fixed) / denom;
                tile = std::min<int>(tile, static_cast<int>(
                    std::min<std::size_t>(nv_sz, max_tile)));
            } else if (requested_tile == 0) {
                tile = std::min<int>(64, static_cast<int>(nv));
            }
            if (tile < 1) continue;
            const auto candidate = make_blocked(threads, tile);
            if (fits(candidate.workspace_bytes)) {
                result = candidate;
                return true;
            }
        }
        return false;
    };
    const auto find_disk = [&](TriplesMemoryPlan& result) {
        if (!has_df_factors) return false;
        const int requested_tile = opts.triples_tile_size > 0
            ? std::min<int>(opts.triples_tile_size, static_cast<int>(nv)) : 0;
        for (int threads = max_threads; threads >= 1; --threads) {
            int tile = requested_tile > 0 ? requested_tile
                : std::min<int>(64, static_cast<int>(nv));
            if (bounded) {
                const std::size_t fixed = saturating_add(
                    saturating_add(disk_retained, work_bytes),
                    saturating_mul(static_cast<std::size_t>(threads),
                                   factor_row_bytes));
                if (fixed > budget) continue;
                const std::size_t denom = saturating_mul(
                    static_cast<std::size_t>(threads),
                    blocked_tile_per_thread);
                const std::size_t max_tile = denom == 0 ? nv_sz
                    : (budget - fixed) / denom;
                tile = std::min<int>(tile, static_cast<int>(
                    std::min<std::size_t>(nv_sz, max_tile)));
            }
            if (tile < 1) continue;
            const auto candidate = make_disk(threads, tile);
            if (fits(candidate.workspace_bytes)) {
                result = candidate;
                return true;
            }
        }
        return false;
    };

    TriplesMemoryPlan plan;
    if (mode == "fast") {
        if (find_fast(true, plan)) return plan;
    } else if (mode == "blocked") {
        if (find_blocked(plan)) return plan;
    } else if (mode == "direct") {
        plan = make_direct(max_threads);
        if (fits(plan.workspace_bytes)) return plan;
    } else if (mode == "disk") {
        if (!has_df_factors)
            throw std::runtime_error(
                "run_ccsd: triples_memory_mode='disk' requires the "
                "density-fitted CCSD route");
        if (find_disk(plan)) return plan;
    } else {  // auto
        if (!bounded) {
            constexpr std::size_t kDefaultFastWorkspace = 512ULL << 20;
            plan = make_fast(max_threads);
            if (plan.workspace_bytes <= kDefaultFastWorkspace) return plan;
        } else if (find_fast(false, plan)) {
            return plan;
        }
        if (find_blocked(plan)) return plan;
        plan = make_direct(max_threads);
        if (fits(plan.workspace_bytes)) return plan;
        if (find_disk(plan)) return plan;
    }

    const std::size_t direct_minimum = saturating_add(low_retained, work_bytes);
    const std::size_t disk_minimum = has_df_factors
        ? std::max(
              low_retained,
              saturating_add(
                  saturating_add(disk_retained, work_bytes),
                  saturating_add(factor_row_bytes,
                                 blocked_tile_per_thread)))
        : std::numeric_limits<std::size_t>::max();
    const std::size_t minimum = std::min(direct_minimum, disk_minimum);
    std::ostringstream msg;
    msg << "run_ccsd: requested triples memory budget of " << budget
        << " bytes cannot hold the selected '" << mode << "' strategy";
    if (mode == "auto" || mode == "direct" || mode == "disk")
        msg << "; minimum modeled triples peak is " << minimum << " bytes";
    else
        msg << "; use triples_memory_mode='auto' or 'direct', or request more memory";
    throw std::runtime_error(msg.str());
}

void release_matrix(Mat& matrix) {
    matrix.resize(0, 0);
}

void release_matrix(RowMat& matrix) {
    matrix.resize(0, 0);
}

void prepare_triples_integrals(IntegralBlocks& V,
                               const TriplesMemoryPlan& plan) {
    // CCSD-only blocks are never read by any triples strategy.
    release_matrix(V.oo_oo);
    release_matrix(V.oo_vv);
    release_matrix(V.vv_vv);
    if (plan.strategy == TriplesMemoryStrategy::Disk) {
        // Spill first, then release both the materialised ov_vv block and the
        // in-core factors.  If writing fails, the partially constructed store
        // removes its file while V still owns the original matrices.
        release_matrix(V.ov_vv);
        V.disk_factors = DiskDFFactorStore::spill(
            V.B_ov, V.B_vv, plan.scratch_directory);
        release_matrix(V.B_ov);
        release_matrix(V.B_vv);
    } else if (plan.factor_direct) {
        // DF blocked/direct term A is regenerated from B_ov/B_vv.
        release_matrix(V.ov_vv);
    } else {
        // Fast and canonical blocked/direct paths retain ov_vv, not DF factors.
        release_matrix(V.B_ov);
        release_matrix(V.B_vv);
    }
}


}  // namespace

// =========================================================================
// Public entry point: dlpno_pair_residual (DLPNO local-solver kernel)
// =========================================================================
void dlpno_pair_residual(
    const Mat& T1, const Mat& T2_flat,
    const RowMat& B_ov, const RowMat& B_oo, const RowMat& B_vv,
    const Mat& f_oo, const Mat& f_vv, const Mat& f_ov,
    Mat& R1, Mat& R2_flat, bool include_ladder, bool include_ring,
    Mat* W1_out, Mat* W2_out, Mat* WX_out) {
    const Eigen::Index no = T1.rows();
    const Eigen::Index nv = T1.cols();
    // Reuse the validated block builder + residual kernel (file-local).
    DFBlocks df{B_ov, B_vv, B_oo, B_ov.rows(), no, nv};
    const IntegralBlocks V = build_integral_blocks(std::move(df), include_ladder);
    // compute_residuals writes into pre-sized outputs (it does not resize).
    R1 = Mat::Zero(no, nv);
    R2_flat = Mat::Zero(no * no, nv * nv);
    compute_residuals(T1, T2_flat, V, f_oo, f_vv, f_ov, R1, R2_flat,
                      -1, -1, include_ladder, include_ring,
                      W1_out, W2_out, WX_out);
}

void dlpno_target_pair_residual(
    const Mat& T1, const Mat& T2_flat,
    const RowMat& B_ov, const RowMat& B_oo, const RowMat& B_vv,
    const Mat& f_oo, const Mat& f_vv, const Mat& f_ov,
    Eigen::Index target_i, Eigen::Index target_j,
    Mat& R1_target, Mat& R2_target, bool include_ladder, bool include_ring,
    Mat* W1_out, Mat* W2_out, Mat* WX_out) {
    const Eigen::Index no = T1.rows();
    const Eigen::Index nv = T1.cols();
    if (target_i < 0 || target_i >= no || target_j < 0 || target_j >= no) {
        throw std::invalid_argument(
            "dlpno_target_pair_residual: target occupied indices are out of range");
    }

    DFBlocks df{B_ov, B_vv, B_oo, B_ov.rows(), no, nv};
    const IntegralBlocks V = build_integral_blocks(std::move(df), include_ladder);
    Mat R1 = Mat::Zero(no, nv);
    Mat R2_flat = Mat::Zero(no * no, nv * nv);
    compute_residuals(
        T1, T2_flat, V, f_oo, f_vv, f_ov, R1, R2_flat, target_i, target_j,
        include_ladder, include_ring, W1_out, W2_out, WX_out);

    R1_target = R1.row(target_i);
    R2_target.resize(nv, nv);
    const Eigen::Index target_ij = target_i * no + target_j;
    for (Eigen::Index a = 0; a < nv; ++a)
        for (Eigen::Index b = 0; b < nv; ++b)
            R2_target(a, b) = R2_flat(target_ij, a * nv + b);
}

// =========================================================================
// Public entry point: dlpno_spatial_triples_correction
// =========================================================================
double dlpno_spatial_triples_correction(
    const Mat& T1, const Mat& T2_flat,
    const RowMat& B_ov, const RowMat& B_oo, const RowMat& B_vv,
    const Eigen::VectorXd& eps_o, const Eigen::VectorXd& eps_v) {
    const Eigen::Index no = T1.rows();
    const Eigen::Index nv = T1.cols();
    const Eigen::Index n_aux = B_ov.rows();

    if (T2_flat.rows() != no * no || T2_flat.cols() != nv * nv) {
        throw std::invalid_argument(
            "dlpno_spatial_triples_correction: T2_flat must have shape "
            "(n_occ*n_occ, n_vir*n_vir)");
    }
    if (B_ov.cols() != no * nv) {
        throw std::invalid_argument(
            "dlpno_spatial_triples_correction: B_ov must have shape "
            "(n_aux, n_occ*n_vir)");
    }
    if (B_oo.rows() != n_aux || B_oo.cols() != no * no) {
        throw std::invalid_argument(
            "dlpno_spatial_triples_correction: B_oo must have shape "
            "(n_aux, n_occ*n_occ)");
    }
    if (B_vv.rows() != n_aux || B_vv.cols() != nv * nv) {
        throw std::invalid_argument(
            "dlpno_spatial_triples_correction: B_vv must have shape "
            "(n_aux, n_vir*n_vir)");
    }
    if (eps_o.size() != no || eps_v.size() != nv) {
        throw std::invalid_argument(
            "dlpno_spatial_triples_correction: epsilon vector sizes must "
            "match n_occ and n_vir");
    }
    if (no < 1 || nv < 1) return 0.0;

    DFBlocks df{B_ov, B_vv, B_oo, n_aux, no, nv};
    const IntegralBlocks V = build_integral_blocks(std::move(df));
    const Mat f_ov = Mat::Zero(no, nv);
    const auto nv_sz_d = static_cast<std::size_t>(nv);
    constexpr std::size_t kTiledThreshold = 512ULL << 20;
    int tile_sz = 0;
    if (const char* e = std::getenv("VIBEQC_TRIPLES_TILE_SIZE")) {
        try { tile_sz = std::stoi(e); } catch (...) {}
    }
    if (tile_sz <= 0 && nv_sz_d * nv_sz_d * nv_sz_d * 16 > kTiledThreshold) {
        tile_sz = 64;
    }
    const auto parts =
        (tile_sz > 0 && nv > tile_sz)
        ? compute_triples_tiled(T1, T2_flat, V, f_ov, eps_o, eps_v, tile_sz)
        : compute_triples(T1, T2_flat, V, f_ov, eps_o, eps_v);
    return parts.fourth_order + parts.disconnected_fifth;
}

// =========================================================================
// Public entry point: spatial_lambda_triples
// =========================================================================
// Closed-shell A-CCSD(T) / Lambda triples correction.
//
// Right-hand triples moment W(T2) from CCSD amplitudes; left-hand moment
// W(L2) + Wd(L1) from Lambda multipliers.  For canonical orbitals
// (f_ov = 0), replacing L by T reduces exactly to the standard (T)
// correction.  OpenMP-parallel over the no^3 occupied triples.
double spatial_lambda_triples(
    const Mat& T1, const Mat& T2_flat,
    const Mat& L1, const Mat& L2_flat,
    const Mat& f_ov,
    const Mat& ov_vv, const Mat& oo_ov, const Mat& ov_ov,
    const Eigen::VectorXd& eps_o, const Eigen::VectorXd& eps_v)
{
    const Eigen::Index no = T1.rows();
    const Eigen::Index nv = T1.cols();
    if (no < 1 || nv < 1) return 0.0;

    static const std::array<std::pair<std::array<int, 3>, double>, 3> perms =
        {{{{0, 1, 2}, +1.0}, {{1, 0, 2}, -1.0}, {{2, 1, 0}, -1.0}}};
    static const triples::Pattern PAT_AAA = {{0, 0, 0}, {0, 0, 0}};
    static const triples::Pattern PAT_AAB = {{0, 0, 1}, {0, 0, 1}};
    constexpr double W_AAA = 1.0 / 18.0;
    constexpr double W_AAB = 0.5;

    IntegralBlocks V;
    V.ov_vv = ov_vv;
    V.oo_ov = oo_ov;
    V.ov_ov = ov_ov;

    double e_t = 0.0;
    #pragma omp parallel for collapse(3) reduction(+:e_t) schedule(static)
    for (Eigen::Index i = 0; i < no; ++i)
        for (Eigen::Index j = 0; j < no; ++j)
            for (Eigen::Index k = 0; k < no; ++k) {
                // ---- Per-triple Lambda energy, mirroring
                //      dlpno_spatial_triple_energy but with separate
                //      right-hand (T2) and left-hand (L1,L2) buffers. ----
                const std::array<Eigen::Index, 3> occ = {{i, j, k}};
                const double d_occ = eps_o(i) + eps_o(j) + eps_o(k);
                const auto nv3 = static_cast<std::size_t>(nv * nv * nv);

                // Accumulate right-hand connected buffer (T2 only)
                auto wc_from_T2 = [&](const triples::Pattern& pat,
                                      std::vector<double>& w) {
                    std::fill(w.begin(), w.end(), 0.0);
                    for (const auto& [po, sgn_o] : perms)
                        for (const auto& [pv, sgn_v] : perms)
                            triples::accumulate_g_image(
                                T2_flat, V, no, nv, occ, pat,
                                po, pv, sgn_o * sgn_v, w);
                };

                // Accumulate left-hand connected + disconnected from L
                auto wcwd_from_L = [&](const triples::Pattern& pat,
                                       std::vector<double>& wc,
                                       std::vector<double>& wd) {
                    std::fill(wc.begin(), wc.end(), 0.0);
                    std::fill(wd.begin(), wd.end(), 0.0);
                    for (const auto& [po, sgn_o] : perms)
                        for (const auto& [pv, sgn_v] : perms) {
                            const double sign = sgn_o * sgn_v;
                            triples::accumulate_g_image(
                                L2_flat, V, no, nv, occ, pat,
                                po, pv, sign, wc);
                            triples::accumulate_d_image(
                                L1, L2_flat, f_ov, V, no, nv,
                                occ, pat, po, pv, sign, wd);
                        }
                };

                auto contract_lam = [&](const triples::Pattern& pat,
                                        double weight) {
                    std::vector<double> wc_r(nv3), wc_l(nv3), wd_l(nv3);
                    wc_from_T2(pat, wc_r);
                    wcwd_from_L(pat, wc_l, wd_l);
                    double e = 0.0;
                    std::size_t idx = 0;
                    for (Eigen::Index a = 0; a < nv; ++a)
                        for (Eigen::Index b = 0; b < nv; ++b)
                            for (Eigen::Index c = 0; c < nv; ++c, ++idx) {
                                const double d = d_occ
                                    - eps_v(a) - eps_v(b) - eps_v(c);
                                e += weight * wc_r[idx]
                                   * (wc_l[idx] + wd_l[idx]) / d;
                            }
                    return e;
                };

                e_t += contract_lam(PAT_AAA, W_AAA);
                e_t += contract_lam(PAT_AAB, W_AAB);
            }
    return e_t;
}

// =========================================================================
// Public entry point: dlpno_spatial_triple_energy
// =========================================================================
double dlpno_spatial_triple_energy(
    Eigen::Index i, Eigen::Index j, Eigen::Index k,
    const Mat& T1, const Mat& T2_flat,
    const RowMat& ov_vv, const RowMat& oo_ov, const RowMat& ov_ov,
    const Eigen::VectorXd& eps_o, const Eigen::VectorXd& eps_v) {
    const Eigen::Index no = T1.rows();
    const Eigen::Index nv = T1.cols();

    if (i < 0 || j < 0 || k < 0 || i >= no || j >= no || k >= no) {
        throw std::invalid_argument(
            "dlpno_spatial_triple_energy: occupied indices out of range");
    }
    if (T2_flat.rows() != no * no || T2_flat.cols() != nv * nv) {
        throw std::invalid_argument(
            "dlpno_spatial_triple_energy: T2_flat must have shape "
            "(n_occ*n_occ, n_vir*n_vir)");
    }
    if (ov_vv.rows() != no * nv || ov_vv.cols() != nv * nv) {
        throw std::invalid_argument(
            "dlpno_spatial_triple_energy: ov_vv must have shape "
            "(n_occ*n_vir, n_vir*n_vir)");
    }
    if (oo_ov.rows() != no * no || oo_ov.cols() != no * nv) {
        throw std::invalid_argument(
            "dlpno_spatial_triple_energy: oo_ov must have shape "
            "(n_occ*n_occ, n_occ*n_vir)");
    }
    if (ov_ov.rows() != no * nv || ov_ov.cols() != no * nv) {
        throw std::invalid_argument(
            "dlpno_spatial_triple_energy: ov_ov must have shape "
            "(n_occ*n_vir, n_occ*n_vir)");
    }
    if (eps_o.size() != no || eps_v.size() != nv) {
        throw std::invalid_argument(
            "dlpno_spatial_triple_energy: epsilon vector sizes must match "
            "n_occ and n_vir");
    }
    if (no < 1 || nv < 1) return 0.0;

    static const std::array<std::pair<std::array<int, 3>, double>, 3> perms =
        {{{{0, 1, 2}, +1.0}, {{1, 0, 2}, -1.0}, {{2, 1, 0}, -1.0}}};
    static const triples::Pattern PAT_AAA = {{0, 0, 0}, {0, 0, 0}};
    static const triples::Pattern PAT_AAB = {{0, 0, 1}, {0, 0, 1}};
    constexpr double W_AAA = 1.0 / 18.0;
    constexpr double W_AAB = 0.5;

    IntegralBlocks V;
    V.ov_vv = ov_vv;
    V.oo_ov = oo_ov;
    V.ov_ov = ov_ov;

    const auto nv3 = static_cast<std::size_t>(nv * nv * nv);
    std::vector<double> wc(nv3), wd(nv3);
    const Mat f_ov = Mat::Zero(no, nv);
    const std::array<Eigen::Index, 3> occ = {{i, j, k}};
    const double d_occ = eps_o(i) + eps_o(j) + eps_o(k);

    auto contract = [&](const triples::Pattern& pat) {
        std::fill(wc.begin(), wc.end(), 0.0);
        std::fill(wd.begin(), wd.end(), 0.0);
        for (const auto& [po, sgn_o] : perms)
            for (const auto& [pv, sgn_v] : perms) {
                const double sign = sgn_o * sgn_v;
                triples::accumulate_g_image(
                    T2_flat, V, no, nv, occ, pat, po, pv, sign, wc);
                triples::accumulate_d_image(
                    T1, T2_flat, f_ov, V, no, nv,
                    occ, pat, po, pv, sign, wd);
            }

        double e = 0.0;
        std::size_t idx = 0;
        for (Eigen::Index a = 0; a < nv; ++a)
            for (Eigen::Index b = 0; b < nv; ++b)
                for (Eigen::Index c = 0; c < nv; ++c, ++idx) {
                    const double d = d_occ - eps_v(a) - eps_v(b) - eps_v(c);
                    e += wc[idx] * (wc[idx] + wd[idx]) / d;
                }
        return e;
    };

    return W_AAA * contract(PAT_AAA) + W_AAB * contract(PAT_AAB);
}

// =========================================================================
// Public entry point: dlpno_project_tno_amplitudes
// =========================================================================
void dlpno_project_tno_amplitudes(
    const Mat& V_T,
    int n_act,
    const std::vector<int>& pair_i,
    const std::vector<int>& pair_j,
    const std::vector<Mat>& U_pno,
    const std::vector<Mat>& T2_pno,
    const std::vector<int>& t1_idx,
    const std::vector<Eigen::VectorXd>& t1_vec,
    Mat& t1_out,
    Mat& T2_out) {

    const Eigen::Index nv = V_T.rows();
    const Eigen::Index n_T = V_T.cols();
    if (n_T < 1 || nv < 1) return;

    const Eigen::Index n_pairs = static_cast<Eigen::Index>(pair_i.size());
    if (n_pairs != static_cast<Eigen::Index>(pair_j.size())
        || n_pairs != static_cast<Eigen::Index>(U_pno.size())
        || n_pairs != static_cast<Eigen::Index>(T2_pno.size())) {
        throw std::invalid_argument(
            "dlpno_project_tno_amplitudes: pair_i, pair_j, U_pno, T2_pno "
            "must have the same length");
    }

    t1_out = Mat::Zero(n_act, n_T);
    T2_out = Mat::Zero(n_act * n_act, n_T * n_T);

    // ---- diagonal singles projection ----
    const Eigen::Index n_t1 = static_cast<Eigen::Index>(t1_idx.size());
    #pragma omp parallel for schedule(static)
    for (Eigen::Index k = 0; k < n_t1; ++k) {
        const Eigen::Index m = static_cast<Eigen::Index>(t1_idx[static_cast<std::size_t>(k)]);
        if (m < 0 || m >= n_act) continue;
        const Eigen::VectorXd& t1_k = t1_vec[static_cast<std::size_t>(k)];
        if (t1_k.size() != nv) continue;
        Eigen::VectorXd proj = V_T.transpose() * t1_k;
        for (Eigen::Index a = 0; a < n_T; ++a)
            t1_out(m, a) = proj(a);
    }

    // ---- pair doubles projection ----
    #pragma omp parallel for schedule(dynamic)
    for (Eigen::Index k = 0; k < n_pairs; ++k) {
        const Eigen::Index m = static_cast<Eigen::Index>(pair_i[static_cast<std::size_t>(k)]);
        const Eigen::Index n = static_cast<Eigen::Index>(pair_j[static_cast<std::size_t>(k)]);
        if (m < 0 || m >= n_act || n < 0 || n >= n_act) continue;

        const Mat& U_k = U_pno[static_cast<std::size_t>(k)];
        const Mat& T2_k = T2_pno[static_cast<std::size_t>(k)];
        if (U_k.rows() != nv || T2_k.rows() != U_k.cols()
            || T2_k.cols() != U_k.cols()) continue;

        // S = V_T^T @ U_k  (n_T x n_pno_k)
        Mat S = V_T.transpose() * U_k;
        // T2_TNO = S @ T2_k @ S^T  (n_T x n_T)
        Mat T2_proj = S * T2_k * S.transpose();

        // Write into the flat (n_act*n_act, n_T*n_T) output at block (m,n).
        const Eigen::Index row0 = m * n_act + n;
        const Eigen::Index col0 = 0;
        for (Eigen::Index a = 0; a < n_T; ++a)
            for (Eigen::Index b = 0; b < n_T; ++b)
                T2_out(row0, col0 + a * n_T + b) = T2_proj(a, b);
    }
}

// =========================================================================
// Public entry point: dlpno_build_tno_density
// =========================================================================
Mat dlpno_build_tno_density(
    const Mat& V_T,
    const std::vector<int>& distinct,
    const std::vector<int>& pair_i,
    const std::vector<int>& pair_j,
    const std::vector<Mat>& U_pno,
    const std::vector<Mat>& T2_pno,
    double tcut_tno) {

    const Eigen::Index n_T = V_T.cols();
    if (tcut_tno <= 0.0 || n_T <= 1) return V_T;

    const Eigen::Index n_dist = static_cast<Eigen::Index>(distinct.size());
    const Eigen::Index n_pairs = static_cast<Eigen::Index>(pair_i.size());

    // Build the TNO pair-amplitude density D (n_T x n_T).
    Mat D = Mat::Zero(n_T, n_T);
    for (Eigen::Index ia = 0; ia < n_dist; ++ia) {
        const int p = distinct[static_cast<std::size_t>(ia)];
        for (Eigen::Index jb = ia; jb < n_dist; ++jb) {
            const int q = distinct[static_cast<std::size_t>(jb)];
            // Find the pair (p,q) or (q,p) in the pair lists.
            Eigen::Index pk = -1;
            for (Eigen::Index k = 0; k < n_pairs; ++k) {
                const int pi = pair_i[static_cast<std::size_t>(k)];
                const int pj = pair_j[static_cast<std::size_t>(k)];
                if ((pi == p && pj == q) || (pi == q && pj == p)) {
                    pk = k; break;
                }
            }
            if (pk < 0) continue;

            const Mat& U_k = U_pno[static_cast<std::size_t>(pk)];
            const Mat& T2_k = T2_pno[static_cast<std::size_t>(pk)];
            // Spq = V_T^T @ U_k  (n_T x n_pno)
            Mat Spq = V_T.transpose() * U_k;
            // Tpq = Spq @ T2_k @ Spq^T  (n_T x n_T)
            Mat Tpq = Spq * T2_k * Spq.transpose();
            D += Tpq * Tpq.transpose() + Tpq.transpose() * Tpq;
        }
    }

    // Symmetrise and diagonalise.
    D = 0.5 * (D + D.transpose());
    Eigen::SelfAdjointEigenSolver<Mat> eigh(D);
    if (eigh.info() != Eigen::Success) return V_T;

    // Keep eigenvectors whose occupation > tcut_tno.
    const Eigen::VectorXd& occ = eigh.eigenvalues();
    std::vector<Eigen::Index> keep_idx;
    for (Eigen::Index a = 0; a < n_T; ++a)
        if (occ(a) > tcut_tno) keep_idx.push_back(a);
    if (keep_idx.empty()) {
        // Keep at least the dominant one.
        Eigen::Index dom = 0;
        double dom_val = occ(0);
        for (Eigen::Index a = 1; a < n_T; ++a)
            if (occ(a) > dom_val) { dom = a; dom_val = occ(a); }
        keep_idx.push_back(dom);
    }

    const Eigen::Index n_keep = static_cast<Eigen::Index>(keep_idx.size());
    Mat V_new(V_T.rows(), n_keep);
    for (Eigen::Index a = 0; a < n_keep; ++a)
        V_new.col(a) = V_T * eigh.eigenvectors().col(keep_idx[static_cast<std::size_t>(a)]);
    return V_new;
}

// =========================================================================
// Public entry point: run_ccsd
// =========================================================================
// Shared closed-shell CCSD(T) core.  Both run_ccsd (canonical RHF
// reference) and run_ccsd_from_mos (explicit MO coefficients + AO Fock,
// e.g. FNO-truncated semicanonical virtuals) funnel here.  The orbital
// energies are supplied by the caller (canonical RHF eigenvalues, or the
// diagonal of C^T F C for semicanonical inputs); they drive the MP2 guess,
// the Jacobi preconditioner, and the canonical-orbital (T) denominators.
static CCSDResult run_ccsd_core(const Molecule& mol,
                                const BasisSet& basis,
                                const Mat& C_all,
                                const Eigen::VectorXd& eps_full,
                                const Mat& F_ao,
                                double e_hf,
                                int ecp_total_ncore,
                                const CCSDOptions& opts) {
    const auto n_orb = static_cast<Eigen::Index>(C_all.cols());
    const int n_elec = effective_electron_count(mol, ecp_total_ncore, "run_ccsd");
    const int n_occ_full = n_elec / 2;
    const int n_frozen = resolve_native_frozen_core(
        mol, opts.n_frozen_core, ecp_total_ncore, "run_ccsd");
    if (n_frozen >= n_occ_full)
        throw std::runtime_error(
            "run_ccsd: n_frozen_core >= n_occ, all occupied orbitals frozen");

    const Eigen::Index no = n_occ_full - n_frozen;
    const Eigen::Index nv = n_orb - n_occ_full;
    if (nv < 1)
        throw std::runtime_error("run_ccsd: no virtual orbitals");
    if (opts.compute_triples) {
        // Validate mode/control spelling before the expensive integral build.
        (void)normalise_triples_memory_mode(opts.triples_memory_mode);
        if (opts.triples_tile_size < 0)
            throw std::invalid_argument("run_ccsd: triples_tile_size must be >= 0");
        if (opts.triples_max_threads < 0)
            throw std::invalid_argument("run_ccsd: triples_max_threads must be >= 0");
    }

    // Correlated MO window: frozen-core orbitals are excluded from the
    // amplitude space; they enter only through the converged AO Fock.
    Mat C_occ = C_all.middleCols(n_frozen, no);
    Mat C_vir = C_all.rightCols(nv);

    const Eigen::VectorXd eps_occ = eps_full.segment(n_frozen, no);
    const Eigen::VectorXd eps_vir = eps_full.tail(nv);

    // MO-basis Fock blocks over the correlated window
    Mat f_mo = build_mo_fock(C_all, F_ao);
    Mat f_oo(no, no), f_vv(nv, nv), f_ov(no, nv);
    for (Eigen::Index i = 0; i < no; ++i)
        for (Eigen::Index j = 0; j < no; ++j)
            f_oo(i, j) = f_mo(n_frozen + i, n_frozen + j);
    for (Eigen::Index a = 0; a < nv; ++a)
        for (Eigen::Index b = 0; b < nv; ++b)
            f_vv(a, b) = f_mo(n_occ_full + a, n_occ_full + b);
    for (Eigen::Index i = 0; i < no; ++i)
        for (Eigen::Index a = 0; a < nv; ++a)
            f_ov(i, a) = f_mo(n_frozen + i, n_occ_full + a);

    if (opts.density_fit && opts.aux_basis.empty())
        throw std::invalid_argument("run_ccsd: density_fit requires aux_basis");

    const CCVariant variant = parse_cc_variant(opts.cc_variant);
    if (variant != CCVariant::CCSD && variant != CCVariant::QCISD &&
        variant != CCVariant::BCCD && opts.compute_triples)
        throw std::invalid_argument(
            "run_ccsd: the (T) correction is defined on converged CCSD "
            "or caller-supplied Brueckner CCD/QCISD amplitudes; set "
            "compute_triples=false for cc_variant='" + opts.cc_variant + "'");
    if (variant != CCVariant::CCSD && opts.compute_triples &&
        opts.triples_variant == "[t]")
        throw std::invalid_argument(
            "run_ccsd: triples_variant='[t]' is defined for CCSD only; "
            "QCISD(T) and BCCD(T) use the standard Raghavachari "
            "correction");
    const bool with_singles = variant_has_singles(variant);
    const bool cc2 = variant == CCVariant::CC2;
    const bool linear = variant_is_linear(variant);
    const bool qci = variant == CCVariant::QCISD;
    // Linear variants: CI-like linear wave operator, so the energy
    // functional drops the quadratic t1*t1 tau term.  QCISD is also
    // CI-like in its energy expression, while retaining selected quadratic
    // terms in the amplitude equations.
    const bool tau_energy = !(linear || qci);

    // Free the construction scratch (DF object + MO B-tensor blocks, or the
    // AO/MO four-index tensors on the canonical path) as soon as the dense
    // integral set V is built: the CCSD iteration + (T) below hold V (incl.
    // vv_vv ~ O(nv^4)), the memory wall.  On the DF path, scoping returns
    // n_aux*(nbf^2 + no*nv + nv^2 + no^2)*8 bytes before the long solve.
    IntegralBlocks V = [&] {
        if (opts.density_fit) {
            const BasisSet aux(mol, opts.aux_basis);
            DensityFitting df_obj(basis, aux);
            DFBlocks df = build_df_blocks(df_obj, C_occ, C_vir);
            return build_integral_blocks(std::move(df));
        }
        // Canonical (non-DF) route: exact four-index MO integrals.
        return build_integral_blocks_canonical(basis, C_occ, C_vir);
    }();
    // These correlated-window/Fock construction matrices are not inputs to
    // either the CCSD iteration or triples phase.
    release_matrix(C_occ);
    release_matrix(C_vir);
    release_matrix(f_mo);

    // Amplitudes; MP2 initial guess t_ij^ab = (ia|jb) / D_ij^ab
    Mat T1 = Mat::Zero(no, nv);
    Mat T2_flat = Mat::Zero(no * no, nv * nv);
    for (Eigen::Index i = 0; i < no; ++i)
        for (Eigen::Index j = 0; j < no; ++j)
            for (Eigen::Index a = 0; a < nv; ++a)
                for (Eigen::Index b = 0; b < nv; ++b)
                    T2_flat(i * no + j, a * nv + b) =
                        V.ov_ov(i * nv + a, j * nv + b)
                        / (eps_occ(i) + eps_occ(j)
                           - eps_vir(a) - eps_vir(b));

    CCSDResult result;
    result.e_hf = e_hf;

    // Keep every iteration-only allocation in this scope.  In particular the
    // DIIS amplitude/residual histories and R1/R2 must be destroyed before a
    // requested-budget triples plan is admitted.
    {
      CCSD_DIIS diis(opts.diis_subspace_size);
      Mat R1 = Mat::Zero(no, nv);
      Mat R2_flat = Mat::Zero(no * no, nv * nv);
      double e_cc_prev =
          compute_energy(T1, T2_flat, f_ov, V.ov_ov, no, nv, tau_energy);

      // Constant term of the linearized residual, R(T = 0): evaluated once
      // through the canonical residual code (R1_0 = f_ov, R2_0 = (ia|jb)).
      Mat R1_0, R2_0;
      if (linear) {
          R1_0 = Mat::Zero(no, nv);
          R2_0 = Mat::Zero(no * no, nv * nv);
          const Mat z1 = Mat::Zero(no, nv);
          const Mat z2 = Mat::Zero(no * no, nv * nv);
          compute_residuals(z1, z2, V, f_oo, f_vv, f_ov, R1_0, R2_0);
      }

      for (int iter = 0; iter < opts.max_iter; ++iter) {
        if (cc2)
            compute_cc2_residuals(T1, T2_flat, V, f_oo, f_vv, f_ov,
                                  eps_occ, eps_vir, R1, R2_flat);
        else if (qci)
            compute_qcisd_residuals(T1, T2_flat, V, f_oo, f_vv, f_ov,
                                    R1, R2_flat);
        else if (linear)
            compute_linear_residuals(T1, T2_flat, V, f_oo, f_vv, f_ov,
                                     R1_0, R2_0, R1, R2_flat);
        else
            compute_residuals(T1, T2_flat, V, f_oo, f_vv, f_ov, R1, R2_flat);

        if (variant == CCVariant::CEPA1 || variant == CCVariant::CEPA2 ||
            variant == CCVariant::CEPA3) {
            const Mat eps_pair =
                compute_pair_energies(T2_flat, V.ov_ov, no, nv);
            apply_cepa_shifts(cepa_shifts(variant, eps_pair), T1, T2_flat,
                              R1, R2_flat);
        }
        // Doubles-only variants: pin T1 at zero by zeroing its residual
        // (the Jacobi update and DIIS then never move it).
        if (!with_singles) R1.setZero();

        const double r1_norm = R1.norm();
        const double r2_norm = R2_flat.norm();
        if (!std::isfinite(r1_norm) || !std::isfinite(r2_norm)) break;

        // Jacobi step, then DIIS over the post-step amplitudes
        update_amplitudes(T1, T2_flat, R1, R2_flat, eps_occ, eps_vir, 1.0);
        if (opts.diis_subspace_size > 0) {
            Mat T1_diis, T2_diis;
            diis.extrapolate(T1, T2_flat, R1, R2_flat, T1_diis, T2_diis);
            T1 = T1_diis;
            T2_flat = T2_diis;
        }

        const double e_corr =
            compute_energy(T1, T2_flat, f_ov, V.ov_ov, no, nv, tau_energy);
        const double e_ccsd = result.e_hf + e_corr;
        const double delta_e = e_corr - e_cc_prev;
        result.cc_trace.push_back(
            {iter + 1, e_ccsd, delta_e, r1_norm, r2_norm,
             static_cast<int>(diis.subspace_size())});

        const bool conv = std::abs(delta_e) < opts.conv_tol_energy
                          && (r1_norm + r2_norm) < opts.conv_tol_residual;
        e_cc_prev = e_corr;

        if (conv) {
            result.converged = true;
            result.n_iter = iter + 1;
            break;
        }
      }
    }

    const double e_corr =
        compute_energy(T1, T2_flat, f_ov, V.ov_ov, no, nv, tau_energy);
    result.e_ccsd_correlation = e_corr;
    result.e_ccsd = result.e_hf + e_corr;
    result.t1_norm = T1.norm();
    result.t2_norm = T2_flat.norm();
    if (!result.converged) result.n_iter = opts.max_iter;

    if (opts.compute_triples && result.converged) {
        // f_oo/f_vv and the solve-only integral blocks are dead at convergence.
        // The planner models the exact retained strategy set, then the release
        // step makes that set real before any per-thread buffer is allocated.
        release_matrix(f_oo);
        release_matrix(f_vv);
        const TriplesMemoryPlan plan = plan_triples_memory(
            T1, T2_flat, V, f_ov, eps_occ, eps_vir, opts);
        prepare_triples_integrals(V, plan);

        result.triples_memory_mode_used = plan.name;
        result.triples_tile_size_used = plan.tile_size;
        result.triples_threads_used = plan.threads;
        result.triples_workspace_bytes = plan.workspace_bytes;
        result.triples_disk_bytes = plan.disk_bytes;

        TriplesEnergies t3;
        if (plan.strategy == TriplesMemoryStrategy::Fast) {
            t3 = compute_triples(T1, T2_flat, V, f_ov, eps_occ, eps_vir,
                                 plan.threads);
        } else if (plan.strategy == TriplesMemoryStrategy::Blocked) {
            t3 = compute_triples_tiled(
                T1, T2_flat, V, f_ov, eps_occ, eps_vir,
                plan.tile_size, plan.threads);
        } else if (plan.strategy == TriplesMemoryStrategy::Disk) {
            t3 = compute_triples_tiled(
                T1, T2_flat, V, f_ov, eps_occ, eps_vir,
                plan.tile_size, plan.threads, plan.factor_block_rows);
        } else {
            t3 = compute_triples_direct(
                T1, T2_flat, V, f_ov, eps_occ, eps_vir, plan.threads);
        }
        result.e_t4 = t3.fourth_order;
        result.e_t5_st = t3.disconnected_fifth;
        // triples_variant selects which correction lands in e_t:
        //   "(t)"  standard Raghavachari CCSD(T)  = E[T] + E_ST
        //          QCISD(T) (Pople/Head-Gordon/Raghavachari Eq. 3.4)
        //          omits the singles-singles triples part and keeps
        //          E[T] + 2 E_ST on QCISD amplitudes.
        //   "[t]"  bracket correction CCSD[T]     = E[T]
        //          (identically Urban's CCSD+T(CCSD))
        if (opts.triples_variant == "[t]") {
            result.e_t = t3.fourth_order;
        } else if (opts.triples_variant == "(t)" ||
                   opts.triples_variant.empty()) {
            result.e_t = qci
                ? t3.fourth_order + 2.0 * t3.disconnected_fifth
                : t3.fourth_order + t3.disconnected_fifth;
        } else {
            throw std::invalid_argument(
                "run_ccsd: unknown triples_variant '" + opts.triples_variant +
                "'; supported: \"(t)\" (standard CCSD(T)) and \"[t]\" "
                "(CCSD[T] = CCSD+T(CCSD))");
        }
    }

    // Copy the public T1 result only after triples so it is not a duplicate
    // retained amplitude matrix at the bounded triples peak.
    result.t1_amplitudes = T1;

    result.e_ccsd_t = result.e_ccsd + result.e_t;
    result.e_total = result.e_ccsd_t;
    return result;
}

CCSDResult run_ccsd(const Molecule& mol,
                    const BasisSet& basis,
                    const RHFResult& rhf,
                    const CCSDOptions& opts) {
    if (!rhf.converged)
        throw std::runtime_error("run_ccsd: RHF reference is not converged");
    if (effective_electron_count(mol, rhf.ecp_total_ncore, "run_ccsd") % 2 != 0
        || mol.multiplicity() != 1)
        throw std::invalid_argument(
            "run_ccsd: requires closed-shell RHF reference");
    // Canonical RHF: orbital energies are the diagonal of f_mo by
    // construction, so passing rhf.mo_energies preserves the exact prior
    // numerical behaviour.
    return run_ccsd_core(mol, basis, rhf.mo_coeffs, rhf.mo_energies, rhf.fock,
                         rhf.energy, rhf.ecp_total_ncore, opts);
}

CCSDResult run_ccsd_from_mos(const Molecule& mol,
                             const BasisSet& basis,
                             const Mat& C,
                             const Mat& F,
                             double e_hf,
                             int ecp_total_ncore,
                             const CCSDOptions& opts) {
    if (effective_electron_count(mol, ecp_total_ncore, "run_ccsd_from_mos") % 2
            != 0
        || mol.multiplicity() != 1)
        throw std::invalid_argument(
            "run_ccsd_from_mos: requires a closed-shell reference");
    if (C.rows() == 0 || C.cols() == 0)
        throw std::invalid_argument(
            "run_ccsd_from_mos: MO coefficient matrix must be non-empty");
    if (F.rows() != C.rows() || F.cols() != C.rows())
        throw std::invalid_argument(
            "run_ccsd_from_mos: AO Fock must be (n_ao, n_ao) matching C rows");
    // Orbital energies from diag(C^T F C): canonical for the occupied
    // block, semicanonical for FNO-truncated virtuals (f_vv diagonal so
    // the (T) canonical-orbital denominators stay valid).
    const Eigen::VectorXd eps = build_mo_fock(C, F).diagonal();
    return run_ccsd_core(mol, basis, C, eps, F, e_hf, ecp_total_ncore, opts);
}

// =========================================================================
// Python-bindable spatial CCSD residual: accepts numpy-compatible flat
// layouts (2D views of the 4D integral tensors Python assembles via
// `_blocks` / `_exact_closed_shell_cc_blocks`) and returns the R1 (no,nv)
// and R2_flat (no*no, nv*nv) residuals.  The caller reshapes its 4D numpy
// arrays to 2D before calling — no copy is needed for C-contiguous data
// when using RowMajor Eigen maps.
//
// This is the workhorse behind the A-CCSD(T) and DLPNO-CCSD pilot paths;
// the C++ compute_residuals is ~30x the speed of the Python einsum
// transcription for moderate n_occ/n_vir on an M-series Mac.
// =========================================================================

void spatial_ccsd_residual(
    const Mat& T1, const Mat& T2_flat,
    const Mat& f_oo_rw, const Mat& f_vv_rw, const Mat& f_ov_rw,
    const Mat& ovov_flat, const Mat& oooo_flat,
    const Mat& ooov_flat, const Mat& oovv_flat,
    const Mat& ovvv_flat, const Mat& vvvv_flat,
    Mat& R1, Mat& R2_flat)
{
    const Eigen::Index no = T1.rows();
    const Eigen::Index nv = T1.cols();
    R1 = Mat::Zero(no, nv);
    R2_flat = Mat::Zero(no * no, nv * nv);
    
    IntegralBlocks V;
    V.ov_ov = ovov_flat;
    V.oo_oo = oooo_flat;
    V.oo_ov = ooov_flat;
    V.oo_vv = oovv_flat;
    V.ov_vv = ovvv_flat;
    V.vv_vv = vvvv_flat;
    compute_residuals(T1, T2_flat, V, f_oo_rw, f_vv_rw, f_ov_rw,
                      R1, R2_flat);
}

// =========================================================================
// Full spatial CCSD solver from pre-built integrals.
// Returns T1, T2_flat, E_corr, converged, n_iter.
// =========================================================================

SpatialCCSDResult spatial_ccsd_solve(
    const Mat& f_oo, const Mat& f_vv, const Mat& f_ov,
    const Mat& ovov_flat, const Mat& oooo_flat,
    const Mat& ooov_flat, const Mat& oovv_flat,
    const Mat& ovvv_flat, const Mat& vvvv_flat,
    double e_hf,
    int max_iter,
    double conv_tol_energy,
    double conv_tol_residual,
    int diis_size)
{
    const Eigen::Index no = f_oo.rows();
    const Eigen::Index nv = f_vv.rows();

    // Integral blocks
    IntegralBlocks V;
    V.ov_ov = ovov_flat;
    V.oo_oo = oooo_flat;
    V.oo_ov = ooov_flat;
    V.oo_vv = oovv_flat;
    V.ov_vv = ovvv_flat;
    V.vv_vv = vvvv_flat;

    // Orbital energy denominators
    Eigen::VectorXd eps_o = f_oo.diagonal();
    Eigen::VectorXd eps_v = f_vv.diagonal();

    // Initial MP2 guess
    Mat T1 = f_ov;
    #pragma omp parallel for collapse(2) schedule(static)
    for (Eigen::Index i = 0; i < no; ++i)
        for (Eigen::Index a = 0; a < nv; ++a)
            T1(i, a) /= (eps_o(i) - eps_v(a));

    Mat T2_flat = Mat::Zero(no * no, nv * nv);
    #pragma omp parallel for collapse(4) schedule(static)
    for (Eigen::Index i = 0; i < no; ++i)
        for (Eigen::Index j = 0; j < no; ++j)
            for (Eigen::Index a = 0; a < nv; ++a)
                for (Eigen::Index b = 0; b < nv; ++b)
                    T2_flat(i * no + j, a * nv + b) =
                        V.ov_ov(i * nv + a, j * nv + b)
                        / (eps_o(i) + eps_o(j) - eps_v(a) - eps_v(b));

    // DIIS history
    std::deque<Mat> amp_hist, res_hist;

    double e_prev = compute_energy(T1, T2_flat, f_ov, V.ov_ov, no, nv, true);

    SpatialCCSDResult result;
    for (int iter = 0; iter < max_iter; ++iter) {
        Mat R1 = Mat::Zero(no, nv);
        Mat R2_flat = Mat::Zero(no * no, nv * nv);
        compute_residuals(T1, T2_flat, V, f_oo, f_vv, f_ov, R1, R2_flat);

        // Jacobi update
        Mat T1_new = T1;
        Mat T2_new = T2_flat;
        #pragma omp parallel for collapse(2) schedule(static)
        for (Eigen::Index i = 0; i < no; ++i)
            for (Eigen::Index a = 0; a < nv; ++a)
                T1_new(i, a) += R1(i, a) / (eps_o(i) - eps_v(a));
        #pragma omp parallel for collapse(4) schedule(static)
        for (Eigen::Index i = 0; i < no; ++i)
            for (Eigen::Index j = 0; j < no; ++j)
                for (Eigen::Index a = 0; a < nv; ++a)
                    for (Eigen::Index b = 0; b < nv; ++b)
                        T2_new(i * no + j, a * nv + b) +=
                            R2_flat(i * no + j, a * nv + b)
                            / (eps_o(i) + eps_o(j) - eps_v(a) - eps_v(b));

        // DIIS
        Mat T1_diis = T1_new;
        Mat T2_diis = T2_new;
        if (diis_size > 0) {
            Mat amp_flat(no * nv + no * no * nv * nv, 1);
            amp_flat.topRows(no * nv) =
                Eigen::Map<const Eigen::VectorXd>(T1_new.data(), T1_new.size());
            amp_flat.bottomRows(no * no * nv * nv) =
                Eigen::Map<const Eigen::VectorXd>(T2_new.data(), T2_new.size());
            amp_hist.push_back(amp_flat);

            Mat res_flat(no * nv + no * no * nv * nv, 1);
            res_flat.topRows(no * nv) =
                Eigen::Map<const Eigen::VectorXd>(R1.data(), R1.size());
            res_flat.bottomRows(no * no * nv * nv) =
                Eigen::Map<const Eigen::VectorXd>(R2_flat.data(), R2_flat.size());
            res_hist.push_back(res_flat);

            while (static_cast<int>(amp_hist.size()) > diis_size) {
                amp_hist.pop_front();
                res_hist.pop_front();
            }

            const int nh = static_cast<int>(amp_hist.size());
            if (nh >= 2) {
                Mat B = Mat::Zero(nh + 1, nh + 1);
                for (int a = 0; a < nh; ++a)
                    for (int b = 0; b < nh; ++b)
                        B(a, b) = res_hist[a].reshaped().dot(res_hist[b].reshaped());
                B.row(nh).setConstant(-1.0);
                B.col(nh).setConstant(-1.0);
                B(nh, nh) = 0.0;
                Eigen::VectorXd rhs = Eigen::VectorXd::Zero(nh + 1);
                rhs(nh) = -1.0;
                Eigen::VectorXd c = B.fullPivLu().solve(rhs);
                if (c.size() == nh + 1) {
                    Eigen::VectorXd combined =
                        Eigen::VectorXd::Zero(no * nv + no * no * nv * nv);
                    for (int k = 0; k < nh; ++k)
                        combined += c(k) * amp_hist[k];
                    T1_diis = Eigen::Map<const Mat>(
                        combined.topRows(no * nv).data(), no, nv);
                    T2_diis = Eigen::Map<const Mat>(
                        combined.bottomRows(no * no * nv * nv).data(),
                        no * no, nv * nv);
                }
            }
        }

        T1 = T1_diis;
        T2_flat = T2_diis;

        const double e_corr = compute_energy(
            T1, T2_flat, f_ov, V.ov_ov, no, nv, true);
        const double delta_e = e_corr - e_prev;
        const double r_norm = R1.norm() + R2_flat.norm();

        if (std::abs(delta_e) < conv_tol_energy && r_norm < conv_tol_residual) {
            result.converged = true;
            result.n_iter = iter + 1;
            result.T1 = T1;
            result.T2_flat = T2_flat;
            result.e_corr = e_corr;
            return result;
        }
        e_prev = e_corr;
    }

    result.n_iter = max_iter;
    result.T1 = T1;
    result.T2_flat = T2_flat;
    result.e_corr = compute_energy(
        T1, T2_flat, f_ov, V.ov_ov, no, nv, true);
    return result;
}

}  // namespace vibeqc
