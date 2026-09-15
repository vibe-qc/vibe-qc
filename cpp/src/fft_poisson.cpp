// Reciprocal-space Poisson solver via FFTW3.
//
// Algorithm (standard plane-wave trick):
//
//   1. Forward real-to-complex FFT of ρ(r) → ρ̃(G).
//   2. Multiply by the reciprocal-space kernel:
//        Ṽ(G) = K̃(G) · ρ̃(G)      for G ≠ 0
//        Ṽ(0) = 0                   (charged cells would blow up here;
//                                     we pin the gauge so returned V
//                                     has zero mean.)
//      K̃(G) = (4π / |G|²) · e^(-|G|² / (4ω²))   for erf-screened
//      K̃(G) =  4π / |G|²                        for unscreened
//   3. Backward complex-to-real FFT of Ṽ(G) → V(r).
//   4. Normalize by N_grid so FFTW's unnormalised back-and-forth
//      comes out to the right absolute scale.
//
// The G vectors for a uniform grid on a general cell with lattice
// matrix A = [a_1 a_2 a_3] and grid dimensions (n_x, n_y, n_z) are
//
//    G(i_x, i_y, i_z) = k_x b_1 + k_y b_2 + k_z b_3
//
// where k_α ∈ {0, 1, ..., n_α/2 - 1, -n_α/2, ..., -1} (standard FFT
// wrap-around convention) and B = [b_1 b_2 b_3] = 2π A^{-T}. For the
// real-to-complex (R2C) transform,
// FFTW stores only the n_z/2 + 1 non-redundant k_z values along the
// last axis; the rest are reconstructed via Hermitian conjugation.
//
// Thread-safety: FFTW plan creation is *not* thread-safe. All plan
// creation runs inside a ``std::mutex``-guarded section. The plan
// *execution* via ``fftw_execute`` is thread-safe as long as the same
// plan isn't used by two threads concurrently — our single-threaded
// inner loop is fine.

#include "vibeqc/fft_poisson.hpp"

#include <fftw3.h>

#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>

namespace vibeqc {

std::string fftw3_version() {
    // ``fftw_version`` is a ``const char*`` global supplied by FFTW
    // (see <fftw3.h>); resolves at link time to the linked library's
    // version string, e.g. "fftw-3.3.10-sse2-avx". We strip the
    // leading "fftw-" prefix so the banner reads consistent with the
    // other libraries (which all report bare semver-style strings).
    const char* raw = fftw_version;
    if (raw == nullptr) {
        return {};
    }
    std::string v(raw);
    constexpr std::string_view prefix = "fftw-";
    if (v.compare(0, prefix.size(), prefix) == 0) {
        v.erase(0, prefix.size());
    }
    return v;
}

namespace {

// Guard FFTW plan create/destroy calls.
std::mutex& fftw_plan_mutex() {
    static std::mutex m;
    return m;
}

// Wrap-around index convention for a 1D FFT of length n:
//   i < n/2    → i
//   i >= n/2   → i - n
inline int wrap_index(int i, int n) {
    return (i < n / 2) ? i : (i - n);
}

void check_lattice(const Eigen::Matrix3d& L) {
    const double tol = 1e-10;
    for (int i = 0; i < 3; ++i) {
        if (L.col(i).norm() < tol) {
            throw std::invalid_argument(
                "fft_poisson: lattice vectors must be non-degenerate");
        }
    }
    if (std::abs(L.determinant()) < tol) {
        throw std::invalid_argument(
            "fft_poisson: lattice matrix must be non-singular");
    }
}

// Enum tag so we can share the bulk of the solver code between
// kernels. ``omega`` is unused when kind == Unscreened.
enum class Kernel { Unscreened, ErfScreened };

ScalarField3D solve_poisson_generic(
    const ScalarField3D& rho,
    const Eigen::Matrix3d& lattice,
    Kernel kind,
    double omega)
{
    check_lattice(lattice);
    if (rho.nx == 0 || rho.ny == 0 || rho.nz == 0) {
        throw std::invalid_argument(
            "fft_poisson: rho has a zero-size dimension");
    }

    const int nx = static_cast<int>(rho.nx);
    const int ny = static_cast<int>(rho.ny);
    const int nz = static_cast<int>(rho.nz);
    const std::size_t n_real = static_cast<std::size_t>(nx) * ny * nz;
    const std::size_t n_cx   = static_cast<std::size_t>(nx) * ny
                                * (static_cast<std::size_t>(nz / 2 + 1));

    // Reciprocal lattice columns b_i, including the 2π factor. This
    // handles skew cells by evaluating |G|² with the full metric.
    const Eigen::Matrix3d recip = 2.0 * M_PI * lattice.inverse().transpose();

    // Copy input into a fresh aligned buffer so we can hand it to
    // FFTW. The returned field uses a separate buffer for the back-
    // transformed result.
    double* rho_buf = fftw_alloc_real(n_real);
    fftw_complex* freq_buf = fftw_alloc_complex(n_cx);
    double* V_buf = fftw_alloc_real(n_real);
    if (!rho_buf || !freq_buf || !V_buf) {
        if (rho_buf)  fftw_free(rho_buf);
        if (freq_buf) fftw_free(freq_buf);
        if (V_buf)    fftw_free(V_buf);
        throw std::runtime_error("fft_poisson: fftw_alloc returned null");
    }
    std::copy(rho.data.begin(), rho.data.end(), rho_buf);

    fftw_plan plan_fwd, plan_bwd;
    {
        std::lock_guard<std::mutex> guard(fftw_plan_mutex());
        plan_fwd = fftw_plan_dft_r2c_3d(
            nx, ny, nz, rho_buf, freq_buf, FFTW_ESTIMATE);
        plan_bwd = fftw_plan_dft_c2r_3d(
            nx, ny, nz, freq_buf, V_buf, FFTW_ESTIMATE);
    }
    if (!plan_fwd || !plan_bwd) {
        fftw_free(rho_buf);
        fftw_free(freq_buf);
        fftw_free(V_buf);
        throw std::runtime_error("fft_poisson: fftw_plan_* failed");
    }

    fftw_execute(plan_fwd);

    // Apply the reciprocal-space kernel.
    const double four_pi = 4.0 * M_PI;
    const double inv_4w2 = (kind == Kernel::ErfScreened)
        ? 0.25 / (omega * omega) : 0.0;
    const int nz_half = nz / 2 + 1;

    for (int kx = 0; kx < nx; ++kx) {
        const int ix_wrap = wrap_index(kx, nx);
        for (int ky = 0; ky < ny; ++ky) {
            const int iy_wrap = wrap_index(ky, ny);
            for (int kz = 0; kz < nz_half; ++kz) {
                // For r2c, kz index already runs 0 .. nz/2 without
                // wrap-around — these are non-negative frequencies.
                const Eigen::Vector3d G =
                    static_cast<double>(ix_wrap) * recip.col(0)
                    + static_cast<double>(iy_wrap) * recip.col(1)
                    + static_cast<double>(kz) * recip.col(2);
                const double G2 = G.squaredNorm();

                double K;
                if (G2 == 0.0) {
                    K = 0.0;   // pin Ṽ(G=0) = 0 (neutral-cell gauge)
                } else {
                    double k_tail = 1.0;
                    if (kind == Kernel::ErfScreened) {
                        k_tail = std::exp(-G2 * inv_4w2);
                    }
                    K = four_pi / G2 * k_tail;
                }

                const std::size_t idx =
                    (static_cast<std::size_t>(kx) * ny + ky)
                    * static_cast<std::size_t>(nz_half) + kz;
                freq_buf[idx][0] *= K;
                freq_buf[idx][1] *= K;
            }
        }
    }

    fftw_execute(plan_bwd);

    // FFTW's paired r2c / c2r leaves the result unnormalised by
    // n_real. Divide at the end.
    const double inv_n = 1.0 / static_cast<double>(n_real);
    ScalarField3D V;
    V.resize(rho.nx, rho.ny, rho.nz);
    for (std::size_t g = 0; g < n_real; ++g) {
        V.data[g] = V_buf[g] * inv_n;
    }

    {
        std::lock_guard<std::mutex> guard(fftw_plan_mutex());
        fftw_destroy_plan(plan_fwd);
        fftw_destroy_plan(plan_bwd);
    }
    fftw_free(rho_buf);
    fftw_free(freq_buf);
    fftw_free(V_buf);

    return V;
}

}  // namespace

ScalarField3D solve_poisson_erf_screened(
    const ScalarField3D& rho,
    const Eigen::Matrix3d& lattice_bohr,
    double omega)
{
    if (!(omega > 0.0)) {
        throw std::invalid_argument(
            "solve_poisson_erf_screened: omega must be > 0");
    }
    return solve_poisson_generic(rho, lattice_bohr,
                                  Kernel::ErfScreened, omega);
}

ScalarField3D solve_poisson_coulomb(
    const ScalarField3D& rho,
    const Eigen::Matrix3d& lattice_bohr)
{
    return solve_poisson_generic(rho, lattice_bohr,
                                  Kernel::Unscreened, 0.0);
}

double cell_volume(const Eigen::Matrix3d& lattice_bohr) {
    return std::abs(lattice_bohr.determinant());
}

double hartree_energy(const ScalarField3D& rho,
                       const ScalarField3D& V,
                       double cell_volume_bohr3) {
    if (rho.data.size() != V.data.size()) {
        throw std::invalid_argument(
            "hartree_energy: rho and V have different grid sizes");
    }
    double s = 0.0;
    for (std::size_t g = 0; g < rho.data.size(); ++g) {
        s += rho.data[g] * V.data[g];
    }
    const double dV = cell_volume_bohr3 / static_cast<double>(rho.data.size());
    return 0.5 * s * dV;
}

}  // namespace vibeqc
