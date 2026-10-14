#include "vibeqc/periodic_gdf_short_range.hpp"
#include "vibeqc/manta_periodic_fitting.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/manta_shells.hpp"
#include "vibeqc/detail/sha256.hpp"
#include <cstring>
#include "vibeqc/init.hpp"
#include "vibeqc/lattice_pair_cells.hpp"
#include "vibeqc/thread_pool.hpp"
#include "vibeqc/aopair_ft.hpp"
#include "vibeqc/periodic_auxiliary_fourier.hpp"
#include "aopair_ft_internal.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <libint2/engine.h>

namespace vibeqc {
namespace {
using Complex = std::complex<double>;
constexpr std::size_t primitive_block_size = 4;

// One jellium convention for value and derivative sources at every k.
// The primed LR sum omits p=0; its finite SR counterpart is removed once.
// The exchange Madelung correction belongs to the SCF assembly, not here.
struct RangeSeparatedCoulombKernel {
    double omega_squared, reciprocal_prefactor, short_range_zero_mode;
    bool zero_transfer;

    RangeSeparatedCoulombKernel(const PeriodicSystem& system,
                               const Eigen::Vector3d& q, double omega)
        : omega_squared(omega * omega) {
        const double pi = std::acos(-1.0);
        const double volume = std::abs(system.lattice.determinant());
        reciprocal_prefactor = 4.0 * pi / volume;
        short_range_zero_mode = pi / (volume * omega_squared);
        const Eigen::Vector3d fractional = system.lattice.transpose() * q / (2.0 * pi);
        zero_transfer = (fractional - fractional.array().round().matrix()).norm() < 1e-12;
    }

    static bool singular(double squared_momentum) {
        return squared_momentum <= 1e-24;
    }

    double long_range(double squared_momentum) const {
        return bare(squared_momentum)
            * std::exp(-squared_momentum / (4.0 * omega_squared));
    }

    double bare(double squared_momentum) const {
        return singular(squared_momentum) ? 0.0
            : reciprocal_prefactor / squared_momentum;
    }
};

std::size_t product(std::size_t a, std::size_t b) {
    if (a && b > std::numeric_limits<std::size_t>::max() / a)
        throw std::overflow_error("GDF SR extent overflow");
    return a * b;
}

std::size_t sum(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::overflow_error("GDF SR workspace extent overflow");
    return a + b;
}

std::vector<Eigen::MatrixXd> make_worker_gradients(int workers, std::size_t atoms) {
    // Called after admission of one matrix and header per worker. The
    // vector fill constructor would also allocate a temporary atom-gradient
    // matrix, exceeding that reservation while copying it into the workers.
    std::vector<Eigen::MatrixXd> gradients;
    gradients.reserve(workers);
    for (int worker = 0; worker < workers; ++worker) {
        gradients.emplace_back(atoms, 3);
        gradients.back().setZero();
    }
    return gradients;
}

struct SRWorkerMemory {
    std::size_t prototype, worker;
};

SRWorkerMemory sr_worker_memory(const libint2::BasisSet& a, const libint2::BasisSet& b,
                               int rank, int derivative_order,
                               std::size_t worker_extra_bytes = 0) {
    const auto original_np = std::max(a.max_nprim(), b.max_nprim());
    const auto np = std::min(original_np, primitive_block_size);
    const int l = std::max(a.max_l(), b.max_l());
    if (np == 0 || l < 0)
        throw std::invalid_argument("GDF SR requires nonempty Gaussian bases");
    // Libint's xs_xx fitting center and paired AO centers can have
    // different generated limits. Basis a is auxiliary, basis b is AO;
    // xs_xs uses the fitting limit for both centers.
    const int fitting_limit = derivative_order == 0
        ? (rank == 2 ? LIBINT2_MAX_AM_2eri : LIBINT2_MAX_AM_3eri)
        : (rank == 2 ? LIBINT2_MAX_AM_2eri1 : LIBINT2_MAX_AM_3eri1);
    int paired_limit = fitting_limit;
    if (rank == 3) {
#if defined(LIBINT2_CENTER_DEPENDENT_MAX_AM_3eri) && LIBINT2_CENTER_DEPENDENT_MAX_AM_3eri
        if (derivative_order == 0) paired_limit = LIBINT2_MAX_AM_default;
#endif
#if defined(LIBINT2_CENTER_DEPENDENT_MAX_AM_3eri1) && LIBINT2_CENTER_DEPENDENT_MAX_AM_3eri1
        if (derivative_order == 1) paired_limit = LIBINT2_MAX_AM_default1;
#endif
    }
    if (a.max_l() > fitting_limit || b.max_l() > paired_limit)
        throw std::invalid_argument("GDF SR angular momentum exceeds the linked libint kernels");
    const auto stack = derivative_order == 0
        ? (rank == 2 ? libint2_need_memory_2eri(l) : libint2_need_memory_3eri(l))
        : (rank == 2 ? libint2_need_memory_2eri1(l) : libint2_need_memory_3eri1(l));
    if (!stack)
        throw std::invalid_argument("GDF SR angular momentum exceeds the linked libint kernels");
    const auto nc = product(std::size_t(l + 1), std::size_t(l + 2)) / 2;
    std::size_t primitive_count = 1, shell_values = 1;
    for (int i = 0; i < rank; ++i) {
        primitive_count = product(primitive_count, np);
        shell_values = product(shell_values, nc);
    }
    // libint Engine::initialize/reset_scratch: primitive data, two shell
    // pair lists, recurrence stack and up to two Cartesian target buffers.
    // Include a fixed allowance for parameter/evaluator allocations, but do
    // not describe this estimate as a whole-process RSS hard limit.
    std::size_t engine_bytes = sum(sizeof(libint2::Engine), 64U * 1024U);
    engine_bytes = sum(engine_bytes, product(primitive_count, sizeof(Libint_t)));
    engine_bytes = sum(engine_bytes, product(product(2, product(np, np)),
        sizeof(libint2::ShellPair::PrimPairData)));
    const auto targets = derivative_order == 0 ? 1U : std::size_t(3 * rank);
    engine_bytes = sum(engine_bytes,
        product(sum(stack, product(product(2, targets), shell_values)), sizeof(double)));
    const auto shell_copies = product(6, sum(sizeof(libint2::Shell), product(32, original_np)));
    const auto per_worker = sum(worker_extra_bytes, sum(engine_bytes,
        sum(product(shell_values, sizeof(Complex)), shell_copies)));
    return {engine_bytes, per_worker};
}

int sr_workers(const libint2::BasisSet& a, const libint2::BasisSet& b,
               int rank, std::size_t tasks, std::size_t workspace_cap,
               int derivative_order = 0, std::size_t worker_extra_bytes = 0) {
    const auto memory = sr_worker_memory(a, b, rank, derivative_order, worker_extra_bytes);
    const auto engine_bytes = memory.prototype, per_worker = memory.worker;
    // The prototype remains alive while its worker copies execute.
    if (engine_bytes > workspace_cap || per_worker > workspace_cap - engine_bytes)
        throw std::length_error("GDF SR engine workspace byte cap exceeded");
    const auto cap = std::min<std::size_t>(
        (workspace_cap - engine_bytes) / per_worker, std::numeric_limits<int>::max());
    return omp_workers_for(tasks, static_cast<int>(cap));
}

void validate(const PeriodicSystem& system, const Eigen::Vector3d& q,
              double omega, double cutoff) {
    if (system.dim != 3 || !system.lattice.allFinite()
        || !system.lattice.inverse().allFinite()
        || !q.allFinite() || !std::isfinite(omega) || omega <= 0.0
        || !std::isfinite(cutoff) || cutoff <= 0.0)
        throw std::invalid_argument("GDF SR requires a nonsingular 3D cell, finite q and positive omega/cutoffs");
}

// Stream a conservative integer box; never allocate a lattice-point array.
// |A*n-center| <= radius implies |n_i-(A^-1 center)_i| <= radius*|row_i(A^-1)|.
// Padding is used only in enumeration, then the physical separation filters
// below choose the terms. This preserves atom-image covariance on skew cells.
struct ImageBox {
    Eigen::Matrix3d lattice, inverse;
    Eigen::Vector3d reach;
    std::size_t cap;
    std::size_t count_bound = 1;
    ImageBox(const PeriodicSystem& system, double radius, std::size_t limit)
        : lattice(system.lattice), inverse(lattice.inverse()), cap(limit) {
        for (int i = 0; i < 3; ++i) reach[i] = radius * inverse.row(i).norm();
        std::size_t count = 1;
        for (int i = 0; i < 3; ++i) {
            const double width = 2.0 * std::ceil(reach[i]) + 5.0;
            if (!std::isfinite(width) || width > 1e8)
                throw std::length_error("GDF SR image candidate cap exceeded");
            const auto extent = static_cast<std::size_t>(width);
            if (!cap || count > cap / extent)
                throw std::length_error("GDF SR image candidate cap exceeded");
            count *= extent;
        }
        if (!cap || count > cap)
            throw std::length_error("GDF SR image candidate cap exceeded");
        count_bound = count;
    }
    template<class F> void each(const Eigen::Vector3d& center, F&& f) const {
        const Eigen::Vector3d frac = inverse * center;
        std::array<long long, 3> lo, hi;
        for (int i = 0; i < 3; ++i) {
            lo[i] = static_cast<long long>(std::floor(frac[i] - reach[i])) - 1;
            hi[i] = static_cast<long long>(std::ceil(frac[i] + reach[i])) + 1;
        }
        for (auto i = lo[0]; i <= hi[0]; ++i)
            for (auto j = lo[1]; j <= hi[1]; ++j)
                for (auto k = lo[2]; k <= hi[2]; ++k)
                    f(lattice * Eigen::Vector3d(double(i), double(j), double(k)));
    }

    template<class F> void each_capsule(
        const Eigen::Vector3d& begin, const Eigen::Vector3d& end, double radius,
        const Eigen::Vector3d& screen_center, double screen_radius, F&& f) const {
        // A capsule projects onto lattice coordinate i as the interval
        // between its endpoints, expanded by radius*|row_i(A^-1)|.
        // The enclosing object's worst-case count remains the screening
        // budget and admission bound. Only enumeration is tightened here.
        const Eigen::Vector3d first = inverse * begin, last = inverse * end;
        const Eigen::Vector3d screened = inverse * screen_center;
        std::array<long long, 3> lo, hi;
        for (int i = 0; i < 3; ++i) {
            const double norm = inverse.row(i).norm();
            double lower = std::min(first[i], last[i]) - radius * norm;
            double upper = std::max(first[i], last[i]) + radius * norm;
            if (std::isfinite(screen_radius)) {
                lower = std::max(lower, screened[i] - screen_radius * norm);
                upper = std::min(upper, screened[i] + screen_radius * norm);
            }
            lo[i] = static_cast<long long>(std::floor(lower)) - 1;
            hi[i] = static_cast<long long>(std::ceil(upper)) + 1;
        }
        for (auto i = lo[0]; i <= hi[0]; ++i)
            for (auto j = lo[1]; j <= hi[1]; ++j)
                for (auto k = lo[2]; k <= hi[2]; ++k)
                    f(lattice * Eigen::Vector3d(double(i), double(j), double(k)));
    }
};

// Absolute shell envelope, including contractions and angular momentum.
// In libint's regular-solid-harmonic convention each angular component is
// bounded by r^l (the spherical-harmonic addition theorem); Cartesian
// monomials obey the same bound. For a <= alpha,
// r^l exp(-alpha r^2) <= [l/(2e(alpha-a))]^(l/2) exp(-a r^2).
// Thus |shell_component(r)| <= exp(log_c-a*r^2), without cancellation or
// the normalized-s approximation of Ye 2021 SI S4. Use a=alpha_min for s,
// alpha_min/2 otherwise. No primitive/image-indexed cache is allocated.
struct ShellEnvelope {
    double a = 0.0, log_c = -std::numeric_limits<double>::infinity();
    explicit ShellEnvelope(const libint2::Shell& shell) {
        const auto amin = *std::min_element(shell.alpha.begin(), shell.alpha.end());
        bool angular = false;
        for (const auto& contraction : shell.contr) angular |= contraction.l != 0;
        a = angular ? amin / 2.0 : amin;
        for (const auto& contraction : shell.contr) {
            double total = -std::numeric_limits<double>::infinity();
            for (std::size_t i = 0; i < shell.alpha.size(); ++i) {
                const double coefficient = std::abs(contraction.coeff[i]);
                if (coefficient == 0.0) continue;
                double term = std::log(coefficient);
                if (contraction.l > 0) {
                    const double half_l = 0.5 * contraction.l;
                    term += half_l * (std::log(half_l / (shell.alpha[i] - a)) - 1.0);
                }
                const double larger = std::max(total, term);
                total = larger + std::log1p(std::exp(std::min(total, term) - larger));
            }
            log_c = std::max(log_c, total);
        }
    }
};

// Positive Gaussian SR interaction, bounded by the largest integrand in
// (2/sqrt(pi))*integral_{eta2}^{eta1} exp(-t^2 R^2) dt. The Gaussian
// charges contribute (pi/a)^(3/2)*(pi/b)^(3/2). The result bounds EVERY
// component, rather than assuming a Schwarz factor has Gaussian decay.
struct GaussianSRBound {
    double log_prefactor, decay;
    GaussianSRBound(double a, double b, double log_c, double omega) {
        const double eta1_squared = 1.0 / (1.0 / a + 1.0 / b);
        const double ratio = eta1_squared / (omega * omega);
        const double root = std::sqrt(1.0 + ratio);
        const double difference = std::sqrt(eta1_squared) * ratio / (root * (root + 1.0));
        decay = 1.0 / (1.0 / a + 1.0 / b + 1.0 / (omega * omega));
        log_prefactor = log_c + 2.5 * std::log(std::acos(-1.0))
            + std::log(2.0 * difference) - 1.5 * (std::log(a) + std::log(b));
    }
};

double screening_log_budget(double error, std::size_t count1, std::size_t count2 = 1) {
    if (!std::isfinite(error) || error < 0.0)
        throw std::invalid_argument("GDF SR integral screening error must be finite and nonnegative");
    // Divide in logarithms: a double-image candidate census can exceed
    // size_t even though each streamed box is separately representable.
    return error > 0.0 ? std::log(error) - std::log(double(count1))
        - std::log(double(count2)) - 1e-8 : -std::numeric_limits<double>::infinity();
}

Eigen::Vector3d origin(const libint2::Shell& shell) {
    return {shell.O[0], shell.O[1], shell.O[2]};
}
void validate_centers(const BasisSet& basis, const PeriodicSystem& system) {
    const Eigen::Matrix3d inverse = system.lattice.inverse();
    for (const auto& shell : basis.libint()) {
        const Eigen::Vector3d fractional = inverse * origin(shell);
        if (!fractional.allFinite() || fractional.cwiseAbs().maxCoeff() > 1e12)
            throw std::invalid_argument("GDF SR basis center cannot be represented safely in image enumeration");
    }
}
libint2::Shell shifted(const libint2::Shell& shell, const Eigen::Vector3d& r) {
    auto result = shell;
    for (int i = 0; i < 3; ++i) result.O[i] += r[i];
    return result;
}

// Energy and weighted derivatives share these exact image domains and
// screening decisions. A derivative holds their discrete membership fixed;
// no derivative tensor indexed by atom or lattice image is materialized.
struct MetricImageDomain {
    ImageBox images;
    const PeriodicSystem& system;
    double cutoff, error, omega, log_budget;
    MetricImageDomain(const PeriodicSystem& sys, double radius, double split,
                      std::size_t cap, double screen)
        : images(sys, radius, cap), system(sys), cutoff(radius), error(screen), omega(split),
          log_budget(screening_log_budget(screen, images.count_bound)) {}

    template<class F> void each(const libint2::Shell& p, const libint2::Shell& s,
                               F&& f) const {
        const ShellEnvelope ep(p), es(s);
        const GaussianSRBound bound(ep.a, es.a, ep.log_c + es.log_c, omega);
        auto visit = [&](const Eigen::Vector3d& r) {
            if (!pair_in_range(p, s, r, cutoff)) return;
            if (error > 0.0 && bound.log_prefactor - bound.decay
                * (origin(p) - origin(s) - r).squaredNorm() < log_budget) return;
            f(r);
        };
        if (error > 0.0) {
            if (bound.log_prefactor < log_budget) return;
            const double radius = std::sqrt((bound.log_prefactor - log_budget) / bound.decay);
            if (radius < cutoff) {
                ImageBox(system, radius, images.cap).each(origin(p) - origin(s), visit);
                return;
            }
        }
        images.each(origin(p) - origin(s), visit);
    }
};

struct ThreeCenterImageDomain {
    ImageBox pair_images, aux_images;
    const PeriodicSystem& system;
    double pair_cutoff, auxiliary_cutoff, omega, error, log_budget;
    std::size_t image_cap;
    ThreeCenterImageDomain(const PeriodicSystem& sys, double pair_radius,
                           double aux_radius, double split, std::size_t cap, double screen)
        : pair_images(sys, pair_radius, cap),
          aux_images(sys, aux_radius + pair_radius / 2.0, cap), system(sys),
          pair_cutoff(pair_radius), auxiliary_cutoff(aux_radius), omega(split),
          error(screen), log_budget(screening_log_budget(
              screen, pair_images.count_bound, aux_images.count_bound)), image_cap(cap) {}

    template<class Start, class Integral, class Finish>
    void each(const libint2::Shell& p, const libint2::Shell& mu,
              const libint2::Shell& nu, Start&& start, Integral&& integral,
              Finish&& finish) const {
        const Eigen::Vector3d A = origin(mu), B = origin(nu), P = origin(p);
        const ShellEnvelope ep(p), em(mu), en(nu);
        const double pair_exponent = em.a + en.a;
        const double beta = em.a * en.a / pair_exponent;
        const GaussianSRBound bound(ep.a, pair_exponent,
            ep.log_c + em.log_c + en.log_c, omega);
        auto visit_pair = [&](const Eigen::Vector3d& r) {
            if (!pair_in_range(mu, nu, r, pair_cutoff)) return;
            const Eigen::Vector3d end = B + r, segment = end - A;
            const double length2 = segment.squaredNorm();
            const double log_pair_bound = bound.log_prefactor - beta * length2;
            if (error > 0.0 && log_pair_bound < log_budget) return;
            const Eigen::Vector3d product_center = (em.a * A + en.a * end) / pair_exponent;
            start(r);
            auto visit = [&](const Eigen::Vector3d& t) {
                const Eigen::Vector3d displacement = P + t - A;
                const double fraction = length2 > 0.0
                    ? std::clamp(displacement.dot(segment) / length2, 0.0, 1.0) : 0.0;
                if ((displacement - fraction * segment).squaredNorm()
                    > auxiliary_cutoff * auxiliary_cutoff) return;
                if (error > 0.0 && log_pair_bound - bound.decay
                    * (P + t - product_center).squaredNorm() < log_budget) return;
                integral(r, t);
            };
            const double radius = error > 0.0
                ? std::sqrt(std::max(0.0, (log_pair_bound - log_budget) / bound.decay))
                : std::numeric_limits<double>::infinity();
            aux_images.each_capsule(A - P, end - P, auxiliary_cutoff,
                                    product_center - P, radius, visit);
            finish(r);
        };
        if (error > 0.0) {
            if (bound.log_prefactor < log_budget) return;
            const double radius = std::sqrt((bound.log_prefactor - log_budget) / beta);
            if (radius < pair_cutoff) {
                ImageBox(system, radius, image_cap).each(A - B, visit_pair);
                return;
            }
        }
        pair_images.each(A - B, visit_pair);
    }
};

template<class F> void primitive_blocks(const libint2::Shell& shell, F&& f) {
    if (shell.nprim() <= primitive_block_size) {
        f(shell);
        return;
    }
    for (std::size_t first = 0; first < shell.nprim(); first += primitive_block_size) {
        const auto last = std::min(shell.nprim(), first + primitive_block_size);
        libint2::svector<double> alpha(shell.alpha.begin() + first, shell.alpha.begin() + last);
        libint2::svector<libint2::Shell::Contraction> contractions;
        for (const auto& c : shell.contr) {
            libint2::svector<double> coefficients(c.coeff.begin() + first, c.coeff.begin() + last);
            contractions.push_back({c.l, c.pure, std::move(coefficients)});
        }
        // Preserve the original contraction coefficients, including their
        // normalization. Renormalizing each block changes the basis.
        const libint2::Shell block(std::move(alpha), std::move(contractions), shell.O, false);
        f(block);
    }
}

libint2::Engine engine_for(const libint2::BasisSet& a,
                          const libint2::BasisSet& b, double omega,
                          libint2::BraKet braket, int derivative_order = 0) {
    using Params = libint2::operator_traits<libint2::Operator::erfc_coulomb>::oper_params_type;
    // Supply the reduced-center braket at construction. Constructing the
    // default four-center engine and calling set(braket) afterwards retains
    // its nprim^4 primitive-data allocation (libint initialize(0) does not
    // resize primdata), even for a two-center metric.
    return libint2::Engine(libint2::Operator::erfc_coulomb,
        std::min(std::max(a.max_nprim(), b.max_nprim()), primitive_block_size),
        std::max(a.max_l(), b.max_l()),
        derivative_order, std::numeric_limits<double>::epsilon(), Params{omega}, braket);
}

void validate_gradient_atoms(const BasisSet& basis, const PeriodicSystem& system) {
    for (std::size_t shell = 0; shell < basis.libint().size(); ++shell) {
        const auto atom = basis.shell_atom_index(shell);
        if (atom < 0 || static_cast<std::size_t>(atom) >= system.unit_cell.size())
            throw std::invalid_argument("GDF SR derivative shell atom index is outside the cell");
    }
}

template<class F> void gdf_parallel_tasks(std::size_t tasks, int workers, F&& work) {
    std::exception_ptr failure;
    #pragma omp parallel for schedule(dynamic) num_threads(workers)
    for (std::size_t task = 0; task < tasks; ++task) {
        try {
            work(task);
        } catch (...) {
            #pragma omp critical(gdf_short_range_failure)
            {
                if (!failure) failure = std::current_exception();
            }
        }
    }
    if (failure) std::rethrow_exception(failure);
}
}  // namespace

std::size_t gdf_short_range_workspace_bytes(
    const BasisSet& orbital, const BasisSet& aux, int n_threads,
    int derivative_order, std::size_t n_atoms) {
    if (n_threads <= 0 || derivative_order < 0 || derivative_order > 1)
        throw std::invalid_argument("GDF SR workspace requires positive threads and derivative order 0 or 1");
    ensure_libint_initialized();
    const auto& a = aux.libint();
    const auto& o = orbital.libint();
    const auto extra = derivative_order
        ? sum(product(product(n_atoms, 3), sizeof(double)), sizeof(Eigen::MatrixXd)) : 0;
    const auto metric = sr_worker_memory(a, a, 2, derivative_order, extra);
    const auto tensor = sr_worker_memory(a, o, 3, derivative_order, extra);
    const auto metric_workers = std::min(std::size_t(n_threads), product(a.size(), a.size()));
    const auto tensor_workers = std::min(std::size_t(n_threads), product(a.size(), product(o.size(), o.size())));
    return std::max(sum(metric.prototype, product(metric_workers, metric.worker)),
                    sum(tensor.prototype, product(tensor_workers, tensor.worker)));
}

Eigen::MatrixXcd compute_gdf_sr_metric(
    const BasisSet& aux, const PeriodicSystem& system,
    const Eigen::Vector3d& q, double omega, double cutoff,
    std::size_t output_byte_cap, std::size_t image_candidate_cap,
    std::size_t workspace_byte_cap, double integral_screen_error) {
    validate(system, q, omega, cutoff);
    validate_centers(aux, system);
    const auto n = aux.nbasis();
    if (product(product(n, n), sizeof(Complex)) > output_byte_cap)
        throw std::length_error("GDF SR metric output byte cap exceeded");
    const MetricImageDomain domain(system, cutoff, omega, image_candidate_cap, integral_screen_error);
    ensure_libint_initialized();
    const auto& shells = aux.libint();
    const auto offsets = shells.shell2bf();
    const auto ns = shells.size();
    const auto tasks = product(ns, ns);
    const int workers = sr_workers(shells, shells, 2, tasks, workspace_byte_cap);
    auto prototype = engine_for(shells, shells, omega, libint2::BraKet::xs_xs);
    std::vector<libint2::Engine> engines(workers, prototype);
    Eigen::MatrixXcd result = Eigen::MatrixXcd::Zero(n, n);
    gdf_parallel_tasks(tasks, workers, [&](std::size_t task) {
        const auto p = task / ns, s = task % ns;
        auto& engine = engines[omp_thread_index()];
        domain.each(shells[p], shells[s], [&](const Eigen::Vector3d& r) {
            const auto ket = shifted(shells[s], r);
            const Complex phase = std::exp(Complex(0.0, q.dot(r)));
            primitive_blocks(shells[p], [&](const libint2::Shell& bra_block) {
                primitive_blocks(ket, [&](const libint2::Shell& ket_block) {
                    engine.compute(bra_block, ket_block);
                    const double* block = engine.results()[0];
                    if (!block) return;
                    for (std::size_t i = 0; i < shells[p].size(); ++i)
                        for (std::size_t j = 0; j < shells[s].size(); ++j)
                            result(offsets[p] + i, offsets[s] + j) += phase * block[i * shells[s].size() + j];
                });
            });
        });
    });
    return result;
}

GDFShortRangeBatch compute_gdf_sr_three_center(
    const BasisSet& orbital, const BasisSet& aux,
    const PeriodicSystem& system, const Eigen::Vector3d& q,
    const Eigen::MatrixXd& ket_kpoints, double omega,
    double pair_cutoff, double auxiliary_cutoff,
    std::size_t output_byte_cap, std::size_t image_candidate_cap,
    std::size_t workspace_byte_cap, double integral_screen_error) {
    validate(system, q, omega, pair_cutoff);
    validate(system, q, omega, auxiliary_cutoff);
    validate_centers(orbital, system);
    validate_centers(aux, system);
    if (ket_kpoints.cols() != 3 || ket_kpoints.rows() == 0 || !ket_kpoints.allFinite())
        throw std::invalid_argument("GDF SR ket kpoints must have finite shape (n_k, 3)");
    const auto no = orbital.nbasis(), na = aux.nbasis();
    const auto nk = static_cast<std::size_t>(ket_kpoints.rows());
    const auto extent = product(product(product(nk, na), no), no);
    if (product(extent, sizeof(Complex)) > output_byte_cap)
        throw std::length_error("GDF SR three-center output byte cap exceeded");
    const ThreeCenterImageDomain domain(system, pair_cutoff, auxiliary_cutoff,
        omega, image_candidate_cap, integral_screen_error);
    ensure_libint_initialized();
    const auto& a = aux.libint();
    const auto& o = orbital.libint();
    const auto ai = a.shell2bf(), oi = o.shell2bf();
    const auto ns = o.size(), tasks = product(product(a.size(), ns), ns);
    const int workers = sr_workers(a, o, 3, tasks, workspace_byte_cap);
    auto prototype = engine_for(a, o, omega, libint2::BraKet::xs_xx);
    std::vector<libint2::Engine> engines(workers, prototype);
    GDFShortRangeBatch result{nk, na, no, std::vector<Complex>(extent)};
    gdf_parallel_tasks(tasks, workers, [&](std::size_t task) {
        const auto p = task / (ns * ns), mu = (task / ns) % ns, nu = task % ns;
        auto& engine = engines[omp_thread_index()];
        const auto np = a[p].size(), nm = o[mu].size(), nn = o[nu].size();
        std::vector<Complex> contracted(np * nm * nn);
        auto ket = o[nu];
        domain.each(a[p], o[mu], o[nu], [&](const Eigen::Vector3d& r) {
            ket = shifted(o[nu], r);
            std::fill(contracted.begin(), contracted.end(), Complex{});
        }, [&](const Eigen::Vector3d&, const Eigen::Vector3d& t) {
            const auto bra = shifted(a[p], t);
            const Complex phase = std::exp(Complex(0.0, -q.dot(t)));
            primitive_blocks(bra, [&](const libint2::Shell& p_block) {
                primitive_blocks(o[mu], [&](const libint2::Shell& mu_block) {
                    primitive_blocks(ket, [&](const libint2::Shell& nu_block) {
                        engine.compute(p_block, mu_block, nu_block);
                        const double* block = engine.results()[0];
                        if (!block) return;
                        for (std::size_t i = 0; i < contracted.size(); ++i)
                            contracted[i] += phase * block[i];
                    });
                });
            });
        }, [&](const Eigen::Vector3d& r) {
            for (std::size_t k = 0; k < nk; ++k) {
                const Complex phase = std::exp(Complex(0.0, ket_kpoints.row(k).dot(r)));
                for (std::size_t i = 0; i < np; ++i)
                    for (std::size_t j = 0; j < nm; ++j)
                        for (std::size_t l = 0; l < nn; ++l)
                            result.data[((k * na + ai[p] + i) * no + oi[mu] + j) * no + oi[nu] + l]
                                += phase * contracted[(i * nm + j) * nn + l];
            }
        });
    });
    return result;
}

Eigen::MatrixXd compute_gdf_sr_metric_gradient_weighted(
    const BasisSet& aux, const PeriodicSystem& system,
    const Eigen::Vector3d& q, double omega, double cutoff,
    const Eigen::Ref<const Eigen::MatrixXcd>& weight,
    std::size_t output_byte_cap, std::size_t image_candidate_cap,
    std::size_t workspace_byte_cap, double integral_screen_error) {
    validate(system, q, omega, cutoff);
    validate_centers(aux, system);
    validate_gradient_atoms(aux, system);
    const auto n = aux.nbasis(), atoms = system.unit_cell.size();
    if (weight.rows() != static_cast<Eigen::Index>(n)
        || weight.cols() != static_cast<Eigen::Index>(n) || !weight.allFinite())
        throw std::invalid_argument("GDF SR metric derivative requires finite matching weights");
    const auto gradient_bytes = product(product(atoms, 3), sizeof(double));
    if (gradient_bytes > output_byte_cap)
        throw std::length_error("GDF SR metric derivative output byte cap exceeded");
    const MetricImageDomain domain(system, cutoff, omega, image_candidate_cap, integral_screen_error);
    ensure_libint_initialized();
    const auto& shells = aux.libint();
    const auto offsets = shells.shell2bf();
    const auto ns = shells.size(), tasks = product(ns, ns);
    const int workers = sr_workers(shells, shells, 2, tasks, workspace_byte_cap,
        1, sum(gradient_bytes, sizeof(Eigen::MatrixXd)));
    auto prototype = engine_for(shells, shells, omega, libint2::BraKet::xs_xs, 1);
    std::vector<libint2::Engine> engines(workers, prototype);
    auto gradients = make_worker_gradients(workers, atoms);
    gdf_parallel_tasks(tasks, workers, [&](std::size_t task) {
        const auto p = task / ns, s = task % ns;
        const auto worker = omp_thread_index();
        auto& engine = engines[worker];
        auto& gradient = gradients[worker];
        const int centers[] = {aux.shell_atom_index(p), aux.shell_atom_index(s)};
        if (centers[0] == centers[1]) return;  // rigid translation of both centers
        domain.each(shells[p], shells[s], [&](const Eigen::Vector3d& r) {
            const auto ket = shifted(shells[s], r);
            const Complex phase = std::exp(Complex(0.0, q.dot(r)));
            primitive_blocks(shells[p], [&](const libint2::Shell& p_block) {
                primitive_blocks(ket, [&](const libint2::Shell& s_block) {
                    engine.compute(p_block, s_block);
                    // A screened contraction clears target 0; later derivative pointers
                    // can retain buffers from the previous shell tuple (Libint contract).
                    if (!engine.results()[0]) return;
                    for (int center = 0; center < 2; ++center)
                        for (int axis = 0; axis < 3; ++axis) {
                            const double* block = engine.results()[3 * center + axis];
                            if (!block) continue;
                            double value = 0.0;
                            for (std::size_t i = 0; i < shells[p].size(); ++i)
                                for (std::size_t j = 0; j < shells[s].size(); ++j)
                                    value += std::real(phase * weight(offsets[p] + i, offsets[s] + j))
                                        * block[i * shells[s].size() + j];
                            gradient(centers[center], axis) += value;
                        }
                });
            });
        });
    });
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(atoms, 3);
    for (const auto& gradient : gradients) result += gradient;
    if (!result.allFinite())
        throw std::invalid_argument("GDF SR derivative produced a nonfinite atom gradient");
    return result;
}

Eigen::MatrixXd compute_gdf_sr_three_center_gradient_weighted(
    const BasisSet& orbital, const BasisSet& aux, const PeriodicSystem& system,
    const Eigen::Vector3d& q, const Eigen::MatrixXd& ket_kpoints,
    double omega, double pair_cutoff, double auxiliary_cutoff,
    GDFShortRangeWeightView weight,
    std::size_t output_byte_cap, std::size_t image_candidate_cap,
    std::size_t workspace_byte_cap, double integral_screen_error) {
    validate(system, q, omega, pair_cutoff);
    validate(system, q, omega, auxiliary_cutoff);
    validate_centers(orbital, system);
    validate_centers(aux, system);
    validate_gradient_atoms(orbital, system);
    validate_gradient_atoms(aux, system);
    const auto no = orbital.nbasis(), na = aux.nbasis(), atoms = system.unit_cell.size();
    const auto nk = static_cast<std::size_t>(ket_kpoints.rows());
    if (ket_kpoints.cols() != 3 || !nk || !ket_kpoints.allFinite()
        || !weight.data || weight.n_k != nk || weight.n_aux != na || weight.n_orb != no)
        throw std::invalid_argument("GDF SR three-center derivative requires matching kpoints and weights");
    const auto weight_count = product(product(product(nk, na), no), no);
    for (std::size_t i = 0; i < weight_count; ++i)
        if (!std::isfinite(weight.data[i].real()) || !std::isfinite(weight.data[i].imag()))
            throw std::invalid_argument("GDF SR three-center derivative requires finite weights");
    const auto gradient_bytes = product(product(atoms, 3), sizeof(double));
    if (gradient_bytes > output_byte_cap)
        throw std::length_error("GDF SR three-center derivative output byte cap exceeded");
    const ThreeCenterImageDomain domain(system, pair_cutoff, auxiliary_cutoff,
        omega, image_candidate_cap, integral_screen_error);
    ensure_libint_initialized();
    const auto& a = aux.libint();
    const auto& o = orbital.libint();
    const auto ai = a.shell2bf(), oi = o.shell2bf();
    const auto ns = o.size(), tasks = product(product(a.size(), ns), ns);
    const int workers = sr_workers(a, o, 3, tasks, workspace_byte_cap,
        1, sum(gradient_bytes, sizeof(Eigen::MatrixXd)));
    auto prototype = engine_for(a, o, omega, libint2::BraKet::xs_xx, 1);
    std::vector<libint2::Engine> engines(workers, prototype);
    auto gradients = make_worker_gradients(workers, atoms);
    gdf_parallel_tasks(tasks, workers, [&](std::size_t task) {
        const auto p = task / (ns * ns), mu = (task / ns) % ns, nu = task % ns;
        const auto worker = omp_thread_index();
        auto& engine = engines[worker];
        auto& gradient = gradients[worker];
        const auto np = a[p].size(), nm = o[mu].size(), nn = o[nu].size();
        const int centers[] = {aux.shell_atom_index(p), orbital.shell_atom_index(mu),
                               orbital.shell_atom_index(nu)};
        if (centers[0] == centers[1] && centers[1] == centers[2]) return;
        std::vector<Complex> image_weight(np * nm * nn);
        auto ket = o[nu];
        domain.each(a[p], o[mu], o[nu], [&](const Eigen::Vector3d& r) {
            ket = shifted(o[nu], r);
            std::fill(image_weight.begin(), image_weight.end(), Complex{});
            for (std::size_t k = 0; k < nk; ++k) {
                const Complex phase = std::exp(Complex(0.0, ket_kpoints.row(k).dot(r)));
                for (std::size_t i = 0; i < np; ++i)
                    for (std::size_t j = 0; j < nm; ++j)
                        for (std::size_t l = 0; l < nn; ++l)
                            image_weight[(i * nm + j) * nn + l] += phase
                                * weight.data[((k * na + ai[p] + i) * no + oi[mu] + j) * no + oi[nu] + l];
            }
        }, [&](const Eigen::Vector3d&, const Eigen::Vector3d& t) {
            const auto bra = shifted(a[p], t);
            const Complex phase = std::exp(Complex(0.0, -q.dot(t)));
            primitive_blocks(bra, [&](const libint2::Shell& p_block) {
                primitive_blocks(o[mu], [&](const libint2::Shell& mu_block) {
                    primitive_blocks(ket, [&](const libint2::Shell& nu_block) {
                        engine.compute(p_block, mu_block, nu_block);
                        // A screened contraction clears target 0; later derivative pointers
                        // can retain buffers from the previous shell tuple (Libint contract).
                        if (!engine.results()[0]) return;
                        for (int center = 0; center < 3; ++center)
                            for (int axis = 0; axis < 3; ++axis) {
                                const double* block = engine.results()[3 * center + axis];
                                if (!block) continue;
                                double value = 0.0;
                                for (std::size_t i = 0; i < image_weight.size(); ++i)
                                    value += std::real(phase * image_weight[i]) * block[i];
                                gradient(centers[center], axis) += value;
                            }
                    });
                });
            });
        }, [](const Eigen::Vector3d&) {});
    });
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(atoms, 3);
    for (const auto& gradient : gradients) result += gradient;
    if (!result.allFinite())
        throw std::invalid_argument("GDF SR derivative produced a nonfinite atom gradient");
    return result;
}

GDFPlaneWaveProjection compute_gdf_plane_wave_projection(
    const BasisSet& orbital, const BasisSet& aux,
    const PeriodicSystem& system, const Eigen::Vector3d& q,
    const Eigen::MatrixXd& ket_kpoints,
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& vectors,
    double pair_cutoff, std::size_t output_byte_cap,
    std::size_t transient_byte_cap, std::size_t image_candidate_cap,
    bool compute_metric) {
    validate(system, q, 1.0, pair_cutoff);
    if (ket_kpoints.cols() != 3 || ket_kpoints.rows() == 0 || !ket_kpoints.allFinite()
        || !vectors.allFinite() || vectors.rows() == 0)
        throw std::invalid_argument("MDF projection requires finite nonempty kpoints and vectors");
    const auto na = aux.nbasis(), no = orbital.nbasis();
    if (na == 0 || no == 0)
        throw std::invalid_argument("MDF projection requires nonempty bases");
    const auto nk = static_cast<std::size_t>(ket_kpoints.rows());
    const auto npw = static_cast<std::size_t>(vectors.rows());
    const auto pairs = product(no, no);
    const auto metric_bytes = compute_metric ? product(product(na, na), sizeof(Complex)) : 0;
    const auto tensor_size = product(product(nk, na), pairs);
    const auto factor_size = product(product(nk, npw), pairs);
    if (sum(metric_bytes, product(sum(tensor_size, factor_size), sizeof(Complex))) > output_byte_cap)
        throw std::length_error("MDF projection output byte cap exceeded");
    constexpr std::size_t vector_block = 128;
    const auto pair_block = std::min<std::size_t>(16, std::max<std::size_t>(1,
        product(pairs, nk) / product(4, std::size_t(std::max(1, omp_max_threads())))));
    const auto aux_bytes = product(product(na, vector_block), sizeof(Complex));
    const auto pair_bytes = product(product(pair_block, vector_block), sizeof(Complex));
    const auto worker_bytes = sum(pair_bytes, ao_pair_fourier_fixed_numeric_workspace_bytes());
    const auto shared_bytes = sum(aux_bytes, sizeof(std::array<double, vector_block>));
    if (sum(shared_bytes, worker_bytes) > transient_byte_cap)
        throw std::length_error("MDF projection Fourier transient byte cap exceeded");
    const auto worker_cap = std::min<std::size_t>(std::numeric_limits<int>::max(),
        (transient_byte_cap - shared_bytes) / worker_bytes);
    const auto pair_tiles = pairs / pair_block + (pairs % pair_block != 0);
    const auto tasks = product(nk, pair_tiles);
    const int workers = omp_workers_for(tasks, static_cast<int>(worker_cap));
    const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
    AuxiliaryFourierVectorView at_zero{zero.data(), zero.data()+1, zero.data()+2, 1};
    // Validate the full basis and image census before any large output.
    auxiliary_gaussian_fourier_panel(aux, at_zero, aux_bytes);
    ao_pair_gaussian_fourier_panel(orbital, system, at_zero, zero,
        0, 0, pair_cutoff, image_candidate_cap, pair_bytes);
    GDFPlaneWaveProjection result;
    if (compute_metric) result.metric = Eigen::MatrixXcd::Zero(na, na);
    result.three_center = {nk, na, no, std::vector<Complex>(tensor_size)};
    result.factors = {nk, npw, no, std::vector<Complex>(factor_size)};
    const RangeSeparatedCoulombKernel kernel(system, q, 1.0);
    for (std::size_t first_g = 0; first_g < npw; first_g += vector_block) {
        const auto ng = std::min(vector_block, npw - first_g);
        const double* ptr = vectors.data() + 3 * first_g;
        AuxiliaryFourierVectorView panel{ptr, ptr+1, ptr+2, ng, 3, 3, 3};
        auto auxiliary = auxiliary_gaussian_fourier_panel(aux, panel, aux_bytes);
        std::array<double, vector_block> weights{};
        for (std::size_t g = 0; g < ng; ++g) {
            const double p2 = vectors.row(first_g + g).squaredNorm();
            if (RangeSeparatedCoulombKernel::singular(p2)) continue;
            weights[g] = kernel.bare(p2);
            if (!std::isfinite(weights[g]))
                throw std::invalid_argument("MDF projection Coulomb weight is nonfinite");
            ++result.reciprocal_vector_count;
        }
        if (compute_metric) {
            const int metric_workers = omp_workers_for(na, omp_max_threads());
            #pragma omp parallel for schedule(static) num_threads(metric_workers)
            for (std::size_t s = 0; s < na; ++s)
                for (std::size_t g = 0; g < ng; ++g)
                    for (std::size_t p = 0; p < na; ++p)
                        result.metric(p, s) += weights[g] * std::conj(auxiliary(p, g)) * auxiliary(s, g);
        }
        std::exception_ptr failure;
        #pragma omp parallel for schedule(dynamic) num_threads(workers)
        for (std::size_t task = 0; task < tasks; ++task) {
            try {
                const auto k = task / pair_tiles;
                const auto first = (task % pair_tiles) * pair_block;
                const auto count = std::min(pair_block, pairs - first);
                auto ao = ao_pair_gaussian_fourier_panel(orbital, system, panel,
                    ket_kpoints.row(k).transpose(), first, count, pair_cutoff,
                    image_candidate_cap, pair_bytes);
                for (std::size_t pair = 0; pair < count; ++pair) {
                    for (std::size_t g = 0; g < ng; ++g)
                        result.factors.data[(k * npw + first_g + g) * pairs + first + pair]
                            = std::sqrt(weights[g]) * ao.data[pair * ng + g];
                    for (std::size_t p = 0; p < na; ++p) {
                        Complex value{};
                        for (std::size_t g = 0; g < ng; ++g)
                            value += weights[g] * std::conj(auxiliary(p, g)) * ao.data[pair * ng + g];
                        result.three_center.data[(k * na + p) * pairs + first + pair] += value;
                    }
                }
            } catch (...) {
                #pragma omp critical(mdf_projection_failure)
                { if (!failure) failure = std::current_exception(); }
            }
        }
        if (failure) std::rethrow_exception(failure);
    }
    return result;
}

GDFRangeSeparatedIntegrals compute_gdf_range_separated_integrals(
    const BasisSet& orbital, const BasisSet& aux,
    const PeriodicSystem& system, const Eigen::Vector3d& q,
    const Eigen::MatrixXd& ket_kpoints,
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& vectors,
    double omega, double pair_cutoff, double auxiliary_cutoff,
    std::size_t output_byte_cap, std::size_t transient_byte_cap,
    std::size_t image_candidate_cap, double integral_screen_error, bool compute_metric,
    const std::string& physical_kernel, double omega_screen) {
    validate(system, q, omega, pair_cutoff);
    validate(system, q, omega, auxiliary_cutoff);
    if (physical_kernel != "coulomb" && physical_kernel != "erfc" && physical_kernel != "erf")
        throw std::invalid_argument("RSGDF physical kernel must be coulomb, erfc or erf");
    const bool screened = physical_kernel != "coulomb";
    const bool real_short_range = physical_kernel != "erf";
    if (!std::isfinite(omega_screen) || (screened
        ? (omega_screen <= 0.0 || !std::isfinite(omega_screen * omega_screen)
            || omega_screen * omega_screen == 0.0) : omega_screen != 0.0))
        throw std::invalid_argument("RSGDF physical range is zero for Coulomb, finite and positive for erfc/erf");
    if (ket_kpoints.cols() != 3 || ket_kpoints.rows() == 0 || !ket_kpoints.allFinite())
        throw std::invalid_argument("RSGDF ket kpoints must have finite shape (n_k, 3)");
    if (!vectors.allFinite() || vectors.rows() == 0)
        throw std::invalid_argument("RSGDF reciprocal vectors must be nonempty and finite");
    const auto na = aux.nbasis(), no = orbital.nbasis();
    const auto nk = static_cast<std::size_t>(ket_kpoints.rows());
    const auto pairs = product(no, no);
    const auto metric_bytes = compute_metric ? product(product(na, na), sizeof(Complex)) : 0;
    const auto tensor_bytes = product(product(product(nk, na), pairs), sizeof(Complex));
    if (metric_bytes > output_byte_cap || tensor_bytes > output_byte_cap - metric_bytes)
        throw std::length_error("RSGDF combined output byte cap exceeded");
    // Fixed pair/vector tiles; reciprocal work has no full AO-square*G
    // transient and does not depend on the steepest Gaussian exponent.
    constexpr std::size_t vector_block = 128;
    // Keep several disjoint pair tiles per worker on small cells. A fixed
    // 16-pair tile leaves only 11 Fourier tasks for 13 AOs at Gamma,
    // regardless of the number of reciprocal panels or available cores.
    // This changes scheduling and scratch size, never a pair's G-sum order.
    const auto target_tasks = product(4, std::size_t(std::max(1, omp_max_threads())));
    const auto pair_block = std::min<std::size_t>(16,
        std::max<std::size_t>(1, product(pairs, nk) / target_tasks));
    const auto aux_bytes = product(product(na, vector_block), sizeof(Complex));
    const auto pair_bytes = product(pair_block * vector_block, sizeof(Complex));
    const auto charge_bytes = product(na, sizeof(Complex));
    const auto fixed_bytes = ao_pair_fourier_fixed_numeric_workspace_bytes()
        + sizeof(std::array<double, vector_block>);
    if (aux_bytes > transient_byte_cap
        || charge_bytes > transient_byte_cap - aux_bytes
        || pair_bytes > transient_byte_cap - aux_bytes - charge_bytes
        || fixed_bytes > transient_byte_cap - aux_bytes - charge_bytes - pair_bytes)
        throw std::length_error("RSGDF Fourier transient byte cap exceeded");
    const auto per_worker_bytes = pair_bytes + ao_pair_fourier_fixed_numeric_workspace_bytes();
    const auto available_worker_bytes = transient_byte_cap - aux_bytes - charge_bytes
        - sizeof(std::array<double, vector_block>);
    const auto worker_cap = std::min<std::size_t>(
        available_worker_bytes / per_worker_bytes, std::numeric_limits<int>::max());
    const auto pair_tiles = pairs / pair_block + (pairs % pair_block != 0);
    const auto fourier_tasks = product(nk, pair_tiles);
    const int fourier_workers = omp_workers_for(fourier_tasks, static_cast<int>(worker_cap));
    // Validate basis conventions/centers and AO enumeration before the SR
    // build. A zero-vector, empty-pair panel performs this without AO output.
    const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
    AuxiliaryFourierVectorView at_zero{zero.data(), zero.data()+1, zero.data()+2, 1};
    auto charges = auxiliary_gaussian_fourier_panel(aux, at_zero, product(na, sizeof(Complex)));
    ao_pair_gaussian_fourier_panel(orbital, system, at_zero, zero,
        0, 0, pair_cutoff, image_candidate_cap, pair_bytes);

    GDFRangeSeparatedIntegrals result;
    if (real_short_range) {
        if (compute_metric)
            result.metric = compute_gdf_sr_metric(aux, system, q, omega,
                auxiliary_cutoff, metric_bytes, image_candidate_cap, transient_byte_cap - charge_bytes,
                integral_screen_error);
        result.three_center = compute_gdf_sr_three_center(orbital, aux, system,
            q, ket_kpoints, omega, pair_cutoff, auxiliary_cutoff,
            tensor_bytes, image_candidate_cap, transient_byte_cap - charge_bytes, integral_screen_error);
    } else {
        // Physical erf is a smooth, independently fitted reciprocal source.
        // No full-minus-SR factor subtraction or numerical real-space tail.
        if (compute_metric) result.metric = Eigen::MatrixXcd::Zero(na, na);
        result.three_center = {nk, na, no,
            std::vector<Complex>(product(product(nk, na), pairs))};
    }
    const RangeSeparatedCoulombKernel kernel(system, q, omega);
    if (real_short_range && kernel.zero_transfer) {
        // Ye 2021 Eq. (12), and its two-center analogue: subtract the
        // finite SR p=0 mode once. The LR primed sum below omits p=0.
        const double g0 = kernel.short_range_zero_mode;
        if (compute_metric)
            for (std::size_t p = 0; p < na; ++p)
                for (std::size_t s = 0; s < na; ++s)
                    result.metric(p, s) -= g0 * std::conj(charges(p, 0)) * charges(s, 0);
        for (std::size_t k = 0; k < nk; ++k) {
            for (std::size_t first = 0; first < pairs; first += pair_block) {
                const auto count = std::min(pair_block, pairs - first);
                auto overlap = ao_pair_gaussian_fourier_panel(orbital, system, at_zero,
                    ket_kpoints.row(k).transpose(), first, count, pair_cutoff,
                    image_candidate_cap, pair_bytes);
                for (std::size_t p = 0; p < na; ++p)
                    for (std::size_t pair = 0; pair < count; ++pair)
                        result.three_center.data[(k * na + p) * pairs + first + pair]
                            -= g0 * std::conj(charges(p, 0)) * overlap.data[pair];
            }
        }
    }
    for (Eigen::Index first_g = 0; first_g < vectors.rows(); first_g += vector_block) {
        const auto ng = static_cast<std::size_t>(std::min<Eigen::Index>(vector_block, vectors.rows() - first_g));
        const double* ptr = vectors.data() + 3 * first_g;
        AuxiliaryFourierVectorView panel_vectors{ptr, ptr+1, ptr+2, ng, 3, 3, 3};
        auto auxiliary = auxiliary_gaussian_fourier_panel(aux, panel_vectors, aux_bytes);
        std::array<double, vector_block> weights{};
        for (std::size_t g = 0; g < ng; ++g) {
            const double p2 = vectors.row(first_g + g).squaredNorm();
            if (RangeSeparatedCoulombKernel::singular(p2)) continue;
            if (!screened) {
                weights[g] = kernel.long_range(p2);
            } else {
                const double physical_exponent = p2 / (4.0 * omega_screen * omega_screen);
                if (!real_short_range) {
                    weights[g] = kernel.bare(p2) * std::exp(-physical_exponent);
                } else {
                    // erfc(eta) + erf(eta) - erf(omega_screen). eta is
                    // numerical, omega_screen physical. expm1 avoids losing
                    // precision when the two ranges nearly coincide.
                    const double numerical_exponent = p2 / (4.0 * kernel.omega_squared);
                    const double difference = numerical_exponent <= physical_exponent
                        ? -std::exp(-numerical_exponent)
                            * std::expm1(-(physical_exponent - numerical_exponent))
                        : std::exp(-physical_exponent)
                            * std::expm1(-(numerical_exponent - physical_exponent));
                    weights[g] = kernel.bare(p2) * difference;
                }
            }
            ++result.reciprocal_vector_count;
        }
        // Columns own disjoint metric slices, and (k,pair-tile) tasks own
        // disjoint T slices. G panels remain ordered: no per-thread metric
        // or tensor copy, and no reduction-order change with team size.
        if (compute_metric) {
            const int metric_workers = omp_workers_for(na, omp_max_threads());
            #pragma omp parallel for schedule(static) num_threads(metric_workers)
            for (std::size_t s = 0; s < na; ++s)
                for (std::size_t g = 0; g < ng; ++g)
                    for (std::size_t p = 0; p < na; ++p)
                        result.metric(p, s) += weights[g] * std::conj(auxiliary(p, g)) * auxiliary(s, g);
        }
        std::exception_ptr failure;
        #pragma omp parallel for schedule(dynamic) num_threads(fourier_workers)
        for (std::size_t task = 0; task < fourier_tasks; ++task) {
            try {
                const auto k = task / pair_tiles;
                const auto first = (task % pair_tiles) * pair_block;
                const auto count = std::min(pair_block, pairs - first);
                auto ao = ao_pair_gaussian_fourier_panel(orbital, system, panel_vectors,
                    ket_kpoints.row(k).transpose(), first, count, pair_cutoff,
                    image_candidate_cap, pair_bytes);
                for (std::size_t p = 0; p < na; ++p)
                    for (std::size_t pair = 0; pair < count; ++pair) {
                        Complex value{};
                        for (std::size_t g = 0; g < ng; ++g)
                            value += weights[g] * std::conj(auxiliary(p, g)) * ao.data[pair * ng + g];
                        result.three_center.data[(k * na + p) * pairs + first + pair] += value;
                    }
            } catch (...) {
                #pragma omp critical(gdf_fourier_failure)
                {
                    if (!failure) failure = std::current_exception();
                }
            }
        }
        if (failure) std::rethrow_exception(failure);
    }
    return result;
}
// Share Fourier center derivatives and image enumeration between the
// combined SR/LR source and the selected bare-Coulomb MDF projection.
static Eigen::MatrixXd compute_gdf_coulomb_gradient_weighted(
    const BasisSet& orbital, const BasisSet& aux,
    const PeriodicSystem& system, const Eigen::Vector3d& q,
    const Eigen::MatrixXd& ket_kpoints,
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& vectors,
    double omega, double pair_cutoff, double auxiliary_cutoff,
    const Eigen::Ref<const Eigen::MatrixXcd>& metric_weight,
    GDFShortRangeWeightView three_center_weight,
    std::size_t output_byte_cap, std::size_t transient_byte_cap,
    std::size_t image_candidate_cap, double integral_screen_error,
    bool plane_wave_projection, GDFShortRangeWeightView factor_weight) {
    validate(system, q, omega, pair_cutoff);
    validate(system, q, omega, auxiliary_cutoff);
    validate_centers(orbital, system);
    validate_centers(aux, system);
    validate_gradient_atoms(orbital, system);
    validate_gradient_atoms(aux, system);
    if (ket_kpoints.cols() != 3 || !ket_kpoints.rows() || !ket_kpoints.allFinite()
        || !vectors.rows() || !vectors.allFinite())
        throw std::invalid_argument("RSGDF derivative requires finite kpoints and reciprocal vectors");
    for (Eigen::Index g = 0; g < vectors.rows(); ++g)
        if (!std::isfinite(vectors.row(g).squaredNorm()))
            throw std::invalid_argument("RSGDF derivative requires finite squared momenta");
    const auto na = aux.nbasis(), no = orbital.nbasis(), atoms = system.unit_cell.size();
    const auto nk = static_cast<std::size_t>(ket_kpoints.rows());
    const auto pairs = product(no, no), tasks = product(nk, pairs);
    const auto gradient_bytes = product(product(atoms, 3), sizeof(double));
    if (gradient_bytes > output_byte_cap)
        throw std::length_error("RSGDF derivative output byte cap exceeded");
    if (metric_weight.rows() != static_cast<Eigen::Index>(na)
        || metric_weight.cols() != static_cast<Eigen::Index>(na)
        || !metric_weight.allFinite() || !three_center_weight.data
        || three_center_weight.n_k != nk || three_center_weight.n_aux != na
        || three_center_weight.n_orb != no)
        throw std::invalid_argument("RSGDF derivative requires matching finite weights");
    const auto weight_count = product(tasks, na);
    for (std::size_t i = 0; i < weight_count; ++i) {
        const auto value = three_center_weight.data[i];
        if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
            throw std::invalid_argument("RSGDF derivative requires finite weights");
    }
    if (plane_wave_projection) {
        const auto npw = static_cast<std::size_t>(vectors.rows());
        if (!factor_weight.data || factor_weight.n_k != nk
            || factor_weight.n_aux != npw || factor_weight.n_orb != no)
            throw std::invalid_argument("MDF projection derivative requires matching factor weights");
        const auto factor_count = product(tasks, npw);
        for (std::size_t i = 0; i < factor_count; ++i) {
            const auto value = factor_weight.data[i];
            if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
                throw std::invalid_argument("MDF projection derivative requires finite factor weights");
        }
    }
    constexpr std::size_t vector_block = 128;
    const auto aux_bytes = product(product(na, vector_block), sizeof(Complex));
    const auto owner_bytes = product(sum(na, no), sizeof(int));
    const auto fixed_bytes = sum(owner_bytes, sum(aux_bytes,
        sizeof(std::array<double, vector_block>)));
    const auto worker_bytes = sum(gradient_bytes, sum(sizeof(Eigen::MatrixXd),
        sum(aopair_ft_detail::center_derivative_numeric_workspace_bytes(),
            sizeof(std::array<Complex, 7 * vector_block>))));
    if (fixed_bytes > transient_byte_cap || worker_bytes > transient_byte_cap - fixed_bytes
        || gradient_bytes >= transient_byte_cap)
        throw std::length_error("RSGDF derivative Fourier workspace byte cap exceeded");
    const auto worker_cap = std::min<std::size_t>(
        (transient_byte_cap - fixed_bytes) / worker_bytes, std::numeric_limits<int>::max());
    const int workers = omp_workers_for(std::max(tasks, na), static_cast<int>(worker_cap));
    // Empty value panel applies the same basis and lattice-conditioning
    // validation without retaining an AO-by-G array.
    const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
    AuxiliaryFourierVectorView at_zero{zero.data(), zero.data()+1, zero.data()+2, 1};
    ao_pair_gaussian_fourier_panel(orbital, system, at_zero, zero,
        0, 0, pair_cutoff, image_candidate_cap, 1);
    const aopair_ft_detail::LongMatrix inverse = system.lattice.cast<long double>().inverse();
    for (std::size_t pair = 0; pair < pairs; ++pair) {
        const auto bra = aopair_ft_detail::ao(orbital, pair / no);
        const auto ket = aopair_ft_detail::ao(orbital, pair % no);
        if (aopair_ft_detail::image_box(bra, ket, inverse, pair_cutoff).candidates
            > image_candidate_cap)
            throw std::length_error("RSGDF derivative AO image candidate cap exceeded");
    }
    Eigen::MatrixXd result;
    if (plane_wave_projection) {
        result = Eigen::MatrixXd::Zero(atoms, 3);
    } else {
        result = compute_gdf_sr_metric_gradient_weighted(
            aux, system, q, omega, auxiliary_cutoff, metric_weight,
            output_byte_cap, image_candidate_cap, transient_byte_cap, integral_screen_error);
        // The temporary second atom gradient is transient, while result is output.
        result += compute_gdf_sr_three_center_gradient_weighted(
            orbital, aux, system, q, ket_kpoints, omega, pair_cutoff, auxiliary_cutoff,
            three_center_weight, gradient_bytes, image_candidate_cap,
            transient_byte_cap - gradient_bytes, integral_screen_error);
    }
    auto owners = [](const BasisSet& basis) {
        std::vector<int> values;
        values.reserve(basis.nbasis());
        for (std::size_t shell = 0; shell < basis.libint().size(); ++shell)
            values.insert(values.end(), basis.libint()[shell].size(), basis.shell_atom_index(shell));
        return values;
    };
    const auto aux_owner = owners(aux), orb_owner = owners(orbital);
    auto gradients = make_worker_gradients(workers, atoms);
    const RangeSeparatedCoulombKernel kernel(system, q, omega);
    auto contract_panel = [&](AuxiliaryFourierVectorView panel, bool zero_mode, std::size_t first_g) {
        const auto ng = panel.count;
        const auto auxiliary = auxiliary_gaussian_fourier_panel(aux, panel, aux_bytes);
        std::array<double, vector_block> kernels{};
        auto momentum = [&](std::size_t g) {
            return Eigen::Vector3d(panel.x[g * panel.x_stride],
                panel.y[g * panel.y_stride], panel.z[g * panel.z_stride]);
        };
        for (std::size_t g = 0; g < ng; ++g) {
            const double p2 = momentum(g).squaredNorm();
            if (!std::isfinite(p2))
                throw std::invalid_argument("RSGDF derivative requires finite squared momenta");
            kernels[g] = plane_wave_projection ? kernel.bare(p2)
                : (zero_mode ? -kernel.short_range_zero_mode : kernel.long_range(p2));
            if (!std::isfinite(kernels[g]))
                throw std::invalid_argument("GDF derivative Coulomb weight is nonfinite");
        }
        gdf_parallel_tasks(na, workers, [&](std::size_t s) {
            auto& gradient = gradients[omp_thread_index()];
            for (std::size_t p = 0; p < na; ++p) {
                if (aux_owner[p] == aux_owner[s]) continue;
                for (std::size_t g = 0; g < ng; ++g) {
                    const Complex value = kernels[g] * metric_weight(p, s)
                        * std::conj(auxiliary(p, g)) * auxiliary(s, g);
                    const auto vector = momentum(g);
                    for (int axis = 0; axis < 3; ++axis) {
                        const double derivative = std::real(Complex(0, vector[axis]) * value);
                        gradient(aux_owner[p], axis) += derivative;
                        gradient(aux_owner[s], axis) -= derivative;
                    }
                }
            }
        });
        gdf_parallel_tasks(tasks, workers, [&](std::size_t task) {
            const auto k = task / pairs, pair = task % pairs;
            const auto mu = pair / no, nu = pair % no;
            const auto bra = aopair_ft_detail::ao(orbital, mu);
            const auto ket = aopair_ft_detail::ao(orbital, nu);
            const auto box = aopair_ft_detail::image_box(bra, ket, inverse, pair_cutoff);
            std::array<Complex, 7 * vector_block> values{};
            aopair_ft_detail::walk_images(bra, ket, system.lattice, box, pair_cutoff * pair_cutoff,
                [&](const std::array<std::int64_t, 3>&, const Eigen::Vector3d& translation,
                    const Eigen::Vector3d& separation, double squared) {
                    for (std::size_t g = 0; g < ng; ++g) {
                        if (kernels[g] == 0.0) continue;
                        const auto leaf = aopair_ft_detail::value_and_center_derivatives(
                            bra, ket, momentum(g), ket_kpoints.row(k).transpose(),
                            translation, separation, squared);
                        for (std::size_t component = 0; component < 7; ++component)
                            values[7 * g + component] += leaf[component];
                    }
                });
            auto& gradient = gradients[omp_thread_index()];
            for (std::size_t g = 0; g < ng; ++g) {
                const auto vector = momentum(g);
                Complex contracted{};
                for (std::size_t p = 0; p < na; ++p) {
                    const Complex coefficient = kernels[g] * std::conj(auxiliary(p, g))
                        * three_center_weight.data[(k * na + p) * pairs + pair];
                    contracted += coefficient;
                    for (int axis = 0; axis < 3; ++axis)
                        gradient(aux_owner[p], axis) += std::real(
                            Complex(0, vector[axis]) * coefficient * values[7 * g]);
                }
                if (plane_wave_projection)
                    contracted += std::sqrt(kernels[g]) * factor_weight.data[
                        (k * factor_weight.n_aux + first_g + g) * pairs + pair];
                for (int axis = 0; axis < 3; ++axis) {
                    gradient(orb_owner[mu], axis) += std::real(contracted * values[7 * g + 1 + axis]);
                    gradient(orb_owner[nu], axis) += std::real(contracted * values[7 * g + 4 + axis]);
                }
            }
        });
    };
    if (!plane_wave_projection && kernel.zero_transfer)
        contract_panel(at_zero, true, 0);
    for (Eigen::Index first = 0; first < vectors.rows(); first += vector_block) {
        const auto count = static_cast<std::size_t>(
            std::min<Eigen::Index>(vector_block, vectors.rows() - first));
        const auto* ptr = vectors.data() + 3 * first;
        contract_panel({ptr, ptr+1, ptr+2, count, 3, 3, 3}, false,
            static_cast<std::size_t>(first));
    }
    for (const auto& gradient : gradients) result += gradient;
    if (!result.allFinite())
        throw std::invalid_argument("RSGDF derivative produced a nonfinite atom gradient");
    return result;
}

Eigen::MatrixXd compute_gdf_range_separated_gradient_weighted(
    const BasisSet& orbital, const BasisSet& aux,
    const PeriodicSystem& system, const Eigen::Vector3d& q,
    const Eigen::MatrixXd& ket_kpoints,
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& vectors,
    double omega, double pair_cutoff, double auxiliary_cutoff,
    const Eigen::Ref<const Eigen::MatrixXcd>& metric_weight,
    GDFShortRangeWeightView three_center_weight,
    std::size_t output_byte_cap, std::size_t transient_byte_cap,
    std::size_t image_candidate_cap, double integral_screen_error) {
    return compute_gdf_coulomb_gradient_weighted(
        orbital, aux, system, q, ket_kpoints, vectors, omega, pair_cutoff,
        auxiliary_cutoff, metric_weight, three_center_weight, output_byte_cap,
        transient_byte_cap, image_candidate_cap, integral_screen_error, false, {});
}

Eigen::MatrixXd compute_gdf_plane_wave_projection_gradient_weighted(
    const BasisSet& orbital, const BasisSet& aux,
    const PeriodicSystem& system, const Eigen::Vector3d& q,
    const Eigen::MatrixXd& ket_kpoints,
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& vectors,
    double pair_cutoff,
    const Eigen::Ref<const Eigen::MatrixXcd>& metric_weight,
    GDFShortRangeWeightView three_center_weight,
    GDFShortRangeWeightView factor_weight,
    std::size_t output_byte_cap, std::size_t transient_byte_cap,
    std::size_t image_candidate_cap) {
    return compute_gdf_coulomb_gradient_weighted(
        orbital, aux, system, q, ket_kpoints, vectors, 1.0, pair_cutoff,
        pair_cutoff, metric_weight, three_center_weight, output_byte_cap,
        transient_byte_cap, image_candidate_cap, 0.0, true, factor_weight);
}

// MANTA replaces only the primitive evaluator. These entry points intentionally
// live beside the existing GDF implementation so both use the exact same
// MetricImageDomain/ThreeCenterImageDomain and optional screening predicates.
namespace manta {
namespace {
struct FittingDigest {
    vibeqc::detail::Sha256 hash;
    void word(Count v) { std::array<std::uint8_t,8> b{};for(int i=0;i<8;++i)b[i]=v>>(56-8*i);hash.update(b.data(),8); }
    void text(const std::string& s) { word(s.size());hash.update(reinterpret_cast<const std::uint8_t*>(s.data()),s.size()); }
    void real(double v) { if(v==0)v=0.;Count bits;std::memcpy(&bits,&v,8);word(bits); }
    template<class M> void matrix(const M& m) { word(m.rows());word(m.cols());for(Eigen::Index i=0;i<m.rows();++i)for(Eigen::Index j=0;j<m.cols();++j)real(m(i,j)); }
};
void fitting_caps(const PeriodicGDFShortRangePlan& p,const ResourceBudget& b) {
    using namespace admission;
    cap(p.modeled_peak_bytes,b.memory_bytes,"MANTA periodic fitting memory budget exceeded");
    cap(p.shell_calls,b.shell_calls,"MANTA periodic fitting shell-call budget exceeded");
    cap(p.primitive_products,b.primitive_quartets,"MANTA periodic fitting primitive-product budget exceeded");
    cap(p.scalar_terms,b.scalar_terms,"MANTA periodic fitting scalar-work budget exceeded");
    cap(p.kernel_terms,b.kernel_terms,"MANTA periodic fitting kernel-work budget exceeded");
}
void fitting_add(Complex v,Complex& total,Complex& error) {
    if(!std::isfinite(v.real()) || !std::isfinite(v.imag()))
        throw std::overflow_error("MANTA periodic fitting contribution is nonfinite");
    const auto y=v-error,next=total+y;error=(next-total)-y;total=next;
    if(!std::isfinite(total.real()) || !std::isfinite(total.imag()))
        throw std::overflow_error("MANTA periodic fitting sum is nonfinite");
}
Complex fitting_phase(double angle,PeriodicGDFShortRangeReceipt& r) {
    admission::require_finite(angle);++r.phase_products;return std::exp(Complex(0.,angle));
}
std::string fitting_identity(bool three,const PeriodicSystem& system,const Eigen::Vector3d& q,
    const md::Matrix& k,const PeriodicGDFShortRangeOptions& o,const PeriodicGDFShortRangeReceipt& r) {
    FittingDigest h;h.text("manta.periodic-gdf-short-range.v1");h.text(r.provider_version);h.text(r.normalization);
    h.text(r.convention);h.word(three);h.text(r.auxiliary_basis_identity);h.matrix(system.lattice);h.matrix(q);
    h.real(o.omega);h.real(o.auxiliary_cutoff_bohr);h.real(o.integral_screen_error);
    if(three) { h.text(r.orbital_basis_identity);h.matrix(k);h.real(o.pair_cutoff_bohr); }
    return h.hash.finish_hex();
}
}
PeriodicGDFShortRangePlan plan_periodic_gdf_short_range(const BasisSet& orbital,const BasisSet& aux,
    const PeriodicSystem& system,Count nk,const PeriodicGDFShortRangeOptions& o,const ResourceBudget& budget) {
    using namespace admission;
    margin(budget);validate(system,Eigen::Vector3d::Zero(),o.omega,o.pair_cutoff_bohr);
    validate(system,Eigen::Vector3d::Zero(),o.omega,o.auxiliary_cutoff_bohr);
    if(!nk) throw std::invalid_argument("MANTA periodic fitting requires nonempty ket momenta");
    cap(nk,std::numeric_limits<Eigen::Index>::max()/3,"MANTA periodic fitting momenta are not addressable");
    const MetricImageDomain metric(system,o.auxiliary_cutoff_bohr,o.omega,o.image_candidate_cap,o.integral_screen_error);
    const ThreeCenterImageDomain three(system,o.pair_cutoff_bohr,o.auxiliary_cutoff_bohr,o.omega,o.image_candidate_cap,o.integral_screen_error);
    PeriodicGDFShortRangePlan p;p.n_aux=aux.nbasis();p.n_orb=orbital.nbasis();p.momenta=nk;p.compute_metric=o.compute_metric;
    p.metric_image_candidates=o.compute_metric?metric.images.count_bound:0;
    p.pair_image_candidates=three.pair_images.count_bound;p.auxiliary_image_candidates=three.aux_images.count_bound;
    const Count mi=p.metric_image_candidates,ti=mul(p.pair_image_candidates,p.auxiliary_image_candidates);
    const Count me=o.compute_metric?square(p.n_aux):0,te=mul(nk,mul(p.n_aux,square(p.n_orb)));
    for(auto e:{me,te}) cap(e,std::numeric_limits<Eigen::Index>::max(),"MANTA periodic fitting output is not addressable");
    p.output_bytes=mul(16,add(me,te));
    // Native outputs/compensation, one exported copy of each output, binding
    // copies, receipt/digest owners and conservative spare output ownership.
    p.driver_live_bytes=add(65536,add(mul(4,p.output_bytes),mul(64,add(nk,1))));
    p.modeled_peak_bytes=add(p.driver_live_bytes,add(budget.other_live_bytes,budget.backend_margin_bytes));
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA periodic fitting output memory budget exceeded");
    auto leaf=budget;leaf.other_live_bytes=0;
    leaf.memory_bytes=leaf.shell_calls=leaf.primitive_quartets=leaf.scalar_terms=leaf.kernel_terms=std::numeric_limits<Count>::max();
    auto request=fitting_request(3);request.kernel="erfc";request.omega=o.omega;
    p.three_center=plan_fitting({&aux,&orbital,&orbital},request,leaf);
    if(o.compute_metric) { request.centers=2;p.metric=plan_fitting({&aux,&aux},request,leaf); }
    p.metric_shell_calls=mul(mi,p.metric.shell_calls);p.three_center_shell_calls=mul(ti,p.three_center.shell_calls);
    p.shell_calls=add(p.metric_shell_calls,p.three_center_shell_calls);
    p.primitive_products=add(mul(mi,p.metric.primitive_products),mul(ti,p.three_center.primitive_products));
    const Count pair_calls=mul(p.pair_image_candidates,p.three_center.shell_calls);
    const Count phases=add(p.shell_calls,mul(nk,pair_calls));
    const Count contractions=add(mul(mi,p.metric.integral_elements),
        mul(p.three_center.integral_elements,add(ti,mul(nk,p.pair_image_candidates))));
    // Candidate distance checks, phase construction and compensated digestion
    // are charged even where the shared geometric/screening selector skips.
    p.scalar_terms=add(mul(8,add(me,te)),mul(128,add(p.shell_calls,add(phases,contractions))));
    p.kernel_terms=add(mul(mi,p.metric.kernel_terms),add(mul(ti,p.three_center.kernel_terms),mul(8,p.scalar_terms)));
    p.modeled_peak_bytes=add(p.modeled_peak_bytes,add(std::max(p.metric.modeled_peak_bytes,p.three_center.modeled_peak_bytes),
        mul(32,p.three_center.tile_elements)));
    fitting_caps(p,budget);validate_centers(orbital,system);validate_centers(aux,system);return p;
}
PeriodicGDFShortRangeResult compute_periodic_gdf_short_range(const BasisSet& orbital,const BasisSet& aux,
    const PeriodicSystem& system,const Eigen::Vector3d& q,const md::Matrix& k,
    const PeriodicGDFShortRangeOptions& o,const ResourceBudget& budget) {
    using namespace admission;
    if(k.cols()!=3) throw std::invalid_argument("MANTA periodic fitting momenta require three components");
    PeriodicGDFShortRangeResult result;result.plan=plan_periodic_gdf_short_range(orbital,aux,system,k.rows(),o,budget);
    const auto& p=result.plan;auto& r=result.receipt;
    if(!q.allFinite() || !k.allFinite()) throw std::invalid_argument("MANTA periodic fitting momenta must be finite");
    r.omega=o.omega;r.pair_cutoff_bohr=o.pair_cutoff_bohr;r.auxiliary_cutoff_bohr=o.auxiliary_cutoff_bohr;
    r.integral_screen_error=o.integral_screen_error;r.modeled_peak_bytes=p.modeled_peak_bytes;
    r.orbital_basis_identity=auxiliary_basis_content_identity_sha256(orbital);
    r.auxiliary_basis_identity=auxiliary_basis_content_identity_sha256(aux);
    if(o.compute_metric) r.metric_source_identity=fitting_identity(false,system,q,k,o,r);
    r.three_center_source_identity=fitting_identity(true,system,q,k,o,r);
    FittingDigest combined;combined.text("manta.periodic-gdf-short-range.joint.v1");
    combined.text(r.metric_source_identity);combined.text(r.three_center_source_identity);r.source_identity=combined.hash.finish_hex();
    const auto& a=aux.libint();const auto& b=orbital.libint();const auto ai=a.shell2bf(),bi=b.shell2bf();
    const Count na=p.n_aux,no=p.n_orb,nk=p.momenta;
    const MetricImageDomain metric(system,o.auxiliary_cutoff_bohr,o.omega,o.image_candidate_cap,o.integral_screen_error);
    const ThreeCenterImageDomain three(system,o.pair_cutoff_bohr,o.auxiliary_cutoff_bohr,o.omega,o.image_candidate_cap,o.integral_screen_error);
    if(o.compute_metric) {
        result.metric_values.assign(square(na),{});std::vector<Complex> errors(square(na));
        for(Count is=0;is<a.size();++is) for(Count js=0;js<a.size();++js)
            metric.each(a[is],a[js],[&](const Eigen::Vector3d& t) {
                const auto ket=shifted(a[js],t);
                const auto values=md::shell_coulomb({&a[is],nullptr,&ket,nullptr},md::RadialKernel::Erfc,o.omega);
                const auto phase=fitting_phase(q.dot(t),r);
                ++r.metric_shell_calls;r.primitive_products+=a[is].nprim()*a[js].nprim();
                for(Count i=0;i<a[is].size();++i) for(Count j=0;j<a[js].size();++j) {
                    const Count out=(ai[is]+i)*na+ai[js]+j;
                    fitting_add(phase*values(i,j),result.metric_values[out],errors[out]);++r.phase_accumulations;
                }
            });
    }
    const Count elements=mul(nk,mul(na,square(no)));
    result.three_center_values.assign(elements,{});std::vector<Complex> errors(elements);
    for(Count ip=0;ip<a.size();++ip) for(Count im=0;im<b.size();++im) for(Count in=0;in<b.size();++in) {
        const Count np=a[ip].size(),nm=b[im].size(),nn=b[in].size(),tile=mul(np,mul(nm,nn));
        std::vector<Complex> contracted(tile),correction(tile);auto ket=b[in];
        three.each(a[ip],b[im],b[in],[&](const Eigen::Vector3d& t) {
            ket=shifted(b[in],t);std::fill(contracted.begin(),contracted.end(),Complex{});
            std::fill(correction.begin(),correction.end(),Complex{});++r.pair_shell_images;
        },[&](const Eigen::Vector3d&,const Eigen::Vector3d& t) {
            const auto bra=shifted(a[ip],t);
            const auto values=md::shell_coulomb({&bra,nullptr,&b[im],&ket},md::RadialKernel::Erfc,o.omega);
            const auto phase=fitting_phase(-q.dot(t),r);
            ++r.three_center_shell_calls;r.primitive_products+=a[ip].nprim()*b[im].nprim()*b[in].nprim();
            for(Count i=0;i<tile;++i) { fitting_add(phase*values.data()[i],contracted[i],correction[i]);++r.phase_accumulations; }
        },[&](const Eigen::Vector3d& t) {
            for(Count ik=0;ik<nk;++ik) {
                const auto phase=fitting_phase(k.row(ik).dot(t),r);
                for(Count i=0;i<np;++i) for(Count j=0;j<nm;++j) for(Count l=0;l<nn;++l) {
                    const Count out=((ik*na+ai[ip]+i)*no+bi[im]+j)*no+bi[in]+l;
                    fitting_add(phase*contracted[(i*nm+j)*nn+l],result.three_center_values[out],errors[out]);++r.phase_accumulations;
                }
            }
        });
    }
    cap(r.metric_shell_calls,p.metric_shell_calls,"MANTA periodic metric executed-work bound exceeded");
    cap(r.three_center_shell_calls,p.three_center_shell_calls,"MANTA periodic coupling executed-work bound exceeded");
    cap(r.primitive_products,p.primitive_products,"MANTA periodic fitting executed primitive bound exceeded");
    return result;
}
namespace {
void gdf_caps(const PeriodicGDFPlan& p,const ResourceBudget& b) {
    using namespace admission;
    cap(p.modeled_peak_bytes,b.memory_bytes,"MANTA GDF memory budget exceeded");
    cap(p.shell_calls,b.shell_calls,"MANTA GDF shell-call budget exceeded");
    cap(p.primitive_products,b.primitive_quartets,"MANTA GDF primitive-product budget exceeded");
    cap(p.scalar_terms,b.scalar_terms,"MANTA GDF scalar-work budget exceeded");
    cap(p.kernel_terms,b.kernel_terms,"MANTA GDF kernel-work budget exceeded");
}
ResourceBudget gdf_leaf_budget(ResourceBudget b) {
    b.memory_bytes=b.shell_calls=b.primitive_quartets=b.scalar_terms=b.kernel_terms=std::numeric_limits<Count>::max();
    b.other_live_bytes=0;return b;
}
std::string gdf_identity(bool metric,const md::Matrix& vectors,const PeriodicGDFReceipt& r) {
    FittingDigest h;h.text("manta.periodic-gdf.v1");h.text(r.provider_version);h.text(r.normalization);
    h.text(r.algorithm);h.text(r.kernel);h.text(r.convention);h.word(metric);
    h.text(metric?r.short_range.metric_source_identity:r.short_range.three_center_source_identity);
    h.matrix(vectors);
    if(r.kernel!="coulomb") h.real(r.physical_omega);
    return h.hash.finish_hex();
}
void validate_physical_gdf(const PeriodicGDFKernelOptions& p,const PeriodicSystem& system) {
    if(p.kernel!="coulomb" && p.kernel!="erfc" && p.kernel!="erf")
        throw std::invalid_argument("MANTA GDF physical kernel must be coulomb, erfc or erf");
    const double squared=p.omega*p.omega;
    if(!std::isfinite(p.omega) || (p.kernel=="coulomb" ? p.omega!=0. :
        p.omega<=0. || !std::isfinite(squared) || squared==0.))
        throw std::invalid_argument("MANTA GDF physical omega is zero for Coulomb, positive and representable for erf/erfc");
    if(p.kernel=="erfc" && !std::isfinite((std::acos(-1.)/std::abs(system.lattice.determinant()))/squared))
        throw std::invalid_argument("MANTA GDF physical finite-zero coefficient is not representable");
}
// Fourier decomposition of erfc(omega*r)/r = erfc(eta*r)/r
// + erf(eta*r)/r - erf(omega*r)/r. Eta is only the numerical split.
// Same Gaussian Fourier convention as Sun et al., JCP 147,164119 (2017).
double physical_gdf_weight(const RangeSeparatedCoulombKernel& numerical,
    const PeriodicGDFKernelOptions& p,double p2,bool zero) {
    if(p.kernel=="coulomb") return zero?-numerical.short_range_zero_mode:numerical.long_range(p2);
    if(zero) return p.kernel=="erf"?0.:
        .25*numerical.reciprocal_prefactor/(p.omega*p.omega)-numerical.short_range_zero_mode;
    if(RangeSeparatedCoulombKernel::singular(p2)) return 0.;
    const double physical=-.25*p2/(p.omega*p.omega);
    if(p.kernel=="erf") return numerical.bare(p2)*std::exp(physical);
    const double split=-.25*p2/numerical.omega_squared;
    if(split==physical) return 0.;
    // Avoid cancellation when the physical and numerical ranges are close.
    const double difference=split>=physical?std::exp(split)*(-std::expm1(physical-split)):
        std::exp(physical)*std::expm1(split-physical);
    return numerical.bare(p2)*difference;
}
void describe_physical_gdf(PeriodicGDFReceipt& r,const PeriodicGDFKernelOptions& p,bool gamma) {
    r.kernel=p.kernel;r.physical_omega=p.omega;r.zero_transfer=gamma;r.real_short_range=p.kernel!="erf";
    r.finite_zero_removed=gamma && r.real_short_range;
    r.physical_zero_included=gamma && p.kernel=="erfc";
    if(p.kernel=="erfc") r.convention="physical-erfc;finite-zero-included-once;numerical-SR-zero-removed-once;no-exchange-correction";
    if(p.kernel=="erf") r.convention="physical-erf;reciprocal-only;zero-pole-omitted;no-exchange-correction";
}
PeriodicGDFShortRangeReceipt describe_gdf_short_range(const BasisSet& orbital,const BasisSet& aux,
    const PeriodicSystem& system,const Eigen::Vector3d& q,const md::Matrix& k,
    const PeriodicGDFShortRangeOptions& o,Count peak,bool real_short_range) {
    PeriodicGDFShortRangeReceipt sr;
    sr.orbital_basis_identity=auxiliary_basis_content_identity_sha256(orbital);
    sr.auxiliary_basis_identity=auxiliary_basis_content_identity_sha256(aux);
    auto effective=o;
    if(!real_short_range) {
        sr.algorithm="not-evaluated";sr.kernel="none";sr.convention="no-real-space-short-range-source";
        effective.omega=effective.auxiliary_cutoff_bohr=effective.integral_screen_error=0.;
    }
    sr.omega=effective.omega;sr.pair_cutoff_bohr=o.pair_cutoff_bohr;sr.auxiliary_cutoff_bohr=effective.auxiliary_cutoff_bohr;
    sr.integral_screen_error=effective.integral_screen_error;sr.modeled_peak_bytes=peak;
    if(o.compute_metric) sr.metric_source_identity=fitting_identity(false,system,q,k,effective,sr);
    sr.three_center_source_identity=fitting_identity(true,system,q,k,effective,sr);
    FittingDigest sj;sj.text("manta.periodic-gdf-short-range.joint.v1");sj.text(sr.metric_source_identity);
    sj.text(sr.three_center_source_identity);sr.source_identity=sj.hash.finish_hex();return sr;
}
}

PeriodicGDFPlan plan_periodic_gdf(const BasisSet& orbital,const BasisSet& aux,
    const PeriodicSystem& system,Count nv,Count nk,const PeriodicGDFShortRangeOptions& o,const ResourceBudget& budget,
    const PeriodicGDFKernelOptions& physical) {
    using namespace admission;
    margin(budget);validate_physical_gdf(physical,system);
    if(!nv) throw std::invalid_argument("MANTA GDF requires a nonempty reciprocal domain");
    cap(nv,std::numeric_limits<Eigen::Index>::max()/3,"MANTA GDF vectors are not addressable");
    const auto leaf=gdf_leaf_budget(budget);
    PeriodicGDFPlan p;p.vectors=nv;p.momenta=nk;p.n_aux=aux.nbasis();p.n_orb=orbital.nbasis();p.compute_metric=o.compute_metric;
    // A count-only plan always admits one extra zero panel. Whether it is
    // needed is decided by the shared q convention only after input admission.
    p.panel_vectors=std::min<Count>(16,nv);p.panel_count=add(nv/16+(nv%16!=0),1);
    p.short_range=plan_periodic_gdf_short_range(orbital,aux,system,nk,o,leaf);
    p.output_bytes=p.short_range.output_bytes;
    p.modeled_peak_bytes=add(p.short_range.modeled_peak_bytes,
        add(budget.other_live_bytes,add(65536,mul(48,nv))));
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA GDF output/input memory budget exceeded");
    p.auxiliary_fourier=plan_fourier(aux,nullptr,p.panel_vectors,leaf);
    p.ao_pair_fourier=plan_fourier(orbital,&orbital,p.panel_vectors,leaf);
    const Count images=p.short_range.pair_image_candidates;
    const Count fs=add(p.auxiliary_fourier.shell_calls,mul(images,p.ao_pair_fourier.shell_calls));
    const Count fp=add(p.auxiliary_fourier.primitive_pairs,mul(images,p.ao_pair_fourier.primitive_pairs));
    p.shell_calls=add(p.short_range.shell_calls,mul(p.panel_count,fs));
    p.primitive_products=add(p.short_range.primitive_products,mul(p.panel_count,fp));
    p.vector_primitive_products=mul(add(nv,1),fp);
    const Count pair_elements=square(p.n_orb),metric_elements=o.compute_metric?square(p.n_aux):0;
    const Count phase=mul(p.panel_count,mul(nk,mul(images,p.ao_pair_fourier.shell_calls)));
    const Count fold=mul(add(nv,1),mul(nk,mul(images,pair_elements)));
    const Count digest=mul(add(nv,1),add(metric_elements,mul(nk,mul(p.n_aux,pair_elements))));
    p.scalar_terms=add(p.short_range.scalar_terms,mul(128,add(p.shell_calls,add(phase,add(fold,digest)))));
    p.kernel_terms=add(p.short_range.kernel_terms,add(mul(p.panel_count,
        add(p.auxiliary_fourier.kernel_terms,mul(images,p.ao_pair_fourier.kernel_terms))),mul(8,p.scalar_terms)));
    // One auxiliary panel plus one shell-pair panel folded into all k rows.
    // No global AO-square by G or k/G/auxiliary/AO tensor is allocated.
    const Count panels=add(mul(16,mul(p.n_aux,p.panel_vectors)),
        add(mul(32,mul(nk,p.ao_pair_fourier.tile_elements)),mul(128,add(nk,p.panel_vectors))));
    p.modeled_peak_bytes=add(p.modeled_peak_bytes,add(panels,
        std::max(p.auxiliary_fourier.modeled_peak_bytes,p.ao_pair_fourier.modeled_peak_bytes)));
    gdf_caps(p,budget);return p;
}

namespace {
RangeSeparatedCoulombKernel validate_gdf_domain(const PeriodicSystem& system,const Eigen::Vector3d& q,
    const md::Matrix& k,const md::Matrix& vectors,const PeriodicGDFShortRangeOptions& o) {
    if(k.cols()!=3 || vectors.cols()!=3)
        throw std::invalid_argument("MANTA GDF momenta and reciprocal vectors require three components");
    if(!q.allFinite() || !k.allFinite() || !vectors.allFinite())
        throw std::invalid_argument("MANTA GDF momenta and reciprocal vectors must be finite");
    const double volume=std::abs(system.lattice.determinant());
    if(!std::isfinite(volume) || volume<=0.)
        throw std::invalid_argument("MANTA GDF cell volume is not representable");
    const RangeSeparatedCoulombKernel kernel(system,q,o.omega);
    if(!std::isfinite(kernel.short_range_zero_mode) || !std::isfinite(kernel.reciprocal_prefactor)
        || !std::isfinite(kernel.omega_squared) || kernel.omega_squared==0.)
        throw std::invalid_argument("MANTA GDF Coulomb kernel is not representable");
    const Eigen::Vector3d fractional_q=system.lattice.transpose()*q/(2.*std::acos(-1.));
    for(Eigen::Index g=0;g<vectors.rows();++g) {
        const Eigen::Vector3d fractional_p=system.lattice.transpose()*vectors.row(g).transpose()/(2.*std::acos(-1.));
        const Eigen::Vector3d index=fractional_p-fractional_q;
        // Existing GDF domains quantize transfer labels to 12 fractional
        // decimals. Admit that half-unit plus arithmetic roundoff, without
        // a loose tolerance that could lose a resolvable small-q pole.
        const double tolerance=5e-13+128.*std::numeric_limits<double>::epsilon()*(1.+std::max(
            fractional_p.cwiseAbs().maxCoeff(),fractional_q.cwiseAbs().maxCoeff()));
        if(!index.allFinite() || !std::isfinite(tolerance)
            || (index-index.array().round().matrix()).cwiseAbs().maxCoeff()>tolerance)
            throw std::invalid_argument("MANTA GDF reciprocal vectors must belong to the G+q lattice");
    }
    return kernel;
}
}
PeriodicGDFResult compute_periodic_gdf(const BasisSet& orbital,const BasisSet& aux,
    const PeriodicSystem& system,const Eigen::Vector3d& q,const md::Matrix& k,const md::Matrix& vectors,
    const PeriodicGDFShortRangeOptions& o,const ResourceBudget& budget,const PeriodicGDFKernelOptions& physical) {
    using namespace admission;
    if(k.cols()!=3 || vectors.cols()!=3)
        throw std::invalid_argument("MANTA GDF momenta and reciprocal vectors require three components");
    PeriodicGDFResult result;result.plan=plan_periodic_gdf(orbital,aux,system,vectors.rows(),k.rows(),o,budget,physical);
    const auto& p=result.plan;auto& r=result.receipt;
    const auto kernel=validate_gdf_domain(system,q,k,vectors,o);
    describe_physical_gdf(r,physical,kernel.zero_transfer);
    PeriodicGDFShortRangeResult sr;
    if(r.real_short_range) sr=compute_periodic_gdf_short_range(orbital,aux,system,q,k,o,gdf_leaf_budget(budget));
    else {
        sr.receipt=describe_gdf_short_range(orbital,aux,system,q,k,o,p.modeled_peak_bytes,false);
        sr.metric_values.resize(o.compute_metric?square(p.n_aux):0);
        sr.three_center_values.resize(mul(p.momenta,mul(p.n_aux,square(p.n_orb))));
    }
    result.metric_values=std::move(sr.metric_values);result.three_center_values=std::move(sr.three_center_values);
    r.short_range=std::move(sr.receipt);r.modeled_peak_bytes=p.modeled_peak_bytes;
    r.shell_calls=r.short_range.metric_shell_calls+r.short_range.three_center_shell_calls;
    r.primitive_products=r.short_range.primitive_products;
    r.phase_products=r.short_range.phase_products;r.phase_accumulations=r.short_range.phase_accumulations;
    if(o.compute_metric) r.metric_source_identity=gdf_identity(true,vectors,r);
    r.three_center_source_identity=gdf_identity(false,vectors,r);
    FittingDigest joint;joint.text("manta.periodic-gdf.joint.v1");joint.text(r.metric_source_identity);
    joint.text(r.three_center_source_identity);r.source_identity=joint.hash.finish_hex();
    std::vector<Complex> metric_error(result.metric_values.size()),three_error(result.three_center_values.size());
    const auto& a=aux.libint();const auto& b=orbital.libint();const auto ai=a.shell2bf(),bi=b.shell2bf();
    const Count na=p.n_aux,no=p.n_orb,nk=p.momenta;
    const ImageBox images(system,o.pair_cutoff_bohr,o.image_candidate_cap);
    auto contract_panel=[&](const md::Matrix& v,bool zero) {
        const Count ng=v.rows();md::ComplexMatrix aft(na,ng);
        std::vector<double> weights(ng);
        for(Count g=0;g<ng;++g) {
            const double squared=v.row(g).squaredNorm();
            require_finite(squared);
            weights[g]=physical_gdf_weight(kernel,physical,squared,zero);
            require_finite(weights[g]);
            if(!zero && !RangeSeparatedCoulombKernel::singular(squared)) ++r.reciprocal_vectors;
        }
        for(Count is=0;is<a.size();++is) {
            const auto tile=md::shell_fourier_panel(&a[is],nullptr,v);
            if(!tile.allFinite()) throw std::overflow_error("MANTA GDF auxiliary Fourier tile is nonfinite");
            aft.middleRows(ai[is],a[is].size())=tile;
            ++r.auxiliary_fourier_shell_calls;++r.shell_calls;r.primitive_products+=a[is].nprim();
            r.vector_primitive_products+=ng*a[is].nprim();
        }
        if(o.compute_metric) for(Count is=0;is<na;++is) for(Count js=0;js<na;++js) for(Count g=0;g<ng;++g) {
            fitting_add(weights[g]*std::conj(aft(is,g))*aft(js,g),result.metric_values[is*na+js],metric_error[is*na+js]);
            ++r.contraction_terms;
        }
        for(Count im=0;im<b.size();++im) for(Count in=0;in<b.size();++in) {
            const Count nm=b[im].size(),nn=b[in].size(),np=mul(nm,nn);
            std::vector<Complex> folded(mul(nk,mul(np,ng))),error(folded.size());
            images.each(origin(b[im])-origin(b[in]),[&](const Eigen::Vector3d& t) {
                if(!pair_in_range(b[im],b[in],t,o.pair_cutoff_bohr)) return;
                const auto ket=shifted(b[in],t);const auto tile=md::shell_fourier_panel(&b[im],&ket,v);
                if(!tile.allFinite()) throw std::overflow_error("MANTA GDF pair Fourier tile is nonfinite");
                ++r.pair_fourier_shell_calls;++r.shell_calls;
                const Count primitives=b[im].nprim()*b[in].nprim();r.primitive_products+=primitives;
                r.vector_primitive_products+=ng*primitives;
                for(Count ik=0;ik<nk;++ik) {
                    const double angle=k.row(ik).dot(t);require_finite(angle);
                    const Complex phase=std::exp(Complex(0.,angle));++r.phase_products;
                    for(Count pair=0;pair<np;++pair) for(Count g=0;g<ng;++g) {
                        const Count out=(ik*np+pair)*ng+g;
                        fitting_add(phase*tile(pair,g),folded[out],error[out]);++r.phase_accumulations;
                    }
                }
            });
            for(Count ik=0;ik<nk;++ik) for(Count ip=0;ip<na;++ip) for(Count imf=0;imf<nm;++imf)
                for(Count inf=0;inf<nn;++inf) for(Count g=0;g<ng;++g) {
                    const Count pair=imf*nn+inf,out=((ik*na+ip)*no+bi[im]+imf)*no+bi[in]+inf;
                    fitting_add(weights[g]*std::conj(aft(ip,g))*folded[(ik*np+pair)*ng+g],
                        result.three_center_values[out],three_error[out]);++r.contraction_terms;
                }
        }
    };
    if(kernel.zero_transfer && r.real_short_range) {
        contract_panel(md::Matrix::Zero(1,3),true);
    }
    for(Count first=0;first<p.vectors;first+=p.panel_vectors) {
        const md::Matrix panel=vectors.middleRows(first,std::min(p.panel_vectors,p.vectors-first));
        contract_panel(panel,false);
    }
    cap(r.shell_calls,p.shell_calls,"MANTA GDF executed shell-call bound exceeded");
    cap(r.primitive_products,p.primitive_products,"MANTA GDF executed primitive bound exceeded");
    cap(r.vector_primitive_products,p.vector_primitive_products,"MANTA GDF executed Fourier-work bound exceeded");
    return result;
}
namespace {
void gdf_tile_caps(const PeriodicGDFTilesPlan& p,const ResourceBudget& b) {
    using namespace admission;
    cap(p.modeled_peak_bytes,b.memory_bytes,"MANTA GDF tiles memory budget exceeded");
    cap(p.shell_calls,b.shell_calls,"MANTA GDF tiles shell-call budget exceeded");
    cap(p.primitive_products,b.primitive_quartets,"MANTA GDF tiles primitive-product budget exceeded");
    cap(p.scalar_terms,b.scalar_terms,"MANTA GDF tiles scalar-work budget exceeded");
    cap(p.kernel_terms,b.kernel_terms,"MANTA GDF tiles kernel-work budget exceeded");
}
void finish_gdf_tile(PeriodicGDFTile& tile) {
    auto& r=tile.receipt;
    r.shell_calls+=r.short_range.metric_shell_calls+r.short_range.three_center_shell_calls;
    r.primitive_products+=r.short_range.primitive_products;
    r.phase_products+=r.short_range.phase_products;r.phase_accumulations+=r.short_range.phase_accumulations;
}
}
PeriodicGDFTilesPlan plan_periodic_gdf_tiles(const BasisSet& orbital,const BasisSet& aux,
    const PeriodicSystem& system,Count nv,Count nk,const PeriodicGDFShortRangeOptions& o,const ResourceBudget& budget,
    const PeriodicGDFKernelOptions& physical) {
    using namespace admission;
    margin(budget);validate_physical_gdf(physical,system);
    if(!nv) throw std::invalid_argument("MANTA GDF tiles require a nonempty reciprocal domain");
    cap(nv,std::numeric_limits<Eigen::Index>::max()/3,"MANTA GDF tile vectors are not addressable");
    const auto leaf=gdf_leaf_budget(budget);
    const auto sr=plan_periodic_gdf_short_range(orbital,aux,system,nk,o,leaf);
    const auto a=census(aux,7),b=census(orbital,6);
    PeriodicGDFTilesPlan p;p.n_aux=aux.nbasis();p.n_orb=orbital.nbasis();p.momenta=nk;p.vectors=nv;
    p.panel_vectors=std::min<Count>(16,nv);p.compute_metric=o.compute_metric;p.external_live_bytes=budget.other_live_bytes;
    const Count panels=add(nv/16+(nv%16!=0),1),images=sr.pair_image_candidates;
    const auto af=plan_fourier(aux,nullptr,p.panel_vectors,leaf),bf=plan_fourier(orbital,&orbital,p.panel_vectors,leaf);
    const Count pair_shells=square(orbital.nshells()),aux_shells=aux.nshells(),pairs=square(p.n_orb);
    p.maximum_tile_elements=mul(nk,sr.three_center.tile_elements);
    cap(p.maximum_tile_elements,std::numeric_limits<Eigen::Index>::max(),"MANTA GDF shell tile is not addressable");
    // Full basis snapshots and numerical import/binding copies. Only the
    // lattice/dimension are retained from PeriodicSystem, never its symmetry
    // metadata or nuclear state. Those remain with the consuming method.
    p.retained_bytes=add(65536,add(mul(4,add(a.bytes,b.bytes)),mul(96,add(nv,add(nk,1)))));
    const Count common=add(p.retained_bytes,add(budget.other_live_bytes,budget.backend_margin_bytes));
    p.metric_peak_bytes=o.compute_metric?add(common,add(mul(48,square(p.n_aux)),
        add(std::max(sr.metric.modeled_peak_bytes,af.modeled_peak_bytes),mul(32,mul(p.n_aux,p.panel_vectors))))):0;
    p.coupling_peak_bytes=add(common,add(add(mul(48,p.maximum_tile_elements),mul(32,sr.three_center.tile_elements)),
        add(std::max({sr.three_center.modeled_peak_bytes,af.modeled_peak_bytes,bf.modeled_peak_bytes}),
            add(mul(32,mul(nk,bf.tile_elements)),mul(64,mul(a.maximum_size,p.panel_vectors))))));
    p.modeled_peak_bytes=std::max(p.metric_peak_bytes,p.coupling_peak_bytes);
    if(o.compute_metric) {
        p.metric_shell_calls=add(sr.metric_shell_calls,mul(panels,af.shell_calls));
        p.metric_primitive_products=add(mul(sr.metric_image_candidates,sr.metric.primitive_products),mul(panels,af.primitive_pairs));
        p.metric_scalar_terms=mul(128,add(p.metric_shell_calls,mul(square(p.n_aux),add(sr.metric_image_candidates,add(nv,1)))));
        p.metric_kernel_terms=add(mul(sr.metric_image_candidates,sr.metric.kernel_terms),add(mul(panels,af.kernel_terms),mul(8,p.metric_scalar_terms)));
    }
    // Shell tiling recomputes auxiliary FT per AO shell pair and AO FT per
    // auxiliary shell. Admission charges that work, including the zero panel.
    const Count shell_ft=add(mul(pair_shells,af.shell_calls),mul(aux_shells,mul(images,bf.shell_calls)));
    const Count primitive_ft=add(mul(pair_shells,af.primitive_pairs),mul(aux_shells,mul(images,bf.primitive_pairs)));
    p.coupling_shell_calls=add(sr.three_center_shell_calls,mul(panels,shell_ft));
    p.coupling_primitive_products=add(mul(mul(images,sr.auxiliary_image_candidates),sr.three_center.primitive_products),mul(panels,primitive_ft));
    const Count phase=mul(panels,mul(nk,mul(images,mul(pair_shells,aux_shells))));
    const Count folds=mul(add(nv,1),mul(nk,mul(images,mul(pairs,aux_shells))));
    const Count digests=mul(add(nv,1),mul(nk,mul(pairs,p.n_aux)));
    p.coupling_scalar_terms=add(sr.scalar_terms,mul(128,add(p.coupling_shell_calls,add(phase,add(folds,digests)))));
    p.coupling_kernel_terms=add(mul(mul(images,sr.auxiliary_image_candidates),sr.three_center.kernel_terms),
        add(mul(panels,add(mul(pair_shells,af.kernel_terms),mul(aux_shells,mul(images,bf.kernel_terms)))),mul(8,p.coupling_scalar_terms)));
    p.shell_calls=add(p.metric_shell_calls,p.coupling_shell_calls);
    p.primitive_products=add(p.metric_primitive_products,p.coupling_primitive_products);
    p.scalar_terms=add(p.metric_scalar_terms,p.coupling_scalar_terms);p.kernel_terms=add(p.metric_kernel_terms,p.coupling_kernel_terms);
    gdf_tile_caps(p,budget);return p;
}

PeriodicGDFTiles::PeriodicGDFTiles(const BasisSet& orbital,const BasisSet& aux,const PeriodicSystem& system,
    const Eigen::Vector3d& q,const md::Matrix& k,const md::Matrix& vectors,
    const PeriodicGDFShortRangeOptions& o,const ResourceBudget& budget,const PeriodicGDFKernelOptions& physical)
    : plan_([&] {
        if(k.cols()!=3 || vectors.cols()!=3) throw std::invalid_argument("MANTA GDF tile input vectors require three components");
        return plan_periodic_gdf_tiles(orbital,aux,system,vectors.rows(),k.rows(),o,budget,physical);
      }()),orbital_(orbital),auxiliary_(aux),q_(q),k_(k),vectors_(vectors),options_(o),physical_(physical) {
    const auto kernel=validate_gdf_domain(system,q,k,vectors,o);
    system_.dim=system.dim;system_.lattice=system.lattice;
    describe_physical_gdf(source_,physical_,kernel.zero_transfer);
    source_.short_range=describe_gdf_short_range(orbital_,auxiliary_,system_,q_,k_,options_,plan_.modeled_peak_bytes,source_.real_short_range);
    if(o.compute_metric) source_.metric_source_identity=gdf_identity(true,vectors_,source_);
    source_.three_center_source_identity=gdf_identity(false,vectors_,source_);
    FittingDigest joint;joint.text("manta.periodic-gdf.joint.v1");joint.text(source_.metric_source_identity);
    joint.text(source_.three_center_source_identity);source_.source_identity=joint.hash.finish_hex();
    source_.modeled_peak_bytes=plan_.modeled_peak_bytes;
}

PeriodicGDFTile PeriodicGDFTiles::metric() const {
    if(!options_.compute_metric) throw std::invalid_argument("MANTA GDF tile source has no admitted metric");
    const auto& a=auxiliary_.libint();const auto offsets=a.shell2bf();const Count n=plan_.n_aux;
    PeriodicGDFTile result;result.shape={n,n};result.offsets={0,0};result.receipt=source_;
    auto& r=result.receipt;result.values.assign(admission::square(n),{});std::vector<Complex> error(result.values.size());
    const MetricImageDomain domain(system_,options_.auxiliary_cutoff_bohr,options_.omega,options_.image_candidate_cap,options_.integral_screen_error);
    if(source_.real_short_range) for(Count is=0;is<a.size();++is) for(Count js=0;js<a.size();++js)
        domain.each(a[is],a[js],[&](const Eigen::Vector3d& t) {
            const auto ket=shifted(a[js],t);const auto values=md::shell_coulomb({&a[is],nullptr,&ket,nullptr},md::RadialKernel::Erfc,options_.omega);
            const auto phase=fitting_phase(q_.dot(t),r.short_range);
            ++r.short_range.metric_shell_calls;r.short_range.primitive_products+=a[is].nprim()*a[js].nprim();
            for(Count i=0;i<a[is].size();++i) for(Count j=0;j<a[js].size();++j) {
                const Count out=(offsets[is]+i)*n+offsets[js]+j;
                fitting_add(phase*values(i,j),result.values[out],error[out]);++r.short_range.phase_accumulations;
            }
        });
    // Match the raw-SR -> LR boundary of the dense reference: SR rounding
    // compensation is not carried into the independently accumulated LR sum.
    std::fill(error.begin(),error.end(),Complex{});
    const RangeSeparatedCoulombKernel kernel(system_,q_,options_.omega);
    auto panel=[&](const md::Matrix& v,bool zero) {
        const Count ng=v.rows();md::ComplexMatrix aft(n,ng);
        for(Count is=0;is<a.size();++is) {
            const auto tile=md::shell_fourier_panel(&a[is],nullptr,v);
            if(!tile.allFinite()) throw std::overflow_error("MANTA GDF metric Fourier tile is nonfinite");
            aft.middleRows(offsets[is],a[is].size())=tile;
            ++r.auxiliary_fourier_shell_calls;++r.shell_calls;r.primitive_products+=a[is].nprim();r.vector_primitive_products+=ng*a[is].nprim();
        }
        std::vector<double> weights(ng);
        for(Count g=0;g<ng;++g) {
            const double p2=v.row(g).squaredNorm();admission::require_finite(p2);
            weights[g]=physical_gdf_weight(kernel,physical_,p2,zero);admission::require_finite(weights[g]);
            if(!zero && !RangeSeparatedCoulombKernel::singular(p2)) ++r.reciprocal_vectors;
        }
        for(Count i=0;i<n;++i) for(Count j=0;j<n;++j) for(Count g=0;g<ng;++g) {
            fitting_add(weights[g]*std::conj(aft(i,g))*aft(j,g),result.values[i*n+j],error[i*n+j]);++r.contraction_terms;
        }
    };
    if(kernel.zero_transfer && source_.real_short_range) panel(md::Matrix::Zero(1,3),true);
    for(Count first=0;first<plan_.vectors;first+=plan_.panel_vectors)
        panel(vectors_.middleRows(first,std::min(plan_.panel_vectors,plan_.vectors-first)),false);
    finish_gdf_tile(result);
    admission::cap(r.shell_calls,plan_.metric_shell_calls,"MANTA GDF metric executed-work bound exceeded");
    admission::cap(r.primitive_products,plan_.metric_primitive_products,"MANTA GDF metric primitive bound exceeded");
    return result;
}

PeriodicGDFTile PeriodicGDFTiles::evaluate(Count ip,Count im,Count in) const {
    if(ip>=auxiliary_.nshells() || im>=orbital_.nshells() || in>=orbital_.nshells())
        throw std::out_of_range("MANTA GDF coupling shell index");
    const auto& a=auxiliary_.libint();const auto& b=orbital_.libint();const auto& sp=a[ip];const auto& sm=b[im];const auto& sn=b[in];
    const Count np=sp.size(),nm=sm.size(),nn=sn.size(),nk=plan_.momenta,tile=np*nm*nn;
    const auto ai=a.shell2bf(),bi=b.shell2bf();PeriodicGDFTile result;
    result.shape={nk,np,nm,nn};result.offsets={0,ai[ip],bi[im],bi[in]};result.receipt=source_;auto& r=result.receipt;
    result.values.assign(admission::mul(nk,tile),{});std::vector<Complex> error(result.values.size()),contracted(tile),correction(tile);
    const ThreeCenterImageDomain domain(system_,options_.pair_cutoff_bohr,options_.auxiliary_cutoff_bohr,options_.omega,
        options_.image_candidate_cap,options_.integral_screen_error);
    auto ket=sn;
    if(source_.real_short_range) domain.each(sp,sm,sn,[&](const Eigen::Vector3d& t) {
        ket=shifted(sn,t);std::fill(contracted.begin(),contracted.end(),Complex{});std::fill(correction.begin(),correction.end(),Complex{});
        ++r.short_range.pair_shell_images;
    },[&](const Eigen::Vector3d&,const Eigen::Vector3d& t) {
        const auto bra=shifted(sp,t);const auto values=md::shell_coulomb({&bra,nullptr,&sm,&ket},md::RadialKernel::Erfc,options_.omega);
        const auto phase=fitting_phase(-q_.dot(t),r.short_range);++r.short_range.three_center_shell_calls;
        r.short_range.primitive_products+=sp.nprim()*sm.nprim()*sn.nprim();
        for(Count i=0;i<tile;++i) { fitting_add(phase*values.data()[i],contracted[i],correction[i]);++r.short_range.phase_accumulations; }
    },[&](const Eigen::Vector3d& t) {
        for(Count ik=0;ik<nk;++ik) {
            const auto phase=fitting_phase(k_.row(ik).dot(t),r.short_range);
            for(Count i=0;i<tile;++i) { fitting_add(phase*contracted[i],result.values[ik*tile+i],error[ik*tile+i]);++r.short_range.phase_accumulations; }
        }
    });
    std::fill(error.begin(),error.end(),Complex{});
    const RangeSeparatedCoulombKernel kernel(system_,q_,options_.omega);const ImageBox images(system_,options_.pair_cutoff_bohr,options_.image_candidate_cap);
    auto panel=[&](const md::Matrix& v,bool zero) {
        const Count ng=v.rows(),pairs=nm*nn;const auto aft=md::shell_fourier_panel(&sp,nullptr,v);
        if(!aft.allFinite()) throw std::overflow_error("MANTA GDF auxiliary Fourier tile is nonfinite");
        ++r.auxiliary_fourier_shell_calls;++r.shell_calls;r.primitive_products+=sp.nprim();r.vector_primitive_products+=ng*sp.nprim();
        std::vector<Complex> folded(nk*pairs*ng),fold_error(folded.size());std::vector<double> weights(ng);
        for(Count g=0;g<ng;++g) {
            const double p2=v.row(g).squaredNorm();admission::require_finite(p2);
            weights[g]=physical_gdf_weight(kernel,physical_,p2,zero);admission::require_finite(weights[g]);
            if(!zero && !RangeSeparatedCoulombKernel::singular(p2)) ++r.reciprocal_vectors;
        }
        images.each(origin(sm)-origin(sn),[&](const Eigen::Vector3d& t) {
            if(!pair_in_range(sm,sn,t,options_.pair_cutoff_bohr)) return;
            const auto translated=shifted(sn,t);const auto ft=md::shell_fourier_panel(&sm,&translated,v);
            if(!ft.allFinite()) throw std::overflow_error("MANTA GDF pair Fourier tile is nonfinite");
            ++r.pair_fourier_shell_calls;++r.shell_calls;r.primitive_products+=sm.nprim()*sn.nprim();
            r.vector_primitive_products+=ng*sm.nprim()*sn.nprim();
            for(Count ik=0;ik<nk;++ik) {
                const double angle=k_.row(ik).dot(t);admission::require_finite(angle);const Complex phase=std::exp(Complex(0.,angle));++r.phase_products;
                for(Count pair=0;pair<pairs;++pair) for(Count g=0;g<ng;++g) {
                    const Count out=(ik*pairs+pair)*ng+g;
                    fitting_add(phase*ft(pair,g),folded[out],fold_error[out]);++r.phase_accumulations;
                }
            }
        });
        for(Count ik=0;ik<nk;++ik) for(Count i=0;i<np;++i) for(Count j=0;j<pairs;++j) for(Count g=0;g<ng;++g) {
            const Count out=(ik*np+i)*pairs+j;
            fitting_add(weights[g]*std::conj(aft(i,g))*folded[(ik*pairs+j)*ng+g],result.values[out],error[out]);++r.contraction_terms;
        }
    };
    if(kernel.zero_transfer && source_.real_short_range) panel(md::Matrix::Zero(1,3),true);
    for(Count first=0;first<plan_.vectors;first+=plan_.panel_vectors)
        panel(vectors_.middleRows(first,std::min(plan_.panel_vectors,plan_.vectors-first)),false);
    finish_gdf_tile(result);
    admission::cap(r.shell_calls,plan_.coupling_shell_calls,"MANTA GDF tile executed-work bound exceeded");
    admission::cap(r.primitive_products,plan_.coupling_primitive_products,"MANTA GDF tile primitive bound exceeded");
    return result;
}
} // namespace manta
}  // namespace vibeqc
