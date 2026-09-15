#include "vibeqc/periodic_gdf_short_range.hpp"
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
            if (std::isfinite(radius) && radius < auxiliary_cutoff + pair_cutoff / 2.0) {
                ImageBox(system, radius, image_cap).each(product_center - P, visit);
            } else {
                aux_images.each((A + end) / 2.0 - P, visit);
            }
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
    std::size_t image_candidate_cap, double integral_screen_error, bool compute_metric) {
    validate(system, q, omega, pair_cutoff);
    validate(system, q, omega, auxiliary_cutoff);
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
    if (compute_metric)
        result.metric = compute_gdf_sr_metric(aux, system, q, omega,
            auxiliary_cutoff, metric_bytes, image_candidate_cap, transient_byte_cap - charge_bytes,
            integral_screen_error);
    result.three_center = compute_gdf_sr_three_center(orbital, aux, system,
        q, ket_kpoints, omega, pair_cutoff, auxiliary_cutoff,
        tensor_bytes, image_candidate_cap, transient_byte_cap - charge_bytes, integral_screen_error);
    const RangeSeparatedCoulombKernel kernel(system, q, omega);
    if (kernel.zero_transfer) {
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
            weights[g] = kernel.long_range(p2);
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
}  // namespace vibeqc
