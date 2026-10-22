#include <iterator>
#include "vibeqc/periodic_gdf_short_range.hpp"
#include "vibeqc/manta_periodic_fitting.hpp"
#include "vibeqc/manta_periodic_gdf_pair_tree.hpp"
#include "vibeqc/manta_periodic_error.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/manta_lattice_errors.hpp"
#include "vibeqc/manta_moments.hpp"
#include "manta_extent_pair.hpp"
#include "manta_ri_interval.hpp"
#include "manta_auxiliary_image_bubbles.hpp"
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
#include <optional>
#include <map>
#include <set>
#include <tuple>
#include <deque>
#include <mutex>
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
    template<class F> void each_index(const Eigen::Vector3d& center, F&& f) const {
        const Eigen::Vector3d frac = inverse * center;
        std::array<long long, 3> lo, hi;
        for (int i = 0; i < 3; ++i) {
            lo[i] = static_cast<long long>(std::floor(frac[i] - reach[i])) - 1;
            hi[i] = static_cast<long long>(std::ceil(frac[i] + reach[i])) + 1;
        }
        for (auto i = lo[0]; i <= hi[0]; ++i)
            for (auto j = lo[1]; j <= hi[1]; ++j)
                for (auto k = lo[2]; k <= hi[2]; ++k)
                    f(lattice * Eigen::Vector3d(double(i), double(j), double(k)),std::array<long long,3>{i,j,k});
    }
    template<class F> void each(const Eigen::Vector3d& center, F&& f) const {
        each_index(center,[&](const Eigen::Vector3d& r,const std::array<long long,3>&) {f(r);});
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

bool inside_auxiliary_capsule(const Eigen::Vector3d& point,const Eigen::Vector3d& begin,
    const Eigen::Vector3d& end,double radius) {
    const Eigen::Vector3d segment=end-begin,displacement=point-begin;
    const double length2=segment.squaredNorm();
    const double fraction=length2>0.?std::clamp(displacement.dot(segment)/length2,0.,1.):0.;
    return !((displacement-fraction*segment).squaredNorm()>radius*radius);
}
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

    template<class F> void each_unscreened_pair(const libint2::Shell& mu,const libint2::Shell& nu,F&& f) const {
        if(error!=0.) throw std::invalid_argument("MANTA ordered pair tree requires the unscreened GDF selector");
        pair_images.each_index(origin(mu)-origin(nu),[&](const Eigen::Vector3d& r,const std::array<long long,3>& index) {
            if(pair_in_range(mu,nu,r,pair_cutoff)) f(r,index);
        });
    }
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
                if (!inside_auxiliary_capsule(P+t,A,end,auxiliary_cutoff)) return;
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
    auto engine = libint2::Engine(libint2::Operator::erfc_coulomb,
        std::min(std::max(a.max_nprim(), b.max_nprim()), primitive_block_size),
        std::max(a.max_l(), b.max_l()),
        derivative_order, std::numeric_limits<double>::epsilon(), Params{omega}, braket);
    // The image domains above already apportion an absolute omission budget
    // using Gaussian/angular envelopes (Ye 2021 SI S4.1). Libint's default
    // primitive screen instead tests coefficient/exponent products before
    // Coulomb and angular prefactors. Even an epsilon screen can omit whole
    // negative p-p image contributions and spoil a nearly cancelling Bloch
    // sum. Do not add that independent, unbudgeted omission to our domains.
    // Set this AFTER construction: the Boys evaluator still needs its
    // supported epsilon precision; set_precision controls primitive omission.
    engine.set_precision(0.0);
    return engine;
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
    // Share the k-independent pair/image Fourier work across bounded ket
    // tiles. Shrink the pair tile to keep its aggregate output near the
    // scalar panel size. A tight caller budget falls back to smaller k tiles,
    // including the original scalar path, without changing image support.
    const auto target_tasks = product(4, std::size_t(std::max(1, omp_max_threads())));
    const auto aux_bytes = product(product(na, vector_block), sizeof(Complex));
    const auto charge_bytes = product(na, sizeof(Complex));
    const auto shared_bytes = sum(sum(aux_bytes, charge_bytes),
        sizeof(std::array<double, vector_block>));
    if (shared_bytes > transient_byte_cap)
        throw std::length_error("RSGDF Fourier transient byte cap exceeded");
    const auto available_worker_bytes = transient_byte_cap - shared_bytes;
    auto k_block = std::min(nk, ao_pair_fourier_k_tile);
    std::size_t pair_block = 0, pair_bytes = 0, per_worker_bytes = 0;
    for (;;) {
        const auto k_tiles = nk / k_block + (nk % k_block != 0);
        pair_block = std::min<std::size_t>(16 / k_block,
            std::max<std::size_t>(1, product(pairs, k_tiles) / target_tasks));
        pair_bytes = product(product(product(k_block, pair_block), vector_block), sizeof(Complex));
        const auto workspace = k_block == 1
            ? ao_pair_fourier_fixed_numeric_workspace_bytes()
            : ao_pair_fourier_batch_numeric_workspace_bytes();
        per_worker_bytes = sum(pair_bytes, workspace);
        if (per_worker_bytes <= available_worker_bytes) break;
        if (k_block == 1)
            throw std::length_error("RSGDF Fourier transient byte cap exceeded");
        k_block = std::max<std::size_t>(1, k_block / 2);
    }
    const auto worker_cap = std::min<std::size_t>(
        available_worker_bytes / per_worker_bytes, std::numeric_limits<int>::max());
    const auto pair_tiles = pairs / pair_block + (pairs % pair_block != 0);
    const auto k_tiles = nk / k_block + (nk % k_block != 0);
    const auto fourier_tasks = product(k_tiles, pair_tiles);
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
        // Columns own disjoint metric slices, and (k-tile,pair-tile) tasks own
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
                const auto first_k = (task / pair_tiles) * k_block;
                const auto nk_tile = std::min(k_block, nk - first_k);
                const auto first = (task % pair_tiles) * pair_block;
                const auto count = std::min(pair_block, pairs - first);
                AOPairFourierPanel ao;
                if (k_block == 1) {
                    ao = ao_pair_gaussian_fourier_panel(orbital, system, panel_vectors,
                        ket_kpoints.row(first_k).transpose(), first, count, pair_cutoff,
                        image_candidate_cap, pair_bytes);
                } else {
                    ao = ao_pair_gaussian_fourier_batch(orbital, system, panel_vectors,
                        ket_kpoints.middleRows(first_k, nk_tile), first, count,
                        pair_cutoff, image_candidate_cap, pair_bytes);
                }
                for (std::size_t k = 0; k < nk_tile; ++k)
                    for (std::size_t p = 0; p < na; ++p)
                        for (std::size_t pair = 0; pair < count; ++pair) {
                            Complex value{};
                            for (std::size_t g = 0; g < ng; ++g)
                                value += weights[g] * std::conj(auxiliary(p, g))
                                    * ao.data[(k * count + pair) * ng + g];
                            result.three_center.data[((first_k + k) * na + p) * pairs + first + pair] += value;
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
namespace verified_coupling = spectral_interval;
// Preparation belongs to one immutable finite source, never to a global
// basis/image registry. Each published extent/moment panel is immutable.
// A source's plan reserves its entire possible cache before input copies;
// lazy construction preserves direct/always-near paths and failure behavior.
struct PeriodicGDFMomentCache {
    struct Pair {
        PairExtent extent;
        md::Point ket_center;
        mutable std::once_flag once;
        mutable md::Matrix moments;
    };
    struct Auxiliary {
        AuxiliaryExtent extent;
        mutable std::once_flag once;
        mutable md::Matrix moments;
    };
    const ExtentOptions extents;
    const Count maximum_pairs,maximum_auxiliary,maximum_supports,capacity;
    mutable std::mutex mutex;
    std::map<std::array<Count,3>,std::shared_ptr<const Pair>> pairs;
    std::map<Count,std::shared_ptr<const Auxiliary>> auxiliary;
    std::map<std::array<Count,3>,Count> supports;
    Count panels=0,elements=0;
    PeriodicGDFMomentCache(const ExtentOptions& e,Count p,Count a,Count s,Count bytes)
        :extents(e),maximum_pairs(p),maximum_auxiliary(a),maximum_supports(s),capacity(bytes) {}
    template<class CountImages> Count support(const std::array<Count,3>& key,CountImages&& count) {
        std::lock_guard<std::mutex> guard(mutex);auto found=supports.find(key);
        if(found!=supports.end()) return found->second;
        admission::cap(admission::add(supports.size(),1),maximum_supports,"MANTA prepared support-count bound exceeded");
        const Count value=count();supports.emplace(key,value);return value;
    }
    static Count entry_bytes(int order) {return admission::add(1024,admission::mul(64,order+1));}
    std::shared_ptr<const Pair> pair(const libint2::Shell& a,const libint2::Shell& b,Count i,Count j,Count ordinal,
        PeriodicGDFShortRangeReceipt& r) {
        const std::array<Count,3> key{i,j,ordinal};std::lock_guard<std::mutex> guard(mutex);
        auto found=pairs.find(key);
        if(found!=pairs.end()) {
            if(found->second->ket_center!=b.O) throw std::logic_error("MANTA prepared pair image does not match its ordered source");
            ++r.multipole_reused_extents;return found->second;
        }
        admission::cap(admission::add(pairs.size(),1),maximum_pairs,"MANTA prepared pair inventory bound exceeded");
        auto entry=std::make_shared<Pair>();Count always_near=0;
        entry->extent=detail::pair_extent(a,b,{i,j},extents,r.multipole_envelope_evaluations,always_near);
        entry->ket_center=b.O;r.multipole_primitive_products+=a.nprim()*b.nprim();
        pairs.emplace(key,entry);return entry;
    }
    std::shared_ptr<const Auxiliary> aux(const libint2::Shell& a,Count i,PeriodicGDFShortRangeReceipt& r) {
        std::lock_guard<std::mutex> guard(mutex);auto found=auxiliary.find(i);
        if(found!=auxiliary.end()) {++r.multipole_reused_extents;return found->second;}
        admission::cap(admission::add(auxiliary.size(),1),maximum_auxiliary,"MANTA prepared auxiliary inventory bound exceeded");
        auto entry=std::make_shared<Auxiliary>();Count always_near=0;
        entry->extent=detail::auxiliary_extent(a,i,extents,r.multipole_envelope_evaluations,always_near);
        r.multipole_primitive_products+=a.nprim();auxiliary.emplace(i,entry);return entry;
    }
    template<class Entry,class Evaluate> const md::Matrix& moments(const std::shared_ptr<const Entry>& entry,
        Evaluate&& evaluate,Count primitives,PeriodicGDFShortRangeReceipt& r) {
        bool constructed=false;
        std::call_once(entry->once,[&] {
            auto values=evaluate();
            if(!values.allFinite()) throw std::overflow_error("MANTA prepared moments are nonfinite");
            entry->moments=std::move(values);
            {std::lock_guard<std::mutex> guard(mutex);panels=admission::add(panels,1);elements=admission::add(elements,entry->moments.size());}
            constructed=true;
        });
        if(constructed) {++r.multipole_moment_panels;r.multipole_primitive_products+=primitives;}
        else ++r.multipole_reused_moment_panels;
        return entry->moments;
    }
    PeriodicGDFPreparationInfo info(const std::string& source) const {
        using namespace admission;
        std::lock_guard<std::mutex> guard(mutex);PeriodicGDFPreparationInfo r;
        r.enabled=true;r.source_identity=source;r.pair_extents=pairs.size();r.auxiliary_extents=auxiliary.size();
        r.moment_panels=panels;r.moment_elements=elements;r.support_tiles=supports.size();r.capacity_bytes=capacity;
        r.modeled_live_bytes=add(8192,add(mul(256,supports.size()),add(mul(entry_bytes(extents.moment_order),add(pairs.size(),auxiliary.size())),mul(8,elements))));
        cap(r.modeled_live_bytes,capacity,"MANTA prepared source storage bound exceeded");return r;
    }
};
// Density-independent plans belong to this exact ordered source. The finite
// byte budget bounds a FIFO working set, not a rectangular image interaction
// inventory. Overflow entries stream through the original producer. Cached
// entries are immutable; an evicted entry may remain borrowed by one action,
// whose plan separately reserves a full entry-capacity workspace.
struct PeriodicGDFPairState {
    std::once_flag once;
    std::shared_ptr<const PeriodicGDFPairTree> geometry;
    std::unique_ptr<BubbleErrorModel> bounds;
    Count maximum_support=0;
};
struct PeriodicGDFPartitionCache {
    struct Step {
        Count node=0;md::Matrix interaction;double error=0.;
        // Global total-degree matrices retain only their structurally allowed
        // entries. Expansion restores the SAME dense matrix before arithmetic.
        // Legacy auxiliary-image partitions keep their existing dense matrix.
        std::vector<double> packed;
    };
    struct Entry {
        std::shared_ptr<const BubbleTree> tree;
        std::vector<Step> steps;
        md::Point ket_center{};
        Count images=0,bytes=0;
    };
    using Key=std::array<Count,4>;
    const Count capacity;
    mutable std::mutex mutex;
    std::map<Key,std::shared_ptr<const Entry>> entries;
    std::deque<Key> fifo;
    Count bytes=0,evictions=0;
    explicit PeriodicGDFPartitionCache(Count cap):capacity(cap) {}
    template<class Build> std::shared_ptr<const Entry> get(const Key& key,Count images,
        const md::Point& center,Count upper,Build&& build,PeriodicGDFShortRangeReceipt& r,
        bool separate_entry_workspace=false) {
        std::lock_guard<std::mutex> guard(mutex);
        auto found=entries.find(key);
        if(found!=entries.end()) {
            if(found->second->images!=images || found->second->ket_center!=center)
                throw std::logic_error("MANTA prepared partition does not match its ordered source");
            ++r.multipole_partition_reuses;return found->second;
        }
        // The global pair action admits a complete constructing entry outside
        // this retained cache, even with zero cache capacity. It can build first
        // and evict for actual storage; a worst-case entry bound must not evict
        // a working set of much smaller entries. Legacy paths reserve first.
        if(!separate_entry_workspace && upper>capacity) {++r.multipole_partition_bypasses;return {};}
        auto evict=[&](Count required) {
            while(bytes>capacity-required) {
                const auto oldest=fifo.front();fifo.pop_front();auto old=entries.find(oldest);
                bytes-=old->second->bytes;entries.erase(old);++evictions;
            }
        };
        if(!separate_entry_workspace) evict(upper);
        auto entry=build();entry->images=images;entry->ket_center=center;
        admission::cap(entry->bytes,upper,"MANTA partition entry storage bound exceeded");
        if(separate_entry_workspace) {
            // A large entry is still useful to this action; do not build it twice
            // or evict other entries when it cannot be retained at all.
            if(entry->bytes>capacity) {++r.multipole_partition_bypasses;return entry;}
            evict(entry->bytes);
        }
        const Count next=admission::add(bytes,entry->bytes);
        fifo.push_back(key);
        try {entries.emplace(key,entry);} catch(...) {fifo.pop_back();throw;}
        bytes=next;
        return entry;
    }
    PeriodicGDFPartitionInfo info(const std::string& source) const {
        std::lock_guard<std::mutex> guard(mutex);PeriodicGDFPartitionInfo r;
        r.enabled=true;r.source_identity=source;r.entries=entries.size();r.evictions=evictions;
        r.modeled_live_bytes=admission::add(8192,bytes);r.capacity_bytes=admission::add(8192,capacity);
        return r;
    }
};
namespace {
struct FittingDigest {
    vibeqc::detail::Sha256 hash;
    void word(Count v) { std::array<std::uint8_t,8> b{};for(int i=0;i<8;++i)b[i]=v>>(56-8*i);hash.update(b.data(),8); }
    void text(const std::string& s) { word(s.size());hash.update(reinterpret_cast<const std::uint8_t*>(s.data()),s.size()); }
    void real(double v) { if(v==0)v=0.;Count bits;std::memcpy(&bits,&v,8);word(bits); }
    template<class M> void matrix(const M& m) { word(m.rows());word(m.cols());for(Eigen::Index i=0;i<m.rows();++i)for(Eigen::Index j=0;j<m.cols();++j)real(m(i,j)); }
};
ExtentOptions coupling_extents(const PeriodicGDFShortRangeOptions& o) {
    auto e=o.coupling_multipole_extents;
    e.moment_order=o.coupling_multipole_order;e.retain_full_moments=true;return e;
}
double coupling_screen(const PeriodicGDFShortRangeOptions& o) {
    if(o.coupling_multipole_pair_tree && (!(o.coupling_multipole_error>0.) || !o.coupling_multipole_reuse))
        throw std::invalid_argument("MANTA coupling_multipole_pair_tree requires positive multipole error and preparation reuse");
    if(!std::isfinite(o.coupling_multipole_error) || o.coupling_multipole_error<0. ||
       (o.coupling_multipole_error>0. && (o.integral_screen_error!=0. || o.coupling_screen_error!=0.)))
        throw std::invalid_argument("MANTA coupling_multipole_error must be finite nonnegative and cannot combine with screening");
    if(!std::isfinite(o.coupling_screen_error) || o.coupling_screen_error<0. ||
       (o.coupling_screen_error>0. && o.integral_screen_error!=0.))
        throw std::invalid_argument("MANTA coupling_screen_error must be finite nonnegative and cannot combine with integral_screen_error");
    return o.coupling_screen_error>0.?o.coupling_screen_error:o.integral_screen_error;
}
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
    h.real(o.omega);h.real(o.auxiliary_cutoff_bohr);h.real(three?coupling_screen(o):o.integral_screen_error);
    if(three) {
        h.text(r.orbital_basis_identity);h.matrix(k);h.real(o.pair_cutoff_bohr);
        if(o.coupling_multipole_error>0.) {
            h.text("ordered-erfc-multipoles.v2;joint-radial-and-interior-moment-bounds");h.real(o.coupling_multipole_error);h.word(o.coupling_multipole_order);
            if(o.coupling_multipole_pair_tree) h.text("ordered-global-pair-bubbles.v1;gamma-hermitian-pair-completion-v1;max-support-allocation;leaf=1;branching=3;depth=32");
            else if(o.coupling_multipole_hierarchy) h.text("auxiliary-image-upward-hierarchy.v1;leaf=1;branching=3;depth=32");
            const auto e=coupling_extents(o);
            for(double x:{e.pointwise_tolerance,e.tail_mass_tolerance,e.minimum_radius,e.maximum_radius}) h.real(x);
        }
    }
    return h.hash.finish_hex();
}


} // namespace
// Neese et al. (2025), doi:10.1021/acs.jpca.4c07415, Eqs.29-36:
// contract the input before far-field translation and project fields back
// afterward. Ordered periodic phases remain producer-owned. No metric,
// Hermitization, canonical pair weights or density-dependent selection here.
// One call-wide gather/scatter over the producer's ordered pair inventory.
// Gaussian panels are immutable source-owned data, but these charges/fields
// live only for this input. This is the pair-moment stage of RI-BUPO (Neese
// et al. 2025, doi:10.1021/acs.jpca.4c07415, Eqs.29-36), without changing the
// selected auxiliary partitions or claiming a global pair-bubble traversal.
struct PeriodicGDFGlobalAction {
    struct Pair {
        std::shared_ptr<const PeriodicGDFMomentCache::Pair> prepared;
        Eigen::Vector3d image;
        Eigen::VectorXcd phases,charge,field,correction;
        Eigen::VectorXd magnitude;
        Count nm=0,nn=0,m0=0,n0=0;
    };
    const std::vector<Complex>& input;
    std::vector<Complex>& output;
    std::vector<Complex>& correction;
    PeriodicGDFCouplingReceipt& receipt;
    const md::Matrix& k;
    const bool forward;
    const Count n,pair_slots,auxiliary_slots;
    std::map<std::array<Count,3>,Pair> pairs;
    struct Auxiliary {
        std::shared_ptr<const PeriodicGDFMomentCache::Auxiliary> prepared;
        Eigen::VectorXcd charge,field,correction;
    };
    std::map<Count,Auxiliary> auxiliary;
    PeriodicGDFGlobalAction(const std::vector<Complex>& x,std::vector<Complex>& y,std::vector<Complex>& error,
        PeriodicGDFCouplingReceipt& r,const md::Matrix& momenta,bool f,const PeriodicGDFTilesPlan& plan)
        :input(x),output(y),correction(error),receipt(r),k(momenta),forward(f),n(plan.n_orb),
         pair_slots(plan.global_pair_slots),auxiliary_slots(plan.global_auxiliary_slots) {}
    void products(Count c) {receipt.multipole_action_products=admission::add(receipt.multipole_action_products,c);}
    Pair& pair(const std::array<Count,3>& key,const std::shared_ptr<const PeriodicGDFMomentCache::Pair>& prepared,
        const md::Matrix& panel,const Eigen::Vector3d& image,Count nm,Count nn,Count m0,Count n0,const std::vector<Complex>* phases=nullptr) {
        if(!prepared || &panel!=&prepared->moments)
            throw std::logic_error("MANTA global pair requires the source-owned moment panel");
        auto found=pairs.find(key);
        if(found!=pairs.end()) {
            const auto& p=found->second;
            if(p.prepared!=prepared || p.image!=image || p.nm!=nm || p.nn!=nn || p.m0!=m0 || p.n0!=n0)
                throw std::logic_error("MANTA global pair does not match its ordered source");
            ++receipt.multipole_global_pair_reuses;return found->second;
        }
        admission::cap(admission::add(pairs.size(),1),pair_slots,"MANTA global pair inventory bound exceeded");
        Pair p;p.prepared=prepared;p.image=image;p.nm=nm;p.nn=nn;p.m0=m0;p.n0=n0;p.phases.resize(k.rows());
        for(Eigen::Index ik=0;ik<k.rows();++ik) {
            if(phases) p.phases[ik]=phases->at(ik);
            else {const double angle=k.row(ik).dot(image);admission::require_finite(angle);p.phases[ik]=std::exp(Complex(0.,angle));}
        }
        p.magnitude=panel.cwiseAbs().colwise().maxCoeff().transpose();
        if(forward) {
            Eigen::VectorXcd folded=Eigen::VectorXcd::Zero(nm*nn),errors=folded;
            for(Count ik=0;ik<Count(k.rows());++ik)for(Count i=0;i<nm;++i)for(Count j=0;j<nn;++j)
                fitting_add(std::conj(p.phases[ik])*input[ik*n*n+(m0+i)*n+n0+j],folded[i*nn+j],errors[i*nn+j]);
            p.charge=panel.transpose()*folded;products(admission::mul(nm*nn,admission::add(k.rows(),panel.cols())));
            if(!p.charge.allFinite()) throw std::overflow_error("MANTA global pair charge is nonfinite");
        } else {p.field=Eigen::VectorXcd::Zero(panel.cols());p.correction=p.field;}
        auto inserted=pairs.emplace(key,std::move(p));++receipt.multipole_global_pairs;return inserted.first->second;
    }
    Auxiliary& aux(Count offset,const md::Matrix& panel,const std::shared_ptr<const PeriodicGDFMomentCache::Auxiliary>& prepared) {
        if(!prepared || &panel!=&prepared->moments)
            throw std::logic_error("MANTA global auxiliary requires the source-owned moment panel");
        auto found=auxiliary.find(offset);
        if(found!=auxiliary.end()) {
            if(found->second.prepared!=prepared) throw std::logic_error("MANTA global auxiliary does not match its source");
            return found->second;
        }
        admission::cap(admission::add(auxiliary.size(),1),auxiliary_slots,"MANTA global auxiliary inventory bound exceeded");
        Auxiliary value;value.prepared=prepared;
        if(forward) {value.field=Eigen::VectorXcd::Zero(panel.cols());value.correction=value.field;}
        else {
            value.charge=panel.transpose()*Eigen::Map<const Eigen::VectorXcd>(input.data()+offset,panel.rows());
            if(!value.charge.allFinite()) throw std::overflow_error("MANTA global auxiliary charge is nonfinite");
            products(admission::mul(panel.rows(),panel.cols()));
        }
        auto inserted=auxiliary.emplace(offset,std::move(value));++receipt.multipole_global_auxiliary;
        return inserted.first->second;
    }
    void add(Auxiliary& a,const Eigen::VectorXcd& field) {
        for(Eigen::Index i=0;i<field.size();++i) fitting_add(field[i],a.field[i],a.correction[i]);
    }
    void add(Pair& p,const Eigen::VectorXcd& value) {
        for(Eigen::Index i=0;i<value.size();++i) fitting_add(value[i],p.field[i],p.correction[i]);
    }
    void finish() {
        if(forward) {
            for(const auto& [offset,a]:auxiliary) {
                const auto& panel=a.prepared->moments;const Eigen::VectorXcd value=panel*a.field;
                for(Count i=0;i<Count(value.size());++i) fitting_add(value[i],output[offset+i],correction[offset+i]);
                products(admission::mul(panel.rows(),panel.cols()));++receipt.multipole_global_auxiliary_projections;
            }
            return;
        }
        for(const auto& [key,p]:pairs) {
            const auto& panel=p.prepared->moments;const Eigen::VectorXcd value=panel*p.field;
            for(Count ik=0;ik<Count(k.rows());++ik)for(Count i=0;i<p.nm;++i)for(Count j=0;j<p.nn;++j) {
                const Count out=ik*n*n+(p.m0+i)*n+p.n0+j;
                fitting_add(p.phases[ik]*value[i*p.nn+j],output[out],correction[out]);
            }
            products(admission::mul(p.nm*p.nn,admission::add(k.rows(),panel.cols())));
            ++receipt.multipole_global_projections;
        }
    }
};
struct PeriodicGDFMultipoleAction {
    const std::vector<Complex>& input;
    std::vector<Complex>& output;
    std::vector<Complex>& correction;
    PeriodicGDFCouplingReceipt& receipt;
    const md::Matrix& k;
    const bool forward;
    const Count n,np,nm,nn,p0,m0,n0;
    Eigen::Vector3d image=Eigen::Vector3d::Zero();
    Eigen::VectorXcd phases,pair_charge,pair_field,pair_correction,auxiliary_charge;
    Eigen::VectorXd pair_magnitude;
    PeriodicGDFGlobalAction* global=nullptr;
    PeriodicGDFGlobalAction::Pair* global_pair=nullptr;
    PeriodicGDFGlobalAction::Auxiliary* global_auxiliary=nullptr;
    bool pair_ready=false,auxiliary_ready=false;
    long double far_magnitude=0.;
    bool verified=false;double verified_far_magnitude=0.;
    PeriodicGDFMultipoleAction(const std::vector<Complex>& x,std::vector<Complex>& y,
        std::vector<Complex>& error,PeriodicGDFCouplingReceipt& r,const md::Matrix& momenta,bool f,
        Count ao,Count ap,Count am,Count an,Count p,Count m,Count v,PeriodicGDFGlobalAction* shared=nullptr,bool audit=false)
        :input(x),output(y),correction(error),receipt(r),k(momenta),forward(f),
         n(ao),np(ap),nm(am),nn(an),p0(p),m0(m),n0(v),global(shared),verified(audit) {}
    void products(Count count) {receipt.multipole_action_products=admission::add(receipt.multipole_action_products,count);}
    void start(const Eigen::Vector3d& r) {image=r;pair_ready=false;global_pair=nullptr;}
    void prepare_pair(const md::Matrix& panel,const std::shared_ptr<const PeriodicGDFMomentCache::Pair>& prepared,
        const PeriodicGDFPartitionCache::Key& key) {
        if(pair_ready) return;
        if(global) {
            global_pair=&global->pair({key[1],key[2],key[3]},prepared,panel,image,nm,nn,m0,n0);
            pair_ready=true;++receipt.multipole_compressed_pairs;return;
        }
        const Count pairs=nm*nn,h=panel.cols();phases.resize(k.rows());
        for(Eigen::Index ik=0;ik<k.rows();++ik) {
            const double angle=k.row(ik).dot(image);admission::require_finite(angle);
            phases[ik]=std::exp(Complex(0.,angle));
        }
        pair_magnitude=panel.cwiseAbs().colwise().maxCoeff().transpose();
        if(forward) {
            Eigen::VectorXcd folded=Eigen::VectorXcd::Zero(pairs),error=folded;
            for(Count ik=0;ik<Count(k.rows());++ik)for(Count i=0;i<nm;++i)for(Count j=0;j<nn;++j)
                fitting_add(std::conj(phases[ik])*input[ik*n*n+(m0+i)*n+n0+j],folded[i*nn+j],error[i*nn+j]);
            pair_charge=panel.transpose()*folded;
            products(admission::mul(pairs,admission::add(k.rows(),h)));
            if(!pair_charge.allFinite()) throw std::overflow_error("MANTA compressed pair charge is nonfinite");
        } else {
            pair_field=Eigen::VectorXcd::Zero(h);pair_correction=pair_field;
        }
        pair_ready=true;++receipt.multipole_compressed_pairs;
    }
    const Eigen::VectorXcd& pair_input() const {return global_pair?global_pair->charge:pair_charge;}
    PeriodicGDFGlobalAction::Auxiliary& auxiliary_data(const md::Matrix& panel,
        const std::shared_ptr<const PeriodicGDFMomentCache::Auxiliary>& prepared) {
        if(!global_auxiliary) global_auxiliary=&global->aux(p0,panel,prepared);
        return *global_auxiliary;
    }
    void add_auxiliary_field(const md::Matrix& panel,const Eigen::VectorXcd& field,
        const std::shared_ptr<const PeriodicGDFMomentCache::Auxiliary>& prepared) {
        global->add(auxiliary_data(panel,prepared),field);
    }
    const Eigen::VectorXcd& auxiliary_input(const md::Matrix& panel,const std::shared_ptr<const PeriodicGDFMomentCache::Auxiliary>& prepared) {
        if(global) return auxiliary_data(panel,prepared).charge;
        if(!auxiliary_ready) {
            auxiliary_charge=panel.transpose()*Eigen::Map<const Eigen::VectorXcd>(input.data()+p0,np);
            products(admission::mul(np,panel.cols()));auxiliary_ready=true;
            if(!auxiliary_charge.allFinite()) throw std::overflow_error("MANTA compressed auxiliary charge is nonfinite");
        }
        return auxiliary_charge;
    }
    void bound(const Eigen::VectorXd& auxiliary,const md::Matrix& interaction) {
        // Bound each auxiliary row and pair column without materializing A I P^T.
        // This positive envelope is density independent; no phase cancellation
        // is credited. The existing fixed-density RI error propagation uses it.
        long double value=0.,correction=0.;
        for(Eigen::Index i=0;i<interaction.rows();++i)for(Eigen::Index j=0;j<interaction.cols();++j) {
            const long double term=static_cast<long double>(auxiliary[i])*std::abs(interaction(i,j))*(global_pair?global_pair->magnitude[j]:pair_magnitude[j]);
            const long double y=term-correction,next=value+y;correction=(next-value)-y;value=next;
        }
        far_magnitude+=periodic_error::upper(value);
        if(!std::isfinite(far_magnitude)) throw std::overflow_error("MANTA compressed coupling magnitude overflow");
        if(verified) {
            namespace vi=verified_coupling;double bound=0.,phase=0.;
            const auto& magnitudes=global_pair?global_pair->magnitude:pair_magnitude;
            const auto& pair_phases=global_pair?global_pair->phases:phases;
            for(Eigen::Index k=0;k<pair_phases.size();++k)
                phase=std::max(phase,vi::magnitude(vi::point(pair_phases[k])));
            for(Eigen::Index i=0;i<interaction.rows();++i)for(Eigen::Index j=0;j<interaction.cols();++j)
                bound=vi::plus_up(bound,vi::times_up(auxiliary[i],vi::times_up(std::abs(interaction(i,j)),magnitudes[j])));
            verified_far_magnitude=vi::plus_up(verified_far_magnitude,vi::times_up(phase,bound));
        }
        ++receipt.multipole_compressed_nodes;
    }
    void add_auxiliary(const Eigen::VectorXcd& value) {
        for(Count i=0;i<np;++i) fitting_add(value[i],output[p0+i],correction[p0+i]);
    }
    void add_pair_field(const Eigen::VectorXcd& value) {
        if(global) {global->add(*global_pair,value);return;}
        for(Eigen::Index i=0;i<value.size();++i) fitting_add(value[i],pair_field[i],pair_correction[i]);
    }
    void finish_pair(const md::Matrix& panel) {
        if(forward || !pair_ready || global) return;
        const Eigen::VectorXcd value=panel*pair_field;
        for(Count ik=0;ik<Count(k.rows());++ik)for(Count i=0;i<nm;++i)for(Count j=0;j<nn;++j) {
            const Count out=ik*n*n+(m0+i)*n+n0+j;
            fitting_add(phases[ik]*value[i*nn+j],output[out],correction[out]);
        }
        products(admission::mul(nm*nn,admission::add(k.rows(),panel.cols())));
    }
};
namespace {

// Admission composes the shared extent, moment and erfc planners. Image
// selectors and basis snapshots are unchanged; no rectangular interaction
// list or replicated basis is allocated. Counts cover an entire traversal.
PeriodicGDFPairTreePlan gdf_pair_geometry_plan(Count pair_capacity,Count orbital_shells,Count auxiliary_shells,
    Count momenta,const PeriodicSystem& system,const PeriodicGDFShortRangeOptions& o,const ResourceBudget& budget) {
    using namespace admission;PeriodicGDFPairTreePlan p;p.pair_capacity=pair_capacity;
    auto leaf=budget;leaf.other_live_bytes=0;
    leaf.memory_bytes=leaf.shell_calls=leaf.primitive_quartets=leaf.scalar_terms=leaf.kernel_terms=std::numeric_limits<Count>::max();
    const auto tree=bupo_detail::OrderedPairImageBubbles::plan(p.pair_capacity,o.coupling_multipole_order,leaf);
    p.node_capacity=tree.maximum_nodes;
    const ImageBox union_images(system,o.pair_cutoff_bohr+o.auxiliary_cutoff_bohr,o.image_candidate_cap);
    p.auxiliary_image_capacity=mul(orbital_shells,union_images.count_bound);
    // Original-frame entries, endpoint bands, query/export copies and union
    // candidates coexist with the shared tree's construction/sort workspace.
    const Count extra=add(8192,add(mul(1024,p.pair_capacity),add(mul(1024,p.node_capacity),
        add(mul(1024,auxiliary_shells),mul(256,add(momenta,1))))));
    p.modeled_peak_bytes=add(tree.modeled_peak_bytes,add(extra,mul(512,p.auxiliary_image_capacity)));
    p.scalar_terms=add(tree.scalar_terms,mul(65536,add(p.pair_capacity,p.node_capacity)));
    p.scalar_terms=add(p.scalar_terms,mul(4096,mul(p.auxiliary_image_capacity,add(p.node_capacity,64))));
    p.kernel_terms=add(tree.kernel_terms,mul(8,p.scalar_terms));return p;
}
PeriodicGDFMultipolePlan plan_coupling_multipoles(const BasisSet& orbital,const BasisSet& aux,
    Count images,Count auxiliary_images,const PeriodicGDFShortRangeOptions& o,ResourceBudget leaf) {
    using namespace admission;
    PeriodicGDFMultipolePlan p;if(o.coupling_multipole_error==0.) return p;
    leaf.other_live_bytes=0;leaf.backend_margin_bytes=65536;
    leaf.memory_bytes=leaf.shell_calls=leaf.primitive_quartets=leaf.scalar_terms=leaf.kernel_terms=std::numeric_limits<Count>::max();
    const auto ext=coupling_extents(o);
    const auto ep=plan_pair_extents(orbital,ext,leaf);const auto ea=plan_auxiliary_extents(aux,ext,leaf);
    const auto gp=plan_erfc_gaussian_interaction(o.coupling_multipole_order,leaf);
    const Count pairs=square(orbital.nshells()),nas=aux.nshells(),triples=mul(pairs,nas);
    const Count pair_scale=mul(2,mul(images,nas)); // Triangular inventory bounds the ordered traversal.
    p.shell_calls=add(mul(pair_scale,ep.shell_pairs),mul(pairs,ea.shells));
    p.primitive_products=add(mul(pair_scale,ep.primitive_pairs),mul(pairs,ea.primitives));
    p.scalar_terms=add(mul(pair_scale,ep.scalar_terms),mul(pairs,ea.scalar_terms));
    p.kernel_terms=add(mul(pair_scale,ep.kernel_terms),mul(pairs,ea.kernel_terms));
    Count peak_pair=0,peak_aux=0;
    for(Count i=0;i<orbital.nshells();++i)for(Count j=0;j<orbital.nshells();++j) {
        const auto m=plan_pair_moments(orbital,{i,j},o.coupling_multipole_order,leaf);
        const Count scale=mul(images,nas);
        p.shell_calls=add(p.shell_calls,scale);p.primitive_products=add(p.primitive_products,mul(scale,m.primitive_pairs));
        p.scalar_terms=add(p.scalar_terms,mul(scale,m.scalar_terms));p.kernel_terms=add(p.kernel_terms,mul(scale,m.kernel_terms));
        peak_pair=std::max(peak_pair,m.modeled_peak_bytes);
    }
    for(Count i=0;i<nas;++i) {
        const auto m=plan_auxiliary_moments(aux,i,o.coupling_multipole_order,leaf);
        p.shell_calls=add(p.shell_calls,pairs);p.primitive_products=add(p.primitive_products,mul(pairs,m.primitives));
        p.scalar_terms=add(p.scalar_terms,mul(pairs,m.scalar_terms));p.kernel_terms=add(p.kernel_terms,mul(pairs,m.kernel_terms));
        peak_aux=std::max(peak_aux,m.modeled_peak_bytes);
    }
    const Count candidates=mul(mul(images,auxiliary_images),triples),moments=gp.coefficients.moments;
    if(o.coupling_multipole_hierarchy) {
        // Stream one selected auxiliary image tree per ordered pair image.
        // Source selection, tree storage, envelopes and upward translations
        // are admitted before any image-indexed allocation. No full T or
        // auxiliary-by-pair image list is retained.
        const auto tp=bupo_detail::AuxiliaryImageBubbles::plan(auxiliary_images,o.coupling_multipole_order,leaf);
        const auto mp=plan_multipoles(o.coupling_multipole_order,leaf);
        const Count trees=mul(images,triples),nodes=tp.maximum_nodes,edges=add(nodes,auxiliary_images);
        const Count radial=mul(nodes,o.coupling_multipole_order+1);
        const Count upward=mul(8,mul(census(aux,7).maximum_size,square(moments)));
        p.scalar_terms=add(p.scalar_terms,mul(trees,add(tp.scalar_terms,add(mul(2,radial),mul(edges,add(mp.scalar_terms,upward))))));
        p.kernel_terms=add(p.kernel_terms,mul(trees,add(tp.kernel_terms,add(
            mul(256,mul(edges,square(o.coupling_multipole_order+1))),mul(edges,add(mp.kernel_terms,mul(8,upward)))))));
        // At most 2*N-1 node queries replace N leaf queries.
        p.scalar_terms=add(p.scalar_terms,mul(candidates,gp.scalar_terms));
        p.kernel_terms=add(p.kernel_terms,mul(candidates,gp.kernel_terms));
        // Tree + envelope producer + traversal/phase vectors + all reusable
        // complex node moment panels and conservative Eigen temporaries.
        p.modeled_peak_bytes=add(mul(2,tp.modeled_peak_bytes),add(mp.modeled_peak_bytes,
            add(mul(128,mul(nodes,mul(census(aux,7).maximum_size,moments))),
                add(mul(128,radial),mul(2048,edges)))));
    }
    const auto ac=census(aux,7),bc=census(orbital,6);
    const Count contraction=mul(4,mul(mul(ac.maximum_size,square(bc.maximum_size)),square(moments)));
    p.scalar_terms=add(p.scalar_terms,mul(candidates,add(128,add(gp.scalar_terms,contraction))));
    p.kernel_terms=add(p.kernel_terms,mul(candidates,add(gp.kernel_terms,mul(8,contraction))));
    // Simultaneously live extents, both moments, coefficient/error work,
    // temporary products and shell-sized output, including Eigen temporaries.
    p.modeled_peak_bytes=add(p.modeled_peak_bytes,add(add(ep.modeled_peak_bytes,ea.modeled_peak_bytes),
        add(add(peak_pair,peak_aux),add(gp.modeled_peak_bytes,mul(32,contraction)))));
    return p;
}

// RI-BUPO paired map (Neese et al. 2025, doi:10.1021/acs.jpca.4c07415,
// Eqs.29-36), here with the ordered finite GDF image support and erfc
// derivatives. The optional auxiliary-image tree uses the same native bubble,
// radial-envelope and Cartesian translation producers as molecular BUPO.
struct OrderedCouplingMultipoles {
    const PeriodicGDFShortRangeOptions& options;
    PeriodicGDFShortRangeReceipt& receipt;
    ExtentOptions extents;
    AuxiliaryExtent auxiliary;PairExtent pair;
    md::Matrix auxiliary_moments,pair_moments;
    std::vector<Eigen::Vector3d> selected_images;
    PeriodicGDFMomentCache* preparation=nullptr;
    PeriodicGDFPartitionCache* partitions=nullptr;
    PeriodicGDFPartitionCache::Key partition_key{};
    PeriodicGDFMultipoleAction* action=nullptr;
    std::shared_ptr<const PeriodicGDFMomentCache::Auxiliary> prepared_auxiliary;
    std::shared_ptr<const PeriodicGDFMomentCache::Pair> prepared_pair;
    const md::Matrix *auxiliary_values=nullptr,*pair_values=nullptr;
    Count pair_ordinal=0;
    long double tolerance=0.,error=0.;
    Count always_near=0;
    bool active=false;
    OrderedCouplingMultipoles(const libint2::Shell& shell,const libint2::Shell& mu,const libint2::Shell& nu,Count index,
        const ThreeCenterImageDomain& domain,const PeriodicGDFShortRangeOptions& o,PeriodicGDFShortRangeReceipt& r,
        PeriodicGDFMomentCache* prepared=nullptr,Count bra_index=0,Count ket_index=0,PeriodicGDFMultipoleAction* compressed=nullptr,PeriodicGDFPartitionCache* cached=nullptr)
        :options(o),receipt(r),extents(coupling_extents(o)),preparation(prepared),partitions(cached),
        partition_key{index,bra_index,ket_index,0},action(compressed),active(o.coupling_multipole_error>0.) {
        if(!active) return;
        // Count the actual shared support without retaining an interaction
        // list. The count and every acceptance decision are density independent.
        auto count=[&] {
            Count candidates=0;
            domain.each(shell,mu,nu,[](const Eigen::Vector3d&) {},
                [&](const Eigen::Vector3d&,const Eigen::Vector3d&) {candidates=admission::add(candidates,1);},
                [](const Eigen::Vector3d&) {});
            return candidates;
        };
        const Count candidates=preparation?preparation->support({index,bra_index,ket_index},count):count();
        receipt.multipole_candidate_images=admission::add(receipt.multipole_candidate_images,candidates);
        if(!candidates) {active=false;return;}
        tolerance=static_cast<long double>(o.coupling_multipole_error)/candidates;
        tolerance*=1-4096*std::numeric_limits<double>::epsilon();
        if(preparation) {prepared_auxiliary=preparation->aux(shell,index,receipt);auxiliary=prepared_auxiliary->extent;}
        else {
            auxiliary=detail::auxiliary_extent(shell,index,extents,receipt.multipole_envelope_evaluations,always_near);
            receipt.multipole_primitive_products+=shell.nprim();
        }
    }
    void start(const libint2::Shell& a,const libint2::Shell& b,Count i,Count j) {
        selected_images.clear();
        if(!active) return;
        if(preparation) {prepared_pair=preparation->pair(a,b,i,j,pair_ordinal,receipt);pair=prepared_pair->extent;}
        else {
            pair=detail::pair_extent(a,b,{i,j},extents,receipt.multipole_envelope_evaluations,always_near);
            receipt.multipole_primitive_products+=a.nprim()*b.nprim();
        }
        partition_key[3]=pair_ordinal++;pair_moments.resize(0,0);pair_values=nullptr;
    }
    const md::Matrix& auxiliary_panel(const libint2::Shell& shell) {
        if(!auxiliary_values) {
            auto evaluate=[&] {return md::shell_auxiliary_moments(shell,shell.O,options.coupling_multipole_order);};
            if(prepared_auxiliary) auxiliary_values=&preparation->moments(prepared_auxiliary,evaluate,shell.nprim(),receipt);
            else {
                auxiliary_moments=evaluate();auxiliary_values=&auxiliary_moments;
                ++receipt.multipole_moment_panels;receipt.multipole_primitive_products+=shell.nprim();
            }
        }
        return *auxiliary_values;
    }
    const md::Matrix& pair_panel(const libint2::Shell& a,const libint2::Shell& b) {
        if(!pair_values) {
            auto evaluate=[&] {return md::shell_moments(a,b,pair.center,options.coupling_multipole_order);};
            if(prepared_pair) pair_values=&preparation->moments(prepared_pair,evaluate,a.nprim()*b.nprim(),receipt);
            else {
                pair_moments=evaluate();pair_values=&pair_moments;
                ++receipt.multipole_moment_panels;receipt.multipole_primitive_products+=a.nprim()*b.nprim();
            }
        }
        return *pair_values;
    }
    md::Matrix evaluate(const libint2::Shell& shifted_aux,const libint2::Shell& a,const libint2::Shell& b) {
        if(active && !auxiliary.always_near && !pair.always_near) {
            auto translated=auxiliary;translated.center=shifted_aux.O;
            long double distance=0.;
            for(int i=0;i<3;++i) {const long double d=static_cast<long double>(translated.center[i])-pair.center[i];distance+=d*d;}
            const long double rho=static_cast<long double>(translated.radius)+pair.radius;
            if(std::sqrt(distance)>rho*(1+128*std::numeric_limits<double>::epsilon())) {
                ++receipt.multipole_trials;
                // The enclosing producer has already admitted this workspace.
                ResourceBudget leaf;leaf.other_live_bytes=0;leaf.backend_margin_bytes=65536;
                leaf.memory_bytes=leaf.shell_calls=leaf.primitive_quartets=leaf.scalar_terms=leaf.kernel_terms=std::numeric_limits<Count>::max();
                std::optional<LatticeGaussianInteraction> interaction;
                try { interaction=erfc_gaussian_interaction(options.omega,translated,pair,options.coupling_multipole_order,leaf); }
                catch(const std::overflow_error&) {} // Unrepresentable bound forces direct evaluation.
                if(interaction && interaction->error.total<=tolerance) {
                    md::Matrix values=auxiliary_panel(shifted_aux)*interaction->matrix.values*pair_panel(a,b).transpose();
                    if(!values.allFinite()) throw std::overflow_error("MANTA periodic multipole contraction is nonfinite");
                    ++receipt.multipole_images;error+=interaction->error.total;
                    return values;
                }
            }
        }
        ++receipt.three_center_shell_calls;receipt.primitive_products+=shifted_aux.nprim()*a.nprim()*b.nprim();
        return md::shell_coulomb({&shifted_aux,nullptr,&a,&b},md::RadialKernel::Erfc,options.omega);
    }
    bool hierarchical() const { return active && options.coupling_multipole_hierarchy; }
    void accumulate(const Eigen::Vector3d& t,const libint2::Shell& shell,const libint2::Shell& a,
        const libint2::Shell& b,const Eigen::Vector3d& q,std::vector<Complex>& total,std::vector<Complex>& correction) {
        if(hierarchical()) {selected_images.push_back(t);return;}
        const auto values=evaluate(shifted(shell,t),a,b);const auto phase=fitting_phase(-q.dot(t),receipt);
        for(Count i=0;i<total.size();++i) {fitting_add(phase*values.data()[i],total[i],correction[i]);++receipt.phase_accumulations;}
    }
    void finish_pair(const libint2::Shell& shell,const libint2::Shell& a,const libint2::Shell& b,
        const Eigen::Vector3d& q,std::vector<Complex>& total,std::vector<Complex>& correction) {
        if(!hierarchical() || selected_images.empty()) return;
        ResourceBudget leaf;leaf.other_live_bytes=0;leaf.backend_margin_bytes=65536;
        leaf.memory_bytes=leaf.shell_calls=leaf.primitive_quartets=leaf.scalar_terms=leaf.kernel_terms=std::numeric_limits<Count>::max();
        std::vector<md::Point> centers;centers.reserve(selected_images.size());
        for(const auto& t:selected_images) centers.push_back(shifted(shell,t).O);
        auto build_tree=[&] {return bupo_detail::AuxiliaryImageBubbles::build(auxiliary,centers,receipt.three_center_source_identity,leaf);};
        auto certify=[&](const BubbleNode& node,const BubbleEnvelope& envelope)->std::optional<LatticeGaussianInteraction> {
            // Keep the original hypot arithmetic and strict separation test.
            const long double distance=std::hypot(static_cast<long double>(node.center[0]-pair.center[0]),
                static_cast<long double>(node.center[1]-pair.center[1]),static_cast<long double>(node.center[2]-pair.center[2]));
            if(node.always_near || pair.always_near || distance<=(static_cast<long double>(node.radius)+pair.radius)*(1+128*std::numeric_limits<double>::epsilon())) return {};
            AuxiliaryExtent target;target.shell=auxiliary.shell;target.center=node.center;target.radius=node.radius;
            target.full_moment_bounds=envelope.full_moments;target.tail_moment_bounds=envelope.tail_moments;target.potential_bound=envelope.potential;
            ++receipt.multipole_trials;
            try {
                auto value=erfc_gaussian_interaction(options.omega,target,pair,options.coupling_multipole_order,leaf);
                if(value.error.total<=tolerance) return value;
            } catch(const std::overflow_error&) {} // Refine an unrepresentable envelope.
            return {};
        };
        std::shared_ptr<const PeriodicGDFPartitionCache::Entry> partition;
        if(partitions) {
            using namespace admission;
            const Count h=Count(options.coupling_multipole_order+1)*(options.coupling_multipole_order+2)*(options.coupling_multipole_order+3)/6;
            const auto tp=bupo_detail::AuxiliaryImageBubbles::plan(centers.size(),options.coupling_multipole_order,leaf);
            const Count upper=add(4096,add(tp.modeled_peak_bytes,mul(tp.maximum_nodes,add(256,mul(8,square(h))))));
            partition=partitions->get(partition_key,centers.size(),b.O,upper,[&] {
                auto result=std::make_shared<PeriodicGDFPartitionCache::Entry>();result->tree=build_tree();
                const auto& ns=result->tree->nodes();result->steps.reserve(ns.size());
                const BubbleErrorModel bounds(result->tree,options.coupling_multipole_order,leaf);
                auto select=[&](auto&& self,Count id)->void {
                    auto interaction=certify(ns[id],bounds.envelopes()[id]);
                    if(interaction) result->steps.push_back({id,std::move(interaction->matrix.values),interaction->error.total});
                    else if(ns[id].child_count) for(int c=0;c<ns[id].child_count;++c) self(self,ns[id].children[c]);
                    else result->steps.push_back({id,{},0.});
                };
                select(select,0);++receipt.multipole_partition_builds;
                result->bytes=add(4096,add(result->tree->retained_bytes(),mul(result->steps.capacity(),sizeof(PeriodicGDFPartitionCache::Step))));
                for(const auto& step:result->steps) result->bytes=add(result->bytes,mul(8,step.interaction.size()));
                return result;
            },receipt);
        }
        const auto tree=partition?partition->tree:build_tree();
        std::optional<BubbleErrorModel> envelopes;
        if(!partition) {envelopes.emplace(tree,options.coupling_multipole_order,leaf);++receipt.multipole_partition_builds;}
        const auto& nodes=tree->nodes();const auto& order=tree->pair_order();
        receipt.multipole_tree_nodes=admission::add(receipt.multipole_tree_nodes,nodes.size());
        std::vector<md::ComplexMatrix> moments(nodes.size());
        std::vector<Eigen::VectorXd> magnitudes(action?nodes.size():0);
        std::vector<Eigen::VectorXcd> fields(action && action->forward?nodes.size():0);
        std::vector<bool> ready(nodes.size());
        Eigen::VectorXd auxiliary_magnitude;
        const bool need_charge=!action || !action->forward;
        auto shift=[](const md::Point& from,const md::Point& to) {
            md::Point d;for(int k=0;k<3;++k)d[k]=from[k]-to[k];return d;
        };
        // M(parent)=sum W(child-parent) M(child). Unit-modulus q phases
        // stay on the auxiliary images; k phases remain on the ordered pair.
        // Every accepted node uses these same moments in both paired actions.
        auto upward=[&](auto&& self,Count id)->const md::ComplexMatrix& {
            auto& result=moments[id];if(ready[id]) return result;
            const auto& auxiliary_moments=auxiliary_panel(shell);
            const Count h=auxiliary_moments.cols();
            if(need_charge) result=md::ComplexMatrix::Zero(action?1:shell.size(),h);
            if(action) {
                magnitudes[id]=Eigen::VectorXd::Zero(h);
                if(!auxiliary_magnitude.size()) auxiliary_magnitude=auxiliary_moments.cwiseAbs().colwise().maxCoeff().transpose();
            }
            md::ComplexMatrix errors=md::ComplexMatrix::Zero(result.rows(),result.cols());
            auto include=[&](const md::ComplexMatrix& v) {
                for(Count k=0;k<Count(v.size());++k) {fitting_add(v.data()[k],result.data()[k],errors.data()[k]);++receipt.phase_accumulations;}
            };
            const auto& node=nodes[id];
            if(node.child_count) {
                for(int c=0;c<node.child_count;++c) {
                    const Count child=node.children[c];const auto& m=self(self,child);
                    const auto w=multipole_translation(shift(nodes[child].center,node.center),options.coupling_multipole_order,leaf);
                    if(need_charge) {
                        include(m*w.values.transpose());
                        if(action) action->products(admission::square(h));
                    }
                    if(action) {
                        if(action->verified) {
                            for(Count i=0;i<h;++i)for(Count j=0;j<h;++j)
                                magnitudes[id][i]=verified_coupling::plus_up(magnitudes[id][i],
                                    verified_coupling::times_up(std::abs(w.values(i,j)),magnitudes[child][j]));
                        } else magnitudes[id]+=w.values.cwiseAbs()*magnitudes[child];
                    }
                    ++receipt.multipole_translations;
                }
            } else for(Count k=node.begin;k<node.end;++k) {
                const Count image=order[k];const auto phase=fitting_phase(-q.dot(selected_images[image]),receipt);
                const auto displacement=shift(centers[image],node.center);
                // Singleton leaves share their image center. Preserve the
                // exact identity without rebuilding a dense W for each image.
                if(displacement==md::Point{}) {
                    if(need_charge) {
                        if(action) {include(phase*action->auxiliary_input(auxiliary_moments,prepared_auxiliary).transpose());action->products(h);}
                        else include(phase*auxiliary_moments);
                    }
                    if(action) {
                        if(action->verified) {
                            const double phase_bound=verified_coupling::magnitude(verified_coupling::point(phase));
                            for(Count i=0;i<h;++i)magnitudes[id][i]=verified_coupling::plus_up(magnitudes[id][i],
                                verified_coupling::times_up(phase_bound,auxiliary_magnitude[i]));
                        } else magnitudes[id]+=auxiliary_magnitude;
                    }
                } else {
                    const auto w=multipole_translation(displacement,options.coupling_multipole_order,leaf);
                    if(need_charge) {
                        if(action) {
                            include(phase*(w.values*action->auxiliary_input(auxiliary_moments,prepared_auxiliary)).transpose());
                            action->products(admission::add(admission::square(h),h));
                        }
                        else include(phase*(auxiliary_moments*w.values.transpose()));
                    }
                    if(action) {
                        if(action->verified) {
                            const double phase_bound=verified_coupling::magnitude(verified_coupling::point(phase));
                            for(Count i=0;i<h;++i)for(Count j=0;j<h;++j)
                                magnitudes[id][i]=verified_coupling::plus_up(magnitudes[id][i],
                                    verified_coupling::times_up(phase_bound,verified_coupling::times_up(std::abs(w.values(i,j)),auxiliary_magnitude[j])));
                        } else magnitudes[id]+=w.values.cwiseAbs()*auxiliary_magnitude;
                    }
                }
                ++receipt.multipole_translations;
            }
            if(action && !action->verified) for(Eigen::Index i=0;i<magnitudes[id].size();++i)
                magnitudes[id][i]=periodic_error::upper(magnitudes[id][i]);
            ready[id]=true;return result;
        };
        auto append=[&](const md::ComplexMatrix& values) {
            if(!values.allFinite()) throw std::overflow_error("MANTA hierarchical GDF contraction is nonfinite");
            for(Count i=0;i<total.size();++i) {fitting_add(values.data()[i],total[i],correction[i]);++receipt.phase_accumulations;}
        };
        auto far=[&](Count id,const md::Matrix& interaction,double bound) {
            const auto& pm=pair_panel(a,b);const auto& am=upward(upward,id);
            if(action) {
                action->prepare_pair(pm,prepared_pair,partition_key);action->bound(magnitudes[id],interaction);
                if(action->forward) fields[id]=interaction*action->pair_input();
                else action->add_pair_field(interaction.transpose()*am.row(0).transpose());
                action->products(admission::square(pm.cols()));
            } else append(am*interaction*pm.transpose());
            const Count represented=nodes[id].end-nodes[id].begin;
            receipt.multipole_images=admission::add(receipt.multipole_images,represented);++receipt.multipole_far_blocks;
            // Uniform descendant bounds, never density/phase cancellation.
            error+=static_cast<long double>(represented)*bound;
        };
        auto near=[&](Count id) {
            const auto& node=nodes[id];
            for(Count k=node.begin;k<node.end;++k) {
                const auto& t=selected_images[order[k]];const auto bra=shifted(shell,t);
                ++receipt.three_center_shell_calls;receipt.primitive_products+=bra.nprim()*a.nprim()*b.nprim();
                const auto value=md::shell_coulomb({&bra,nullptr,&a,&b},md::RadialKernel::Erfc,options.omega);
                append(fitting_phase(-q.dot(t),receipt)*value);
            }
        };
        if(partition) {
            for(const auto& step:partition->steps) {
                if(step.interaction.size()) far(step.node,step.interaction,step.error);
                else near(step.node);
            }
        } else {
            auto visit=[&](auto&& self,Count id)->void {
                const auto interaction=certify(nodes[id],envelopes->envelopes()[id]);
                if(interaction) far(id,interaction->matrix.values,interaction->error.total);
                else if(nodes[id].child_count) for(int c=0;c<nodes[id].child_count;++c) self(self,nodes[id].children[c]);
                else near(id);
            };
            visit(visit,0);
        }
        // Adjoint of the same upward translations. Forward fields propagate
        // to auxiliary leaves; reverse fields project to the ordered AO pair.
        // Structural activity follows accepted nodes, never input values.
        if(action && action->pair_ready) {
            if(action->forward) {
                const auto& ap=auxiliary_panel(shell);
                for(Count id=0;id<nodes.size();++id) {
                    if(!fields[id].size()) continue;
                    const auto& node=nodes[id];
                    if(node.child_count) for(int c=0;c<node.child_count;++c) {
                        const Count child=node.children[c];
                        const auto w=multipole_translation(shift(nodes[child].center,node.center),options.coupling_multipole_order,leaf);
                        // Accepted nodes form an antichain: an accepted parent
                        // has no separately accepted descendant contribution.
                        if(fields[child].size()) throw std::logic_error("MANTA compressed far partition overlaps");
                        fields[child]=w.values.transpose()*fields[id];
                        ++action->receipt.multipole_action_translations;action->products(admission::square(ap.cols()));
                    } else for(Count k=node.begin;k<node.end;++k) {
                        const Count image=order[k];const auto phase=fitting_phase(-q.dot(selected_images[image]),receipt);
                        const auto displacement=shift(centers[image],node.center);
                        if(displacement==md::Point{}) {
                            if(action->global) action->add_auxiliary_field(ap,std::conj(phase)*fields[id],prepared_auxiliary);
                            else action->add_auxiliary(std::conj(phase)*(ap*fields[id]));
                        } else {
                            const auto w=multipole_translation(displacement,options.coupling_multipole_order,leaf);
                            if(action->global) action->add_auxiliary_field(ap,std::conj(phase)*(w.values.transpose()*fields[id]),prepared_auxiliary);
                            else action->add_auxiliary(std::conj(phase)*(ap*(w.values.transpose()*fields[id])));
                            action->products(admission::square(ap.cols()));
                        }
                        ++action->receipt.multipole_action_translations;action->products(action->global?Count(ap.cols()):admission::mul(ap.rows(),admission::add(ap.cols(),1)));
                    }
                }
            } else action->finish_pair(pair_panel(a,b));
        }
    }
    void finish() {
        if(active) receipt.evaluated_multipole_error=std::max(receipt.evaluated_multipole_error,periodic_error::upper(error));
    }
};
}
namespace {
void add_pair_sr_receipt(PeriodicGDFShortRangeReceipt&,const PeriodicGDFShortRangeReceipt&);
ResourceBudget gdf_leaf_budget(ResourceBudget);
void gdf_caps(const PeriodicGDFPlan&,const ResourceBudget&);
}
PeriodicGDFShortRangePlan plan_periodic_gdf_short_range(const BasisSet& orbital,const BasisSet& aux,
    const PeriodicSystem& system,Count nk,const PeriodicGDFShortRangeOptions& o,const ResourceBudget& budget) {
    using namespace admission;
    if(o.coupling_multipole_pair_tree) {
        auto local=o;local.coupling_multipole_pair_tree=false;
        auto p=plan_periodic_gdf_short_range(orbital,aux,system,nk,local,gdf_leaf_budget(budget));
        const auto tiles=plan_periodic_gdf_tiles(orbital,aux,system,1,nk,o,gdf_leaf_budget(budget));
        p.modeled_peak_bytes=add(p.modeled_peak_bytes,add(tiles.modeled_peak_bytes,budget.other_live_bytes));
        p.shell_calls=add(p.shell_calls,tiles.coupling_shell_calls);
        p.three_center_shell_calls=add(p.three_center_shell_calls,tiles.coupling_shell_calls);
        p.primitive_products=add(p.primitive_products,tiles.coupling_primitive_products);
        p.scalar_terms=add(p.scalar_terms,tiles.coupling_scalar_terms);p.kernel_terms=add(p.kernel_terms,tiles.coupling_kernel_terms);
        fitting_caps(p,budget);return p;
    }
    margin(budget);validate(system,Eigen::Vector3d::Zero(),o.omega,o.pair_cutoff_bohr);
    validate(system,Eigen::Vector3d::Zero(),o.omega,o.auxiliary_cutoff_bohr);
    if(!nk) throw std::invalid_argument("MANTA periodic fitting requires nonempty ket momenta");
    cap(nk,std::numeric_limits<Eigen::Index>::max()/3,"MANTA periodic fitting momenta are not addressable");
    const MetricImageDomain metric(system,o.auxiliary_cutoff_bohr,o.omega,o.image_candidate_cap,o.integral_screen_error);
    const ThreeCenterImageDomain three(system,o.pair_cutoff_bohr,o.auxiliary_cutoff_bohr,o.omega,o.image_candidate_cap,coupling_screen(o));
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
    p.multipole=plan_coupling_multipoles(orbital,aux,p.pair_image_candidates,p.auxiliary_image_candidates,o,leaf);
    p.shell_calls=add(p.shell_calls,p.multipole.shell_calls);p.primitive_products=add(p.primitive_products,p.multipole.primitive_products);
    p.scalar_terms=add(p.scalar_terms,p.multipole.scalar_terms);p.kernel_terms=add(p.kernel_terms,p.multipole.kernel_terms);
    p.modeled_peak_bytes=add(p.modeled_peak_bytes,p.multipole.modeled_peak_bytes);
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
    r.integral_screen_error=o.integral_screen_error;r.coupling_screen_error=o.coupling_screen_error;r.coupling_multipole_error=o.coupling_multipole_error;r.coupling_multipole_hierarchy=o.coupling_multipole_hierarchy;r.coupling_multipole_pair_tree=o.coupling_multipole_pair_tree;r.modeled_peak_bytes=p.modeled_peak_bytes;
    r.orbital_basis_identity=auxiliary_basis_content_identity_sha256(orbital);
    r.auxiliary_basis_identity=auxiliary_basis_content_identity_sha256(aux);
    if(o.compute_metric) r.metric_source_identity=fitting_identity(false,system,q,k,o,r);
    r.three_center_source_identity=fitting_identity(true,system,q,k,o,r);
    FittingDigest combined;combined.text("manta.periodic-gdf-short-range.joint.v1");
    combined.text(r.metric_source_identity);combined.text(r.three_center_source_identity);r.source_identity=combined.hash.finish_hex();
    const auto& a=aux.libint();const auto& b=orbital.libint();const auto ai=a.shell2bf(),bi=b.shell2bf();
    const Count na=p.n_aux,no=p.n_orb,nk=p.momenta;
    const MetricImageDomain metric(system,o.auxiliary_cutoff_bohr,o.omega,o.image_candidate_cap,o.integral_screen_error);
    const ThreeCenterImageDomain three(system,o.pair_cutoff_bohr,o.auxiliary_cutoff_bohr,o.omega,o.image_candidate_cap,coupling_screen(o));
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
    if(o.coupling_multipole_pair_tree) {
        auto source_options=o;source_options.compute_metric=false;
        const md::Matrix vectors=q.transpose();
        const PeriodicGDFTiles source(orbital,aux,system,q,k,vectors,source_options,gdf_leaf_budget(budget));
        for(Count ip=0;ip<a.size();++ip)for(Count im=0;im<b.size();++im)for(Count in=0;in<b.size();++in) {
            const auto tile=source.pair_tree_tile(ip,im,in);const Count np=a[ip].size(),nm=b[im].size(),nn=b[in].size();
            for(Count ik=0;ik<nk;++ik)for(Count i=0;i<np;++i)for(Count j=0;j<nm;++j)for(Count v=0;v<nn;++v)
                result.three_center_values[((ik*na+ai[ip]+i)*no+bi[im]+j)*no+bi[in]+v]=tile.values[((ik*np+i)*nm+j)*nn+v];
            add_pair_sr_receipt(r,tile.receipt.short_range);
        }
        cap(r.three_center_shell_calls,p.three_center_shell_calls,"MANTA global SR executed shell bound exceeded");
        cap(add(r.primitive_products,r.multipole_primitive_products),p.primitive_products,"MANTA global SR executed primitive bound exceeded");
        return result;
    }
    for(Count ip=0;ip<a.size();++ip) for(Count im=0;im<b.size();++im) for(Count in=0;in<b.size();++in) {
        const Count np=a[ip].size(),nm=b[im].size(),nn=b[in].size(),tile=mul(np,mul(nm,nn));
        std::vector<Complex> contracted(tile),correction(tile);auto ket=b[in];
        OrderedCouplingMultipoles multipole(a[ip],b[im],b[in],ip,three,o,r);
        three.each(a[ip],b[im],b[in],[&](const Eigen::Vector3d& t) {
            ket=shifted(b[in],t);multipole.start(b[im],ket,im,in);std::fill(contracted.begin(),contracted.end(),Complex{});
            std::fill(correction.begin(),correction.end(),Complex{});++r.pair_shell_images;
        },[&](const Eigen::Vector3d&,const Eigen::Vector3d& t) {
            multipole.accumulate(t,a[ip],b[im],ket,q,contracted,correction);
        },[&](const Eigen::Vector3d& t) {
            multipole.finish_pair(a[ip],b[im],ket,q,contracted,correction);
            for(Count ik=0;ik<nk;++ik) {
                const auto phase=fitting_phase(k.row(ik).dot(t),r);
                for(Count i=0;i<np;++i) for(Count j=0;j<nm;++j) for(Count l=0;l<nn;++l) {
                    const Count out=((ik*na+ai[ip]+i)*no+bi[im]+j)*no+bi[in]+l;
                    fitting_add(phase*contracted[(i*nm+j)*nn+l],result.three_center_values[out],errors[out]);++r.phase_accumulations;
                }
            }
        });
        multipole.finish();
    }
    cap(r.metric_shell_calls,p.metric_shell_calls,"MANTA periodic metric executed-work bound exceeded");
    cap(r.three_center_shell_calls,p.three_center_shell_calls,"MANTA periodic coupling executed-work bound exceeded");
    cap(add(r.primitive_products,r.multipole_primitive_products),p.primitive_products,"MANTA periodic fitting executed primitive bound exceeded");
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
        effective.coupling_multipole_pair_tree=false;
        effective.omega=effective.auxiliary_cutoff_bohr=effective.integral_screen_error=effective.coupling_screen_error=effective.coupling_multipole_error=0.;
    }
    sr.omega=effective.omega;sr.pair_cutoff_bohr=o.pair_cutoff_bohr;sr.auxiliary_cutoff_bohr=effective.auxiliary_cutoff_bohr;
    sr.integral_screen_error=effective.integral_screen_error;sr.coupling_screen_error=effective.coupling_screen_error;sr.coupling_multipole_error=effective.coupling_multipole_error;sr.coupling_multipole_hierarchy=effective.coupling_multipole_error>0. && effective.coupling_multipole_hierarchy;sr.coupling_multipole_pair_tree=effective.coupling_multipole_pair_tree;sr.modeled_peak_bytes=peak;
    if(o.compute_metric) sr.metric_source_identity=fitting_identity(false,system,q,k,effective,sr);
    sr.three_center_source_identity=fitting_identity(true,system,q,k,effective,sr);
    FittingDigest sj;sj.text("manta.periodic-gdf-short-range.joint.v1");sj.text(sr.metric_source_identity);
    sj.text(sr.three_center_source_identity);sr.source_identity=sj.hash.finish_hex();return sr;
}
void describe_coupling_reference(PeriodicGDFReceipt& r,const BasisSet& orbital,const BasisSet& aux,
    const PeriodicSystem& system,const Eigen::Vector3d& q,const md::Matrix& k,const md::Matrix& vectors,
    const PeriodicGDFShortRangeOptions& o) {
    r.coupling_error_bound=r.real_short_range?std::max(o.coupling_screen_error,o.coupling_multipole_error):0.;
    r.reference_three_center_source_identity=r.three_center_source_identity;
    if(r.coupling_error_bound>0.) {
        auto exact=o;exact.coupling_screen_error=exact.coupling_multipole_error=0.;exact.coupling_multipole_pair_tree=false;auto reference=r;
        reference.short_range=describe_gdf_short_range(orbital,aux,system,q,k,exact,0,r.real_short_range);
        r.reference_three_center_source_identity=gdf_identity(false,vectors,reference);
    }
}
}

PeriodicGDFPlan plan_periodic_gdf(const BasisSet& orbital,const BasisSet& aux,
    const PeriodicSystem& system,Count nv,Count nk,const PeriodicGDFShortRangeOptions& o,const ResourceBudget& budget,
    const PeriodicGDFKernelOptions& physical) {
    using namespace admission;
    margin(budget);validate_physical_gdf(physical,system);
    if(o.coupling_multipole_pair_tree) {
        auto local=o;local.coupling_multipole_pair_tree=false;
        auto p=plan_periodic_gdf(orbital,aux,system,nv,nk,local,gdf_leaf_budget(budget),physical);
        const auto tiles=plan_periodic_gdf_tiles(orbital,aux,system,nv,nk,o,gdf_leaf_budget(budget),physical);
        p.modeled_peak_bytes=add(p.modeled_peak_bytes,add(tiles.modeled_peak_bytes,budget.other_live_bytes));
        p.shell_calls=add(p.shell_calls,tiles.shell_calls);p.primitive_products=add(p.primitive_products,tiles.primitive_products);
        p.scalar_terms=add(p.scalar_terms,tiles.scalar_terms);p.kernel_terms=add(p.kernel_terms,tiles.kernel_terms);
        // Materialized shell exports repeat Fourier panels per shell triple.
        p.vector_primitive_products=add(p.vector_primitive_products,tiles.primitive_products);
        gdf_caps(p,budget);return p;
    }
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
    if(o.coupling_multipole_pair_tree) {
        const PeriodicGDFTiles source(orbital,aux,system,q,k,vectors,o,gdf_leaf_budget(budget),physical);
        r=source.source();result.three_center_values.resize(mul(p.momenta,mul(p.n_aux,square(p.n_orb))));
        auto include=[&](const PeriodicGDFTile& tile) {
            const auto& t=tile.receipt;add_pair_sr_receipt(r.short_range,t.short_range);
            r.shell_calls=add(r.shell_calls,t.shell_calls);r.primitive_products=add(r.primitive_products,t.primitive_products);
            r.vector_primitive_products=add(r.vector_primitive_products,t.vector_primitive_products);
            r.phase_products=add(r.phase_products,t.phase_products);r.phase_accumulations=add(r.phase_accumulations,t.phase_accumulations);
            r.auxiliary_fourier_shell_calls=add(r.auxiliary_fourier_shell_calls,t.auxiliary_fourier_shell_calls);
            r.pair_fourier_shell_calls=add(r.pair_fourier_shell_calls,t.pair_fourier_shell_calls);
            r.contraction_terms=add(r.contraction_terms,t.contraction_terms);r.reciprocal_vectors=std::max(r.reciprocal_vectors,t.reciprocal_vectors);
            r.evaluated_coupling_error_bound=std::max(r.evaluated_coupling_error_bound,t.evaluated_coupling_error_bound);
        };
        if(o.compute_metric) {const auto metric=source.metric();result.metric_values=metric.values;include(metric);}
        for(Count ip=0;ip<aux.nshells();++ip)for(Count im=0;im<orbital.nshells();++im)for(Count in=0;in<orbital.nshells();++in) {
            const auto tile=source.evaluate(ip,im,in);include(tile);const Count np=tile.shape[1],nm=tile.shape[2],nn=tile.shape[3];
            for(Count ik=0;ik<p.momenta;++ik)for(Count i=0;i<np;++i)for(Count j=0;j<nm;++j)for(Count v=0;v<nn;++v)
                result.three_center_values[((ik*p.n_aux+tile.offsets[1]+i)*p.n_orb+tile.offsets[2]+j)*p.n_orb+tile.offsets[3]+v]=tile.values[((ik*np+i)*nm+j)*nn+v];
        }
        r.coupling_error_evaluated=true;r.modeled_peak_bytes=p.modeled_peak_bytes;
        cap(r.shell_calls,p.shell_calls,"MANTA global dense GDF executed shell bound exceeded");
        cap(r.primitive_products,p.primitive_products,"MANTA global dense GDF primitive bound exceeded");
        cap(r.vector_primitive_products,p.vector_primitive_products,"MANTA global dense GDF Fourier bound exceeded");
        return result;
    }
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
    describe_coupling_reference(r,orbital,aux,system,q,k,vectors,o);
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
    r.shell_calls+=r.short_range.multipole_moment_panels;r.primitive_products+=r.short_range.multipole_primitive_products;
    r.evaluated_coupling_error_bound=r.real_short_range?std::max(o.coupling_screen_error,r.short_range.evaluated_multipole_error):0.;
    r.coupling_error_evaluated=true;
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
    r.shell_calls+=r.short_range.metric_shell_calls+r.short_range.three_center_shell_calls+r.short_range.multipole_moment_panels;
    r.primitive_products+=r.short_range.primitive_products+r.short_range.multipole_primitive_products;
    if(tile.shape.size()==4) {
        r.evaluated_coupling_error_bound=r.real_short_range?std::max(r.short_range.coupling_screen_error,r.short_range.evaluated_multipole_error):0.;
        r.coupling_error_evaluated=true;
    }
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
    auto local=o;local.coupling_multipole_pair_tree=false;
    const auto sr=plan_periodic_gdf_short_range(orbital,aux,system,nk,local,leaf);
    if(o.coupling_multipole_pair_tree) {
        coupling_screen(o);
        if(physical.kernel=="erf") throw std::invalid_argument("MANTA global pair tree requires a real short-range source");
    }
    const auto a=census(aux,7),b=census(orbital,6);
    PeriodicGDFTilesPlan p;p.n_aux=aux.nbasis();p.n_orb=orbital.nbasis();p.momenta=nk;p.vectors=nv;
    p.panel_vectors=std::min<Count>(16,nv);p.compute_metric=o.compute_metric;p.external_live_bytes=budget.other_live_bytes;
    p.coupling_error_bound=physical.kernel=="erf"?0.:std::max(o.coupling_screen_error,o.coupling_multipole_error);
    const Count panels=add(nv/16+(nv%16!=0),1),images=sr.pair_image_candidates;
    const auto af=plan_fourier(aux,nullptr,p.panel_vectors,leaf),bf=plan_fourier(orbital,&orbital,p.panel_vectors,leaf);
    const Count pair_shells=square(orbital.nshells()),aux_shells=aux.nshells(),pairs=square(p.n_orb);
    p.maximum_tile_elements=mul(nk,sr.three_center.tile_elements);
    cap(p.maximum_tile_elements,std::numeric_limits<Eigen::Index>::max(),"MANTA GDF shell tile is not addressable");
    // Full basis snapshots and numerical import/binding copies. Only the
    // lattice/dimension are retained from PeriodicSystem, never its symmetry
    // metadata or nuclear state. Those remain with the consuming method.
    p.retained_bytes=add(65536,add(mul(4,add(a.bytes,b.bytes)),mul(96,add(nv,add(nk,1)))));
    if(o.coupling_multipole_error>0. && o.coupling_multipole_reuse && physical.kernel!="erf") {
        const Count moments=Count(o.coupling_multipole_order+1)*(o.coupling_multipole_order+2)*(o.coupling_multipole_order+3)/6;
        p.preparation_pair_slots=mul(images,pair_shells);p.preparation_auxiliary_slots=aux_shells;
        p.preparation_support_slots=mul(aux_shells,pair_shells);
        const Count slots=add(p.preparation_pair_slots,p.preparation_auxiliary_slots);
        const Count elements=mul(moments,add(mul(images,pairs),p.n_aux));
        // Per-source cache contains only independent pair-image and auxiliary
        // extents/moments, never an auxiliary-by-pair interaction list or T/B.
        // Reserve every possible slot even when it is not yet materialized.
        p.preparation_cache_bytes=add(8192,add(mul(256,p.preparation_support_slots),add(mul(PeriodicGDFMomentCache::entry_bytes(o.coupling_multipole_order),slots),mul(8,elements))));
        p.retained_bytes=add(p.retained_bytes,p.preparation_cache_bytes);
    }
    if(o.coupling_multipole_error>0. && (o.coupling_multipole_hierarchy || o.coupling_multipole_pair_tree) && o.coupling_multipole_reuse &&
       o.coupling_multipole_partition_bytes && physical.kernel!="erf") {
        p.partition_cache_bytes=add(8192,o.coupling_multipole_partition_bytes);
        // One borrowed or constructing entry can outlive eviction. Reserve
        // the entire entry cap separately from the retained working set.
        p.partition_workspace_bytes=o.coupling_multipole_partition_bytes;
        p.retained_bytes=add(p.retained_bytes,p.partition_cache_bytes);
    }
    if(o.coupling_multipole_pair_tree) {
        const auto gp=gdf_pair_geometry_plan(p.preparation_pair_slots,orbital.nshells(),aux.nshells(),nk,system,o,leaf);
        const auto mp=plan_multipoles(o.coupling_multipole_order,leaf);
        const auto interaction=plan_erfc_gaussian_interaction(o.coupling_multipole_order,leaf);
        const Count nodes=gp.node_capacity,h=mp.moments,slots=gp.pair_capacity;
        p.global_pair_slots=slots;p.global_auxiliary_slots=aux_shells;
        p.pair_tree_bytes=add(gp.modeled_peak_bytes,add(8192,mul(nodes,add(512,mul(128,o.coupling_multipole_order+1)))));
        p.pair_tree_bytes=add(p.pair_tree_bytes,mul(512,slots)); // Temporary conjugate-orbit map and leaf indices.
        p.retained_bytes=add(p.retained_bytes,p.pair_tree_bytes);
        // Lazy moment charges, compensated fields, uniform positive envelopes,
        // phase/fold vectors, near blocks and one translation/interaction workspace.
        p.pair_tree_action_bytes=add(16384,add(mul(256,mul(nodes,h)),add(mul(slots,add(2048,
            add(mul(64,nk),mul(128,square(b.maximum_size))))),add(mul(256,square(h)),mul(aux_shells,add(1024,mul(128,h)))))));
        // An uncached/bypassed auxiliary image still owns its complete
        // antichain of interaction matrices until application. This workspace
        // is required even when the retained partition byte limit is zero.
        p.pair_tree_action_bytes=add(p.pair_tree_action_bytes,
            add(4096,mul(nodes,add(256,mul(8,square(h))))));
        // Per-node and per-shell-pair positive error/magnitude ledgers plus
        // call-private input/output block norms. One auxiliary shell at a time.
        p.pair_tree_action_bytes=add(p.pair_tree_action_bytes,
            add(4096,add(mul(64,nodes),add(mul(128,pair_shells),mul(32,aux_shells)))));
        const Count queries=mul(aux_shells,mul(gp.auxiliary_image_capacity,nodes));
        // Include repeated materialized shell exports, even though paired
        // actions traverse the tree once. No cache-hit savings in admission.
        p.pair_tree_scalar_terms=add(gp.scalar_terms,mul(pair_shells,add(mul(queries,add(4096,interaction.scalar_terms)),
            mul(nodes,add(mp.scalar_terms,mul(256,square(h)))))));
        p.pair_tree_scalar_terms=add(p.pair_tree_scalar_terms,mul(65536,add(slots,mul(nodes,square(o.coupling_multipole_order+1)))));
        p.pair_tree_scalar_terms=add(p.pair_tree_scalar_terms,mul(256,mul(pair_shells,
            add(mul(aux_shells,add(nodes,pair_shells)),add(mul(nk,pairs),p.n_aux)))));
        const Count packing=mul(16,mul(pair_shells,mul(queries,square(h))));
        p.pair_tree_scalar_terms=add(p.pair_tree_scalar_terms,packing);
        p.coupling_kernel_terms=add(mul(8,packing),add(gp.kernel_terms,mul(pair_shells,add(mul(queries,interaction.kernel_terms),mul(nodes,mp.kernel_terms)))));
    }
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
    p.coupling_kernel_terms=add(p.coupling_kernel_terms,add(mul(mul(images,sr.auxiliary_image_candidates),sr.three_center.kernel_terms),
        add(mul(panels,add(mul(pair_shells,af.kernel_terms),mul(aux_shells,mul(images,bf.kernel_terms)))),mul(8,p.coupling_scalar_terms))));
    p.coupling_shell_calls=add(p.coupling_shell_calls,sr.multipole.shell_calls);
    p.coupling_primitive_products=add(p.coupling_primitive_products,sr.multipole.primitive_products);
    // sr.scalar_terms already includes the multipole scalar work.
    p.coupling_kernel_terms=add(p.coupling_kernel_terms,sr.multipole.kernel_terms);
    p.coupling_peak_bytes=add(p.coupling_peak_bytes,sr.multipole.modeled_peak_bytes);
    if(!o.coupling_multipole_pair_tree && o.coupling_multipole_error>0. && o.coupling_multipole_hierarchy && o.coupling_multipole_compress && physical.kernel!="erf") {
        const Count h=Count(o.coupling_multipole_order+1)*(o.coupling_multipole_order+2)*(o.coupling_multipole_order+3)/6;
        const Count auxiliary_images=sr.auxiliary_image_candidates;
        const Count nodes=add(auxiliary_images,auxiliary_images)-1;
        const Count trees=mul(images,mul(aux_shells,pair_shells)),pair_size=square(b.maximum_size);
        // Additional compressed charges, downward fields, positive magnitude
        // envelopes, phase/fold buffers and Eigen temporaries. Keep the direct
        // residual tile and all old producers/work bounds admitted as well.
        p.compressed_action_bytes=add(8192,add(mul(192,mul(nodes,h)),add(mul(64,nodes),
            add(mul(256,square(h)),mul(128,add(mul(h,add(pair_size,a.maximum_size)),mul(nk,pair_size)))))));
        const Count per_tree=add(mul(add(nodes,auxiliary_images),add(square(h),mul(h,a.maximum_size))),
            add(mul(nk,pair_size),mul(h,add(pair_size,a.maximum_size))));
        p.compressed_action_scalar_terms=mul(256,mul(trees,per_tree));
        p.coupling_peak_bytes=add(p.coupling_peak_bytes,p.compressed_action_bytes);
        p.coupling_scalar_terms=add(p.coupling_scalar_terms,p.compressed_action_scalar_terms);
        p.coupling_kernel_terms=add(p.coupling_kernel_terms,mul(8,p.compressed_action_scalar_terms));
    }
    if(p.compressed_action_bytes && p.preparation_cache_bytes && o.coupling_multipole_global_actions) {
        const Count h=Count(o.coupling_multipole_order+1)*(o.coupling_multipole_order+2)*(o.coupling_multipole_order+3)/6;
        p.global_pair_slots=mul(images,pair_shells);p.global_auxiliary_slots=aux_shells;
        // Map/key owners, phases, compressed charges or compensated fields,
        // positive pair magnitudes and auxiliary charges. Source-owned real
        // moment panels are borrowed, not copied. Local producer temporaries
        // remain reserved in the compressed action plan.
        p.global_action_bytes=add(8192,add(mul(p.global_pair_slots,add(2048,add(mul(96,h),mul(64,nk)))),
            mul(aux_shells,add(1024,mul(64,h)))));
        const Count visits=mul(aux_shells,p.global_pair_slots);
        p.global_action_scalar_terms=mul(4096,add(visits,aux_shells));
        p.coupling_peak_bytes=add(p.coupling_peak_bytes,p.global_action_bytes);
        p.coupling_scalar_terms=add(p.coupling_scalar_terms,p.global_action_scalar_terms);
        p.coupling_kernel_terms=add(p.coupling_kernel_terms,mul(8,p.global_action_scalar_terms));
    }
    if(p.partition_cache_bytes) {
        const Count h=Count(o.coupling_multipole_order+1)*(o.coupling_multipole_order+2)*(o.coupling_multipole_order+3)/6;
        const Count nodes=add(sr.auxiliary_image_candidates,sr.auxiliary_image_candidates)-1;
        const Count trees=mul(images,mul(aux_shells,pair_shells));
        p.partition_scalar_terms=mul(64,mul(trees,add(64,mul(nodes,add(square(h),32)))));
        p.coupling_peak_bytes=add(p.coupling_peak_bytes,p.partition_workspace_bytes);
        p.coupling_scalar_terms=add(p.coupling_scalar_terms,p.partition_scalar_terms);
        p.coupling_kernel_terms=add(p.coupling_kernel_terms,mul(8,p.partition_scalar_terms));
    }
    if(p.pair_tree_bytes) {
        // Gamma completes the exact conjugate pair orbit in both linear maps.
        // The count-only plan admits two sequential raw passes for every q.
        p.pair_tree_action_bytes=add(p.pair_tree_action_bytes,mul(64,add(mul(nk,pairs),p.n_aux)));
        p.coupling_shell_calls=mul(2,p.coupling_shell_calls);p.coupling_primitive_products=mul(2,p.coupling_primitive_products);
        p.coupling_scalar_terms=mul(2,p.coupling_scalar_terms);p.coupling_kernel_terms=mul(2,p.coupling_kernel_terms);
        p.pair_tree_scalar_terms=mul(2,p.pair_tree_scalar_terms);
    }
    p.coupling_peak_bytes=add(p.coupling_peak_bytes,p.pair_tree_action_bytes);
    p.coupling_scalar_terms=add(p.coupling_scalar_terms,p.pair_tree_scalar_terms);
    p.coupling_kernel_terms=add(p.coupling_kernel_terms,mul(8,p.pair_tree_scalar_terms));
    p.modeled_peak_bytes=std::max(p.metric_peak_bytes,p.coupling_peak_bytes);
    if(p.preparation_cache_bytes) {
        // Three integer key comparisons per balanced-map level, at most
        // a conservative 128 levels for a Count-addressable red-black tree, plus
        // publication/receipt bookkeeping. Cold producer work stays reserved.
        const Count lookups=mul(aux_shells,mul(pair_shells,add(images,2)));
        const Count work=mul(4096,lookups);
        p.coupling_scalar_terms=add(p.coupling_scalar_terms,work);
        p.coupling_kernel_terms=add(p.coupling_kernel_terms,mul(8,work));
    }
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
    describe_coupling_reference(source_,orbital_,auxiliary_,system_,q_,k_,vectors_,options_);
    FittingDigest joint;joint.text("manta.periodic-gdf.joint.v1");joint.text(source_.metric_source_identity);
    joint.text(source_.three_center_source_identity);source_.source_identity=joint.hash.finish_hex();
    source_.modeled_peak_bytes=plan_.modeled_peak_bytes;
    if(plan_.preparation_cache_bytes) preparation_=std::make_shared<PeriodicGDFMomentCache>(coupling_extents(options_),
        plan_.preparation_pair_slots,plan_.preparation_auxiliary_slots,plan_.preparation_support_slots,plan_.preparation_cache_bytes);
    if(plan_.partition_cache_bytes) partitions_=std::make_shared<PeriodicGDFPartitionCache>(options_.coupling_multipole_partition_bytes);
    if(plan_.pair_tree_bytes) pair_state_=std::make_shared<PeriodicGDFPairState>();
}
PeriodicGDFPartitionInfo PeriodicGDFTiles::partition_info() const {
    if(partitions_) return partitions_->info(source_.three_center_source_identity);
    PeriodicGDFPartitionInfo r;r.source_identity=source_.three_center_source_identity;return r;
}
PeriodicGDFPreparationInfo PeriodicGDFTiles::preparation_info() const {
    if(preparation_) return preparation_->info(source_.three_center_source_identity);
    PeriodicGDFPreparationInfo result;result.source_identity=source_.three_center_source_identity;return result;
}


namespace {
md::Point gdf_point(const Eigen::Vector3d& p) {return {p[0],p[1],p[2]};}
Eigen::Vector3d gdf_vector(const md::Point& p) {return {p[0],p[1],p[2]};}
Eigen::Vector3d gdf_image(const Eigen::Matrix3d& lattice,const std::array<long long,3>& n) {
    return lattice*Eigen::Vector3d(double(n[0]),double(n[1]),double(n[2]));
}
std::array<long long,3> gdf_frame(const Eigen::Matrix3d& inverse,const Eigen::Vector3d& p) {
    const Eigen::Vector3d fractional=inverse*p;std::array<long long,3> result;
    for(int k=0;k<3;++k) {
        double x=fractional[k];admission::require_finite(x);
        if(std::abs(x)>1e12) throw std::invalid_argument("MANTA GDF frame exceeds supported image coordinates");
        const double nearest=std::nearbyint(x);
        if(std::abs(x-nearest)<=32*std::numeric_limits<double>::epsilon()*(1+std::abs(x))) x=nearest;
        result[k]=static_cast<long long>(std::floor(x));
    }
    return result;
}
void gdf_pair_tree_caps(const PeriodicGDFPairTreePlan& p,const ResourceBudget& b) {
    using namespace admission;
    cap(p.modeled_peak_bytes,b.memory_bytes,"MANTA GDF pair tree memory budget exceeded");
    cap(p.shell_calls,b.shell_calls,"MANTA GDF pair tree shell budget exceeded");
    cap(p.primitive_products,b.primitive_quartets,"MANTA GDF pair tree primitive budget exceeded");
    cap(p.scalar_terms,b.scalar_terms,"MANTA GDF pair tree scalar budget exceeded");
    cap(p.kernel_terms,b.kernel_terms,"MANTA GDF pair tree kernel budget exceeded");
}
}
PeriodicGDFPairTreePlan plan_periodic_gdf_pair_tree(const PeriodicGDFTiles& source,const ResourceBudget& budget) {
    using namespace admission;margin(budget);
    if(!source.preparation_ || !source.source_.real_short_range)
        throw std::invalid_argument("MANTA GDF pair tree requires prepared real short-range multipoles");
    const auto& sp=source.plan_;
    auto p=gdf_pair_geometry_plan(sp.preparation_pair_slots,source.orbital_.nshells(),source.auxiliary_.nshells(),
        sp.momenta,source.system_,source.options_,budget);
    p.modeled_peak_bytes=add(budget.other_live_bytes,add(sp.retained_bytes,p.modeled_peak_bytes));
    p.shell_calls=sp.coupling_shell_calls;p.primitive_products=sp.coupling_primitive_products;
    p.scalar_terms=add(sp.coupling_scalar_terms,p.scalar_terms);
    p.kernel_terms=add(sp.coupling_kernel_terms,add(p.kernel_terms,mul(8,sp.coupling_scalar_terms)));
    gdf_pair_tree_caps(p,budget);return p;
}
std::shared_ptr<PeriodicGDFPairTree> build_periodic_gdf_pair_tree(const PeriodicGDFTiles& source,const ResourceBudget& budget) {
    const auto plan=plan_periodic_gdf_pair_tree(source,budget);
    auto out=std::make_shared<PeriodicGDFPairTree>();out->plan_=plan;
    out->source_=source.source_.three_center_source_identity;out->lattice_=source.system_.lattice;
    out->q_=source.q_;out->k_=source.k_;out->cutoff_=source.options_.auxiliary_cutoff_bohr;
    out->pair_cutoff_=source.options_.pair_cutoff_bohr;out->image_cap_=source.options_.image_candidate_cap;
    const auto& orbit=source.orbital_.libint();const auto& aux=source.auxiliary_.libint();
    const auto inverse=out->lattice_.inverse().eval();double coordinate=1.;
    for(const auto& shell:orbit) {
        coordinate=std::max(coordinate,origin(shell).cwiseAbs().maxCoeff());
        out->bra_centers_.push_back(gdf_point(origin(shell)-gdf_image(out->lattice_,gdf_frame(inverse,origin(shell)))));
    }
    for(const auto& shell:aux) {
        out->auxiliary_centers_.push_back(shell.O);out->auxiliary_frames_.push_back(gdf_frame(inverse,origin(shell)));
        coordinate=std::max(coordinate,origin(shell).cwiseAbs().maxCoeff());
    }
    PairExtents extents;extents.receipt.source_identity=out->source_;
    const auto& o=source.options_;const ThreeCenterImageDomain domain(source.system_,o.pair_cutoff_bohr,
        o.auxiliary_cutoff_bohr,o.omega,o.image_candidate_cap,coupling_screen(o));
    auto& preparation=out->preparation_;
    for(Count i=0;i<orbit.size();++i) for(Count j=0;j<orbit.size();++j) {
        Count ordinal=0;const auto frame=gdf_frame(inverse,origin(orbit[i]));
        const Eigen::Vector3d shift=gdf_image(out->lattice_,frame);
        domain.each_unscreened_pair(orbit[i],orbit[j],[&](const Eigen::Vector3d& r,const std::array<long long,3>& index) {
            admission::cap(admission::add(out->entries_.size(),1),plan.pair_capacity,"MANTA GDF pair tree inventory exceeds admission");
            const auto ket=shifted(orbit[j],r);const auto prepared=source.preparation_->pair(orbit[i],ket,i,j,ordinal,preparation);
            PeriodicGDFPairTreeEntry e;e.bra_shell=i;e.ket_shell=j;e.pair_ordinal=ordinal++;
            e.ket_image=index;e.bra_frame=frame;e.translation=gdf_point(r);e.frame_translation=gdf_point(shift);
            e.bra_center=orbit[i].O;e.ket_center=ket.O;out->entries_.push_back(e);
            auto extent=prepared->extent;extent.center=gdf_point(gdf_vector(extent.center)-shift);
            extents.pairs.push_back(std::move(extent));
            coordinate=std::max({coordinate,origin(ket).cwiseAbs().maxCoeff(),shift.cwiseAbs().maxCoeff()});
        });
    }
    auto leaf=gdf_leaf_budget(budget);
    leaf.memory_bytes=leaf.shell_calls=leaf.primitive_quartets=leaf.scalar_terms=leaf.kernel_terms=std::numeric_limits<Count>::max();
    out->tree_=bupo_detail::OrderedPairImageBubbles::build(extents,leaf);
    const auto& nodes=out->tree_->nodes();const auto& order=out->tree_->pair_order();out->capsules_.resize(nodes.size());
    // Each endpoint band encloses corresponding endpoints of every descendant
    // segment. Their Hausdorff distance from this common segment is <= spread.
    // This is separate from Gaussian radii, which qualify far expansions.
    for(Count id=0;id<nodes.size();++id) {
        const auto& node=nodes[id];auto& band=out->capsules_[id];
        Eigen::Vector3d low_a=Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity()),low_b=low_a;
        Eigen::Vector3d high_a=-low_a,high_b=-low_b;
        for(Count k=node.begin;k<node.end;++k) {
            const auto& e=out->entries_[order[k]];const Eigen::Vector3d shift=gdf_vector(e.frame_translation);
            const Eigen::Vector3d a=gdf_vector(e.bra_center)-shift,b=gdf_vector(e.ket_center)-shift;
            low_a=low_a.cwiseMin(a);high_a=high_a.cwiseMax(a);low_b=low_b.cwiseMin(b);high_b=high_b.cwiseMax(b);
        }
        const Eigen::Vector3d a=low_a/2+high_a/2,b=low_b/2+high_b/2;
        band.begin=gdf_point(a);band.end=gdf_point(b);long double spread=0.;
        for(Count k=node.begin;k<node.end;++k) {
            const auto& e=out->entries_[order[k]];const Eigen::Vector3d shift=gdf_vector(e.frame_translation);
            const std::array<Eigen::Vector3d,2> deltas{(gdf_vector(e.bra_center)-shift-a).eval(),(gdf_vector(e.ket_center)-shift-b).eval()};
            for(const auto& delta:deltas)
                spread=std::max(spread,std::hypot(static_cast<long double>(delta[0]),static_cast<long double>(delta[1]),static_cast<long double>(delta[2])));
        }
        band.spread=periodic_error::upper(spread);
    }
    out->roundoff_guard_=1024*std::numeric_limits<double>::epsilon()*coordinate;
    return out;
}
Eigen::Vector3d PeriodicGDFPairTree::position(Count auxiliary,const std::array<long long,3>& image) const {
    if(auxiliary>=auxiliary_centers_.size()) throw std::out_of_range("MANTA GDF auxiliary shell index");
    std::array<long long,3> shifted{};
    for(int k=0;k<3;++k) {
        if(std::abs(static_cast<long double>(image[k]))>1e12L)
            throw std::invalid_argument("MANTA GDF auxiliary image exceeds supported coordinates");
        shifted[k]=image[k]-auxiliary_frames_[auxiliary][k];
    }
    const Eigen::Vector3d value=gdf_vector(auxiliary_centers_[auxiliary])+gdf_image(lattice_,shifted);
    if(!value.allFinite()) throw std::overflow_error("MANTA GDF auxiliary image position is nonfinite");
    return value;
}
md::Point PeriodicGDFPairTree::auxiliary_center(Count auxiliary,const std::array<long long,3>& image) const {
    return gdf_point(position(auxiliary,image));
}
std::array<long long,3> PeriodicGDFPairTree::source_auxiliary_image(Count entry,Count auxiliary,
    const std::array<long long,3>& image) const {
    position(auxiliary,image);
    if(entry>=entries_.size()) throw std::out_of_range("MANTA GDF pair entry index");
    std::array<long long,3> result;
    for(int k=0;k<3;++k) result[k]=image[k]+entries_[entry].bra_frame[k]-auxiliary_frames_[auxiliary][k];
    return result;
}
PeriodicGDFCapsuleMask PeriodicGDFPairTree::classify(Count id,Count auxiliary,const std::array<long long,3>& image) const {
    if(id>=tree_->nodes().size()) throw std::out_of_range("MANTA GDF pair node index");
    const auto point=position(auxiliary,image);const auto& node=tree_->nodes()[id];const auto& band=capsules_[id];
    PeriodicGDFCapsuleMask result;result.pairs=node.end-node.begin;
    if(!node.child_count) {
        // Leaves use the original source-frame arithmetic and exact shared
        // predicate, including cutoff equality. No approximate mask decision.
        const Count entry=tree_->pair_order()[node.begin];const auto& e=entries_[entry];
        const auto index=source_auxiliary_image(entry,auxiliary,image);
        const Eigen::Vector3d original=gdf_vector(auxiliary_centers_[auxiliary])+gdf_image(lattice_,index);
        result.membership=inside_auxiliary_capsule(original,gdf_vector(e.bra_center),gdf_vector(e.ket_center),cutoff_)?"inside":"outside";
        return result;
    }
    Eigen::Matrix<long double,3,1> a,b,x;
    for(int k=0;k<3;++k) {a[k]=band.begin[k];b[k]=band.end[k];x[k]=point[k];}
    const auto segment=(b-a).eval(),delta=(x-a).eval();const long double length2=segment.squaredNorm();
    const long double f=length2>0?std::clamp(delta.dot(segment)/length2,0.L,1.L):0.L;
    const long double distance=(delta-f*segment).norm();
    // Roundoff uncertainty refines to exact leaf membership. Image/frame
    // reindexing changes representation only, never the finite support.
    const long double guard=roundoff_guard_+1024*std::numeric_limits<double>::epsilon()*
        (1+point.cwiseAbs().maxCoeff()+a.cwiseAbs().maxCoeff()+b.cwiseAbs().maxCoeff()+cutoff_+band.spread);
    result.membership=distance+band.spread+guard<cutoff_?"inside":
        distance-band.spread-guard>cutoff_?"outside":"mixed";
    return result;
}
std::vector<std::array<long long,3>> PeriodicGDFPairTree::auxiliary_images(Count auxiliary) const {
    const Eigen::Vector3d center=position(auxiliary,{0,0,0});
    PeriodicSystem system;system.dim=3;system.lattice=lattice_;
    const ImageBox box(system,pair_cutoff_+cutoff_,image_cap_);
    std::set<std::array<long long,3>> candidates;
    // Any selected capsule point is within (pair cutoff + auxiliary cutoff)
    // of its bra center. Enumerate those padded integer boxes, deduplicate
    // integer labels (never rounded Cartesian coordinates), then apply the
    // exact shared masks. No auxiliary-by-pair membership matrix is retained.
    for(const auto& bra:bra_centers_) box.each_index(gdf_vector(bra)-center,
        [&](const Eigen::Vector3d&,const std::array<long long,3>& image) {
            if(candidates.find(image)!=candidates.end()) return;
            admission::cap(admission::add(candidates.size(),1),plan_.auxiliary_image_capacity,
                "MANTA GDF auxiliary union inventory exceeds admission");
            candidates.insert(image);
        });
    std::vector<std::array<long long,3>> result;
    for(const auto& image:candidates) if(partition(auxiliary,image).included_pairs) result.push_back(image);
    return result;
}
PeriodicGDFCapsulePartition PeriodicGDFPairTree::partition(Count auxiliary,const std::array<long long,3>& image) const {
    position(auxiliary,image);PeriodicGDFCapsulePartition result;result.source_identity=source_;
    const auto& nodes=tree_->nodes();std::vector<Count> stack{0};
    while(!stack.empty()) {
        const Count id=stack.back();stack.pop_back();++result.visited_nodes;
        const auto mask=classify(id,auxiliary,image);
        if(mask.membership=="inside") {result.included_nodes.push_back(id);result.included_pairs+=mask.pairs;}
        else if(mask.membership=="outside") result.excluded_pairs+=mask.pairs;
        else {
            ++result.refined_nodes;const auto& node=nodes[id];
            for(int k=node.child_count;k>0;--k) stack.push_back(node.children[k-1]);
        }
    }
    if(result.included_pairs+result.excluded_pairs!=entries_.size())
        throw std::logic_error("MANTA GDF pair mask traversal is incomplete");
    return result;
}
std::vector<Complex> PeriodicGDFPairTree::pair_phase(Count entry) const {
    if(entry>=entries_.size()) throw std::out_of_range("MANTA GDF pair entry index");
    const auto& e=entries_[entry];std::vector<Complex> result(k_.rows());
    for(Eigen::Index k=0;k<k_.rows();++k) {
        const double angle=k_.row(k).dot(gdf_vector(e.translation))-q_.dot(gdf_vector(e.frame_translation));
        admission::require_finite(angle);result[k]=std::exp(Complex(0.,angle));
    }
    return result;
}
Complex PeriodicGDFPairTree::auxiliary_phase(Count auxiliary,const std::array<long long,3>& image) const {
    position(auxiliary,image);std::array<long long,3> index;
    for(int k=0;k<3;++k) index[k]=image[k]-auxiliary_frames_[auxiliary][k];
    const double angle=-q_.dot(gdf_image(lattice_,index));admission::require_finite(angle);
    return std::exp(Complex(0.,angle));
}

namespace {
void add_pair_sr_receipt(PeriodicGDFShortRangeReceipt& a,const PeriodicGDFShortRangeReceipt& b) {
    using R=PeriodicGDFShortRangeReceipt;
    for(auto member:{&R::multipole_trials,&R::multipole_images,&R::multipole_moment_panels,&R::multipole_primitive_products,
        &R::multipole_envelope_evaluations,&R::multipole_candidate_images,&R::multipole_tree_nodes,&R::multipole_far_blocks,
        &R::multipole_translations,&R::multipole_reused_extents,&R::multipole_reused_moment_panels,&R::multipole_partition_builds,
        &R::multipole_partition_reuses,&R::multipole_partition_bypasses,&R::metric_shell_calls,&R::three_center_shell_calls,
        &R::pair_shell_images,&R::primitive_products,&R::phase_products,&R::phase_accumulations})
        a.*member=admission::add(a.*member,b.*member);
    a.evaluated_multipole_error=std::max(a.evaluated_multipole_error,b.evaluated_multipole_error);
}
void add_pair_action_receipt(PeriodicGDFCouplingReceipt& r,const PeriodicGDFShortRangeReceipt& s) {
#define PAIR_COUNT(name) r.name=admission::add(r.name,s.name)
    PAIR_COUNT(multipole_trials);PAIR_COUNT(multipole_images);PAIR_COUNT(multipole_moment_panels);
    PAIR_COUNT(multipole_envelope_evaluations);PAIR_COUNT(multipole_tree_nodes);PAIR_COUNT(multipole_far_blocks);
    PAIR_COUNT(multipole_translations);PAIR_COUNT(multipole_reused_extents);PAIR_COUNT(multipole_reused_moment_panels);
    PAIR_COUNT(multipole_partition_builds);PAIR_COUNT(multipole_partition_reuses);PAIR_COUNT(multipole_partition_bypasses);
#undef PAIR_COUNT
    r.shell_calls=admission::add(r.shell_calls,admission::add(s.three_center_shell_calls,s.multipole_moment_panels));
    r.primitive_products=admission::add(r.primitive_products,admission::add(s.primitive_products,s.multipole_primitive_products));
}
}
void PeriodicGDFTiles::prepare_pair_tree(PeriodicGDFShortRangeReceipt& receipt) const {
    if(!pair_state_) throw std::logic_error("MANTA global pair route was not admitted");
    std::call_once(pair_state_->once,[&] {
        const auto leaf=gdf_leaf_budget(ResourceBudget{});
        auto geometry=build_periodic_gdf_pair_tree(*this,leaf);
        if(source_.zero_transfer) {
            const Eigen::Vector3d fractional=system_.lattice.transpose()*q_/(2*std::acos(-1.));
            if((fractional-fractional.array().round().matrix()).norm()>
                64*std::numeric_limits<double>::epsilon()*(1+fractional.norm()))
                throw std::invalid_argument("MANTA global pair completion requires a reciprocal Gamma representative within roundoff");
            // The literal finite selector must itself possess this symmetry.
            // Floating-point cutoff ties can break capsule conjugacy; refuse
            // those sources instead of projecting away a finite-domain defect.
            const auto& entries=geometry->entries();const auto& nodes=geometry->tree()->nodes();
            const auto& permutation=geometry->tree()->pair_order();
            using Key=std::tuple<Count,Count,std::array<long long,3>>;
            std::map<Key,Count> lookup;std::vector<Count> mate(entries.size()),leaves(entries.size());
            for(Count i=0;i<entries.size();++i) lookup.emplace(Key{entries[i].bra_shell,entries[i].ket_shell,entries[i].ket_image},i);
            for(Count i=0;i<nodes.size();++i) if(!nodes[i].child_count) leaves[permutation[nodes[i].begin]]=i;
            for(Count i=0;i<entries.size();++i) {
                const auto& e=entries[i];auto reverse=e.ket_image;for(auto& k:reverse) k=-k;
                const auto found=lookup.find(Key{e.ket_shell,e.bra_shell,reverse});
                if(found==lookup.end()) throw std::invalid_argument("MANTA global pair Gamma requires conjugate-closed finite pair support");
                mate[i]=found->second;
            }
            for(Count ip=0;ip<auxiliary_.nshells();++ip) for(const auto& image:geometry->auxiliary_images(ip)) {
                const auto partition=geometry->partition(ip,image);
                for(const Count id:partition.included_nodes) for(Count k=nodes[id].begin;k<nodes[id].end;++k) {
                    const Count i=permutation[k];const auto& e=entries[i];const auto& other=entries[mate[i]];
                    auto counterpart=image;
                    for(int axis=0;axis<3;++axis) counterpart[axis]+=e.bra_frame[axis]-other.bra_frame[axis]-e.ket_image[axis];
                    if(geometry->classify(leaves[mate[i]],ip,counterpart).membership!="inside")
                        throw std::invalid_argument("MANTA global pair Gamma requires conjugate-closed finite capsule support");
                }
            }
        }
        auto bounds=std::make_unique<BubbleErrorModel>(geometry->tree(),options_.coupling_multipole_order,leaf);
        const ThreeCenterImageDomain domain(system_,options_.pair_cutoff_bohr,options_.auxiliary_cutoff_bohr,
            options_.omega,options_.image_candidate_cap,0.);
        const auto& a=auxiliary_.libint();const auto& b=orbital_.libint();Count maximum=0;
        for(Count ip=0;ip<a.size();++ip) for(Count im=0;im<b.size();++im) for(Count in=0;in<b.size();++in) {
            const auto count=preparation_->support({ip,im,in},[&] {
                Count n=0;domain.each(a[ip],b[im],b[in],[](const Eigen::Vector3d&) {},
                    [&](const Eigen::Vector3d&,const Eigen::Vector3d&) {n=admission::add(n,1);},[](const Eigen::Vector3d&) {});return n;
            });
            maximum=std::max(maximum,count);
            receipt.multipole_candidate_images=admission::add(receipt.multipole_candidate_images,count);
        }
        add_pair_sr_receipt(receipt,geometry->preparation_receipt());
        pair_state_->geometry=std::move(geometry);pair_state_->bounds=std::move(bounds);pair_state_->maximum_support=maximum;
    });
}

// Global ordered pair-bubble RI-BUPO: Neese et al. (2025),
// doi:10.1021/acs.jpca.4c07415, Eqs.29-36. The shared GDF producer owns
// discrete support and Bloch frames; only wholly included nodes can undergo
// the separate native Gaussian erfc error test. All input/field state is local.
struct PeriodicGDFPairContractions {
    const PeriodicGDFTiles& source;
    const PeriodicGDFPairTree& geometry;
    const BubbleTree& tree;
    const libint2::BasisSet& orbital;
    const libint2::BasisSet& auxiliary;
    PeriodicGDFShortRangeReceipt& receipt;
    ResourceBudget leaf=gdf_leaf_budget(ResourceBudget{});
    const int order;
    const Count h,n,nk;
    std::vector<Count> interaction_rows;
    Count packed_elements=0;
    const std::vector<Complex>* input=nullptr;
    std::vector<Complex>& output;
    std::vector<Complex>& correction;
    PeriodicGDFCouplingReceipt* action=nullptr;
    std::vector<double>* component_bounds=nullptr;
    std::vector<double>* reference_magnitudes=nullptr;
    std::vector<double>* reference_actions=nullptr;
    const bool forward;
    std::array<Count,3> filter{};
    std::unique_ptr<PeriodicGDFGlobalAction> global;
    std::vector<std::shared_ptr<const PeriodicGDFMomentCache::Pair>> prepared_pairs;
    std::vector<const md::Matrix*> panels;
    std::vector<std::vector<Complex>> phases;
    std::vector<Eigen::VectorXcd> charges,fields,field_errors,near_pairs,near_errors;
    std::vector<Eigen::VectorXd> magnitudes;
    std::vector<bool> ready;
    // Each node envelope is uniform over its descendants. Accumulate it
    // once per accepted interaction, propagate to leaves, then sum only
    // original R images belonging to the SAME orbital shell pair. Keep one
    // auxiliary shell at a time; never form an auxiliary-by-pair tensor.
    struct Bounds {long double error=0.,magnitude=0.;};
    std::vector<Bounds> node_bounds,pair_bounds;
    std::vector<long double> input_bounds,output_bounds,output_magnitudes;
    long double maximum_error=0.,action_error=0.,action_error_norm=0.;
    double maximum_step_error=0.;
    static void positive_add(long double value,long double& sum) {
        periodic_error::positive_add(value,sum);
    }
    void add_error_norm(Count elements,long double error) {
        if(error!=0.) {
            const double contribution=periodic_error::upper(std::sqrt(static_cast<long double>(elements))*error);
            action_error_norm=std::nextafter(std::hypot(action_error_norm,static_cast<long double>(contribution)),
                std::numeric_limits<long double>::infinity());
        }
    }
    Count block(Count index) const {
        const auto& e=geometry.entries()[index];return e.bra_shell*orbital.size()+e.ket_shell;
    }
    PeriodicGDFPairContractions(const PeriodicGDFTiles& s,PeriodicGDFShortRangeReceipt& r,
        std::vector<Complex>& y,std::vector<Complex>& e,const std::vector<Complex>* x,
        PeriodicGDFCouplingReceipt* a,bool f,std::array<Count,3> selected={},
        std::vector<double>* components=nullptr,std::vector<double>* magnitudes_out=nullptr,
        std::vector<double>* actions_out=nullptr)
        :source(s),geometry(*s.pair_state_->geometry),tree(*geometry.tree()),orbital(s.orbital_.libint()),
         auxiliary(s.auxiliary_.libint()),receipt(r),order(s.options_.coupling_multipole_order),
         h(Count(order+1)*(order+2)*(order+3)/6),n(s.plan_.n_orb),nk(s.plan_.momenta),
         input(x),output(y),correction(e),action(a),component_bounds(components),
         reference_magnitudes(magnitudes_out),reference_actions(actions_out),forward(f),filter(selected),
         prepared_pairs(geometry.entries().size()),panels(prepared_pairs.size(),nullptr),phases(prepared_pairs.size()),
         charges(a?tree.nodes().size():0),fields(a?tree.nodes().size():0),field_errors(fields.size()),
         near_pairs(a?prepared_pairs.size():0),near_errors(near_pairs.size()),magnitudes(charges.size()),ready(charges.size()),
         node_bounds(tree.nodes().size()),pair_bounds(admission::square(orbital.size())),
         input_bounds(a?(f?pair_bounds.size():auxiliary.size()):0),output_bounds(a && !f?pair_bounds.size():0),
         output_magnitudes(output_bounds.size()) {
        // md::hermites orders rows by total degree. The existing producer
        // explicitly zeros entries with degree(row)+degree(column)>order.
        // Keep every allowed value verbatim, including zeros and signed zeros.
        for(const auto& index:md::hermites(order)) {
            const Count degree=order-index[0]-index[1]-index[2];
            const Count columns=(degree+1)*(degree+2)*(degree+3)/6;
            interaction_rows.push_back(columns);packed_elements+=columns;
        }
        if(action) {
            global=std::make_unique<PeriodicGDFGlobalAction>(*input,output,correction,*action,s.k_,f,s.plan_);
            auto norm=[](Complex z) {return periodic_error::upper(std::hypot(static_cast<long double>(z.real()),static_cast<long double>(z.imag())));};
            if(forward) {
                const auto bf=orbital.shell2bf();
                for(Count im=0;im<orbital.size();++im)for(Count in=0;in<orbital.size();++in)
                    for(Count k=0;k<nk;++k)for(Count i=0;i<orbital[im].size();++i)for(Count j=0;j<orbital[in].size();++j)
                        positive_add(norm((*input)[k*n*n+(bf[im]+i)*n+bf[in]+j]),input_bounds[im*orbital.size()+in]);
            } else {
                const auto af=auxiliary.shell2bf();
                for(Count ip=0;ip<auxiliary.size();++ip)for(Count i=0;i<auxiliary[ip].size();++i)
                    positive_add(norm((*input)[af[ip]+i]),input_bounds[ip]);
            }
        }
        for(Count i=0;i<phases.size();++i) phases[i]=geometry.pair_phase(i);
        receipt.multipole_tree_nodes=admission::add(receipt.multipole_tree_nodes,tree.nodes().size());
        receipt.phase_products=admission::add(receipt.phase_products,admission::mul(phases.size(),nk));
    }
    void products(Count value) {if(action) global->products(value);}
    static md::Point difference(const md::Point& a,const md::Point& b) {
        md::Point d;for(int k=0;k<3;++k)d[k]=a[k]-b[k];return d;
    }
    bool selected(Count index) const {
        const auto& e=geometry.entries()[index];return action || (e.bra_shell==filter[1] && e.ket_shell==filter[2]);
    }
    const md::Matrix& pair_panel(Count index) {
        if(!panels[index]) {
            const auto& e=geometry.entries()[index];const auto& bra=orbital[e.bra_shell];auto ket=orbital[e.ket_shell];ket.O=e.ket_center;
            auto prepared=source.preparation_->pair(bra,ket,e.bra_shell,e.ket_shell,e.pair_ordinal,receipt);
            panels[index]=&source.preparation_->moments(prepared,[&] {return md::shell_moments(bra,ket,prepared->extent.center,order);},bra.nprim()*ket.nprim(),receipt);
            prepared_pairs[index]=std::move(prepared);
        }
        return *panels[index];
    }
    PeriodicGDFGlobalAction::Pair& pair(Count index) {
        const auto& e=geometry.entries()[index];const auto& panel=pair_panel(index);const auto bf=orbital.shell2bf();
        return global->pair({e.bra_shell,e.ket_shell,e.pair_ordinal},prepared_pairs[index],panel,gdf_vector(e.translation),
            orbital[e.bra_shell].size(),orbital[e.ket_shell].size(),bf[e.bra_shell],bf[e.ket_shell],&phases[index]);
    }
    void add_vector(const Eigen::VectorXcd& value,Eigen::VectorXcd& sum,Eigen::VectorXcd& error) {
        if(!sum.size()) {sum=Eigen::VectorXcd::Zero(value.size());error=sum;}
        for(Eigen::Index i=0;i<value.size();++i) fitting_add(value[i],sum[i],error[i]);
    }
    void upward(Count id) {
        if(ready[id]) return;
        const auto& node=tree.nodes()[id];magnitudes[id]=Eigen::VectorXd::Zero(h);
        if(forward) charges[id]=Eigen::VectorXcd::Zero(h);
        Eigen::VectorXcd error=Eigen::VectorXcd::Zero(forward?h:0);
        if(node.child_count) for(int k=0;k<node.child_count;++k) {
            const Count child=node.children[k];upward(child);
            const auto w=multipole_translation(difference(tree.nodes()[child].center,node.center),order,leaf);
            if(forward) add_vector(w.values*charges[child],charges[id],error);
            magnitudes[id]=magnitudes[id].cwiseMax(w.values.cwiseAbs()*magnitudes[child]);
            ++receipt.multipole_translations;products(admission::mul(2,admission::square(h)));
        } else {
            const Count index=tree.pair_order()[node.begin];const auto& panel=pair_panel(index);
            const auto delta=difference(tree.pairs()[index].center,node.center);
            magnitudes[id]=panel.cwiseAbs().colwise().maxCoeff().transpose();
            if(delta==md::Point{}) {if(forward) charges[id]=pair(index).charge;}
            else {
                const auto w=multipole_translation(delta,order,leaf);
                magnitudes[id]=w.values.cwiseAbs()*magnitudes[id];
                if(forward) charges[id]=w.values*pair(index).charge;
                ++receipt.multipole_translations;products(admission::mul(2,admission::square(h)));
            }
        }
        for(Eigen::Index k=0;k<magnitudes[id].size();++k) magnitudes[id][k]=periodic_error::upper(magnitudes[id][k]);
        if(forward && !charges[id].allFinite()) throw std::overflow_error("MANTA global pair charge is nonfinite");
        ready[id]=true;
    }
    void tile_add(Count index,Complex phase,const md::Matrix& values) {
        const Count elements=values.size();
        for(Count k=0;k<nk;++k)for(Count i=0;i<elements;++i) {
            const Count out=k*elements+i;
            fitting_add(phases[index][k]*phase*values.data()[i],output[out],correction[out]);++receipt.phase_accumulations;
        }
    }
    void near(Count ip,const std::array<long long,3>& image,Count index,Complex phase) {
        if(!selected(index)) return;
        const auto& e=geometry.entries()[index];auto ket=orbital[e.ket_shell];ket.O=e.ket_center;
        const auto as=shifted(auxiliary[ip],gdf_image(source.system_.lattice,geometry.source_auxiliary_image(index,ip,image)));
        const auto value=md::shell_coulomb({&as,nullptr,&orbital[e.bra_shell],&ket},md::RadialKernel::Erfc,source.options_.omega);
        ++receipt.three_center_shell_calls;receipt.primitive_products+=as.nprim()*orbital[e.bra_shell].nprim()*ket.nprim();
        if(!action) {tile_add(index,phase,value);return;}
        positive_add(periodic_error::upper(value.cwiseAbs().maxCoeff()),pair_bounds[block(index)].magnitude);
        const auto af=auxiliary.shell2bf(),bf=orbital.shell2bf();const Count nm=orbital[e.bra_shell].size(),nn=ket.size();
        if(forward) {
            if(!near_pairs[index].size()) {
                near_pairs[index]=Eigen::VectorXcd::Zero(nm*nn);Eigen::VectorXcd error=near_pairs[index];
                for(Count k=0;k<nk;++k)for(Count i=0;i<nm;++i)for(Count j=0;j<nn;++j)
                    fitting_add(std::conj(phases[index][k])*(*input)[k*n*n+(bf[e.bra_shell]+i)*n+bf[e.ket_shell]+j],near_pairs[index][i*nn+j],error[i*nn+j]);
                products(admission::mul(nk,nm*nn));
            }
            const Eigen::VectorXcd result=std::conj(phase)*(value*near_pairs[index]);
            for(Count i=0;i<as.size();++i) fitting_add(result[i],output[af[ip]+i],correction[af[ip]+i]);
        } else add_vector(phase*(value.transpose()*Eigen::Map<const Eigen::VectorXcd>(input->data()+af[ip],as.size())),near_pairs[index],near_errors[index]);
        products(admission::mul(as.size(),admission::add(nm*nn,1)));
    }
    void finish_bounds(Count ip) {
        // Shared BubbleTree stores parents before children. Descendant leaves
        // represent ordered pair images, not distinct AO output components.
        const auto& nodes=tree.nodes();
        for(Count id=0;id<nodes.size();++id) {
            const auto& node=nodes[id];const auto bound=node_bounds[id];
            if(node.child_count) for(int k=0;k<node.child_count;++k) {
                auto& child=node_bounds[node.children[k]];
                positive_add(bound.error,child.error);positive_add(bound.magnitude,child.magnitude);
            } else {
                auto& pair=pair_bounds[block(tree.pair_order()[node.begin])];
                positive_add(bound.error,pair.error);positive_add(bound.magnitude,pair.magnitude);
            }
        }
        long double error=0.,magnitude=0.,absolute_action=0.;
        for(Count b=0;b<pair_bounds.size();++b) {
            const auto bound=pair_bounds[b];maximum_error=std::max(maximum_error,bound.error);
            if(!action) continue;
            action->maximum_coupling=std::max(action->maximum_coupling,periodic_error::upper(bound.magnitude));
            magnitude=std::max(magnitude,bound.magnitude+bound.error);
            if(forward) {
                positive_add(periodic_error::upper(bound.error*input_bounds[b]),error);
                positive_add(periodic_error::upper((bound.magnitude+bound.error)*input_bounds[b]),absolute_action);
            } else {
                positive_add(periodic_error::upper(bound.error*input_bounds[ip]),output_bounds[b]);
                positive_add(periodic_error::upper((bound.magnitude+bound.error)*input_bounds[ip]),output_magnitudes[b]);
            }
        }
        if(action) {
            const auto af=auxiliary.shell2bf();
            for(Count i=0;i<auxiliary[ip].size();++i) {
                (*reference_magnitudes)[af[ip]+i]=periodic_error::upper(magnitude);
                if(forward) {
                    (*component_bounds)[af[ip]+i]=periodic_error::upper(error);
                    (*reference_actions)[af[ip]+i]=periodic_error::upper(absolute_action);
                }
            }
            if(forward) {
                action_error=std::max(action_error,error);
                add_error_norm(auxiliary[ip].size(),error);
            }
        }
    }
    PeriodicGDFPartitionCache::Step packed_step(Count id,const md::Matrix& matrix,double error) const {
        PeriodicGDFPartitionCache::Step step{id,{},error};step.packed.resize(packed_elements);
        Count offset=0;
        for(Count i=0;i<h;++i) {
            for(Count j=0;j<interaction_rows[i];++j) step.packed[offset++]=matrix(i,j);
            for(Count j=interaction_rows[i];j<h;++j)
                if(matrix(i,j)!=0.) throw std::logic_error("MANTA total-degree interaction has an unexpected coefficient");
        }
        return step;
    }
    void run() {
        // One expansion buffer is covered by the existing single-interaction
        // workspace. Dense multiplication order and error evaluation stay fixed.
        md::Matrix interaction(h,h);
        const auto& nodes=tree.nodes();const auto& permutation=tree.pair_order();
        const Count first=action?0:filter[0],last=action?auxiliary.size():first+1;
        const long double tolerance=source.pair_state_->maximum_support?
            static_cast<long double>(source.options_.coupling_multipole_error)/source.pair_state_->maximum_support*
            (1-4096*std::numeric_limits<double>::epsilon()):0.;
        for(Count ip=first;ip<last;++ip) {
            std::fill(node_bounds.begin(),node_bounds.end(),Bounds{});
            std::fill(pair_bounds.begin(),pair_bounds.end(),Bounds{});
            const auto prepared=source.preparation_->aux(auxiliary[ip],ip,receipt);
            const md::Matrix* auxiliary_panel=nullptr;
            auto panel=[&]() -> const md::Matrix& {
                if(!auxiliary_panel) auxiliary_panel=&source.preparation_->moments(prepared,[&] {
                    return md::shell_auxiliary_moments(auxiliary[ip],auxiliary[ip].O,order);
                },auxiliary[ip].nprim(),receipt);
                return *auxiliary_panel;
            };
            const auto images=geometry.auxiliary_images(ip);Count ordinal=0;
            for(const auto& image:images) {
                const auto center=geometry.auxiliary_center(ip,image);auto auxiliary_extent=prepared->extent;auxiliary_extent.center=center;
                auto build=[&] {
                    auto entry=std::make_shared<PeriodicGDFPartitionCache::Entry>();entry->tree=geometry.tree();entry->steps.reserve(nodes.size());
                    auto visit=[&](auto&& self,Count id)->void {
                        const auto mask=geometry.classify(id,ip,image);if(mask.membership=="outside") return;
                        const auto& node=nodes[id];
                        if(mask.membership=="inside" && !node.always_near && !auxiliary_extent.always_near) {
                            const long double distance=std::hypot(static_cast<long double>(center[0]-node.center[0]),
                                static_cast<long double>(center[1]-node.center[1]),static_cast<long double>(center[2]-node.center[2]));
                            if(distance>(static_cast<long double>(node.radius)+auxiliary_extent.radius)*(1+128*std::numeric_limits<double>::epsilon())) {
                                PairExtent target;target.center=node.center;target.radius=node.radius;
                                const auto& envelope=source.pair_state_->bounds->envelopes()[id];
                                target.full_moment_bounds=envelope.full_moments;target.tail_moment_bounds=envelope.tail_moments;target.potential_bound=envelope.potential;
                                ++receipt.multipole_trials;
                                try {
                                    auto interaction=erfc_gaussian_interaction(source.options_.omega,auxiliary_extent,target,order,leaf);
                                    if(interaction.error.total<=tolerance) {
                                        entry->steps.push_back(packed_step(id,interaction.matrix.values,interaction.error.total));return;
                                    }
                                } catch(const std::overflow_error&) {} // Refine unrepresentable envelopes.
                            }
                        }
                        if(node.child_count) for(int c=0;c<node.child_count;++c) self(self,node.children[c]);
                        else entry->steps.push_back({id,{},0.});
                    };
                    visit(visit,0);++receipt.multipole_partition_builds;
                    // Every entry borrows the SAME source-owned global tree; its
                    // storage is already admitted in pair_tree_bytes, not per image.
                    // Move into an exactly sized antichain allocation. The old
                    // nodes.size() reservation belongs only to construction;
                    // retaining it for every auxiliary image wastes the cache.
                    std::vector<PeriodicGDFPartitionCache::Step>(std::make_move_iterator(entry->steps.begin()),
                        std::make_move_iterator(entry->steps.end())).swap(entry->steps);
                    entry->bytes=admission::add(4096,admission::mul(entry->steps.capacity(),sizeof(PeriodicGDFPartitionCache::Step)));
                    for(const auto& step:entry->steps) entry->bytes=admission::add(entry->bytes,admission::mul(8,step.packed.capacity()));
                    return entry;
                };
                std::shared_ptr<const PeriodicGDFPartitionCache::Entry> partition;
                if(source.partitions_) {
                    const auto upper=admission::add(4096,admission::mul(nodes.size(),admission::add(256,admission::mul(8,admission::square(h)))));
                    partition=source.partitions_->get({ip,ordinal,0,0},geometry.entries().size(),center,upper,build,receipt,true);
                }
                if(!partition) partition=build();++ordinal;
                const Complex phase=geometry.auxiliary_phase(ip,image);++receipt.phase_products;
                for(const auto& step:partition->steps) {
                    const auto& node=nodes[step.node];
                    if(step.packed.empty()) {
                        for(Count k=node.begin;k<node.end;++k) near(ip,image,permutation[k],phase);
                        continue;
                    }
                    if(step.packed.size()!=packed_elements) throw std::logic_error("MANTA packed partition has an inconsistent order");
                    interaction.setZero();Count offset=0;
                    for(Count i=0;i<h;++i) for(Count j=0;j<interaction_rows[i];++j)
                        interaction(i,j)=step.packed[offset++];
                    const Count represented=node.end-node.begin;
                    receipt.multipole_images=admission::add(receipt.multipole_images,represented);++receipt.multipole_far_blocks;
                    maximum_step_error=std::max(maximum_step_error,step.error);
                    positive_add(step.error,node_bounds[step.node].error);
                    const auto& ap=panel();
                    if(action) {
                        upward(step.node);const auto af=auxiliary.shell2bf();auto& aux=global->aux(af[ip],ap,prepared);
                        const Eigen::VectorXd am=ap.cwiseAbs().colwise().maxCoeff().transpose();
                        positive_add(periodic_error::upper(am.dot(interaction.cwiseAbs()*magnitudes[step.node])),node_bounds[step.node].magnitude);
                        if(forward) global->add(aux,std::conj(phase)*(interaction*charges[step.node]));
                        else add_vector(phase*(interaction.transpose()*aux.charge),fields[step.node],field_errors[step.node]);
                        ++action->multipole_compressed_nodes;if(represented>1) ++action->multipole_pair_groups;
                        products(admission::add(admission::square(h),h));
                    } else for(Count k=node.begin;k<node.end;++k) {
                        const Count index=permutation[k];if(!selected(index)) continue;
                        const auto& pm=pair_panel(index);const auto delta=difference(tree.pairs()[index].center,node.center);
                        if(delta==md::Point{}) tile_add(index,phase,(ap*interaction*pm.transpose()).eval());
                        else {
                            const auto w=multipole_translation(delta,order,leaf);++receipt.multipole_translations;
                            tile_add(index,phase,(ap*interaction*w.values*pm.transpose()).eval());
                        }
                    }
                }
            }
            finish_bounds(ip);
        }
        if(action && !forward) {
            for(Count b=0;b<output_bounds.size();++b) {
                const auto error=output_bounds[b];action_error=std::max(action_error,error);
                const Count elements=admission::mul(nk,admission::mul(orbital[b/orbital.size()].size(),orbital[b%orbital.size()].size()));
                add_error_norm(elements,error);
                const auto bf=orbital.shell2bf();const Count im=b/orbital.size(),in=b%orbital.size();
                for(Count k=0;k<nk;++k)for(Count i=0;i<orbital[im].size();++i)for(Count j=0;j<orbital[in].size();++j) {
                    const Count out=k*n*n+(bf[im]+i)*n+bf[in]+j;
                    (*component_bounds)[out]=periodic_error::upper(error);
                    (*reference_actions)[out]=periodic_error::upper(output_magnitudes[b]);
                }
            }
            for(Count id=0;id<nodes.size();++id) if(fields[id].size()) {
                const auto& node=nodes[id];
                if(node.child_count) for(int k=0;k<node.child_count;++k) {
                    const Count child=node.children[k];const auto w=multipole_translation(difference(nodes[child].center,node.center),order,leaf);
                    add_vector(w.values.transpose()*fields[id],fields[child],field_errors[child]);
                    ++action->multipole_action_translations;products(admission::square(h));
                } else {
                    const Count index=permutation[node.begin];const auto delta=difference(tree.pairs()[index].center,node.center);
                    if(delta==md::Point{}) global->add(pair(index),fields[id]);
                    else {
                        const auto w=multipole_translation(delta,order,leaf);global->add(pair(index),w.values.transpose()*fields[id]);
                        ++action->multipole_action_translations;products(admission::square(h));
                    }
                }
            }
            const auto bf=orbital.shell2bf();
            for(Count index=0;index<near_pairs.size();++index) if(near_pairs[index].size()) {
                const auto& e=geometry.entries()[index];const Count nm=orbital[e.bra_shell].size(),nn=orbital[e.ket_shell].size();
                for(Count k=0;k<nk;++k)for(Count i=0;i<nm;++i)for(Count j=0;j<nn;++j) {
                    const Count out=k*n*n+(bf[e.bra_shell]+i)*n+bf[e.ket_shell]+j;
                    fitting_add(phases[index][k]*near_pairs[index][i*nn+j],output[out],correction[out]);
                }
                products(admission::mul(nk,nm*nn));
            }
        }
        if(action) {
            global->finish();action->multipole_compressed_pairs=global->pairs.size();
            action->approximation_bound=periodic_error::upper(action_error);
            action->approximation_l2_bound=periodic_error::upper(action_error_norm);
        }
        // Both bounds hold independently. The old support-count ceiling also
        // preserves the admitted epsilon under very large positive ledgers.
        receipt.evaluated_multipole_error=std::min(periodic_error::upper(maximum_error),
            periodic_error::upper(static_cast<long double>(maximum_step_error)*source.pair_state_->maximum_support));
    }
};
PeriodicGDFTile PeriodicGDFTiles::pair_tree_tile_impl(Count ip,Count im,Count in) const {
    if(ip>=auxiliary_.nshells() || im>=orbital_.nshells() || in>=orbital_.nshells()) throw std::out_of_range("MANTA GDF coupling shell index");
    const auto& a=auxiliary_.libint();const auto& b=orbital_.libint();const auto af=a.shell2bf(),bf=b.shell2bf();
    PeriodicGDFTile result;result.shape={plan_.momenta,a[ip].size(),b[im].size(),b[in].size()};result.offsets={0,af[ip],bf[im],bf[in]};
    result.receipt=source_;auto& sr=result.receipt.short_range;prepare_pair_tree(sr);
    result.values.resize(plan_.momenta*a[ip].size()*b[im].size()*b[in].size());std::vector<Complex> correction(result.values.size());
    PeriodicGDFPairContractions(*this,sr,result.values,correction,nullptr,nullptr,false,{ip,im,in}).run();
    finish_gdf_tile(result);return result;
}
PeriodicGDFTile PeriodicGDFTiles::pair_tree_tile(Count ip,Count im,Count in) const {
    auto result=pair_tree_tile_impl(ip,im,in);if(!source_.zero_transfer) return result;
    const auto mate=pair_tree_tile_impl(ip,in,im);const Count np=result.shape[1],nm=result.shape[2],nn=result.shape[3];
    // The exact ordered Gamma sphere/capsule source is invariant under
    // (mu,nu,R,T) -> (nu,mu,-R,T-R). Complete that conjugate pair orbit in
    // the approximate COUPLING, not only in the resulting Fock matrix.
    // Convex averaging preserves its uniform error bound relative to this
    // same Hermitian exact finite source (arithmetic roundoff is separate).
    for(Count k=0;k<plan_.momenta;++k)for(Count p=0;p<np;++p)for(Count i=0;i<nm;++i)for(Count j=0;j<nn;++j) {
        const Count out=((k*np+p)*nm+i)*nn+j,other=((k*np+p)*nn+j)*nm+i;
        result.values[out]=.5*(result.values[out]+std::conj(mate.values[other]));
    }
    add_pair_sr_receipt(result.receipt.short_range,mate.receipt.short_range);
    result.receipt.shell_calls=admission::add(result.receipt.shell_calls,mate.receipt.shell_calls);
    result.receipt.primitive_products=admission::add(result.receipt.primitive_products,mate.receipt.primitive_products);
    result.receipt.phase_products=admission::add(result.receipt.phase_products,mate.receipt.phase_products);
    result.receipt.phase_accumulations=admission::add(result.receipt.phase_accumulations,mate.receipt.phase_accumulations);
    result.receipt.evaluated_coupling_error_bound=std::max(result.receipt.evaluated_coupling_error_bound,mate.receipt.evaluated_coupling_error_bound);
    return result;
}
double PeriodicGDFTiles::apply_pair_tree(const std::vector<Complex>& input,std::vector<Complex>& output,
    std::vector<Complex>& correction,bool forward,PeriodicGDFCouplingReceipt& r,
    std::vector<double>& components,std::vector<double>& magnitudes,std::vector<double>& actions) const {
    PeriodicGDFShortRangeReceipt sr;prepare_pair_tree(sr);
    PeriodicGDFPairContractions(*this,sr,output,correction,&input,&r,forward,{},&components,&magnitudes,&actions).run();
    if(source_.zero_transfer) {
        const Count n=plan_.n_orb,nk=plan_.momenta;
        std::vector<Complex> transformed(input.size()),mate(output.size()),error(output.size());
        if(forward) {
            for(Count k=0;k<nk;++k)for(Count i=0;i<n;++i)for(Count j=0;j<n;++j)
                transformed[(k*n+i)*n+j]=std::conj(input[(k*n+j)*n+i]);
        } else for(Count i=0;i<input.size();++i) transformed[i]=std::conj(input[i]);
        PeriodicGDFShortRangeReceipt sr2;PeriodicGDFCouplingReceipt r2;
        std::vector<double> mate_components(components.size()),mate_magnitudes(magnitudes.size()),mate_actions(actions.size());
        PeriodicGDFPairContractions(*this,sr2,mate,error,&transformed,&r2,forward,{},&mate_components,&mate_magnitudes,&mate_actions).run();
        // Complete the SAME Gamma orbit for omission envelopes. Adjoint
        // components transpose with the numerical output; auxiliary maxima
        // are invariant under that pair permutation. No cancellation credit.
        for(Count i=0;i<components.size();++i) {
            const Count other=forward?i:(i/(n*n)*n*n+(i%n)*n+(i/n)%n);
            components[i]=periodic_error::upper((static_cast<long double>(components[i])+mate_components[other])/2);
            actions[i]=periodic_error::upper((static_cast<long double>(actions[i])+mate_actions[other])/2);
        }
        for(Count i=0;i<magnitudes.size();++i)
            magnitudes[i]=periodic_error::upper((static_cast<long double>(magnitudes[i])+mate_magnitudes[i])/2);
        if(forward) for(Count i=0;i<output.size();++i) output[i]=.5*(output[i]+std::conj(mate[i]));
        else for(Count k=0;k<nk;++k)for(Count i=0;i<n;++i)for(Count j=0;j<n;++j) {
            const Count out=(k*n+i)*n+j;output[out]=.5*(output[out]+std::conj(mate[(k*n+j)*n+i]));
        }
        std::fill(correction.begin(),correction.end(),Complex{});
        add_pair_sr_receipt(sr,sr2);
        using R=PeriodicGDFCouplingReceipt;
        for(auto member:{&R::multipole_compressed_pairs,&R::multipole_compressed_nodes,&R::multipole_action_products,
            &R::multipole_action_translations,&R::multipole_global_pairs,&R::multipole_global_pair_reuses,
            &R::multipole_global_auxiliary,&R::multipole_global_projections,&R::multipole_global_auxiliary_projections,&R::multipole_pair_groups})
            r.*member=admission::add(r.*member,r2.*member);
        r.maximum_coupling=periodic_error::upper((static_cast<long double>(r.maximum_coupling)+r2.maximum_coupling)/2);
        r.approximation_bound=periodic_error::upper((static_cast<long double>(r.approximation_bound)+r2.approximation_bound)/2);
        r.approximation_l2_bound=periodic_error::upper((static_cast<long double>(r.approximation_l2_bound)+r2.approximation_l2_bound)/2);
    }
    add_pair_action_receipt(r,sr);return sr.evaluated_multipole_error;
}

std::shared_ptr<const PeriodicGDFTiles> PeriodicGDFTiles::refine_multipole(double tolerance,const ResourceBudget& budget) const {
    if(!std::isfinite(tolerance) || tolerance<=0. || tolerance>options_.coupling_multipole_error)
        throw std::invalid_argument("MANTA multipole refinement requires a tighter positive tolerance");
    auto options=options_;options.coupling_multipole_error=tolerance;
    return std::make_shared<PeriodicGDFTiles>(orbital_,auxiliary_,system_,q_,k_,vectors_,options,budget,physical_);
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
    if(!options_.coupling_multipole_pair_tree) return evaluate_impl(ip,im,in,nullptr);
    auto result=pair_tree_tile(ip,im,in);const auto residual=evaluate_impl(ip,im,in,nullptr,true);
    for(Count i=0;i<result.values.size();++i) {
        result.values[i]+=residual.values[i];
        if(!std::isfinite(result.values[i].real()) || !std::isfinite(result.values[i].imag()))
            throw std::overflow_error("MANTA global GDF tile is nonfinite");
    }
    auto& a=result.receipt;const auto& b=residual.receipt;
    a.shell_calls=admission::add(a.shell_calls,b.shell_calls);a.primitive_products=admission::add(a.primitive_products,b.primitive_products);
    a.vector_primitive_products=b.vector_primitive_products;a.reciprocal_vectors=b.reciprocal_vectors;
    a.auxiliary_fourier_shell_calls=b.auxiliary_fourier_shell_calls;a.pair_fourier_shell_calls=b.pair_fourier_shell_calls;
    a.phase_products=admission::add(a.phase_products,b.phase_products);a.phase_accumulations=admission::add(a.phase_accumulations,b.phase_accumulations);
    a.contraction_terms=b.contraction_terms;return result;
}
PeriodicGDFTile PeriodicGDFTiles::evaluate_impl(Count ip,Count im,Count in,PeriodicGDFMultipoleAction* action,bool residual) const {
    if(ip>=auxiliary_.nshells() || im>=orbital_.nshells() || in>=orbital_.nshells())
        throw std::out_of_range("MANTA GDF coupling shell index");
    const auto& a=auxiliary_.libint();const auto& b=orbital_.libint();const auto& sp=a[ip];const auto& sm=b[im];const auto& sn=b[in];
    const Count np=sp.size(),nm=sm.size(),nn=sn.size(),nk=plan_.momenta,tile=np*nm*nn;
    const auto ai=a.shell2bf(),bi=b.shell2bf();PeriodicGDFTile result;
    result.shape={nk,np,nm,nn};result.offsets={0,ai[ip],bi[im],bi[in]};result.receipt=source_;auto& r=result.receipt;
    result.values.assign(admission::mul(nk,tile),{});std::vector<Complex> error(result.values.size()),contracted(tile),correction(tile);
    const ThreeCenterImageDomain domain(system_,options_.pair_cutoff_bohr,options_.auxiliary_cutoff_bohr,options_.omega,
        options_.image_candidate_cap,coupling_screen(options_));
    auto ket=sn;
    auto multipole_options=options_;if(!source_.real_short_range || residual) multipole_options.coupling_multipole_error=0.;
    OrderedCouplingMultipoles multipole(sp,sm,sn,ip,domain,multipole_options,r.short_range,preparation_.get(),im,in,action,partitions_.get());
    if(source_.real_short_range && !residual) domain.each(sp,sm,sn,[&](const Eigen::Vector3d& t) {
        if(action) action->start(t);
        ket=shifted(sn,t);multipole.start(sm,ket,im,in);std::fill(contracted.begin(),contracted.end(),Complex{});std::fill(correction.begin(),correction.end(),Complex{});
        ++r.short_range.pair_shell_images;
    },[&](const Eigen::Vector3d&,const Eigen::Vector3d& t) {
        multipole.accumulate(t,sp,sm,ket,q_,contracted,correction);
    },[&](const Eigen::Vector3d& t) {
        multipole.finish_pair(sp,sm,ket,q_,contracted,correction);
        for(Count ik=0;ik<nk;++ik) {
            const auto phase=fitting_phase(k_.row(ik).dot(t),r.short_range);
            for(Count i=0;i<tile;++i) { fitting_add(phase*contracted[i],result.values[ik*tile+i],error[ik*tile+i]);++r.short_range.phase_accumulations; }
        }
    });
    multipole.finish();
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
PeriodicGDFCouplingPlan plan_periodic_gdf_coupling(const PeriodicGDFTilesPlan& source,const ResourceBudget& budget) {
    using namespace admission;
    margin(budget);
    if(!source.n_aux || !source.n_orb || !source.momenta)
        throw std::invalid_argument("MANTA GDF coupling requires a nonempty finite source");
    PeriodicGDFCouplingPlan p;
    p.n_aux=source.n_aux;p.n_orb=source.n_orb;p.momenta=source.momenta;
    p.maximum_tile_elements=source.maximum_tile_elements;
    const Count mesh=mul(p.momenta,square(p.n_orb)),elements=mul(p.n_aux,mesh);
    // Source snapshot/single-tile workspace once, plus input, binding copies,
    // compensated output and owned export. No auxiliary-by-AO-pair tensor.
    // Also cover component/reference envelopes, their Gamma mates, residual
    // workspace and owned binding exports. No auxiliary-by-pair array.
    const Count action_bytes=mul(192,add(mesh,p.n_aux));
    p.modeled_peak_bytes=add(source.coupling_peak_bytes-source.external_live_bytes,
        add(action_bytes,add(budget.other_live_bytes,budget.backend_margin_bytes)));
    p.shell_calls=source.coupling_shell_calls;p.primitive_products=source.coupling_primitive_products;
    const Count scalar=mul(64,add(elements,add(mesh,p.n_aux)));
    p.scalar_terms=add(source.coupling_scalar_terms,scalar);
    p.kernel_terms=add(source.coupling_kernel_terms,mul(8,scalar));
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA GDF coupling action memory budget exceeded");
    cap(p.shell_calls,budget.shell_calls,"MANTA GDF coupling action shell-call budget exceeded");
    cap(p.primitive_products,budget.primitive_quartets,"MANTA GDF coupling action primitive budget exceeded");
    cap(p.scalar_terms,budget.scalar_terms,"MANTA GDF coupling action scalar budget exceeded");
    cap(p.kernel_terms,budget.kernel_terms,"MANTA GDF coupling action work budget exceeded");
    return p;
}

PeriodicGDFCouplingPlan PeriodicGDFTiles::plan_coupling(const ResourceBudget& budget) const {
    return plan_periodic_gdf_coupling(plan_,budget);
}
PeriodicGDFCouplingPlan plan_periodic_gdf_coupling_norm_bounds(const PeriodicGDFCouplingPlan& base,const ResourceBudget& budget) {
    using namespace admission;margin(budget);auto p=base;
    p.modeled_peak_bytes=add(p.modeled_peak_bytes,mul(32,mul(p.momenta,square(p.n_orb))));
    // Same traversal and magnitude panels, with outward elementary arithmetic.
    p.scalar_terms=mul(32,p.scalar_terms);p.kernel_terms=mul(32,p.kernel_terms);
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA coupling norm memory budget exceeded");
    cap(p.shell_calls,budget.shell_calls,"MANTA coupling norm shell-call budget exceeded");
    cap(p.primitive_products,budget.primitive_quartets,"MANTA coupling norm primitive budget exceeded");
    cap(p.scalar_terms,budget.scalar_terms,"MANTA coupling norm scalar budget exceeded");
    cap(p.kernel_terms,budget.kernel_terms,"MANTA coupling norm work budget exceeded");return p;
}
PeriodicGDFCouplingPlan PeriodicGDFTiles::plan_coupling_norm_bounds(const ResourceBudget& budget) const {
    if(plan_.pair_tree_bytes)throw std::invalid_argument("MANTA metric coupling norms do not yet support pair-tree actions");
    return plan_periodic_gdf_coupling_norm_bounds(plan_coupling(budget),budget);
}
PeriodicGDFCouplingResult PeriodicGDFTiles::coupling_norm_bounds(const ResourceBudget& budget) const {
    const auto p=plan_coupling_norm_bounds(budget);verified_coupling::require_arithmetic();
    return apply_coupling(std::vector<Complex>(p.momenta*p.n_orb*p.n_orb),true,budget,true);
}
PeriodicGDFCouplingResult PeriodicGDFTiles::forward(const std::vector<Complex>& input,const ResourceBudget& budget) const {
    return apply_coupling(input,true,budget);
}
PeriodicGDFCouplingResult PeriodicGDFTiles::adjoint(const std::vector<Complex>& input,const ResourceBudget& budget) const {
    return apply_coupling(input,false,budget);
}
PeriodicGDFCouplingResult PeriodicGDFTiles::apply_coupling(const std::vector<Complex>& input,bool forward,
    const ResourceBudget& budget,bool verified) const {
    using namespace admission;
    PeriodicGDFCouplingResult result;result.plan=verified?plan_coupling_norm_bounds(budget):plan_coupling(budget);
    if(verified)verified_coupling::require_arithmetic();
    const auto& p=result.plan;const Count n2=square(p.n_orb),mesh=mul(p.momenta,n2);
    if(input.size()!=(forward?mesh:p.n_aux)) throw std::invalid_argument("MANTA GDF coupling input dimensions");
    FittingDigest values;values.text("manta.periodic-gdf-coupling-input.v1");
    values.text(forward?"AO-pair":"auxiliary");values.word(input.size());
    for(const auto z:input) { require_finite(z.real());require_finite(z.imag());values.real(z.real());values.real(z.imag()); }
    auto& r=result.receipt;r.direction=forward?"forward":"adjoint";
    r.coupling_identity=source_.three_center_source_identity;r.input_identity=values.hash.finish_hex();
    r.reference_coupling_identity=source_.reference_three_center_source_identity;
    r.coupling_error_bound=source_.coupling_error_bound;r.approximation_evaluated=r.coupling_error_bound>0.;
    if(r.coupling_error_bound>0.) {
        long double norm=0.;
        for(const auto z:input) periodic_error::positive_add(periodic_error::upper(
            std::hypot(static_cast<long double>(z.real()),static_cast<long double>(z.imag()))),norm);
        r.input_l1_bound=periodic_error::upper(norm);
        r.approximation_bound=periodic_error::upper(static_cast<long double>(r.coupling_error_bound)*r.input_l1_bound);
    }
    if(plan_.compressed_action_bytes) r.algorithm="ordered-periodic-gdf-compressed-v1";
    if(plan_.global_action_bytes) r.algorithm="ordered-periodic-gdf-global-panels-v1";
    if(plan_.pair_tree_bytes) r.algorithm="ordered-periodic-gdf-global-pair-tree-v1";
    FittingDigest snapshot;snapshot.text(r.algorithm);snapshot.text(r.coupling_identity);
    snapshot.text(r.direction);snapshot.text(r.input_identity);r.snapshot_identity=snapshot.hash.finish_hex();
    r.modeled_peak_bytes=p.modeled_peak_bytes;
    result.shape=forward?std::vector<Count>{p.n_aux}:std::vector<Count>{p.momenta,p.n_orb,p.n_orb};
    result.values.assign(forward?p.n_aux:mesh,{});std::vector<Complex> correction(result.values.size());
    if(verified)result.metric_column_norm_bounds.assign(mesh,0.);
    // Sun et al. (2017), DOI 10.1063/1.4998644, Eqs.16-17 supplies T(k-q,k).
    // A paired coupling/adjoint is the prerequisite for RI-BUPO's quadratic
    // energy (Neese et al., 2025, DOI 10.1021/acs.jpca.4c07415, Eqs.29-36).
    // Both directions use the same ordered producer partition; hierarchical
    // far blocks may contract through moments without a coupling tile. Near,
    // reciprocal and zero terms keep the common tile path. No canonical pair
    // fold, density-dependent selection or metric refit is used. The opt-in
    // global pair route completes Gamma conjugate orbits in the source
    // coupling after qualifying the literal finite support.
    double evaluated_error=0.;
    result.approximation_bounds.assign(result.values.size(),0.);
    if(r.approximation_evaluated) {
        result.reference_magnitude_bounds.assign(p.n_aux,0.);
        result.reference_action_bounds.assign(result.values.size(),0.);
    }
    if(plan_.pair_tree_bytes) evaluated_error=apply_pair_tree(input,result.values,correction,forward,r,
        result.approximation_bounds,result.reference_magnitude_bounds,result.reference_action_bounds);
    const double pair_magnitude=r.maximum_coupling;
    const auto pair_reference_magnitudes=result.reference_magnitude_bounds;
    std::fill(result.reference_magnitude_bounds.begin(),result.reference_magnitude_bounds.end(),0.);
    const auto& a=auxiliary_.libint();const auto& b=orbital_.libint();
    const auto ao=a.shell2bf(),bo=b.shell2bf();const bool compressed=plan_.compressed_action_bytes>0;
    PeriodicGDFGlobalAction global(input,result.values,correction,r,k_,forward,plan_);
    for(Count ip=0;ip<auxiliary_.nshells();++ip)for(Count im=0;im<orbital_.nshells();++im)
        for(Count in=0;in<orbital_.nshells();++in) {
            PeriodicGDFMultipoleAction action(input,result.values,correction,r,k_,forward,p.n_orb,
                a[ip].size(),b[im].size(),b[in].size(),ao[ip],bo[im],bo[in],plan_.global_action_bytes?&global:nullptr,verified);
            const auto tile=evaluate_impl(ip,im,in,compressed?&action:nullptr,plan_.pair_tree_bytes>0);++r.shell_tiles;
            double residual_maximum=0.;
            evaluated_error=std::max(evaluated_error,tile.receipt.evaluated_coupling_error_bound);
            r.multipole_partition_builds=add(r.multipole_partition_builds,tile.receipt.short_range.multipole_partition_builds);
            r.multipole_partition_reuses=add(r.multipole_partition_reuses,tile.receipt.short_range.multipole_partition_reuses);
            r.multipole_partition_bypasses=add(r.multipole_partition_bypasses,tile.receipt.short_range.multipole_partition_bypasses);
            r.multipole_trials=add(r.multipole_trials,tile.receipt.short_range.multipole_trials);
            r.multipole_images=add(r.multipole_images,tile.receipt.short_range.multipole_images);
            r.multipole_moment_panels=add(r.multipole_moment_panels,tile.receipt.short_range.multipole_moment_panels);
            r.multipole_tree_nodes=add(r.multipole_tree_nodes,tile.receipt.short_range.multipole_tree_nodes);
            r.multipole_far_blocks=add(r.multipole_far_blocks,tile.receipt.short_range.multipole_far_blocks);
            r.multipole_translations=add(r.multipole_translations,tile.receipt.short_range.multipole_translations);
            r.multipole_reused_extents=add(r.multipole_reused_extents,tile.receipt.short_range.multipole_reused_extents);
            r.multipole_reused_moment_panels=add(r.multipole_reused_moment_panels,tile.receipt.short_range.multipole_reused_moment_panels);
            r.multipole_envelope_evaluations=add(r.multipole_envelope_evaluations,tile.receipt.short_range.multipole_envelope_evaluations);
            r.shell_calls=add(r.shell_calls,tile.receipt.shell_calls);
            r.primitive_products=add(r.primitive_products,tile.receipt.primitive_products);
            r.integral_elements_consumed=add(r.integral_elements_consumed,tile.values.size());
            const Count np=tile.shape[1],nm=tile.shape[2],nn=tile.shape[3];
            for(Count k=0;k<p.momenta;++k)for(Count a=0;a<np;++a)for(Count i=0;i<nm;++i)for(Count j=0;j<nn;++j) {
                const Count aux=tile.offsets[1]+a,pair=k*n2+(tile.offsets[2]+i)*p.n_orb+tile.offsets[3]+j;
                const auto t=tile.values[((k*np+a)*nm+i)*nn+j];const Count out=forward?aux:pair;
                if(verified) {
                    const double norm=verified_coupling::plus_up(verified_coupling::magnitude(verified_coupling::point(t)),
                        compressed?action.verified_far_magnitude:0.);
                    result.metric_column_norm_bounds[pair]=verified_coupling::plus_up(result.metric_column_norm_bounds[pair],norm);
                }
                if(r.coupling_error_bound>0.) {
                    residual_maximum=std::max(residual_maximum,std::abs(t));
                    // Exact LR/zero residual plus the source-owned SR bound;
                    // other producers use their own far/omission envelope.
                    const long double magnitude=std::hypot(static_cast<long double>(t.real()),static_cast<long double>(t.imag()));
                    const long double omitted=plan_.pair_tree_bytes?pair_reference_magnitudes[aux]:
                        static_cast<long double>(tile.receipt.evaluated_coupling_error_bound)+(compressed?action.far_magnitude:0.);
                    result.reference_magnitude_bounds[aux]=std::max(result.reference_magnitude_bounds[aux],
                        periodic_error::upper(magnitude+omitted));
                    // The pair route already accumulated its SR action from
                    // the same shell-block ledger. Add only the exact residual;
                    // other sources include their own uniform omitted part.
                    const auto x=input[forward?pair:aux];
                    const long double input_magnitude=periodic_error::upper(std::hypot(
                        static_cast<long double>(x.real()),static_cast<long double>(x.imag())));
                    const long double term=(magnitude+(plan_.pair_tree_bytes?0.L:omitted))*input_magnitude;
                    result.reference_action_bounds[out]=periodic_error::upper(
                        static_cast<long double>(result.reference_action_bounds[out])+term);
                }
                const auto value=forward?std::conj(t)*input[pair]:t*input[aux];
                if(!std::isfinite(value.real()) || !std::isfinite(value.imag()))
                    throw std::overflow_error("MANTA GDF coupling contraction is nonfinite");
                fitting_add(value,result.values[out],correction[out]);++r.contraction_products;
            }
            if(r.coupling_error_bound>0.) r.maximum_coupling=std::max(r.maximum_coupling,
                plan_.pair_tree_bytes?periodic_error::upper(static_cast<long double>(residual_maximum)+pair_magnitude):
                compressed?periodic_error::upper(static_cast<long double>(residual_maximum)+action.far_magnitude):residual_maximum);
        }
    global.finish();
    for(auto z:result.values) if(!std::isfinite(z.real()) || !std::isfinite(z.imag()))
        throw std::overflow_error("MANTA GDF coupling output is nonfinite");
    if(r.coupling_error_bound>0.) {
        r.maximum_coupling=periodic_error::upper(r.maximum_coupling);
        r.coupling_error_bound=evaluated_error;
        const double uniform=periodic_error::upper(static_cast<long double>(evaluated_error)*r.input_l1_bound);
        r.approximation_bound=plan_.pair_tree_bytes?std::min(r.approximation_bound,uniform):uniform;
        const double l2=periodic_error::upper(std::sqrt(static_cast<long double>(result.values.size()))*r.approximation_bound);
        r.approximation_l2_bound=plan_.pair_tree_bytes?std::min(r.approximation_l2_bound,l2):l2;
        for(auto& value:result.approximation_bounds)
            value=plan_.pair_tree_bytes?std::min(value,r.approximation_bound):r.approximation_bound;
        const double reference_maximum=periodic_error::upper(static_cast<long double>(r.maximum_coupling)+r.coupling_error_bound);
        for(auto& value:result.reference_magnitude_bounds) value=std::min(value,reference_maximum);
        const double absolute_action=periodic_error::upper(static_cast<long double>(reference_maximum)*r.input_l1_bound);
        for(Count i=0;i<result.reference_action_bounds.size();++i) {
            const double column_action=forward?periodic_error::upper(
                static_cast<long double>(result.reference_magnitude_bounds[i])*r.input_l1_bound):absolute_action;
            result.reference_action_bounds[i]=std::min(result.reference_action_bounds[i],column_action);
        }
    }
    cap(r.shell_calls,p.shell_calls,"MANTA GDF coupling executed shell bound exceeded");
    cap(r.primitive_products,p.primitive_products,"MANTA GDF coupling executed primitive bound exceeded");
    if(r.integral_elements_consumed!=mul(p.n_aux,mesh) || r.contraction_products!=r.integral_elements_consumed)
        throw std::logic_error("MANTA GDF coupling traversal is incomplete");
    result.metric_norm_qualified=verified;
    return result;
}
} // namespace manta
}  // namespace vibeqc
