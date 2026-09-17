#include "vibeqc/opentrustregion.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#ifdef VIBEQC_HAS_OPENTRUSTREGION
#include <opentrustregion.h>
#endif

namespace vibeqc {
bool has_opentrustregion() {
#ifdef VIBEQC_HAS_OPENTRUSTREGION
    return true;
#else
    return false;
#endif
}
void validate_opentrustregion(const OpenTrustRegionOptions& o, int max_iter) {
    if (o.subsystem_solver != "davidson" && o.subsystem_solver != "jacobi-davidson"
        && o.subsystem_solver != "tcg")
        throw std::invalid_argument("OpenTrustRegion subsystem_solver must be davidson, jacobi-davidson, or tcg");
    if (o.stability != "none" && o.stability != "check" && o.stability != "follow")
        throw std::invalid_argument("OpenTrustRegion stability must be none, check, or follow");
    for (double x : {o.gradient_rms_tolerance, o.initial_trust_radius,
                     o.stability_tolerance, o.stability_residual_tolerance})
        if (!std::isfinite(x) || x <= 0)
            throw std::invalid_argument("OpenTrustRegion tolerances and radius must be finite and positive");
    if (o.stability_residual_tolerance > 0.1 * o.stability_tolerance)
        throw std::invalid_argument("OpenTrustRegion stability residual tolerance must be at most 0.1 times its stability threshold");
    if (max_iter < 1 || o.max_micro_iterations < 1 || o.stability_max_iterations < 1 || o.seed < 0)
        throw std::invalid_argument("OpenTrustRegion iteration limits must be positive and seed nonnegative; NOITER is unsupported");
    if (!has_opentrustregion())
        throw std::runtime_error("OpenTrustRegion backend unavailable: rebuild with -DVIBEQC_ENABLE_OPENTRUSTREGION=ON (see installation documentation)");
}
#ifdef VIBEQC_HAS_OPENTRUSTREGION
namespace {
static_assert(sizeof(c_int) == 4, "vibe-qc OpenTrustRegion requires the LP64/int32 ABI");
static_assert(sizeof(c_real) == 8 && sizeof(c_bool) == 1, "Unexpected OpenTrustRegion ABI");
std::mutex otr_mutex;
thread_local bool inside_otr = false;
struct Context {
    OrbitalObjective& objective;
    OpenTrustRegionReport& report;
    OrbitalModel model;
    std::exception_ptr exception;
    double hessian_shift = 0.0;
};
// This pointer is protected by otr_mutex, including stability_check and init.
Context* active = nullptr;
struct Entry {
    std::unique_lock<std::mutex> lock;
    explicit Entry(Context& c) {
        if (inside_otr) throw std::runtime_error("OpenTrustRegion nested entry is unsupported");
        lock = std::unique_lock<std::mutex>(otr_mutex);
        inside_otr = true;
        active = &c;
    }
    ~Entry() { active = nullptr; inside_otr = false; }
};
template<class F> c_int guard(F&& f) noexcept {
    try {
        if (active->exception) return 1;
        f();
        return 0;
    } catch (...) {
        active->exception = std::current_exception();
        return 1; // Upstream adds callback origin (1100/1200/1300/1500).
    }
}
Eigen::VectorXd vector(const double* p) {
    Eigen::VectorXd v = Eigen::Map<const Eigen::VectorXd>(p, active->objective.size());
    if (!v.allFinite()) throw std::runtime_error("OpenTrustRegion supplied a nonfinite vector");
    return v;
}
c_int hessian_callback(const double* v, double* out) noexcept {
    return guard([&] {
        const auto x = vector(v);
        auto y = active->objective.hessian(x);
        ++active->report.response_evaluations;
        if (y.size() != x.size() || !y.allFinite())
            throw std::runtime_error("OpenTrustRegion Hessian action has invalid size or nonfinite values");
        y.noalias() += active->hessian_shift * x;
        Eigen::Map<Eigen::VectorXd>(out, y.size()) = y;
    });
}
c_int update_callback(const double* k, double* e, double* g, double* d, hess_x_fp* h) noexcept {
    return guard([&] {
        auto m = active->objective.update(vector(k));
        const int n = active->objective.size();
        if (!std::isfinite(m.energy) || m.gradient.size() != n || m.diagonal.size() != n
            || !m.gradient.allFinite() || !m.diagonal.allFinite())
            throw std::runtime_error("OpenTrustRegion objective returned an invalid model");
        active->model = std::move(m);
        *e = active->model.energy;
        Eigen::Map<Eigen::VectorXd>(g, n) = active->model.gradient;
        Eigen::Map<Eigen::VectorXd>(d, n) = active->model.diagonal;
        *h = hessian_callback;
        ++active->report.accepted_evaluations;
    });
}
c_int trial_callback(const double* k, double* e) noexcept {
    return guard([&] {
        *e = active->objective.trial(vector(k));
        ++active->report.trial_evaluations;
        if (!std::isfinite(*e)) throw std::runtime_error("OpenTrustRegion trial energy is nonfinite");
    });
}
c_int convergence_callback(bool* converged) noexcept {
    return guard([&] { *converged = active->objective.converged(); });
}
void logger_callback(const char* message) noexcept {
    // Even allocation failures must not unwind through Fortran.
    guard([&] { active->report.log.emplace_back(message); });
}
void verify_settings_abi(const solver_settings_type& s) {
    if (!s.initialized || s.conv_tol != 1e-5 || s.start_trust_radius != 0.4
        || s.n_macro != 150 || s.n_micro != 50 || s.seed != 42
        || std::string(s.subsystem_solver) != "davidson"
        || s.stability_settings.n_iter != 100 || s.stability_settings.conv_tol != 1e-8)
        throw std::runtime_error("OpenTrustRegion C/Fortran settings ABI mismatch; use the pinned int32 build");
}
}
#endif
OpenTrustRegionReport optimize_orbitals(OrbitalObjective& objective,
    const OpenTrustRegionOptions& o, int max_iter) {
    validate_opentrustregion(o, max_iter);
    OpenTrustRegionReport r;
#ifdef VIBEQC_HAS_OPENTRUSTREGION
    if (objective.size() < 0) throw std::invalid_argument("Negative orbital parameter count");
    Context context{objective, r, {}, {}, 0.0};
    Entry entry(context);
    r.backend = "opentrustregion";
    r.subsystem_solver = o.subsystem_solver;
    r.stability_policy = o.stability;
    r.stability_threshold = o.stability_tolerance;
    r.version = "2.0.0 (8fa7769ae66233a566868a6bf03cdcbdb1ee69d0)";
    auto s = solver_settings_init();
    verify_settings_abi(s);
    s.logger = logger_callback;
    s.verbose = 3;
    s.conv_check = convergence_callback;
    // The host callback is the physical convergence contract. Upstream also
    // stops at machine precision; the host revalidates the returned state.
    s.conv_tol = std::numeric_limits<double>::min();
    s.start_trust_radius = o.initial_trust_radius;
    s.n_macro = max_iter;
    s.n_micro = o.max_micro_iterations;
    s.seed = o.seed;
    s.line_search = o.line_search;
    s.stability = o.stability == "follow";
    std::strcpy(s.subsystem_solver, o.subsystem_solver.c_str());
    s.stability_settings.n_iter = o.stability_max_iterations;
    s.stability_settings.conv_tol = o.stability_residual_tolerance;
    s.stability_settings.seed = o.seed;
    s.stability_settings.logger = logger_callback;
    if (objective.size() == 0) {
        // Upstream divides by sqrt(n) and rejects n=0. Identity is exact here.
        double energy = 0;
        hess_x_fp action = nullptr;
        r.error_code = update_callback(nullptr, &energy, nullptr, nullptr, &action);
        r.termination = "no_rotation_parameters";
        r.stability_checked = o.stability != "none";
        r.stability_converged = r.stability_checked;
        r.stable = r.stability_checked;
    } else {
        r.error_code = solver(update_callback, trial_callback, objective.size(), s);
        r.termination = r.error_code == 0 ? "solver_terminated" :
            (r.error_code == 102 ? "iteration_limit" : "upstream_error");
        if (r.error_code == 202) {
            // The solver can itself invoke stability at an initial stationary
            // point or with follow enabled. This is the pinned C error code,
            // not a verdict inferred from diagnostic text.
            r.termination = "stability_inconclusive";
            r.stability_checked = true;
            r.stability_threshold = 0.01;
        }
        if (!r.error_code && o.stability != "none" && !context.exception) {
            // Pinned upstream hardcodes stable iff lambda > -0.01. Shift H
            // by (tol - 0.01) so this *separate verdict* tests lambda > -tol.
            // The physical Hessian passed to the optimizer is never shifted.
            context.hessian_shift = o.stability_tolerance - 0.01;
            Eigen::VectorXd diag = context.model.diagonal.array() + context.hessian_shift;
            Eigen::VectorXd mode = Eigen::VectorXd::Zero(objective.size());
            bool stable = false;
            const int code = stability_check(diag.data(), hessian_callback,
                objective.size(), &stable, s.stability_settings, mode.data());
            r.stability_checked = true;
            r.stability_threshold = o.stability_tolerance;
            r.stability_converged = code == 0;
            r.stable = code == 0 && stable;
            if (code) { r.error_code = code; r.termination = "stability_inconclusive"; }
        }
    }
    if (context.exception) {
        r.termination = "callback_failure";
        try { std::rethrow_exception(context.exception); }
        catch (const std::exception& e) { r.callback_error = e.what(); }
        catch (...) { r.callback_error = "Unknown callback exception"; }
    }
    if (objective.size() == 0) r.gradient_rms = 0.0;
    if (context.model.gradient.size())
        r.gradient_rms = context.model.gradient.norm() / std::sqrt(double(objective.size()));
#endif
    return r;
}
} // namespace vibeqc
