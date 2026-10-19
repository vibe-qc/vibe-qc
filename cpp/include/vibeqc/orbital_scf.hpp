#pragma once
#include "vibeqc/orbital_objective.hpp"
#include "vibeqc/rhf.hpp"

namespace vibeqc {
template<class Options> void validate_orbital_optimizer(const Options& opts) {
    if (opts.orbital_optimizer == "native") return;
    if (opts.orbital_optimizer != "opentrustregion")
        throw std::invalid_argument("orbital_optimizer must be native or opentrustregion");
    validate_opentrustregion(opts.opentrustregion, opts.max_iter);
    if (opts.newton_threshold != 0 || opts.trah_threshold != 0 || opts.soscf_threshold != 0
        || opts.quadratic_fallback_iter != 0)
        throw std::invalid_argument("OpenTrustRegion conflicts with explicit Newton/TRAH/SOSCF/quadratic settings; disable their thresholds");
    if (opts.cosx || !opts.dft_plus_u_sites.empty())
        throw std::invalid_argument("OpenTrustRegion does not support COSX or DFT+U response; use exact/RIJK exchange and disable +U, or select native");
}
inline void validate_orbital_xc(const Functional& f) {
    if (f.is_range_separated() || f.kind() == XCKind::MGGA || f.is_external() || f.needs_vv10())
        throw std::invalid_argument("OpenTrustRegion supports native LDA/GGA and global hybrids only; range-separated, meta-GGA and nonlocal XC response are unsupported");
}
struct OrbitalSCFRun {
    MolecularOrbitalState state;
    OpenTrustRegionReport report;
    std::vector<SCFIteration> trace;
    bool converged = false;
};
template<class Options> OrbitalSCFRun run_orbital_scf(const Options& opts,
    const Eigen::MatrixXd& X, const Eigen::MatrixXd& S, const Eigen::MatrixXd& H,
    double nuclear, const JKBuilder& jk, const Eigen::MatrixXd& da,
    const Eigen::MatrixXd& db, int na, int nb, bool restricted, double exchange,
    OrbitalXCFunction xc = {}) {
    validate_orbital_optimizer(opts);
    if (jk.has_post_scf_exchange_correction())
        throw std::invalid_argument("OpenTrustRegion requires a consistent energy/response JK builder; COSX correction is unsupported");
    jk.set_schwarz_threshold(opts.schwarz_threshold);
    jk.reset_state();
    const Eigen::MatrixXd ca = orbital_seed(X,S,da,na,restricted?2:1);
    const Eigen::MatrixXd cb = restricted ? ca : orbital_seed(X,S,db,nb,1);
    MolecularOrbitalObjective objective(S,H,nuclear,jk,ca,cb,na,nb,restricted,
        exchange,std::move(xc),opts.conv_tol_energy,opts.conv_tol_grad,
        opts.opentrustregion.gradient_rms_tolerance);
    // Establish a physical energy predecessor, including stationary restarts.
    objective.update(Eigen::VectorXd::Zero(objective.size()));
    OrbitalSCFRun out;
    out.report = optimize_orbitals(objective, opts.opentrustregion, opts.max_iter);
    out.report.manifold = restricted ? "real restricted occupied-virtual" : "real unrestricted occupied-virtual";
    out.state = objective.final_state();
    out.report.energy_converged = objective.energy_converged()
        && std::abs(out.state.energy - objective.state().energy) < opts.conv_tol_energy;
    out.report.gradient_converged = objective.gradient_converged()
        && out.state.residual < opts.conv_tol_grad;
    out.report.fock_evaluations = objective.fock_evaluations;
    out.report.final_residual = out.state.residual;
    out.report.gradient_rms = objective.gradient_rms();
    out.converged = out.report.error_code == 0 && out.report.callback_error.empty()
        && out.report.energy_converged && out.report.gradient_converged;
    if (out.converged) out.report.termination = objective.size() ? "converged" : "no_rotation_parameters";
    else if (out.report.termination == "solver_terminated") out.report.termination = "physical_convergence_failed";
    int iter = 0;
    for (const auto& row : objective.trace()) {
        if (!iter++) continue; // Initial predecessor, not a solver callback.
        SCFIteration r;
        r.iter = iter-1; r.energy = row.energy; r.delta_e = row.delta_e;
        r.grad_norm = row.residual; r.wall_s = row.wall_s;
        out.trace.push_back(r);
    }
    return out;
}
template<class Result> void assign_orbital_common(Result& r, const OrbitalSCFRun& run, double nuclear) {
    r.opentrustregion = run.report;
    r.scf_trace = run.trace;
    r.n_iter = run.report.accepted_evaluations;
    r.energy = run.state.energy;
    r.e_electronic = r.energy-nuclear;
    r.converged = run.converged;
    // OTR's verdict has its own manifold/threshold metadata. Do not populate
    // legacy eigenvalue fields: the C API does not return an eigenvalue.
    r.internal_instability = run.report.stability_converged && !run.report.stable;
}
template<class Result> void assign_orbital_restricted(Result& r, const OrbitalSCFRun& run, double nuclear) {
    assign_orbital_common(r,run,nuclear);
    r.mo_coeffs = run.state.ca; r.mo_energies = run.state.epsa;
    r.density = run.state.da; r.fock = run.state.fa;
}
template<class Result> void assign_orbital_unrestricted(Result& r, const OrbitalSCFRun& run,
    double nuclear, const Eigen::MatrixXd& S, int na, int nb) {
    assign_orbital_common(r,run,nuclear);
    r.mo_coeffs_alpha = run.state.ca; r.mo_energies_alpha = run.state.epsa;
    r.mo_coeffs_beta = run.state.cb; r.mo_energies_beta = run.state.epsb;
    r.density_alpha = run.state.da; r.fock_alpha = run.state.fa;
    r.density_beta = run.state.db; r.fock_beta = run.state.fb;
    const double spin = 0.5*(na-nb);
    r.s_squared_ideal = spin*(spin+1);
    r.s_squared = r.s_squared_ideal + nb
        - (run.state.ca.leftCols(na).transpose()*S*run.state.cb.leftCols(nb)).squaredNorm();
}
} // namespace vibeqc
