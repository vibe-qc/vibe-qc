#pragma once
#include "vibeqc/opentrustregion.hpp"
#include "vibeqc/jk_builder.hpp"
#include "vibeqc/xc_kernel.hpp"
#include <memory>
#include <chrono>

namespace vibeqc {
struct OrbitalXC {
    double energy = 0.0;
    Eigen::MatrixXd alpha, beta;
    std::shared_ptr<XCKernelBuilder> restricted_kernel;
    std::shared_ptr<UHFXCKernelBuilder> unrestricted_kernel;
};
// Kernel construction is requested only for accepted states. Trial evaluations
// own their potential buffers and never replace the frozen response kernel.
using OrbitalXCFunction = std::function<OrbitalXC(
    const Eigen::MatrixXd&, const Eigen::MatrixXd&, bool)>;
struct MolecularOrbitalState {
    Eigen::MatrixXd ca, cb, da, db, fa, fb;
    Eigen::VectorXd epsa, epsb;
    OrbitalXC xc;
    double energy = 0, core = 0, coulomb = 0, exchange = 0;
    double residual = 0;
};
struct OrbitalTrace {
    double energy, delta_e, residual, wall_s;
};
class MolecularOrbitalObjective final : public OrbitalObjective {
public:
    // C exp(K), K_ai=k_ai, K_ia=-k_ai. Column-major (a + nvir*i),
    // alpha before beta; Euclidean parameter norm. Restricted D=2 Co Co^T,
    // g=4 F_ai; unrestricted D_sigma=Co Co^T, g_sigma=2 F_sigma_ai.
    MolecularOrbitalObjective(const Eigen::MatrixXd& S, const Eigen::MatrixXd& H,
        double nuclear, const JKBuilder& jk, const Eigen::MatrixXd& ca,
        const Eigen::MatrixXd& cb, int na, int nb, bool restricted,
        double exchange, OrbitalXCFunction xc, double energy_tol,
        double residual_tol, double rms_tol);
    int size() const override;
    OrbitalModel update(const Eigen::VectorXd&) override;
    double trial(const Eigen::VectorXd&) override;
    Eigen::VectorXd hessian(const Eigen::VectorXd&) override;
    bool converged() const override;
    const MolecularOrbitalState& state() const { return accepted_; }
    MolecularOrbitalState final_state();
    const std::vector<OrbitalTrace>& trace() const { return trace_; }
    bool energy_converged() const;
    bool gradient_converged() const;
    double gradient_rms() const { return gradient_rms_; }
    int fock_evaluations = 0;
private:
    MolecularOrbitalState evaluate(const Eigen::MatrixXd&, const Eigen::MatrixXd&, bool);
    Eigen::MatrixXd rotate(const Eigen::MatrixXd&, const Eigen::VectorXd&, int, int) const;
    OrbitalModel model(const MolecularOrbitalState&) const;
    const Eigen::MatrixXd S_, H_;
    const double nuclear_, exchange_, energy_tol_, residual_tol_, rms_tol_;
    const JKBuilder& jk_;
    const int na_, nb_;
    const bool restricted_;
    OrbitalXCFunction xc_;
    MolecularOrbitalState accepted_;
    std::vector<OrbitalTrace> trace_;
    double gradient_rms_ = 0;
    std::chrono::steady_clock::time_point last_update_ = std::chrono::steady_clock::now();
};
OrbitalXCFunction make_orbital_rks_xc(const BasisSet&, const Grid&, const std::string&);
OrbitalXCFunction make_orbital_uks_xc(const BasisSet&, const Grid&, const std::string&);
// Recover an integer-occupied seed from a density in the retained AO space.
// Idempotent restart densities preserve their occupied subspace exactly.
Eigen::MatrixXd orbital_seed(const Eigen::MatrixXd& X, const Eigen::MatrixXd& S,
    const Eigen::MatrixXd& density, int nocc, double occupation);
} // namespace vibeqc
