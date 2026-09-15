// SCC charge-vector mixer hierarchy for semiempirical methods.
//
// The ab initio SCF accelerators (DIIS, EDIIS, ADIIS, KDIIS, etc.
// in diis.hpp / ediis.hpp / kdiis.hpp) operate on nbf x nbf Fock
// and density matrices with error vectors in the AO or MO basis.
// Semiempirical SCC iterates on a much smaller vector -- shell
// charges dq_shell (n_shells ~ 10-100) -- so we need vector-space
// mixers:
//
//   Simple       -- linear mixing: dq = alpha*dq_new + (1-alpha)*dq_old
//   DIIS         -- Pulay's DIIS in charge-vector space
//   Broyden      -- modified Broyden quasi-Newton (CP2K/tblite pattern)
//   BroydenEyert -- the tblite/xtb modified Broyden with inverse-norm
//                   history weighting (omega) and the charge-difference
//                   u-vector term (broyden.f90, Eyert scheme).  This is
//                   the mixer the xtb binary uses; it contracts metal /
//                   ionic supercell maps where the normalised-history
//                   Broyden and simple mixing orbit (IID 130 family).
//   Newton       -- full Newton step on the charge-map fixed point with a
//                   finite-difference Jacobian and a residual-reduction line
//                   search (reaches fixed points that linear mixing orbits,
//                   e.g. the 4-layer MgO(100) slab repeller, IID 141)

#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <deque>
#include <string>
#include <vector>

namespace vibeqc {
namespace semiempirical {

// ---------------------------------------------------------------------------
// Selector enum
// ---------------------------------------------------------------------------

enum class SCCMixer {
    Simple,       // linear mixing (default)
    DIIS,         // Pulay DIIS on charge residuals
    Broyden,      // modified Broyden quasi-Newton
    BroydenEyert, // tblite/xtb Eyert Broyden with omega weighting
    Newton,       // FD-Jacobian Newton with residual-reduction line search
};

inline std::string to_string(SCCMixer m) {
    switch (m) {
        case SCCMixer::Simple:       return "Simple";
        case SCCMixer::DIIS:         return "DIIS";
        case SCCMixer::Broyden:      return "Broyden";
        case SCCMixer::BroydenEyert: return "BroydenEyert";
        case SCCMixer::Newton:       return "Newton";
    }
    return "?";
}

// ---------------------------------------------------------------------------
// Base class
// ---------------------------------------------------------------------------

class ChargeMixer {
public:
    virtual ~ChargeMixer() = default;

    // Mix new charges into the current iteration.
    //   dq_out   = mixed charge vector (output)
    //   dq_in    = raw charges from this iteration
    //   iter     = 1-based SCC iteration count
    virtual void mix(Eigen::VectorXd& dq_out,
                     const Eigen::VectorXd& dq_in,
                     int iter) = 0;

    // Convergence diagnostic: current residual norm (optional).
    virtual double error() const { return 0.0; }

    // Reset internal state for a new SCC cycle.
    virtual void reset() {}

    // Name for logging.
    virtual std::string name() const = 0;
};

// ---------------------------------------------------------------------------
// Simple linear mixer
// ---------------------------------------------------------------------------

class SimpleMixer : public ChargeMixer {
public:
    explicit SimpleMixer(double alpha = 0.2) : alpha_(alpha) {}

    void mix(Eigen::VectorXd& dq_out,
             const Eigen::VectorXd& dq_in,
             int /*iter*/) override {
        if (dq_out.size() == 0) {
            dq_out = dq_in;
        } else {
            dq_out = alpha_ * dq_in + (1.0 - alpha_) * dq_out;
        }
    }

    void set_alpha(double a) { alpha_ = a; }
    double alpha() const { return alpha_; }

    std::string name() const override { return "Simple"; }

private:
    double alpha_;
};

// ---------------------------------------------------------------------------
// Pulay DIIS in charge-vector space
// ---------------------------------------------------------------------------

class ChargeDIISMixer : public ChargeMixer {
public:
    explicit ChargeDIISMixer(std::size_t max_subspace = 6,
                             double damping = 0.0,
                             int diis_start = 3)
        : max_subspace_(max_subspace),
          damping_(std::max(0.01, damping)),
          diis_start_(diis_start) {}

    void mix(Eigen::VectorXd& dq_out,
             const Eigen::VectorXd& dq_in,
             int iter) override;

    double error() const override { return last_error_; }

    void reset() override {
        q_history_.clear();
        e_history_.clear();
        last_error_ = 0.0;
    }

    std::string name() const override { return "DIIS"; }

private:
    std::size_t max_subspace_;
    double damping_;
    int diis_start_;
    std::deque<Eigen::VectorXd> q_history_;
    std::deque<Eigen::VectorXd> e_history_;
    double last_error_ = 0.0;
};

// ---------------------------------------------------------------------------
// Modified Broyden quasi-Newton mixer (CP2K/tblite style)
// ---------------------------------------------------------------------------

class BroydenMixer : public ChargeMixer {
public:
    // memory: max history vectors (typ. 4-8)
    // damping: initial mixing parameter (typ. 0.4-0.7)
    explicit BroydenMixer(int memory = 6, double damping = 0.4);

    void mix(Eigen::VectorXd& dq_out,
             const Eigen::VectorXd& dq_in,
             int iter) override;

    double error() const override { return error_; }

    void reset() override;

    std::string name() const override { return "Broyden"; }

private:
    int memory_;
    double damping_;
    int dim_ = 0;
    int step_ = 0;
    double error_ = 0.0;
    double w0_ = 0.01;

    // History storage
    Eigen::VectorXd q_last_;
    Eigen::VectorXd dq_last_;
    Eigen::MatrixXd U_;   // (dim, memory) difference vectors
    Eigen::MatrixXd V_;   // (dim, memory) residual differences
    Eigen::MatrixXd A_;   // (memory, memory) B matrix

    void allocate(int dim);
    void update_jacobian();
};

// ---------------------------------------------------------------------------
// Eyert modified Broyden mixer (tblite broyden.f90, the xtb mixer)
// ---------------------------------------------------------------------------

class EyertBroydenMixer : public ChargeMixer {
public:
    // memory: max history vectors. tblite defaults its Broyden memory
    // to the full iteration budget (memory = maxiter); pass the SCC
    // loop's max_iter for the xtb-faithful trajectory. A shorter
    // window is a sliding-history approximation.
    // alpha: mixing parameter (xtb bromix default 0.4)
    explicit EyertBroydenMixer(int memory = 10, double alpha = 0.4);

    void mix(Eigen::VectorXd& dq_out,
             const Eigen::VectorXd& dq_in,
             int iter) override;

    double error() const override { return error_; }

    void reset() override;

    std::string name() const override { return "BroydenEyert"; }

private:
    int memory_;
    double alpha_;
    int dim_ = 0;
    double error_ = 0.0;

    // tblite slot-wise history (circular, per-slot vectors).
    Eigen::VectorXd q_last_;     // q at the previous iteration
    Eigen::VectorXd dq_last_;    // residual dq at the previous iteration
    Eigen::MatrixXd df_;         // (dim, memory) normalized residual diffs
    Eigen::MatrixXd u_;          // (dim, memory) update vectors
    Eigen::VectorXd omega_;      // (memory) inverse-norm weights
};

}  // namespace semiempirical
}  // namespace vibeqc
