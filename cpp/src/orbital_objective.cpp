#include "vibeqc/orbital_objective.hpp"
#include <unsupported/Eigen/MatrixFunctions>
#include <Eigen/Eigenvalues>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace vibeqc {
namespace {
Eigen::VectorXd flatten(const Eigen::MatrixXd& a) {
    return Eigen::Map<const Eigen::VectorXd>(a.data(), a.size());
}
Eigen::MatrixXd density(const Eigen::MatrixXd& c, int n) {
    return c.leftCols(n) * c.leftCols(n).transpose();
}
double dot(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
    return (a.array() * b.array()).sum();
}
void canonicalize(Eigen::MatrixXd& c, Eigen::VectorXd& eps,
                  const Eigen::MatrixXd& f, int nocc) {
    eps.resize(c.cols());
    for (auto span : {std::pair<int,int>{0, nocc}, {nocc, int(c.cols()) - nocc}}) {
        if (!span.second) continue;
        Eigen::MatrixXd block = c.middleCols(span.first, span.second);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(block.transpose() * f * block);
        if (eig.info() != Eigen::Success) throw std::runtime_error("Orbital block diagonalization failed");
        c.middleCols(span.first, span.second) = block * eig.eigenvectors();
        eps.segment(span.first, span.second) = eig.eigenvalues();
    }
}
}
Eigen::MatrixXd orbital_seed(const Eigen::MatrixXd& X, const Eigen::MatrixXd& S,
    const Eigen::MatrixXd& d, int nocc, double occupation) {
    if (nocc < 0 || nocc > X.cols() || d.rows() != S.rows() || d.cols() != S.cols()
        || !d.allFinite()) throw std::invalid_argument("Invalid OpenTrustRegion starting density");
    Eigen::MatrixXd p = X.transpose() * S * d * S * X / occupation;
    p = (0.5 * (p + p.transpose())).eval();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(p);
    if (eig.info() != Eigen::Success) throw std::runtime_error("Starting density diagonalization failed");
    return X * eig.eigenvectors().rowwise().reverse();
}
MolecularOrbitalObjective::MolecularOrbitalObjective(
    const Eigen::MatrixXd& S, const Eigen::MatrixXd& H, double nuclear,
    const JKBuilder& jk, const Eigen::MatrixXd& ca, const Eigen::MatrixXd& cb,
    int na, int nb, bool restricted, double exchange, OrbitalXCFunction xc,
    double energy_tol, double residual_tol, double rms_tol)
    : S_(S), H_(H), nuclear_(nuclear), exchange_(exchange), energy_tol_(energy_tol),
      residual_tol_(residual_tol), rms_tol_(rms_tol), jk_(jk), na_(na), nb_(nb),
      restricted_(restricted), xc_(std::move(xc)) {
    if (na < 0 || nb < 0 || na > ca.cols() || (!restricted && nb > cb.cols())
        || ca.rows() != S.rows() || (!restricted && cb.rows() != S.rows())
        || !ca.allFinite() || (!restricted && !cb.allFinite()))
        throw std::invalid_argument("Invalid OpenTrustRegion orbital dimensions");
    for (double x : {energy_tol_, residual_tol_, rms_tol_})
        if (!std::isfinite(x) || x <= 0) throw std::invalid_argument("SCF tolerances must be finite and positive");
    accepted_.ca = ca;
    accepted_.cb = restricted ? ca : cb;
}
int MolecularOrbitalObjective::size() const {
    return na_ * (accepted_.ca.cols() - na_)
        + (restricted_ ? 0 : nb_ * (accepted_.cb.cols() - nb_));
}
Eigen::MatrixXd MolecularOrbitalObjective::rotate(const Eigen::MatrixXd& c,
    const Eigen::VectorXd& v, int n, int offset) const {
    const int nv = c.cols() - n;
    Eigen::MatrixXd k = Eigen::MatrixXd::Zero(c.cols(), c.cols());
    if (nv * n) {
        k.bottomLeftCorner(nv, n) = Eigen::Map<const Eigen::MatrixXd>(v.data()+offset, nv, n);
        k.topRightCorner(n, nv) = -k.bottomLeftCorner(nv, n).transpose();
    }
    return c * k.exp();
}
MolecularOrbitalState MolecularOrbitalObjective::evaluate(
    const Eigen::MatrixXd& ca, const Eigen::MatrixXd& cb, bool response) {
    MolecularOrbitalState s;
    s.ca = ca; s.cb = restricted_ ? ca : cb;
    s.da = (restricted_ ? 2.0 : 1.0) * density(ca, na_);
    s.db = restricted_ ? Eigen::MatrixXd::Zero(S_.rows(), S_.cols()).eval() : density(cb, nb_);
    const Eigen::MatrixXd total = s.da + s.db;
    // Stateless entry points: never build_g_rhf/build_*_slot. In particular,
    // neither a trial nor a signed response density can pollute a delta-D cache.
    const Eigen::MatrixXd j = jk_.build_J(total);
    Eigen::MatrixXd ka = Eigen::MatrixXd::Zero(S_.rows(), S_.cols()), kb = ka;
    if (exchange_ != 0) {
        ka = jk_.build_K(s.da);
        if (!restricted_) kb = jk_.build_K(s.db);
    }
    s.fa = H_ + j - (restricted_ ? 0.5 : 1.0) * exchange_ * ka;
    s.fb = restricted_ ? s.fa : (H_ + j - exchange_ * kb).eval();
    s.core = dot(total, H_);
    s.coulomb = 0.5 * dot(total, j);
    s.exchange = -0.5 * exchange_ * ((restricted_ ? 0.5 : 1.0) * dot(s.da, ka) + dot(s.db, kb));
    if (xc_) {
        s.xc = xc_(s.da, s.db, response);
        s.fa += s.xc.alpha;
        if (!restricted_) s.fb += s.xc.beta;
        else s.fb = s.fa;
    }
    s.energy = nuclear_ + s.core + s.coulomb + s.exchange + s.xc.energy;
    s.residual = (s.fa * s.da * S_ - S_ * s.da * s.fa).norm();
    if (!restricted_)
        s.residual = std::max(s.residual, (s.fb * s.db * S_ - S_ * s.db * s.fb).norm());
    ++fock_evaluations;
    if (!std::isfinite(s.energy) || !s.fa.allFinite() || !s.fb.allFinite()
        || !std::isfinite(s.residual)) throw std::runtime_error("Nonfinite orbital energy/Fock/residual");
    return s;
}
OrbitalModel MolecularOrbitalObjective::model(const MolecularOrbitalState& s) const {
    OrbitalModel m;
    m.energy = s.energy;
    m.gradient.resize(size()); m.diagonal.resize(size());
    int offset = 0;
    auto spin = [&](const Eigen::MatrixXd& c, const Eigen::MatrixXd& f, int n, double factor) {
        const int nv = c.cols() - n;
        Eigen::MatrixXd fm = c.transpose() * f * c;
        m.gradient.segment(offset, n*nv) = factor * flatten(fm.bottomLeftCorner(nv, n));
        // Scaled Fock-gap diagonal approximation, ONLY for preconditioning.
        // No positive floor or orbital-energy approximation in hessian().
        for (int i=0; i<n; ++i) for (int a=0; a<nv; ++a)
            m.diagonal[offset+a+nv*i] = factor * (fm(n+a,n+a)-fm(i,i));
        offset += n*nv;
    };
    spin(s.ca, s.fa, na_, restricted_ ? 4 : 2);
    if (!restricted_) spin(s.cb, s.fb, nb_, 2);
    return m;
}
OrbitalModel MolecularOrbitalObjective::update(const Eigen::VectorXd& v) {
    if (v.size() != size() || !v.allFinite()) throw std::invalid_argument("Invalid orbital rotation");
    auto next = evaluate(rotate(accepted_.ca,v,na_,0),
        restricted_ ? accepted_.cb : rotate(accepted_.cb,v,nb_,na_*(accepted_.ca.cols()-na_)), true);
    auto m = model(next);
    if (!m.gradient.allFinite() || !m.diagonal.allFinite()) throw std::runtime_error("Nonfinite orbital derivatives");
    gradient_rms_ = size() ? m.gradient.norm()/std::sqrt(double(size())) : 0;
    const double de = trace_.empty() ? 0 : next.energy - accepted_.energy;
    const auto now = std::chrono::steady_clock::now();
    trace_.push_back({next.energy,de,next.residual,
        std::chrono::duration<double>(now-last_update_).count()});
    last_update_ = now;
    accepted_ = std::move(next); // Commit only after all model work succeeds.
    return m;
}
double MolecularOrbitalObjective::trial(const Eigen::VectorXd& v) {
    if (v.size() != size() || !v.allFinite()) throw std::invalid_argument("Invalid orbital trial");
    return evaluate(rotate(accepted_.ca,v,na_,0),
        restricted_ ? accepted_.cb : rotate(accepted_.cb,v,nb_,na_*(accepted_.ca.cols()-na_)), false).energy;
}
Eigen::VectorXd MolecularOrbitalObjective::hessian(const Eigen::VectorXd& v) {
    if (v.size() != size() || !v.allFinite() || trace_.empty()) throw std::invalid_argument("Invalid orbital response input");
    const auto& s = accepted_;
    const int va = s.ca.cols()-na_, vb = s.cb.cols()-nb_, off = va*na_;
    const Eigen::MatrixXd a = Eigen::Map<const Eigen::MatrixXd>(v.data(),va,na_);
    Eigen::MatrixXd b = Eigen::MatrixXd::Zero(vb,nb_);
    if (!restricted_ && vb*nb_) b = Eigen::Map<const Eigen::MatrixXd>(v.data()+off,vb,nb_);
    auto delta = [](const Eigen::MatrixXd& c, int n, const Eigen::MatrixXd& k) -> Eigen::MatrixXd {
        Eigen::MatrixXd t = c.rightCols(c.cols()-n)*k*c.leftCols(n).transpose();
        return t+t.transpose();
    };
    Eigen::MatrixXd da = (restricted_?2:1)*delta(s.ca,na_,a);
    Eigen::MatrixXd db = restricted_ ? Eigen::MatrixXd::Zero(S_.rows(),S_.cols()).eval() : delta(s.cb,nb_,b);
    const Eigen::MatrixXd j = jk_.build_J(da+db);
    Eigen::MatrixXd fa = j, fb = j;
    if (exchange_ != 0) {
        fa -= (restricted_?0.5:1)*exchange_*jk_.build_K(da);
        if (!restricted_) fb -= exchange_*jk_.build_K(db);
    }
    if (restricted_ && s.xc.restricted_kernel) fa += s.xc.restricted_kernel->apply(da);
    if (!restricted_ && s.xc.unrestricted_kernel) {
        const auto w = s.xc.unrestricted_kernel->apply(da,db);
        fa += w.alpha; fb += w.beta;
    }
    Eigen::VectorXd out(size());
    auto spin = [](const Eigen::MatrixXd& c, const Eigen::MatrixXd& f,
                   const Eigen::MatrixXd& df, const Eigen::MatrixXd& k, int n) -> Eigen::MatrixXd {
        const int nv = c.cols()-n;
        const Eigen::MatrixXd fm = c.transpose()*f*c;
        // Exact off-stationary local Hessian: full oo/vv Fock blocks.
        return fm.bottomRightCorner(nv,nv)*k - k*fm.topLeftCorner(n,n)
             + c.rightCols(nv).transpose()*df*c.leftCols(n);
    };
    out.head(off) = (restricted_?4:2)*flatten(spin(s.ca,s.fa,fa,a,na_));
    if (!restricted_) out.tail(vb*nb_) = 2*flatten(spin(s.cb,s.fb,fb,b,nb_));
    return out;
}
bool MolecularOrbitalObjective::energy_converged() const {
    return trace_.size() > 1 && std::abs(trace_.back().delta_e) < energy_tol_;
}
bool MolecularOrbitalObjective::gradient_converged() const {
    return !trace_.empty() && accepted_.residual < residual_tol_ && gradient_rms_ < rms_tol_;
}
bool MolecularOrbitalObjective::converged() const { return energy_converged() && gradient_converged(); }
MolecularOrbitalState MolecularOrbitalObjective::final_state() {
    // Fresh physical Fock and energy, retaining accepted occupied subspaces.
    auto final = evaluate(accepted_.ca, accepted_.cb, false);
    canonicalize(final.ca, final.epsa, final.fa, na_);
    if (restricted_) { final.cb = final.ca; final.epsb = final.epsa; }
    else canonicalize(final.cb, final.epsb, final.fb, nb_);
    return final;
}
} // namespace vibeqc
