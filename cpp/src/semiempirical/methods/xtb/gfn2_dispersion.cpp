#include "vibeqc/semiempirical/methods/xtb/gfn2_dispersion.hpp"
#include "vibeqc/semiempirical/core/parameter_identity.hpp"
#include <cmath>
#include <stdexcept>
#include <utility>

namespace vibeqc::semiempirical::xtb {

GFN2D4Data::GFN2D4Data(Eigen::VectorXi numbers, Eigen::MatrixXd positions,
        Eigen::VectorXi owners, Eigen::VectorXd shifted_reference_charges,
        Eigen::VectorXd effective_charges, Eigen::VectorXd hardness,
        Eigen::VectorXd cn_weights, Eigen::MatrixXd kernel,
        double neutral_atm_energy, Eigen::VectorXd cn_weight_derivatives,
        Eigen::MatrixXd cn_jacobian, Eigen::MatrixXd radial_kernel,
        Eigen::MatrixXd neutral_atm_gradient)
    : numbers_(std::move(numbers)), owners_(std::move(owners)),
      positions_(std::move(positions)), kernel_(std::move(kernel)),
      cn_jacobian_(std::move(cn_jacobian)), radial_kernel_(std::move(radial_kernel)),
      atm_gradient_(std::move(neutral_atm_gradient)),
      qref_(std::move(shifted_reference_charges)), zeff_(std::move(effective_charges)),
      hardness_(std::move(hardness)), weights_(std::move(cn_weights)),
      dweights_(std::move(cn_weight_derivatives)), atm_energy_(neutral_atm_energy) {
    const auto n = numbers_.size(), nr = owners_.size();
    const auto valid_matrix = [](const Eigen::MatrixXd& m, auto rows, auto cols) {
        return m.rows() == rows && m.cols() == cols && m.allFinite();
    };
    if (n == 0 || nr == 0 || (numbers_.array() <= 0).any()
        || (owners_.array() < 0).any() || (owners_.array() >= n).any()
        || !valid_matrix(positions_, n, 3) || !valid_matrix(kernel_, nr, nr)
        || !valid_matrix(radial_kernel_, nr, nr)
        || !valid_matrix(cn_jacobian_, 3*n, n)
        || !valid_matrix(atm_gradient_, n, 3) || !std::isfinite(atm_energy_))
        throw std::invalid_argument("GFN2 D4 operands have invalid shapes or values");
    for (const auto* v : {&qref_, &zeff_, &hardness_, &weights_, &dweights_})
        if (v->size() != nr || !v->allFinite())
            throw std::invalid_argument("GFN2 D4 reference operands must be finite and aligned");
    if ((qref_.array() <= 0.).any() || (zeff_.array() <= 0.).any()
        || (hardness_.array() <= 0.).any() || (weights_.array() < 0.).any()
        || (kernel_.array() > 0.).any()
        || !(kernel_.array() == kernel_.transpose().array()).all()
        || !(radial_kernel_.array() == radial_kernel_.transpose().array()).all())
        throw std::invalid_argument("GFN2 D4 kernel must be symmetric with physical weights");
    for (Eigen::Index i = 0; i < nr; ++i)
        for (Eigen::Index j = 0; j < nr; ++j)
            if (owners_(i) == owners_(j)
                && (kernel_(i,j) != 0. || radial_kernel_(i,j) != 0.))
                throw std::invalid_argument("GFN2 D4 same-atom kernel blocks must be zero");
    for (Eigen::Index a = 0; a < n; ++a) {
        if (!(owners_.array() == a).any())
            throw std::invalid_argument("GFN2 D4 has an atom without reference operands");
        for (Eigen::Index b = 0; b < a; ++b)
            if ((positions_.row(a)-positions_.row(b)).norm() <= 1e-10)
                throw std::invalid_argument("GFN2 D4 requires distinct atom positions");
    }
    detail::CanonicalParameterHasher hash("xtb.gfn2.d4-operands.v1");
    hash.add_size(n); hash.add_size(nr);
    for (const auto& v : {numbers_, owners_})
        for (Eigen::Index i = 0; i < v.size(); ++i) hash.add_int(v(i));
    for (const auto* v : {&qref_, &zeff_, &hardness_, &weights_, &dweights_})
        for (Eigen::Index i = 0; i < v->size(); ++i) hash.add_double((*v)(i));
    for (const auto* matrix : {&positions_, &kernel_, &cn_jacobian_, &radial_kernel_, &atm_gradient_})
        for (Eigen::Index i = 0; i < matrix->rows(); ++i)
            for (Eigen::Index j = 0; j < matrix->cols(); ++j) hash.add_double((*matrix)(i,j));
    hash.add_double(atm_energy_);
    sha256_ = hash.finish();
}

void GFN2D4Data::require_geometry(const Molecule& mol) const {
    if (mol.atoms().size() != static_cast<std::size_t>(numbers_.size()))
        throw std::invalid_argument("GFN2 D4 operands do not match the molecular geometry");
    for (Eigen::Index a = 0; a < numbers_.size(); ++a) {
        if (mol.atoms()[a].Z != numbers_(a))
            throw std::invalid_argument("GFN2 D4 operands do not match the molecular elements");
        for (int k = 0; k < 3; ++k)
            if (mol.atoms()[a].xyz[k] != positions_(a,k))
                throw std::invalid_argument("GFN2 D4 operands do not match the molecular geometry");
    }
}

GFN2D4Evaluation GFN2D4Data::evaluate(const Eigen::VectorXd& charges,
                                    bool with_gradient) const {
    const auto n = numbers_.size(), nr = owners_.size();
    if (charges.size() != n || !charges.allFinite())
        throw std::invalid_argument("GFN2 D4 requires finite atom-sized charges");
    Eigen::VectorXd w(nr), dw(nr), wcn(nr);
    for (Eigen::Index i = 0; i < nr; ++i) {
        const double q = charges(owners_(i)) + zeff_(i);
        double z = std::exp(3.), dz = 0.;
        if (q > 0.) {
            const double inner = std::exp(hardness_(i)*(1.-qref_(i)/q));
            z = std::exp(3.*(1.-inner));
            dz = -3.*hardness_(i)*inner*z*qref_(i)/(q*q);
        }
        w(i) = weights_(i)*z; dw(i) = weights_(i)*dz; wcn(i) = dweights_(i)*z;
    }
    const Eigen::VectorXd field = kernel_*w;
    GFN2D4Evaluation result;
    result.energy = .5*w.dot(field)+atm_energy_;
    result.potential = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd vcn = Eigen::VectorXd::Zero(n);
    for (Eigen::Index i = 0; i < nr; ++i) {
        result.potential(owners_(i)) += dw(i)*field(i);
        vcn(owners_(i)) += wcn(i)*field(i);
    }
    if (with_gradient) {
        const Eigen::VectorXd flat = cn_jacobian_*vcn;
        result.gradient = atm_gradient_;
        for (Eigen::Index a = 0; a < n; ++a)
            for (int k = 0; k < 3; ++k) result.gradient(a,k) += flat(3*a+k);
        for (Eigen::Index i = 0; i < nr; ++i)
            for (Eigen::Index j = 0; j < i; ++j) {
                const int a = owners_(i), b = owners_(j);
                if (a == b) continue;
                const Eigen::RowVector3d delta = positions_.row(a)-positions_.row(b);
                const Eigen::RowVector3d g = w(i)*w(j)*radial_kernel_(i,j)*delta/delta.norm();
                result.gradient.row(a) += g; result.gradient.row(b) -= g;
            }
    }
    if (!std::isfinite(result.energy) || !result.potential.allFinite()
        || (with_gradient && !result.gradient.allFinite()))
        throw std::runtime_error("GFN2 D4 evaluation produced nonfinite operands");
    return result;
}
} // namespace vibeqc::semiempirical::xtb
