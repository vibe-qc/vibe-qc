#pragma once

#include "vibeqc/manta.hpp"
#include "vibeqc/manta_md.hpp"

namespace vibeqc::manta {
// Exact exp(i q.R) for stored finite vectors, with no libm accuracy premise.
struct BlochPhasePlan {
    Count modeled_peak_bytes=0,scalar_terms=0,kernel_terms=0;
};
struct BlochPhaseCertificate {
    double real_lower=0.,real_upper=0.,imag_lower=0.,imag_upper=0.,error_bound=0.;
    Count squarings=0;
};
BlochPhasePlan plan_bloch_phase(const ResourceBudget&);
BlochPhaseCertificate certify_bloch_phase(const Eigen::Vector3d&,const Eigen::Vector3d&,
    std::complex<double> evaluated,const ResourceBudget&);
struct FourierPlan {
    std::vector<Count> shape;
    Count shell_calls=0, primitive_pairs=0, vector_primitive_products=0;
    Count kernel_terms=0, tile_elements=0, output_bytes=0, modeled_peak_bytes=0;
};
struct FourierReceipt {
    std::string provider="manta", provider_version="md-fp64-v1";
    std::string normalization="libint-shell-axial-v1", convention="exp(-i*q.r)";
    std::string source_identity;
    Count shell_calls=0, primitive_pairs=0, vector_primitive_products=0;
    Count integral_elements=0, modeled_peak_bytes=0;
};
struct FourierResult {
    std::vector<Count> shape;
    std::vector<std::complex<double>> values;
    FourierReceipt receipt;
};
// A null ket denotes a single auxiliary Gaussian, with no orbital product.
// Output order is (q,bra,ket), or (q,auxiliary) for the single-center form.
FourierPlan plan_fourier(const BasisSet& bra,const BasisSet* ket,Count vectors,const ResourceBudget&);
FourierResult compute_fourier_panel(const BasisSet& bra,const BasisSet* ket,
                                    const md::Matrix& vectors,const ResourceBudget&);
} // namespace vibeqc::manta
