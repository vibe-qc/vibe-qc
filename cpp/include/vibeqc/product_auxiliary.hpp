#pragma once
#include "vibeqc/basis.hpp"
#include <cstdint>
#include <memory>

namespace vibeqc {
// Shared basis metadata producer, independent of integral/Fock providers.
// Primitive product ranges and even tempering: Stoychev/Auer/Neese (2017),
// DOI 10.1021/acs.jctc.6b01041, Sec.2, Eqs.5-7. This is not AutoAux:
// no effective-exponent, valence/ECP or angular pruning heuristic is used.
struct ProductAuxiliaryOptions {
    double beta=2.;
    std::uint64_t refinement=0;
    int maximum_angular_momentum=8;
};
struct ProductAuxiliaryLimits {
    std::uint64_t memory_bytes=std::uint64_t{256}<<20,work_units=1000000000;
    std::uint64_t maximum_shells=4096,maximum_functions=32768;
};
struct ProductAuxiliaryPlan {
    std::uint64_t centers=0,channels=0,original_shells=0,original_functions=0;
    std::uint64_t maximum_generated_shells=0,maximum_generated_functions=0;
    std::uint64_t maximum_shells=0,maximum_functions=0,modeled_peak_bytes=0,work_units=0;
};
struct ProductAuxiliarySpan {
    std::uint64_t atom=0,coarse_intervals=0;
    int angular_momentum=0;
    double minimum_exponent=0.,maximum_product_exponent=0.,upper_grid_exponent=0.;
};
struct ProductAuxiliaryReceipt {
    std::string algorithm="atom-local-primitive-product-even-tempered-cartesian-v1";
    std::string qualification="candidate-basis-only;no-fitting-or-energy-accuracy-certificate";
    std::string citation_method="product_auxiliary_basis";
    std::string orbital_identity,seed_identity,basis_identity,identity;
    ProductAuxiliaryOptions options;
    std::uint64_t added_shells=0,added_functions=0,duplicate_seed_shells=0;
    std::vector<ProductAuxiliarySpan> spans;
};
ProductAuxiliaryPlan plan_product_auxiliary_basis(const Molecule&,const BasisSet& orbital,
    const BasisSet* seed,const ProductAuxiliaryOptions&,const ProductAuxiliaryLimits&);
class ProductAuxiliaryBasis {
public:
    ProductAuxiliaryBasis(const Molecule&,const BasisSet&,const BasisSet*,const ProductAuxiliaryOptions&,const ProductAuxiliaryLimits&);
    const BasisSet& basis() const { return *basis_; }
    bool matches_orbital(const BasisSet&) const;
    bool matches_seed(const BasisSet&) const;
    bool matches_basis(const BasisSet&) const;
    const ProductAuxiliaryPlan& plan() const { return plan_; }
    const ProductAuxiliaryReceipt& receipt() const { return receipt_; }
private:
    std::uint64_t orbital_shells_=0,orbital_functions_=0,orbital_primitives_=0,seed_primitives_=0;
    ProductAuxiliaryPlan plan_;
    ProductAuxiliaryReceipt receipt_;
    std::unique_ptr<BasisSet> basis_;
};
// Shared, immutable candidate metadata prepared before method integral values.
// Limits apply to the aggregate preparation, including all retained candidates.
struct ProductAuxiliarySequencePlan {
    std::uint64_t candidates=0,modeled_peak_bytes=0,work_units=0;
    std::uint64_t maximum_shells=0,maximum_functions=0;
};
ProductAuxiliarySequencePlan plan_product_auxiliary_sequence(const Molecule&,const BasisSet&,const BasisSet&,
    const ProductAuxiliaryOptions&,std::uint64_t refinements,const ProductAuxiliaryLimits&);
class ProductAuxiliarySequence {
public:
    ProductAuxiliarySequence(const Molecule&,const BasisSet&,const BasisSet&,
        const ProductAuxiliaryOptions&,std::uint64_t,const ProductAuxiliaryLimits&);
    const ProductAuxiliarySequencePlan& plan() const { return plan_; }
    const ProductAuxiliaryOptions& initial() const { return initial_; }
    std::uint64_t maximum_refinements() const { return candidates_.size()-1; }
    const ProductAuxiliaryBasis& candidate(std::uint64_t i) const { return *candidates_.at(i); }
    bool matches_sources(const BasisSet& orbital,const BasisSet& seed) const {
        return candidate(0).matches_orbital(orbital) && candidate(0).matches_seed(seed);
    }
private:
    ProductAuxiliarySequencePlan plan_;
    ProductAuxiliaryOptions initial_;
    std::vector<std::unique_ptr<ProductAuxiliaryBasis>> candidates_;
};
} // namespace vibeqc
