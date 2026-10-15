#pragma once
#include "vibeqc/manta_ri.hpp"
#include "vibeqc/manta_bupo.hpp"
#include "vibeqc/manta_bupo_ri.hpp"
#include "vibeqc/rhf.hpp"
#include "vibeqc/uhf.hpp"
#include "vibeqc/rks.hpp"
#include "vibeqc/uks.hpp"

namespace vibeqc::manta {
struct MolecularHFEngine {
    std::string backend="manta",coulomb="exact",exchange="exact";
    ResourceBudget budget;
    RIOptions ri;
    BupoOptions bupo;
    BupoFittingOptions ri_bupo;
    Count channel_batch=1;
};
struct MolecularGradientSource {
    ResourceBudget budget;
    RIOptions ri;
    std::string coulomb,exchange;
    std::optional<BasisSet> auxiliary;
};
bool matches_molecular_engine_density(const Eigen::MatrixXd& alpha,const Eigen::MatrixXd& beta,const MolecularEngineProvenance&);
bool matches_molecular_engine_source(const Molecule&,const BasisSet&,const MolecularEngineProvenance&);
RHFResult run_rhf_with_engine(const Molecule&,const BasisSet&,const RHFOptions&,const MolecularHFEngine&,const BasisSet* auxiliary=nullptr);
UHFResult run_uhf_with_engine(const Molecule&,const BasisSet&,const UHFOptions&,const MolecularHFEngine&,const BasisSet* auxiliary=nullptr);
RKSResult run_rks_with_engine(const Molecule&,const BasisSet&,const RKSOptions&,const MolecularHFEngine&,const BasisSet* auxiliary=nullptr);
UKSResult run_uks_with_engine(const Molecule&,const BasisSet&,const UKSOptions&,const MolecularHFEngine&,const BasisSet* auxiliary=nullptr);
} // namespace vibeqc::manta
