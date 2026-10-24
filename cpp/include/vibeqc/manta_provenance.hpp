#pragma once
#include <cstdint>
#include <string>
#include <memory>

namespace vibeqc::manta {
struct MolecularGradientSource;
struct MolecularEngineProvenance {
    // Native-only immutable fitting configuration. A serialized receipt is
    // evidence, not enough information to reconstruct arbitrary auxiliary bases.
    std::shared_ptr<const MolecularGradientSource> gradient_source;
    std::string method,provider_version,functional,xc_provider,xc_source_identity;
    std::string range_exchange_source_identity;
    std::uint64_t channel_batch=1,xc_grid_points=0,vv10_grid_points=0;
    std::uint64_t range_exchange_shell_calls=0,range_exchange_build_calls=0;
    bool exchange_active=true;
    double full_exchange_fraction=1.,range_exchange_fraction=0.,range_omega=0.;
    std::string orbital_basis_identity,molecule_identity;
    std::string requested_backend,one_electron_provider,coulomb_provider,exchange_provider;
    std::string coulomb_algorithm,exchange_algorithm;
    std::string confirmation_coulomb_algorithm,confirmation_coulomb_source_identity;
    std::string bupo_partition_identity,bupo_near_provider;
    std::uint64_t confirmation_coulomb_build_calls=0,bupo_build_calls=0,bupo_order=0;
    std::uint64_t bupo_near_shell_calls=0,bupo_far_blocks=0,bupo_moment_panels=0,bupo_moment_primitive_pairs=0;
    double bupo_integral_tolerance=0.,bupo_maximum_integral_error=0.,bupo_last_j_bound=0.;
    double bupo_setup_wall_seconds=0.;
    std::uint64_t bupo_moment_primitive_products=0,ri_bupo_metric_rank=0;
    std::string ri_bupo_forward_source_identity,ri_bupo_adjoint_source_identity,ri_bupo_metric_factor_identity;
    double ri_bupo_last_energy_bound=0.,ri_bupo_last_forward_bound=0.,ri_bupo_last_adjoint_bound=0.;
    double ri_bupo_last_metric_residual=0.,ri_bupo_metric_inverse_norm_bound=0.;
    std::string one_electron_source_identity,coulomb_source_identity,exchange_source_identity;
    std::string terminal_density_alpha_identity,terminal_density_beta_identity;
    std::string fallback_reason;
    std::uint64_t one_electron_shell_calls=0,jk_shell_calls=0,jk_build_calls=0;
    std::uint64_t primitive_integrals_evaluated=0,integral_modeled_peak_bytes=0;
    bool primitive_evaluations_measured=false,full_build_confirmed=false,stationarity_confirmed=false;
    double terminal_energy_delta=0.,terminal_fock_delta=0.,terminal_commutator_norm=0.;
    double one_electron_wall_seconds=0.,jk_wall_seconds=0.;
};
} // namespace vibeqc::manta
