#include "vibeqc/periodic_correlation_real_local_provider.hpp"

#include <optional>

#include "periodic_correlation_real_local_provider_internal.hpp"

namespace vibeqc {
namespace {

namespace local = periodic_correlation_local_detail;
namespace real = periodic_correlation_real_local_detail;
using real::add;
using real::mul;
using Memory = PeriodicCorrelationRealLocalProviderMemoryPlan;
using Options = PeriodicCorrelationRealLocalProviderOptions;
using Caps = PeriodicCorrelationRealLocalProviderCaps;
using Panel = PeriodicCorrelationLocalOrbitalFactorPanel;
using Complex = std::complex<double>;

using real::density_index;
void validate_basis(const PeriodicCorrelationAdmittedReference& reference,
                    const PeriodicCorrelationRealLocalBasis& basis) {
    if (!basis.state_handle() || basis.state_handle().get() != reference.state_handle().get()
        || basis.contract_version() != kPeriodicCorrelationRealLocalBasisContractVersion
        || basis.allocation_identity() != reference.dimensions().allocation_identity)
        throw std::invalid_argument("real-local provider and basis require identical state owners and allocation");
    (void) basis.occupied_indices_data(); (void) basis.f_oo_data();
    if (basis.identity_sha256().size() != 64 || basis.local_basis_identity_sha256().size() != 64)
        throw std::logic_error("real-local provider basis lacks numerical certificate identities");
}
void validate_options(const Options& o) {
    real::tolerance_pair(o.reversal_absolute_tolerance, o.reversal_relative_tolerance);
    real::tolerance_pair(o.conjugacy_absolute_tolerance, o.conjugacy_relative_tolerance);
    real::tolerance_pair(o.self_q_absolute_tolerance, o.self_q_relative_tolerance);
    if (!std::isfinite(o.maximum_eri_projection_error) || o.maximum_eri_projection_error <= 0
        || !std::isfinite(o.maximum_scalar_roundoff_error) || o.maximum_scalar_roundoff_error <= 0)
        throw std::invalid_argument("real-local provider projection and scalar error budgets must be finite positive");
}
void option_wire(real::Digest& h, const Options& o) {
    h.real(o.reversal_absolute_tolerance); h.real(o.reversal_relative_tolerance);
    h.real(o.conjugacy_absolute_tolerance); h.real(o.conjugacy_relative_tolerance);
    h.real(o.self_q_absolute_tolerance); h.real(o.self_q_relative_tolerance);
    h.real(o.maximum_eri_projection_error); h.real(o.maximum_scalar_roundoff_error);
}
void admit(const PeriodicCorrelationAdmittedReference& reference, const Memory& m, const Caps& caps) {
    if (!caps.maximum_owned_numerical_bytes || m.peak_owned_numerical_bytes > caps.maximum_owned_numerical_bytes)
        throw std::length_error("real-local provider owned numerical byte cap is missing or exceeded");
    if (!caps.maximum_work_units || m.work_units > caps.maximum_work_units
        || !caps.maximum_scalar_work_units || m.scalar_work_units > caps.maximum_scalar_work_units)
        throw std::length_error("real-local provider construction or scalar work cap is missing or exceeded");
    if (!caps.maximum_factor_panels || m.factor_panels > caps.maximum_factor_panels
        || !caps.maximum_tile_visits || m.tile_visits > caps.maximum_tile_visits
        || !caps.maximum_reader_tile_bytes || m.maximum_reader_tile_bytes > caps.maximum_reader_tile_bytes)
        throw std::length_error("real-local provider factor panel or reader tile cap is missing or exceeded");
    if (!reference.budget().memory_limit_bytes || m.required_node_memory_bytes > reference.budget().memory_limit_bytes)
        throw std::length_error("real-local provider live inventory exceeds admitted node memory");
}

}  // namespace

Memory plan_periodic_correlation_real_local_provider(
    const PeriodicCorrelationAdmittedReference& reference, const PeriodicCorrelationFactorStreamSchedule& schedule,
    const PeriodicCorrelationPrivateFactorReader& reader, const PeriodicCorrelationWannier& wannier,
    const PeriodicCorrelationPAODomain& domain, const PeriodicCorrelationPAOSpace& space,
    const PeriodicCorrelationRealLocalBasis& basis) {
    real::float_environment(); validate_basis(reference, basis);
    const auto& shape = schedule.shape();
    const auto occupied = basis.memory().occupied_count;
    const auto selected = basis.virtual_selection();
    const auto count = std::min(shape.auxiliary_block, shape.n_auxiliary);
    const auto full = plan_periodic_correlation_local_orbital_factor_panel(
        reference, schedule, reader, wannier, domain, space, occupied, selected, 0, 0, count);
    const auto last_begin = mul(shape.auxiliary_tile_count-1, shape.auxiliary_block);
    const auto last = plan_periodic_correlation_local_orbital_factor_panel(reference, schedule, reader, wannier,
        domain, space, occupied, selected, 0, last_begin, shape.n_auxiliary-last_begin);
    Memory m;
    m.n_cells = shape.n_kpoints; m.n_auxiliary = shape.n_auxiliary;
    m.occupied_count = occupied; m.virtual_count = selected.count; m.orbital_count = full.orbital_count;
    const auto square = mul(m.orbital_count, m.orbital_count);
    m.density_count = mul(m.orbital_count, add(m.orbital_count, 1))/2;
    m.row_count = mul(m.n_cells, m.n_auxiliary);
    m.self_inverse_q_count = 1;
    for (auto dimension : reference.state().mesh())
        m.self_inverse_q_count = mul(m.self_inverse_q_count, dimension % 2 == 0 ? 2 : 1);
    m.retained_row_bytes = mul(8, mul(m.row_count, m.density_count));
    m.norm_workspace_bytes = mul(24, m.density_count);
    m.building_panel_peak_bytes = full.peak_owned_numerical_bytes;
    m.retained_partner_panel_bytes = m.n_cells == m.self_inverse_q_count ? 0
        : add(full.retained_output_bytes, full.retained_index_bytes);
    m.peak_owned_numerical_bytes = add(add(m.retained_row_bytes, m.norm_workspace_bytes),
        add(m.building_panel_peak_bytes, m.retained_partner_panel_bytes));
    m.live_basis_bytes = basis.memory().retained_output_bytes;
    m.basis_index_alias_bytes = full.caller_index_bytes;
    if (m.basis_index_alias_bytes != basis.memory().retained_index_bytes
        || basis.memory().orbital_count != m.orbital_count || basis.memory().n_cells != m.n_cells)
        throw std::logic_error("real-local provider basis inventory differs from its seal");
    m.caller_gauge_bytes = full.caller_gauge_bytes; m.live_wannier_bytes = full.live_wannier_bytes;
    m.live_domain_bytes = full.live_domain_bytes; m.live_space_bytes = full.live_space_bytes;
    m.live_reader_numeric_bytes = full.live_reader_numeric_bytes;
    m.live_reader_control_bytes = full.live_reader_control_bytes;
    m.maximum_reader_tile_bytes = full.maximum_reader_tile_bytes;
    m.factor_panels = mul(m.n_cells, shape.auxiliary_tile_count);
    m.tile_visits = mul(m.n_cells, add(mul(shape.auxiliary_tile_count-1, full.tile_visits), last.tile_visits));
    if (m.tile_visits != shape.tile_count) throw std::logic_error("real-local provider traversal is not one full store pass");
    const auto panel_work = mul(m.n_cells, add(mul(shape.auxiliary_tile_count-1, full.work_units), last.work_units));
    const auto audit_work = mul(512, add(mul(m.row_count, add(square, m.density_count)),
        add(m.caller_gauge_bytes/16, add(mul(occupied, occupied), m.factor_panels))));
    m.work_units = add(panel_work, audit_work);
    m.scalar_work_units = real::dot_interval_work(m.row_count);
    for (auto bytes : {m.retained_row_bytes, m.norm_workspace_bytes, m.peak_owned_numerical_bytes,
                       mul(512, m.factor_panels)}) real::extent(bytes);
    if (m.retained_row_bytes/8 > std::vector<double>().max_size()
        || m.norm_workspace_bytes/8 > std::vector<double>().max_size())
        throw std::length_error("real-local provider exceeds vector extent");
    // The basis indices ARE the panel's borrowed index view, not another copy.
    const auto live = add(m.peak_owned_numerical_bytes, add(m.live_basis_bytes, add(m.caller_gauge_bytes,
        add(m.live_wannier_bytes, add(m.live_domain_bytes, add(m.live_space_bytes,
        add(m.live_reader_numeric_bytes, m.live_reader_control_bytes)))))));
    const auto& dimensions = reference.dimensions(); const auto& budget = reference.budget();
    m.required_node_memory_bytes = add(add(dimensions.external_bytes, dimensions.shared_bytes),
        add(mul(budget.mpi_ranks, add(dimensions.per_rank_bytes, dimensions.localization_window_bytes_per_rank)),
            mul(mul(budget.mpi_ranks, budget.workers_per_rank), live)));
    return m;
}

const double* PeriodicCorrelationRealLocalProvider::rows_data() const {
    if (!state_ || rows_.size() != memory_.retained_row_bytes/8)
        throw std::logic_error("real-local provider owner is consumed");
    return rows_.data();
}
const std::string& PeriodicCorrelationRealLocalProvider::store_identity_sha256() const {
    if (source_kind_ != PeriodicCorrelationRealLocalProviderSourceKind::PrivateFactorStore)
        throw std::logic_error("Gaussian real-local provider has no factor store identity");
    return store_;
}
const std::string& PeriodicCorrelationRealLocalProvider::source_context_identity_sha256() const {
    if (!matched_finite_gaussian_hf_recipe())
        throw std::logic_error("real-local provider has no live Gaussian source context receipt");
    return source_context_;
}
const std::string& PeriodicCorrelationRealLocalProvider::hf_reference_source_identity_sha256() const {
    if (!matched_finite_gaussian_hf_recipe())
        throw std::logic_error("real-local provider has no live actual Gaussian HF receipt");
    return hf_source_;
}
double PeriodicCorrelationRealLocalProvider::row(std::size_t k, std::size_t left, std::size_t right) const {
    if (k >= memory_.row_count) throw std::out_of_range("real-local provider row is out of range");
    return rows_data()[k*memory_.density_count+density_index(left, right, memory_.orbital_count)];
}
PeriodicCorrelationRealLocalIntegral PeriodicCorrelationRealLocalProvider::integral(
    std::size_t p, std::size_t q, std::size_t r, std::size_t s) const {
    const auto* values = rows_data();
    const auto x = density_index(p,q,memory_.orbital_count), y = density_index(r,s,memory_.orbital_count);
    return real::dot_interval(values+x,rows_.size()-x,memory_.density_count,
        values+y,rows_.size()-y,memory_.density_count,memory_.row_count,
        memory_.scalar_work_units,options_.maximum_scalar_roundoff_error);
}
BoundedRestrictedCCSDIntegralProvider PeriodicCorrelationRealLocalProvider::integral_provider(
    const PeriodicCorrelationRealLocalBasis& basis) const {
    (void) rows_data(); (void) basis.occupied_indices_data();
    if (basis.state_handle().get() != state_.get() || basis.identity_sha256() != certificate_
        || basis.local_basis_identity_sha256() != basis_)
        throw std::invalid_argument("real-local CCSD provider requires its exact native basis certificate owner");
    BoundedRestrictedCCSDIntegralProvider out;
    out.context = const_cast<PeriodicCorrelationRealLocalProvider*>(this);
    out.value = [](std::uint64_t p, std::uint64_t q, std::uint64_t r, std::uint64_t s, void* context) {
        return static_cast<const PeriodicCorrelationRealLocalProvider*>(context)->integral(p,q,r,s).value;
    };
    out.retained_numerical_bytes = add(memory_.retained_row_bytes, basis.memory().retained_index_bytes);
    out.maximum_transient_numerical_bytes = 0;
    return out;
}

PeriodicCorrelationRealLocalProvider make_periodic_correlation_real_local_provider(
    const PeriodicCorrelationAdmittedReference& reference, const PeriodicCorrelationFactorStreamSchedule& schedule,
    const PeriodicCorrelationPrivateFactorReader& reader, const PeriodicCorrelationWannier& wannier,
    const Complex* gauges, std::size_t gauge_count, const PeriodicCorrelationPAODomain& domain,
    const PeriodicCorrelationPAOSpace& space, const PeriodicCorrelationRealLocalBasis& basis,
    const Options& options, const Caps& caps) {
    const auto memory = plan_periodic_correlation_real_local_provider(reference,schedule,reader,wannier,domain,space,basis);
    admit(reference,memory,caps); validate_options(options);
    const auto* labels = basis.occupied_indices_data();
    real::indices(labels,mul(2,memory.occupied_count),memory.occupied_count,reference.state());
    const auto gauge = local::validate_gauges(wannier,gauges,gauge_count);
    const auto selected = basis.virtual_selection();
    if (real::local_basis_identity(reference,wannier,gauge,domain,space,labels,memory.occupied_count,selected)
        != basis.local_basis_identity_sha256())
        throw std::invalid_argument("real-local provider selected orbital payload differs from its basis certificate");
    PeriodicCorrelationRealLocalProvider result;
    result.memory_ = memory; result.options_ = options; result.state_ = reference.state_handle();
    result.basis_ = basis.local_basis_identity_sha256(); result.certificate_ = basis.identity_sha256();
    result.store_ = reader.storage_identity_sha256();
    result.rows_.resize(static_cast<std::size_t>(memory.retained_row_bytes/8));
    real::Digest consumed("vibeqc.periodic.correlation.real-local-provider.consumed-panels");
    consumed.string(result.store_); consumed.string(result.basis_); consumed.u64(memory.factor_panels);
    auto& diagnostic = result.diagnostics_;
    const auto n = memory.orbital_count, aux = memory.n_auxiliary;
    const auto& shape = schedule.shape();
    {
        std::vector<double> norms(static_cast<std::size_t>(memory.norm_workspace_bytes/8));
        auto build = [&](std::uint64_t q, std::uint64_t begin, std::uint64_t count,
                         std::string& source, std::string& whitener) {
            const auto plan = plan_periodic_correlation_local_orbital_factor_panel(
                reference,schedule,reader,wannier,domain,space,memory.occupied_count,selected,q,begin,count);
            PeriodicCorrelationLocalFactorCaps panel_caps;
            panel_caps.maximum_owned_numerical_bytes = plan.peak_owned_numerical_bytes;
            panel_caps.maximum_work_units = plan.work_units; panel_caps.maximum_tile_visits = plan.tile_visits;
            panel_caps.maximum_reader_tile_bytes = plan.maximum_reader_tile_bytes;
            auto panel = build_periodic_correlation_local_orbital_factor_panel(reference,schedule,reader,wannier,
                gauges,gauge_count,domain,space,labels,mul(2,memory.occupied_count),memory.occupied_count,
                selected,q,begin,count,panel_caps);
            if (panel.state_handle().get() != reference.state_handle().get() || panel.local_basis_identity_sha256() != result.basis_
                || panel.store_identity_sha256() != result.store_ || panel.ao_basis_identity_sha256() != reader.ao_basis_identity_sha256()
                || panel.auxiliary_basis_identity_sha256() != reader.auxiliary_basis_identity_sha256()
                || panel.q_index() != q || panel.auxiliary_begin() != begin || panel.memory().auxiliary_count != count
                || panel.memory().orbital_count != n || panel.source_identity_sha256().size() != 64
                || panel.whitener_payload_identity_sha256().size() != 64)
                throw std::logic_error("real-local provider panel provenance or shape differs from its native source");
            if (source.empty()) { source = panel.source_identity_sha256(); whitener = panel.whitener_payload_identity_sha256(); }
            else if (source != panel.source_identity_sha256() || whitener != panel.whitener_payload_identity_sha256())
                throw std::invalid_argument("real-local provider q source or original-auxiliary whitener changed across slices");
            consumed.u64(q); consumed.u64(begin); consumed.u64(count); consumed.string(panel.identity_sha256());
            consumed.string(panel.source_identity_sha256()); consumed.string(panel.whitener_payload_identity_sha256());
            diagnostic.factor_panels_built = add(diagnostic.factor_panels_built,1);
            diagnostic.tile_visits = add(diagnostic.tile_visits,panel.memory().tile_visits);
            return panel;
        };
        for (std::uint64_t q = 0; q < memory.n_cells; ++q) {
            const auto partner_q = real::negative_cell(q,reference.state().mesh());
            if (q > partner_q) continue;
            std::string source, whitener, partner_source, partner_whitener;
            for (std::uint64_t tile = 0; tile < shape.auxiliary_tile_count; ++tile) {
                const auto begin = tile*shape.auxiliary_block, count = std::min(shape.auxiliary_block,aux-begin);
                auto panel = build(q,begin,count,source,whitener);
                std::optional<Panel> partner;
                if (q != partner_q) partner.emplace(build(partner_q,begin,count,partner_source,partner_whitener));
                real::convert_provider_panel_pair(panel,partner ? &*partner : static_cast<const Panel*>(nullptr),
                    q,partner_q,begin,count,memory,options,diagnostic,result.rows_.data(),norms.data());
            }
        }
        real::finish_provider_norms(memory,diagnostic,norms.data());
    }
    if (diagnostic.factor_panels_built != memory.factor_panels || diagnostic.tile_visits != memory.tile_visits)
        throw std::logic_error("real-local provider did not consume its admitted full store pass");
    real::finish_provider_projection(memory,options,diagnostic);
    result.consumed_ = consumed.finish();
    real::Digest payload("vibeqc.periodic.correlation.real-local-provider.rows");
    payload.u64(memory.row_count); payload.u64(memory.density_count);
    for (auto value : result.rows_) payload.real(value);
    result.payload_ = payload.finish();
    real::Digest identity("vibeqc.periodic.correlation.real-local-provider.identity");
    identity.string(reference.state().state_identity_sha256()); identity.string(reference.dimensions().allocation_identity);
    identity.string(result.basis_); identity.string(result.certificate_); identity.string(result.store_);
    identity.string(result.consumed_); identity.string(result.payload_); identity.string(real::kFloatPolicy);
    identity.string("finite-source-global-auxiliary-ri;hf-operator-match-and-image-tail-uncertified");
    option_wire(identity,options);
    identity.u64(diagnostic.factor_panels_built); identity.u64(diagnostic.tile_visits);
    for (auto value : {diagnostic.maximum_reversal_error,diagnostic.maximum_conjugacy_error,
        diagnostic.maximum_self_q_imaginary_magnitude,diagnostic.maximum_original_density_norm,
        diagnostic.maximum_reversal_norm,diagnostic.maximum_covariance_projection_norm,
        diagnostic.maximum_real_row_conversion_norm,diagnostic.orientation_error_bound,
        diagnostic.covariance_error_bound,diagnostic.conversion_error_bound,diagnostic.maximum_eri_projection_error_bound})
        identity.real(value);
    result.identity_ = identity.finish();
    return result;
}

}  // namespace vibeqc
