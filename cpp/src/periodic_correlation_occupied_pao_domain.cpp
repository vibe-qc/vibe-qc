#include "vibeqc/periodic_correlation_occupied_pao_domain.hpp"

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "vibeqc/kmesh_address.hpp"
#include "periodic_correlation_real_local_internal.hpp"

namespace vibeqc {
namespace {
namespace local = periodic_correlation_local_detail;
namespace arithmetic = periodic_correlation_real_local_detail;
using arithmetic::add;
using arithmetic::mul;
using arithmetic::finite;
using Complex = std::complex<double>;
using Sum = arithmetic::Sum;
using Digest = local::Digest;
using Options = PeriodicCorrelationOccupiedPAODomainOptions;
using Inventory = PeriodicCorrelationOccupiedPAODomainInventory;
using Caps = PeriodicCorrelationOccupiedPAODomainCaps;
using Plan = PeriodicCorrelationOccupiedPAODomainMemoryPlan;
using Diagnostics = PeriodicCorrelationOccupiedPAODomainDiagnostics;
using AtomCell = PeriodicCorrelationAtomCell;
constexpr double kTwoPi = 6.283185307179586476925286766559005768;
constexpr std::uint64_t kFixedControls = 65536;
static_assert(sizeof(Complex) == 16 && sizeof(AtomCell) == 16,
              "occupied PAO domain payload accounting requires complex128 and two uint64 labels");

void environment() {
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__) || FLT_EVAL_METHOD != 0
    throw std::invalid_argument("occupied PAO domains require strict binary64 arithmetic without fast/finite-only math");
#endif
    arithmetic::float_environment();
}

void options_valid(const Options& o) {
    const std::array<double, 8> controls{o.mulliken_population_cutoff, o.pao_tail_cutoff,
        o.normalization_tolerance, o.maximum_population_imaginary_magnitude,
        o.maximum_pao_imaginary_magnitude, o.maximum_negative_absolute_population,
        o.maximum_seed_omitted_absolute_population, o.maximum_expanded_omitted_absolute_population};
    for (const auto x : controls) if (!std::isfinite(x) || x < 0.0)
        throw std::invalid_argument("occupied PAO domain cuts and budgets require explicit finite nonnegative values");
    if (!(o.normalization_tolerance > 0.0 && o.normalization_tolerance < 1.0))
        throw std::invalid_argument("occupied PAO domain normalization tolerance must lie in (0,1)");
}

void metadata(const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationIAOOptimizerResult& opt, const PeriodicCorrelationWannier& w) {
    environment();
    local::validate_wannier(ref, w);
    const auto& m = opt.memory();
    if (opt.contract_version() != kPeriodicCorrelationIAOOptimizerContractVersion || !opt.converged()
        || opt.state_handle() != ref.state_handle()
        || opt.allocation_identity() != ref.dimensions().allocation_identity
        || m.n_points != w.n_cells() || m.n_basis != w.n_basis() || m.n_active != w.n_home_occupied()
        || opt.optimizer_identity_sha256().size() != 64 || w.wannier_identity_sha256().size() != 64
        || w.localization_identity_sha256() != opt.optimizer_identity_sha256()
        // These are different digest schemas (the optimizer also seals its
        // active-band indices). The payload equality is checked by hashing
        // the actual optimizer gauges in the Wannier schema after admission.
        || w.gauge_payload_sha256().size() != 64 || opt.gauge_payload_sha256().size() != 64
        || !w.options().require_time_reversal || !w.options().require_real_home_coefficients
        || !w.diagnostics().time_reversal_compatible || !w.diagnostics().real_home_coefficients_compatible)
        throw std::invalid_argument("occupied PAO domain requires the converged native optimizer and its own real/TR Wannier owner");
    (void) opt.state();
    (void) opt.gauges_data();
}

// Conservative scalar-loop reservations, not hardware FLOPs or runtime.
// Every mask/mapping candidate is counted, including rejected candidates.
std::uint64_t tail_work(const Plan& p, std::uint64_t seeds,
    std::uint64_t effective, std::uint64_t virtuals, std::uint64_t axis_sum) {
    if (!seeds) return 0;
    const auto kn = mul(p.n_cells, p.n_basis);
    const auto projector = mul(kn, add(effective, mul(virtuals, mul(2, p.n_basis))));
    const auto fourier = mul(kn, mul(p.n_basis, axis_sum));
    const auto candidates = mul(p.n_basis, mul(seeds, mul(p.n_cells, p.n_basis)));
    const auto score_passes = mul(p.n_atoms, mul(seeds, p.n_cells));
    return mul(64, add(add(projector, fourier), add(candidates,
        add(mul(3, score_passes), add(mul(p.n_atoms, p.n_basis), mul(kn, p.n_basis))))));
}

std::string mapping_digest(const std::uint64_t* map, std::uint64_t n, std::uint64_t atoms) {
    Digest digest("vibeqc.periodic.correlation.occupied-pao-domain.ao-atom-map");
    digest.u64(n); digest.u64(atoms);
    for (std::uint64_t mu = 0; mu < n; ++mu) {
        if (map[mu] >= atoms) throw std::out_of_range("occupied PAO domain AO-to-atom label is out of range");
        digest.u64(map[mu]);
    }
    // No atom-sized count allocation; even rejected mapping candidates were
    // admitted. An empty atom would make full mode/topology ambiguous.
    for (std::uint64_t a = 0; a < atoms; ++a) {
        bool found = false;
        for (std::uint64_t mu = 0; mu < n; ++mu) found |= map[mu] == a;
        if (!found) throw std::invalid_argument("occupied PAO domain atom has no mapped AO");
    }
    return digest.finish();
}

void source_payloads(const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationIAOOptimizerResult& opt, const PeriodicCorrelationWannier& w) {
    metadata(ref, opt, w);
    const auto& state = ref.state();
    const auto n = w.n_basis(), active = w.n_home_occupied();
    local::validate_gauges(w, opt.gauges_data(), static_cast<std::size_t>(w.memory().gauge_element_count));
    Digest optimizer_digest("vibeqc.periodic.correlation.iao-optimizer.gauges");
    optimizer_digest.u64(w.n_cells()); optimizer_digest.u64(active);
    for (std::uint64_t i = 0; i < w.memory().gauge_element_count; ++i)
        optimizer_digest.complex(opt.gauges_data()[i]);
    for (std::uint64_t k = 0; k < w.n_cells(); ++k)
        for (std::uint64_t i = 0; i < active; ++i) optimizer_digest.u64(opt.active_band(k, i));
    if (optimizer_digest.finish() != opt.gauge_payload_sha256())
        throw std::invalid_argument("occupied PAO domain optimizer payload differs from its native receipt");
    Digest digest("vibeqc.periodic.correlation.wannier.coefficients");
    for (auto division : state.mesh()) digest.u32(division);
    digest.u64(n); digest.u64(active);
    for (std::uint64_t k = 0; k < w.n_cells(); ++k) {
        const auto& mask = state.correlated_occupied_mask(k);
        std::uint64_t row = 0;
        for (std::size_t b = 0; b < mask.size(); ++b) if (mask[b]) {
            if (row >= active || opt.active_band(k, row) != b)
                throw std::invalid_argument("occupied PAO domain optimizer active band order differs from the reference");
            ++row;
        }
        if (row != active) throw std::invalid_argument("occupied PAO domain active mask count differs");
        const auto* values = w.cell_coefficients(k);
        for (std::uint64_t j = 0; j < n * active; ++j) digest.complex(values[j]);
    }
    if (digest.finish() != w.coefficient_payload_sha256())
        throw std::invalid_argument("occupied PAO domain Wannier payload differs from its native receipt");
}

// Same native scalar convention/axis order as occupied_fock.cpp. The
// caller owns and admits the line scratch; no FFT allocations or phase table.
// Canonical conjugate characters and exact integer Nyquist parity keep even
// meshes correct; no fractional-coordinate round-trip or nearest-cell search.
Complex positive_character(std::uint64_t m, std::uint64_t r, std::uint64_t extent) {
    if (!m) return {1.0, 0.0};
    if (extent % 2 == 0 && m == extent / 2) return {r % 2 == 0 ? 1.0 : -1.0, 0.0};
    const auto canonical = std::min(m, extent - m);
    const auto residue = (canonical * r) % extent;  // RegularKMesh bounds each division.
    double turns = double(residue) / double(extent);
    if (turns > 0.5) turns -= 1.0;
    const Complex phase(std::cos(kTwoPi * turns), std::sin(kTwoPi * turns));
    return m == canonical ? phase : std::conj(phase);
}

void inverse_fourier(Complex* panel, Complex* line, const std::array<int, 3>& mesh,
    std::uint64_t kcount, std::uint64_t width) {
    std::uint64_t stride = 1;
    for (int axis = 2; axis >= 0; --axis) {
        const auto extent = std::uint64_t(mesh[axis]);
        const auto outer_count = kcount / (extent * stride);
        const double weight = 1.0 / double(extent);
        if (extent != 1) {
            for (std::uint64_t outer = 0; outer < outer_count; ++outer)
                for (std::uint64_t inner = 0; inner < stride; ++inner)
                    for (std::uint64_t col = 0; col < width; ++col) {
                        for (std::uint64_t m = 0; m < extent; ++m)
                            line[m] = panel[((outer * extent + m) * stride + inner) * width + col];
                        for (std::uint64_t r = 0; r < extent; ++r) {
                            Sum sum;
                            for (std::uint64_t m = 0; m < extent; ++m)
                                sum.include(positive_character(m, r, extent) * (line[m] * weight));
                            panel[((outer * extent + r) * stride + inner) * width + col] = sum.value();
                        }
                    }
        }
        stride *= extent;
    }
}

// Same retained-space projector contraction as pao_domain.cpp. In particular
// this is NOT I-Cocc*Cocc^H*S if the SCF discarded any overlap directions.
// Raw complex values are retained through Fourier and the tail modulus.
void projector_column(const PeriodicRestrictedMeanFieldState& state,
    std::uint64_t k, std::uint64_t column, Complex* out, Complex* correction) {
    const auto n = state.n_basis();
    std::fill_n(out, n, Complex{}); std::fill_n(correction, n, Complex{});
    const auto& c = state.coefficients(k);
    const auto& s = state.overlap(k);
    const auto& mask = state.virtual_mask(k);
    for (std::size_t b = 0; b < mask.size(); ++b) if (mask[b]) {
        Sum overlap;
        for (std::uint64_t nu = 0; nu < n; ++nu) overlap.include(std::conj(c(nu, b)) * s(nu, column));
        const auto value = overlap.value();
        for (std::uint64_t nu = 0; nu < n; ++nu)
            arithmetic::accumulate(c(nu, b) * value, out[nu], correction[nu]);
    }
    for (std::uint64_t nu = 0; nu < n; ++nu) out[nu] = finite(out[nu] + correction[nu]);
}

std::uint64_t difference_cell(std::uint64_t left, std::uint64_t right,
    const std::array<int, 3>& mesh) {
    // The same canonical last-axis-fast torus as RegularKMesh/topology.
    // No +/-minimum-image tie break: even-mesh involutions remain exact.
    std::array<std::uint64_t, 3> d{};
    for (int axis = 2; axis >= 0; --axis) {
        const auto size = std::uint64_t(mesh[axis]);
        const auto a = left % size, b = right % size;
        d[axis] = (a + size - b) % size;
        left /= size; right /= size;
    }
    return (d[0] * mesh[1] + d[1]) * mesh[2] + d[2];
}

void add_real(double term, double& sum, double& correction) {
    finite(term);
    const double next = finite(sum + term);
    correction = finite(correction + (std::abs(sum) >= std::abs(term)
        ? (sum - next) + term : (term - next) + sum));
    sum = next;
}

void gate(double measured, double maximum, const char* message) {
    if (finite(measured) > maximum) throw std::invalid_argument(message);
}

std::uint64_t population_bytes(const Plan& p) {
    return add(add(p.retained_population_bytes, p.selection_mark_bytes),
        add(p.coefficient_panel_bytes, p.shared_scratch_bytes));
}
std::uint64_t tail_bytes(const Plan& p, std::uint64_t seeds) {
    return add(population_bytes(p), add(mul(16, seeds), mul(16, mul(p.n_cells, seeds))));
}
std::uint64_t output_bytes(const Plan& p, std::uint64_t seeds, std::uint64_t expanded) {
    return add(add(p.retained_population_bytes, p.selection_mark_bytes), mul(16, add(seeds, expanded)));
}
}  // namespace

PeriodicCorrelationOccupiedPAODomainMemoryPlan plan_periodic_correlation_occupied_pao_domain(
    const PeriodicCorrelationAdmittedReference& ref, const PeriodicCorrelationIAOOptimizerResult& opt,
    const PeriodicCorrelationWannier& w, std::uint64_t atoms, const Options& options,
    const Inventory& inventory, const Caps& caps) {
    options_valid(options); metadata(ref, opt, w);
    if (!atoms || atoms > w.n_basis())
        throw std::invalid_argument("occupied PAO domain requires 1 <= atom_count <= nao");
    if (!caps.maximum_atom_cells || !caps.maximum_seed_atom_cells || !caps.maximum_expanded_atom_cells
        || !caps.maximum_owned_numerical_bytes || !caps.maximum_control_storage_bytes
        || !caps.maximum_worker_bytes || !caps.maximum_work_units || !inventory.backend_allowance_bytes)
        throw std::invalid_argument("occupied PAO domain requires positive explicit caps and backend allowance");
    const RegularKMesh mesh(w.mesh(), {0, 0, 0});
    Plan p;
    p.n_cells = w.n_cells(); p.n_basis = w.n_basis(); p.n_atoms = atoms; p.n_active = w.n_home_occupied();
    if (mesh.size() != p.n_cells) throw std::invalid_argument("occupied PAO domain mesh extent differs");
    const auto kn = mul(p.n_cells, p.n_basis), ka = mul(p.n_cells, p.n_active);
    p.atom_cell_count = mul(p.n_cells, atoms);
    if (p.atom_cell_count > caps.maximum_atom_cells)
        throw std::length_error("occupied PAO domain atom-cell candidate cap exceeded");
    p.seed_count_upper = std::min(p.atom_cell_count, caps.maximum_seed_atom_cells);
    p.expanded_count_upper = std::min(p.atom_cell_count, caps.maximum_expanded_atom_cells);
    if (p.expanded_count_upper < p.seed_count_upper)
        throw std::invalid_argument("occupied PAO domain expanded cap must cover the seed cap");
    if (options.full_domain && (p.seed_count_upper != p.atom_cell_count || p.expanded_count_upper != p.atom_cell_count))
        throw std::length_error("occupied PAO domain full mode exceeds seed/expanded cap");
    p.borrowed_optimizer_numerical_bytes = add(mul(16, mul(ka, p.n_active)), mul(8, ka));
    p.borrowed_wannier_numerical_bytes = mul(16, mul(kn, p.n_active));
    if (p.borrowed_optimizer_numerical_bytes != opt.memory().output_numerical_bytes
        || p.borrowed_wannier_numerical_bytes != w.memory().retained_coefficient_bytes)
        throw std::invalid_argument("occupied PAO domain native owner payload census differs");
    p.caller_mapping_bytes = mul(8, p.n_basis);
    p.retained_population_bytes = mul(8, p.atom_cell_count);
    p.retained_domain_bytes_upper = mul(16, add(p.seed_count_upper, p.expanded_count_upper));
    p.coefficient_panel_bytes = mul(16, kn);
    const auto max_axis = std::uint64_t(*std::max_element(w.mesh().begin(), w.mesh().end()));
    p.shared_scratch_bytes = mul(16, std::max(mul(2, p.n_basis), max_axis));
    p.selection_mark_bytes = p.atom_cell_count;
    p.tail_score_bytes_upper = options.full_domain ? 0 : mul(16, mul(p.n_cells, p.seed_count_upper));
    // The compact seed list is constructed while the population panel is
    // still live, including in full mode (which has no tail phase).
    p.population_phase_owned_bytes = add(population_bytes(p), mul(16, p.seed_count_upper));
    p.tail_phase_owned_bytes_upper = options.full_domain ? 0 : tail_bytes(p, p.seed_count_upper);
    p.output_phase_owned_bytes_upper = output_bytes(p, p.seed_count_upper, p.expanded_count_upper);
    p.peak_owned_numerical_bytes = std::max({p.population_phase_owned_bytes,
        p.tail_phase_owned_bytes_upper, p.output_phase_owned_bytes_upper});
    p.fixed_control_storage_bytes = kFixedControls;
    // Logical owner objects and active identity characters; no numerical
    // source payload is counted twice. Container capacity is external margin.
    p.borrowed_owner_control_bytes = add(sizeof(PeriodicCorrelationIAOOptimizerResult),
        add(sizeof(PeriodicCorrelationWannier), 16 * 64));
    const auto controls = add(p.fixed_control_storage_bytes, p.borrowed_owner_control_bytes);
    if (add(controls, inventory.other_live_control_bytes) > caps.maximum_control_storage_bytes)
        throw std::length_error("occupied PAO domain control storage cap exceeded");
    const auto live = add(add(p.borrowed_optimizer_numerical_bytes, p.borrowed_wannier_numerical_bytes),
        add(p.caller_mapping_bytes, inventory.other_live_numerical_bytes));
    p.worker_bytes = add(add(p.peak_owned_numerical_bytes, live),
        add(controls, add(inventory.other_live_control_bytes, inventory.backend_allowance_bytes)));
    const auto& b = ref.budget(); const auto& d = ref.dimensions();
    p.required_node_memory_bytes = add(add(d.external_bytes, d.shared_bytes),
        add(mul(b.mpi_ranks, add(d.per_rank_bytes, d.localization_window_bytes_per_rank)),
            mul(mul(b.mpi_ranks, b.workers_per_rank), p.worker_bytes)));
    const auto effective = ref.state().n_effective_orbitals();
    const auto axis_sum = add(add(std::uint64_t(w.mesh()[0]), w.mesh()[1]), w.mesh()[2]);
    p.validation_work_units = mul(128, add(add(mul(p.n_cells, effective), mul(ka, p.n_active)),
        add(mul(kn, p.n_active), add(mul(atoms, p.n_basis), add(p.n_basis, p.atom_cell_count)))));
    p.population_work_units = mul(64, add(mul(kn, add(effective, add(p.n_active, p.n_basis))),
        add(mul(kn, axis_sum), mul(p.atom_cell_count, p.n_basis))));
    p.tail_work_units_upper = options.full_domain ? 0 : tail_work(p, p.seed_count_upper,
        effective, ref.state().n_virtual(), axis_sum);
    p.planned_work_units = add(p.validation_work_units, add(p.population_work_units, p.tail_work_units_upper));
    for (auto bytes : {p.peak_owned_numerical_bytes, p.borrowed_optimizer_numerical_bytes,
            p.borrowed_wannier_numerical_bytes, p.caller_mapping_bytes, p.tail_score_bytes_upper})
        arithmetic::extent(bytes);
    if (kn > std::vector<Complex>().max_size() || p.shared_scratch_bytes / 16 > std::vector<Complex>().max_size()
        || p.atom_cell_count > std::vector<double>().max_size()
        || p.atom_cell_count > std::vector<std::uint8_t>().max_size()
        || p.seed_count_upper > std::vector<AtomCell>().max_size()
        || p.expanded_count_upper > std::vector<AtomCell>().max_size()
        || p.tail_score_bytes_upper / 8 > std::vector<double>().max_size())
        throw std::length_error("occupied PAO domain payload exceeds native container extents");
    if (p.peak_owned_numerical_bytes > caps.maximum_owned_numerical_bytes)
        throw std::length_error("occupied PAO domain owned numerical byte cap exceeded");
    if (p.worker_bytes > caps.maximum_worker_bytes || !b.memory_limit_bytes
        || p.required_node_memory_bytes > b.memory_limit_bytes)
        throw std::length_error("occupied PAO domain worker/node memory cap exceeded");
    if (p.planned_work_units > caps.maximum_work_units)
        throw std::length_error("occupied PAO domain work cap exceeded before source scans");
    return p;
}

PeriodicCorrelationOccupiedPAODomain select_periodic_correlation_occupied_pao_domain(
    const PeriodicCorrelationAdmittedReference& ref, const PeriodicCorrelationIAOOptimizerResult& opt,
    const PeriodicCorrelationWannier& w, const std::uint64_t* map, std::size_t accessible,
    std::uint64_t atoms, std::uint64_t occupied, const Options& options,
    const Inventory& inventory, const Caps& caps) {
    const auto o = options; const auto live = inventory; const auto limits = caps;
    const auto p = plan_periodic_correlation_occupied_pao_domain(ref, opt, w, atoms, o, live, limits);
    if (occupied >= p.n_active) throw std::out_of_range("occupied PAO domain home occupied index is out of range");
    if (!map || accessible != p.n_basis || reinterpret_cast<std::uintptr_t>(map) % alignof(std::uint64_t))
        throw std::invalid_argument("occupied PAO domain requires aligned uint64 AO mapping of exactly nao elements");
    if (p.caller_mapping_bytes > std::numeric_limits<std::uintptr_t>::max() - reinterpret_cast<std::uintptr_t>(map))
        throw std::overflow_error("occupied PAO domain mapping pointer extent overflows");
    const auto map_identity = mapping_digest(map, p.n_basis, atoms);
    source_payloads(ref, opt, w);
    PeriodicCorrelationOccupiedPAODomain result;
    result.state_ = ref.state_handle(); result.occupied_ = occupied;
    result.memory_ = p; result.options_ = o;
    result.allocation_identity_ = ref.dimensions().allocation_identity;
    result.optimizer_identity_ = opt.optimizer_identity_sha256();
    result.wannier_identity_ = w.wannier_identity_sha256(); result.mapping_identity_ = map_identity;
    result.populations_.resize(p.atom_cell_count);
    std::vector<std::uint8_t> marked(p.atom_cell_count, 0);
    auto& diag = result.diagnostics_;
    diag.actual_peak_owned_numerical_bytes = population_bytes(p);
    diag.charged_work_units = add(p.validation_work_units, p.population_work_units);
    std::uint64_t seeds = 0, expanded = 0;
    const auto& state = ref.state();
    const auto n = p.n_basis, nk = p.n_cells;
    {
        std::vector<Complex> panel(p.coefficient_panel_bytes / 16);
        std::vector<Complex> scratch(p.shared_scratch_bytes / 16);
        for (std::uint64_t k = 0; k < nk; ++k) {
            local::fill_occupied_column(state, opt.gauges_data(), occupied, 0, k, scratch.data());
            const auto& s = state.overlap(k);
            for (std::uint64_t mu = 0; mu < n; ++mu) {
                Sum row;
                for (std::uint64_t nu = 0; nu < n; ++nu) row.include(s(mu, nu) * scratch[nu]);
                panel[k*n+mu] = row.value();
            }
        }
        inverse_fourier(panel.data(), scratch.data(), state.mesh(), nk, n);
        Sum total, negative, omitted;
        for (std::uint64_t cell = 0; cell < nk; ++cell) {
            const auto* beta = w.cell_coefficients(cell);
            for (std::uint64_t a = 0; a < atoms; ++a) {
                Sum population;
                for (std::uint64_t mu = 0; mu < n; ++mu) if (map[mu] == a)
                    population.include(std::conj(beta[mu*p.n_active+occupied]) * panel[cell*n+mu]);
                const auto value = population.value();
                diag.maximum_population_imaginary_magnitude = std::max(diag.maximum_population_imaginary_magnitude,
                    std::abs(value.imag()));
                gate(std::abs(value.imag()), o.maximum_population_imaginary_magnitude,
                    "occupied PAO domain atom population violates its imaginary budget");
                const double signed_population = value.real();
                const auto index = cell*atoms+a;
                result.populations_[index] = signed_population;
                total.include({signed_population, 0});
                if (signed_population < 0) negative.include({-signed_population, 0});
                if (o.full_domain || signed_population > o.mulliken_population_cutoff) {
                    marked[index] = 1; ++seeds;
                } else omitted.include({std::abs(signed_population), 0});
            }
        }
        diag.population_sum = total.value().real();
        diag.normalization_residual = std::abs(finite(diag.population_sum - 1.0));
        diag.negative_absolute_population = negative.value().real();
        diag.seed_omitted_absolute_population = omitted.value().real();
        gate(diag.normalization_residual, o.normalization_tolerance, "occupied PAO domain Mulliken normalization failed");
        gate(diag.negative_absolute_population, o.maximum_negative_absolute_population,
            "occupied PAO domain negative signed population exceeds its explicit budget");
        gate(diag.seed_omitted_absolute_population, o.maximum_seed_omitted_absolute_population,
            "occupied PAO domain seed omitted absolute population exceeds its explicit budget");
        if (seeds > p.seed_count_upper) throw std::length_error("occupied PAO domain actual seed cap exceeded");
        result.seeds_.resize(seeds);
        std::uint64_t out = 0;
        for (std::uint64_t index = 0; index < p.atom_cell_count; ++index) if (marked[index])
            result.seeds_[out++] = {index / atoms, index % atoms};
        // Seed storage is now live above the original population arrays.
        diag.actual_peak_owned_numerical_bytes = std::max(diag.actual_peak_owned_numerical_bytes,
            add(population_bytes(p), mul(16, seeds)));
        if (!o.full_domain && seeds) {
            const auto exact_tail_bytes = tail_bytes(p, seeds);
            if (exact_tail_bytes > p.peak_owned_numerical_bytes || exact_tail_bytes > limits.maximum_owned_numerical_bytes)
                throw std::length_error("occupied PAO domain actual tail phase exceeds admitted bytes");
            const auto axis_sum = add(add(std::uint64_t(state.mesh()[0]), state.mesh()[1]), state.mesh()[2]);
            const auto work = tail_work(p, seeds, state.n_effective_orbitals(), state.n_virtual(), axis_sum);
            diag.charged_work_units = add(diag.charged_work_units, work);
            if (diag.charged_work_units > limits.maximum_work_units)
                throw std::length_error("occupied PAO domain actual tail work exceeds its admission");
            diag.actual_peak_owned_numerical_bytes = std::max(diag.actual_peak_owned_numerical_bytes, exact_tail_bytes);
            const auto score_count = mul(nk, seeds);
            std::vector<double> scores(mul(2, score_count));
            for (std::uint64_t atom_b = 0; atom_b < atoms; ++atom_b) {
                std::fill(scores.begin(), scores.end(), 0.0);
                for (std::uint64_t mu = 0; mu < n; ++mu) if (map[mu] == atom_b) {
                    for (std::uint64_t k = 0; k < nk; ++k)
                        projector_column(state, k, mu, panel.data()+k*n, scratch.data());
                    inverse_fourier(panel.data(), scratch.data(), state.mesh(), nk, n);
                    ++diag.projected_pao_columns;
                    for (const auto value : panel) {
                        diag.maximum_pao_imaginary_magnitude = std::max(diag.maximum_pao_imaginary_magnitude,
                            std::abs(value.imag()));
                        gate(std::abs(value.imag()), o.maximum_pao_imaginary_magnitude,
                            "occupied PAO domain home PAO violates its imaginary budget");
                    }
                    for (std::uint64_t seed_index = 0; seed_index < seeds; ++seed_index) {
                        const auto seed = result.seeds_[seed_index];
                        for (std::uint64_t cell_b = 0; cell_b < nk; ++cell_b) {
                            const auto delta = difference_cell(seed.cell, cell_b, state.mesh());
                            const auto score = seed_index*nk+cell_b;
                            for (std::uint64_t nu = 0; nu < n; ++nu) if (map[nu] == seed.atom)
                                add_real(std::abs(panel[delta*n+nu]), scores[score], scores[score_count+score]);
                        }
                    }
                }
                for (std::uint64_t seed_index = 0; seed_index < seeds; ++seed_index)
                    for (std::uint64_t cell_b = 0; cell_b < nk; ++cell_b) {
                        const auto index = seed_index*nk+cell_b;
                        const double strength = finite(scores[index] + scores[score_count+index]);
                        if (strength < 0) throw std::overflow_error("occupied PAO tail strength became negative");
                        diag.maximum_pao_tail_strength = std::max(diag.maximum_pao_tail_strength, strength);
                        if (strength > o.pao_tail_cutoff) marked[cell_b*atoms+atom_b] = 1;
                    }
            }
        }
    }  // All coefficient, Fourier and score buffers released before output.
    Sum omitted;
    for (std::uint64_t index = 0; index < p.atom_cell_count; ++index) {
        if (marked[index]) ++expanded;
        else omitted.include({std::abs(result.populations_[index]), 0});
    }
    diag.expanded_omitted_absolute_population = omitted.value().real();
    gate(diag.expanded_omitted_absolute_population, o.maximum_expanded_omitted_absolute_population,
        "occupied PAO domain expanded omitted absolute population exceeds its explicit budget");
    if (expanded > p.expanded_count_upper) throw std::length_error("occupied PAO domain actual expanded cap exceeded");
    result.expanded_.resize(expanded);
    std::uint64_t out = 0;
    for (std::uint64_t index = 0; index < p.atom_cell_count; ++index) if (marked[index])
        result.expanded_[out++] = {index / atoms, index % atoms};
    diag.seed_count = seeds; diag.expanded_count = expanded;
    diag.retained_numerical_bytes = add(p.retained_population_bytes, mul(16, add(seeds, expanded)));
    diag.actual_peak_owned_numerical_bytes = std::max(diag.actual_peak_owned_numerical_bytes,
        output_bytes(p, seeds, expanded));
    if (diag.actual_peak_owned_numerical_bytes > p.peak_owned_numerical_bytes)
        throw std::logic_error("occupied PAO domain actual payload exceeded its conservative plan");
    source_payloads(ref, opt, w);
    if (map_identity != mapping_digest(map, n, atoms)
        || result.optimizer_identity_ != opt.optimizer_identity_sha256()
        || result.wannier_identity_ != w.wannier_identity_sha256())
        throw std::invalid_argument("occupied PAO domain source or mapping changed during construction");
    Digest payload("vibeqc.periodic.correlation.occupied-pao-domain.payload");
    payload.u64(p.atom_cell_count);
    for (const auto value : result.populations_) payload.real(value);
    payload.u64(seeds);
    for (const auto row : result.seeds_) { payload.u64(row.cell); payload.u64(row.atom); }
    payload.u64(expanded);
    for (const auto row : result.expanded_) { payload.u64(row.cell); payload.u64(row.atom); }
    result.payload_identity_ = payload.finish();
    Digest identity("vibeqc.periodic.correlation.occupied-pao-domain");
    identity.string(state.state_identity_sha256()); identity.string(state.calculation_identity());
    identity.string(result.allocation_identity_); identity.string(result.optimizer_identity_);
    identity.string(result.wannier_identity_); identity.string(result.mapping_identity_);
    identity.u64(occupied); identity.u32(o.full_domain);
    identity.real(o.mulliken_population_cutoff); identity.real(o.pao_tail_cutoff);
    identity.real(o.normalization_tolerance); identity.real(o.maximum_population_imaginary_magnitude);
    identity.real(o.maximum_pao_imaginary_magnitude); identity.real(o.maximum_negative_absolute_population);
    identity.real(o.maximum_seed_omitted_absolute_population); identity.real(o.maximum_expanded_omitted_absolute_population);
    identity.string("signed-gross-no-spin2;strict-cuts;one-step-original-row-to-PAO-column;retained-virtual-only;exact-Gamma-torus");
    identity.string(result.payload_identity_); result.identity_ = identity.finish();
    return result;
}

void PeriodicCorrelationOccupiedPAODomain::require_live() const {
    if (!state_ || populations_.size() != memory_.atom_cell_count
        || seeds_.size() != diagnostics_.seed_count || expanded_.size() != diagnostics_.expanded_count)
        throw std::logic_error("occupied PAO domain owner is consumed");
}
const double* PeriodicCorrelationOccupiedPAODomain::populations_data() const { require_live(); return populations_.data(); }
const PeriodicCorrelationAtomCell* PeriodicCorrelationOccupiedPAODomain::seeds_data() const { require_live(); return seeds_.data(); }
const PeriodicCorrelationAtomCell* PeriodicCorrelationOccupiedPAODomain::expanded_data() const { require_live(); return expanded_.data(); }
double PeriodicCorrelationOccupiedPAODomain::population(std::size_t cell, std::size_t atom) const {
    require_live();
    if (cell >= memory_.n_cells || atom >= memory_.n_atoms) throw std::out_of_range("occupied PAO domain population index is out of range");
    return populations_[cell*memory_.n_atoms+atom];
}
PeriodicCorrelationAtomCell PeriodicCorrelationOccupiedPAODomain::seed(std::size_t index) const {
    require_live();
    if (index >= seeds_.size()) throw std::out_of_range("occupied PAO domain seed index is out of range");
    return seeds_[index];
}
PeriodicCorrelationAtomCell PeriodicCorrelationOccupiedPAODomain::expanded(std::size_t index) const {
    require_live();
    if (index >= expanded_.size()) throw std::out_of_range("occupied PAO domain expanded index is out of range");
    return expanded_[index];
}
}  // namespace vibeqc
