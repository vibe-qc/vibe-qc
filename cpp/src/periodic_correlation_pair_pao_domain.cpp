#include "vibeqc/periodic_correlation_pair_pao_domain.hpp"

#include <algorithm>
#include <array>
#include <cfloat>
#include <limits>
#include <stdexcept>
#include "periodic_correlation_real_local_internal.hpp"

namespace vibeqc {
namespace {
namespace local = periodic_correlation_local_detail;
namespace arithmetic = periodic_correlation_real_local_detail;
using arithmetic::add;
using arithmetic::mul;
using Digest = local::Digest;
using Domain = PeriodicCorrelationOccupiedPAODomain;
using Topology = PeriodicCorrelationTranslationPairTopology;
using Reference = PeriodicCorrelationAdmittedReference;
using Inventory = PeriodicCorrelationPairPAODomainInventory;
using Caps = PeriodicCorrelationPairPAODomainCaps;
using Plan = PeriodicCorrelationPairPAODomainMemoryPlan;
using AtomCell = PeriodicCorrelationAtomCell;
static_assert(sizeof(AtomCell) == 16, "pair PAO atom-cell storage requires two uint64 labels");

void environment() {
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__) || FLT_EVAL_METHOD != 0
    throw std::invalid_argument("pair PAO domain receipts require strict binary64 without fast/finite-only math");
#endif
    arithmetic::float_environment();
}

std::uint64_t translate(std::uint64_t cell, std::uint64_t shift, const std::array<int, 3>& mesh) {
    std::array<std::uint64_t, 3> sum{};
    for (int axis = 2; axis >= 0; --axis) {
        const auto n = std::uint64_t(mesh[axis]);
        sum[axis] = (cell % n + shift % n) % n;
        cell /= n; shift /= n;
    }
    return (sum[0]*mesh[1]+sum[1])*mesh[2]+sum[2];
}

void domain_metadata(const Reference& ref, const Domain& domain, std::uint64_t atoms) {
    const auto& m = domain.memory(); const auto& d = domain.diagnostics();
    if (domain.contract_version() != kPeriodicCorrelationOccupiedPAODomainVersion
        || domain.state_handle() != ref.state_handle()
        || domain.allocation_identity() != ref.dimensions().allocation_identity
        || m.n_cells != ref.state().n_kpoints() || m.n_basis != ref.state().n_basis()
        || m.n_active != ref.state().n_correlated_occupied() || m.n_atoms != atoms
        || m.atom_cell_count != mul(m.n_cells, atoms)
        || d.seed_count > m.atom_cell_count || d.expanded_count > m.atom_cell_count
        || d.retained_numerical_bytes != add(mul(8, m.atom_cell_count), mul(16, add(d.seed_count, d.expanded_count)))
        || domain.optimizer_identity_sha256().size() != 64 || domain.wannier_identity_sha256().size() != 64
        || domain.occupied_pao_domain_identity_sha256().size() != 64)
        throw std::invalid_argument("pair PAO domain endpoint state, allocation or native census differs");
    // Extent checks only, not a payload scan, including a consumed owner.
    (void) domain.populations_data(); (void) domain.seeds_data(); (void) domain.expanded_data();
}

void metadata(const Reference& ref, const Topology& topology, std::uint64_t row,
    const Domain& home, const Domain& partner, std::uint64_t atoms) {
    environment(); local::validate_reference(ref);
    if (!atoms || atoms > ref.state().n_basis())
        throw std::invalid_argument("pair PAO domain requires 1 <= atom_count <= nao");
    if (topology.contract_version() != kPeriodicCorrelationTranslationPairTopologyContractVersion
        || topology.state_handle() != ref.state_handle()
        || topology.allocation_identity() != ref.dimensions().allocation_identity
        || topology.calculation_identity() != ref.state().calculation_identity()
        || topology.state_identity_sha256() != ref.state().state_identity_sha256()
        || topology.mesh() != ref.state().mesh() || topology.is_shift() != ref.state().is_shift()
        || topology.n_cells() != ref.state().n_kpoints()
        || topology.n_home_occupied() != ref.state().n_correlated_occupied()
        || topology.row_count() != ref.dimensions().expected_pair_candidate_count
        || topology.topology_identity_sha256().size() != 64)
        throw std::invalid_argument("pair PAO domain topology state or allocation differs");
    const auto counts = estimate_periodic_correlation_translation_pair_counts(topology.mesh(), topology.n_home_occupied());
    if (counts.candidate_count != topology.row_count() || counts.cell_count != topology.n_cells()
        || counts.self_inverse_translation_count != topology.self_inverse_translation_count()
        || counts.placed_pair_count != topology.placed_pair_count())
        throw std::invalid_argument("pair PAO domain topology counts differ from the exact torus");
    if (row >= topology.row_count()) throw std::out_of_range("pair PAO domain topology row is out of range");
    domain_metadata(ref, home, atoms); domain_metadata(ref, partner, atoms);
    const auto& pair = topology.row(row);
    if (home.home_occupied_index() != pair.home_orbital || partner.home_occupied_index() != pair.partner_orbital
        || home.optimizer_identity_sha256() != partner.optimizer_identity_sha256()
        || home.wannier_identity_sha256() != partner.wannier_identity_sha256()
        || home.mapping_identity_sha256() != partner.mapping_identity_sha256())
        throw std::invalid_argument("pair PAO domain endpoint labels, global localization or AO mapping differ");
    if (pair.home_orbital == pair.partner_orbital
        && home.occupied_pao_domain_identity_sha256() != partner.occupied_pao_domain_identity_sha256())
        throw std::invalid_argument("same-orbital pair PAO endpoints require identical domain identities for translation covariance");
}

void domain_payload(const Domain& domain) {
    const auto& m = domain.memory(); const auto& d = domain.diagnostics();
    Digest digest("vibeqc.periodic.correlation.occupied-pao-domain.payload");
    digest.u64(m.atom_cell_count);
    const auto* populations = domain.populations_data();
    for (std::uint64_t i = 0; i < m.atom_cell_count; ++i) digest.real(populations[i]);
    const auto rows = [&](const AtomCell* data, std::uint64_t count) {
        digest.u64(count);
        std::uint64_t previous = 0;
        for (std::uint64_t i = 0; i < count; ++i) {
            const auto value = data[i];
            if (value.cell >= m.n_cells || value.atom >= m.n_atoms)
                throw std::out_of_range("pair PAO source atom-cell label is out of range");
            const auto linear = value.cell*m.n_atoms+value.atom;
            if (i && linear <= previous)
                throw std::invalid_argument("pair PAO source atom-cell rows are not sorted unique");
            previous = linear; digest.u64(value.cell); digest.u64(value.atom);
        }
    };
    rows(domain.seeds_data(), d.seed_count); rows(domain.expanded_data(), d.expanded_count);
    if (digest.finish() != domain.payload_identity_sha256())
        throw std::invalid_argument("pair PAO source domain payload differs from its native seal");
}

void topology_payload(const Reference& ref, const Topology& topology) {
    // Exact existing topology scalar wire, not a newly labelled row list.
    Digest digest("vibeqc.periodic.correlation.translation-pair-topology");
    digest.u32(ref.contract_version()); digest.u32(ref.dimensions().allocation_contract_version);
    digest.u32(ref.state().digest_version()); digest.string(ref.state().state_identity_sha256());
    digest.string(ref.dimensions().calculation_identity); digest.string(ref.dimensions().allocation_identity);
    for (auto value : topology.mesh()) digest.u32(value);
    for (auto value : topology.is_shift()) digest.u32(value);
    digest.u64(topology.n_cells()); digest.u64(topology.n_home_occupied());
    digest.u64(topology.self_inverse_translation_count()); digest.u64(topology.row_count());
    digest.u64(topology.placed_pair_count()); digest.string("unclassified");
    for (std::uint64_t i = 0; i < topology.row_count(); ++i) {
        const auto& row = topology.row(i);
        digest.u64(row.home_orbital); digest.u64(row.partner_orbital);
        digest.u64(row.translation_linear_index); digest.u64(row.placed_multiplicity);
    }
    if (digest.finish() != topology.topology_identity_sha256())
        throw std::invalid_argument("pair PAO topology payload differs from its native seal");
}

std::string map_payload(const std::uint64_t* map, std::uint64_t n, std::uint64_t atoms) {
    Digest digest("vibeqc.periodic.correlation.occupied-pao-domain.ao-atom-map");
    digest.u64(n); digest.u64(atoms);
    for (std::uint64_t i = 0; i < n; ++i) {
        if (map[i] >= atoms) throw std::out_of_range("pair PAO AO-to-atom label is out of range");
        digest.u64(map[i]);
    }
    for (std::uint64_t a = 0; a < atoms; ++a) {
        bool found = false;
        for (std::uint64_t i = 0; i < n; ++i) found |= map[i] == a;
        if (!found) throw std::invalid_argument("pair PAO mapping names an atom with no AOs");
    }
    return digest.finish();
}

void sources(const Reference& ref, const Topology& topology, std::uint64_t row,
    const Domain& home, const Domain& partner, const std::uint64_t* map, std::uint64_t atoms) {
    metadata(ref, topology, row, home, partner, atoms);
    topology_payload(ref, topology); domain_payload(home);
    if (&home != &partner) domain_payload(partner);
    const auto digest = map_payload(map, ref.state().n_basis(), atoms);
    if (digest != home.mapping_identity_sha256() || digest != partner.mapping_identity_sha256())
        throw std::invalid_argument("pair PAO mapping content differs from both sealed endpoint mappings");
}
}  // namespace

PeriodicCorrelationPairPAODomainMemoryPlan plan_periodic_correlation_pair_pao_domain(
    const Reference& ref, const Topology& topology, std::uint64_t row,
    const Domain& home, const Domain& partner, std::uint64_t atoms,
    const Inventory& inventory, const Caps& caps) {
    metadata(ref, topology, row, home, partner, atoms);
    if (!caps.maximum_atom_cells || !caps.maximum_union_atom_cells || !caps.maximum_ao_columns
        || !caps.maximum_topology_rows || !caps.maximum_owned_numerical_bytes
        || !caps.maximum_control_storage_bytes || !caps.maximum_worker_bytes || !caps.maximum_work_units
        || !inventory.backend_allowance_bytes)
        throw std::invalid_argument("pair PAO domain requires positive explicit caps and backend allowance");
    Plan p;
    p.n_cells = topology.n_cells(); p.n_basis = ref.state().n_basis(); p.n_atoms = atoms;
    p.atom_cell_count = mul(p.n_cells, atoms);
    if (p.atom_cell_count > caps.maximum_atom_cells || topology.row_count() > caps.maximum_topology_rows)
        throw std::length_error("pair PAO atom-cell/topology candidate cap exceeded");
    p.distinct_domain_owners = &home == &partner ? 1 : 2;
    p.borrowed_domain_numerical_bytes = home.diagnostics().retained_numerical_bytes;
    if (&home != &partner) p.borrowed_domain_numerical_bytes = add(p.borrowed_domain_numerical_bytes,
        partner.diagnostics().retained_numerical_bytes);
    p.borrowed_topology_row_bytes = mul(32, topology.row_count());
    p.caller_mapping_bytes = mul(8, p.n_basis);
    p.union_atom_count_upper = std::min({p.atom_cell_count, caps.maximum_union_atom_cells,
        add(home.diagnostics().expanded_count, partner.diagnostics().expanded_count)});
    p.ao_column_count_upper = std::min({mul(p.n_cells, p.n_basis),
        mul(p.n_basis, p.union_atom_count_upper), caps.maximum_ao_columns});
    p.mark_bytes = p.atom_cell_count;
    p.retained_atom_bytes_upper = mul(16, p.union_atom_count_upper);
    p.retained_ao_bytes_upper = mul(16, p.ao_column_count_upper);
    p.peak_owned_numerical_bytes = add(p.mark_bytes, add(p.retained_atom_bytes_upper, p.retained_ao_bytes_upper));
    p.fixed_control_storage_bytes = 65536;
    p.borrowed_owner_control_bytes = add(sizeof(Topology)+4*64,
        mul(p.distinct_domain_owners, sizeof(Domain)+7*64));
    const auto controls = add(add(p.fixed_control_storage_bytes, p.borrowed_owner_control_bytes),
        inventory.other_live_control_bytes);
    p.worker_bytes = add(add(p.peak_owned_numerical_bytes, p.borrowed_domain_numerical_bytes),
        add(p.borrowed_topology_row_bytes, add(p.caller_mapping_bytes,
            add(inventory.other_live_numerical_bytes, add(controls, inventory.backend_allowance_bytes)))));
    const auto& b = ref.budget(); const auto& d = ref.dimensions();
    p.required_node_memory_bytes = add(add(d.external_bytes, d.shared_bytes),
        add(mul(b.mpi_ranks, add(d.per_rank_bytes, d.localization_window_bytes_per_rank)),
            mul(mul(b.mpi_ranks, b.workers_per_rank), p.worker_bytes)));
    // Both source hash passes, original rejected mapping candidates and all
    // topology rows are charged before any scan. Fixed receipt/control work
    // and O(log row_count) native resolver fit the additive reservation.
    p.validation_work_units = mul(128, add(1024, add(topology.row_count(),
        add(p.borrowed_domain_numerical_bytes/8, add(p.n_basis, mul(p.n_basis, atoms))))));
    p.union_work_units = mul(32, add(add(home.diagnostics().expanded_count, partner.diagnostics().expanded_count),
        add(p.atom_cell_count, mul(3, mul(p.n_cells, p.n_basis)))));
    p.planned_work_units = add(p.validation_work_units, p.union_work_units);
    for (const auto bytes : {p.peak_owned_numerical_bytes, p.borrowed_domain_numerical_bytes,
             p.borrowed_topology_row_bytes, p.caller_mapping_bytes}) arithmetic::extent(bytes);
    if (p.atom_cell_count > std::vector<std::uint8_t>().max_size()
        || p.union_atom_count_upper > std::vector<AtomCell>().max_size()
        || mul(2, p.ao_column_count_upper) > std::vector<std::uint64_t>().max_size())
        throw std::length_error("pair PAO domain output exceeds native container extents");
    if (p.peak_owned_numerical_bytes > caps.maximum_owned_numerical_bytes)
        throw std::length_error("pair PAO domain owned numerical cap exceeded");
    if (controls > caps.maximum_control_storage_bytes)
        throw std::length_error("pair PAO domain total control storage cap exceeded");
    if (p.worker_bytes > caps.maximum_worker_bytes || !b.memory_limit_bytes
        || p.required_node_memory_bytes > b.memory_limit_bytes)
        throw std::length_error("pair PAO domain worker/node memory cap exceeded");
    if (p.planned_work_units > caps.maximum_work_units)
        throw std::length_error("pair PAO domain work cap exceeded before source scans");
    return p;
}

PeriodicCorrelationPairPAODomain make_periodic_correlation_pair_pao_domain(
    const Reference& ref, const Topology& topology, std::uint64_t row,
    const Domain& home, const Domain& partner, const std::uint64_t* map, std::size_t accessible,
    std::uint64_t atoms, const Inventory& inventory, const Caps& caps) {
    const auto live = inventory; const auto limits = caps;
    const auto p = plan_periodic_correlation_pair_pao_domain(ref, topology, row, home, partner, atoms, live, limits);
    if (!map || accessible != p.n_basis || reinterpret_cast<std::uintptr_t>(map) % alignof(std::uint64_t))
        throw std::invalid_argument("pair PAO domain requires aligned uint64 mapping with exactly nao elements");
    if (p.caller_mapping_bytes > std::numeric_limits<std::uintptr_t>::max() - reinterpret_cast<std::uintptr_t>(map))
        throw std::overflow_error("pair PAO mapping pointer extent overflows");
    sources(ref, topology, row, home, partner, map, atoms);
    const auto pair = topology.row(row);
    const auto resolution = resolve_periodic_correlation_placed_pair(topology,
        pair.home_orbital, 0, pair.partner_orbital, pair.translation_linear_index);
    if (resolution.row_index != row || resolution.common_translation_cell != 0 || resolution.transpose)
        throw std::invalid_argument("pair PAO domain canonical row differs from the native placed-pair resolver");
    PeriodicCorrelationPairPAODomain out;
    out.state_ = ref.state_handle(); out.memory_ = p; out.row_index_ = row; out.row_ = pair;
    out.energy_weight_ = periodic_correlation_translation_pair_energy_weight(topology, row);
    out.allocation_identity_ = ref.dimensions().allocation_identity;
    out.topology_identity_ = topology.topology_identity_sha256();
    out.home_identity_ = home.occupied_pao_domain_identity_sha256();
    out.partner_identity_ = partner.occupied_pao_domain_identity_sha256();
    out.mapping_identity_ = home.mapping_identity_sha256();
    std::vector<std::uint8_t> mark(p.atom_cell_count, 0);
    for (std::uint64_t i = 0; i < home.diagnostics().expanded_count; ++i) {
        const auto item = home.expanded(i); mark[item.cell*atoms+item.atom] = 1;
    }
    for (std::uint64_t i = 0; i < partner.diagnostics().expanded_count; ++i) {
        const auto item = partner.expanded(i);
        mark[translate(item.cell, pair.translation_linear_index, topology.mesh())*atoms+item.atom] = 1;
    }
    std::uint64_t union_count = 0, columns = 0;
    for (const auto flag : mark) if (flag) ++union_count;
    for (std::uint64_t cell = 0; cell < p.n_cells; ++cell)
        for (std::uint64_t mu = 0; mu < p.n_basis; ++mu) if (mark[cell*atoms+map[mu]]) ++columns;
    if (union_count > p.union_atom_count_upper || columns > p.ao_column_count_upper)
        throw std::length_error("pair PAO domain actual union atom/AO cap exceeded before output allocation");
    out.atoms_.resize(union_count); out.columns_.resize(mul(2, columns));
    std::uint64_t offset = 0;
    for (std::uint64_t i = 0; i < p.atom_cell_count; ++i) if (mark[i]) out.atoms_[offset++] = {i/atoms, i%atoms};
    offset = 0;
    for (std::uint64_t cell = 0; cell < p.n_cells; ++cell)
        for (std::uint64_t mu = 0; mu < p.n_basis; ++mu) if (mark[cell*atoms+map[mu]]) {
            out.columns_[2*offset] = cell; out.columns_[2*offset+1] = mu; ++offset;
        }
    out.diagnostics_.union_atom_count = union_count; out.diagnostics_.ao_column_count = columns;
    out.diagnostics_.retained_numerical_bytes = mul(16, add(union_count, columns));
    out.diagnostics_.actual_peak_owned_numerical_bytes = add(p.mark_bytes, out.diagnostics_.retained_numerical_bytes);
    out.diagnostics_.charged_work_units = p.planned_work_units;
    sources(ref, topology, row, home, partner, map, atoms);
    if (out.topology_identity_ != topology.topology_identity_sha256()
        || out.home_identity_ != home.occupied_pao_domain_identity_sha256()
        || out.partner_identity_ != partner.occupied_pao_domain_identity_sha256())
        throw std::invalid_argument("pair PAO source owner changed during construction");
    Digest row_digest("vibeqc.periodic.correlation.pair-pao-domain.row");
    row_digest.string(out.topology_identity_); row_digest.u64(row);
    row_digest.u64(pair.home_orbital); row_digest.u64(pair.partner_orbital);
    row_digest.u64(pair.translation_linear_index); row_digest.u64(pair.placed_multiplicity);
    row_digest.u64(out.energy_weight_); out.row_identity_ = row_digest.finish();
    Digest payload("vibeqc.periodic.correlation.pair-pao-domain.payload");
    payload.u64(union_count);
    for (const auto item : out.atoms_) { payload.u64(item.cell); payload.u64(item.atom); }
    payload.u64(columns);
    for (const auto label : out.columns_) payload.u64(label);
    out.payload_identity_ = payload.finish();
    Digest identity("vibeqc.periodic.correlation.pair-pao-domain");
    identity.string(ref.state().state_identity_sha256()); identity.string(ref.state().calculation_identity());
    identity.string(out.allocation_identity_); identity.string(out.row_identity_);
    identity.string(out.home_identity_); identity.string(out.partner_identity_); identity.string(out.mapping_identity_);
    identity.string("initial-pair-domain;home-union-translated-partner;all-parent-AOs;not-CCSD-extended-domain");
    identity.string(out.payload_identity_); out.identity_ = identity.finish();
    return out;
}

void PeriodicCorrelationPairPAODomain::require_live() const {
    if (!state_ || atoms_.size() != diagnostics_.union_atom_count
        || columns_.size() != 2*diagnostics_.ao_column_count)
        throw std::logic_error("pair PAO domain owner is consumed");
}
const AtomCell* PeriodicCorrelationPairPAODomain::atom_cells_data() const { require_live(); return atoms_.data(); }
const std::uint64_t* PeriodicCorrelationPairPAODomain::cell_ao_indices_data() const { require_live(); return columns_.data(); }
AtomCell PeriodicCorrelationPairPAODomain::atom(std::size_t index) const {
    require_live();
    if (index >= atoms_.size()) throw std::out_of_range("pair PAO atom index is out of range");
    return atoms_[index];
}
PeriodicPAODomainColumn PeriodicCorrelationPairPAODomain::column(std::size_t index) const {
    require_live();
    if (index >= diagnostics_.ao_column_count) throw std::out_of_range("pair PAO column index is out of range");
    return {columns_[2*index], columns_[2*index+1]};
}
PeriodicPAODomainColumn PeriodicCorrelationPairPAODomain::translated_column(std::size_t index, std::uint64_t cell) const {
    const auto value = column(index);
    if (cell >= memory_.n_cells) throw std::out_of_range("pair PAO common translation is not a canonical cell");
    return {translate(value.cell, cell, state_->mesh()), value.ao};
}
}  // namespace vibeqc
