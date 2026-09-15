#include "vibeqc/semiempirical/kpoints_occupations.hpp"

#include <algorithm>
#include <cmath>
#include <ios>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace vibeqc {
namespace semiempirical {
namespace {

void compensated_add(double value, double& sum, double& compensation) {
    const double corrected = value - compensation;
    const double updated = sum + corrected;
    compensation = (updated - sum) - corrected;
    sum = updated;
}

double accumulation_tolerance(
    std::size_t term_count,
    double accumulated_value) {
    // Forward-error bound gamma_(n+1) for one rounded product per term
    // followed by sequential accumulation, with IEEE unit roundoff u.
    const double unit_roundoff =
        0.5 * std::numeric_limits<double>::epsilon();
    const double operations =
        (static_cast<double>(term_count) + 1.0) * unit_roundoff;
    if (operations >= 1.0) {
        return std::numeric_limits<double>::infinity();
    }
    const double gamma = operations / (1.0 - operations);
    return std::max(
        1.0e-12,
        gamma * std::max(1.0, std::abs(accumulated_value)));
}

std::vector<double> resolve_weights(
    std::size_t n_kpoints,
    const std::vector<double>& weights) {
    if (n_kpoints == 0) {
        throw std::invalid_argument(
            "k-point occupations: no k-points");
    }
    if (weights.empty()) {
        return std::vector<double>(
            n_kpoints, 1.0 / static_cast<double>(n_kpoints));
    }
    if (weights.size() != n_kpoints) {
        throw std::invalid_argument(
            "k-point occupations: weights must match k-point count");
    }
    double weight_sum = 0.0;
    double weight_compensation = 0.0;
    for (double weight : weights) {
        if (!std::isfinite(weight) || weight < 0.0) {
            throw std::invalid_argument(
                "k-point occupations: weights must be finite and nonnegative");
        }
        compensated_add(weight, weight_sum, weight_compensation);
    }
    const double weight_tolerance =
        256.0 * std::numeric_limits<double>::epsilon() *
        std::max(1.0, static_cast<double>(n_kpoints));
    if (std::abs(weight_sum - 1.0) > weight_tolerance) {
        throw std::invalid_argument(
            "k-point occupations: weights must sum to 1");
    }
    return weights;
}

double fermi_occupation(double energy, double chemical_potential, double temp) {
    const double argument = std::clamp(
        (energy - chemical_potential) / temp, -50.0, 50.0);
    return 2.0 / (1.0 + std::exp(argument));
}

struct WeightedBandState {
    double energy;
    std::size_t kpoint;
    Eigen::Index band;
    double weight;
};

KPointOccupationResult global_zero_temperature_occupations(
    const std::vector<Eigen::VectorXd>& eps_per_k,
    const std::vector<double>& weights,
    double target,
    double capacity,
    double capacity_tolerance,
    double min_resolvable_frontier_gap) {
    KPointOccupationResult result;
    result.occupations_per_k.reserve(eps_per_k.size());
    for (const auto& energies : eps_per_k) {
        result.occupations_per_k.push_back(
            Eigen::VectorXd::Zero(energies.size()));
    }

    std::vector<WeightedBandState> states;
    std::size_t state_count = 0;
    for (const auto& energies : eps_per_k) {
        state_count += static_cast<std::size_t>(energies.size());
    }
    states.reserve(state_count);
    for (std::size_t ik = 0; ik < eps_per_k.size(); ++ik) {
        if (weights[ik] == 0.0) {
            continue;
        }
        for (Eigen::Index band = 0; band < eps_per_k[ik].size(); ++band) {
            states.push_back(
                {eps_per_k[ik](band), ik, band, weights[ik]});
        }
    }
    if (states.empty()) {
        throw std::invalid_argument(
            "k-point occupations: at least one k-point must have weight");
    }
    std::sort(
        states.begin(), states.end(),
        [](const WeightedBandState& lhs, const WeightedBandState& rhs) {
            return lhs.energy < rhs.energy;
        });

    if (target <= capacity_tolerance) {
        result.fermi_level = states.front().energy;
        return result;
    }
    if (target >= capacity - capacity_tolerance) {
        for (auto& occupations : result.occupations_per_k) {
            occupations.setConstant(2.0);
        }
        result.fermi_level = states.back().energy;
        return result;
    }

    double energy_scale = 1.0;
    for (const auto& state : states) {
        energy_scale = std::max(energy_scale, std::abs(state.energy));
    }
    // This is a roundoff equality test, not a physical broadening. A finite
    // k-point band calculation has one weighted particle constraint, and an
    // energy-degenerate Fermi group shares one occupation in the T -> 0
    // ensemble limit (Weinert and Davenport, Phys. Rev. B 45, 13709, 1992,
    // DOI 10.1103/PhysRevB.45.13709).
    const double energy_tolerance =
        kZeroTemperatureDegeneracyUlps *
        std::numeric_limits<double>::epsilon() * energy_scale;
    // Bound on a degeneracy group's span (#544); see the header.
    const double max_group_span = kDegenerateGroupSpanFactor * energy_tolerance;
    const double count_tolerance = std::max(
        capacity_tolerance,
        accumulation_tolerance(states.size(), capacity));

    double filled = 0.0;
    double filled_compensation = 0.0;
    double boundary_occupation = 0.0;
    bool has_fractional_boundary = false;
    bool found_boundary = false;
    std::size_t group_begin = 0;
    while (group_begin < states.size()) {
        // Transitive grouping (#544): chain neighbours that are within
        // energy_tolerance of each other (the states are sorted, so the
        // difference is non-negative), bounded by max_group_span. The
        // pre-#544 rule compared to states[group_begin] -- a ball around
        // the first member -- and could split a roundoff-degenerate
        // manifold into two groups one ulp apart.
        std::size_t group_end = group_begin + 1;
        while (
            group_end < states.size() &&
            states[group_end].energy - states[group_end - 1].energy <=
                energy_tolerance &&
            states[group_end].energy - states[group_begin].energy <=
                max_group_span) {
            ++group_end;
        }

        double group_weight = 0.0;
        double group_weight_compensation = 0.0;
        double group_energy_sum = 0.0;
        double group_energy_compensation = 0.0;
        for (std::size_t index = group_begin; index < group_end; ++index) {
            compensated_add(
                states[index].weight,
                group_weight,
                group_weight_compensation);
            compensated_add(
                states[index].energy,
                group_energy_sum,
                group_energy_compensation);
        }
        const double group_capacity = 2.0 * group_weight;
        const double next_filled = filled + group_capacity;

        if (target > next_filled + count_tolerance) {
            for (std::size_t index = group_begin; index < group_end; ++index) {
                const auto& state = states[index];
                result.occupations_per_k[state.kpoint](state.band) = 2.0;
            }
            compensated_add(
                group_capacity, filled, filled_compensation);
            group_begin = group_end;
            continue;
        }

        if (target >= next_filled - count_tolerance) {
            for (std::size_t index = group_begin; index < group_end; ++index) {
                const auto& state = states[index];
                result.occupations_per_k[state.kpoint](state.band) = 2.0;
            }
            compensated_add(
                group_capacity, filled, filled_compensation);
            if (group_end < states.size()) {
                result.fermi_level = 0.5 * (
                    states[group_end - 1].energy + states[group_end].energy);
                // Issue #434: this is the branch that CUTS -- the boundary
                // falls between two distinct groups, so one state is filled
                // and the next is left empty. When those two are closer than
                // the fill can resolve, which of them is "lower" is decided
                // by last-ulp arithmetic that differs between hosts, and an
                // electron moves with it. Record it; the drivers refuse on
                // the accepted spectrum (see the header for why not here).
                //
                // The predicate is deliberately a bare "<=" against the
                // tolerance and does NOT exclude gaps below energy_tolerance.
                // Since #544 the grouping is transitive, so two adjacent
                // groups are more than energy_tolerance apart unless the
                // span bound max_group_span ended a long ulp-spaced chain;
                // that split is exactly the case this guard must still
                // catch, so no lower bound is applied here.
                result.frontier_gap =
                    states[group_end].energy - states[group_end - 1].energy;
                result.frontier_cut_unresolved =
                    result.frontier_gap <= min_resolvable_frontier_gap;
            } else {
                // Completely filled model space: no empty state above the
                // boundary, hence no frontier to cut and nothing to guard.
                result.fermi_level = states[group_end - 1].energy;
            }
        } else {
            boundary_occupation = std::clamp(
                (target - filled) / group_weight, 0.0, 2.0);
            has_fractional_boundary = true;
            for (std::size_t index = group_begin; index < group_end; ++index) {
                const auto& state = states[index];
                result.occupations_per_k[state.kpoint](state.band) =
                    boundary_occupation;
            }
            compensated_add(
                boundary_occupation * group_weight,
                filled,
                filled_compensation);
            result.fermi_level = group_energy_sum /
                                 static_cast<double>(group_end - group_begin);
        }
        found_boundary = true;
        break;
    }

    if (!found_boundary || std::abs(target - filled) > count_tolerance) {
        throw std::runtime_error(
            "k-point occupations: failed to satisfy weighted electron count");
    }

    // Zero-weight points do not constrain the particle count. Populate them
    // from the same global step function for deterministic result objects.
    for (std::size_t ik = 0; ik < eps_per_k.size(); ++ik) {
        if (weights[ik] != 0.0) {
            continue;
        }
        for (Eigen::Index band = 0; band < eps_per_k[ik].size(); ++band) {
            const double energy = eps_per_k[ik](band);
            if (energy < result.fermi_level - energy_tolerance) {
                result.occupations_per_k[ik](band) = 2.0;
            } else if (
                std::abs(energy - result.fermi_level) <= energy_tolerance) {
                result.occupations_per_k[ik](band) =
                    has_fractional_boundary ? boundary_occupation : 1.0;
            }
        }
    }
    return result;
}

}  // namespace

// Measured band edges + gaps from the occupations actually used (issue
// #426). The classification convention -- valence n > t, conduction
// n < 2 - t, fractional states in both classes, unclamped gaps, NaN only
// for a class empty in the model space -- is documented on
// KPointBandEdges in the header.
KPointBandEdges compute_kpoint_band_edges(
    const std::vector<Eigen::VectorXd>& eps_per_k,
    const std::vector<Eigen::VectorXd>& occupations_per_k,
    double occupation_tolerance) {
    if (!std::isfinite(occupation_tolerance) ||
        occupation_tolerance <= 0.0 || occupation_tolerance >= 1.0) {
        throw std::invalid_argument(
            "k-point band edges: occupation tolerance must be in (0, 1)");
    }
    if (eps_per_k.size() != occupations_per_k.size()) {
        throw std::invalid_argument(
            "k-point band edges: spectra and occupations must match "
            "k-point for k-point");
    }

    KPointBandEdges edges;
    edges.valence_max_per_k.reserve(eps_per_k.size());
    edges.conduction_min_per_k.reserve(eps_per_k.size());

    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double infinity = std::numeric_limits<double>::infinity();
    double valence_max = -infinity;
    double conduction_min = infinity;
    double direct_gap = infinity;
    // Occupancy-partition edges for gap_above_fermi_manifold: a
    // fractionally occupied state counts as occupied for the upper edge
    // and is NOT a candidate for the lower one.
    double partition_occupied_max = -infinity;
    double partition_empty_min = infinity;
    bool any_fractional = false;

    for (std::size_t ik = 0; ik < eps_per_k.size(); ++ik) {
        const auto& energies = eps_per_k[ik];
        const auto& occupations = occupations_per_k[ik];
        if (energies.size() != occupations.size()) {
            throw std::invalid_argument(
                "k-point band edges: spectra and occupations must match "
                "band for band");
        }
        double valence_max_k = -infinity;
        double conduction_min_k = infinity;
        for (Eigen::Index band = 0; band < energies.size(); ++band) {
            const double energy = energies(band);
            const double occupation = occupations(band);
            if (!std::isfinite(energy) || !std::isfinite(occupation)) {
                throw std::invalid_argument(
                    "k-point band edges: energies and occupations must "
                    "be finite");
            }
            if (occupation > occupation_tolerance) {
                valence_max_k = std::max(valence_max_k, energy);
                partition_occupied_max =
                    std::max(partition_occupied_max, energy);
            } else {
                partition_empty_min = std::min(partition_empty_min, energy);
            }
            if (occupation < 2.0 - occupation_tolerance) {
                conduction_min_k = std::min(conduction_min_k, energy);
            }
            if (occupation > occupation_tolerance &&
                occupation < 2.0 - occupation_tolerance) {
                any_fractional = true;
            }
        }
        const bool has_valence = std::isfinite(valence_max_k);
        const bool has_conduction = std::isfinite(conduction_min_k);
        edges.valence_max_per_k.push_back(has_valence ? valence_max_k : nan);
        edges.conduction_min_per_k.push_back(
            has_conduction ? conduction_min_k : nan);
        if (has_valence && valence_max_k > valence_max) {
            valence_max = valence_max_k;
            edges.valence_band_max_k = static_cast<int>(ik);
        }
        if (has_conduction && conduction_min_k < conduction_min) {
            conduction_min = conduction_min_k;
            edges.conduction_band_min_k = static_cast<int>(ik);
        }
        if (has_valence && has_conduction &&
            conduction_min_k - valence_max_k < direct_gap) {
            direct_gap = conduction_min_k - valence_max_k;
            edges.direct_gap_k = static_cast<int>(ik);
        }
    }

    if (edges.valence_band_max_k >= 0) {
        edges.valence_band_max = valence_max;
    }
    if (edges.conduction_band_min_k >= 0) {
        edges.conduction_band_min = conduction_min;
    }
    if (edges.valence_band_max_k >= 0 && edges.conduction_band_min_k >= 0) {
        edges.indirect_gap = conduction_min - valence_max;
    }
    if (edges.direct_gap_k >= 0) {
        edges.direct_gap = direct_gap;
    }
    // "Distance to the next entirely empty state" -- reported under its own
    // name so it can never be mistaken for the band gap (sec8-r4 F1).
    if (std::isfinite(partition_occupied_max) &&
        std::isfinite(partition_empty_min)) {
        edges.gap_above_fermi_manifold =
            partition_empty_min - partition_occupied_max;
    }
    // The structural gapless flag: E_F pinned inside a partially filled
    // manifold, or a measured band overlap. A screen must test this rather
    // than the sign of a float -- under a zero-temperature global aufbau
    // fill the pooled gap can never be negative, so a sign test alone is
    // vacuous on mesh rows.
    edges.is_metallic =
        any_fractional ||
        (std::isfinite(edges.indirect_gap) && edges.indirect_gap <= 0.0);
    return edges;
}

void reject_unresolved_frontier_cut(
    const KPointOccupationResult& occupation,
    const KPointOccupationOptions& options,
    const char* who) {
    if (!occupation.frontier_cut_unresolved) {
        return;
    }
    // CODATA 2018 Hartree -> eV, quoted so the message is actionable to a
    // reader who thinks in eV rather than Ha.
    constexpr double kHartreeToElectronVolt = 27.211386245988;
    std::ostringstream message;
    message.setf(std::ios::scientific);
    message.precision(3);
    message
        << who
        << ": the zero-temperature Aufbau fill cut a frontier it cannot "
           "resolve. The highest occupied and lowest empty states are "
           "separated by only "
        << occupation.frontier_gap << " Ha ("
        << occupation.frontier_gap * kHartreeToElectronVolt
        << " eV), at or below min_resolvable_frontier_gap = "
        << options.min_resolvable_frontier_gap
        << " Ha. At that separation which of the two states carries the "
           "electrons is decided by last-ulp arithmetic, so the occupation "
           "-- and every quantity built from it -- can differ between hosts "
           "and between builds while still reporting convergence. Supply a "
           "smearing temperature (occupation_options.smearing_temperature, "
           "e.g. 0.005 Ha) so the frontier is shared rather than cut. "
           "Setting min_resolvable_frontier_gap = 0 disables this check and "
           "restores the unstable zero-temperature fill.";
    throw std::runtime_error(message.str());
}

void validate_kpoint_occupation_options(
    const KPointOccupationOptions& options,
    const char* caller) {
    if (!std::isfinite(options.smearing_temperature) ||
        options.smearing_temperature < 0.0) {
        throw std::invalid_argument(
            std::string(caller) +
            ": smearing_temperature must be finite and >= 0");
    }
    if (!std::isfinite(options.min_resolvable_frontier_gap) ||
        options.min_resolvable_frontier_gap < 0.0) {
        throw std::invalid_argument(
            std::string(caller) +
            ": min_resolvable_frontier_gap must be finite and >= 0 "
            "(0 disables the issue-434 unresolvable-frontier guard)");
    }
}

Eigen::VectorXd gamma_degenerate_frontier_occupations(
    const Eigen::VectorXd& eps,
    int n_occ,
    double energy_tolerance,
    double max_occupation) {
    if (n_occ < 0 || n_occ > eps.size()) {
        throw std::invalid_argument(
            "gamma frontier occupations: n_occ out of range");
    }
    if (!std::isfinite(max_occupation) || max_occupation <= 0.0) {
        throw std::invalid_argument(
            "gamma frontier occupations: max_occupation must be finite and > 0");
    }
    if (n_occ == 0) {
        return Eigen::VectorXd::Zero(eps.size());
    }
    if (n_occ == eps.size()) {
        return Eigen::VectorXd::Constant(eps.size(), max_occupation);
    }

    // Roundoff equality scale, matched to the global multi-k zero-temperature
    // ensemble in global_zero_temperature_occupations. This is a degeneracy
    // test, not a physical broadening.
    double tolerance = energy_tolerance;
    if (tolerance < 0.0) {
        double energy_scale = 1.0;
        for (Eigen::Index band = 0; band < eps.size(); ++band) {
            energy_scale = std::max(energy_scale, std::abs(eps(band)));
        }
        tolerance = kZeroTemperatureDegeneracyUlps *
                    std::numeric_limits<double>::epsilon() * energy_scale;
    }
    // Bound on the manifold's span (#544); see the header.
    const double max_span = kDegenerateGroupSpanFactor * tolerance;

    const Eigen::Index frontier = n_occ - 1;
    const Eigen::Index lumo = n_occ;

    Eigen::VectorXd occupations = Eigen::VectorXd::Zero(eps.size());

    // An open gap above the frontier makes hard Aufbau unique: no manifold
    // is cut, so the exact historical vector is returned unchanged.
    if (eps(lumo) - eps(frontier) > tolerance) {
        occupations.head(n_occ).setConstant(max_occupation);
        return occupations;
    }

    // The frontier cuts a roundoff-degenerate manifold. Occupy the whole
    // manifold with equal fractional occupations (Weinert & Davenport T -> 0
    // ensemble), preserving the electron count exactly. Equal occupation
    // makes the density an invariant of the eigenspace, so density,
    // energy-weighted density, gradient, and stress are all unique.
    // Transitive extension (#544): a neighbour belongs to the manifold when
    // it lies within `tolerance` of the member next to it, not of the
    // frontier state, so a roundoff-degenerate manifold is never chopped
    // between two indistinguishable members. The manifold's total span is
    // bounded by max_span. (eps is sorted ascending.)
    Eigen::Index lo = frontier;
    while (lo > 0 && eps(lo) - eps(lo - 1) <= tolerance &&
           eps(lumo) - eps(lo - 1) <= max_span) {
        --lo;
    }
    Eigen::Index hi = lumo + 1;
    while (hi < eps.size() && eps(hi) - eps(hi - 1) <= tolerance &&
           eps(hi) - eps(lo) <= max_span) {
        ++hi;
    }

    occupations.head(lo).setConstant(max_occupation);
    const double manifold_occupation = max_occupation
        * static_cast<double>(n_occ - lo) / static_cast<double>(hi - lo);
    occupations.segment(lo, hi - lo).setConstant(manifold_occupation);
    return occupations;
}

KPointOccupationResult compute_closed_shell_kpoint_occupations(
    const std::vector<Eigen::VectorXd>& eps_per_k,
    const std::vector<double>& weights,
    double n_electrons_per_cell,
    int n_occ_each,
    const KPointOccupationOptions& options) {
    validate_kpoint_occupation_options(options, "k-point occupations");
    const auto resolved_weights = resolve_weights(eps_per_k.size(), weights);
    if (!std::isfinite(n_electrons_per_cell)) {
        throw std::invalid_argument(
            "k-point occupations: electron count must be finite");
    }
    if (n_occ_each < 0) {
        throw std::invalid_argument(
            "k-point occupations: n_occ_each must be >= 0");
    }

    double minimum_energy = std::numeric_limits<double>::infinity();
    double maximum_energy = -std::numeric_limits<double>::infinity();
    double capacity = 0.0;
    double capacity_compensation = 0.0;
    for (std::size_t ik = 0; ik < eps_per_k.size(); ++ik) {
        const auto& energies = eps_per_k[ik];
        if (energies.size() == 0 || n_occ_each > energies.size()) {
            throw std::invalid_argument(
                "k-point occupations: invalid per-k band count");
        }
        if (!energies.allFinite()) {
            throw std::invalid_argument(
                "k-point occupations: band energies must be finite");
        }
        minimum_energy = std::min(minimum_energy, energies.minCoeff());
        maximum_energy = std::max(maximum_energy, energies.maxCoeff());
        compensated_add(
            resolved_weights[ik] * 2.0 * energies.size(),
            capacity,
            capacity_compensation);
    }
    const double target = n_electrons_per_cell;
    const double capacity_tolerance = accumulation_tolerance(
        eps_per_k.size(), capacity);
    if (target < -1.0e-12 || target > capacity + capacity_tolerance) {
        throw std::invalid_argument(
            "k-point occupations: electron count is outside band capacity");
    }

    KPointOccupationResult result;
    result.occupations_per_k.reserve(eps_per_k.size());
    const double temperature = options.smearing_temperature;
    if (temperature == 0.0) {
        if (eps_per_k.size() > 1) {
            return global_zero_temperature_occupations(
                eps_per_k,
                resolved_weights,
                target,
                capacity,
                capacity_tolerance,
                options.min_resolvable_frontier_gap);
        }

        // Preserve the historical one-spectrum hard-Aufbau path exactly for
        // Gamma-point, molecular, and SECCM callers.
        double homo = -std::numeric_limits<double>::infinity();
        double lumo = std::numeric_limits<double>::infinity();
        for (const auto& energies : eps_per_k) {
            Eigen::VectorXd occupations =
                Eigen::VectorXd::Zero(energies.size());
            occupations.head(n_occ_each).setConstant(2.0);
            result.occupations_per_k.push_back(std::move(occupations));
            if (n_occ_each > 0) {
                homo = std::max(homo, energies(n_occ_each - 1));
            }
            if (n_occ_each < energies.size()) {
                lumo = std::min(lumo, energies(n_occ_each));
            }
        }
        if (std::isfinite(homo + lumo)) {
            result.fermi_level = 0.5 * (homo + lumo);
        }
        return result;
    }

    if (target <= 1.0e-12) {
        for (const auto& energies : eps_per_k) {
            result.occupations_per_k.push_back(
                Eigen::VectorXd::Zero(energies.size()));
        }
        result.fermi_level = minimum_energy;
        return result;
    }
    if (target >= capacity - capacity_tolerance) {
        for (const auto& energies : eps_per_k) {
            result.occupations_per_k.push_back(
                Eigen::VectorXd::Constant(energies.size(), 2.0));
        }
        result.fermi_level = maximum_energy;
        return result;
    }

    auto particle_count = [&](double chemical_potential) {
        double count = 0.0;
        for (std::size_t ik = 0; ik < eps_per_k.size(); ++ik) {
            for (int band = 0; band < eps_per_k[ik].size(); ++band) {
                count += resolved_weights[ik] * fermi_occupation(
                    eps_per_k[ik](band), chemical_potential, temperature);
            }
        }
        return count;
    };

    double lower = minimum_energy - 10.0 * temperature - 1.0;
    double upper = maximum_energy + 10.0 * temperature + 1.0;
    double expansion = std::max({100.0, 100.0 * temperature, 1.0});
    bool bracketed = false;
    for (int iteration = 0; iteration < 20; ++iteration) {
        if (particle_count(lower) <= target &&
            target <= particle_count(upper)) {
            bracketed = true;
            break;
        }
        lower -= expansion;
        upper += expansion;
        expansion *= 2.0;
    }
    if (!bracketed) {
        throw std::runtime_error(
            "k-point occupations: failed to bracket electron count");
    }

    for (int iteration = 0; iteration < 200; ++iteration) {
        const double middle = 0.5 * (lower + upper);
        if (particle_count(middle) > target) {
            upper = middle;
        } else {
            lower = middle;
        }
        if (upper - lower < 1.0e-14) {
            break;
        }
    }
    result.fermi_level = 0.5 * (lower + upper);

    for (std::size_t ik = 0; ik < eps_per_k.size(); ++ik) {
        Eigen::VectorXd occupations(eps_per_k[ik].size());
        for (int band = 0; band < eps_per_k[ik].size(); ++band) {
            occupations(band) = fermi_occupation(
                eps_per_k[ik](band), result.fermi_level, temperature);
            const double fraction = std::clamp(
                occupations(band) / 2.0, 1.0e-300, 1.0 - 1.0e-15);
            const double band_entropy = -2.0 * (
                fraction * std::log(fraction) +
                (1.0 - fraction) * std::log(1.0 - fraction));
            result.entropy += resolved_weights[ik] * band_entropy;
        }
        result.occupations_per_k.push_back(std::move(occupations));
    }
    return result;
}

}  // namespace semiempirical
}  // namespace vibeqc
