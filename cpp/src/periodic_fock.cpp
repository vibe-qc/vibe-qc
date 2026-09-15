#include "vibeqc/periodic_fock.hpp"

#include <string>

#include "vibeqc/init.hpp"
#include "vibeqc/lattice_pair_cells.hpp"
#include "vibeqc/schwarz.hpp"
#include "vibeqc/thread_pool.hpp"

#include <libint2/engine.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vibeqc {

namespace {

// Local alias so existing code keeps using the short name.
inline std::vector<libint2::Shell> shift_shells(
    const libint2::BasisSet& shells, const Eigen::Vector3d& dr) {
    return shift_shells_to_cell(shells, dr);
}

// Hash an integer lattice index for unordered_map lookup.
struct LatticeIndexHash {
    std::size_t operator()(const Eigen::Vector3i& v) const noexcept {
        // FNV-1a-ish. Cells live in a small bounded box; any decent hash works.
        const std::uint64_t a = static_cast<std::uint64_t>(v[0] + 1024);
        const std::uint64_t b = static_cast<std::uint64_t>(v[1] + 1024);
        const std::uint64_t c = static_cast<std::uint64_t>(v[2] + 1024);
        return static_cast<std::size_t>((a * 2654435761ULL) ^
                                        (b * 40503ULL) ^ c);
    }
};
struct LatticeIndexEq {
    bool operator()(const Eigen::Vector3i& a,
                    const Eigen::Vector3i& b) const noexcept {
        return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
    }
};
using CellIndexMap =
    std::unordered_map<Eigen::Vector3i, int, LatticeIndexHash, LatticeIndexEq>;
using WeightLookup =
    std::unordered_map<Eigen::Vector3i, Eigen::MatrixXd, LatticeIndexHash, LatticeIndexEq>;

// A bounded worker-local cache of primitive shell pairs. Absolute shell
// positions are keys: translating a precomputed Gaussian product center
// would introduce different rounding from the direct integral path.
class PrimitivePairCache {
    struct Entry {
        std::size_t a = std::numeric_limits<std::size_t>::max();
        std::size_t b = std::numeric_limits<std::size_t>::max();
        libint2::ShellPair pair;
    };
public:
    PrimitivePairCache(std::size_t max_nprim, double precision,
                       libint2::ScreeningMethod screening)
        : ln_precision_(std::log(precision)), screening_(screening) {
        constexpr std::size_t budget = 4 * 1024 * 1024;
        const auto entry_bytes = sizeof(Entry) + max_nprim * max_nprim
            * sizeof(libint2::ShellPair::PrimPairData);
        entries_.resize(std::min<std::size_t>(4096, budget / entry_bytes));
        for (auto& entry : entries_)
            entry.pair.primpairs.reserve(max_nprim * max_nprim);
    }

    const libint2::ShellPair* get(std::size_t a, std::size_t b,
                                 const libint2::Shell& sa,
                                 const libint2::Shell& sb) {
        if (entries_.empty()) return nullptr;
        auto& entry = entries_[(a * 2654435761ULL ^ b * 40503ULL) % entries_.size()];
        if (entry.a != a || entry.b != b) {
            // Match Engine's canonical primitive iteration order. Reversing
            // only the labels lets compute2 undo its input-pair permutation
            // without changing primitive sums or Gaussian product rounding.
#if LIBINT2_SHELLQUARTET_SET == LIBINT2_SHELLQUARTET_SET_STANDARD
            const bool reverse = sa.contr[0].l < sb.contr[0].l;
#else
            const bool reverse = sa.contr[0].l > sb.contr[0].l;
#endif
            entry.pair.init(reverse ? sb : sa, reverse ? sa : sb,
                            ln_precision_, screening_);
            if (reverse) {
                for (auto& primitive : entry.pair.primpairs)
                    std::swap(primitive.p1, primitive.p2);
                for (auto& coordinate : entry.pair.AB) coordinate = -coordinate;
            }
            entry.a = a;
            entry.b = b;
        }
        return &entry.pair;
    }
private:
    double ln_precision_;
    libint2::ScreeningMethod screening_;
    std::vector<Entry> entries_;
};

CellIndexMap build_cell_index_map(const std::vector<LatticeCell>& cells) {
    CellIndexMap m;
    m.reserve(cells.size() * 2);
    for (std::size_t i = 0; i < cells.size(); ++i) {
        m[cells[i].index] = static_cast<int>(i);
    }
    return m;
}

// Dense displacement -> position lookup over an integer bounding box.
//
// The fused SR J/K traversal (build_jk_2e_real_space_impl) queries cell
// displacements ``h = c_a - c_b`` O(n_out x n_pad^2) times per Fock
// build; on the padded ket-image balls of the corrected-gauge BIPOLE
// route that unordered_map hash floor measurably dominates (registry
// BIPOLE-SR-PAD-OVERCONSERVATIVE / BIPOLE-EXACT-ZONE-UNBOUNDED,
// 2026-08-05). Every queried displacement is a difference of two cell
// indices from one list, so it lies in the Minkowski-difference box of
// that list — a small dense array (~15x the cell count in entries for a
// ball-shaped list) turns the lookup into two compares + one load. The
// stored contents are IDENTICAL to the map this replaces, so traversal
// decisions — and the emitted J/K bits — are unchanged.
class DenseCellTable {
public:
    DenseCellTable() = default;
    DenseCellTable(const Eigen::Vector3i& lo, const Eigen::Vector3i& hi)
        : lo_(lo.cast<long long>()),
          dims_(hi.cast<long long>() - lo_ + WideIndex::Ones()) {
        // Disconnected physical pair neighborhoods can have a huge empty
        // bounding box. Keep the dense fast path bounded and store only
        // present labels when that box would exceed its memory budget.
        constexpr std::size_t entry_budget = 16 * 1024 * 1024 / sizeof(int);
        std::size_t entries = 1;
        for (int axis = 0; axis < 3; ++axis) {
            if (dims_[axis] <= 0 ||
                static_cast<unsigned long long>(dims_[axis]) > entry_budget / entries) {
                sparse_ = true;
                return;
            }
            entries *= static_cast<std::size_t>(dims_[axis]);
        }
        pos_.assign(entries, -1);
    }

    // Entries outside the box are silently dropped: the box is built to
    // contain every reachable query, so such entries are unreachable —
    // exactly like a map entry that is never looked up.
    void insert(const Eigen::Vector3i& h, int value) {
        if (sparse_) {
            if (in_bounds(h)) sparse_pos_[h] = value;
            return;
        }
        const std::ptrdiff_t flat = flatten(h);
        if (flat >= 0) pos_[static_cast<std::size_t>(flat)] = value;
    }

    int at(const Eigen::Vector3i& h) const {
        if (sparse_) {
            const auto found = sparse_pos_.find(h);
            return found == sparse_pos_.end() ? -1 : found->second;
        }
        const std::ptrdiff_t flat = flatten(h);
        return (flat < 0) ? -1 : pos_[static_cast<std::size_t>(flat)];
    }

private:
    using WideIndex = Eigen::Matrix<long long, 3, 1>;
    bool in_bounds(const Eigen::Vector3i& h) const {
        const WideIndex position = h.cast<long long>() - lo_;
        return (position.array() >= 0).all() && (position.array() < dims_.array()).all();
    }
    std::ptrdiff_t flatten(const Eigen::Vector3i& h) const {
        const long long a = static_cast<long long>(h[0]) - lo_[0];
        const long long b = static_cast<long long>(h[1]) - lo_[1];
        const long long c = static_cast<long long>(h[2]) - lo_[2];
        if (a < 0 || b < 0 || c < 0 ||
            a >= dims_[0] || b >= dims_[1] || c >= dims_[2]) {
            return -1;
        }
        return (static_cast<std::ptrdiff_t>(a) * dims_[1] + b) * dims_[2] + c;
    }

    WideIndex lo_ = WideIndex::Zero();
    WideIndex dims_ = WideIndex::Zero();
    std::vector<int> pos_;
    bool sparse_ = false;
    CellIndexMap sparse_pos_;
};

// Componentwise index bounds of a cell list (empty list -> zero box).
void cell_index_bounds(const std::vector<LatticeCell>& cells,
                       Eigen::Vector3i& lo, Eigen::Vector3i& hi) {
    lo = Eigen::Vector3i::Zero();
    hi = Eigen::Vector3i::Zero();
    for (const auto& c : cells) {
        lo = lo.cwiseMin(c.index);
        hi = hi.cwiseMax(c.index);
    }
}

// A balanced Cartesian range index for the *supplied* summation domain.
// It never invents lattice points or assumes an orthogonal/3D lattice.
// Queries visit only boxes intersecting the erfc interaction sphere (#21).
class CellRangeIndex {
    struct Node {
        Eigen::Vector3d lo, hi;
        std::size_t begin, end;
        int left = -1, right = -1;
    };
public:
    explicit CellRangeIndex(const std::vector<LatticeCell>& cells)
        : cells_(cells), order_(cells.size()) {
        std::iota(order_.begin(), order_.end(), 0);
        if (!cells.empty()) build(0, cells.size());
    }

    template<class Visitor>
    bool visit(const Eigen::Vector3d& center, double radius,
               Visitor&& visitor) const {
        return nodes_.empty() || visit_node(0, center, radius * radius, visitor);
    }

private:
    int build(std::size_t begin, std::size_t end) {
        Node node;
        node.begin = begin;
        node.end = end;
        node.lo = node.hi = cells_[order_[begin]].r_cart;
        for (auto p = begin + 1; p < end; ++p) {
            node.lo = node.lo.cwiseMin(cells_[order_[p]].r_cart);
            node.hi = node.hi.cwiseMax(cells_[order_[p]].r_cart);
        }
        const int pos = static_cast<int>(nodes_.size());
        nodes_.push_back(node);
        if (end - begin > 16) {
            Eigen::Index axis;
            (node.hi - node.lo).maxCoeff(&axis);
            const auto mid = begin + (end - begin) / 2;
            std::nth_element(order_.begin() + begin, order_.begin() + mid,
                             order_.begin() + end, [&](int a, int b) {
                return cells_[a].r_cart[axis] < cells_[b].r_cart[axis];
            });
            const int left = build(begin, mid);
            const int right = build(mid, end);
            nodes_[pos].left = left;
            nodes_[pos].right = right;
        }
        return pos;
    }

    template<class Visitor>
    bool visit_node(int pos, const Eigen::Vector3d& center, double radius2,
                    Visitor& visitor) const {
        const auto& node = nodes_[pos];
        const auto nearest = center.cwiseMax(node.lo).cwiseMin(node.hi);
        if ((center - nearest).squaredNorm() > radius2) return true;
        if (node.left < 0) {
            for (auto p = node.begin; p < node.end; ++p) {
                const int c = order_[p];
                if ((cells_[c].r_cart - center).squaredNorm() <= radius2 &&
                    !visitor(c)) return false;
            }
            return true;
        }
        return visit_node(node.left, center, radius2, visitor) &&
               visit_node(node.right, center, radius2, visitor);
    }

    const std::vector<LatticeCell>& cells_;
    std::vector<int> order_;
    std::vector<Node> nodes_;
};

struct ErfcGaussianAmplitude { double exponent, weight; };

std::vector<ErfcGaussianAmplitude> erfc_radial_amplitudes(
        const libint2::Shell& shell, double eta) {
    using Gaussian = ErfcGaussianAmplitude;
    if (shell.contr.size() != 1 || shell.alpha.empty()) return {};
    const auto& contraction = shell.contr.front();
    std::vector<Gaussian> amplitudes;
    for (std::size_t p = 0; p < shell.alpha.size(); ++p) {
        const double alpha = shell.alpha[p];
        if (!(alpha > 0.0) || !std::isfinite(alpha)) return {};
        const int l = contraction.l;
        // Every Cartesian monomial and normalized real solid
        // harmonic obeys |P_l(r)| <= r^l. For 0 < eta < 1,
        // r^l exp(-eta alpha r^2) <= [l/(2 e eta alpha)]^(l/2).
        // Taking absolute contraction coefficients, then squaring,
        // yields a positive Gaussian upper bound on each AO^2.
        const double scale = l == 0 ? 1.0
            : std::pow(l / (2.0 * std::exp(1.0) * eta * alpha), 0.5 * l);
        amplitudes.push_back({(l == 0 ? 1.0 : 1.0 - eta) * alpha,
            std::abs(contraction.coeff[p]) * scale});
    }
    if (amplitudes.size() > 8) {
        // Bound a long contraction with at most eight groups:
        // each group's slowest exponent bounds all its members.
        // Keeping distinct exponent ranges avoids assigning the
        // compact primitives the most diffuse primitive's tail.
        std::sort(amplitudes.begin(), amplitudes.end(),
            [](const auto& a, const auto& b) { return a.exponent < b.exponent; });
        std::vector<Gaussian> grouped;
        for (std::size_t group = 0; group < 8; ++group) {
            const auto begin = group * amplitudes.size() / 8;
            const auto end = (group + 1) * amplitudes.size() / 8;
            double weight = 0.0;
            for (auto p = begin; p < end; ++p) weight += amplitudes[p].weight;
            grouped.push_back({amplitudes[begin].exponent, weight});
        }
        amplitudes = std::move(grouped);
    }
    return amplitudes;
}

// Positive radial envelopes for the charge-pair Schwarz inequality
//   |(ab|cd)_SR| <= min(sqrt((aa|cc)_SR (bb|dd)_SR),
//                      sqrt((aa|dd)_SR (bb|cc)_SR)).
// Sun (2023), Eq. 51, doi:10.1063/5.0155815. Unlike a Gaussian
// product-center attenuation, this bounds angular and contracted tails.
class ErfcChargeBounds {
    using Gaussian = ErfcGaussianAmplitude;
    struct Term { double weight, decay, at_zero; };
    struct RadiusEntry { double bound, radius; };
    using Envelope = std::vector<Gaussian>;
    using Alternatives = std::vector<std::vector<Term>>;
public:
    ErfcChargeBounds(const libint2::BasisSet& shells,
                     const std::vector<LatticeCell>& cells, double omega,
                     bool requested) : ns_(shells.size()), nc_(cells.size()) {
        constexpr std::size_t budget = 64 * 1024 * 1024;
        if (!requested || ns_ == 0 || nc_ == 0 || ns_ > 256 ||
            nc_ > (budget / (ns_ * ns_) - sizeof(double) - sizeof(std::vector<RadiusEntry>))
                / (sizeof(double) + sizeof(RadiusEntry))) return;
        std::vector<std::vector<Envelope>> envelopes;
        for (const auto& shell : shells) {
            // The numerical engines use single-contraction shells. Unknown
            // representations keep the ordinary Schwarz-only traversal.
            if (shell.contr.size() != 1 || shell.alpha.empty()) return;
            const auto& contraction = shell.contr.front();
            std::vector<Envelope> choices;
            // Each eta gives an independent upper bound. Taking the minimum
            // improves tightness without making these an accuracy parameter.
            for (double eta : {0.125, 0.25, 0.5}) {
                auto amplitudes = erfc_radial_amplitudes(shell, eta);
                if (amplitudes.empty()) return;
                Envelope charge;
                for (std::size_t p = 0; p < amplitudes.size(); ++p)
                    for (std::size_t q = 0; q <= p; ++q)
                        charge.push_back({amplitudes[p].exponent + amplitudes[q].exponent,
                            amplitudes[p].weight * amplitudes[q].weight * (p == q ? 1.0 : 2.0)});
                choices.push_back(std::move(charge));
                if (contraction.l == 0) break;
            }
            envelopes.push_back(std::move(choices));
        }
        ceilings_.resize(ns_ * ns_);
        values_.resize(nc_ * ns_ * ns_);
        radii_.resize(ns_ * ns_);
        const double pi = std::acos(-1.0);
        double max_cell_norm = 0.0;
        for (const auto& cell : cells) max_cell_norm = std::max(max_cell_norm, cell.r_cart.norm());
        // Each work item owns one pair's table. Scratch contains at most
        // 9 * 36^2 terms (under 300 KiB) per worker, independent of cells.
        #pragma omp parallel for schedule(dynamic)
        for (int pair = 0; pair < static_cast<int>(ns_ * ns_); ++pair) {
            const auto a = static_cast<std::size_t>(pair) / ns_;
            const auto b = static_cast<std::size_t>(pair) % ns_;
            Alternatives alternatives;
            for (const auto& ea : envelopes[a])
                for (const auto& eb : envelopes[b]) {
                    std::vector<Term> terms;
                    terms.reserve(ea.size() * eb.size());
                    for (const auto& ga : ea)
                        for (const auto& gb : eb) {
                            const double rho = ga.exponent * gb.exponent / (ga.exponent + gb.exponent);
                            terms.push_back({ga.weight * gb.weight * std::pow(pi * pi / (ga.exponent * gb.exponent), 1.5),
                                std::sqrt(1.0 / (1.0 / rho + 1.0 / (omega * omega))),
                                2.0 * std::sqrt(rho / pi)});
                        }
                    alternatives.push_back(std::move(terms));
                }
            const double geometry_slack = 1e-10 * (1.0 + 2.0 * max_cell_norm
                + Eigen::Vector3d(shells[a].O[0], shells[a].O[1], shells[a].O[2]).norm()
                + Eigen::Vector3d(shells[b].O[0], shells[b].O[1], shells[b].O[2]).norm());
            const auto evaluate = [&](double r) {
                // Re-anchoring a pair can round differently from constructing
                // its two absolute shell positions. Lower distance outwards.
                r = std::max(0.0, r - geometry_slack);
                double result = std::numeric_limits<double>::infinity();
                for (const auto& terms : alternatives) {
                    double value = 0.0;
                    for (const auto& term : terms) {
                        // The smeared erfc interaction equals
                        // [erfc(sqrt(mw)R)-erfc(sqrt(rho)R)]/R.
                        // Dropping the positive second term avoids Boys-tail
                        // subtraction; the full Coulomb origin value also
                        // bounds it. Both bounds are positive and decreasing.
                        const double kernel = r == 0.0 ? term.at_zero
                            : std::min(term.at_zero, std::erfc(term.decay * r) / r);
                        value += term.weight * kernel;
                    }
                    if (!std::isfinite(value)) value = std::numeric_limits<double>::infinity();
                    result = std::min(result, value);
                }
                // Outward numerical allowance on positive sums and square root.
                return std::sqrt(result) * (1.0 + 1.0e-12);
            };
            ceilings_[pair] = evaluate(0.0);
            auto& row = radii_[pair];
            row.reserve(nc_);
            const Eigen::Vector3d offset(shells[a].O[0] - shells[b].O[0],
                shells[a].O[1] - shells[b].O[1], shells[a].O[2] - shells[b].O[2]);
            for (std::size_t c = 0; c < nc_; ++c) {
                const double r = (offset - cells[c].r_cart).norm();
                const double bound = evaluate(r);
                values_[c * ns_ * ns_ + pair] = bound;
                row.push_back({bound, r});
            }
            std::sort(row.begin(), row.end(), [](const auto& a, const auto& b) { return a.bound > b.bound; });
            double radius = 0.0;
            for (auto& entry : row) { radius = std::max(radius, entry.radius); entry.radius = radius; }
        }
        enabled_ = true;
    }
    bool enabled() const { return enabled_; }
    double at(int cell, std::size_t a, std::size_t b) const {
        if (!enabled_) return std::numeric_limits<double>::infinity();
        return cell < 0 ? ceiling(a, b) : values_[static_cast<std::size_t>(cell) * ns_ * ns_ + a * ns_ + b];
    }
    double ceiling(std::size_t a, std::size_t b) const { return ceilings_[a * ns_ + b]; }
    double radius(std::size_t a, std::size_t b, double required) const {
        if (!(required > 0.0)) return std::numeric_limits<double>::infinity();
        const auto& row = radii_[a * ns_ + b];
        const auto end = std::partition_point(row.begin(), row.end(),
            [&](const auto& entry) { return entry.bound >= required; });
        return end == row.begin() ? -1.0 : std::prev(end)->radius;
    }
private:
    std::size_t ns_, nc_;
    bool enabled_ = false;
    std::vector<double> values_, ceilings_;
    std::vector<std::vector<RadiusEntry>> radii_;
};

// Integrating positive Gaussian envelopes of both AO products retains their
// overlap decay as well as their separation decay. Each product is enclosed
// by its total positive mass, slowest exponent and a ball of product centers.
// For R > 0, the smeared erfc kernel is bounded by exp(-m_omega R^2)/R.
// This combines angular tails and overlap without multiplying an estimated
// attenuation into a Schwarz factor.
class ErfcProductBounds {
    struct Node {
        Eigen::Vector3d lo, hi, cell_lo, cell_hi;
        double radius, inverse_exponent, log_mass;
        std::size_t begin, end;
        int left = -1, right = -1;
    };
public:
    struct Entry {
        double log_mass = std::numeric_limits<double>::infinity();
        double inverse_exponent = std::numeric_limits<double>::infinity();
        Eigen::Vector3d center = Eigen::Vector3d::Zero();
        double radius = std::numeric_limits<double>::infinity();
    };
    ErfcProductBounds(const libint2::BasisSet& shells,
                      const std::vector<LatticeCell>& cells, bool requested)
        : ns_(shells.size()) {
        constexpr std::size_t budget = 128 * 1024 * 1024;
        if (!requested || ns_ == 0 || ns_ > 256 || cells.empty() ||
            cells.size() > budget / sizeof(Entry) / (ns_ * ns_)) return;
        std::vector<std::vector<ErfcGaussianAmplitude>> amplitudes;
        for (const auto& shell : shells) {
            amplitudes.push_back(erfc_radial_amplitudes(shell, 0.25));
            if (amplitudes.back().empty()) return;
        }
        entries_.resize(cells.size() * ns_ * ns_);
        const double pi = std::acos(-1.0);
        double max_cell_norm = 0.0;
        for (const auto& cell : cells) max_cell_norm = std::max(max_cell_norm, cell.r_cart.norm());
        #pragma omp parallel for schedule(dynamic)
        for (int key = 0; key < static_cast<int>(entries_.size()); ++key) {
            const auto b = static_cast<std::size_t>(key) % ns_;
            const auto a = static_cast<std::size_t>(key) / ns_ % ns_;
            const auto cell = static_cast<std::size_t>(key) / (ns_ * ns_);
            const Eigen::Vector3d A(shells[a].O[0], shells[a].O[1], shells[a].O[2]);
            const Eigen::Vector3d B = Eigen::Vector3d(shells[b].O[0],
                shells[b].O[1], shells[b].O[2]) + cells[cell].r_cart;
            Eigen::Vector3d lower = Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
            Eigen::Vector3d upper = -lower;
            const double geometry_slack = 1e-10 * (1.0 + A.norm() + B.norm() + 2.0 * max_cell_norm);
            const double distance = std::max(0.0, (A - B).norm() - geometry_slack);
            double mass = 0.0, exponent = std::numeric_limits<double>::infinity();
            for (const auto& ga : amplitudes[a])
                for (const auto& gb : amplitudes[b]) {
                    const double sum = ga.exponent + gb.exponent;
                    const Eigen::Vector3d center = (ga.exponent * A + gb.exponent * B) / sum;
                    lower = lower.cwiseMin(center);
                    upper = upper.cwiseMax(center);
                    exponent = std::min(exponent, sum);
                    mass += ga.weight * gb.weight * std::pow(pi / sum, 1.5)
                        * std::exp(-ga.exponent * gb.exponent / sum * distance * distance);
                }
            if (!(mass > 0.0) || !std::isfinite(mass) || !(exponent > 0.0) ||
                !lower.allFinite() || !upper.allFinite()) continue;
            auto& entry = entries_[key];
            entry.log_mass = std::log(mass) + 1e-12;
            entry.inverse_exponent = 1.0 / exponent;
            entry.center = 0.5 * (lower + upper);
            entry.radius = 0.5 * (upper - lower).norm() + geometry_slack;
        }
        enabled_ = true;
        // A bounded product-center hierarchy also rejects whole exchange
        // neighborhoods using overlap mass, rather than visiting every cell
        // in the looser charge-pair sphere. Its leaves index supplied cells.
        const auto count_nodes = [&](auto&& self, std::size_t n) -> std::size_t {
            return n <= 16 ? 1 : 1 + self(self, n / 2) + self(self, n - n / 2);
        };
        const auto nodes_per_pair = count_nodes(count_nodes, cells.size());
        constexpr std::size_t tree_budget = 64 * 1024 * 1024;
        const auto pair_bytes = cells.size() * sizeof(int) + nodes_per_pair * sizeof(Node) + sizeof(int);
        if (ns_ * ns_ > tree_budget / pair_bytes) return;
        order_.resize(entries_.size());
        nodes_.resize(ns_ * ns_ * nodes_per_pair);
        roots_.resize(ns_ * ns_);
        #pragma omp parallel for schedule(dynamic)
        for (int pair = 0; pair < static_cast<int>(ns_ * ns_); ++pair) {
            const auto begin = static_cast<std::size_t>(pair) * cells.size();
            std::iota(order_.begin() + begin, order_.begin() + begin + cells.size(), 0);
            int next = static_cast<int>(static_cast<std::size_t>(pair) * nodes_per_pair);
            const auto entry_at = [&](int cell) -> const Entry& {
                return entries_[static_cast<std::size_t>(cell) * ns_ * ns_ + pair];
            };
            const auto build = [&](auto&& self, std::size_t first, std::size_t last) -> int {
                const int pos = next++;
                auto& node = nodes_[pos];
                node.begin = first; node.end = last;
                node.lo = node.hi = entry_at(order_[first]).center;
                node.cell_lo = node.cell_hi = cells[order_[first]].r_cart;
                node.radius = node.inverse_exponent = 0.0;
                node.log_mass = -std::numeric_limits<double>::infinity();
                for (auto i = first; i < last; ++i) {
                    const auto& entry = entry_at(order_[i]);
                    node.lo = node.lo.cwiseMin(entry.center);
                    node.hi = node.hi.cwiseMax(entry.center);
                    node.cell_lo = node.cell_lo.cwiseMin(cells[order_[i]].r_cart);
                    node.cell_hi = node.cell_hi.cwiseMax(cells[order_[i]].r_cart);
                    node.radius = std::max(node.radius, entry.radius);
                    node.inverse_exponent = std::max(node.inverse_exponent, entry.inverse_exponent);
                    node.log_mass = std::max(node.log_mass, entry.log_mass);
                }
                if (last - first > 16) {
                    Eigen::Index axis;
                    (node.hi - node.lo).maxCoeff(&axis);
                    const auto mid = first + (last - first) / 2;
                    std::nth_element(order_.begin() + first, order_.begin() + mid,
                        order_.begin() + last, [&](int a, int b) {
                            return entry_at(a).center[axis] < entry_at(b).center[axis];
                        });
                    node.left = self(self, first, mid);
                    node.right = self(self, mid, last);
                }
                return pos;
            };
            roots_[pair] = build(build, begin, begin + cells.size());
        }
    }
    bool enabled() const { return enabled_; }
    bool indexed() const { return !roots_.empty(); }
    const Entry& at(int cell, std::size_t a, std::size_t b) const {
        return entries_[static_cast<std::size_t>(cell) * ns_ * ns_ + a * ns_ + b];
    }
    static double query_radius(const Entry& a, const Entry& b,
                               double omega, double log_density, double log_threshold) {
        const double inverse_decay = a.inverse_exponent + b.inverse_exponent + 1.0 / (omega * omega);
        const double excess = a.log_mass + b.log_mass + log_density - log_threshold;
        if (!std::isfinite(inverse_decay) || !std::isfinite(excess))
            return std::numeric_limits<double>::infinity();
        // For R >= 1 bohr, exp(-m R^2)/R <= exp(-m R^2).
        return std::max(1.0, std::sqrt(std::max(0.0, excess) * inverse_decay))
            + a.radius + b.radius;
    }
    static bool skip(const Entry& a, const Entry& b, const Eigen::Vector3d& shift,
                     double omega, double log_density, double log_threshold) {
        const double distance = (a.center - b.center - shift).norm() - a.radius - b.radius;
        if (!(distance > 0.0)) return false;
        const double inverse_decay = a.inverse_exponent + b.inverse_exponent + 1.0 / (omega * omega);
        const double log_bound = a.log_mass + b.log_mass + log_density
            - distance * distance / inverse_decay - std::log(distance);
        return log_bound < log_threshold;
    }
    template<class Visitor>
    bool visit(std::size_t a, std::size_t b, const Entry& left,
               const Eigen::Vector3d& shift, double omega, double log_density,
               double log_threshold, const Eigen::Vector3d& charge_center,
               double charge_radius, const std::vector<LatticeCell>& cells,
               Visitor&& visitor) const {
        const Eigen::Vector3d center = left.center - shift;
        const double radius2 = charge_radius * charge_radius;
        const auto walk = [&](auto&& self, int pos) -> bool {
            const auto& node = nodes_[pos];
            const auto nearest_cell = charge_center.cwiseMax(node.cell_lo).cwiseMin(node.cell_hi);
            if ((charge_center - nearest_cell).squaredNorm() > radius2) return true;
            const auto nearest = center.cwiseMax(node.lo).cwiseMin(node.hi);
            const double distance = (center - nearest).norm() - left.radius - node.radius;
            if (distance > 0.0) {
                // Node maxima give one upper bound for every contained
                // product. Larger Gaussian variance means slower erfc decay.
                const double inverse_decay = left.inverse_exponent + node.inverse_exponent
                    + 1.0 / (omega * omega);
                if (left.log_mass + node.log_mass + log_density
                        - distance * distance / inverse_decay - std::log(distance) < log_threshold)
                    return true;
            }
            if (node.left < 0) {
                for (auto i = node.begin; i < node.end; ++i) {
                    const int cell = order_[i];
                    if ((cells[cell].r_cart - charge_center).squaredNorm() > radius2) continue;
                    if (skip(left, at(cell, a, b), shift, omega, log_density, log_threshold)) continue;
                    if (!visitor(cell)) return false;
                }
                return true;
            }
            return self(self, node.left) && self(self, node.right);
        };
        return walk(walk, roots_[a * ns_ + b]);
    }
private:
    std::size_t ns_;
    bool enabled_ = false;
    std::vector<Entry> entries_;
    std::vector<Node> nodes_;
    std::vector<int> order_, roots_;
};

}  // namespace

JKMatrices build_jk_gamma_molecular_limit(const BasisSet& basis,
                                          const PeriodicSystem& system,
                                          const LatticeSumOptions& opts,
                                          const Eigen::MatrixXd& P,
                                          double omega) {
    ensure_libint_initialized();

    const auto& shells_ref = basis.libint();
    const int nbf = static_cast<int>(basis.nbasis());
    if (P.rows() != nbf || P.cols() != nbf) {
        throw std::runtime_error(
            "build_jk_gamma_molecular_limit: density shape mismatch");
    }
    if (omega < 0.0) {
        throw std::runtime_error(
            "build_jk_gamma_molecular_limit: omega must be non-negative");
    }

    if (opts.pair_complete_1e) {
        const auto cells = pair_complete_eri_cells(system, opts, shells_ref);
        LatticeMatrixSet density;
        density.nbf = nbf;
        density.cells = cells;
        density.blocks.assign(cells.size(), P);
        const auto lattice = build_jk_2e_real_space(basis, system, opts, density, omega);
        JKMatrices result{Eigen::MatrixXd::Zero(nbf, nbf), Eigen::MatrixXd::Zero(nbf, nbf)};
        for (const auto& block : lattice.J.blocks) result.J += block;
        for (const auto& block : lattice.K.blocks) result.K += block;
        result.J = 0.5 * (result.J + result.J.transpose()).eval();
        result.K = 0.5 * (result.K + result.K.transpose()).eval();
        return result;
    }

    // Cell list shared between both lattice-index sums. Consistent with
    // Phase 12a's convention that cutoff_bohr bounds the μν real-space
    // sum; the two indices share the same cell lattice here because in
    // the Γ-only molecular limit their cutoffs are effectively coupled
    // through the ν ↔ P-image symmetry.
    const auto cells = direct_lattice_cells(system, opts.cutoff_bohr);

    // When ω > 0 we use libint's erfc-screened Coulomb kernel
    // (erfc(ω·r_12)/r_12) — the short-range Ewald-split piece. The
    // scalar ω is passed to the engine via set_params.
    const libint2::Operator op = (omega > 0.0)
        ? libint2::Operator::erfc_coulomb
        : libint2::Operator::coulomb;
    libint2::Engine prototype(op, shells_ref.max_nprim(),
                              shells_ref.max_l(), 0);
    if (omega > 0.0) {
        prototype.set_params(omega);
    }
    auto engines = make_engine_pool(prototype);
    const auto shell2bf = shells_ref.shell2bf();
    const std::size_t nshells = shells_ref.size();

    // Pre-shift shells for every cell; amortise across the doubled loop.
    // Memory: O(N_c × n_shells × shell-size) — small. Shared across threads.
    std::vector<std::vector<libint2::Shell>> shells_at(cells.size());
    for (std::size_t c = 0; c < cells.size(); ++c) {
        shells_at[c] = shift_shells(shells_ref, cells[c].r_cart);
    }

    // ---- Cauchy–Schwarz pre-pass ------------------------------------------
    //
    // The Γ-only molecular-limit J/K kernel has loop shape (c_g, c_p) with
    // ν shifted by g and {λ, σ} both shifted by p. For each shell quartet:
    //
    //   J: (μ_0 ν_g | λ_p σ_p) — bra-pair displacement c_g, ket-pair 0.
    //   K: (μ_0 λ_p | ν_g σ_p) — bra-pair displacement c_p, ket-pair c_p−c_g.
    //
    // Schwarz: |⟨ab|cd⟩| ≤ Q_ab · Q_cd. Skip the quartet when the bound
    // (× density envelope) falls below ``opts.schwarz_threshold``.
    const double schwarz_thr = opts.schwarz_threshold;
    const bool screen = (schwarz_thr > 0.0);
    const auto Q = screen
        ? compute_schwarz_factors_per_cell(shells_ref, shells_at, prototype)
        : std::vector<std::vector<double>>{};
    const auto Q_max = screen
        ? max_q_per_cell(Q)
        : std::vector<double>{};
    const double D_max = screen
        ? std::max(1.0, std::max(std::fabs(P.maxCoeff()),
                                  std::fabs(P.minCoeff())))
        : 0.0;
    // Sx3: per-shell-pair density envelope for LinK-style screening.
    Eigen::MatrixXd Dpair;
    if (screen) {
        Dpair = Eigen::MatrixXd::Zero(
            static_cast<Eigen::Index>(nshells),
            static_cast<Eigen::Index>(nshells));
        for (std::size_t s = 0; s < nshells; ++s) {
            const auto bf_s = shell2bf[s];
            const auto n_s = shells_ref[s].size();
            for (std::size_t t = 0; t < nshells; ++t) {
                const auto bf_t = shell2bf[t];
                const auto n_t = shells_ref[t].size();
                double mx = 0.0;
                for (std::size_t i = 0; i < n_s; ++i)
                    for (std::size_t j = 0; j < n_t; ++j)
                        mx = std::max(mx, std::fabs(P(static_cast<Eigen::Index>(bf_s + i),
                                                        static_cast<Eigen::Index>(bf_t + j))));
                Dpair(static_cast<Eigen::Index>(s), static_cast<Eigen::Index>(t)) = mx;
            }
        }
    }
    CellIndexMap cell_index_map;
    if (screen) cell_index_map = build_cell_index_map(cells);
    // Q[0] indexes the zero-cell (Γ ket of J). It is always the first cell
    // returned by direct_lattice_cells, but we look it up via the index map
    // for safety.
    int c_zero_idx = -1;
    if (screen) {
        auto it = cell_index_map.find(Eigen::Vector3i(0, 0, 0));
        if (it != cell_index_map.end()) c_zero_idx = it->second;
    }
    const double q_zero_max =
        (screen && c_zero_idx >= 0) ? Q_max[c_zero_idx] : 0.0;

    // Parallelise the double cell loop over a flat (c_g, c_p) index. Each
    // iteration accumulates into thread-local J and K buffers; we reduce
    // across threads at the end. No write races without this because J
    // and K would otherwise be hot shared accumulators across all the
    // nested shell-quartet loops.
    const int n_c = static_cast<int>(cells.size());
    const int n_pairs = n_c * n_c;
    const int n_threads = omp_max_threads();
    std::vector<Eigen::MatrixXd> Jm_tls(
        n_threads, Eigen::MatrixXd::Zero(nbf, nbf));
    std::vector<Eigen::MatrixXd> Km_tls(
        n_threads, Eigen::MatrixXd::Zero(nbf, nbf));

    #pragma omp parallel for schedule(dynamic)
    for (int idx = 0; idx < n_pairs; ++idx) {
        const int c_g = idx / n_c;
        const int c_p = idx % n_c;
        const auto tid = static_cast<std::size_t>(omp_thread_index());
        auto& engine = engines[tid];
        const auto& buf = engine.results();
        auto& Jm_local = Jm_tls[tid];
        auto& Km_local = Km_tls[tid];

        const auto& shells_g = shells_at[c_g];       // ν shifted by g
        const auto& shells_p = shells_at[c_p];       // λ, σ shifted by g_λ

        // Cell index of c_p − c_g for the K ket-pair displacement.
        int c_pg_idx = -1;
        if (screen) {
            const Eigen::Vector3i pg = cells[c_p].index - cells[c_g].index;
            auto it = cell_index_map.find(pg);
            if (it != cell_index_map.end()) c_pg_idx = it->second;
        }

        // Cell-level Schwarz: skip the entire shell-quartet loop when
        // neither J nor K can possibly contribute above the threshold.
        if (screen) {
            const bool j_possible =
                (Q_max[c_g] * q_zero_max * D_max >= schwarz_thr);
            const double q_pg_max =
                (c_pg_idx >= 0) ? Q_max[c_pg_idx] : 0.0;
            const bool k_possible =
                (Q_max[c_p] * q_pg_max * D_max >= schwarz_thr);
            if (!j_possible && !k_possible) continue;
        }

        for (std::size_t s1 = 0; s1 < shells_ref.size(); ++s1) {
            const auto bf1 = shell2bf[s1];
            const auto n1 = shells_ref[s1].size();

            for (std::size_t s2 = 0; s2 < shells_g.size(); ++s2) {
                const auto bf2 = shell2bf[s2];
                const auto n2 = shells_g[s2].size();
                // J bra: (s1_0, s2_g). K ket: (s2_g, s4_p).
                const double q12_J = screen
                    ? Q[c_g][s1 * nshells + s2]
                    : 0.0;

                for (std::size_t s3 = 0; s3 < shells_p.size(); ++s3) {
                    const auto bf3 = shell2bf[s3];
                    const auto n3 = shells_p[s3].size();
                    // K bra: (s1_0, s3_p).
                    const double q13_K = screen
                        ? Q[c_p][s1 * nshells + s3]
                        : 0.0;

                    for (std::size_t s4 = 0; s4 < shells_p.size(); ++s4) {
                        const auto bf4 = shell2bf[s4];
                        const auto n4 = shells_p[s4].size();

                        // J: (μ_0 ν_g | λ_p σ_p) — ket pair at c=0.
                        bool do_J = true;
                        if (screen) {
                            if (c_zero_idx < 0) {
                                do_J = false;
                            } else {
                                const double q34_J =
                                    Q[c_zero_idx][s3 * nshells + s4];
                                // Sx3: shell-pair density envelope for the
                                // ket pair (λ_p σ_p) — the density element
                                // that multiplies (μ_0 ν_g | λ_p σ_p).
                                const double den34 = Dpair(
                                    static_cast<Eigen::Index>(s3),
                                    static_cast<Eigen::Index>(s4));
                                if (q12_J * q34_J * den34 < schwarz_thr)
                                    do_J = false;
                            }
                        }
                        if (do_J) {
                            engine.compute(shells_ref[s1], shells_g[s2],
                                           shells_p[s3], shells_p[s4]);
                            if (const double* blk = buf[0]) {
                                for (std::size_t i = 0; i < n1; ++i)
                                for (std::size_t j = 0; j < n2; ++j)
                                for (std::size_t k = 0; k < n3; ++k)
                                for (std::size_t l = 0; l < n4; ++l) {
                                    const double v = blk[
                                        ((i * n2 + j) * n3 + k) * n4 + l];
                                    Jm_local(bf1 + i, bf2 + j) +=
                                        P(bf3 + k, bf4 + l) * v;
                                }
                            }
                        }

                        // K: (μ_0 λ_p | ν_g σ_p) — ket pair at c_p − c_g.
                        bool do_K = true;
                        if (screen) {
                            if (c_pg_idx < 0) {
                                do_K = false;
                            } else {
                                const double q24_K =
                                    Q[c_pg_idx][s2 * nshells + s4];
                                // Sx3: shell-pair density envelope for the
                                // density pair (λ σ) — P(λ, σ) multiplies
                                // (μ_0 λ_p | ν_g σ_p) in the K build.
                                const double den34 = Dpair(
                                    static_cast<Eigen::Index>(s3),
                                    static_cast<Eigen::Index>(s4));
                                if (q13_K * q24_K * den34 * 0.5 < schwarz_thr)
                                    do_K = false;
                            }
                        }
                        if (do_K) {
                            engine.compute(shells_ref[s1], shells_p[s3],
                                           shells_g[s2], shells_p[s4]);
                            if (const double* blk = buf[0]) {
                                for (std::size_t i = 0; i < n1; ++i)
                                for (std::size_t j = 0; j < n2; ++j)
                                for (std::size_t k = 0; k < n3; ++k)
                                for (std::size_t l = 0; l < n4; ++l) {
                                    const double v = blk[
                                        ((i * n3 + k) * n2 + j) * n4 + l];
                                    Km_local(bf1 + i, bf2 + j) +=
                                        P(bf3 + k, bf4 + l) * v;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Reduce thread-local accumulators.
    Eigen::MatrixXd Jm = Eigen::MatrixXd::Zero(nbf, nbf);
    Eigen::MatrixXd Km = Eigen::MatrixXd::Zero(nbf, nbf);
    for (const auto& m : Jm_tls) Jm += m;
    for (const auto& m : Km_tls) Km += m;
    return JKMatrices{Jm, Km};
}

JKLatticeMatrixSets build_jk_2e_real_space_impl(
        const BasisSet& basis,
        const PeriodicSystem& system,
        const LatticeSumOptions& opts,
        const LatticeMatrixSet& P_real_space,
        const std::vector<LatticeCell>& cells,
        bool compute_exchange,
        double omega,
        // Phase SYM3b: when non-empty, build output (bra-cell c_g) blocks
        // ONLY for these cell indices; the internal c_λ/c_σ lattice sum stays
        // FULL over `cells`, so each computed block is exact. Other blocks are
        // left zero for the caller to fill by point-group reconstruction. An
        // empty list builds every cell (the original full behaviour).
        const std::vector<int>& output_indices = {},
        // Phase SYM3b shell-pair mask: when non-empty (and parallel to
        // `output_indices`), `output_shell_masks[oi]` is an nshells*nshells
        // row-major flag array; only output shell pairs (s1_home, s2_cg) with a
        // non-zero flag are built. This restricts the build to the
        // atom-pair-orbit *representative* sub-blocks (not just rep cells), the
        // finer reduction the point-group reconstruction actually needs. The
        // internal c_λ/c_σ/s3/s4 sum stays full, so each emitted sub-block is
        // exact. Empty → build every output shell pair of each output cell.
        const std::vector<std::vector<uint8_t>>& output_shell_masks = {},
        const std::vector<std::vector<uint8_t>>& density_shell_masks = {},
        const Eigen::MatrixXd* gamma_density = nullptr,
        bool density_derivative = false,
        // Skip mask for the dormant PDR 1988 Ch. II.4c quartet prototype.
        // Parallel to the (c_g * n_c + c_p) cell-pair flat index.
        // Each entry is a sorted vector of shell-quartet flat indices
        // (s1*nsh^3 + s2*nsh^2 + s3*nsh + s4); quartets in this mask
        // are FAR-FIELD and SKIPPED from the exact ERI evaluation.
        // Empty (default) → no skip, existing behaviour.
        const std::vector<std::vector<int>>& bipolar_skip_mask = {}) {
    ensure_libint_initialized();

    const auto& shells_ref = basis.libint();
    const int nbf = static_cast<int>(basis.nbasis());
    if (P_real_space.nbf != nbf) {
        throw std::runtime_error(
            "build_jk_2e_real_space: density nbf mismatch");
    }
    if (omega < 0.0) {
        throw std::runtime_error(
            "build_jk_2e_real_space: omega must be non-negative");
    }

    const auto n_c = cells.size();

    // Pre-shift shells once per cell.
    auto shift_shells = [&](const Eigen::Vector3d& dr) {
        std::vector<libint2::Shell> out(shells_ref.begin(), shells_ref.end());
        for (auto& s : out) {
            s.O[0] += dr[0]; s.O[1] += dr[1]; s.O[2] += dr[2];
        }
        return out;
    };
    std::vector<std::vector<libint2::Shell>> shells_at(n_c);
    for (std::size_t c = 0; c < n_c; ++c) {
        shells_at[c] = shift_shells(cells[c].r_cart);
    }

    // Density cell lookup by integer index. We'll use it to fetch P(h)
    // where h = g_σ − g_λ. Store blocks flat and index by cell position.
    // This and the `cells` displacement lookup below are dense
    // bounding-box tables (DenseCellTable above): the traversal queries
    // them O(n_out × n_pad²) times per Fock build, and unordered_map
    // probing on that floor was a measured hot spot on padded BIPOLE
    // ket-image balls (registry BIPOLE-SR-PAD-OVERCONSERVATIVE). Every
    // query is either a cell index or a difference of two cell indices,
    // so the Minkowski-difference box of `cells` contains all of them;
    // density cells outside it are unreachable exactly as they were
    // unreachable map entries. Same contents -> same traversal -> same
    // J/K bits.
    Eigen::Vector3i cells_idx_lo, cells_idx_hi;
    cell_index_bounds(cells, cells_idx_lo, cells_idx_hi);
    const Eigen::Vector3i diff_lo = cells_idx_lo - cells_idx_hi;
    const Eigen::Vector3i diff_hi = cells_idx_hi - cells_idx_lo;
    DenseCellTable p_cell_index(diff_lo, diff_hi);
    for (std::size_t i = 0; i < P_real_space.cells.size(); ++i) {
        p_cell_index.insert(P_real_space.cells[i].index,
                            static_cast<int>(i));
    }
    auto p_block_idx = [&](int c_lam, int c_sig) -> int {
        const Eigen::Vector3i h = cells[c_sig].index - cells[c_lam].index;
        return p_cell_index.at(h);
    };

    // Optionally switch to the erfc-screened Coulomb kernel for Ewald
    // short-range ERIs (same convention as build_jk_gamma_molecular_limit).
    const libint2::Operator op = (omega > 0.0)
        ? libint2::Operator::erfc_coulomb
        : libint2::Operator::coulomb;
    libint2::Engine prototype(op, shells_ref.max_nprim(),
                              shells_ref.max_l(), 0);
    if (omega > 0.0) {
        prototype.set_params(omega);
    }
    auto engines = make_engine_pool(prototype);
    const auto shell2bf = shells_ref.shell2bf();
    const std::size_t nshells = shells_ref.size();

    // ---- Cauchy–Schwarz pre-pass ------------------------------------------
    //
    // |⟨μ ν_g | λ_λ σ_σ⟩|  ≤  Q[c_g][s1, s2] · Q[c_h][s3, s4]
    //
    // (c_h = c_σ − c_λ for J; for K the cell decomposition differs —
    // see below). Skip the libint quartet call when the bound × density
    // envelope falls below ``opts.schwarz_threshold``. Without this the
    // cost is O(n_c³ · n_shells⁴) libint quartet calls per Fock build —
    // O(10⁹) for LiH/STO-3G with cutoff 15 bohr; with screening it
    // drops to seconds.
    const double schwarz_thr = opts.schwarz_threshold;
    const bool screen = (schwarz_thr > 0.0);
    const auto Q = screen
        ? compute_schwarz_factors_per_cell(shells_ref, shells_at, prototype)
        : std::vector<std::vector<double>>{};
    const double D_max = screen ? density_envelope(P_real_space.blocks) : 0.0;
    // Cell-level Schwarz: ``Q_max[c] = max_{s_a, s_b} Q[c][s_a, s_b]``.
    // For any cell triple (c_g, c_lam, c_sig) the bound on *any* shell
    // quartet is at most ``Q_max[c_g] · Q_max[c_h] · D_max`` (J) or
    // ``Q_max[c_lam] · Q_max[c_kh] · D_max · 0.5·|alpha|`` (K). If both
    // fail the threshold we can skip the entire shell-quartet loop —
    // a 10-100× speedup on real crystals where many triples have
    // negligible shell-pair overlap.
    //
    // Sx3: Density-weighted (LinK-style) per-shell-pair screening.
    // Compute per-shell-pair |D| max for each density block so the
    // shell-quartet screening below uses the local density envelope
    // instead of the global D_max.  For insulators the off-diagonal
    // density blocks decay exponentially, so this buys genuine O(N)
    // exchange screening.
    std::vector<Eigen::MatrixXd> Dpair_per_block;
    if (screen) {
        Dpair_per_block.reserve(P_real_space.blocks.size());
        for (const auto& B : P_real_space.blocks) {
            Eigen::MatrixXd Dpair(nshells, nshells);
            if (B.size() == 0) {
                Dpair.setZero();
            } else {
                for (std::size_t s = 0; s < nshells; ++s) {
                    const auto bf_s = shell2bf[s];
                    const auto n_s = shells_ref[s].size();
                    for (std::size_t t = 0; t < nshells; ++t) {
                        const auto bf_t = shell2bf[t];
                        const auto n_t = shells_ref[t].size();
                        double mx = 0.0;
                        for (std::size_t i = 0; i < n_s; ++i)
                            for (std::size_t j = 0; j < n_t; ++j)
                                mx = std::max(mx, std::fabs(B(static_cast<Eigen::Index>(bf_s + i),
                                                                static_cast<Eigen::Index>(bf_t + j))));
                        Dpair(static_cast<Eigen::Index>(s), static_cast<Eigen::Index>(t)) = mx;
                    }
                }
            }
            Dpair_per_block.push_back(std::move(Dpair));
        }
    }
    const auto Q_max = screen
        ? max_q_per_cell(Q)
        : std::vector<double>{};
    // Map cell-index → position in `cells`, for K-displacement lookup
    // (c_sig.index − c_g.index may not be in `cells` — if so its Q is 0
    // by truncation and the K bound vanishes). Dense bounding-box table
    // for the same reason as `p_cell_index` above.
    DenseCellTable cell_pos_table;
    if (screen) {
        cell_pos_table = DenseCellTable(diff_lo, diff_hi);
        for (std::size_t i = 0; i < cells.size(); ++i) {
            cell_pos_table.insert(cells[i].index, static_cast<int>(i));
        }
    }

    // Charge-pair Schwarz bounds include the angular polynomial and every
    // contraction coefficient (#755). Sparse range searches enclose the
    // accepted terms of this same predicate; no QQR attenuation remains.
    const bool valid_charge_geometry = std::all_of(cells.begin(), cells.end(), [&](const auto& cell) {
        const Eigen::Vector3d expected = system.lattice * cell.index.template cast<double>();
        return (expected - cell.r_cart).norm() <= 1e-12 * (1.0 + expected.norm());
    });
    const ErfcChargeBounds charge_bounds(shells_ref, cells, omega,
        screen && opts.sr_range_screening && omega > 0.0 && valid_charge_geometry);
    const ErfcProductBounds product_bounds(shells_ref, cells, charge_bounds.enabled());
    const double ln_thr = screen ? std::log(schwarz_thr) : 0.0;
    constexpr double LN_STAGE1_MARGIN = 1e-6;
    std::vector<std::vector<double>> lnQ;
    std::vector<Eigen::MatrixXd> lnDpair_per_block;
    if (charge_bounds.enabled() || (screen && opts.pair_complete_1e)) {
        lnQ = Q;
        for (auto& row : lnQ)
            for (auto& value : row) value = std::log(value);
        for (const auto& density : Dpair_per_block)
            lnDpair_per_block.push_back(density.array().log().matrix());
    }
    const double interaction_cutoff = eri_interaction_cutoff(opts);
    const auto physical_quartet = [&](int g, int lam, int sig, std::size_t s1,
                                      std::size_t s2, std::size_t s3, std::size_t s4,
                                      bool exchange) {
        if (!opts.pair_complete_1e) return true;
        const auto& a = shells_ref[s1];
        const auto& b = shells_at[g][s2];
        const auto& c = shells_at[lam][s3];
        const auto& d = shells_at[sig][s4];
        return exchange ? pair_products_in_range(a, c, b, d, opts.cutoff_bohr, interaction_cutoff)
                        : pair_products_in_range(a, b, c, d, opts.cutoff_bohr, interaction_cutoff);
    };
    const auto charge_skip = [&](int g, int lam, int sig, std::size_t s1,
                                 std::size_t s2, std::size_t s3, std::size_t s4,
                                 double density, bool exchange) {
        if (!charge_bounds.enabled()) return false;
        const int p_minus_g = cell_pos_table.at(cells[lam].index - cells[g].index);
        double bound = charge_bounds.at(sig, s1, s4)
            * charge_bounds.at(p_minus_g, s2, s3);
        if (exchange) {
            const int h = cell_pos_table.at(cells[sig].index - cells[lam].index);
            bound = std::min(bound, charge_bounds.at(g, s1, s2)
                * charge_bounds.at(h, s3, s4));
        } else {
            const int t = cell_pos_table.at(cells[sig].index - cells[g].index);
            bound = std::min(bound, charge_bounds.at(lam, s1, s3)
                * charge_bounds.at(t, s2, s4));
        }
        if (bound * density * (exchange ? 0.5 : 1.0) < schwarz_thr) return true;
        if (!product_bounds.enabled()) return false;
        const int right_cell = cell_pos_table.at(cells[sig].index
            - cells[exchange ? g : lam].index);
        if (right_cell < 0) return false;
        const auto& left = exchange ? product_bounds.at(lam, s1, s3)
                                    : product_bounds.at(g, s1, s2);
        const auto& right = exchange ? product_bounds.at(right_cell, s2, s4)
                                     : product_bounds.at(right_cell, s3, s4);
        return ErfcProductBounds::skip(left, right, cells[exchange ? g : lam].r_cart,
            omega, std::log(density) + (exchange ? std::log(0.5) : 0.0), ln_thr);
    };

    // Pair-specific lattice selection, PDR (1988), Ch. II.4b(iii),
    // doi:10.1007/978-3-642-93385-1. The finite domain remains caller-owned.
    bool sparse = screen && (omega > 0.0 || opts.pair_complete_1e) && opts.sr_sparse_traversal
        && std::isfinite(omega) && std::isfinite(D_max)
        && std::all_of(Q_max.begin(), Q_max.end(), [](double q) {
            return std::isfinite(q) && q >= 0.0;
        });
    std::vector<int> q_order;
    std::vector<double> cell_norm(n_c);
    double coordinate_slack = 0.0;
    if (sparse) {
        q_order.resize(n_c);
        std::iota(q_order.begin(), q_order.end(), 0);
        std::sort(q_order.begin(), q_order.end(), [&](int a, int b) {
            return Q_max[a] > Q_max[b];
        });
        double max_coordinate = 0.0;
        for (std::size_t c = 0; c < n_c; ++c) {
            cell_norm[c] = cells[c].r_cart.norm();
            max_coordinate = std::max(max_coordinate, cell_norm[c]);
            // Explicit-domain callers can supply duplicate indices. Keep
            // their historical multiplicities via exhaustive traversal.
            if (cell_pos_table.at(cells[c].index) != static_cast<int>(c))
                sparse = false;
            const Eigen::Vector3d expected = system.lattice * cells[c].index.cast<double>();
            if ((expected - cells[c].r_cart).norm() >
                    1e-12 * (1.0 + expected.norm())) sparse = false;
        }
        coordinate_slack = 1e-10 * (1.0 + max_coordinate);
    }
    const std::vector<LatticeCell> no_cells;
    const CellRangeIndex range_index(sparse ? cells : no_cells);
    // Hierarchical shell-pair bounds: reject a prefix before expanding its
    // remaining shell indices. Every factor bounds the corresponding
    // factor in the existing quartet Schwarz test, with the same product
    // order. These are bounds on that test, not extra integral estimates.
    std::vector<std::vector<double>> q_row_max, d_row_max;
    std::vector<double> d_block_max;
    if (sparse) {
        q_row_max.assign(n_c, std::vector<double>(nshells, 0.0));
        for (std::size_t c = 0; c < n_c; ++c)
            for (std::size_t s = 0; s < nshells; ++s)
                for (std::size_t t = 0; t < nshells; ++t)
                    q_row_max[c][s] = std::max(q_row_max[c][s], Q[c][s * nshells + t]);
        d_row_max.assign(Dpair_per_block.size(), std::vector<double>(nshells, 0.0));
        d_block_max.assign(Dpair_per_block.size(), 0.0);
        for (std::size_t p = 0; p < Dpair_per_block.size(); ++p) {
            for (std::size_t s = 0; s < nshells; ++s) {
                for (std::size_t t = 0; t < nshells; ++t)
                    d_row_max[p][s] = std::max(d_row_max[p][s], Dpair_per_block[p](s, t));
                d_block_max[p] = std::max(d_block_max[p], d_row_max[p][s]);
            }
        }
    }
    const auto candidates_for = [&](int g, auto&& append) {
        const auto add = [&](int lam, int sig) {
            return p_block_idx(lam, sig) < 0 || append(lam, sig);
        };
        for (int h : q_order) {
            if (Q_max[g] * Q_max[h] * D_max < schwarz_thr) break;
            const int p = p_cell_index.at(cells[h].index);
            if (p < 0 || Q_max[g] * Q_max[h] * d_block_max[p] < schwarz_thr) continue;
            for (std::size_t lam = 0; lam < n_c; ++lam) {
                const int sig = cell_pos_table.at(cells[lam].index + cells[h].index);
                if (sig >= 0 && !add(lam, sig)) return false;
            }
        }
        if (compute_exchange && !q_order.empty()) {
            const double max_q = Q_max[q_order.front()];
            for (int lam : q_order) {
                if (Q_max[lam] * max_q * D_max * 0.5 < schwarz_thr) break;
                for (int t : q_order) {
                    if (Q_max[lam] * Q_max[t] * D_max * 0.5 < schwarz_thr) break;
                    const int sig = cell_pos_table.at(cells[g].index + cells[t].index);
                    if (sig >= 0 && !add(lam, sig)) return false;
                }
            }
        }
        return true;
    };

    // Allocate result components: one nbf × nbf block per cell in `cells`.
    LatticeMatrixSet J_set;
    J_set.nbf = nbf;
    J_set.cells = cells;  // same cell list as ν-shift
    J_set.blocks.assign(n_c, Eigen::MatrixXd::Zero(nbf, nbf));

    LatticeMatrixSet K_set;
    K_set.nbf = nbf;
    K_set.cells = cells;
    K_set.blocks.assign(n_c, Eigen::MatrixXd::Zero(nbf, nbf));

    // Parallelise over c_g — each thread writes to distinct J/K blocks,
    // so no synchronisation is needed even though the inner
    // (c_λ, c_σ, s1..s4) loops accumulate into those blocks.
    // Phase SYM3b: iterate the bra-cell loop over the caller's output subset
    // when supplied (point-group orbit representatives), else over all cells.
    const int n_c_i = static_cast<int>(n_c);
    const bool use_subset = !output_indices.empty();
    const int n_out = use_subset
        ? static_cast<int>(output_indices.size())
        : n_c_i;
    // Output positions index THIS build's cell list. A caller that derived
    // them from some other cell list would otherwise read off the end of
    // `cells` and segfault; fail with the reason instead.
    for (int oi : output_indices) {
        if (oi < 0 || oi >= n_c_i) {
            throw std::runtime_error(
                "build_jk_2e_real_space: output_indices entry " +
                std::to_string(oi) + " is outside this build's cell list (" +
                std::to_string(n_c_i) + " cells). Output positions must "
                "index direct_lattice_cells(system, opts.cutoff_bohr).");
        }
    }
    const bool use_shell_mask = !output_shell_masks.empty();
    if (use_shell_mask &&
        static_cast<int>(output_shell_masks.size()) != n_out) {
        throw std::runtime_error(
            "build_jk_2e_real_space: output_shell_masks must be parallel to "
            "output_indices (one nshells*nshells mask per output cell)");
    }
    const bool use_density_mask = !density_shell_masks.empty();
    if (use_density_mask &&
        density_shell_masks.size() != P_real_space.blocks.size()) {
        throw std::runtime_error(
            "build_jk_2e_real_space: density_shell_masks must be parallel "
            "to the real-space density blocks");
    }
    if (gamma_density != nullptr &&
        (gamma_density->rows() != nbf || gamma_density->cols() != nbf)) {
        throw std::runtime_error(
            "build_jk_2e_real_space: gamma_density must be nbf x nbf");
    }
    if (gamma_density != nullptr && density_derivative) {
        throw std::runtime_error(
            "build_jk_2e_real_space: Gamma and real-space density "
            "derivatives are mutually exclusive");
    }
    const auto validate_shell_masks = [&](const auto& masks,
                                          const char* name) {
        for (const auto& mask : masks) {
            if (mask.size() != nshells * nshells) {
                throw std::runtime_error(
                    std::string("build_jk_2e_real_space: ") + name +
                    " entries must each contain nshells*nshells flags");
            }
        }
    };
    validate_shell_masks(output_shell_masks, "output_shell_masks");
    validate_shell_masks(density_shell_masks, "density_shell_masks");
    const int n_threads = omp_max_threads();
    std::vector<Eigen::MatrixXd> J_adjoint_tls;
    std::vector<Eigen::MatrixXd> K_adjoint_tls;
    std::vector<std::vector<Eigen::MatrixXd>> J_density_adjoint_tls;
    std::vector<std::vector<Eigen::MatrixXd>> K_density_adjoint_tls;
    if (gamma_density != nullptr) {
        J_adjoint_tls.assign(
            static_cast<std::size_t>(n_threads),
            Eigen::MatrixXd::Zero(nbf, nbf));
        K_adjoint_tls.assign(
            static_cast<std::size_t>(n_threads),
            Eigen::MatrixXd::Zero(nbf, nbf));
    }
    if (density_derivative) {
        const auto zero_blocks = std::vector<Eigen::MatrixXd>(
            P_real_space.blocks.size(),
            Eigen::MatrixXd::Zero(nbf, nbf));
        J_density_adjoint_tls.assign(
            static_cast<std::size_t>(n_threads), zero_blocks);
        K_density_adjoint_tls.assign(
            static_cast<std::size_t>(n_threads), zero_blocks);
    }
    std::vector<int> output_density_positions(
        static_cast<std::size_t>(n_out), -1);
    if (density_derivative) {
        for (int oi = 0; oi < n_out; ++oi) {
            const int c_g = use_subset ? output_indices[oi] : oi;
            const int outer_pos = p_cell_index.at(cells[c_g].index);
            if (outer_pos < 0) {
                throw std::runtime_error(
                    "build_jk_2e_real_space: output cell is absent from "
                    "the real-space density support");
            }
            output_density_positions[static_cast<std::size_t>(oi)] =
                outer_pos;
        }
    }
    std::uint64_t triples_considered = 0;
    std::uint64_t triples_possible = 0;
    std::uint64_t quartets_considered = 0;
    // Separate bra/ket caches keep both returned pointers valid even on a
    // hash collision. At most 8 MiB of pair storage per worker; eviction
    // recomputes the same data. The exhaustive comparison path bypasses it.
    std::vector<PrimitivePairCache> bra_pairs, ket_pairs;
    if (sparse) {
        for (int t = 0; t < n_threads; ++t) {
            bra_pairs.emplace_back(shells_ref.max_nprim(), prototype.precision(),
                                   prototype.screening_method());
            ket_pairs.emplace_back(shells_ref.max_nprim(), prototype.precision(),
                                   prototype.screening_method());
        }
    }
    // A shell pair, rather than a cell maximum, is the unit of the sparse
    // join. J joins a density-weighted ket pair with a bra-specific lattice
    // sphere. K joins two Schwarz-ordered pair lists through sigma = g+t.
    // The final cell-union and quartet tests remain the direct tests below.
    struct ShellPairEntry {
        int cell;
        std::size_t a, b;
        double q, density, log_weight;
    };
    bool pair_traversal = sparse && (charge_bounds.enabled() || opts.pair_complete_1e)
        && n_c > 0 && n_c <= 16384
        && nshells > 0 && nshells <= 256
        && gamma_density == nullptr && !density_derivative
        && bipolar_skip_mask.empty();
    const bool pair_traversal_attempted = pair_traversal;
    const double pair_density_max = pair_traversal && !d_block_max.empty()
        ? *std::max_element(d_block_max.begin(), d_block_max.end()) : 0.0;
    std::vector<ShellPairEntry> coulomb_pairs;
    std::vector<std::vector<ShellPairEntry>> exchange_pairs;
    if (pair_traversal) {
        exchange_pairs.resize(nshells);
        constexpr std::size_t pair_bytes_limit = 64 * 1024 * 1024;
        std::size_t pair_bytes = 0;
        const auto append_pair = [&](auto& list, const ShellPairEntry& entry) {
            if (list.size() == list.capacity()) {
                const auto capacity = std::max<std::size_t>(16, 2 * list.capacity());
                // Account for both old and new buffers during reallocation.
                if (capacity * sizeof(ShellPairEntry) > pair_bytes_limit - pair_bytes)
                    return false;
                const auto old = list.capacity();
                list.reserve(capacity);
                pair_bytes += (list.capacity() - old) * sizeof(ShellPairEntry);
            }
            list.push_back(entry);
            return true;
        };
        const double max_q = q_order.empty() ? 0.0 : Q_max[q_order.front()];
        for (std::size_t h = 0; h < n_c && pair_traversal; ++h) {
            const int p = p_cell_index.at(cells[h].index);
            for (std::size_t a = 0; a < nshells && pair_traversal; ++a) {
                for (std::size_t b = 0; b < nshells; ++b) {
                    if (opts.pair_complete_1e &&
                        !pair_in_range(shells_ref[a], shells_at[h][b], opts.cutoff_bohr)) continue;
                    const double q = Q[h][a * nshells + b];
                    if (q <= 0.0) continue;
                    if (compute_exchange && q * max_q * pair_density_max * 0.5 >= schwarz_thr) {
                        if (!append_pair(exchange_pairs[a], ShellPairEntry{
                                static_cast<int>(h), a, b, q, 0.0, 0.0})) {
                            pair_traversal = false;
                            break;
                        }
                    }
                    if (p >= 0) {
                        const double den = Dpair_per_block[p](a, b);
                        if (den > 0.0 && max_q * q * den >= schwarz_thr) {
                            if (!append_pair(coulomb_pairs, ShellPairEntry{
                                    static_cast<int>(h), a, b, q, den,
                                    lnQ[h][a * nshells + b] + lnDpair_per_block[p](a, b)})) {
                                pair_traversal = false;
                                break;
                            }
                        }
                    }
                }
            }
        }
        if (pair_traversal) {
            std::sort(coulomb_pairs.begin(), coulomb_pairs.end(), [](const auto& a, const auto& b) {
                return a.log_weight > b.log_weight;
            });
            for (auto& row : exchange_pairs)
                std::sort(row.begin(), row.end(), [](const auto& a, const auto& b) { return a.q > b.q; });
        } else {
            std::vector<ShellPairEntry>().swap(coulomb_pairs);
            std::vector<std::vector<ShellPairEntry>>().swap(exchange_pairs);
        }
    }
    const auto cell_possible = [&](int g, int lam, int sig, int h, int kh) {
        if (!screen) return true;
        bool j = h >= 0 && Q_max[g] * Q_max[h] * D_max >= schwarz_thr;
        bool k = compute_exchange && kh >= 0
            && Q_max[lam] * Q_max[kh] * D_max * 0.5 >= schwarz_thr;
        return j || k;
    };
    std::vector<std::uint8_t> pair_outputs_done(n_out, 0);
    std::vector<std::uint8_t> output_traversal(n_out, 0);
    std::uint64_t pair_fallback_outputs =
        pair_traversal_attempted && !pair_traversal ? n_out : 0;
    if (pair_traversal) {
        const auto origin = [](const libint2::Shell& shell) {
            return Eigen::Vector3d(shell.O[0], shell.O[1], shell.O[2]);
        };
        double query_slack = coordinate_slack;
        for (const auto& shell : shells_ref)
            query_slack = std::max(query_slack, 1e-10 * (1.0 + origin(shell).norm()));
        constexpr std::size_t max_work = 8 * 1024 * 1024;
        std::vector<std::vector<std::uint64_t>> pair_work(n_threads);
        // Each task owns one output shell pair and its complete AO sums.
        // Batching output cells keeps several expensive diffuse pairs ready
        // at once, without allocating a gate table for every output cell.
        const int n_pairs = static_cast<int>(nshells * nshells);
        const std::size_t gate_words = (n_c * n_c + 31) / 32;
        constexpr std::size_t gate_bytes_limit = 64 * 1024 * 1024;
        const int batch_width = static_cast<int>(std::min<std::size_t>(n_out,
            std::max<std::size_t>(1, gate_bytes_limit / (sizeof(std::uint64_t) * gate_words))));
        for (int batch_begin = 0; batch_begin < n_out; batch_begin += batch_width) {
            const int batch_size = std::min(batch_width, n_out - batch_begin);
            // Two atomic bits memoize the historical J/K union gate.
            std::vector<std::atomic<std::uint64_t>> gates(batch_size * gate_words);
            for (auto& word : gates) word.store(0, std::memory_order_relaxed);
            std::vector<std::atomic<bool>> complete_outputs(batch_size);
            std::vector<std::atomic<std::uint64_t>> seen_counts(batch_size);
            std::vector<std::atomic<std::uint64_t>> possible_counts(batch_size);
            std::vector<std::atomic<std::uint64_t>> visited_counts(batch_size);
            for (int local = 0; local < batch_size; ++local) {
                complete_outputs[local].store(true, std::memory_order_relaxed);
                seen_counts[local].store(0, std::memory_order_relaxed);
                possible_counts[local].store(0, std::memory_order_relaxed);
                visited_counts[local].store(0, std::memory_order_relaxed);
            }
            #pragma omp parallel for schedule(dynamic)
            for (int task = 0; task < batch_size * n_pairs; ++task) {
                const int local = task / n_pairs;
                const int oi = batch_begin + local;
                const int pair = task % n_pairs;
                auto& output_complete = complete_outputs[local];
                if (!output_complete.load(std::memory_order_relaxed)) continue;
                const int c_g = use_subset ? output_indices[oi] : oi;
                const auto* mask_oi = use_shell_mask ? &output_shell_masks[oi] : nullptr;
                const auto s1 = static_cast<std::size_t>(pair) / nshells;
                const auto s2 = static_cast<std::size_t>(pair) % nshells;
                if (mask_oi && (*mask_oi)[pair] == 0) continue;
                const auto& shells_g = shells_at[c_g];
                if (opts.pair_complete_1e && !compute_exchange &&
                    !pair_in_range(shells_ref[s1], shells_g[s2], opts.cutoff_bohr)) continue;
                auto& J_g = J_set.blocks[c_g];
                auto& K_g = K_set.blocks[c_g];
                std::uint64_t seen = 0, possible = 0, visited = 0;
                const auto tid = static_cast<std::size_t>(omp_thread_index());
                auto& engine = engines[tid];
                const auto& buf = engine.results();
                const auto admitted_cell = [&](int lam, int sig) {
                    const auto flat = static_cast<std::size_t>(lam) * n_c + sig;
                    const unsigned shift = (flat % 32) * 2;
                    auto& word = gates[local * gate_words + flat / 32];
                    auto state = (word.load(std::memory_order_relaxed) >> shift) & 3;
                    if (state == 0) {
                        const int h = cell_pos_table.at(cells[sig].index - cells[lam].index);
                        const int kh = compute_exchange
                            ? cell_pos_table.at(cells[sig].index - cells[c_g].index) : -1;
                        state = cell_possible(c_g, lam, sig, h, kh) ? 2 : 1;
                        const auto prior = word.fetch_or(state << shift, std::memory_order_relaxed);
                        if (((prior >> shift) & 3) == 0) {
                            ++seen;
                            possible += state == 2;
                        }
                    }
                    return state == 2;
                };
                // At most 64 MiB of entries per worker. A full buffer marks
                // this output for a complete restart through the direct path.
                auto& work = pair_work[tid];
                work.clear();
                bool complete = true;
                const auto append = [&](int lam, int sig, std::size_t s3,
                                        std::size_t s4, bool exchange) {
                    if (!physical_quartet(c_g, lam, sig, s1, s2, s3, s4, exchange)) return true;
                    if (work.size() == max_work) return false;
                    // Reserve once per active worker, without initializing
                    // unused entries. Reuse avoids repeated growth copies
                    // and keeps transient storage within the same budget.
                    if (work.capacity() == 0) work.reserve(max_work);
                    work.push_back(((((static_cast<std::uint64_t>(lam) * n_c + sig)
                        * nshells + s3) * nshells + s4) << 1) | exchange);
                    return true;
                };
                const auto& a = shells_ref[s1];
                const auto& b = shells_g[s2];
                const double q12 = Q[c_g][s1 * nshells + s2];
                for (const auto& ket : coulomb_pairs) {
                    // Log ordering avoids underflow in very diffuse
                    // density-weighted pairs. Keep the existing log margin
                    // and test the original product association below.
                    if (lnQ[c_g][s1 * nshells + s2] + ket.log_weight
                            < ln_thr - LN_STAGE1_MARGIN) break;
                    const double q = q12 * ket.q * ket.density;
                    if (q < schwarz_thr) continue;
                    const auto s3 = ket.a, s4 = ket.b;
                    const int h = ket.cell;
                    // C13(lambda) C24(sigma-g) bounds the integral. The
                    // second factor never exceeds its zero-distance ceiling.
                    double radius = std::numeric_limits<double>::infinity();
                    if (charge_bounds.enabled()) {
                        const double required = schwarz_thr * (1.0 - 1e-12) /
                            (ket.density * charge_bounds.ceiling(s2, s4));
                        radius = charge_bounds.radius(s1, s3, required);
                    }
                    if (radius < 0.0) continue;
                    Eigen::Vector3d center = origin(shells_ref[s1]) - origin(shells_ref[s3]);
                    if (product_bounds.enabled()) {
                        const auto& bra_product = product_bounds.at(c_g, s1, s2);
                        const auto& ket_product = product_bounds.at(h, s3, s4);
                        const double product_radius = ErfcProductBounds::query_radius(
                            bra_product, ket_product, omega, std::log(ket.density), ln_thr - 1e-12);
                        // Either necessary bound encloses every accepted term.
                        // Query the smaller sphere, then test both bounds.
                        if (product_radius < radius) {
                            radius = product_radius;
                            center = bra_product.center - ket_product.center;
                        }
                    }
                    if (opts.pair_complete_1e && interaction_cutoff < radius) {
                        radius = interaction_cutoff;
                        center = 0.5 * (origin(a) + origin(b) - origin(shells_ref[s3])
                                        - origin(shells_at[h][s4]));
                    }
                    complete = range_index.visit(center, radius + query_slack, [&](int lam) {
                        const int sig = cell_pos_table.at(cells[lam].index + cells[h].index);
                        if (sig < 0 || !admitted_cell(lam, sig)) return true;
                        ++visited;
                        if (charge_skip(c_g, lam, sig, s1, s2, s3, s4, ket.density, false)) return true;
                        return append(lam, sig, s3, s4, false);
                    });
                    if (!complete) break;
                }
                if (!complete) { output_complete.store(false); continue; }
                if (compute_exchange) {
                    const auto& bras = exchange_pairs[s1];
                    const auto& kets = exchange_pairs[s2];
                    const double max_q24 = kets.empty() ? 0.0 : kets.front().q;
                    for (const auto& bra : bras) {
                        if (bra.q * max_q24 * pair_density_max * 0.5 < schwarz_thr) break;
                        const int lam = bra.cell;
                        const auto s3 = bra.b;
                        const int p_minus_g = cell_pos_table.at(cells[lam].index - cells[c_g].index);
                        const double c23 = charge_bounds.enabled()
                            ? charge_bounds.at(p_minus_g, s2, s3) : 0.0;
                        for (std::size_t s4 = 0; s4 < nshells; ++s4) {
                            // The cross-pair bound is C14(sigma) C23(lambda-g).
                            double radius = charge_bounds.enabled()
                                ? charge_bounds.radius(s1, s4,
                                    schwarz_thr * (1.0 - 1e-12) / (0.5 * pair_density_max * c23))
                                : std::numeric_limits<double>::infinity();
                            if (radius < 0.0) continue;
                            Eigen::Vector3d center = origin(a) - origin(shells_ref[s4]);
                            if (opts.pair_complete_1e && opts.cutoff_bohr < radius) {
                                radius = opts.cutoff_bohr;
                                center = origin(b) - origin(shells_ref[s4]);
                            }
                            if (opts.pair_complete_1e && 2.0 * interaction_cutoff < radius) {
                                radius = 2.0 * interaction_cutoff;
                                center = origin(a) + origin(shells_at[lam][s3])
                                       - origin(b) - origin(shells_ref[s4]);
                            }
                            const auto accept_sigma = [&](int sig) {
                                const int t = cell_pos_table.at(cells[sig].index - cells[c_g].index);
                                if (t < 0) return true;
                                const int p = p_block_idx(lam, sig);
                                if (p < 0) return true;
                                const double den = Dpair_per_block[p](s3, s4);
                                if (bra.q * Q[t][s2 * nshells + s4] * den * 0.5 < schwarz_thr ||
                                    !admitted_cell(lam, sig)) return true;
                                ++visited;
                                if (charge_skip(c_g, lam, sig, s1, s2, s3, s4, den, true)) return true;
                                return append(lam, sig, s3, s4, true);
                            };
                            if (product_bounds.indexed()) {
                                complete = product_bounds.visit(s2, s4,
                                    product_bounds.at(lam, s1, s3), cells[c_g].r_cart,
                                    omega, std::log(0.5 * pair_density_max), ln_thr - 1e-12,
                                    center - cells[c_g].r_cart, radius + query_slack, cells,
                                    [&](int t) {
                                        const int sig = cell_pos_table.at(cells[t].index + cells[c_g].index);
                                        return sig < 0 || accept_sigma(sig);
                                    });
                            } else {
                                complete = range_index.visit(center, radius + query_slack, accept_sigma);
                            }
                            if (!complete) break;
                        }
                        if (!complete) break;
                    }
                }
                if (!complete) { output_complete.store(false); continue; }
                // For each output AO pair, restore lambda/sigma/s3/s4
                // order before contraction. J and K have separate sums.
                std::sort(work.begin(), work.end());
                for (auto encoded : work) {
                    const bool exchange = (encoded & 1) != 0;
                    auto key = encoded >> 1;
                    const auto s4 = key % nshells; key /= nshells;
                    const auto s3 = key % nshells; key /= nshells;
                    const auto sig = key % n_c;
                    const auto lam = key / n_c;
                    const auto& c = shells_at[lam][s3];
                    const auto& d = shells_at[sig][s4];
                    const auto& density = P_real_space.blocks[p_block_idx(lam, sig)];
                    const auto n1 = a.size(), n2 = b.size(), n3 = c.size(), n4 = d.size();
                    const auto bf1 = shell2bf[s1], bf2 = shell2bf[s2];
                    const auto bf3 = shell2bf[s3], bf4 = shell2bf[s4];
                    if (exchange) {
                        if (omega == 0.0) engine.compute(a, c, b, d);
                        else engine.compute2<libint2::Operator::erfc_coulomb, libint2::BraKet::xx_xx, 0>(a, c, b, d,
                            bra_pairs[tid].get(n_c * nshells + s1, lam * nshells + s3, a, c),
                            ket_pairs[tid].get(c_g * nshells + s2, sig * nshells + s4, b, d));
                        if (const double* block = buf[0])
                            for (std::size_t i = 0; i < n1; ++i)
                            for (std::size_t k = 0; k < n3; ++k)
                            for (std::size_t j = 0; j < n2; ++j)
                            for (std::size_t l = 0; l < n4; ++l)
                                K_g(bf1 + i, bf2 + j) += density(bf3 + k, bf4 + l)
                                    * block[((i * n3 + k) * n2 + j) * n4 + l];
                    } else {
                        if (omega == 0.0) engine.compute(a, b, c, d);
                        else engine.compute2<libint2::Operator::erfc_coulomb, libint2::BraKet::xx_xx, 0>(a, b, c, d,
                            bra_pairs[tid].get(n_c * nshells + s1, c_g * nshells + s2, a, b),
                            ket_pairs[tid].get(lam * nshells + s3, sig * nshells + s4, c, d));
                        if (const double* block = buf[0])
                            for (std::size_t i = 0; i < n1; ++i)
                            for (std::size_t j = 0; j < n2; ++j)
                            for (std::size_t k = 0; k < n3; ++k)
                            for (std::size_t l = 0; l < n4; ++l)
                                J_g(bf1 + i, bf2 + j) += density(bf3 + k, bf4 + l)
                                    * block[((i * n2 + j) * n3 + k) * n4 + l];
                    }
                }
                seen_counts[local].fetch_add(seen, std::memory_order_relaxed);
                possible_counts[local].fetch_add(possible, std::memory_order_relaxed);
                visited_counts[local].fetch_add(visited, std::memory_order_relaxed);
            }
            for (int local = 0; local < batch_size; ++local) {
                const int oi = batch_begin + local;
                if (complete_outputs[local].load(std::memory_order_relaxed)) {
                    pair_outputs_done[oi] = 1;
                    output_traversal[oi] = 1;
                    triples_considered += seen_counts[local].load(std::memory_order_relaxed);
                    triples_possible += possible_counts[local].load(std::memory_order_relaxed);
                    quartets_considered += visited_counts[local].load(std::memory_order_relaxed);
                } else {
                    // Workers have joined before incomplete outputs restart.
                    ++pair_fallback_outputs;
                    const int c_g = use_subset ? output_indices[oi] : oi;
                    J_set.blocks[c_g].setZero();
                    K_set.blocks[c_g].setZero();
                }
            }
        }
    }
    #pragma omp parallel for schedule(dynamic) reduction(+:triples_considered,triples_possible,quartets_considered)
    for (int oi = 0; oi < n_out; ++oi) {
        if (pair_outputs_done[oi]) continue;
        const int c_g = use_subset ? output_indices[oi] : oi;
        const std::vector<uint8_t>* mask_oi =
            use_shell_mask ? &output_shell_masks[oi] : nullptr;
        const auto tid = static_cast<std::size_t>(omp_thread_index());
        auto& engine = engines[tid];
        const auto& buf = engine.results();

        const auto& shells_g = shells_at[c_g];
        Eigen::MatrixXd& J_g = J_set.blocks[c_g];
        Eigen::MatrixXd& K_g = K_set.blocks[c_g];
        const Eigen::MatrixXd* P_outer = gamma_density;
        if (density_derivative) {
            P_outer = &P_real_space.blocks[
                static_cast<std::size_t>(
                    output_density_positions[static_cast<std::size_t>(oi)])];
        }

        std::vector<std::pair<int, int>> candidates;
        constexpr std::size_t max_candidates = 262144;
        const bool use_candidates = sparse && candidates_for(c_g, [&](int lam, int sig) {
            if (candidates.size() == max_candidates) return false;
            candidates.emplace_back(lam, sig);
            return true;
        });
        // Keep the original lambda/sigma accumulation order and the UNION
        // of J/K possibilities. The legacy cell test gates the whole
        // quartet loop, not J and K independently.
        if (use_candidates) {
            std::sort(candidates.begin(), candidates.end());
            candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
        }
        std::vector<std::uint64_t> candidate_bits;
        std::size_t bitmap_count = 0;
        if (sparse && !use_candidates && n_c <= 4096) {
            // A periodic density need not decay between images. When its
            // support fills the pair list, a bitmap still represents the
            // entire pair domain within the same 2 MiB scratch budget.
            // Release the list first; duplicate J/K hits consume one bit.
            std::vector<std::pair<int, int>>().swap(candidates);
            candidate_bits.assign((n_c * n_c + 63) / 64, 0);
            candidates_for(c_g, [&](int lam, int sig) {
                const std::size_t index = static_cast<std::size_t>(lam) * n_c + sig;
                const std::uint64_t mask = std::uint64_t{1} << (index % 64);
                auto& word = candidate_bits[index / 64];
                bitmap_count += (word & mask) == 0;
                word |= mask;
                return true;
            });
        }
        const bool use_bitmap = !candidate_bits.empty();
        output_traversal[oi] = (use_candidates || use_bitmap) ? 2 : 3;
        const std::size_t count = use_candidates ? candidates.size()
            : use_bitmap ? bitmap_count : n_c * n_c;
        triples_considered += count;
        std::size_t bitmap_word = 0;
        for (std::size_t pair = 0; pair < count; ++pair) {
            std::size_t flat_pair = pair;
            if (use_bitmap) {
                while (candidate_bits[bitmap_word] == 0) ++bitmap_word;
                auto& word = candidate_bits[bitmap_word];
                flat_pair = bitmap_word * 64 + __builtin_ctzll(word);
                word &= word - 1;
            }
            const std::size_t c_lam = use_candidates
                ? static_cast<std::size_t>(candidates[pair].first) : flat_pair / n_c;
            const std::size_t c_sig = use_candidates
                ? static_cast<std::size_t>(candidates[pair].second) : flat_pair % n_c;
            const auto& shells_lam = shells_at[c_lam];
            const auto& shells_sig = shells_at[c_sig];

            const int p_idx = p_block_idx(
                static_cast<int>(c_lam), static_cast<int>(c_sig));
            if (p_idx < 0) {
                continue;  // h = g_σ − g_λ outside density cutoff
            }
            const Eigen::MatrixXd* P_block =
                &P_real_space.blocks[static_cast<std::size_t>(p_idx)];
            const std::vector<uint8_t>* density_mask =
                use_density_mask
                ? &density_shell_masks[static_cast<std::size_t>(p_idx)]
                : nullptr;

            // Index of the c_h = c_σ − c_λ cell in `cells`, for the J
            // Schwarz bound. (Same displacement that p_block uses to
            // index P_real_space; here we want it inside `cells`.)
            int c_h_idx = -1;
            if (screen) {
                const Eigen::Vector3i h =
                    cells[c_sig].index - cells[c_lam].index;
                c_h_idx = cell_pos_table.at(h);
            }
            // Cell index for the K bound's second pair displacement
            // (c_σ − c_g): the K integral pairs (s2_g, s4_σ) so the
            // shell-pair displacement seen by Q is c_σ − c_g.
            int c_kh_idx = -1;
            if (screen && compute_exchange) {
                const Eigen::Vector3i kh =
                    cells[c_sig].index - cells[c_g].index;
                c_kh_idx = cell_pos_table.at(kh);
            }

            if (!cell_possible(c_g, c_lam, c_sig, c_h_idx, c_kh_idx)) continue;
            ++triples_possible;

            // For each AO shell quartet, compute J and K integrals.
            // Bipolar dispatch: pre-build per-(c_g, c_lam) skip lookup.
            const bool use_bipolar = !bipolar_skip_mask.empty();
            const std::vector<int>* bp_skip_vec = nullptr;
            if (use_bipolar) {
                int bp_idx = static_cast<int>(c_g) * n_c_i
                    + static_cast<int>(c_lam);
                if (bp_idx >= 0 &&
                    bp_idx < static_cast<int>(bipolar_skip_mask.size())) {
                    bp_skip_vec = &bipolar_skip_mask[
                        static_cast<std::size_t>(bp_idx)];
                }
            }

            for (std::size_t s1 = 0; s1 < shells_ref.size(); ++s1) {
                const auto bf1 = shell2bf[s1];
                const auto n1 = shells_ref[s1].size();
                for (std::size_t s2 = 0; s2 < shells_g.size(); ++s2) {
                    const auto bf2 = shell2bf[s2];
                    const auto n2 = shells_g[s2].size();
                    // SYM3b shell-pair mask: skip output shell pairs
                    // (s1_home, s2_cg) that are not orbit representatives;
                    // the caller reconstructs them by point-group rotation.
                    if (mask_oi && (*mask_oi)[s1 * nshells + s2] == 0)
                        continue;
                    // Schwarz factor for the (s1_0, s2_g) pair.
                    const double q12 = screen
                        ? Q[c_g][s1 * nshells + s2]
                        : 0.0;
                    double k_row = 0.0;
                    if (sparse) {
                        const double j_max = c_h_idx >= 0 ? Q_max[c_h_idx] : 0.0;
                        k_row = compute_exchange && c_kh_idx >= 0
                            ? q_row_max[c_kh_idx][s2] : 0.0;
                        const double den = d_block_max[p_idx];
                        if (q12 * j_max * den < schwarz_thr &&
                            q_row_max[c_lam][s1] * k_row * den * 0.5 < schwarz_thr)
                            continue;
                    }
                    for (std::size_t s3 = 0; s3 < shells_lam.size(); ++s3) {
                        const auto bf3 = shell2bf[s3];
                        const auto n3 = shells_lam[s3].size();
                        // Schwarz factor for the (s1_0, s3_λ) pair —
                        // K integral's first half (relative displacement
                        // = c_λ).
                        const double q13 = (screen && compute_exchange)
                            ? Q[c_lam][s1 * nshells + s3]
                            : 0.0;
                        if (sparse) {
                            const double j_row = c_h_idx >= 0
                                ? q_row_max[c_h_idx][s3] : 0.0;
                            const double den = d_row_max[p_idx][s3];
                            bool j_prefix = q12 * j_row * den >= schwarz_thr;
                            bool k_prefix = q13 * k_row * den * 0.5 >= schwarz_thr;
                            if (!j_prefix && !k_prefix) continue;
                        }
                        for (std::size_t s4 = 0; s4 < shells_sig.size(); ++s4) {
                            ++quartets_considered;
                            const auto bf4 = shell2bf[s4];
                            const auto n4 = shells_sig[s4].size();

                            // ---- Dormant quartet-prototype far-path skip --
                            // Quartets in the skip mask are far-field;
                            // their contribution is computed by the Python
                            // multipole contractor and added separately.
                            if (bp_skip_vec && !bp_skip_vec->empty()) {
                                const int q_flat = static_cast<int>(
                                    ((s1 * nshells + s2) * nshells + s3)
                                        * nshells + s4);
                                if (std::binary_search(
                                        bp_skip_vec->begin(),
                                        bp_skip_vec->end(), q_flat)) {
                                    continue;  // far-field: skip exact ERI
                                }
                            }

                            // -- J piece -----------------------------
                            bool do_J = physical_quartet(c_g, c_lam, c_sig, s1, s2, s3, s4, false);
                            if (screen) {
                                if (c_h_idx < 0) {
                                    do_J = false;
                                } else {
                                    const double q34 =
                                        Q[c_h_idx][s3 * nshells + s4];
                                    // Sx3: shell-pair density max from
                                    // the local density block (the
                                    // outer triple-level p_idx — it is
                                    // >= 0 here by the `continue`
                                    // above; the previous per-quartet
                                    // p_block_idx recomputation was
                                    // loop-invariant waste).
                                    const double den12 =
                                        Dpair_per_block[p_idx](
                                            static_cast<Eigen::Index>(s3),
                                            static_cast<Eigen::Index>(s4));
                                    if (q12 * q34 * den12 < schwarz_thr)
                                        do_J = false;
                                    else if (charge_skip(c_g, c_lam, c_sig,
                                            s1, s2, s3, s4, den12, false))
                                        do_J = false;
                                }
                            }
                            if (do_J) {
                                if (sparse) {
                                    engine.compute2<libint2::Operator::erfc_coulomb,
                                        libint2::BraKet::xx_xx, 0>(
                                        shells_ref[s1], shells_g[s2], shells_lam[s3], shells_sig[s4],
                                        bra_pairs[tid].get(n_c * nshells + s1,
                                            c_g * nshells + s2, shells_ref[s1], shells_g[s2]),
                                        ket_pairs[tid].get(c_lam * nshells + s3,
                                            c_sig * nshells + s4, shells_lam[s3], shells_sig[s4]));
                                } else {
                                    engine.compute(shells_ref[s1], shells_g[s2],
                                                   shells_lam[s3], shells_sig[s4]);
                                }
                                if (const double* blk = buf[0]) {
                                    for (std::size_t i = 0; i < n1; ++i)
                                    for (std::size_t j = 0; j < n2; ++j)
                                    for (std::size_t k = 0; k < n3; ++k)
                                    for (std::size_t l = 0; l < n4; ++l) {
                                        const double v = blk[
                                            ((i * n2 + j) * n3 + k) * n4 + l];
                                        J_g(bf1 + i, bf2 + j) +=
                                            (*P_block)(bf3 + k, bf4 + l) * v;
                                        if (gamma_density != nullptr &&
                                            (!density_mask ||
                                             (*density_mask)[
                                                 s3 * nshells + s4] != 0)) {
                                            J_adjoint_tls[tid](
                                                bf3 + k, bf4 + l) +=
                                                (*P_outer)(bf1 + i,
                                                           bf2 + j) * v;
                                        }
                                        if (density_derivative &&
                                            (!density_mask ||
                                             (*density_mask)[
                                                 s3 * nshells + s4] != 0)) {
                                            J_density_adjoint_tls[tid][
                                                static_cast<std::size_t>(p_idx)](
                                                    bf3 + k, bf4 + l) +=
                                                (*P_outer)(bf1 + i,
                                                           bf2 + j) * v;
                                        }
                                    }
                                }
                            }

                            // -- K piece (exchange) -----------------
                            if (compute_exchange) {
                                bool do_K = physical_quartet(c_g, c_lam, c_sig, s1, s2, s3, s4, true);
                                if (screen) {
                                    if (c_kh_idx < 0) {
                                        do_K = false;
                                    } else {
                                        const double q24 =
                                            Q[c_kh_idx][s2 * nshells + s4];
                                        // Sx3: shell-pair density
                                        // envelope for P(λ,σ) — the
                                        // density element that multiplies
                                        // (μ_0 λ_λ | ν_g σ_σ) in the K
                                        // build (outer p_idx >= 0 as
                                        // above).
                                        const double den34 =
                                            Dpair_per_block[p_idx](
                                                static_cast<Eigen::Index>(s3),
                                                static_cast<Eigen::Index>(s4));
                                        if (q13 * q24 * den34 * 0.5
                                                < schwarz_thr)
                                            do_K = false;
                                        else if (charge_skip(c_g, c_lam, c_sig,
                                                s1, s2, s3, s4, den34, true))
                                            do_K = false;
                                    }
                                }
                                if (do_K) {
                                    if (sparse) {
                                        engine.compute2<libint2::Operator::erfc_coulomb,
                                            libint2::BraKet::xx_xx, 0>(
                                            shells_ref[s1], shells_lam[s3], shells_g[s2], shells_sig[s4],
                                            bra_pairs[tid].get(n_c * nshells + s1,
                                                c_lam * nshells + s3, shells_ref[s1], shells_lam[s3]),
                                            ket_pairs[tid].get(c_g * nshells + s2,
                                                c_sig * nshells + s4, shells_g[s2], shells_sig[s4]));
                                    } else {
                                        engine.compute(shells_ref[s1], shells_lam[s3],
                                                       shells_g[s2], shells_sig[s4]);
                                    }
                                    if (const double* blk = buf[0]) {
                                        for (std::size_t i = 0; i < n1; ++i)
                                        for (std::size_t j = 0; j < n2; ++j)
                                        for (std::size_t k = 0; k < n3; ++k)
                                        for (std::size_t l = 0; l < n4; ++l) {
                                            const double v = blk[
                                                ((i * n3 + k) * n2 + j) * n4 + l];
                                            K_g(bf1 + i, bf2 + j) +=
                                                (*P_block)(bf3 + k, bf4 + l) * v;
                                            if (gamma_density != nullptr &&
                                                (!density_mask ||
                                                 (*density_mask)[
                                                     s3 * nshells + s4] != 0)) {
                                                K_adjoint_tls[tid](
                                                    bf3 + k, bf4 + l) +=
                                                    (*P_outer)(bf1 + i,
                                                               bf2 + j) * v;
                                            }
                                            if (density_derivative &&
                                                (!density_mask ||
                                                 (*density_mask)[
                                                     s3 * nshells + s4] != 0)) {
                                                K_density_adjoint_tls[tid][
                                                    static_cast<std::size_t>(
                                                        p_idx)](
                                                        bf3 + k,
                                                        bf4 + l) +=
                                                    (*P_outer)(bf1 + i,
                                                               bf2 + j) * v;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Eigen::MatrixXd J_gamma;
    Eigen::MatrixXd K_gamma;
    LatticeMatrixSet J_density_derivative;
    LatticeMatrixSet K_density_derivative;
    if (gamma_density != nullptr) {
        Eigen::MatrixXd J_forward = Eigen::MatrixXd::Zero(nbf, nbf);
        Eigen::MatrixXd K_forward = Eigen::MatrixXd::Zero(nbf, nbf);
        for (const auto& block : J_set.blocks) J_forward += block;
        for (const auto& block : K_set.blocks) K_forward += block;
        Eigen::MatrixXd J_adjoint = Eigen::MatrixXd::Zero(nbf, nbf);
        Eigen::MatrixXd K_adjoint = Eigen::MatrixXd::Zero(nbf, nbf);
        for (const auto& block : J_adjoint_tls) J_adjoint += block;
        for (const auto& block : K_adjoint_tls) K_adjoint += block;
        J_gamma = 0.5 * (J_forward + J_adjoint);
        K_gamma = 0.5 * (K_forward + K_adjoint);
        // The independent Gamma variable is a real symmetric density.
        J_gamma = 0.5 * (J_gamma + J_gamma.transpose().eval());
        K_gamma = 0.5 * (K_gamma + K_gamma.transpose().eval());
    }
    if (density_derivative) {
        J_density_derivative.nbf = nbf;
        J_density_derivative.cells = P_real_space.cells;
        J_density_derivative.blocks.assign(
            P_real_space.blocks.size(), Eigen::MatrixXd::Zero(nbf, nbf));
        K_density_derivative.nbf = nbf;
        K_density_derivative.cells = P_real_space.cells;
        K_density_derivative.blocks.assign(
            P_real_space.blocks.size(), Eigen::MatrixXd::Zero(nbf, nbf));

        // Forward action: the outer density P(g) contracts only the emitted
        // Fock-output cells. The adjoint action lands on whichever P(h) block
        // the internal lambda/sigma displacement selected.
        for (int oi = 0; oi < n_out; ++oi) {
            const int c_g = use_subset ? output_indices[oi] : oi;
            const auto p = static_cast<std::size_t>(
                output_density_positions[static_cast<std::size_t>(oi)]);
            J_density_derivative.blocks[p] += 0.5 * J_set.blocks[c_g];
            K_density_derivative.blocks[p] += 0.5 * K_set.blocks[c_g];
        }
        for (std::size_t p = 0; p < P_real_space.blocks.size(); ++p) {
            for (std::size_t tid = 0;
                 tid < static_cast<std::size_t>(n_threads); ++tid) {
                J_density_derivative.blocks[p] +=
                    0.5 * J_density_adjoint_tls[tid][p];
                K_density_derivative.blocks[p] +=
                    0.5 * K_density_adjoint_tls[tid][p];
            }
        }
    }
    auto result = JKLatticeMatrixSets{
        std::move(J_set), std::move(K_set),
        std::move(J_gamma), std::move(K_gamma),
        std::move(J_density_derivative),
        std::move(K_density_derivative), triples_considered, triples_possible,
        quartets_considered};
    result.shell_pair_outputs = std::count(output_traversal.begin(), output_traversal.end(), 1);
    result.cell_sparse_outputs = std::count(output_traversal.begin(), output_traversal.end(), 2);
    result.exhaustive_outputs = std::count(output_traversal.begin(), output_traversal.end(), 3);
    result.shell_pair_fallback_outputs = pair_fallback_outputs;
    result.cell_sparse_fallback_outputs = sparse ? result.exhaustive_outputs : 0;
    result.charge_screening_available = charge_bounds.enabled();
    result.product_screening_available = product_bounds.enabled();
    return result;
}

namespace {

JKLatticeMatrixSets build_jk_automatic_domain(
        const BasisSet& basis, const PeriodicSystem& system,
        const LatticeSumOptions& opts, const LatticeMatrixSet& density,
        bool exchange, double omega, const std::vector<int>& subset = {},
        const std::vector<std::vector<uint8_t>>& masks = {}) {
    if (!opts.pair_complete_1e) {
        const auto cells = direct_lattice_cells(system, opts.cutoff_bohr);
        return build_jk_2e_real_space_impl(
            basis, system, opts, density, cells, exchange, omega, subset, masks);
    }
    const auto internal = pair_complete_eri_cells(system, opts, basis.libint());
    const auto& outputs = internal;
    const auto positions = build_cell_index_map(internal);
    std::vector<int> selected = subset;
    if (selected.empty()) {
        selected.resize(outputs.size());
        std::iota(selected.begin(), selected.end(), 0);
    }
    std::vector<int> mapped;
    mapped.reserve(selected.size());
    for (int output : selected) {
        if (output < 0 || static_cast<std::size_t>(output) >= outputs.size())
            throw std::invalid_argument("build_jk_2e_real_space: output index outside pair domain");
        mapped.push_back(positions.at(outputs[output].index));
    }
    auto result = build_jk_2e_real_space_impl(
        basis, system, opts, density, internal, exchange, omega, mapped, masks);
    // Exchange output and contracting density can extend beyond either
    // integral product. Preserve the full physical quartet enclosure.
    const auto trim = [&](LatticeMatrixSet& set) {
        std::vector<Eigen::MatrixXd> blocks;
        blocks.reserve(outputs.size());
        for (const auto& cell : outputs)
            blocks.push_back(std::move(set.blocks[positions.at(cell.index)]));
        set.cells = outputs;
        set.blocks = std::move(blocks);
    };
    trim(result.J);
    trim(result.K);
    return result;
}

}  // namespace

JKLatticeMatrixSets build_jk_2e_real_space(const BasisSet& basis,
                                           const PeriodicSystem& system,
                                           const LatticeSumOptions& opts,
                                           const LatticeMatrixSet& P_real_space,
                                           double omega) {
    return build_jk_automatic_domain(
        basis, system, opts, P_real_space, true, omega);
}

JKLatticeMatrixSets build_jk_2e_real_space_explicit(
        const BasisSet& basis,
        const PeriodicSystem& system,
        const LatticeSumOptions& opts,
        const LatticeMatrixSet& P_real_space,
        const std::vector<LatticeCell>& cells,
        double omega) {
    return build_jk_2e_real_space_impl(
        basis, system, opts, P_real_space, cells, true, omega);
}

// Phase SYM3b: symmetry-reduced output build. Builds J(g)/K(g) only for the
// bra cells in `output_indices` (point-group orbit representatives), with the
// FULL internal lattice sum over the cutoff cell list — so each emitted block
// is exact and the non-emitted blocks (left zero) are recovered by the caller
// via point-group reconstruction. The |G|-fold compute reduction the SYM3
// storage round-trip (vibeqc.symmetry_integrals) was the validation substrate
// for. `output_indices` are positions into the cutoff cell list.
JKLatticeMatrixSets build_jk_2e_real_space_output_subset(
        const BasisSet& basis,
        const PeriodicSystem& system,
        const LatticeSumOptions& opts,
        const LatticeMatrixSet& P_real_space,
        const std::vector<int>& output_indices,
        double omega) {
    return build_jk_automatic_domain(
        basis, system, opts, P_real_space, true, omega, output_indices);
}

// Phase SYM3b shell-pair mask: like build_jk_2e_real_space_output_subset, but
// builds only the atom-pair-orbit *representative* sub-blocks within each
// output (representative) cell, given a per-output-cell nshells*nshells flag
// array (`output_shell_masks[oi][s1*nshells + s2] != 0` → build that output
// shell pair). This is the finer reduction: the whole-cell subset still builds
// every shell pair of a rep cell, whereas the reconstruction only needs the
// rep atom-pair sub-blocks. Internal lattice sum stays full → each emitted
// sub-block is exact; non-emitted entries are recovered by point-group
// rotation. `output_indices` are positions into the cutoff cell list.
JKLatticeMatrixSets build_jk_2e_real_space_output_subset_masked(
        const BasisSet& basis,
        const PeriodicSystem& system,
        const LatticeSumOptions& opts,
        const LatticeMatrixSet& P_real_space,
        const std::vector<int>& output_indices,
        const std::vector<std::vector<uint8_t>>& output_shell_masks,
        double omega) {
    return build_jk_automatic_domain(
        basis, system, opts, P_real_space, true, omega, output_indices,
        output_shell_masks);
}

// Pair-resolved truncation (M1): the caller controls the internal summation
// cell list, the output subset, and the per-output shell-pair masks — the
// impl always supported them; the other wrappers hard-derive `cells` from
// opts.cutoff_bohr. See the header for the domain semantics.
JKLatticeMatrixSets build_jk_2e_real_space_domains(
        const BasisSet& basis,
        const PeriodicSystem& system,
        const LatticeSumOptions& opts,
        const LatticeMatrixSet& P_real_space,
        const std::vector<LatticeCell>& cells,
        const std::vector<int>& output_indices,
        const std::vector<std::vector<uint8_t>>& output_shell_masks,
        double omega,
        bool compute_exchange) {
    return build_jk_2e_real_space_impl(
        basis, system, opts, P_real_space, cells, compute_exchange, omega,
        output_indices, output_shell_masks);
}

JKLatticeMatrixSets build_jk_2e_real_space_bipolar_dispatch(
        const BasisSet& basis,
        const PeriodicSystem& system,
        const LatticeSumOptions& opts,
        const LatticeMatrixSet& P_real_space,
        const std::vector<std::vector<int>>& bipolar_skip_mask,
        double omega,
        bool compute_exchange) {
    // Same traversal as build_jk_2e_real_space_domains but with the
    // bipolar far-field quartet skip mask: quartets listed in
    // bipolar_skip_mask[c_g * n_c + c_lam] are SKIPPED from exact
    // ERI evaluation (their contribution is computed separately via
    // the Python multipole far-field contractor).
    //
    // Pisani-Dovesi-Roetti (1988), Ch. II.4c, is the expansion source.
    // The skip-mask mechanism and classifier are implementation prototypes.
    if (opts.pair_complete_1e) {
        throw std::invalid_argument(
            "The radial bipolar skip mask does not describe the physical pair domain");
    }
    const auto cells = direct_lattice_cells(system, opts.cutoff_bohr);
    return build_jk_2e_real_space_impl(
        basis, system, opts, P_real_space, cells, compute_exchange, omega,
        {}, /* output_indices: all cells */
        {}, /* output_shell_masks: all pairs */
        {}, /* density_shell_masks */
        nullptr, /* gamma_density */
        false, /* density_derivative */
        bipolar_skip_mask);
}

JKLatticeMatrixSets build_jk_2e_real_space_domains_gamma_derivative(
        const BasisSet& basis,
        const PeriodicSystem& system,
        const LatticeSumOptions& opts,
        const LatticeMatrixSet& P_real_space,
        const std::vector<LatticeCell>& cells,
        const std::vector<int>& output_indices,
        double omega,
        bool compute_exchange,
        const std::vector<std::vector<uint8_t>>& output_shell_masks,
        const std::vector<std::vector<uint8_t>>& density_shell_masks,
        const Eigen::MatrixXd& gamma_density) {
    if (P_real_space.blocks.empty()) {
        throw std::runtime_error(
            "build_jk_2e_real_space_domains_gamma_derivative: empty density");
    }
    Eigen::MatrixXd gamma = gamma_density;
    if (gamma.size() == 0) {
        gamma = P_real_space.blocks.front();
        for (const auto& block : P_real_space.blocks) {
            if (block.rows() != gamma.rows() || block.cols() != gamma.cols() ||
                !block.isApprox(gamma, 1.0e-12)) {
                throw std::runtime_error(
                    "build_jk_2e_real_space_domains_gamma_derivative "
                    "requires an explicit gamma_density when the real-space "
                    "density blocks are not homogeneous");
            }
        }
    }
    return build_jk_2e_real_space_impl(
        basis, system, opts, P_real_space, cells, compute_exchange, omega,
        output_indices, output_shell_masks, density_shell_masks, &gamma);
}

JKLatticeMatrixSets build_jk_2e_real_space_domains_density_derivative(
        const BasisSet& basis,
        const PeriodicSystem& system,
        const LatticeSumOptions& opts,
        const LatticeMatrixSet& P_real_space,
        const std::vector<LatticeCell>& cells,
        const std::vector<int>& output_indices,
        double omega,
        bool compute_exchange,
        const std::vector<std::vector<uint8_t>>& output_shell_masks,
        const std::vector<std::vector<uint8_t>>& density_shell_masks) {
    if (P_real_space.blocks.empty()) {
        throw std::runtime_error(
            "build_jk_2e_real_space_domains_density_derivative: empty density");
    }
    return build_jk_2e_real_space_impl(
        basis, system, opts, P_real_space, cells, compute_exchange, omega,
        output_indices, output_shell_masks, density_shell_masks, nullptr, true);
}

LatticeMatrixSet build_fock_2e_real_space(const BasisSet& basis,
                                          const PeriodicSystem& system,
                                          const LatticeSumOptions& opts,
                                          const LatticeMatrixSet& P_real_space,
                                          double exchange_scale,
                                          double omega) {
    const bool need_exchange = (exchange_scale != 0.0);
    if (omega < 0.0) {
        throw std::runtime_error(
            "build_fock_2e_real_space: omega must be non-negative");
    }
    if (!need_exchange) {
        return build_jk_automatic_domain(
            basis, system, opts, P_real_space, false, omega).J;
    }
    JKLatticeMatrixSets jk = build_jk_automatic_domain(
        basis, system, opts, P_real_space, true, omega);
    for (std::size_t c = 0; c < jk.J.blocks.size(); ++c) {
        jk.J.blocks[c] -= 0.5 * exchange_scale * jk.K.blocks[c];
    }
    return std::move(jk.J);
}

// ---- Phase SYM3b explicit-cell variant -----------------------------------
// Same kernel as build_jk_gamma_molecular_limit, but accepts a caller-
// supplied cell list.  The original entry point delegates here.
JKMatrices build_jk_gamma_molecular_limit_explicit(
        const BasisSet& basis,
        const PeriodicSystem& system,
        const std::vector<LatticeCell>& cells,
        const LatticeSumOptions& opts,
        const Eigen::MatrixXd& P,
        double omega) {
    ensure_libint_initialized();

    const auto& shells_ref = basis.libint();
    const int nbf = static_cast<int>(basis.nbasis());
    if (P.rows() != nbf || P.cols() != nbf)
        throw std::runtime_error("build_jk_gamma_molecular_limit_explicit: density shape mismatch");
    if (omega < 0.0)
        throw std::runtime_error("build_jk_gamma_molecular_limit_explicit: omega must be non-negative");

    if (opts.pair_complete_1e) {
        LatticeMatrixSet density;
        density.nbf = nbf;
        CellIndexMap positions;
        const auto add_density_cell = [&](const Eigen::Vector3i& index) {
            if (positions.emplace(index, static_cast<int>(density.cells.size())).second) {
                density.cells.push_back(LatticeCell{index, system.lattice * index.cast<double>()});
                density.blocks.push_back(P);
            }
        };
        for (const auto& cell : cells) add_density_cell(cell.index);
        for (const auto& a : cells)
            for (const auto& b : cells) add_density_cell(b.index - a.index);
        const auto lattice = build_jk_2e_real_space_impl(
            basis, system, opts, density, cells, true, omega);
        JKMatrices result{Eigen::MatrixXd::Zero(nbf, nbf), Eigen::MatrixXd::Zero(nbf, nbf)};
        for (const auto& block : lattice.J.blocks) result.J += block;
        for (const auto& block : lattice.K.blocks) result.K += block;
        result.J = 0.5 * (result.J + result.J.transpose()).eval();
        result.K = 0.5 * (result.K + result.K.transpose()).eval();
        return result;
    }

    const libint2::Operator op = (omega > 0.0)
        ? libint2::Operator::erfc_coulomb
        : libint2::Operator::coulomb;
    libint2::Engine prototype(op, shells_ref.max_nprim(), shells_ref.max_l(), 0);
    if (omega > 0.0) prototype.set_params(omega);
    auto engines = make_engine_pool(prototype);
    const auto shell2bf = shells_ref.shell2bf();
    const std::size_t nshells = shells_ref.size();

    std::vector<std::vector<libint2::Shell>> shells_at(cells.size());
    for (std::size_t c = 0; c < cells.size(); ++c)
        shells_at[c] = shift_shells(shells_ref, cells[c].r_cart);

    const double schwarz_thr = opts.schwarz_threshold;
    const bool screen = (schwarz_thr > 0.0);
    const auto Q = screen
        ? compute_schwarz_factors_per_cell(shells_ref, shells_at, prototype)
        : std::vector<std::vector<double>>{};
    const auto Q_max = screen ? max_q_per_cell(Q) : std::vector<double>{};
    const double D_max = screen
        ? std::max(1.0, std::max(std::fabs(P.maxCoeff()), std::fabs(P.minCoeff())))
        : 0.0;
    CellIndexMap cell_index_map;
    if (screen) cell_index_map = build_cell_index_map(cells);
    int c_zero_idx = -1;
    if (screen) {
        auto it = cell_index_map.find(Eigen::Vector3i(0, 0, 0));
        if (it != cell_index_map.end()) c_zero_idx = it->second;
    }
    const double q_zero_max = (screen && c_zero_idx >= 0) ? Q_max[c_zero_idx] : 0.0;

    const int n_c = static_cast<int>(cells.size());
    const int n_pairs = n_c * n_c;
    const int n_threads = omp_max_threads();
    std::vector<Eigen::MatrixXd> Jm_tls(n_threads, Eigen::MatrixXd::Zero(nbf, nbf));
    std::vector<Eigen::MatrixXd> Km_tls(n_threads, Eigen::MatrixXd::Zero(nbf, nbf));

    #pragma omp parallel for schedule(dynamic)
    for (int idx = 0; idx < n_pairs; ++idx) {
        const int c_g = idx / n_c;
        const int c_p = idx % n_c;
        const auto tid = static_cast<std::size_t>(omp_thread_index());
        auto& engine = engines[tid];
        const auto& buf = engine.results();
        auto& Jm_local = Jm_tls[tid];
        auto& Km_local = Km_tls[tid];
        const auto& shells_g = shells_at[c_g];
        const auto& shells_p = shells_at[c_p];
        int c_pg_idx = -1;
        if (screen) {
            const Eigen::Vector3i pg = cells[c_p].index - cells[c_g].index;
            auto it = cell_index_map.find(pg);
            if (it != cell_index_map.end()) c_pg_idx = it->second;
        }
        if (screen) {
            const bool j_possible = (Q_max[c_g] * q_zero_max * D_max >= schwarz_thr);
            const double q_pg_max = (c_pg_idx >= 0) ? Q_max[c_pg_idx] : 0.0;
            const bool k_possible = (Q_max[c_p] * q_pg_max * D_max >= schwarz_thr);
            if (!j_possible && !k_possible) continue;
        }
        for (std::size_t s1 = 0; s1 < shells_ref.size(); ++s1) {
            const auto bf1 = shell2bf[s1]; const auto n1 = shells_ref[s1].size();
            for (std::size_t s2 = 0; s2 < shells_g.size(); ++s2) {
                const auto bf2 = shell2bf[s2]; const auto n2 = shells_g[s2].size();
                const double q12_J = screen ? Q[c_g][s1 * nshells + s2] : 0.0;
                for (std::size_t s3 = 0; s3 < shells_p.size(); ++s3) {
                    const auto bf3 = shell2bf[s3]; const auto n3 = shells_p[s3].size();
                    const double q13_K = screen ? Q[c_p][s1 * nshells + s3] : 0.0;
                    for (std::size_t s4 = 0; s4 < shells_p.size(); ++s4) {
                        const auto bf4 = shell2bf[s4]; const auto n4 = shells_p[s4].size();
                        bool do_J = true;
                        if (screen) {
                            if (c_zero_idx < 0) { do_J = false; }
                            else {
                                const double q34_J = Q[c_zero_idx][s3 * nshells + s4];
                                if (q12_J * q34_J * D_max < schwarz_thr) do_J = false;
                            }
                        }
                        if (do_J) {
                            engine.compute(shells_ref[s1], shells_g[s2], shells_p[s3], shells_p[s4]);
                            if (const double* blk = buf[0]) {
                                for (std::size_t i = 0; i < n1; ++i)
                                for (std::size_t j = 0; j < n2; ++j)
                                for (std::size_t k = 0; k < n3; ++k)
                                for (std::size_t l = 0; l < n4; ++l) {
                                    const double v = blk[((i * n2 + j) * n3 + k) * n4 + l];
                                    Jm_local(bf1+i, bf2+j) += P(bf3+k, bf4+l) * v;
                                }
                            }
                        }
                        bool do_K = true;
                        if (screen) {
                            if (c_pg_idx < 0) { do_K = false; }
                            else {
                                const double q24_K = Q[c_pg_idx][s2 * nshells + s4];
                                if (q13_K * q24_K * D_max < schwarz_thr) do_K = false;
                            }
                        }
                        if (do_K) {
                            engine.compute(shells_ref[s1], shells_p[s3], shells_g[s2], shells_p[s4]);
                            if (const double* blk = buf[0]) {
                                for (std::size_t i = 0; i < n1; ++i)
                                for (std::size_t j = 0; j < n2; ++j)
                                for (std::size_t k = 0; k < n3; ++k)
                                for (std::size_t l = 0; l < n4; ++l) {
                                    const double v = blk[((i * n3 + k) * n2 + j) * n4 + l];
                                    Km_local(bf1+i, bf2+j) += P(bf3+k, bf4+l) * v;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Eigen::MatrixXd Jm = Eigen::MatrixXd::Zero(nbf, nbf);
    Eigen::MatrixXd Km = Eigen::MatrixXd::Zero(nbf, nbf);
    for (const auto& m : Jm_tls) Jm += m;
    for (const auto& m : Km_tls) Km += m;
    return JKMatrices{Jm, Km};
}

// ---- Phase M3b per-cell-pair J/K contributions ---------------------------
// Same Γ-only molecular-limit shell-quartet loop as
// build_jk_gamma_molecular_limit_explicit, but each (c_g, c_p) pair's J and K
// contributions are stored separately instead of being summed. Parallelised
// over the supplied pair list; each thread writes to its own output slot, so
// no reduction or locking is needed.
std::vector<PairJKContribution> build_jk_pair_contributions(
        const BasisSet& basis,
        const PeriodicSystem& system,
        const std::vector<LatticeCell>& cells,
        const std::vector<std::pair<int, int>>& pairs,
        const LatticeSumOptions& opts,
        const Eigen::MatrixXd& P,
        double omega) {
    if (opts.pair_complete_1e) {
        throw std::invalid_argument(
            "build_jk_pair_contributions: physical quartet support requires "
            "the three-image domains API; the two-image decomposition is unsupported");
    }
    ensure_libint_initialized();

    const auto& shells_ref = basis.libint();
    const int nbf = static_cast<int>(basis.nbasis());
    if (P.rows() != nbf || P.cols() != nbf)
        throw std::runtime_error("build_jk_pair_contributions: density shape mismatch");
    if (omega < 0.0)
        throw std::runtime_error("build_jk_pair_contributions: omega must be non-negative");

    const int n_c = static_cast<int>(cells.size());
    for (const auto& pr : pairs) {
        if (pr.first < 0 || pr.first >= n_c ||
            pr.second < 0 || pr.second >= n_c)
            throw std::runtime_error(
                "build_jk_pair_contributions: pair cell index out of range");
    }

    const libint2::Operator op = (omega > 0.0)
        ? libint2::Operator::erfc_coulomb
        : libint2::Operator::coulomb;
    libint2::Engine prototype(op, shells_ref.max_nprim(), shells_ref.max_l(), 0);
    if (omega > 0.0) prototype.set_params(omega);
    auto engines = make_engine_pool(prototype);
    const auto shell2bf = shells_ref.shell2bf();
    const std::size_t nshells = shells_ref.size();

    std::vector<std::vector<libint2::Shell>> shells_at(cells.size());
    for (std::size_t c = 0; c < cells.size(); ++c)
        shells_at[c] = shift_shells(shells_ref, cells[c].r_cart);

    const double schwarz_thr = opts.schwarz_threshold;
    const bool screen = (schwarz_thr > 0.0);
    const auto Q = screen
        ? compute_schwarz_factors_per_cell(shells_ref, shells_at, prototype)
        : std::vector<std::vector<double>>{};
    const auto Q_max = screen ? max_q_per_cell(Q) : std::vector<double>{};
    const double D_max = screen
        ? std::max(1.0, std::max(std::fabs(P.maxCoeff()), std::fabs(P.minCoeff())))
        : 0.0;
    CellIndexMap cell_index_map;
    if (screen) cell_index_map = build_cell_index_map(cells);
    int c_zero_idx = -1;
    if (screen) {
        auto it = cell_index_map.find(Eigen::Vector3i(0, 0, 0));
        if (it != cell_index_map.end()) c_zero_idx = it->second;
    }
    const double q_zero_max = (screen && c_zero_idx >= 0) ? Q_max[c_zero_idx] : 0.0;

    // Pre-allocate one output slot per pair; each thread owns a distinct slot.
    const int n_pairs = static_cast<int>(pairs.size());
    std::vector<PairJKContribution> out(pairs.size());
    for (int i = 0; i < n_pairs; ++i) {
        out[i].c_g = pairs[i].first;
        out[i].c_p = pairs[i].second;
        out[i].J_contrib = Eigen::MatrixXd::Zero(nbf, nbf);
        out[i].K_contrib = Eigen::MatrixXd::Zero(nbf, nbf);
    }

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n_pairs; ++i) {
        const int c_g = pairs[i].first;
        const int c_p = pairs[i].second;
        const auto tid = static_cast<std::size_t>(omp_thread_index());
        auto& engine = engines[tid];
        const auto& buf = engine.results();
        Eigen::MatrixXd& Jm_local = out[i].J_contrib;
        Eigen::MatrixXd& Km_local = out[i].K_contrib;
        const auto& shells_g = shells_at[c_g];
        const auto& shells_p = shells_at[c_p];
        int c_pg_idx = -1;
        if (screen) {
            const Eigen::Vector3i pg = cells[c_p].index - cells[c_g].index;
            auto it = cell_index_map.find(pg);
            if (it != cell_index_map.end()) c_pg_idx = it->second;
        }
        if (screen) {
            const bool j_possible = (Q_max[c_g] * q_zero_max * D_max >= schwarz_thr);
            const double q_pg_max = (c_pg_idx >= 0) ? Q_max[c_pg_idx] : 0.0;
            const bool k_possible = (Q_max[c_p] * q_pg_max * D_max >= schwarz_thr);
            if (!j_possible && !k_possible) continue;
        }
        for (std::size_t s1 = 0; s1 < shells_ref.size(); ++s1) {
            const auto bf1 = shell2bf[s1]; const auto n1 = shells_ref[s1].size();
            for (std::size_t s2 = 0; s2 < shells_g.size(); ++s2) {
                const auto bf2 = shell2bf[s2]; const auto n2 = shells_g[s2].size();
                const double q12_J = screen ? Q[c_g][s1 * nshells + s2] : 0.0;
                for (std::size_t s3 = 0; s3 < shells_p.size(); ++s3) {
                    const auto bf3 = shell2bf[s3]; const auto n3 = shells_p[s3].size();
                    const double q13_K = screen ? Q[c_p][s1 * nshells + s3] : 0.0;
                    for (std::size_t s4 = 0; s4 < shells_p.size(); ++s4) {
                        const auto bf4 = shell2bf[s4]; const auto n4 = shells_p[s4].size();
                        bool do_J = true;
                        if (screen) {
                            if (c_zero_idx < 0) { do_J = false; }
                            else {
                                const double q34_J = Q[c_zero_idx][s3 * nshells + s4];
                                if (q12_J * q34_J * D_max < schwarz_thr) do_J = false;
                            }
                        }
                        if (do_J) {
                            engine.compute(shells_ref[s1], shells_g[s2], shells_p[s3], shells_p[s4]);
                            if (const double* blk = buf[0]) {
                                for (std::size_t i_ = 0; i_ < n1; ++i_)
                                for (std::size_t j = 0; j < n2; ++j)
                                for (std::size_t k = 0; k < n3; ++k)
                                for (std::size_t l = 0; l < n4; ++l) {
                                    const double v = blk[((i_ * n2 + j) * n3 + k) * n4 + l];
                                    Jm_local(bf1+i_, bf2+j) += P(bf3+k, bf4+l) * v;
                                }
                            }
                        }
                        bool do_K = true;
                        if (screen) {
                            if (c_pg_idx < 0) { do_K = false; }
                            else {
                                const double q24_K = Q[c_pg_idx][s2 * nshells + s4];
                                if (q13_K * q24_K * D_max < schwarz_thr) do_K = false;
                            }
                        }
                        if (do_K) {
                            engine.compute(shells_ref[s1], shells_p[s3], shells_g[s2], shells_p[s4]);
                            if (const double* blk = buf[0]) {
                                for (std::size_t i_ = 0; i_ < n1; ++i_)
                                for (std::size_t j = 0; j < n2; ++j)
                                for (std::size_t k = 0; k < n3; ++k)
                                for (std::size_t l = 0; l < n4; ++l) {
                                    const double v = blk[((i_ * n3 + k) * n2 + j) * n4 + l];
                                    Km_local(bf1+i_, bf2+j) += P(bf3+k, bf4+l) * v;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return out;
}


// ---- CCM (Cyclic Cluster Model) WSSC-weighted four-center J/K build ------
// Two methods, selectable via `method`:
//   "bra_home" — bra at home, ket imaged, bra-ket symmetrised via a
//        second ERI pass (lambda-home, sigma-home | mu_{-p}, nu_{-p}).
//        Matches vibe-qc's padded ccm_eri frame, validated to ~1e-5.
//   "union12"  — nu imaged over WSC(mu), lambda in union WSC, K
//        symmetrised per spec (AICCM reference frame, research).
JKMatrices build_jk_ccm_weighted(
    const BasisSet& basis,
    const PeriodicSystem& system,
    const std::vector<LatticeCell>& cells,
    const LatticeSumOptions& opts,
    const Eigen::MatrixXd& P_gamma,
    const std::vector<Eigen::Vector3i>& weight_cells,
    const std::vector<Eigen::MatrixXd>& weight_matrices,
    const std::string& method,
    double omega) {
    ensure_libint_initialized();

    const auto& shells_ref = basis.libint();
    const int nbf = static_cast<int>(basis.nbasis());

    const auto shell_info = basis.shells();
    std::vector<int> atom_of_shell(shell_info.size());
    for (std::size_t s = 0; s < shell_info.size(); ++s)
        atom_of_shell[s] = shell_info[s].atom_index;

    if (weight_matrices.empty())
        throw std::runtime_error("build_jk_ccm_weighted: empty weights");
    const int n_atoms = static_cast<int>(weight_matrices[0].rows());
    if (P_gamma.rows() != nbf || P_gamma.cols() != nbf)
        throw std::runtime_error("build_jk_ccm_weighted: density shape mismatch");
    if (omega < 0.0)
        throw std::runtime_error("build_jk_ccm_weighted: omega >= 0 required");
    if (weight_cells.size() != weight_matrices.size())
        throw std::runtime_error("build_jk_ccm_weighted: weight size mismatch");

    WeightLookup wlookup;
    wlookup.reserve(weight_cells.size());
    for (std::size_t wi = 0; wi < weight_cells.size(); ++wi) {
        const auto& W = weight_matrices[wi];
        if (W.rows() != n_atoms || W.cols() != n_atoms)
            throw std::runtime_error("build_jk_ccm_weighted: weight shape mismatch");
        wlookup[weight_cells[wi]] = W;
    }
    auto w = [&](const Eigen::Vector3i& g, int A, int B) -> double {
        auto it = wlookup.find(g);
        if (it == wlookup.end()) return 0.0;
        if (A < 0 || A >= n_atoms || B < 0 || B >= n_atoms) return 0.0;
        return it->second(A, B);
    };

    const libint2::Operator op = (omega > 0.0)
        ? libint2::Operator::erfc_coulomb : libint2::Operator::coulomb;
    libint2::Engine prototype(op, shells_ref.max_nprim(), shells_ref.max_l(), 0);
    if (omega > 0.0) prototype.set_params(omega);
    auto engines = make_engine_pool(prototype);
    const auto shell2bf = shells_ref.shell2bf();
    const std::size_t nshells = shells_ref.size();

    std::vector<std::vector<libint2::Shell>> shells_at(cells.size());
    for (std::size_t c = 0; c < cells.size(); ++c)
        shells_at[c] = shift_shells(shells_ref, cells[c].r_cart);

    const double schwarz_thr = opts.schwarz_threshold;
    const bool screen = (schwarz_thr > 0.0);
    const auto Q = screen
        ? compute_schwarz_factors_per_cell(shells_ref, shells_at, prototype)
        : std::vector<std::vector<double>>{};
    const auto Q_max = screen ? max_q_per_cell(Q) : std::vector<double>{};
    const double D_max = screen
        ? std::max(1.0, std::max(std::fabs(P_gamma.maxCoeff()),
                                  std::fabs(P_gamma.minCoeff()))) : 0.0;
    CellIndexMap cell_index_map;
    if (screen) cell_index_map = build_cell_index_map(cells);
    int c_zero_idx = -1;
    if (screen) {
        auto it = cell_index_map.find(Eigen::Vector3i(0, 0, 0));
        if (it != cell_index_map.end()) c_zero_idx = it->second;
    }

    const int n_c = static_cast<int>(cells.size());
    const int n_threads = omp_max_threads();
    const Eigen::Vector3i home(0, 0, 0);

    // Find home cell.
    int home_idx = -1;
    for (int ci = 0; ci < n_c; ++ci)
        if (cells[ci].index == home) { home_idx = ci; break; }
    if (home_idx < 0)
        throw std::runtime_error("build_jk_ccm_weighted: home cell not found");
    const auto& shells_home = shells_at[home_idx];

    // Pre-build map from cell index to position in cells array (for
    // looking up the negated cell -c_p).
    std::unordered_map<Eigen::Vector3i, int, LatticeIndexHash, LatticeIndexEq> cell_pos_map;
    for (int ci = 0; ci < n_c; ++ci)
        cell_pos_map[cells[ci].index] = ci;

    if (method == "bra_home") {
        // ================================================================
        // bra_home: ket-folded + bra-folded with independent sigma offset.
        //
        // Ket-folded: w_Jk = W0[a,b] * (Wp[a,c]+Wp[b,c])/2 * W{gd}[c,d]
        //             ERI = (mu_0 nu_0 | lambda_p sigma_q)  [q = p + gd]
        // Bra-folded: w_Jb = W{gd}[c,d] * (W{-p}[c,a]+W{-q}[d,a])/2 * W0[a,b]
        //             ERI = (lambda_0 sigma_0 | mu_{-p} nu_{-q})
        // J = 0.5*(Jk + Jb), same for K.
        // ================================================================

        // Pre-build sorted list of weight-cell offsets (gd values).
        std::vector<Eigen::Vector3i> gd_list(weight_cells.begin(), weight_cells.end());

        std::vector<Eigen::MatrixXd> Jk_tls(n_threads, Eigen::MatrixXd::Zero(nbf, nbf));
        std::vector<Eigen::MatrixXd> Kk_tls(n_threads, Eigen::MatrixXd::Zero(nbf, nbf));
        std::vector<Eigen::MatrixXd> Jb_tls(n_threads, Eigen::MatrixXd::Zero(nbf, nbf));
        std::vector<Eigen::MatrixXd> Kb_tls(n_threads, Eigen::MatrixXd::Zero(nbf, nbf));

        const int n_gd = static_cast<int>(gd_list.size());
        const int n_triples = n_c * n_gd;

        #pragma omp parallel for schedule(dynamic)
        for (int idx = 0; idx < n_triples; ++idx) {
            const int c_p = idx / n_gd;
            const int igd  = idx % n_gd;
            const Eigen::Vector3i gd = gd_list[igd];
            const Eigen::Vector3i cell_p = cells[c_p].index;
            const Eigen::Vector3i cell_q(cell_p[0]+gd[0], cell_p[1]+gd[1], cell_p[2]+gd[2]);
            const Eigen::Vector3i neg_p(-cell_p[0], -cell_p[1], -cell_p[2]);
            const Eigen::Vector3i neg_q(-cell_q[0], -cell_q[1], -cell_q[2]);

            const auto tid = static_cast<std::size_t>(omp_thread_index());
            auto& engine = engines[tid];
            const auto& buf = engine.results();
            auto& Jk = Jk_tls[tid]; auto& Kk = Kk_tls[tid];
            auto& Jb = Jb_tls[tid]; auto& Kb = Kb_tls[tid];

            const auto& shells_p = shells_at[c_p];

            // Find shells at cell_q and cell_{-q}.
            int q_idx = -1, nq_idx = -1;
            auto it_q = cell_pos_map.find(cell_q);
            if (it_q != cell_pos_map.end()) q_idx = it_q->second;
            auto it_nq = cell_pos_map.find(neg_q);
            if (it_nq != cell_pos_map.end()) nq_idx = it_nq->second;
            if (q_idx < 0) continue;   // sigma cell must exist
            const auto& shells_q = shells_at[q_idx];
            const std::vector<libint2::Shell>* shells_nq = (nq_idx >= 0) ? &shells_at[nq_idx] : nullptr;

            for (std::size_t s1 = 0; s1 < shells_ref.size(); ++s1) {
                const auto bf1 = shell2bf[s1]; const auto n1 = shells_ref[s1].size();
                const int a = atom_of_shell[s1];                       // mu (home)
                for (std::size_t s2 = 0; s2 < shells_home.size(); ++s2) {
                    const auto bf2 = shell2bf[s2]; const auto n2 = shells_home[s2].size();
                    const int b = atom_of_shell[s2];                    // nu (home)
                    for (std::size_t s3 = 0; s3 < shells_p.size(); ++s3) {
                        const auto bf3 = shell2bf[s3]; const auto n3 = shells_p[s3].size();
                        const int c = atom_of_shell[s3];               // lambda (cell p)
                        for (std::size_t s4 = 0; s4 < shells_q.size(); ++s4) {
                            const auto bf4 = shell2bf[s4]; const auto n4 = shells_q[s4].size();
                            const int d = atom_of_shell[s4];            // sigma (cell q)

                            // ---- ket-folded J: (mu_0 nu_0 | lambda_p sigma_q)
                            {
                                const double w_ab = w(home, a, b);
                                const double w_ac = w(cell_p, a, c);
                                const double w_bc = w(cell_p, b, c);
                                const double w_cd = w(gd, c, d);
                                const double w_Jk = w_ab * 0.5 * (w_ac + w_bc) * w_cd;
                                const Eigen::Vector3i neg_gd(-gd[0], -gd[1], -gd[2]);
                                const double w_Jb = w(home, c, d) * 0.5
                                    * (w(neg_p, c, a) + w(neg_p, d, a)) * w(neg_gd, a, b);

                                if (w_Jk != 0.0 || w_Jb != 0.0) {
                                    if (w_Jk != 0.0) {
                                        engine.compute(shells_ref[s1], shells_home[s2],
                                                       shells_p[s3], shells_q[s4]);
                                        if (const double* blk = buf[0])
                                            for (std::size_t i = 0; i < n1; ++i)
                                            for (std::size_t j = 0; j < n2; ++j)
                                            for (std::size_t k = 0; k < n3; ++k)
                                            for (std::size_t l = 0; l < n4; ++l) {
                                                const double v = blk[((i*n2+j)*n3+k)*n4+l];
                                                Jk(bf1+i, bf2+j) += P_gamma(bf3+k, bf4+l) * v * w_Jk;
                                            }
                                    }
                                    if (w_Jb != 0.0 && shells_nq) {
                                        const auto& snq = *shells_nq;
                                        engine.compute(shells_ref[s3], shells_ref[s4],
                                                       snq[s1], snq[s2]);
                                        if (const double* blk = buf[0])
                                            for (std::size_t i = 0; i < n3; ++i)
                                            for (std::size_t j = 0; j < n4; ++j)
                                            for (std::size_t k = 0; k < n1; ++k)
                                            for (std::size_t l = 0; l < n2; ++l) {
                                                const double v = blk[((i*n4+j)*n1+k)*n2+l];
                                                Jb(bf1+k, bf2+l) += P_gamma(bf4+i, bf3+j) * v * w_Jb;
                                            }
                                    }
                                }
                            }

                            // ---- ket-folded K: (mu_0 lambda_p | nu_0 sigma_q)
                            {
                                const double wk_ab = w(cell_p, a, c);
                                const double wk_ac = w(home, a, b);
                                const double wk_bc = w(neg_p, c, b);
                                const double wk_cd = w(cell_q, b, d);  // No: w(gd, b, d)?
                                // Wait — K has: bra=(mu_0, lambda_p), ket=(nu_0, sigma_q)
                                // ket-pair displacement = q-0 = q. So omega_cd = W[q][b,d]? No.
                                // Actually: nu at home (cell 0), sigma at cell q. 
                                // The WSSC weight omega_cd is w(cell_q, b, d) = W[q][b,d].
                                // But we need w(cell_q)[b,d], not w(gd)[b,d].
                                // gd = cell_q - cell_p. cell_q is the actual cell of sigma.
                                // w(cell_q, b, d) uses the ACTUAL cell of sigma relative to nu at home.
                                const double w_Kk_cd = w(cell_q, b, d);
                                const double w_Kk = wk_ab * 0.5 * (wk_ac + wk_bc) * w_Kk_cd;
                                const double w_Kb = w(home, b, d) * 0.5
                                    * (w(home, b, a) + w(home, d, a)) * w(neg_p, a, c);

                                if (w_Kk != 0.0 || w_Kb != 0.0) {
                                    if (w_Kk != 0.0) {
                                        engine.compute(shells_ref[s1], shells_p[s3],
                                                       shells_home[s2], shells_q[s4]);
                                        if (const double* blk = buf[0])
                                            for (std::size_t i = 0; i < n1; ++i)
                                            for (std::size_t j = 0; j < n2; ++j)
                                            for (std::size_t k = 0; k < n3; ++k)
                                            for (std::size_t l = 0; l < n4; ++l) {
                                                const double v = blk[((i*n3+k)*n2+j)*n4+l];
                                                Kk(bf1+i, bf2+j) += P_gamma(bf3+k, bf4+l) * v * w_Kk;
                                            }
                                    }
                                    if (w_Kb != 0.0 && shells_nq) {
                                        const auto& snq = *shells_nq;
                                        engine.compute(shells_home[s2], shells_q[s4],
                                                       snq[s1], shells_ref[s3]);
                                        if (const double* blk = buf[0])
                                            for (std::size_t i = 0; i < n2; ++i)
                                            for (std::size_t j = 0; j < n4; ++j)
                                            for (std::size_t k = 0; k < n1; ++k)
                                            for (std::size_t l = 0; l < n3; ++l) {
                                                const double v = blk[((i*n4+j)*n1+k)*n3+l];
                                                Kb(bf1+k, bf2+i) += P_gamma(bf4+j, bf3+l) * v * w_Kb;
                                            }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Reduce and symmetrise
        Eigen::MatrixXd Jkm = Eigen::MatrixXd::Zero(nbf, nbf);
        Eigen::MatrixXd Kkm = Eigen::MatrixXd::Zero(nbf, nbf);
        Eigen::MatrixXd Jbm = Eigen::MatrixXd::Zero(nbf, nbf);
        Eigen::MatrixXd Kbm = Eigen::MatrixXd::Zero(nbf, nbf);
        for (const auto& m : Jk_tls) Jkm += m;
        for (const auto& m : Kk_tls) Kkm += m;
        for (const auto& m : Jb_tls) Jbm += m;
        for (const auto& m : Kb_tls) Kbm += m;
        return JKMatrices{0.5*(Jkm+Jbm), 0.5*(Kkm+Kbm)};
    }

    if (method == "bra_home_full") {
        // ================================================================
        // bra_home_full: build effective ERI tensor V, symmetrise, contract.
        // Matches the padded route's ccm_eri exactly: V is built with
        // bra at home, ket imaged (c_p + gd), then bra-ket symmetrised
        // as V_sym = 0.5*(V + V^T), then J = contract(P, V_sym).
        // O(nbf^4) memory — small-system reference implementation.
        // ================================================================

        const int nbf2 = nbf * nbf;
        std::vector<Eigen::MatrixXd> Vj_tls(n_threads, Eigen::MatrixXd::Zero(nbf2, nbf2));

        std::vector<Eigen::Vector3i> gd_list(weight_cells.begin(), weight_cells.end());
        const int n_gd = static_cast<int>(gd_list.size());
        const int n_triples = n_c * n_gd;

        #pragma omp parallel for schedule(dynamic)
        for (int idx = 0; idx < n_triples; ++idx) {
            const int c_p = idx / n_gd;
            const int igd  = idx % n_gd;
            const Eigen::Vector3i gd = gd_list[igd];
            const Eigen::Vector3i cell_p = cells[c_p].index;
            const Eigen::Vector3i cell_q(cell_p[0]+gd[0], cell_p[1]+gd[1], cell_p[2]+gd[2]);

            const auto tid = static_cast<std::size_t>(omp_thread_index());
            auto& engine = engines[tid];
            const auto& buf = engine.results();
            auto& Vj = Vj_tls[tid];
            const auto& shells_p = shells_at[c_p];

            int q_idx = -1;
            auto it_q = cell_pos_map.find(cell_q);
            if (it_q != cell_pos_map.end()) q_idx = it_q->second;
            if (q_idx < 0) continue;
            const auto& shells_q = shells_at[q_idx];

            // Bra (mu_0 nu_0) is the home cell: BOTH bra indices must use the
            // same (reference) shell set. Mixing shells_ref[s1] with
            // shells_home[s2] makes mu and nu come from distinct shell objects
            // and breaks the mu<->nu symmetry of the effective tensor (the
            // wrap/long-range elements come out asymmetric). Use shells_ref for
            // both — verified against the padded ccm_eri reference.
            for (std::size_t s1 = 0; s1 < shells_ref.size(); ++s1) {
                const auto bf1 = shell2bf[s1]; const auto n1 = shells_ref[s1].size();
                const int a = atom_of_shell[s1];
                for (std::size_t s2 = 0; s2 < shells_ref.size(); ++s2) {
                    const auto bf2 = shell2bf[s2]; const auto n2 = shells_ref[s2].size();
                    const int b = atom_of_shell[s2];
                    for (std::size_t s3 = 0; s3 < shells_p.size(); ++s3) {
                        const auto bf3 = shell2bf[s3]; const auto n3 = shells_p[s3].size();
                        const int c = atom_of_shell[s3];
                        for (std::size_t s4 = 0; s4 < shells_q.size(); ++s4) {
                            const auto bf4 = shell2bf[s4]; const auto n4 = shells_q[s4].size();
                            const int d = atom_of_shell[s4];

                            // eq-18 four-center weight, bra at home (matches padded
                            // ccm_eri): w_J = omega_ab(0) * 0.5(omega_ac+omega_bc)(p)
                            //                * omega_cd(gd).
                            const double w_ab = w(home, a, b);
                            if (w_ab == 0.0) continue;
                            const double w_J = w_ab * 0.5
                                * (w(cell_p, a, c) + w(cell_p, b, c)) * w(gd, c, d);
                            if (w_J == 0.0) continue;
                            // One effective ERI tensor V[mu nu, la si] from the
                            // home-bra / ket-imaged integral (mu_0 nu_0 | lambda_p sigma_q).
                            // J and K are both contracted from its bra-ket
                            // symmetrisation below (no separate K tensor).
                            engine.compute(shells_ref[s1], shells_ref[s2],
                                           shells_p[s3], shells_q[s4]);
                            if (const double* blk = buf[0])
                                for (std::size_t i = 0; i < n1; ++i)
                                for (std::size_t j = 0; j < n2; ++j)
                                for (std::size_t k = 0; k < n3; ++k)
                                for (std::size_t l = 0; l < n4; ++l) {
                                    const double v = blk[((i*n2+j)*n3+k)*n4+l];
                                    const int mu = bf1 + i, nu = bf2 + j;
                                    const int la = bf3 + k, si = bf4 + l;
                                    Vj(mu*nbf+nu, la*nbf+si) += v * w_J;
                                }
                        }
                    }
                }
            }
        }

        // Reduce across threads.
        Eigen::MatrixXd V_full = Eigen::MatrixXd::Zero(nbf2, nbf2);
        for (const auto& m : Vj_tls) V_full += m;

        // Bra-ket symmetrise: V_sym = 0.5*(V + V^T) — the (mu nu|la si) <-> (la si|mu nu)
        // swap, exactly the padded ccm_eri's 0.5*(eff + eff^{(cd|ab)}). NB: assign
        // to a SEPARATE matrix — `V_full = 0.5*(V_full + V_full.transpose())`
        // aliases in Eigen (the transpose reads entries already overwritten) and
        // silently yields an ASYMMETRIC tensor, breaking the SCF.
        const Eigen::MatrixXd V_sym = 0.5 * (V_full + V_full.transpose());

        // Contract BOTH J and K from the SAME symmetrised tensor, matching the
        // padded run_ccm_rhf contractions J=einsum("mnrs,rs"), K=einsum("msrn,rs"):
        //   J[mu,nu] = sum_{la,si} P[la,si] * V_sym[mu*n+nu, la*n+si]
        //   K[mu,nu] = sum_{la,si} P[la,si] * V_sym[mu*n+si, la*n+nu]
        Eigen::MatrixXd Jm = Eigen::MatrixXd::Zero(nbf, nbf);
        Eigen::MatrixXd Km = Eigen::MatrixXd::Zero(nbf, nbf);
        for (int mu = 0; mu < nbf; ++mu) {
            for (int nu = 0; nu < nbf; ++nu) {
                double jsum = 0.0, ksum = 0.0;
                for (int la = 0; la < nbf; ++la) {
                    for (int si = 0; si < nbf; ++si) {
                        const double p = P_gamma(la, si);
                        if (p != 0.0) {
                            jsum += p * V_sym(mu*nbf+nu, la*nbf+si);
                            ksum += p * V_sym(mu*nbf+si, la*nbf+nu);
                        }
                    }
                }
                Jm(mu, nu) = jsum;
                Km(mu, nu) = ksum;
            }
        }
        // The eq-18 CCM frame is slightly non-Hermitian (~1e-3 — the residual
        // cyclic-invariance breaking, present in the padded reference too).
        // Hermitise J and K so the Fock is symmetric and the SCF gradient
        // converges (mirrors run_ccm_rhf's F = 0.5*(F + F^T)). Separate
        // destinations: A = 0.5*(A + A.transpose()) aliases in Eigen.
        const Eigen::MatrixXd Jh = 0.5 * (Jm + Jm.transpose());
        const Eigen::MatrixXd Kh = 0.5 * (Km + Km.transpose());
        return JKMatrices{Jh, Kh};
    }

    if (method == "bra_home_full-direct") {
        // ================================================================
        // bra_home_full-direct: the INTEGRAL-DIRECT form of "bra_home_full"
        // (eq-18 weight; Python method "union12"). Same triple loop, same
        // w_J, same bra-ket symmetrisation as the full-tensor branch above,
        // but folds each weighted quartet straight into J/K rather than the
        // O(nbf^4) tensor V. See the aiccm2026dev-a-direct branch below for the
        // fold algebra; the full "bra_home_full" branch is preserved above as
        // the comparison reference. Result agrees to ~1e-12 (summation reorder).
        // ================================================================

        std::vector<Eigen::MatrixXd> J_tls(n_threads, Eigen::MatrixXd::Zero(nbf, nbf));
        std::vector<Eigen::MatrixXd> K_tls(n_threads, Eigen::MatrixXd::Zero(nbf, nbf));

        std::vector<Eigen::Vector3i> gd_list(weight_cells.begin(), weight_cells.end());
        const int n_gd = static_cast<int>(gd_list.size());
        const int n_triples = n_c * n_gd;

        #pragma omp parallel for schedule(dynamic)
        for (int idx = 0; idx < n_triples; ++idx) {
            const int c_p = idx / n_gd;
            const int igd  = idx % n_gd;
            const Eigen::Vector3i gd = gd_list[igd];
            const Eigen::Vector3i cell_p = cells[c_p].index;
            const Eigen::Vector3i cell_q(cell_p[0]+gd[0], cell_p[1]+gd[1], cell_p[2]+gd[2]);

            const auto tid = static_cast<std::size_t>(omp_thread_index());
            auto& engine = engines[tid];
            const auto& buf = engine.results();
            auto& Jloc = J_tls[tid];
            auto& Kloc = K_tls[tid];
            const auto& shells_p = shells_at[c_p];

            int q_idx = -1;
            auto it_q = cell_pos_map.find(cell_q);
            if (it_q != cell_pos_map.end()) q_idx = it_q->second;
            if (q_idx < 0) continue;
            const auto& shells_q = shells_at[q_idx];

            // Opt-in Schwarz screening (active only when screen). Ket pair
            // (lambda_p, sigma_q) separation = gd -> Q at idx(gd); bra is home ->
            // Q at c_zero_idx. eq-18 weight w_J in [0,1], so w_J*Q_bra*Q_ket*D_max
            // bounds the J/K contribution. See the aiccm2026dev-a-direct note.
            int gd_idx = -1;
            if (screen) { auto itg = cell_index_map.find(gd); if (itg != cell_index_map.end()) gd_idx = itg->second; }
            const bool do_screen = screen && gd_idx >= 0 && c_zero_idx >= 0;

            for (std::size_t s1 = 0; s1 < shells_ref.size(); ++s1) {
                const auto bf1 = shell2bf[s1]; const auto n1 = shells_ref[s1].size();
                const int a = atom_of_shell[s1];
                for (std::size_t s2 = 0; s2 < shells_ref.size(); ++s2) {
                    const auto bf2 = shell2bf[s2]; const auto n2 = shells_ref[s2].size();
                    const int b = atom_of_shell[s2];
                    const double w_ab0 = w(home, a, b);
                    if (w_ab0 == 0.0) continue;
                    const double q_bra = do_screen ? Q[c_zero_idx][s1*nshells + s2] : 0.0;
                    if (do_screen && q_bra * Q_max[gd_idx] * D_max < schwarz_thr) continue;
                    for (std::size_t s3 = 0; s3 < shells_p.size(); ++s3) {
                        const auto bf3 = shell2bf[s3]; const auto n3 = shells_p[s3].size();
                        const int c = atom_of_shell[s3];
                        for (std::size_t s4 = 0; s4 < shells_q.size(); ++s4) {
                            const auto bf4 = shell2bf[s4]; const auto n4 = shells_q[s4].size();
                            const int d = atom_of_shell[s4];

                            // eq-18 four-center weight, bra at home (matches the
                            // full bra_home_full branch above).
                            const double w_ab = w(home, a, b);
                            if (w_ab == 0.0) continue;
                            const double w_J = w_ab * 0.5
                                * (w(cell_p, a, c) + w(cell_p, b, c)) * w(gd, c, d);
                            if (w_J == 0.0) continue;
                            // Per-quartet Schwarz bound.
                            if (do_screen &&
                                w_J * q_bra * Q[gd_idx][s3*nshells + s4] * D_max < schwarz_thr)
                                continue;
                            engine.compute(shells_ref[s1], shells_ref[s2],
                                           shells_p[s3], shells_q[s4]);
                            if (const double* blk = buf[0])
                                for (std::size_t i = 0; i < n1; ++i)
                                for (std::size_t j = 0; j < n2; ++j)
                                for (std::size_t k = 0; k < n3; ++k)
                                for (std::size_t l = 0; l < n4; ++l) {
                                    const double t = blk[((i*n2+j)*n3+k)*n4+l] * w_J;
                                    if (t == 0.0) continue;
                                    const int mu = bf1 + i, nu = bf2 + j;
                                    const int la = bf3 + k, si = bf4 + l;
                                    Jloc(mu, nu) += 0.5 * t * P_gamma(la, si);
                                    Jloc(la, si) += 0.5 * t * P_gamma(mu, nu);
                                    Kloc(mu, si) += 0.5 * t * P_gamma(la, nu);
                                    Kloc(la, nu) += 0.5 * t * P_gamma(mu, si);
                                }
                        }
                    }
                }
            }
        }

        Eigen::MatrixXd Jm = Eigen::MatrixXd::Zero(nbf, nbf);
        Eigen::MatrixXd Km = Eigen::MatrixXd::Zero(nbf, nbf);
        for (const auto& m : J_tls) Jm += m;
        for (const auto& m : K_tls) Km += m;
        // Same final Hermitisation as the full bra_home_full branch.
        const Eigen::MatrixXd Jh = 0.5 * (Jm + Jm.transpose());
        const Eigen::MatrixXd Kh = 0.5 * (Km + Km.transpose());
        return JKMatrices{Jh, Kh};
    }

    if (method == "aiccm2026dev-a" || method == "aiccmdev") {  // "aiccmdev" = deprecated alias
        // ================================================================
        // aiccm2026dev-a: the symmetric Born-von Karman-torus four-center
        // (AICCM_ALGORITHM.md §13; Python padded.ccm_eri_symmetric). Two
        // changes vs bra_home_full (eq 18), both needed for exact 8-fold
        // permutation symmetry on ANY lattice:
        //   * symmetric bridge  1/4(w_ac + w_bc + w_ad + w_bd)  -- treats the
        //     ket functions rho, sigma identically (eq 18 uses 1/2(w_ac+w_bc),
        //     singling out the ket anchor rho);
        //   * independent minimum-image fold -- rho at g_c and sigma at g_e are
        //     EACH the min image of the home bra; the ket-pair weight w_rho_sigma
        //     is taken at g_e - g_c (eq 18 chains sigma to rho at g_c+g_d).
        // V is then bra-ket symmetrised and contracted exactly as bra_home_full.
        // O(nbf^4) memory -- small-system reference (matches the padded route).
        // ================================================================

        const int nbf2 = nbf * nbf;
        std::vector<Eigen::MatrixXd> Vj_tls(n_threads, Eigen::MatrixXd::Zero(nbf2, nbf2));

        const std::vector<Eigen::Vector3i> wcells(weight_cells.begin(), weight_cells.end());
        const int n_w = static_cast<int>(wcells.size());
        const int n_pairs = n_w * n_w;

        #pragma omp parallel for schedule(dynamic)
        for (int idx = 0; idx < n_pairs; ++idx) {
            const Eigen::Vector3i gc = wcells[idx / n_w];          // rho cell
            const Eigen::Vector3i ge = wcells[idx % n_w];          // sigma cell
            const Eigen::Vector3i grel(ge[0]-gc[0], ge[1]-gc[1], ge[2]-gc[2]);
            // rho,sigma must be a minimum-image pair (w_rho_sigma at g_e-g_c).
            if (wlookup.find(grel) == wlookup.end()) continue;

            int gc_idx = -1, ge_idx = -1;
            auto it_c = cell_pos_map.find(gc); if (it_c != cell_pos_map.end()) gc_idx = it_c->second;
            auto it_e = cell_pos_map.find(ge); if (it_e != cell_pos_map.end()) ge_idx = it_e->second;
            if (gc_idx < 0 || ge_idx < 0) continue;

            const auto tid = static_cast<std::size_t>(omp_thread_index());
            auto& engine = engines[tid];
            const auto& buf = engine.results();
            auto& Vj = Vj_tls[tid];
            const auto& shells_c = shells_at[gc_idx];
            const auto& shells_e = shells_at[ge_idx];

            // Bra (mu_0 nu_0) at home: both bra indices use shells_ref (see the
            // bra_home_full note on preserving the mu<->nu symmetry).
            for (std::size_t s1 = 0; s1 < shells_ref.size(); ++s1) {
                const auto bf1 = shell2bf[s1]; const auto n1 = shells_ref[s1].size();
                const int a = atom_of_shell[s1];
                for (std::size_t s2 = 0; s2 < shells_ref.size(); ++s2) {
                    const auto bf2 = shell2bf[s2]; const auto n2 = shells_ref[s2].size();
                    const int b = atom_of_shell[s2];
                    const double w_ab = w(home, a, b);
                    if (w_ab == 0.0) continue;
                    for (std::size_t s3 = 0; s3 < shells_c.size(); ++s3) {
                        const auto bf3 = shell2bf[s3]; const auto n3 = shells_c[s3].size();
                        const int c = atom_of_shell[s3];
                        for (std::size_t s4 = 0; s4 < shells_e.size(); ++s4) {
                            const auto bf4 = shell2bf[s4]; const auto n4 = shells_e[s4].size();
                            const int d = atom_of_shell[s4];

                            const double w_cd = w(grel, c, d);
                            if (w_cd == 0.0) continue;
                            const double bridge = 0.25 * (w(gc, a, c) + w(gc, b, c)
                                                          + w(ge, a, d) + w(ge, b, d));
                            const double w4 = w_ab * bridge * w_cd;
                            if (w4 == 0.0) continue;

                            // (mu_0 nu_0 | lambda_{g_c} sigma_{g_e})
                            engine.compute(shells_ref[s1], shells_ref[s2],
                                           shells_c[s3], shells_e[s4]);
                            if (const double* blk = buf[0])
                                for (std::size_t i = 0; i < n1; ++i)
                                for (std::size_t j = 0; j < n2; ++j)
                                for (std::size_t k = 0; k < n3; ++k)
                                for (std::size_t l = 0; l < n4; ++l) {
                                    const double v = blk[((i*n2+j)*n3+k)*n4+l];
                                    const int mu = bf1 + i, nu = bf2 + j;
                                    const int la = bf3 + k, si = bf4 + l;
                                    Vj(mu*nbf+nu, la*nbf+si) += v * w4;
                                }
                        }
                    }
                }
            }
        }

        Eigen::MatrixXd V_full = Eigen::MatrixXd::Zero(nbf2, nbf2);
        for (const auto& m : Vj_tls) V_full += m;
        // Bra-ket symmetrise (the remaining generator; mu<->nu and rho<->sigma
        // already hold by construction). Separate destination -- aliasing note
        // in bra_home_full applies.
        const Eigen::MatrixXd V_sym = 0.5 * (V_full + V_full.transpose());

        Eigen::MatrixXd Jm = Eigen::MatrixXd::Zero(nbf, nbf);
        Eigen::MatrixXd Km = Eigen::MatrixXd::Zero(nbf, nbf);
        for (int mu = 0; mu < nbf; ++mu) {
            for (int nu = 0; nu < nbf; ++nu) {
                double jsum = 0.0, ksum = 0.0;
                for (int la = 0; la < nbf; ++la) {
                    for (int si = 0; si < nbf; ++si) {
                        const double p = P_gamma(la, si);
                        if (p != 0.0) {
                            jsum += p * V_sym(mu*nbf+nu, la*nbf+si);
                            ksum += p * V_sym(mu*nbf+si, la*nbf+nu);
                        }
                    }
                }
                Jm(mu, nu) = jsum;
                Km(mu, nu) = ksum;
            }
        }
        // The symmetric tensor already gives Hermitian J/K; this is a no-op
        // safeguard (separate destinations -- Eigen aliasing).
        const Eigen::MatrixXd Jh = 0.5 * (Jm + Jm.transpose());
        const Eigen::MatrixXd Kh = 0.5 * (Km + Km.transpose());
        return JKMatrices{Jh, Kh};
    }

    if (method == "aiccm2026dev-a-direct" || method == "aiccmdev-direct") {
        // ================================================================
        // aiccm2026dev-a-direct: the INTEGRAL-DIRECT form of "aiccm2026dev-a"
        // (Phase 3b). Identical quartet loop, weights, and bra-ket
        // symmetrisation as the full-tensor branch above, but each weighted
        // quartet block is folded straight into J and K instead of accumulated
        // into the O(nbf^4) effective tensor V. Peak memory drops from
        // (n_threads+2)*nbf^4 (Vj_tls + V_full + V_sym) to n_threads*2*nbf^2
        // (thread-local J/K) -- this is what makes 3-D cells at production basis
        // fit in RAM. The dense "aiccm2026dev-a" branch is PRESERVED above as the
        // small-cluster comparison reference (and the byte-for-byte gate target).
        //
        // Exactness of the fold: with V_sym = 0.5*(V + V^T) and the full-branch
        // contractions  J[mn] = sum_{ls} P[ls] V_sym[mn,ls],
        //                K[mn] = sum_{ls} P[ls] V_sym[ms,ln],
        // a single block  t = (mu nu | la si) * w4  contributes
        //   J[mu,nu] += 0.5 t P[la,si]      J[la,si] += 0.5 t P[mu,nu]
        //   K[mu,si] += 0.5 t P[la,nu]      K[la,nu] += 0.5 t P[mu,si]
        // (the paired terms are the V and V^T halves of V_sym; the home-bra /
        // imaged-ket loop never visits the transpose itself). A reduction +
        // final Hermitisation match the full branch; result agrees to ~1e-12
        // (summation reorder), NOT bit-for-bit -- hence opt-in, not a silent
        // replacement of the dense reference.
        // ================================================================

        std::vector<Eigen::MatrixXd> J_tls(n_threads, Eigen::MatrixXd::Zero(nbf, nbf));
        std::vector<Eigen::MatrixXd> K_tls(n_threads, Eigen::MatrixXd::Zero(nbf, nbf));

        const std::vector<Eigen::Vector3i> wcells(weight_cells.begin(), weight_cells.end());
        const int n_w = static_cast<int>(wcells.size());
        const int n_pairs = n_w * n_w;

        #pragma omp parallel for schedule(dynamic)
        for (int idx = 0; idx < n_pairs; ++idx) {
            const Eigen::Vector3i gc = wcells[idx / n_w];          // rho cell
            const Eigen::Vector3i ge = wcells[idx % n_w];          // sigma cell
            const Eigen::Vector3i grel(ge[0]-gc[0], ge[1]-gc[1], ge[2]-gc[2]);
            // rho,sigma must be a minimum-image pair (w_rho_sigma at g_e-g_c).
            if (wlookup.find(grel) == wlookup.end()) continue;

            int gc_idx = -1, ge_idx = -1;
            auto it_c = cell_pos_map.find(gc); if (it_c != cell_pos_map.end()) gc_idx = it_c->second;
            auto it_e = cell_pos_map.find(ge); if (it_e != cell_pos_map.end()) ge_idx = it_e->second;
            if (gc_idx < 0 || ge_idx < 0) continue;

            const auto tid = static_cast<std::size_t>(omp_thread_index());
            auto& engine = engines[tid];
            const auto& buf = engine.results();
            auto& Jloc = J_tls[tid];
            auto& Kloc = K_tls[tid];
            const auto& shells_c = shells_at[gc_idx];
            const auto& shells_e = shells_at[ge_idx];

            // Opt-in Schwarz screening (active only when screen, i.e.
            // opts.schwarz_threshold > 0; off by default keeps the kernel exact).
            // The ket pair (lambda_{g_c}, sigma_{g_e}) Schwarz factor is
            // translation-invariant -> Q at separation grel = g_e - g_c; the bra
            // is home-home -> Q at c_zero_idx. WSSC weights are in [0,1] so
            // |w4| <= 1, making w4 * Q_bra * Q_ket * D_max a valid upper bound on
            // the |w4 (mu nu|la si) P[la,si]| J/K contribution.
            int grel_idx = -1;
            if (screen) { auto itg = cell_index_map.find(grel); if (itg != cell_index_map.end()) grel_idx = itg->second; }
            const bool do_screen = screen && grel_idx >= 0 && c_zero_idx >= 0;

            // Bra (mu_0 nu_0) at home: both bra indices use shells_ref (see the
            // bra_home_full note on preserving the mu<->nu symmetry).
            for (std::size_t s1 = 0; s1 < shells_ref.size(); ++s1) {
                const auto bf1 = shell2bf[s1]; const auto n1 = shells_ref[s1].size();
                const int a = atom_of_shell[s1];
                for (std::size_t s2 = 0; s2 < shells_ref.size(); ++s2) {
                    const auto bf2 = shell2bf[s2]; const auto n2 = shells_ref[s2].size();
                    const int b = atom_of_shell[s2];
                    const double w_ab = w(home, a, b);
                    if (w_ab == 0.0) continue;
                    const double q_bra = do_screen ? Q[c_zero_idx][s1*nshells + s2] : 0.0;
                    // Bra-level early-out: even the largest ket pair can't survive.
                    if (do_screen && q_bra * Q_max[grel_idx] * D_max < schwarz_thr) continue;
                    for (std::size_t s3 = 0; s3 < shells_c.size(); ++s3) {
                        const auto bf3 = shell2bf[s3]; const auto n3 = shells_c[s3].size();
                        const int c = atom_of_shell[s3];
                        for (std::size_t s4 = 0; s4 < shells_e.size(); ++s4) {
                            const auto bf4 = shell2bf[s4]; const auto n4 = shells_e[s4].size();
                            const int d = atom_of_shell[s4];

                            const double w_cd = w(grel, c, d);
                            if (w_cd == 0.0) continue;
                            const double bridge = 0.25 * (w(gc, a, c) + w(gc, b, c)
                                                          + w(ge, a, d) + w(ge, b, d));
                            const double w4 = w_ab * bridge * w_cd;
                            if (w4 == 0.0) continue;
                            // Per-quartet Schwarz bound.
                            if (do_screen &&
                                w4 * q_bra * Q[grel_idx][s3*nshells + s4] * D_max < schwarz_thr)
                                continue;

                            // (mu_0 nu_0 | lambda_{g_c} sigma_{g_e})
                            engine.compute(shells_ref[s1], shells_ref[s2],
                                           shells_c[s3], shells_e[s4]);
                            if (const double* blk = buf[0])
                                for (std::size_t i = 0; i < n1; ++i)
                                for (std::size_t j = 0; j < n2; ++j)
                                for (std::size_t k = 0; k < n3; ++k)
                                for (std::size_t l = 0; l < n4; ++l) {
                                    const double t = blk[((i*n2+j)*n3+k)*n4+l] * w4;
                                    if (t == 0.0) continue;
                                    const int mu = bf1 + i, nu = bf2 + j;
                                    const int la = bf3 + k, si = bf4 + l;
                                    // V and V^T halves of V_sym, folded into J:
                                    Jloc(mu, nu) += 0.5 * t * P_gamma(la, si);
                                    Jloc(la, si) += 0.5 * t * P_gamma(mu, nu);
                                    // ...and into K:
                                    Kloc(mu, si) += 0.5 * t * P_gamma(la, nu);
                                    Kloc(la, nu) += 0.5 * t * P_gamma(mu, si);
                                }
                        }
                    }
                }
            }
        }

        Eigen::MatrixXd Jm = Eigen::MatrixXd::Zero(nbf, nbf);
        Eigen::MatrixXd Km = Eigen::MatrixXd::Zero(nbf, nbf);
        for (const auto& m : J_tls) Jm += m;
        for (const auto& m : K_tls) Km += m;
        // Final Hermitisation matches the full branch (separate destinations --
        // Eigen aliasing note above).
        const Eigen::MatrixXd Jh = 0.5 * (Jm + Jm.transpose());
        const Eigen::MatrixXd Kh = 0.5 * (Km + Km.transpose());
        return JKMatrices{Jh, Kh};
    }


    // ================================================================
    // union12: nu imaged, K symmetrised (AICCM reference frame)
    // ================================================================
    const int n_pairs = n_c * n_c;
    std::vector<Eigen::MatrixXd> J_tls(n_threads, Eigen::MatrixXd::Zero(nbf, nbf));
    std::vector<Eigen::MatrixXd> K_tls(n_threads, Eigen::MatrixXd::Zero(nbf, nbf));

    #pragma omp parallel for schedule(dynamic)
    for (int idx = 0; idx < n_pairs; ++idx) {
        const int c_g = idx / n_c;
        const int c_p = idx % n_c;
        const auto tid = static_cast<std::size_t>(omp_thread_index());
        auto& engine = engines[tid];
        const auto& buf = engine.results();
        auto& Jloc = J_tls[tid];
        auto& Kloc = K_tls[tid];
        const auto& shells_g = shells_at[c_g];
        const auto& shells_p = shells_at[c_p];
        const Eigen::Vector3i cell_g = cells[c_g].index;
        const Eigen::Vector3i cell_p = cells[c_p].index;
        const Eigen::Vector3i pg(cell_p[0]-cell_g[0], cell_p[1]-cell_g[1], cell_p[2]-cell_g[2]);
        const Eigen::Vector3i gp(cell_g[0]-cell_p[0], cell_g[1]-cell_p[1], cell_g[2]-cell_p[2]);

        int c_pg_idx = -1;
        if (screen) {
            auto it = cell_index_map.find(cells[c_p].index - cells[c_g].index);
            if (it != cell_index_map.end()) c_pg_idx = it->second;
        }
        if (screen) {
            const bool jp = (Q_max[c_g] * ((c_zero_idx>=0)?Q_max[c_zero_idx]:0.0) * D_max >= schwarz_thr);
            const double qpg = (c_pg_idx>=0) ? Q_max[c_pg_idx] : 0.0;
            if (!jp && !(Q_max[c_p]*qpg*D_max >= schwarz_thr)) continue;
        }

        for (std::size_t s1 = 0; s1 < shells_ref.size(); ++s1) {
            const auto bf1 = shell2bf[s1]; const auto n1 = shells_ref[s1].size();
            const int a = atom_of_shell[s1];
            for (std::size_t s2 = 0; s2 < shells_g.size(); ++s2) {
                const auto bf2 = shell2bf[s2]; const auto n2 = shells_g[s2].size();
                const int b = atom_of_shell[s2];
                for (std::size_t s3 = 0; s3 < shells_p.size(); ++s3) {
                    const auto bf3 = shell2bf[s3]; const auto n3 = shells_p[s3].size();
                    const int c = atom_of_shell[s3];
                    for (std::size_t s4 = 0; s4 < shells_p.size(); ++s4) {
                        const auto bf4 = shell2bf[s4]; const auto n4 = shells_p[s4].size();
                        const int d = atom_of_shell[s4];

                        const double w_ab = w(cell_g, a, b);
                        if (w_ab == 0.0) continue;
                        const double w_ac = w(cell_p, a, c);
                        const double w_bc = w(pg, b, c);
                        const double w_cd = w(home, c, d);
                        const double w_J = w_ab * 0.5*(w_ac+w_bc) * w_cd;

                        if (w_J != 0.0) {
                            engine.compute(shells_ref[s1], shells_g[s2], shells_p[s3], shells_p[s4]);
                            if (const double* blk = buf[0])
                                for (std::size_t i = 0; i < n1; ++i)
                                for (std::size_t j = 0; j < n2; ++j)
                                for (std::size_t k = 0; k < n3; ++k)
                                for (std::size_t l = 0; l < n4; ++l) {
                                    const double v = blk[((i*n2+j)*n3+k)*n4+l];
                                    Jloc(bf1+i, bf2+j) += P_gamma(bf3+k, bf4+l) * v * w_J;
                                }
                        }

                        // K1: V[mu,lambda,nu,sigma] -> (mu_0 lambda_p | nu_g sigma_p)
                        const double wk1_ab = w(cell_p, a, c);
                        if (wk1_ab != 0.0) {
                            const double wk1_ac = w(cell_g, a, b);
                            const double wk1_bc = w(gp, c, b);
                            const double wk1_cd = w(home, b, d);
                            const double w_K1 = wk1_ab * 0.5*(wk1_ac+wk1_bc) * wk1_cd;
                            if (w_K1 != 0.0) {
                                engine.compute(shells_ref[s1], shells_p[s3], shells_g[s2], shells_p[s4]);
                                if (const double* blk = buf[0])
                                    for (std::size_t i = 0; i < n1; ++i)
                                    for (std::size_t j = 0; j < n2; ++j)
                                    for (std::size_t k = 0; k < n3; ++k)
                                    for (std::size_t l = 0; l < n4; ++l) {
                                        const double v = blk[((i*n3+k)*n2+j)*n4+l];
                                        Kloc(bf1+i, bf2+j) += P_gamma(bf3+k, bf4+l) * v * w_K1;
                                    }
                            }
                        }
                        // K2: V[mu,sigma,nu,lambda] -> (mu_0 sigma_p | nu_g lambda_p)
                        const double wk2_ab = w(cell_p, a, d);
                        if (wk2_ab != 0.0) {
                            const double wk2_ac = w(cell_g, a, b);
                            const double wk2_bc = w(gp, d, b);
                            const double wk2_cd = w(home, b, c);
                            const double w_K2 = wk2_ab * 0.5*(wk2_ac+wk2_bc) * wk2_cd;
                            if (w_K2 != 0.0) {
                                engine.compute(shells_ref[s1], shells_p[s4], shells_g[s2], shells_p[s3]);
                                if (const double* blk = buf[0])
                                    for (std::size_t i = 0; i < n1; ++i)
                                    for (std::size_t j = 0; j < n2; ++j)
                                    for (std::size_t k = 0; k < n4; ++k)
                                    for (std::size_t l = 0; l < n3; ++l) {
                                        const double v = blk[((i*n4+k)*n2+j)*n3+l];
                                        Kloc(bf1+i, bf2+j) += P_gamma(bf4+k, bf3+l) * v * w_K2;
                                    }
                            }
                        }
                    }
                }
            }
        }
    }

    Eigen::MatrixXd Jm = Eigen::MatrixXd::Zero(nbf, nbf);
    Eigen::MatrixXd Km = Eigen::MatrixXd::Zero(nbf, nbf);
    for (const auto& m : J_tls) Jm += m;
    for (const auto& m : K_tls) Km += m;
    Km *= 0.5;
    return JKMatrices{Jm, Km};
}
}  // namespace vibeqc
