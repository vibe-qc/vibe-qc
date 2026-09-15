#include "vibeqc/periodic_gaussian_bloch_iao.hpp"

#include <algorithm>
#include <array>
#include <cfenv>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

#include "vibeqc/aopair_ft.hpp"
#include "vibeqc/cart_to_sph_data.hpp"
#include "vibeqc/detail/sha256.hpp"

namespace vibeqc {
namespace {
using I = std::uint64_t;
using Z = std::complex<double>;
using Options = PeriodicGaussianBlochIAOOptions;
using Caps = PeriodicGaussianBlochIAOCaps;
using Plan = PeriodicGaussianBlochIAOMemoryPlan;
using Diagnostics = PeriodicGaussianBlochIAODiagnostics;
static_assert(sizeof(double) == 8 && sizeof(Z) == 16 && sizeof(int) == 4
              && std::numeric_limits<double>::is_iec559 && std::numeric_limits<double>::digits == 53,
              "Gaussian Bloch IAO requires binary64 and 32-bit atom indices");
static_assert(kAuxiliaryBasisContentDigestVersion == 1U, "update Gaussian Bloch IAO basis-wire census");
static_assert(cart_to_sph_data::kMaxL == 6, "update Gaussian Bloch IAO primitive-loop bound for a new angular range");
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error "Gaussian Bloch IAO forbids fast/finite-only math"
#endif
#if FLT_EVAL_METHOD != 0
#error "Gaussian Bloch IAO requires binary64 evaluation"
#endif
I add(I a, I b) {
    if (b > std::numeric_limits<I>::max() - a) throw std::overflow_error("Gaussian Bloch IAO count sum overflow");
    return a + b;
}
I mul(I a, I b) {
    if (a && b > std::numeric_limits<I>::max() / a) throw std::overflow_error("Gaussian Bloch IAO count product overflow");
    return a * b;
}
void limit(I value, I cap, const char* message) { if (value > cap) throw std::length_error(message); }
double finite(double x) {
    if (!std::isfinite(x)) throw std::overflow_error("Gaussian Bloch IAO arithmetic is not finite");
    return x;
}
double magnitude(Z x) { return finite(std::abs(x)); }
void check(Z a, Z b, double absolute, double relative, double& maximum, const char* message) {
    const double error = magnitude(a - b); maximum = std::max(maximum, error);
    const double boundary = finite(absolute + finite(relative * std::max(magnitude(a), magnitude(b))));
    if (error > boundary) throw std::invalid_argument(message);
}
void tolerance(double a, double r) {
    if (!std::isfinite(a) || a < 0 || a >= 1 || !std::isfinite(r) || r < 0 || r >= 1 || (a == 0 && r == 0))
        throw std::invalid_argument("Gaussian Bloch IAO requires explicit finite nonnegative audit tolerance pairs below one");
}
void controls(const Options& o, const PeriodicCorrelationBlochIAOOptions& io, const Caps& c) {
    if (!std::isfinite(o.image_cutoff_bohr) || o.image_cutoff_bohr < 0)
        throw std::invalid_argument("Gaussian Bloch IAO finite image cutoff must be nonnegative");
    finite(o.image_cutoff_bohr * o.image_cutoff_bohr);
    tolerance(o.geometry_absolute_tolerance, o.geometry_relative_tolerance);
    tolerance(o.overlap_absolute_tolerance, o.overlap_relative_tolerance);
    tolerance(o.structural_absolute_tolerance, o.structural_relative_tolerance);
    tolerance(o.projection_absolute_tolerance, o.projection_relative_tolerance);
    for (const auto pair : std::array<std::array<double, 2>, 5>{{
            {io.minimal_rank_absolute_floor, io.minimal_rank_relative_floor},
            {io.depolarized_rank_absolute_floor, io.depolarized_rank_relative_floor},
            {io.iao_rank_absolute_floor, io.iao_rank_relative_floor},
            {io.schur_negative_absolute_tolerance, io.schur_negative_relative_tolerance},
            {io.validation_absolute_tolerance, io.validation_relative_tolerance}}}) {
        if (!std::isfinite(pair[0]) || pair[0] < 0 || !std::isfinite(pair[1]) || pair[1] < 0
            || pair[1] >= 1 || (pair[0] == 0 && pair[1] == 0))
            throw std::invalid_argument("Gaussian Bloch IAO requires explicit valid native IAO rank/audit controls");
    }
    if (io.validation_absolute_tolerance >= 1 || !io.jacobi_max_sweeps || !io.maximum_work_units
        || !std::isfinite(io.jacobi_relative_tolerance) || io.jacobi_relative_tolerance <= 0
        || io.jacobi_relative_tolerance >= 1)
        throw std::invalid_argument("Gaussian Bloch IAO native IAO solver controls are invalid");
    for (I x : {c.maximum_owned_numerical_bytes, c.maximum_work_units, c.maximum_atom_count,
            c.maximum_shell_count, c.maximum_contraction_count, c.maximum_primitive_numeric_lanes,
            c.maximum_basis_content_wire_bytes, c.maximum_borrowed_active_numeric_bytes,
            c.maximum_panel_pairs, c.maximum_total_image_candidates})
        if (!x) throw std::invalid_argument("Gaussian Bloch IAO caps must all be positive");
    volatile double tiny = std::numeric_limits<double>::denorm_min(), one = 1.0, zero = 0.0;
    if (std::fegetround() != FE_TONEAREST || !(tiny > 0) || std::fma(tiny, one, zero) != tiny)
        throw std::invalid_argument("Gaussian Bloch IAO requires round-to-nearest and gradual underflow");
}

class Digest {
public:
    explicit Digest(const char* domain) { text(domain); u32(kPeriodicGaussianBlochIAOVersion); }
    void u32(std::uint32_t x) { std::array<std::uint8_t, 4> b{}; for (unsigned i = 0; i < 4; ++i) b[i] = x >> (24 - 8 * i); h_.update(b.data(), b.size()); }
    void u64(I x) { std::array<std::uint8_t, 8> b{}; for (unsigned i = 0; i < 8; ++i) b[i] = x >> (56 - 8 * i); h_.update(b.data(), b.size()); }
    void real(double x) { finite(x); if (x == 0) x = 0; I bits; std::memcpy(&bits, &x, 8); u64(bits); }
    void value(Z x) { real(x.real()); real(x.imag()); }
    void text(const std::string& s) { u64(s.size()); h_.update(reinterpret_cast<const std::uint8_t*>(s.data()), s.size()); }
    std::string finish() { return h_.finish_hex(); }
private: detail::Sha256 h_;
};

struct Counts {
    I shells = 0, contractions = 0, primitives = 0, wire = 150, numeric = 0, max_primitives = 0;
    I work() const { return add(add(mul(128, wire), mul(64, add(shells, contractions))), 4096); }
    void admit(const Caps& c) const {
        limit(shells, c.maximum_shell_count, "Gaussian Bloch IAO shell scan cap exceeded");
        limit(contractions, c.maximum_contraction_count, "Gaussian Bloch IAO contraction scan cap exceeded");
        limit(primitives, c.maximum_primitive_numeric_lanes, "Gaussian Bloch IAO primitive scan cap exceeded");
        limit(wire, std::min(c.maximum_basis_content_wire_bytes, std::numeric_limits<I>::max() / 8 - 4096),
              "Gaussian Bloch IAO basis wire cap or hash range exceeded");
        limit(numeric, c.maximum_borrowed_active_numeric_bytes, "Gaussian Bloch IAO borrowed source cap exceeded");
        limit(work(), c.maximum_work_units, "Gaussian Bloch IAO basis scan work cap exceeded");
    }
};
PeriodicGaussianBasisInventory inspect(const BasisSet& basis, const PeriodicSystem& system,
                                      Counts& counts, const Caps& caps) {
    PeriodicGaussianBasisInventory out; out.content_wire_bytes = 75;
    if (basis.libint().empty()) throw std::invalid_argument("Gaussian Bloch IAO requires nonempty explicit bases");
    limit(add(counts.shells, basis.nshells()), caps.maximum_shell_count, "Gaussian Bloch IAO shell scan cap exceeded");
    for (std::size_t s = 0; s < basis.nshells(); ++s) {
        const auto& shell = basis.libint()[s]; const I np = shell.alpha.size();
        if (!np || shell.contr.empty()) throw std::invalid_argument("Gaussian Bloch IAO has an empty shell");
        ++counts.shells; ++out.shell_count;
        counts.primitives = add(counts.primitives, np); counts.wire = add(counts.wire, 8);
        counts.numeric = add(counts.numeric, mul(8, add(3, np)));
        counts.max_primitives = std::max(counts.max_primitives, np);
        out.exponent_count = add(out.exponent_count, np); out.content_wire_bytes = add(out.content_wire_bytes, 8);
        counts.admit(caps);
        limit(add(counts.contractions, shell.contr.size()), caps.maximum_contraction_count,
              "Gaussian Bloch IAO contraction scan cap exceeded");
        const int atom = basis.shell_atom_index(s);
        if (atom < 0 || static_cast<std::size_t>(atom) >= system.unit_cell.size())
            throw std::invalid_argument("Gaussian Bloch IAO shell atom index is outside the explicit unit cell");
        for (unsigned axis = 0; axis < 3; ++axis)
            if (!std::isfinite(shell.O[axis]) || shell.O[axis] != system.unit_cell[atom].xyz[axis])
                throw std::invalid_argument("Gaussian Bloch IAO shell origin differs from its explicit unit-cell atom");
        for (const auto& contraction : shell.contr) {
            if (contraction.l < 0 || contraction.l > cart_to_sph_data::kMaxL
                || (!contraction.pure && contraction.l != 0) || contraction.coeff.size() != np)
                throw std::invalid_argument("Gaussian Bloch IAO supports pure spherical L<=6 or Cartesian s with matching primitive extents");
            ++counts.contractions; ++out.contraction_count;
            counts.primitives = add(counts.primitives, np); counts.numeric = add(counts.numeric, mul(8, np));
            const I record = add(45, mul(16, np)); counts.wire = add(counts.wire, record);
            out.content_wire_bytes = add(out.content_wire_bytes, record);
            out.coefficient_count = add(out.coefficient_count, np);
            out.function_count = add(out.function_count, 2U * static_cast<I>(contraction.l) + 1U);
            counts.admit(caps);
        }
    }
    if (out.function_count != basis.nbasis()) throw std::invalid_argument("Gaussian Bloch IAO basis extent mismatch");
    out.borrowed_active_numeric_bytes = mul(8, add(mul(3, out.shell_count), add(out.exponent_count, out.coefficient_count)));
    return out;
}

struct Stream { const BasisSet* bra; const BasisSet* ket; I role; bool conjugate; };
std::array<Stream, 7> streams(const BasisSet& ao, const BasisSet& minimal) {
    return {{{&ao, &ao, 0, false}, {&ao, &minimal, 1, false}, {&minimal, &minimal, 2, false},
             {&minimal, &ao, 3, false}, {&ao, &ao, 4, true}, {&ao, &minimal, 5, true},
             {&minimal, &minimal, 6, true}}};
}
Eigen::Vector3d cartesian(const Eigen::Matrix3d& reciprocal, const Eigen::Vector3d& fractional) {
    Eigen::Vector3d value;
    for (unsigned row = 0; row < 3; ++row) {
        double sum = 0;
        for (int column = 2; column >= 0; --column) sum = finite(std::fma(reciprocal(row, column), fractional[column], sum));
        value[row] = sum == 0 ? 0 : sum;
    }
    return value;
}
struct Prepared {
    Plan plan;
    Diagnostics diagnostics;
    Eigen::Matrix3d reciprocal;
    Eigen::Vector3d k, kbar;
    std::string ao_digest, minimal_digest;
};
Prepared prepare(const PeriodicCorrelationAdmittedReference& reference, const BasisSet& ao,
                 const BasisSet& minimal, const PeriodicSystem& system, std::size_t point,
                 I other, const Options& options, const PeriodicCorrelationBlochIAOOptions& io,
                 const Caps& caps) {
    controls(options, io, caps);
    if (!reference.state_handle()) throw std::invalid_argument("Gaussian Bloch IAO requires an admitted state owner");
    const auto& state = reference.state();
    if (system.dim != 3 || state.periodic_dimension() != 3 || state.is_shift() != std::array<int, 3>{0, 0, 0}
        || point >= state.n_kpoints() || system.unit_cell.empty())
        throw std::invalid_argument("Gaussian Bloch IAO requires a valid full Gamma-mesh 3D point and explicit atoms");
    limit(system.unit_cell.size(), caps.maximum_atom_count, "Gaussian Bloch IAO atom scan cap exceeded");
    limit(add(ao.nshells(), minimal.nshells()), caps.maximum_shell_count, "Gaussian Bloch IAO shell scan cap exceeded");
    if (ao.nbasis() != state.n_basis()) throw std::invalid_argument("Gaussian Bloch IAO AO dimension differs from admitted reference");
    const RegularKMesh mesh(state.mesh());
    Prepared prep; auto& m = prep.plan; auto& d = prep.diagnostics;
    m.point = point; m.conjugate_point = mesh.index(mesh.negate(mesh.address(point)));
    m.n_basis = state.n_basis(); m.n_effective = state.n_effective_orbitals();
    m.n_occupied = add(state.n_correlated_occupied(), state.n_frozen_core()); m.n_minimal = minimal.nbasis();
    const auto iao_plan = plan_periodic_correlation_bloch_iao(m.n_basis, m.n_effective, m.n_occupied, m.n_minimal, io.jacobi_max_sweeps);
    m.overlap_and_label_bytes = iao_plan.borrowed_input_bytes;
    m.panel_pairs = std::min(caps.maximum_panel_pairs, mul(m.n_basis, m.n_basis));
    m.panel_output_bytes = mul(16, m.panel_pairs); m.fixed_fourier_workspace_bytes = ao_pair_fourier_fixed_numeric_workspace_bytes();
    m.iao_owned_peak_bytes = iao_plan.peak_owned_numerical_bytes;
    m.output_numerical_bytes = iao_plan.output_numerical_bytes;
    m.peak_owned_numerical_bytes = add(m.overlap_and_label_bytes,
        std::max(add(m.panel_output_bytes, m.fixed_fourier_workspace_bytes), m.iao_owned_peak_bytes));
    limit(m.peak_owned_numerical_bytes, caps.maximum_owned_numerical_bytes, "Gaussian Bloch IAO owned numerical byte cap exceeded");
    limit(m.peak_owned_numerical_bytes, static_cast<I>(std::numeric_limits<std::ptrdiff_t>::max()), "Gaussian Bloch IAO address extent exceeded");
    m.iao_work_units = iao_plan.maximum_work_units;
    limit(m.iao_work_units, io.maximum_work_units, "Gaussian Bloch IAO native IAO work cap exceeded");
    m.borrowed_geometry_numeric_bytes = add(72, mul(28, system.unit_cell.size())); // A, atom Z/xyz
    Counts count; count.numeric = m.borrowed_geometry_numeric_bytes; count.admit(caps);
    m.ao = inspect(ao, system, count, caps); m.minimal = inspect(minimal, system, count, caps);
    // Basis shell-to-atom indices are discrete payload read in addition to
    // the reusable basis inventory's origin/exponent/coefficient lanes.
    m.borrowed_basis_numeric_bytes = add(add(m.ao.borrowed_active_numeric_bytes,
        m.minimal.borrowed_active_numeric_bytes), mul(4, count.shells));
    const I borrowed = add(m.borrowed_basis_numeric_bytes, m.borrowed_geometry_numeric_bytes);
    limit(borrowed, caps.maximum_borrowed_active_numeric_bytes, "Gaussian Bloch IAO borrowed source cap exceeded");
    m.other_live_numerical_bytes = other; m.total_borrowed_numerical_bytes = add(borrowed, other);
    const auto& dims = reference.dimensions(); const auto& budget = reference.budget();
    m.required_node_memory_bytes = add(add(dims.external_bytes, dims.shared_bytes),
        add(mul(budget.mpi_ranks, add(dims.per_rank_bytes, dims.localization_window_bytes_per_rank)),
            mul(mul(budget.mpi_ranks, budget.workers_per_rank), add(m.peak_owned_numerical_bytes, m.total_borrowed_numerical_bytes))));
    if (!budget.memory_limit_bytes) throw std::invalid_argument("Gaussian Bloch IAO node budget must be positive");
    limit(m.required_node_memory_bytes, budget.memory_limit_bytes, "Gaussian Bloch IAO live owners exceed admitted node memory");
    m.basis_scan_work_units = add(count.work(), mul(512, system.unit_cell.size()));
    const auto roles = streams(ao, minimal); const unsigned nr = m.point == m.conjugate_point ? 4 : 7;
    I pair_count = 0;
    for (unsigned role = 0; role < nr; ++role) {
        const I pairs = mul(roles[role].bra->nbasis(), roles[role].ket->nbasis());
        pair_count = add(pair_count, pairs);
        m.panel_calls = add(m.panel_calls, add(pairs / m.panel_pairs, pairs % m.panel_pairs != 0));
    }
    const I per_scan = add(count.wire, add(count.shells, add(count.contractions, 256)));
    const I scans = mul(m.panel_calls, per_scan);
    const I lookups = mul(pair_count, add(add(count.shells, count.contractions), 32));
    m.preflight_work_units = add(m.basis_scan_work_units,
        mul(512, add(add(scans, lookups), caps.maximum_total_image_candidates)));
    limit(add(m.preflight_work_units, m.iao_work_units), caps.maximum_work_units,
          "Gaussian Bloch IAO preflight work reservation exceeds cap");
    // All scans/capacity checks precede full numeric/hash scans and image walks.
    for (const auto& atom : system.unit_cell) {
        if (atom.Z <= 0 || atom.Z > 118) throw std::invalid_argument("Gaussian Bloch IAO atom labels require physical atomic numbers");
        for (double x : atom.xyz) finite(x);
    }
    if (!system.lattice.allFinite()) throw std::invalid_argument("Gaussian Bloch IAO direct lattice is not finite");
    prep.reciprocal = system.reciprocal_lattice();
    if (!prep.reciprocal.allFinite()) throw std::invalid_argument("Gaussian Bloch IAO reciprocal lattice is not finite");
    for (unsigned i = 0; i < 3; ++i) for (unsigned j = 0; j < 3; ++j) {
        check(prep.reciprocal(i, j), state.reciprocal_lattice()(i, j),
            options.geometry_absolute_tolerance, options.geometry_relative_tolerance,
            d.maximum_geometry_residual, "Gaussian Bloch IAO original direct cell does not match admitted reciprocal lattice");
        double dual = 0;
        for (int a = 2; a >= 0; --a) dual = finite(std::fma(system.lattice(a, i), prep.reciprocal(a, j), dual));
        check(dual, i == j ? 2 * std::acos(-1.0) : 0.0,
            options.geometry_absolute_tolerance, options.geometry_relative_tolerance,
            d.maximum_geometry_residual, "Gaussian Bloch IAO direct-reciprocal duality audit failed");
    }
    prep.k = cartesian(prep.reciprocal, mesh.fractional_at(point));
    prep.kbar = cartesian(prep.reciprocal, mesh.fractional_at(m.conjugate_point));
    for (unsigned a = 0; a < 3; ++a) {
        check(prep.k[a], state.kpoint_cartesian(point)[a], options.geometry_absolute_tolerance,
            options.geometry_relative_tolerance, d.maximum_geometry_residual, "Gaussian Bloch IAO point phase convention mismatch");
        check(prep.kbar[a], state.kpoint_cartesian(m.conjugate_point)[a], options.geometry_absolute_tolerance,
            options.geometry_relative_tolerance, d.maximum_geometry_residual, "Gaussian Bloch IAO conjugate point phase convention mismatch");
    }
    prep.ao_digest = auxiliary_basis_content_identity_sha256(ao);
    prep.minimal_digest = auxiliary_basis_content_identity_sha256(minimal);
    const AuxiliaryFourierVectorView empty{};
    for (unsigned role = 0; role < nr; ++role) {
        const auto& stream = roles[role]; const I pairs = mul(stream.bra->nbasis(), stream.ket->nbasis());
        for (I begin = 0; begin < pairs;) {
            const I rows = std::min(m.panel_pairs, pairs - begin);
            if (m.image_candidate_count == caps.maximum_total_image_candidates)
                throw std::length_error("Gaussian Bloch IAO total image candidate cap exceeded");
            const auto panel = ao_pair_gaussian_fourier_panel(*stream.bra, *stream.ket, system,
                empty, stream.conjugate ? prep.kbar : prep.k, begin, rows, options.image_cutoff_bohr,
                caps.maximum_total_image_candidates - m.image_candidate_count, 1);
            if (!panel.data.empty() || panel.output_bytes != 0)
                throw std::logic_error("Gaussian Bloch IAO count-only Fourier preflight unexpectedly allocated values");
            m.image_candidate_count = add(m.image_candidate_count, panel.image_candidate_count);
            m.retained_pair_image_count = add(m.retained_pair_image_count, panel.retained_pair_image_count);
            begin += rows;
        }
    }
    // A value call scans each image twice (preflight and integration).
    // 262144 primitive-loop units cover MD/pure L<=6 polynomial loops;
    // this is an intentionally conservative loop bound, not a FLOP promise.
    const I primitive_work = mul(262144, mul(m.image_candidate_count, mul(count.max_primitives, count.max_primitives)));
    m.overlap_evaluation_work_units = add(primitive_work,
        mul(512, add(add(scans, lookups), add(mul(2, m.image_candidate_count), mul(8, pair_count)))));
    m.maximum_work_units = add(m.preflight_work_units, add(m.overlap_evaluation_work_units, m.iao_work_units));
    limit(m.maximum_work_units, caps.maximum_work_units, "Gaussian Bloch IAO total counted work cap exceeded");
    limit(add(mul(16, pair_count), 4096), std::numeric_limits<I>::max() / 8,
          "Gaussian Bloch IAO raw source hash extent exceeded");
    d.charged_work_units = m.maximum_work_units;
    return prep;
}

double average(double a, double b) {
    if (a == b) return a;
    int exponent = 0; std::frexp(std::max(std::abs(a), std::abs(b)), &exponent);
    const double x = std::scalbn(a, -exponent), y = std::scalbn(b, -exponent);
    if (std::scalbn(x, exponent) != a || std::scalbn(y, exponent) != b)
        throw std::overflow_error("Gaussian Bloch IAO Hermitian averaging loses input range");
    return finite(std::scalbn(x + y, exponent - 1));
}
void project(std::vector<Z>& cross, std::vector<Z>& minimal, I r, bool trim,
             const Options& o, Diagnostics& d) {
    for (I i = 0; i < r; ++i) for (I j = i; j < r; ++j) {
        const Z a = minimal[i * r + j], b = minimal[j * r + i];
        check(a, std::conj(b), o.structural_absolute_tolerance, o.structural_relative_tolerance,
            d.maximum_minimal_hermitian_defect, "Gaussian Bloch IAO raw minimal overlap is not Hermitian");
        Z value = i == j ? Z(a.real(), 0) : Z(average(a.real(), b.real()), average(a.imag(), -b.imag()));
        if (trim) value = Z(value.real(), 0);
        check(value, a, o.projection_absolute_tolerance, o.projection_relative_tolerance,
            d.maximum_projection_correction, "Gaussian Bloch IAO minimal-overlap projection correction exceeds cap");
        check(std::conj(value), b, o.projection_absolute_tolerance, o.projection_relative_tolerance,
            d.maximum_projection_correction, "Gaussian Bloch IAO minimal-overlap projection correction exceeds cap");
        minimal[i * r + j] = value; minimal[j * r + i] = std::conj(value);
    }
    if (trim) for (auto& value : cross) {
        const Z real(value.real(), 0);
        check(value, real, o.projection_absolute_tolerance, o.projection_relative_tolerance,
            d.maximum_projection_correction, "Gaussian Bloch IAO cross-overlap real projection correction exceeds cap");
        value = real;
    }
}
}  // namespace

PeriodicGaussianBlochIAOMemoryPlan plan_periodic_gaussian_bloch_iao(
    const PeriodicCorrelationAdmittedReference& reference, const BasisSet& ao, const BasisSet& minimal,
    const PeriodicSystem& system, std::size_t point, I other, const Options& o,
    const PeriodicCorrelationBlochIAOOptions& io, const Caps& caps) {
    return prepare(reference, ao, minimal, system, point, other, o, io, caps).plan;
}
PeriodicGaussianBlochIAOPoint::PeriodicGaussianBlochIAOPoint(PeriodicCorrelationBlochIAO&& iao) : iao_(std::move(iao)) {}
const PeriodicCorrelationBlochIAO& PeriodicGaussianBlochIAOPoint::iao() const { (void) iao_.state(); return iao_; }

PeriodicGaussianBlochIAOPoint make_periodic_gaussian_bloch_iao(
    const PeriodicCorrelationAdmittedReference& reference, const BasisSet& ao, const BasisSet& minimal,
    const PeriodicSystem& system, std::size_t point, I other, const Options& options,
    const PeriodicCorrelationBlochIAOOptions& io, const Caps& caps) {
    auto prep = prepare(reference, ao, minimal, system, point, other, options, io, caps);
    const auto& m = prep.plan; auto& d = prep.diagnostics;
    const auto& state = reference.state(); const I b = m.n_basis, r = m.n_minimal;
    const bool trim = m.point == m.conjugate_point;
    std::vector<Z> cross(b * r), small(r * r);
    std::vector<I> labels(r);
    I column = 0;
    for (std::size_t s = 0; s < minimal.nshells(); ++s)
        for (const auto& contraction : minimal.libint()[s].contr)
            for (int component = 0; component < 2 * contraction.l + 1; ++component)
                labels[column++] = static_cast<I>(minimal.shell_atom_index(s));
    if (column != r) throw std::logic_error("Gaussian Bloch IAO minimal atom-label extent changed");
    const double zero = 0;
    const AuxiliaryFourierVectorView vector{&zero, &zero, &zero, 1, 1, 1, 1};
    Digest raw("vibeqc.periodic.gaussian-bloch-iao.raw-overlaps");
    raw.u64(point); raw.u64(m.conjugate_point); raw.u64(b); raw.u64(r);
    const auto roles = streams(ao, minimal); const unsigned nr = trim ? 4 : 7;
    raw.u32(nr);
    for (unsigned role = 0; role < nr; ++role) {
        const auto& stream = roles[role]; const I width = stream.ket->nbasis();
        const I pairs = mul(stream.bra->nbasis(), width); raw.u64(role); raw.u64(pairs);
        for (I begin = 0; begin < pairs;) {
            const I rows = std::min(m.panel_pairs, pairs - begin);
            if (d.evaluated_image_candidate_count == m.image_candidate_count)
                throw std::logic_error("Gaussian Bloch IAO image enumeration changed after preflight");
            const auto panel = ao_pair_gaussian_fourier_panel(*stream.bra, *stream.ket, system,
                vector, stream.conjugate ? prep.kbar : prep.k, begin, rows, options.image_cutoff_bohr,
                m.image_candidate_count - d.evaluated_image_candidate_count, m.panel_output_bytes);
            ++d.evaluated_panel_count;
            d.evaluated_image_candidate_count = add(d.evaluated_image_candidate_count, panel.image_candidate_count);
            d.evaluated_pair_image_count = add(d.evaluated_pair_image_count, panel.retained_pair_image_count);
            for (I row = 0; row < rows; ++row) {
                const I index = begin + row, i = index / width, j = index % width;
                const Z value = panel.data[row]; raw.value(value);
                if (trim) check(value, Z(value.real(), 0), options.structural_absolute_tolerance,
                    options.structural_relative_tolerance, d.maximum_trim_imaginary_magnitude,
                    "Gaussian Bloch IAO raw TRIM overlap is not real");
                if (role == 0 || role == 4) {
                    const I k = stream.conjugate ? m.conjugate_point : m.point;
                    const I l = stream.conjugate ? m.point : m.conjugate_point;
                    check(value, state.overlap(k)(i, j), options.overlap_absolute_tolerance,
                        options.overlap_relative_tolerance, d.maximum_s11_reference_residual,
                        "Gaussian Bloch IAO regenerated S11 does not match admitted overlap");
                    check(value, std::conj(state.overlap(l)(i, j)), options.overlap_absolute_tolerance,
                        options.overlap_relative_tolerance, d.maximum_s11_reference_conjugacy_residual,
                        "Gaussian Bloch IAO S11 does not match conjugate admitted overlap");
                } else if (role == 1) cross[index] = value;
                else if (role == 2) small[index] = value;
                else if (role == 3) check(value, std::conj(cross[j * r + i]), options.structural_absolute_tolerance,
                    options.structural_relative_tolerance, d.maximum_cross_adjoint_residual,
                    "Gaussian Bloch IAO raw S21 differs from S12 adjoint");
                else check(value, std::conj(role == 5 ? cross[index] : small[index]), options.structural_absolute_tolerance,
                    options.structural_relative_tolerance, d.maximum_overlap_time_reversal_residual,
                    "Gaussian Bloch IAO independently generated opposite-k overlap violates time reversal");
            }
            begin += rows;
        } // panel released before next panel and before IAO workspace
    }
    if (d.evaluated_panel_count != m.panel_calls || d.evaluated_image_candidate_count != m.image_candidate_count
        || d.evaluated_pair_image_count != m.retained_pair_image_count)
        throw std::logic_error("Gaussian Bloch IAO count/value enumeration changed");
    const std::string raw_identity = raw.finish();
    project(cross, small, r, trim, options, d);
    Digest source("vibeqc.periodic.gaussian-bloch-iao.source");
    source.text(prep.ao_digest); source.text(prep.minimal_digest); source.text(raw_identity);
    source.text("Sun10/16;p=0;positive-ket-phase;no-Nk-or-volume-factor;original-direct-cell;"
                "separation-cutoff;lexicographic-images;padded-long-double-box;no-infinite-tail-bound;"
                "fixed-reverse-index-FMA-cartesian;Gamma-mesh;basis-standard-pure-L<=6");
    source.u64(point); source.u64(m.conjugate_point);
    for (int n : state.mesh()) source.u32(static_cast<std::uint32_t>(n));
    for (unsigned i = 0; i < 3; ++i) for (unsigned j = 0; j < 3; ++j) {
        source.real(system.lattice(i, j)); source.real(prep.reciprocal(i, j));
    }
    source.real(options.image_cutoff_bohr); source.u64(system.unit_cell.size());
    for (const auto& atom : system.unit_cell) { source.u32(atom.Z); for (double x : atom.xyz) source.real(x); }
    for (I label : labels) source.u64(label);
    source.u64(m.image_candidate_count); source.u64(m.retained_pair_image_count);
    const std::string source_identity = source.finish();
    const PeriodicCorrelationBlochIAOInputProvenance provenance{
        source_identity, prep.minimal_digest, add(m.overlap_and_label_bytes, m.total_borrowed_numerical_bytes)};
    auto iao = make_periodic_correlation_bloch_iao(reference, point, cross.data(), cross.size(),
        small.data(), small.size(), labels.data(), labels.size(), r, provenance, m.iao_owned_peak_bytes, io);
    PeriodicGaussianBlochIAOPoint output(std::move(iao));
    output.memory_ = m; output.diagnostics_ = d; output.options_ = options;
    output.ao_identity_ = prep.ao_digest; output.minimal_identity_ = prep.minimal_digest;
    output.raw_identity_ = raw_identity; output.source_identity_ = source_identity;
    Digest match("vibeqc.periodic.gaussian-bloch-iao.reference-match");
    match.text(source_identity); match.text(state.state_identity_sha256()); match.text(output.iao_.iao_identity_sha256());
    match.text("numerical-S11-match-only;HF-AO-basis-source-not-authenticated;no-Fock-Hcore-Coulomb-certificate");
    for (double x : {options.geometry_absolute_tolerance, options.geometry_relative_tolerance,
            options.overlap_absolute_tolerance, options.overlap_relative_tolerance,
            options.structural_absolute_tolerance, options.structural_relative_tolerance,
            options.projection_absolute_tolerance, options.projection_relative_tolerance,
            d.maximum_geometry_residual, d.maximum_s11_reference_residual,
            d.maximum_s11_reference_conjugacy_residual, d.maximum_cross_adjoint_residual,
            d.maximum_overlap_time_reversal_residual, d.maximum_minimal_hermitian_defect,
            d.maximum_trim_imaginary_magnitude, d.maximum_projection_correction}) match.real(x);
    output.match_identity_ = match.finish();
    return output;
}
}  // namespace vibeqc
