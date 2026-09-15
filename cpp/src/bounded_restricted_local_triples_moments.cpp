#include "vibeqc/bounded_restricted_local_triples_moments.hpp"
#include "vibeqc/detail/sha256.hpp"

#include <algorithm>
#include <cfenv>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace vibeqc {
namespace {
using U = std::uint64_t;
using Input = BoundedRestrictedLocalTriplesMomentsInput;
using Options = BoundedRestrictedLocalTriplesMomentsOptions;
using Inventory = BoundedRestrictedLocalTriplesMomentsInventory;
using Caps = BoundedRestrictedLocalTriplesMomentsCaps;
using Plan = BoundedRestrictedLocalTriplesMomentsMemoryPlan;
using Result = BoundedRestrictedLocalTriplesMomentsResult;
using Reader = BoundedRestrictedPairCCSDAmplitudes;
using View = BoundedRestrictedCCSDRealView;
using Integrals = BoundedRestrictedCCSDIntegralProvider;
static_assert(sizeof(double) == 8 && sizeof(std::size_t) == 8
    && std::numeric_limits<double>::is_iec559, "local TNO moments require 64-bit IEEE storage");
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0) || FLT_EVAL_METHOD != 0
#error "local TNO moments require strict binary64 evaluation"
#endif
U add(U a, U b) {
    if (b > std::numeric_limits<U>::max() - a)
        throw std::overflow_error("local TNO moments count addition overflow");
    return a + b;
}
U mul(U a, U b) {
    if (a && b > std::numeric_limits<U>::max() / a)
        throw std::overflow_error("local TNO moments count multiplication overflow");
    return a * b;
}
U square(U a) { return mul(a, a); }
U cube(U a) { return mul(square(a), a); }
void extent(U bytes) {
    if (bytes > static_cast<U>(std::numeric_limits<std::ptrdiff_t>::max())
        || bytes > std::numeric_limits<U>::max() / 8 - 65536)
        throw std::overflow_error("local TNO moments address or SHA extent overflow");
}
void view(View v, U count) {
    const auto address = reinterpret_cast<std::uintptr_t>(v.data);
    const U bytes = mul(8, count);
    extent(bytes);
    if (v.element_count != count || !v.data || address % alignof(double)
        || bytes > std::numeric_limits<std::uintptr_t>::max() - address)
        throw std::invalid_argument("local TNO moments require exact aligned borrowed views");
}
double finite(double x) {
    if (!std::isfinite(x)) throw std::overflow_error("local TNO moments nonfinite arithmetic or integral");
    return x;
}
void environment() {
    volatile double tiny = std::numeric_limits<double>::denorm_min();
    volatile double twice = tiny + tiny;
    volatile double recovered = twice * 0.5;
    if (FLT_RADIX != 2 || std::numeric_limits<double>::digits != 53
        || std::numeric_limits<double>::max_exponent != 1024
        || std::fegetround() != FE_TONEAREST || tiny == 0 || twice == 0 || recovered != tiny)
        throw std::runtime_error("local TNO moments require nearest binary64 and gradual underflow");
}
struct Sum {
    double value = 0, correction = 0;
    void include(double x) {
        finite(x);
        const auto next = finite(value + x);
        correction = finite(correction + (std::abs(value) >= std::abs(x)
            ? (value - next) + x : (x - next) + value));
        value = next;
    }
    double total() const { return finite(value + correction); }
};
struct Digest {
    detail::Sha256 h;
    explicit Digest(const char* domain) {
        h.update(reinterpret_cast<const std::uint8_t*>(domain), std::strlen(domain));
        integer(1);
    }
    void integer(U x) {
        std::array<std::uint8_t,8> bytes{};
        for (U b = 0; b < 8; ++b) bytes[b] = static_cast<std::uint8_t>(x >> (56 - 8 * b));
        h.update(bytes.data(), bytes.size());
    }
    void number(double x) {
        if (!std::isfinite(x)) throw std::invalid_argument("local TNO moments source must be finite");
        if (x == 0) x = 0;
        U bits = 0;
        std::memcpy(&bits, &x, 8);
        integer(bits);
    }
    void text(const std::string& x) {
        integer(x.size());
        h.update(reinterpret_cast<const std::uint8_t*>(x.data()), x.size());
    }
    void values(View v) {
        integer(v.element_count);
        for (U a = 0; a < v.element_count; ++a) number(v.data[a]);
    }
};
void controls(const Options& options, const Inventory& inventory) {
    if (!std::isfinite(options.coefficient_orthogonality_tolerance)
        || options.coefficient_orthogonality_tolerance <= 0
        || options.coefficient_orthogonality_tolerance >= 1)
        throw std::invalid_argument("local TNO moments Gram tolerance must be explicit in (0,1)");
    if (!std::isfinite(options.amplitude_symmetry_tolerance) || options.amplitude_symmetry_tolerance < 0)
        throw std::invalid_argument("local TNO moments symmetry tolerance must be nonnegative finite");
    if (!options.maximum_integral_work_units_per_call || !inventory.numerical_replicas
        || !inventory.fixed_backend_margin_bytes_per_replica)
        throw std::invalid_argument("local TNO moments integral work, replicas and backend margin must be positive");
}
void limit(U value, U cap, const char* message) {
    if (!cap || value > cap) throw std::length_error(message);
}
void admit(const Plan& p, const Caps& caps) {
    limit(p.n_occupied, caps.maximum_occupied_count, "local TNO moments occupied cap exceeded");
    limit(p.common_virtual_dimension, caps.maximum_common_virtual_dimension, "local TNO moments common dimension cap exceeded");
    limit(p.rank, caps.maximum_rank, "local TNO moments rank cap exceeded");
    limit(p.peak_owned_numerical_bytes, caps.maximum_owned_numerical_bytes, "local TNO moments owned byte cap exceeded");
    limit(p.per_replica_inventoried_bytes, caps.maximum_per_replica_inventoried_bytes, "local TNO moments replica byte cap exceeded");
    limit(p.required_node_inventoried_bytes, caps.maximum_node_inventoried_bytes, "local TNO moments node byte cap exceeded");
    limit(p.common_singles_calls, caps.maximum_common_singles_calls, "local TNO moments common singles cap exceeded");
    limit(p.common_doubles_calls, caps.maximum_common_doubles_calls, "local TNO moments common doubles cap exceeded");
    limit(p.transformed_integral_calls, caps.maximum_transformed_integral_calls, "local TNO moments transformed integral cap exceeded");
    limit(p.common_integral_calls, caps.maximum_common_integral_calls, "local TNO moments common integral cap exceeded");
    limit(p.work_units_upper_bound, caps.maximum_work_units, "local TNO moments work cap exceeded");
}
std::string coefficient_identity(View coefficients, U n, U r) {
    Digest d("vibeqc.local-tno-moments.coefficients");
    d.integer(n); d.integer(r); d.values(coefficients);
    return d.h.finish_hex();
}
std::string source_identity(const Input& in, const Plan& p, const Options& options,
    const std::string& reader_identity) {
    Digest d("vibeqc.local-tno-moments.input");
    d.text(reader_identity); d.integer(p.n_occupied); d.integer(p.common_virtual_dimension);
    d.integer(p.rank); d.integer(in.ccsd_snapshot_id);
    for (auto label : in.occupied) d.integer(label);
    d.values(in.target_coefficients); d.values(in.original_f_ov);
    d.number(options.coefficient_orthogonality_tolerance); d.number(options.amplitude_symmetry_tolerance);
    d.integer(options.maximum_integral_work_units_per_call);
    d.integer(p.integral_retained_numerical_bytes); d.integer(p.integral_maximum_transient_numerical_bytes);
    return d.h.finish_hex();
}
double validate_frame(const Input& in, const Plan& p, const Options& options) {
    const auto n = p.common_virtual_dimension, r = p.rank;
    // Explicit finite scans precede products. The original block is not
    // replaced by C-projected small values: every original lane must be zero.
    for (U x = 0; x < n * r; ++x)
        if (!std::isfinite(in.target_coefficients.data[x]))
            throw std::invalid_argument("local TNO moments source must be finite");
    for (U x = 0; x < p.n_occupied * n; ++x) {
        const auto f = in.original_f_ov.data[x];
        if (!std::isfinite(f)) throw std::invalid_argument("local TNO moments source must be finite");
        if (f != 0) throw std::invalid_argument("local TNO moments require exactly zero ORIGINAL Fov");
    }
    double error = 0;
    for (U a = 0; a < r; ++a) for (U b = 0; b < r; ++b) {
        Sum gram;
        for (U x = 0; x < n; ++x)
            gram.include(in.target_coefficients.data[x * r + a] * in.target_coefficients.data[x * r + b]);
        error = finite(std::hypot(error, finite(gram.total() - (a == b ? 1.0 : 0.0))));
    }
    if (error > options.coefficient_orthogonality_tolerance)
        throw std::invalid_argument("local TNO moments target frame is not orthonormal");
    return error;
}
struct ProjectedAmplitudes {
    const Reader& reader;
    const double* coefficients;
    U o, n, r, singles_limit, doubles_limit;
    mutable U singles_calls = 0, doubles_calls = 0;
    static double singles(U i, U a, const void* context) {
        const auto& self = *static_cast<const ProjectedAmplitudes*>(context);
        if (i >= self.o || a >= self.r) throw std::out_of_range("local TNO moments singles label out of range");
        if (self.n > self.singles_limit - self.singles_calls)
            throw std::length_error("local TNO moments exhausted common singles census");
        Sum sum;
        for (U x = 0; x < self.n; ++x) {
            ++self.singles_calls;
            sum.include(self.coefficients[x * self.r + a] * self.reader.singles(i, x));
        }
        return sum.total();
    }
    static double doubles(U i, U j, U a, U b, const void* context) {
        const auto& self = *static_cast<const ProjectedAmplitudes*>(context);
        if (i >= self.o || j >= self.o || a >= self.r || b >= self.r)
            throw std::out_of_range("local TNO moments doubles label out of range");
        if (square(self.n) > self.doubles_limit - self.doubles_calls)
            throw std::length_error("local TNO moments exhausted common doubles census");
        // Reader construction proves exact pair covariance and exact diagonal
        // local symmetry. Choose one represented projection for that algebraic
        // orbit; this is not repair of arbitrary caller-supplied amplitudes.
        if (i > j) { std::swap(i, j); std::swap(a, b); }
        if (i == j && a > b) std::swap(a, b);
        Sum sum;
        for (U x = 0; x < self.n; ++x) {
            Sum row;
            for (U y = 0; y < self.n; ++y) {
                ++self.doubles_calls;
                row.include(self.reader.doubles(i, j, x, y) * self.coefficients[y * self.r + b]);
            }
            sum.include(self.coefficients[x * self.r + a] * row.total());
        }
        return sum.total();
    }
};
struct TransformedIntegrals {
    Integrals original;
    const double* coefficients;
    U o, n, r, transformed_limit, common_limit, per_query_limit, work_per_common, work_limit;
    U transformed_calls = 0, common_calls = 0, work = 0;
    std::array<U,5> virtual_leg_census{};
    Digest receipt{"vibeqc.local-tno-moments.consumed-common-integrals"};
    static double value(U p, U q, U rlabel, U s, void* context) {
        auto& self = *static_cast<TransformedIntegrals*>(context);
        const std::array<U,4> labels{p,q,rlabel,s};
        std::array<U,4> lengths{1,1,1,1};
        U legs = 0, required = 1;
        for (U slot = 0; slot < 4; ++slot) {
            if (labels[slot] >= self.o + self.r)
                throw std::out_of_range("local TNO moments transformed integral label out of range");
            if (labels[slot] >= self.o) { lengths[slot] = self.n; ++legs; required = mul(required, self.n); }
        }
        // Admit the full scalar transform BEFORE loops or opaque callbacks.
        const U required_work = mul(required, self.work_per_common);
        if (self.transformed_calls >= self.transformed_limit || required > self.per_query_limit
            || required > self.common_limit - self.common_calls || required_work > self.work_limit - self.work)
            throw std::length_error("local TNO moments exhausted integral call or work census");
        ++self.transformed_calls; ++self.virtual_leg_census[legs];
        Sum sum;
        for (U x0 = 0; x0 < lengths[0]; ++x0) for (U x1 = 0; x1 < lengths[1]; ++x1)
            for (U x2 = 0; x2 < lengths[2]; ++x2) for (U x3 = 0; x3 < lengths[3]; ++x3) {
                const std::array<U,4> x{x0,x1,x2,x3};
                std::array<U,4> common = labels;
                double coefficient = 1.0;
                for (U slot = 0; slot < 4; ++slot) if (labels[slot] >= self.o) {
                    common[slot] = self.o + x[slot];
                    coefficient = finite(coefficient * self.coefficients[x[slot] * self.r + labels[slot] - self.o]);
                }
                ++self.common_calls; self.work += self.work_per_common;
                const double eri = self.original.value(common[0], common[1], common[2], common[3], self.original.context);
                environment(); finite(eri);
                for (auto label : common) self.receipt.integer(label);
                self.receipt.number(eri);
                sum.include(coefficient * eri);
            }
        return sum.total();
    }
};
constexpr U fixed_controls = 65536 + 3 * sizeof(Plan) + 2 * sizeof(Result)
    + 2 * sizeof(Input) + 2 * sizeof(Options) + 2 * sizeof(Inventory) + sizeof(Caps)
    + sizeof(ProjectedAmplitudes) + sizeof(TransformedIntegrals) + 4 * sizeof(Digest)
    + 8 * sizeof(std::string) + 2 * sizeof(std::vector<double>);
Plan finish_plan(const BoundedRestrictedPairCCSDAmplitudesMemoryPlan& source, U r,
    U integral_retained, U integral_transient, const Options& options, const Inventory& inventory) {
    controls(options, inventory);
    const U o = source.n_occupied, n = source.common_virtual_dimension;
    if (!o || !n || !r || r > n)
        throw std::invalid_argument("local TNO moments require positive occupied/common/rank dimensions and rank<=common");
    (void)add(o, n); (void)add(o, r);
    const U nr = mul(n,r), on = mul(o,n), orank = mul(o,r), r3 = cube(r);
    Plan p;
    p.n_occupied = o; p.common_virtual_dimension = n; p.rank = r;
    p.reader_borrowed_numerical_bytes = source.borrowed_numerical_bytes;
    p.reader_borrowed_table_bytes = source.borrowed_table_bytes;
    p.reader_control_storage_bytes = source.fixed_control_storage_bytes;
    p.target_coefficient_bytes = mul(8,nr); p.original_f_ov_bytes = mul(8,on);
    p.integral_retained_numerical_bytes = integral_retained;
    p.integral_maximum_transient_numerical_bytes = integral_transient;
    p.projected_zero_f_ov_bytes = mul(8,orank); p.output_numerical_bytes = mul(16,r3);
    p.peak_owned_numerical_bytes = add(p.projected_zero_f_ov_bytes,p.output_numerical_bytes);
    p.maximum_projected_singles_work_units_per_query = add(mul(n,source.maximum_singles_work_units_per_query),mul(64,add(n,1)));
    p.maximum_projected_doubles_work_units_per_query = add(mul(square(n),source.maximum_doubles_work_units_per_query),mul(128,add(square(n),1)));
    BoundedRestrictedCCSDAmplitudeProvider amplitudes;
    amplitudes.retained_numerical_bytes = add(p.reader_borrowed_numerical_bytes,p.target_coefficient_bytes);
    amplitudes.maximum_singles_work_units_per_query = p.maximum_projected_singles_work_units_per_query;
    amplitudes.maximum_doubles_work_units_per_query = p.maximum_projected_doubles_work_units_per_query;
    p.moments = plan_bounded_restricted_triples_moments_accessor(o,r,amplitudes,
        integral_retained,integral_transient);
    p.common_singles_calls = mul(n,p.moments.singles_calls);
    p.common_doubles_calls = mul(square(n),p.moments.doubles_calls);
    p.transformed_integral_calls = p.moments.kernel.integral_calls;
    p.one_virtual_integral_calls = mul(mul(6,o),r3);
    p.two_virtual_integral_calls = mul(3,r3);
    p.three_virtual_integral_calls = mul(mul(6,r),r3);
    p.common_integral_calls = add(add(mul(p.one_virtual_integral_calls,n),
        mul(p.two_virtual_integral_calls,square(n))),mul(p.three_virtual_integral_calls,cube(n)));
    p.maximum_common_integral_calls_per_transformed_query = square(square(n));
    p.control_storage_reservation_bytes = add(fixed_controls,add(p.reader_borrowed_table_bytes,
        add(p.reader_control_storage_bytes,p.moments.fixed_control_storage_bytes)));
    // The child total already counts output, the live generated zero Fov,
    // actual reader/C payload and original ERI retained/transient roles once.
    p.per_replica_inventoried_bytes = add(add(p.moments.kernel.total_live_numerical_bytes,p.original_f_ov_bytes),
        add(p.control_storage_reservation_bytes,add(inventory.other_live_bytes_per_replica,inventory.fixed_backend_margin_bytes_per_replica)));
    p.required_node_inventoried_bytes = add(inventory.external_node_bytes,
        mul(inventory.numerical_replicas,p.per_replica_inventoried_bytes));
    p.validation_work_units = mul(2,add(source.validation_work_units,
        add(mul(4096,add(add(nr,on),1)),mul(128,mul(n,square(r))))));
    p.projection_work_units = p.moments.amplitude_work_units_upper_bound;
    // Includes label construction, finite/FP checks and streamed SHA receipt
    // for each actual common ERI. Opaque callback work is additional.
    p.integral_work_units = add(mul(4096,add(p.common_integral_calls,p.transformed_integral_calls)),
        mul(p.common_integral_calls,options.maximum_integral_work_units_per_call));
    p.work_units_upper_bound = add(add(p.validation_work_units,p.projection_work_units),
        add(p.integral_work_units,add(p.moments.kernel.kernel_work_units_upper_bound,
            mul(4096,add(add(orank,r3),1)))));
    for (U bytes : {p.peak_owned_numerical_bytes,p.original_f_ov_bytes,p.target_coefficient_bytes,
            p.control_storage_reservation_bytes,p.required_node_inventoried_bytes}) extent(bytes);
    if (orank > std::vector<double>().max_size())
        throw std::overflow_error("local TNO moments zero-block vector extent overflow");
    return p;
}
} // namespace

Plan plan_bounded_restricted_local_triples_moments(const Input& in, const Reader& reader,
    const Integrals& integrals, const Options& options, const Inventory& inventory) {
    return finish_plan(reader.memory(),in.rank,integrals.retained_numerical_bytes,
        integrals.maximum_transient_numerical_bytes,options,inventory);
}

Plan plan_bounded_restricted_local_triples_moments_upper(U o, U n, U r, U singles_rank, U pair_rank,
    U integral_retained, U integral_transient, const Options& options, const Inventory& inventory) {
    controls(options,inventory);
    const BoundedRestrictedPairCCSDAmplitudesInventory reader_inventory{inventory.numerical_replicas,
        inventory.external_node_bytes,inventory.other_live_bytes_per_replica,inventory.fixed_backend_margin_bytes_per_replica};
    const auto source = plan_bounded_restricted_pair_ccsd_amplitudes_upper(o,n,singles_rank,pair_rank,reader_inventory);
    auto p = finish_plan(source,r,integral_retained,integral_transient,options,inventory);
    p.uniform_rank_upper_bound = true;
    return p;
}

Result bounded_restricted_local_triples_moments(const Input& supplied, const Reader& reader,
    const Integrals& supplied_integrals, const Options& supplied_options,
    const Inventory& supplied_inventory, const Caps& supplied_caps) {
    const auto input = supplied; const auto integrals = supplied_integrals;
    const auto options = supplied_options; const auto inventory = supplied_inventory; const auto caps = supplied_caps;
    const auto p = plan_bounded_restricted_local_triples_moments(input,reader,integrals,options,inventory);
    admit(p,caps);
    if (!integrals.value) throw std::invalid_argument("local TNO moments require an original common integral callback");
    if (!input.ccsd_snapshot_id) throw std::invalid_argument("local TNO moments require a positive CCSD snapshot");
    for (auto label : input.occupied)
        if (label >= p.n_occupied) throw std::out_of_range("local TNO moments occupied label out of range");
    view(input.target_coefficients,mul(p.common_virtual_dimension,p.rank));
    view(input.original_f_ov,mul(p.n_occupied,p.common_virtual_dimension));
    environment(); reader.validate_immutable_snapshot();
    Result result;
    result.memory_ = p;
    result.gram_error_ = validate_frame(input,p,options);
    result.reader_ = reader.snapshot_identity_sha256();
    result.coefficients_ = coefficient_identity(input.target_coefficients,p.common_virtual_dimension,p.rank);
    result.input_ = source_identity(input,p,options,result.reader_);
    std::vector<double> zero_fov(static_cast<std::size_t>(p.projected_zero_f_ov_bytes/8));
    ProjectedAmplitudes projected{reader,input.target_coefficients.data,p.n_occupied,p.common_virtual_dimension,p.rank,
        p.common_singles_calls,p.common_doubles_calls};
    TransformedIntegrals transformed{integrals,input.target_coefficients.data,p.n_occupied,p.common_virtual_dimension,p.rank,
        p.transformed_integral_calls,p.common_integral_calls,p.maximum_common_integral_calls_per_transformed_query,
        options.maximum_integral_work_units_per_call,mul(p.common_integral_calls,options.maximum_integral_work_units_per_call)};
    const BoundedRestrictedCCSDAmplitudeProvider amplitudes{&ProjectedAmplitudes::singles,&ProjectedAmplitudes::doubles,&projected,
        add(p.reader_borrowed_numerical_bytes,p.target_coefficient_bytes),0,
        p.maximum_projected_singles_work_units_per_query,p.maximum_projected_doubles_work_units_per_query};
    const Integrals target_integrals{&TransformedIntegrals::value,&transformed,
        integrals.retained_numerical_bytes,integrals.maximum_transient_numerical_bytes};
    const BoundedRestrictedTriplesMomentsOperatorInput op{p.n_occupied,p.rank,input.occupied,input.ccsd_snapshot_id,
        {zero_fov.data(),zero_fov.size()},options.amplitude_symmetry_tolerance};
    BoundedRestrictedTriplesMomentsAccessorCaps leaf;
    leaf.kernel = {p.moments.kernel.peak_owned_numerical_bytes,p.moments.kernel.total_live_numerical_bytes,
        p.moments.kernel.integral_calls,p.moments.kernel.kernel_work_units_upper_bound};
    leaf.maximum_singles_calls = p.moments.singles_calls; leaf.maximum_doubles_calls = p.moments.doubles_calls;
    leaf.maximum_amplitude_work_units = p.moments.amplitude_work_units_upper_bound;
    leaf.maximum_control_storage_bytes = p.moments.fixed_control_storage_bytes;
    result.moments_ = bounded_restricted_triples_moments_accessor(op,amplitudes,target_integrals,leaf);
    environment(); reader.validate_immutable_snapshot();
    (void)validate_frame(input,p,options);
    if (reader.snapshot_identity_sha256() != result.reader_
        || source_identity(input,p,options,result.reader_) != result.input_)
        throw std::invalid_argument("local TNO moments immutable source changed during production");
    if (projected.singles_calls != p.common_singles_calls || projected.doubles_calls != p.common_doubles_calls
        || transformed.transformed_calls != p.transformed_integral_calls || transformed.common_calls != p.common_integral_calls
        || transformed.work != mul(p.common_integral_calls,options.maximum_integral_work_units_per_call)
        || transformed.virtual_leg_census[0] || transformed.virtual_leg_census[4]
        || transformed.virtual_leg_census[1] != p.one_virtual_integral_calls
        || transformed.virtual_leg_census[2] != p.two_virtual_integral_calls
        || transformed.virtual_leg_census[3] != p.three_virtual_integral_calls)
        throw std::logic_error("local TNO moments consumed query census is incomplete");
    result.singles_calls_ = projected.singles_calls; result.doubles_calls_ = projected.doubles_calls;
    result.integral_calls_ = transformed.common_calls; result.transformed_calls_ = transformed.transformed_calls;
    result.integrals_ = transformed.receipt.h.finish_hex();
    Digest payload("vibeqc.local-tno-moments.output");
    payload.text(result.input_); payload.text(result.integrals_);
    const auto& moments = result.moments_.moments;
    payload.values({moments.connected.data(),moments.connected.size()});
    payload.values({moments.singles.data(),moments.singles.size()});
    result.payload_ = payload.h.finish_hex();
    return result;
}

} // namespace vibeqc
