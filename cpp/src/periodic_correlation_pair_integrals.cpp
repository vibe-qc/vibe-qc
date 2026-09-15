#include "vibeqc/periodic_correlation_pair_integrals.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "periodic_correlation_local_factors_internal.hpp"

namespace vibeqc {
namespace {
using U = std::uint64_t;
using C = std::complex<double>;
using Selection = PeriodicCorrelationPairIntegralSelection;
using Memory = PeriodicCorrelationPairIntegralMemoryPlan;
using Digest = periodic_correlation_local_detail::Digest;
static_assert(sizeof(C) == 16, "pair-integral accounting requires complex128");
U add(U a, U b) {
    if (b > std::numeric_limits<U>::max() - a)
        throw std::overflow_error("pair integral count addition overflows uint64");
    return a + b;
}
U mul(U a, U b) {
    if (a && b > std::numeric_limits<U>::max() / a)
        throw std::overflow_error("pair integral count multiplication overflows uint64");
    return a * b;
}
void finite(C x) {
    if (!std::isfinite(x.real()) || !std::isfinite(x.imag()))
        throw std::overflow_error("pair integral contraction has a non-finite lane");
}
void accumulate(C term, C& value, C& correction) {
    finite(term);
    const C next = value + term;
    finite(next);
    const double re = std::abs(value.real()) >= std::abs(term.real())
        ? (value.real() - next.real()) + term.real() : (term.real() - next.real()) + value.real();
    const double im = std::abs(value.imag()) >= std::abs(term.imag())
        ? (value.imag() - next.imag()) + term.imag() : (term.imag() - next.imag()) + value.imag();
    correction += C(re, im); finite(correction); value = next;
}
PeriodicCorrelationLocalOrbitalSelection local(const Selection& s, bool right, U begin, U count) {
    PeriodicCorrelationLocalOrbitalSelection out;
    out.occupied_index = right ? s.occupied_j : s.occupied_i;
    out.occupied_cell = right ? s.cell_j : s.cell_i;
    out.virtual_begin = begin; out.virtual_count = count;
    out.virtual_translation_cell = s.virtual_translation_cell;
    return out;
}
void selection_digest(Digest& d, const Selection& s) {
    d.u64(s.occupied_i); d.u64(s.cell_i); d.u64(s.occupied_j); d.u64(s.cell_j);
    d.u64(s.virtual_begin); d.u64(s.virtual_count);
    d.u64(s.virtual_translation_cell); d.u64(s.virtual_block);
}
void caps(const PeriodicCorrelationAdmittedReference& r, const Memory& m,
          const PeriodicCorrelationPairIntegralCaps& c) {
    if (!c.maximum_owned_numerical_bytes || c.maximum_owned_numerical_bytes < m.peak_owned_numerical_bytes)
        throw std::length_error("pair integral owned numerical byte cap is missing or exceeded");
    if (!c.maximum_work_units || c.maximum_work_units < m.work_units_upper_bound
        || !c.maximum_factor_builds || c.maximum_factor_builds < m.factor_builds
        || !c.maximum_tile_visits || c.maximum_tile_visits < m.tile_visits_upper_bound)
        throw std::length_error("pair integral work, factor-build or tile-visit cap is missing or exceeded");
    if (!r.budget().memory_limit_bytes || m.required_node_memory_bytes > r.budget().memory_limit_bytes)
        throw std::length_error("pair integral outer and inner live inventory exceeds admitted node memory");
}
}  // namespace

Memory plan_periodic_correlation_pair_integral_block(
    const PeriodicCorrelationAdmittedReference& r, const PeriodicCorrelationFactorStreamSchedule& schedule,
    const PeriodicCorrelationPrivateFactorReader& reader, const PeriodicCorrelationWannier& w,
    const PeriodicCorrelationPAODomain& domain, const PeriodicCorrelationPAOSpace& space,
    const Selection& s) {
    if (!s.virtual_count || !s.virtual_block || s.virtual_begin >= space.retained_dimension()
        || s.virtual_count > space.retained_dimension() - s.virtual_begin)
        throw std::invalid_argument("pair integral virtual selection or block is invalid");
    const U vb = std::min(s.virtual_block, s.virtual_count);
    const auto& shape = schedule.shape();
    const U ab = std::min(shape.auxiliary_block, shape.n_auxiliary);
    Memory m;
    m.maximum_factor = plan_periodic_correlation_local_factor_block(r, schedule, reader, w,
        domain, space, local(s, false, s.virtual_begin, vb), 0, 0, ab);
    // Validate the second occupied label before any allocation or disk read.
    (void) plan_periodic_correlation_local_coefficient_panel(r, w, domain, space,
        local(s, true, s.virtual_begin, vb));
    const auto& f = m.maximum_factor;
    m.n_virtual = s.virtual_count;
    m.retained_output_bytes = mul(16, mul(m.n_virtual, m.n_virtual));
    m.compensation_bytes = m.retained_output_bytes;
    m.retained_left_factor_bytes = f.retained_output_bytes;
    m.maximum_factor_owned_bytes = f.peak_owned_numerical_bytes;
    const U extra = add(add(m.retained_output_bytes, m.compensation_bytes), m.retained_left_factor_bytes);
    m.peak_owned_numerical_bytes = add(extra, m.maximum_factor_owned_bytes);
    const U nvblocks = (s.virtual_count - 1) / vb + 1;
    m.factor_builds = mul(mul(shape.n_kpoints, shape.auxiliary_tile_count),
                          mul(nvblocks, add(1, nvblocks)));
    m.tile_visits_upper_bound = mul(m.factor_builds, f.tile_visits);
    // 64 units per complex accumulation and 256 per consumed-factor digest;
    // maximum block plans conservatively cover partial final blocks.
    m.work_units_upper_bound = add(f.caller_gauge_bytes / 16, add(mul(m.factor_builds, add(f.work_units, 256)),
        add(mul(64, mul(mul(shape.n_kpoints, shape.n_auxiliary), mul(m.n_virtual, m.n_virtual))),
            mul(8, mul(m.n_virtual, m.n_virtual)))));
    const U replicas = mul(r.budget().mpi_ranks, r.budget().workers_per_rank);
    m.required_node_memory_bytes = add(f.required_node_memory_bytes, mul(replicas, extra));
    if (m.retained_output_bytes / 16 > std::vector<C>().max_size()
        || m.peak_owned_numerical_bytes > static_cast<U>(std::numeric_limits<std::ptrdiff_t>::max())
        || m.retained_output_bytes > std::numeric_limits<U>::max() / 8 - 8192
        || mul(80, m.factor_builds) > std::numeric_limits<U>::max() / 8 - 8192)
        throw std::length_error("pair integral numerical or SHA extent is too large");
    return m;
}

const C* PeriodicCorrelationPairIntegralBlock::data() const {
    if (!state_ || values_.size() != memory_.retained_output_bytes / 16)
        throw std::logic_error("pair integral block is consumed");
    return values_.data();
}

PeriodicCorrelationPairIntegralBlock build_periodic_correlation_pair_integral_block(
    const PeriodicCorrelationAdmittedReference& r, const PeriodicCorrelationFactorStreamSchedule& schedule,
    const PeriodicCorrelationPrivateFactorReader& reader, const PeriodicCorrelationWannier& w,
    const C* gauges, std::size_t gauge_count, const PeriodicCorrelationPAODomain& domain,
    const PeriodicCorrelationPAOSpace& space, const Selection& s,
    const PeriodicCorrelationPairIntegralCaps& controls) {
    const auto m = plan_periodic_correlation_pair_integral_block(r, schedule, reader, w, domain, space, s);
    caps(r, m, controls);
    (void) periodic_correlation_local_detail::validate_gauges(w, gauges, gauge_count);
    PeriodicCorrelationPairIntegralBlock out;
    out.memory_ = m; out.selection_ = s; out.store_ = reader.storage_identity_sha256();
    out.values_.resize(static_cast<std::size_t>(m.retained_output_bytes / 16));
    const auto& shape = schedule.shape();
    PeriodicCorrelationLocalFactorCaps inner;
    inner.maximum_owned_numerical_bytes = m.maximum_factor_owned_bytes;
    inner.maximum_work_units = m.maximum_factor.work_units;
    inner.maximum_tile_visits = m.maximum_factor.tile_visits;
    inner.maximum_reader_tile_bytes = m.maximum_factor.maximum_reader_tile_bytes;
    Digest consumed("vibeqc.periodic.correlation.pair-integrals.consumed-factors");
    consumed.string(out.store_); consumed.u64(m.factor_builds);
    const auto record = [&](const PeriodicCorrelationLocalFactorBlock& block) {
        if (out.factor_builds_ >= m.factor_builds)
            throw std::logic_error("pair integral factor-build census was exceeded");
        out.tile_visits_ = add(out.tile_visits_, block.memory().tile_visits);
        if (out.tile_visits_ > m.tile_visits_upper_bound)
            throw std::logic_error("pair integral tile-visit census was exceeded");
        consumed.u64(out.factor_builds_++); consumed.string(block.identity_sha256());
    };
    {
        std::vector<C> correction(out.values_.size());
        for (U q = 0; q < shape.n_kpoints; ++q) {
            for (U atile = 0; atile < shape.auxiliary_tile_count; ++atile) {
                const U begin = atile * shape.auxiliary_block;
                const U ab = std::min(shape.auxiliary_block, shape.n_auxiliary - begin);
                for (U a0 = 0; a0 < s.virtual_count;) {
                    const U va = std::min(s.virtual_block, s.virtual_count - a0);
                    const auto left = build_periodic_correlation_local_factor_block(r, schedule, reader, w,
                        gauges, gauge_count, domain, space, local(s, false, s.virtual_begin + a0, va),
                        q, begin, ab, PeriodicCorrelationLocalFactorOrientation::VirtualOccupied, inner);
                    record(left);
                    for (U b0 = 0; b0 < s.virtual_count;) {
                        const U vb = std::min(s.virtual_block, s.virtual_count - b0);
                        const auto right = build_periodic_correlation_local_factor_block(r, schedule, reader, w,
                            gauges, gauge_count, domain, space, local(s, true, s.virtual_begin + b0, vb),
                            q, begin, ab, PeriodicCorrelationLocalFactorOrientation::OccupiedVirtual, inner);
                        record(right);
                        if (left.source_identity_sha256() != right.source_identity_sha256()
                            || left.whitener_payload_identity_sha256() != right.whitener_payload_identity_sha256())
                            throw std::logic_error("pair integral factors have inconsistent q source or whitener");
                        const C* l = left.data(); const C* rr = right.data();
                        for (U p = 0; p < ab; ++p) for (U a = 0; a < va; ++a) for (U b = 0; b < vb; ++b) {
                            const U index = (a0 + a) * s.virtual_count + b0 + b;
                            accumulate(std::conj(l[p * va + a]) * rr[p * vb + b],
                                       out.values_[index], correction[index]);
                        }
                        b0 += vb;
                    }
                    a0 += va;
                }
            }
        }
        for (std::size_t i = 0; i < out.values_.size(); ++i) {
            out.values_[i] += correction[i]; finite(out.values_[i]);
        }
    }
    if (out.factor_builds_ != m.factor_builds)
        throw std::logic_error("pair integral factor-build census is incomplete");
    out.consumed_ = consumed.finish();
    Digest payload("vibeqc.periodic.correlation.pair-integrals.payload");
    payload.u64(s.virtual_count);
    for (const auto value : out.values_) payload.complex(value);
    out.payload_ = payload.finish();
    Digest identity("vibeqc.periodic.correlation.pair-integrals.identity");
    identity.string(r.state().state_identity_sha256()); identity.string(r.state().calculation_identity());
    identity.string(r.dimensions().allocation_identity); identity.string(out.store_);
    identity.string(w.wannier_identity_sha256()); identity.string(w.gauge_payload_sha256());
    identity.string(domain.pao_domain_identity_sha256()); identity.string(space.pao_space_identity_sha256());
    selection_digest(identity, s); identity.string(out.consumed_); identity.string(out.payload_);
    identity.string("all-q;conjugate-VO-times-OV;global-auxiliary-RI;finite-image;complex");
    out.identity_ = identity.finish(); out.state_ = r.state_handle();
    return out;
}
}  // namespace vibeqc
