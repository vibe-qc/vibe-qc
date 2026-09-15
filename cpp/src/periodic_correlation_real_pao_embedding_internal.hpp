#pragma once

// One shared physical pair-frame codec. This is a provenance comparison,
// not validation of numerical payloads or a new geometry certificate.
#include "vibeqc/periodic_correlation_real_pao_embedding.hpp"
#include "periodic_correlation_local_factors_internal.hpp"

namespace vibeqc {
namespace periodic_correlation_real_pao_embedding_detail {
inline std::string pair_frame_identity(const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationPAODomain& domain,const PeriodicCorrelationRealPAOSpace& space,
    const PeriodicCorrelationVirtualBlockSelection& selection) {
    periodic_correlation_local_detail::Digest h("vibeqc.periodic.correlation.real-pao-embedding.frame");
    h.string(ref.state().state_identity_sha256());h.string(ref.dimensions().allocation_identity);
    h.string(domain.pao_domain_identity_sha256());h.string(space.space().pao_space_identity_sha256());
    h.string(space.identity_sha256());
    h.u64(selection.begin);h.u64(selection.count);h.u64(selection.translation_cell);h.u64(0);
    return h.finish();
}
} // namespace periodic_correlation_real_pao_embedding_detail
} // namespace vibeqc
