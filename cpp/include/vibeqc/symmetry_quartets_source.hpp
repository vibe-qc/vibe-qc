#pragma once

#include "vibeqc/symmetry_quartets.hpp"
#include "vibeqc/manta.hpp"
#include "vibeqc/manta_molecular_source.hpp"
#include "vibeqc/detail/sha256.hpp"
#include <memory>

namespace vibeqc {
// Execution bridge to an already audited immutable molecular source. This
// does not supply group, screening, arithmetic or equivariance evidence.
SymmetryPlan plan_symmetry_quartet_source(const manta::OrderedJKBuilder&,
    const SymmetryBudget&);

class SymmetryQuartetSourceAccumulator {
public:
    SymmetryQuartetSourceAccumulator(SymmetryQuartetAccumulator&,
        std::shared_ptr<manta::OrderedJKBuilder>, const std::string& source_identity,
        const std::string& representative_digest, std::uint64_t representatives,
        const SymmetryBudget&);
    ~SymmetryQuartetSourceAccumulator();
    void consume_orbit(const std::array<std::int64_t,4>& representative,
        std::uint64_t operations, const std::int64_t* mapped_shells,
        std::uint64_t identity, std::uint8_t mode);
    void bind_support(std::shared_ptr<const manta::MolecularSourceSupport>,
        std::uint64_t operations,std::uint64_t shells,const std::int64_t* destinations,
        std::uint64_t identity,std::uint8_t mode);
    void consume_key(const std::array<std::int64_t,4>& representative);
    SymmetryQuartetTargets finish();
    void invalidate() noexcept;
    const std::array<std::uint64_t,5>& work() const { return work_; }
private:
    void consume(const std::array<std::int64_t,4>& representative,
        std::uint64_t operations,const std::int64_t* mapped,
        std::uint64_t identity,std::uint8_t mode);
    SymmetryQuartetAccumulator& accumulator_;
    std::shared_ptr<manta::OrderedJKBuilder> builder_;
    std::unique_ptr<manta::Workspace> workspace_;
    std::vector<double> zeros_;
    std::shared_ptr<const manta::MolecularSourceSupport> support_;
    std::vector<std::int64_t> mapped_;
    std::uint64_t identity_=0;
    std::uint8_t mode_=0;
    std::string source_, expected_digest_;
    detail::Sha256 digest_;
    std::uint64_t expected_representatives_;
    std::array<std::uint64_t,5> work_{};
    bool valid_=true, finished_=false;
};
} // namespace vibeqc
