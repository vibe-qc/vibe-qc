#include "vibeqc/semiempirical/core/method_registry.hpp"

#include <algorithm>
#include <stdexcept>

namespace vibeqc {
namespace semiempirical {

SemiempiricalMethodRegistry& SemiempiricalMethodRegistry::instance() {
    static SemiempiricalMethodRegistry reg;
    return reg;
}

void SemiempiricalMethodRegistry::register_method(
    const SemiempiricalMethodPlugin& plugin) {
    // Don't allow duplicate registrations
    for (const auto& existing : methods_) {
        if (existing.config.name == plugin.config.name) {
            throw std::runtime_error(
                "SemiempiricalMethodRegistry: method '" +
                plugin.config.name + "' already registered");
        }
    }
    methods_.push_back(plugin);
}

const SemiempiricalMethodPlugin* SemiempiricalMethodRegistry::find(
    const std::string& name) const {
    for (const auto& m : methods_) {
        if (m.config.name == name) return &m;
    }
    return nullptr;
}

std::vector<std::string> SemiempiricalMethodRegistry::method_names() const {
    std::vector<std::string> names;
    names.reserve(methods_.size());
    for (const auto& m : methods_) names.push_back(m.config.name);
    return names;
}

std::vector<SemiempiricalMethodConfig>
SemiempiricalMethodRegistry::family_methods(MethodFamily family) const {
    std::vector<SemiempiricalMethodConfig> result;
    for (const auto& m : methods_) {
        if (m.config.family == family) result.push_back(m.config);
    }
    return result;
}

}  // namespace semiempirical
}  // namespace vibeqc
