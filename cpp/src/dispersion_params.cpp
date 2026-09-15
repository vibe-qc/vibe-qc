// Per-functional D3(BJ) parameter registry.
//
// The numbers (s6, s8, a1, a2) are specific to each density functional.
// The generated table is the complete D3(BJ) registry from the pinned
// simple-dftd3 parameters.toml; see scripts/extract_d3bj_parameters.py.

#include "vibeqc/dispersion.hpp"

#include <algorithm>
#include <cctype>
#include <string>
#include <unordered_map>

namespace vibeqc {

namespace {

std::string canonical_name(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        if (c != '-') out.push_back(static_cast<char>(std::tolower(c)));
    }
    // BP86 is conventionally labelled "bp" by simple-dftd3.
    if (out == "bp86") return "bp";
    return out;
}

const std::unordered_map<std::string, D3BJParams>& params_table() {
    static const std::unordered_map<std::string, D3BJParams> t = {
#include "dispersion_params_data.inc"
    };
    return t;
}

}  // namespace

std::optional<D3BJParams> d3bj_params_for(const std::string& functional) {
    const auto key = canonical_name(functional);
    const auto& t = params_table();
    auto it = t.find(key);
    if (it == t.end()) return std::nullopt;
    return it->second;
}

}  // namespace vibeqc
