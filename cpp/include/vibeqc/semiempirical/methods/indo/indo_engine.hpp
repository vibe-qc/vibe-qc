// MSINDO SCF engine — faithful C++ port, parameter-driven (full H–Xe table).
//
// Mirrors the validated reference engine in
// python/vibeqc/semiempirical/methods/msindo.py (oracle parity to µHa).
// Header-only: includes the STO kernel and Eigen so it can be exercised both
// from a standalone test and the pybind layer without a separate translation
// unit.  Parameters load from the same JSON bundles the Python engine uses
// (msindo_params.json / msindo_params_nddo.json) via
// ``load_msindo_params_from_json``; a built-in H–F embedded set is provided
// for backward compatibility.
//
// Local orbital order: 0=s, 1..3=px,py,pz, 4..8=d (HARMTR column convention).
//
// Scope (as of this commit):
//   closed-shell RHF  — INDO s/p/d (full Z=1–54)
//   open-shell UHF    — INDO light s/p and named heavy d/p-block fixtures
//   NDDO              — closed-shell RHF (H, Li–F, Na–Cl; Al–Cl SPDD included)
//   extended cores    — 1s/1s2s2p/3s3p3d plus [Kr]/[Kr]4d10 frozen cores
//
// © Mulliken Center for Theoretical Chemistry, University of Bonn (method);
// independent vibe-qc re-implementation.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "vibeqc/semiempirical/methods/indo/msindo_integrals.hpp"
#include "vibeqc/solvation_cpcm.hpp"

namespace vibeqc {
namespace semiempirical {
namespace indo {

// --------------------------------------------------------------------------- //
// MsindoParameterSet — the full per-element parameter table (Z=1..54).         //
// --------------------------------------------------------------------------- //
// Each field is a map Z -> value; zero for elements not parametrized / absent.
// AL is the anti-penetration matrix: per central Z, an 11-tuple over partner
// shell-groups (datas.f SHD groups).  Loaded from JSON via
// ``load_msindo_params_from_json`` at module-init time.
//
// The NDDO bundle (msindo_params_nddo.json) overrides a subset of these fields
// for H, Li–F, Na–Cl — mirroring MSINDO's IF(NDDO) CALL NDDOPARAM.  Load it
// via ``load_msindo_nddo_params_from_json`` and merge into the base set.

struct MsindoParameterSet {
    // Valence two-centre Slater exponents (bohr^-1).
    std::map<int, double> MUS, MUP, MUD;
    // Valence one-centre Slater exponents (bohr^-1).
    std::map<int, double> MUSE, MUPE, MUDE;
    // Resonance K betas.
    std::map<int, double> KSS, KPS, KPP, KDS, KDP, KDD;
    // Ionization potentials (Hartree).
    std::map<int, double> IPOTS, IPOTP, IPOTD;
    // Frozen-core Slater exponents (incl. [Kr]/[Kr]4d¹⁰ 4s/4p/4d for Rb–Xe).
    std::map<int, double> TAU1S, TAU2S, TAU2P, TAU3S, TAU3P, TAU3D, TAU4S, TAU4P, TAU4D;
    // Frozen-core potentials.
    std::map<int, double> FCP1S, FCP2S, FCP2P, FCP3S, FCP3P, FCP3D, FCP4S, FCP4P, FCP4D;
    // Shielding corrections.
    std::map<int, double> SCP3D, SCP4S, SCP4P;
    // Valence occupations.
    std::map<int, double> LS, MP, ND;
    // AL anti-penetration matrix: per central Z, an 11-tuple over partner groups.
    // The 11 partner shell-groups (datas.f SHD):
    //   0: 1-2, 1: 3-5, 2: 6-10, 3: 11-12, 4: 13-18,
    //   5: 19-20, 6: 21-30, 7: 31-36, 8: 37-38, 9: 39-48, 10: 49-54
    std::map<int, std::array<double, 11>> AL;

    // Helper: return a scalar, 0.0 if missing.
    static double get(const std::map<int, double>& m, int z) {
        auto it = m.find(z);
        return it == m.end() ? 0.0 : it->second;
    }
    static int iget(const std::map<int, double>& m, int z) {
        auto it = m.find(z);
        return it == m.end() ? 0 : static_cast<int>(it->second);
    }

    // ---- Principal quantum numbers (independent of parameter set) ----
    static int n_principal(int z) {
        if (z <= 2) return 1;
        if (z <= 10) return 2;
        if (z <= 18) return 3;
        if (z <= 36) return 4;
        return 5;
    }
    static int n_p_principal(int z) { return z == 1 ? 2 : n_principal(z); }
    static int n_d_principal(int z) {
        if (z <= 30) return 3;   // Al–Ar (3d), Sc–Zn (3d)
        if (z <= 48) return 4;   // Ga–Kr (4d), Y–Cd (4d)
        return 5;                 // In–Xe (5d)
    }
    static int eff_core_charge(int z) {
        if (z <= 2) return z;
        if (z <= 10) return z - 2;
        if (z <= 18) return z - 10;
        if (z <= 30) return z - 18;
        if (z <= 36) return z - 28;
        if (z <= 48) return z - 36;
        return z - 46;
    }
    int n_basis(int z) const {
        if (z <= 2) return 1;
        if (z <= 12) return 4;
        return (get(MUD, z) != 0.0) ? 9 : 4;
    }

    // AL lookup: partner shell-group index for a given Z.
    static int al_group(int z) {
        if (z <= 2) return 0;
        if (z <= 5) return 1;
        if (z <= 10) return 2;
        if (z <= 12) return 3;
        if (z <= 18) return 4;
        if (z <= 20) return 5;
        if (z <= 30) return 6;
        if (z <= 36) return 7;
        if (z <= 38) return 8;
        if (z <= 48) return 9;
        return 10;
    }
    double AL_val(int zk, int zl) const {
        int grp = al_group(zl);
        auto it = AL.find(zk);
        if (it == AL.end()) return 0.0;
        return it->second[grp];
    }

    // ---- Built-in embedded H–F set (backward compatibility) ----
    static MsindoParameterSet embedded_hf() {
        MsindoParameterSet p;
        // MUS
        p.MUS = {{1,1.1576},{2,1.7000},{6,1.7874},{7,2.0423},{8,2.3538},{9,2.4974}};
        p.MUP = {{1,0.1270},{2,0.0000},{6,1.6770},{7,1.8161},{8,2.1559},{9,2.3510}};
        p.MUD = {};
        p.MUSE = {{1,1.0060},{2,1.7000},{6,1.6266},{7,1.8098},{8,2.1109},{9,2.3408}};
        p.MUPE = {{1,0.1270},{2,0.0000},{6,1.5572},{7,1.7326},{8,1.9055},{9,2.2465}};
        p.MUDE = {};
        p.KSS = {{1,0.1449},{2,0.0000},{6,0.0867},{7,0.1031},{8,0.1242},{9,0.1769}};
        p.KPS = {{1,0.0000},{2,0.0000},{6,0.0867},{7,0.1031},{8,0.1242},{9,0.1769}};
        p.KPP = {{1,0.0000},{2,0.0000},{6,0.0478},{7,0.0524},{8,0.0760},{9,0.0127}};
        p.KDS = {}; p.KDP = {}; p.KDD = {};
        p.IPOTS = {{1,-0.5000},{2,-0.0905},{6,-0.8195},{7,-1.0346},{8,-1.6838},{9,-2.0238}};
        p.IPOTP = {{1,-0.1047},{2,0.0000},{6,-0.3824},{7,-0.4602},{8,-0.5780},{9,-0.6868}};
        p.IPOTD = {};
        p.TAU1S = {{6,5.0830},{7,6.8167},{8,7.3271},{9,8.6043}};
        p.TAU2S = {}; p.TAU2P = {}; p.TAU3S = {}; p.TAU3P = {}; p.TAU3D = {};
        p.FCP1S = {{6,10.430},{7,14.760},{8,19.550},{9,25.190}};
        p.FCP2S = {}; p.FCP2P = {}; p.FCP3S = {}; p.FCP3P = {}; p.FCP3D = {};
        p.SCP3D = {}; p.SCP4S = {}; p.SCP4P = {};
        p.LS = {{1,1},{2,2},{6,2},{7,2},{8,2},{9,2}};
        p.MP = {{1,0},{2,0},{6,2},{7,3},{8,4},{9,5}};
        p.ND = {};
        // AL: H col-0 (partner 1-2) and col-2 (partner 6-10)
        p.AL[1] = {{0.3856,0.4575,0.5038,0.8272,0.5488,0.524,0.5217,0.7579,0,0,0}};
        p.AL[2] = {{0,0,0,0,0,0,0,0,0,0,0}};
        p.AL[6] = {{0.4936,0.5104,0.6776,0.6605,0.818,0,0.8644,0.9662,0,0,0}};
        p.AL[7] = {{0.2960,0.3100,0.3268,0.4219,0.4429,0,0.4680,0.5233,0,0,0}};
        p.AL[8] = {{0.2485,0.2647,0.2246,0.3178,0.3459,0,0.3845,0.4563,0,0,0}};
        p.AL[9] = {{0.1521,0.1583,0.1059,0.1679,0.1807,0,0.1925,0.2134,0,0,0}};
        return p;
    }
};

// --------------------------------------------------------------------------- //
// JSON parameter loading (thin parsers — no external JSON library dependency). //
// --------------------------------------------------------------------------- //
// These parse the vibe-qc bundled JSON bundles (msindo_params.json /
// msindo_params_nddo.json) into a MsindoParameterSet.  The format is simple
// enough for a hand-rolled parser: one JSON object per Z, flat scalar fields
// plus an "AL" list-of-floats per element.

namespace detail {
inline std::string json_get_string(const std::string& json, size_t& pos) {
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) ++pos;
    if (pos >= json.size() || json[pos] != '"') return "";
    ++pos;
    std::string out;
    while (pos < json.size() && json[pos] != '"') {
        if (json[pos] == '\\' && pos + 1 < json.size()) { ++pos; }
        out += json[pos++];
    }
    if (pos < json.size()) ++pos; // skip closing "
    return out;
}
inline double json_get_number(const std::string& json, size_t& pos) {
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) ++pos;
    size_t start = pos;
    if (pos < json.size() && (json[pos] == '-' || json[pos] == '+')) ++pos;
    while (pos < json.size() && std::isdigit(static_cast<unsigned char>(json[pos]))) ++pos;
    if (pos < json.size() && json[pos] == '.') { ++pos; while (pos < json.size() && std::isdigit(static_cast<unsigned char>(json[pos]))) ++pos; }
    if (pos < json.size() && (json[pos] == 'e' || json[pos] == 'E')) {
        ++pos; if (pos < json.size() && (json[pos] == '-' || json[pos] == '+')) ++pos;
        while (pos < json.size() && std::isdigit(static_cast<unsigned char>(json[pos]))) ++pos;
    }
    return std::stod(json.substr(start, pos - start));
}
inline void json_skip_value(const std::string& json, size_t& pos) {
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) ++pos;
    if (pos >= json.size()) return;
    char c = json[pos];
    if (c == '"') { json_get_string(json, pos); }
    else if (c == '{' || c == '[') {
        char close = (c == '{') ? '}' : ']';
        int depth = 1;
        ++pos;
        while (pos < json.size() && depth > 0) {
            if (json[pos] == c) ++depth;
            else if (json[pos] == close) --depth;
            else if (json[pos] == '"') { json_get_string(json, pos); continue; }
            ++pos;
        }
    } else { json_get_number(json, pos); }
}
}  // namespace detail

inline MsindoParameterSet load_msindo_params_from_json(const std::string& json_str) {
    MsindoParameterSet p;
    size_t pos = 0;
    // Navigate to "elements" object
    auto find_key = [&](const std::string& key) {
        while (pos < json_str.size()) {
            while (pos < json_str.size() && std::isspace(static_cast<unsigned char>(json_str[pos]))) ++pos;
            if (pos >= json_str.size()) break;
            if (json_str[pos] == '"') {
                std::string k = detail::json_get_string(json_str, pos);
                while (pos < json_str.size() && std::isspace(static_cast<unsigned char>(json_str[pos]))) ++pos;
                if (pos < json_str.size() && json_str[pos] == ':') ++pos;
                if (k == key) return true;
                detail::json_skip_value(json_str, pos);
                while (pos < json_str.size() && std::isspace(static_cast<unsigned char>(json_str[pos]))) ++pos;
                if (pos < json_str.size() && json_str[pos] == ',') ++pos;
            } else { ++pos; }
        }
        return false;
    };
    if (!find_key("elements")) return p;
    while (pos < json_str.size() && std::isspace(static_cast<unsigned char>(json_str[pos]))) ++pos;
    if (pos < json_str.size() && json_str[pos] == ':') ++pos;
    while (pos < json_str.size() && std::isspace(static_cast<unsigned char>(json_str[pos]))) ++pos;
    if (pos >= json_str.size() || json_str[pos] != '{') return p;
    ++pos; // skip {

    // Iterate over Z keys in "elements"
    int depth = 1;
    while (pos < json_str.size() && depth > 0) {
        while (pos < json_str.size() && std::isspace(static_cast<unsigned char>(json_str[pos]))) ++pos;
        if (pos >= json_str.size()) break;
        if (json_str[pos] == '}') { --depth; ++pos; continue; }
        if (json_str[pos] == ',') { ++pos; continue; }
        if (json_str[pos] != '"') { ++pos; continue; }
        std::string zkey = detail::json_get_string(json_str, pos);
        while (pos < json_str.size() && std::isspace(static_cast<unsigned char>(json_str[pos]))) ++pos;
        if (pos < json_str.size() && json_str[pos] == ':') ++pos;
        while (pos < json_str.size() && std::isspace(static_cast<unsigned char>(json_str[pos]))) ++pos;
        if (pos >= json_str.size() || json_str[pos] != '{') { detail::json_skip_value(json_str, pos); continue; }
        ++pos; // skip {
        int z = std::stoi(zkey);

        // Read fields inside the element object
        int elem_depth = 1;
        std::vector<double> al_vals;
        while (pos < json_str.size() && elem_depth > 0) {
            while (pos < json_str.size() && std::isspace(static_cast<unsigned char>(json_str[pos]))) ++pos;
            if (pos >= json_str.size()) break;
            if (json_str[pos] == '}') { --elem_depth; ++pos; continue; }
            if (json_str[pos] == ',') { ++pos; continue; }
            std::string field = detail::json_get_string(json_str, pos);
            while (pos < json_str.size() && std::isspace(static_cast<unsigned char>(json_str[pos]))) ++pos;
            if (pos < json_str.size() && json_str[pos] == ':') ++pos;

            if (field == "AL") {
                // parse array of numbers
                while (pos < json_str.size() && std::isspace(static_cast<unsigned char>(json_str[pos]))) ++pos;
                if (pos < json_str.size() && json_str[pos] == '[') {
                    ++pos;
                    al_vals.clear();
                    while (pos < json_str.size()) {
                        while (pos < json_str.size() && std::isspace(static_cast<unsigned char>(json_str[pos]))) ++pos;
                        if (pos >= json_str.size() || json_str[pos] == ']') { ++pos; break; }
                        al_vals.push_back(detail::json_get_number(json_str, pos));
                        while (pos < json_str.size() && std::isspace(static_cast<unsigned char>(json_str[pos]))) ++pos;
                        if (pos < json_str.size() && json_str[pos] == ',') ++pos;
                    }
                }
            } else {
                double val = detail::json_get_number(json_str, pos);
                if (field == "MUS") p.MUS[z] = val;
                else if (field == "MUP") p.MUP[z] = val;
                else if (field == "MUD") p.MUD[z] = val;
                else if (field == "MUSE") p.MUSE[z] = val;
                else if (field == "MUPE") p.MUPE[z] = val;
                else if (field == "MUDE") p.MUDE[z] = val;
                else if (field == "KSS") p.KSS[z] = val;
                else if (field == "KPS") p.KPS[z] = val;
                else if (field == "KPP") p.KPP[z] = val;
                else if (field == "KDS") p.KDS[z] = val;
                else if (field == "KDP") p.KDP[z] = val;
                else if (field == "KDD") p.KDD[z] = val;
                else if (field == "IPOTS") p.IPOTS[z] = val;
                else if (field == "IPOTP") p.IPOTP[z] = val;
                else if (field == "IPOTD") p.IPOTD[z] = val;
                else if (field == "TAU1S") p.TAU1S[z] = val;
                else if (field == "TAU2S") p.TAU2S[z] = val;
                else if (field == "TAU2P") p.TAU2P[z] = val;
                else if (field == "TAU3S") p.TAU3S[z] = val;
                else if (field == "TAU3P") p.TAU3P[z] = val;
                else if (field == "TAU3D") p.TAU3D[z] = val;
                else if (field == "TAU4S") p.TAU4S[z] = val;
                else if (field == "TAU4P") p.TAU4P[z] = val;
                else if (field == "TAU4D") p.TAU4D[z] = val;
                else if (field == "FCP1S") p.FCP1S[z] = val;
                else if (field == "FCP2S") p.FCP2S[z] = val;
                else if (field == "FCP2P") p.FCP2P[z] = val;
                else if (field == "FCP3S") p.FCP3S[z] = val;
                else if (field == "FCP3P") p.FCP3P[z] = val;
                else if (field == "FCP3D") p.FCP3D[z] = val;
                else if (field == "FCP4S") p.FCP4S[z] = val;
                else if (field == "FCP4P") p.FCP4P[z] = val;
                else if (field == "FCP4D") p.FCP4D[z] = val;
                else if (field == "SCP3D") p.SCP3D[z] = val;
                else if (field == "SCP4S") p.SCP4S[z] = val;
                else if (field == "SCP4P") p.SCP4P[z] = val;
                else if (field == "LS") p.LS[z] = val;
                else if (field == "MP") p.MP[z] = val;
                else if (field == "ND") p.ND[z] = val;
            }
        }
        // Store AL
        if (!al_vals.empty()) {
            std::array<double, 11> al_arr{};
            for (size_t i = 0; i < al_vals.size() && i < 11; ++i) al_arr[i] = al_vals[i];
            p.AL[z] = al_arr;
        }
    }
    return p;
}

// NDDO overrides: merge NDDO-specific scalars + AL onto an existing base set.
inline void merge_msindo_nddo_overrides(MsindoParameterSet& base, const std::string& json_str) {
    size_t pos = 0;
    auto find_key = [&](const std::string& key) {
        while (pos < json_str.size()) {
            while (pos < json_str.size() && std::isspace(static_cast<unsigned char>(json_str[pos]))) ++pos;
            if (pos >= json_str.size()) break;
            if (json_str[pos] == '"') {
                std::string k = detail::json_get_string(json_str, pos);
                while (pos < json_str.size() && std::isspace(static_cast<unsigned char>(json_str[pos]))) ++pos;
                if (pos < json_str.size() && json_str[pos] == ':') ++pos;
                if (k == key) return true;
                detail::json_skip_value(json_str, pos);
                while (pos < json_str.size() && std::isspace(static_cast<unsigned char>(json_str[pos]))) ++pos;
                if (pos < json_str.size() && json_str[pos] == ',') ++pos;
            } else { ++pos; }
        }
        return false;
    };
    if (!find_key("elements")) return;
    while (pos < json_str.size() && std::isspace(static_cast<unsigned char>(json_str[pos]))) ++pos;
    if (pos < json_str.size() && json_str[pos] == ':') ++pos;
    while (pos < json_str.size() && std::isspace(static_cast<unsigned char>(json_str[pos]))) ++pos;
    if (pos >= json_str.size() || json_str[pos] != '{') return;
    ++pos;

    int depth = 1;
    while (pos < json_str.size() && depth > 0) {
        while (pos < json_str.size() && std::isspace(static_cast<unsigned char>(json_str[pos]))) ++pos;
        if (pos >= json_str.size()) break;
        if (json_str[pos] == '}') { --depth; ++pos; continue; }
        if (json_str[pos] == ',') { ++pos; continue; }
        if (json_str[pos] != '"') { ++pos; continue; }
        std::string zkey = detail::json_get_string(json_str, pos);
        while (pos < json_str.size() && std::isspace(static_cast<unsigned char>(json_str[pos]))) ++pos;
        if (pos < json_str.size() && json_str[pos] == ':') ++pos;
        while (pos < json_str.size() && std::isspace(static_cast<unsigned char>(json_str[pos]))) ++pos;
        if (pos >= json_str.size() || json_str[pos] != '{') { detail::json_skip_value(json_str, pos); continue; }
        ++pos;
        int z = std::stoi(zkey);
        int elem_depth = 1;
        std::vector<double> al_vals;
        while (pos < json_str.size() && elem_depth > 0) {
            while (pos < json_str.size() && std::isspace(static_cast<unsigned char>(json_str[pos]))) ++pos;
            if (pos >= json_str.size()) break;
            if (json_str[pos] == '}') { --elem_depth; ++pos; continue; }
            if (json_str[pos] == ',') { ++pos; continue; }
            std::string field = detail::json_get_string(json_str, pos);
            while (pos < json_str.size() && std::isspace(static_cast<unsigned char>(json_str[pos]))) ++pos;
            if (pos < json_str.size() && json_str[pos] == ':') ++pos;
            if (field == "AL") {
                while (pos < json_str.size() && std::isspace(static_cast<unsigned char>(json_str[pos]))) ++pos;
                if (pos < json_str.size() && json_str[pos] == '[') {
                    ++pos; al_vals.clear();
                    while (pos < json_str.size()) {
                        while (pos < json_str.size() && std::isspace(static_cast<unsigned char>(json_str[pos]))) ++pos;
                        if (pos >= json_str.size() || json_str[pos] == ']') { ++pos; break; }
                        al_vals.push_back(detail::json_get_number(json_str, pos));
                        while (pos < json_str.size() && std::isspace(static_cast<unsigned char>(json_str[pos]))) ++pos;
                        if (pos < json_str.size() && json_str[pos] == ',') ++pos;
                    }
                }
            } else {
                double val = detail::json_get_number(json_str, pos);
                if (field == "MUS") base.MUS[z] = val;
                else if (field == "MUP") base.MUP[z] = val;
                else if (field == "MUD") base.MUD[z] = val;
                else if (field == "MUSE") base.MUSE[z] = val;
                else if (field == "MUPE") base.MUPE[z] = val;
                else if (field == "MUDE") base.MUDE[z] = val;
                else if (field == "KSS") base.KSS[z] = val;
                else if (field == "KPS") base.KPS[z] = val;
                else if (field == "KPP") base.KPP[z] = val;
                else if (field == "KDS") base.KDS[z] = val;
                else if (field == "KDP") base.KDP[z] = val;
                else if (field == "KDD") base.KDD[z] = val;
                else if (field == "IPOTS") base.IPOTS[z] = val;
                else if (field == "IPOTP") base.IPOTP[z] = val;
                else if (field == "IPOTD") base.IPOTD[z] = val;
                else if (field == "TAU1S") base.TAU1S[z] = val;
                else if (field == "TAU2S") base.TAU2S[z] = val;
                else if (field == "TAU2P") base.TAU2P[z] = val;
                else if (field == "SCP3D") base.SCP3D[z] = val;
            }
        }
        if (!al_vals.empty()) {
            std::array<double, 11> al_arr{};
            for (size_t i = 0; i < al_vals.size() && i < 11; ++i) al_arr[i] = al_vals[i];
            base.AL[z] = al_arr;
        }
    }
}

// --------------------------------------------------------------------------- //
// Backward-compatible parameter accessors (old API, embedded H–F scope).       //
// These delegate to the embedded_hf() set.  New code should use the            //
// parameterised forms below.                                                   //
// --------------------------------------------------------------------------- //
namespace legacy {
inline double MUS(int z)  { return MsindoParameterSet::embedded_hf().MUS[z]; }
inline double MUP(int z)  { return MsindoParameterSet::embedded_hf().MUP[z]; }
inline double MUSE(int z) { return MsindoParameterSet::embedded_hf().MUSE[z]; }
inline double MUPE(int z) { return MsindoParameterSet::embedded_hf().MUPE[z]; }
inline double KSS(int z)  { return MsindoParameterSet::embedded_hf().KSS[z]; }
inline double KPS(int z)  { return MsindoParameterSet::embedded_hf().KPS[z]; }
inline double KPP(int z)  { return MsindoParameterSet::embedded_hf().KPP[z]; }
inline double IPOTS(int z){ return MsindoParameterSet::embedded_hf().IPOTS[z]; }
inline double IPOTP(int z){ return MsindoParameterSet::embedded_hf().IPOTP[z]; }
inline double TAU1S(int z){ return MsindoParameterSet::embedded_hf().TAU1S[z]; }
inline double FCP1S(int z){ return MsindoParameterSet::embedded_hf().FCP1S[z]; }
inline int LS(int z)      { return MsindoParameterSet::embedded_hf().iget(MsindoParameterSet::embedded_hf().LS, z); }
inline int MP(int z)      { return MsindoParameterSet::embedded_hf().iget(MsindoParameterSet::embedded_hf().MP, z); }
inline int ND(int z)      { return MsindoParameterSet::embedded_hf().iget(MsindoParameterSet::embedded_hf().ND, z); }
inline double AL(int zk, int zl) { return MsindoParameterSet::embedded_hf().AL_val(zk, zl); }
}  // namespace legacy

// --------------------------------------------------------------------------- //
// One-center integrals (calsla.f / atomic_reference.f / gij1.f).              //
// --------------------------------------------------------------------------- //
struct SlaterCondon {
    double F0SS=0, F0SP=0, F0PP=0, G1SP=0, F2PP=0;
    double F0SD=0, F0PD=0, F0DD=0, G1PD=0, F2PD=0, F2DD=0;
    double G2SD=0, G3PD=0, F4DD=0;
    double I1SPPD=0, I2SDPP=0, I2SDDD=0;
};

// Parameterised version.
inline SlaterCondon slater_condon(int z, const MsindoParameterSet& p) {
    double zs = MsindoParameterSet::get(p.MUSE, z);
    double zp = MsindoParameterSet::get(p.MUPE, z);
    double zd = MsindoParameterSet::get(p.MUDE, z);
    SlaterCondon f;
    if (z <= 2) {
        f.F0SS = 5.0 / 8.0 * zs;
        if (z == 1) {
            f.F0SP = radint(0, 2, 2, 2, 2, zs, zp, zs, zp);
            f.F0PP = 93.0 / 256.0 * zp;
            f.G1SP = radint(1, 2, 2, 2, 2, zs, zp, zp, zs);
            f.F2PP = radint(2, 2, 2, 2, 2, zp, zp, zp, zp);
        }
        return f;
    }
    int n = MsindoParameterSet::n_principal(z);
    f.F0SS = radint(0, n, n, n, n, zs, zs, zs, zs);
    f.F0SP = radint(0, n, n, n, n, zs, zp, zs, zp);
    f.F0PP = radint(0, n, n, n, n, zp, zp, zp, zp);
    f.G1SP = radint(1, n, n, n, n, zs, zp, zp, zs);
    f.F2PP = radint(2, n, n, n, n, zp, zp, zp, zp);
    if (zd != 0.0) {
        int nd = MsindoParameterSet::n_d_principal(z);
        f.F0SD = radint(0, n, nd, n, nd, zs, zd, zs, zd);
        f.F0PD = radint(0, n, nd, n, nd, zp, zd, zp, zd);
        f.F0DD = radint(0, nd, nd, nd, nd, zd, zd, zd, zd);
        f.G1PD = radint(1, n, nd, nd, n, zp, zd, zd, zp);
        f.F2PD = radint(2, n, nd, n, nd, zp, zd, zp, zd);
        f.F2DD = radint(2, nd, nd, nd, nd, zd, zd, zd, zd);
        f.G2SD = radint(2, n, nd, nd, n, zs, zd, zd, zs);
        f.G3PD = radint(3, n, nd, nd, n, zp, zd, zd, zp);
        f.F4DD = radint(4, nd, nd, nd, nd, zd, zd, zd, zd);
        f.I1SPPD = radint(1, n, n, n, nd, zs, zp, zp, zd);
        f.I2SDPP = radint(2, n, n, nd, n, zs, zp, zd, zp);
        f.I2SDDD = radint(2, n, nd, nd, nd, zs, zd, zd, zd);
    }
    return f;
}

// Backward-compatible overload.
inline SlaterCondon slater_condon(int z) {
    return slater_condon(z, MsindoParameterSet::embedded_hf());
}

struct OneCenter2e { double GSS, GSP, HSP, GPP, GP2, HPP; };

inline OneCenter2e one_center_2e(int z, const MsindoParameterSet& p) {
    SlaterCondon f = slater_condon(z, p);
    OneCenter2e o{f.F0SS, 0, 0, 0, 0, 0};
    if (p.n_basis(z) >= 4) {
        o.GSP = f.F0SP;
        o.HSP = f.G1SP / 3.0;
        o.GPP = f.F0PP + 4.0 / 25.0 * f.F2PP;
        o.GP2 = f.F0PP - 2.0 / 25.0 * f.F2PP;
        o.HPP = 3.0 / 25.0 * f.F2PP;
    }
    return o;
}

inline OneCenter2e one_center_2e(int z) {
    return one_center_2e(z, MsindoParameterSet::embedded_hf());
}

// Full 9×9 one-center GMUNU matrix (gij1.f).  Upper triangle = Coulomb,
// lower triangle = exchange.  Orbital order: 0=s, 1..3=px,py,pz, 4..8=d.
inline Eigen::Matrix<double, 9, 9> one_center_gmunu(int z, const MsindoParameterSet& p) {
    SlaterCondon f = slater_condon(z, p);
    Eigen::Matrix<double, 9, 9> G = Eigen::Matrix<double, 9, 9>::Zero();
    G(0, 0) = f.F0SS;
    int nb = p.n_basis(z);
    if (nb >= 4) {
        for (int pi = 1; pi <= 3; ++pi) {
            G(0, pi) = f.F0SP;
            G(pi, 0) = f.G1SP / 3.0;
            G(pi, pi) = f.F0PP + 4.0 / 25.0 * f.F2PP;
        }
        // p-p off-diagonal Coulomb / exchange pairs
        std::pair<int,int> pp_pairs[3] = {{1,2},{1,3},{2,3}};
        for (auto& ab : pp_pairs) {
            G(ab.first, ab.second) = f.F0PP - 2.0 / 25.0 * f.F2PP;
            G(ab.second, ab.first) = 3.0 / 25.0 * f.F2PP;
        }
    }
    if (nb >= 9) {
        double F0SD=f.F0SD, F0PD=f.F0PD, F0DD=f.F0DD;
        double G1PD=f.G1PD, F2PD=f.F2PD, F2DD=f.F2DD;
        double G2SD=f.G2SD, G3PD=f.G3PD, F4DD=f.F4DD;
        for (int d = 4; d < 9; ++d) {
            G(0, d) = F0SD;
            G(d, 0) = G2SD / 5.0;
            G(d, d) = F0DD + 4.0 / 49.0 * F2DD + 36.0 / 441.0 * F4DD;
        }
        // p-d Coulomb (upper)
        G(1, 4) = G(2, 4) = F0PD - 2.0 / 35.0 * F2PD;
        G(3, 4) = F0PD + 4.0 / 35.0 * F2PD;
        for (auto ab : std::initializer_list<std::pair<int,int>>{{1,5},{3,5},{2,6},{3,6},{1,7},{2,7},{1,8},{2,8}})
            G(ab.first, ab.second) = F0PD + 2.0 / 35.0 * F2PD;
        for (auto ab : std::initializer_list<std::pair<int,int>>{{2,5},{1,6},{3,7},{3,8}})
            G(ab.first, ab.second) = F0PD - 4.0 / 35.0 * F2PD;
        // p-d exchange (lower)
        G(4, 1) = G(4, 2) = G1PD / 15.0 + 18.0 / 245.0 * G3PD;
        G(4, 3) = 4.0 / 15.0 * G1PD + 27.0 / 245.0 * G3PD;
        for (auto ab : std::initializer_list<std::pair<int,int>>{{5,1},{5,3},{6,2},{6,3},{7,1},{7,2},{8,1},{8,2}})
            G(ab.first, ab.second) = 3.0 / 15.0 * G1PD + 24.0 / 245.0 * G3PD;
        for (auto ab : std::initializer_list<std::pair<int,int>>{{5,2},{6,1},{7,3},{8,3}})
            G(ab.first, ab.second) = 15.0 / 245.0 * G3PD;
        // d-d Coulomb (upper)
        G(4, 5) = G(4, 6) = F0DD + 2.0 / 49.0 * F2DD - 24.0 / 441.0 * F4DD;
        G(4, 7) = G(4, 8) = F0DD - 4.0 / 49.0 * F2DD + 6.0 / 441.0 * F4DD;
        for (auto ab : std::initializer_list<std::pair<int,int>>{{5,6},{5,7},{5,8},{6,7},{6,8}})
            G(ab.first, ab.second) = F0DD - 2.0 / 49.0 * F2DD - 4.0 / 441.0 * F4DD;
        G(7, 8) = F0DD + 4.0 / 49.0 * F2DD - 34.0 / 441.0 * F4DD;
        // d-d exchange (lower)
        G(5, 4) = G(6, 4) = F2DD / 49.0 + 30.0 / 441.0 * F4DD;
        G(7, 4) = G(8, 4) = 4.0 / 49.0 * F2DD + 15.0 / 441.0 * F4DD;
        for (auto ab : std::initializer_list<std::pair<int,int>>{{6,5},{7,5},{8,5},{7,6},{8,6}})
            G(ab.first, ab.second) = 3.0 / 49.0 * F2DD + 20.0 / 441.0 * F4DD;
        G(8, 7) = 35.0 / 441.0 * F4DD;
    }
    return G;
}

// EINZI hybrid d integrals HYB(1..21), 1-indexed (einzi.f).
inline std::array<double, 22> einzi_hyb(int z, const MsindoParameterSet& p) {
    SlaterCondon f = slater_condon(z, p);
    double s3 = std::sqrt(3.0), s5 = std::sqrt(5.0);
    double R1SPPD=f.I1SPPD, R2SDPP=f.I2SDPP, R2SDDD=f.I2SDDD;
    double R2PPDD=f.F2PD, R1PDPD=f.G1PD, R3PDPD=f.G3PD;
    double R2DDDD=f.F2DD, R4DDDD=f.F4DD;
    std::array<double, 22> h{};
    h[1] = -1.0/3.0/s5 * R1SPPD;
    h[2] = 1.0/s3/s5 * R1SPPD;
    h[3] = 2.0/3.0/s5 * R1SPPD;
    h[4] = -1.0/5.0/s5 * R2SDPP;
    h[5] = 2.0/5.0/s5 * R2SDPP;
    h[6] = 2.0/7.0/s5 * R2SDDD;
    h[7] = 1.0/7.0/s5 * R2SDDD;
    h[8] = s3/5.0/s5 * R2SDPP;
    h[9] = s3/7.0/s5 * R2SDDD;
    h[10] = -2.0*s3/35.0 * R2PPDD;
    h[11] = 3.0/35.0 * R2PPDD;
    h[12] = s3/35.0 * R2PPDD;
    h[13] = -s3/15.0*R1PDPD - 3.0*s3/245.0*R3PDPD;
    h[14] = -s3/15.0*R1PDPD + 12.0*s3/245.0*R3PDPD;
    h[15] = 1.0/5.0*R1PDPD - 6.0/245.0*R3PDPD;
    h[16] = 2.0*s3/15.0*R1PDPD - 9.0*s3/245.0*R3PDPD;
    h[17] = 3.0/49.0 * R3PDPD;
    h[18] = 1.0/5.0*R1PDPD - 3.0/35.0*R3PDPD;
    h[19] = s3/49.0*R2DDDD - 5.0*s3/441.0*R4DDDD;
    h[20] = -2.0*s3/49.0*R2DDDD + 10.0*s3/441.0*R4DDDD;
    h[21] = 3.0/49.0*R2DDDD - 5.0/147.0*R4DDDD;
    return h;
}

// ENEG diagonal one-electron energies (atomic_reference.f).
// Returns (U_ss, U_pp, U_dd).
inline std::array<double, 3> eneg(int z, const MsindoParameterSet& p) {
    SlaterCondon f = slater_condon(z, p);
    double ipots = MsindoParameterSet::get(p.IPOTS, z);
    double ipotp = MsindoParameterSet::get(p.IPOTP, z);
    double ipotd = MsindoParameterSet::get(p.IPOTD, z);
    if (z == 1) return {ipots, ipotp, 0.0};
    if (z == 2) return {ipots - f.F0SS, 0.0, 0.0};
    // Alkali (Li/Na/K/Rb s^1): one electron, no electron-electron correction.
    if (z == 3 || z == 11 || z == 19 || z == 37)
        return {ipots, ipotp, 0.0};
    // Transition metals (Sc-Zn 3d + Y-Pd 4d): occupied-d ENEG.  atomic_reference.f's
    // Y-Cd block (DO L=39,48) is term-for-term identical to Sc-Zn (DO L=21,30).
    if ((z >= 21 && z <= 30) || (z >= 39 && z <= 46)) {
        double sel = MsindoParameterSet::get(p.LS, z);
        double pel = MsindoParameterSet::get(p.MP, z);
        double de  = MsindoParameterSet::get(p.ND, z);
        double e1 = ipots - (sel-1)*f.F0SS - pel*f.F0SP - de*f.F0SD
                    + pel*f.G1SP/6.0 + de*f.G2SD/10.0;
        double e2 = ipotp - (sel-1)*f.F0SP - de*f.F0PD
                    + (sel-1)*f.G1SP/6.0 + de*f.G1PD/15.0
                    + 3.0/70.0*de*f.G3PD;
        double e3 = ipotd - (de-1)*f.F0DD
                    + 2.0/63.0*(de-1)*(f.F2DD+f.F4DD)
                    - sel*f.F0SD + sel*f.G2SD/10.0
                    - pel*f.F0PD + pel*f.G1PD/15.0
                    + 3.0/70.0*pel*f.G3PD;
        return {e1*(1.0-MsindoParameterSet::get(p.SCP4S,z)),
                e2*(1.0-MsindoParameterSet::get(p.SCP4P,z)),
                e3*(1.0-MsindoParameterSet::get(p.SCP3D,z))};
    }
    // Alkaline earth (Be/Mg/Ca/Sr s^2): special empty-p ENEG
    if (z == 4 || z == 12 || z == 20 || z == 38)
        return {ipots - f.F0SS, ipotp - f.F0SP + f.G1SP/6.0, 0.0};
    // Ag (4d¹⁰5s¹), Cd (4d¹⁰5s²): occupied-4d elements that atomic_reference.f
    // computes via the In–Xe loop (DO L=47,54), which runs after the Y–Cd loop
    // and OVERWRITES L=47,48 — the In–Xe p-block formula carrying the *occupied*
    // d shell (DEL=ND=10).  Lockstep with python msindo.eneg's {47,48} branch.
    if (z == 47 || z == 48) {
        double sel = MsindoParameterSet::get(p.LS, z);
        double pel = MsindoParameterSet::get(p.MP, z);
        double de  = MsindoParameterSet::get(p.ND, z);
        double selm1 = sel - 1.0, pelm1 = pel - 1.0;
        double e1 = ipots - selm1*f.F0SS - pel*f.F0SP - de*f.F0SD
                    + pel*f.G1SP/6.0 + de*f.G2SD/10.0;
        double e2 = ipotp - pelm1*(f.F0PP - 2.0/25.0*f.F2PP)
                    - sel*(f.F0SP - f.G1SP/6.0)
                    - de*(f.F0PD - f.G1PD/15.0 - 3.0/70.0*f.G3PD);
        double e3 = ipotd - sel*f.F0SD + sel*f.G2SD/10.0
                    - pelm1*(f.F0PD + f.G1PD/15.0 + 3.0/70.0*f.G3PD);
        e3 *= (1.0 - MsindoParameterSet::get(p.SCP3D, z));
        return {e1, e2, e3};
    }
    // General p-block (B, C-F, Al-Ar, Ga-Kr, In-Xe, ...)
    double pel = MsindoParameterSet::get(p.MP, z);
    double pelm1 = pel - 1.0;
    double e1 = ipots - f.F0SS - pel*f.F0SP + pel*f.G1SP/6.0;
    double e2 = ipotp - 2.0*f.F0SP - pelm1*f.F0PP + f.G1SP/3.0 + 2.0/25.0*pelm1*f.F2PP;
    double e3 = 0.0;
    if (p.n_basis(z) >= 9) {
        if ((z >= 31 && z <= 36) || (z >= 49 && z <= 54)) {
            // Ga-Kr (4th-row) + In-Xe (5th-row) p-block: negated G1PD/G3PD exchange
            e3 = ipotd - 2.0*f.F0SD - pelm1*f.F0PD
                 - pelm1*f.G1PD/15.0 + f.G2SD/5.0
                 - 3.0/70.0*pelm1*f.G3PD;
        } else {
            e3 = ipotd - 2.0*f.F0SD - pelm1*f.F0PD
                 + pelm1*f.G1PD/15.0 + f.G2SD/5.0
                 + 3.0/70.0*pelm1*f.G3PD;
        }
        e3 *= (1.0 - MsindoParameterSet::get(p.SCP3D, z));
    }
    return {e1, e2, e3};
}

// Backward-compatible (returns 2 elements).
inline std::array<double, 2> eneg(int z) {
    auto e = eneg(z, MsindoParameterSet::embedded_hf());
    return {e[0], e[1]};
}

// ATENG atomic reference energy (atomic_reference.f).
inline double ateng(int z, const MsindoParameterSet& p) {
    SlaterCondon f = slater_condon(z, p);
    double ipots = MsindoParameterSet::get(p.IPOTS, z);
    if (z == 1) return ipots;
    if (z == 2) return 2.0*(ipots - f.F0SS) + f.F0SS;
    if (z == 3 || z == 11 || z == 19 || z == 37) return ipots; // alkali s^1
    // Occupied-d TM reference (Sc-Zn + Y-Pd); Ag/Cd (47,48) use the general
    // p-block ateng below (PEL=0 → 2·U_ss + F0SS), matching python msindo.ateng.
    if ((z >= 21 && z <= 30) || (z >= 39 && z <= 46)) {
        double sel=MsindoParameterSet::get(p.LS,z), pel=MsindoParameterSet::get(p.MP,z), de=MsindoParameterSet::get(p.ND,z);
        auto u = eneg(z, p);
        double a = sel*u[0] + pel*u[1] + de*u[2]
                 + sel*(sel-1)/2.0*f.F0SS
                 + pel*(pel-1)/2.0*(f.F0PP - 2.0/25.0*f.F2PP)
                 + de*(de-1)/2.0*(f.F0DD - 2.0/63.0*(f.F2DD+f.F4DD))
                 + sel*pel*(f.F0SP - 1.0/6.0*f.G1SP)
                 + sel*de*(f.F0SD - 1.0/10.0*f.G2SD)
                 + pel*de*(f.F0PD - 1.0/15.0*f.G1PD - 3.0/70.0*f.G3PD);
        // Per-element multiplet corrections
        double corr = 0.0;
        if (z == 21) corr = -0.0010609939;
        else if (z == 22) corr = -58.0/441.0*f.F2DD + 5.0/441.0*f.F4DD;
        else if (z == 23) corr = -93.0/441.0*f.F2DD - 30.0/441.0*f.F4DD;
        else if (z == 24) corr = -25.0/63.0*(f.F2DD+f.F4DD) - f.G2SD/2.0;
        else if (z == 25) corr = -25.0/63.0*(f.F2DD+f.F4DD);
        else if (z == 26) corr = -15.0/63.0*(f.F2DD+f.F4DD);
        else if (z == 27) corr = -93.0/441.0*f.F2DD - 30.0/441.0*f.F4DD;
        else if (z == 28) corr = -58.0/441.0*f.F2DD + 5.0/441.0*f.F4DD;
        return a + corr;
    }
    if (z == 4 || z == 12 || z == 20 || z == 38) return 2.0*eneg(z,p)[0] + f.F0SS;
    double pel = MsindoParameterSet::get(p.MP, z);
    double pelm1 = pel - 1.0;
    auto u = eneg(z, p);
    double a = 2.0*u[0] + pel*u[1] + f.F0SS + pel*pelm1*f.F0PP/2.0
             + 2.0*pel*f.F0SP - pel/3.0*f.G1SP - pel*pelm1/25.0*f.F2PP;
    // Multiplet corrections for main-group
    if (z == 6 || z == 8 || z == 14 || z == 16 || z == 32 || z == 34) a -= 3.0/25.0*f.F2PP;
    if (z == 7 || z == 15 || z == 33) a -= 9.0/25.0*f.F2PP;
    // 3rd row extra ATENG corrections
    if (z == 13) a += -0.0000297976;
    else if (z == 14) a += -0.0000300569;
    else if (z == 16) a += -0.0000342370;
    else if (z == 17) a += -0.0000368735;
    return a;
}

inline double ateng(int z) { return ateng(z, MsindoParameterSet::embedded_hf()); }

struct MsindoCosmoBMatrix {
    Eigen::MatrixXd B;
    std::vector<std::array<int, 3>> pairs;
    int nsto = 0;
};

inline std::array<double, 9> cosmo_local_penetration_diag(
        int z, double dist, int nb, const MsindoParameterSet& p) {
    std::array<double, 9> M{};
    int ns = MsindoParameterSet::n_principal(z);
    M[0] = v2int(ns, 0, 0, MsindoParameterSet::get(p.MUS, z), dist);
    if (nb >= 4) {
        int npr = MsindoParameterSet::n_p_principal(z);
        double mup = MsindoParameterSet::get(p.MUP, z);
        M[1] = v2int(npr, 1, 0, mup, dist);
        M[2] = v2int(npr, 1, 1, mup, dist);
        M[3] = M[2];
    }
    if (nb >= 9) {
        int nd = MsindoParameterSet::n_d_principal(z);
        double mud = MsindoParameterSet::get(p.MUD, z);
        M[4] = v2int(nd, 2, 0, mud, dist);
        M[5] = v2int(nd, 2, 1, mud, dist);
        M[6] = M[5];
        M[7] = v2int(nd, 2, 2, mud, dist);
        M[8] = M[7];
    }
    return M;
}

inline double cosmo_v2sas_element(
        const double T[9][9], const std::array<double, 9>& M, int a, int b, int nb) {
    double val = 0.0;
    for (int c = 0; c < nb; ++c) val += T[a][c] * M[c] * T[b][c];
    return val;
}

inline MsindoCosmoBMatrix build_cosmo_b_matrix(
        const std::vector<int>& Z,
        const Eigen::Ref<const Eigen::MatrixXd>& coords_bohr,
        const Eigen::Ref<const Eigen::MatrixXd>& cavity_points,
        const MsindoParameterSet& p) {
    const int natom = static_cast<int>(Z.size());
    std::vector<std::array<int, 2>> blocks;
    blocks.reserve(natom);
    int nsto = 0;
    for (int z : Z) {
        int nb = p.n_basis(z);
        blocks.push_back({nsto, nsto + nb});
        nsto += nb;
    }

    std::vector<std::array<int, 3>> pairs;
    std::vector<std::vector<std::array<int, 3>>> pair_lookup(natom);
    int comp = nsto;
    for (int ia = 0; ia < natom; ++ia) {
        int lo = blocks[ia][0];
        int nb = blocks[ia][1] - blocks[ia][0];
        if (nb >= 4) {
            for (int mu = lo + 2; mu <= lo + 3; ++mu) {
                for (int nu = lo + 1; nu < mu; ++nu) {
                    pairs.push_back({comp, mu, nu});
                    pair_lookup[ia].push_back({comp, mu - lo, nu - lo});
                    ++comp;
                }
            }
        }
        if (nb >= 9) {
            for (int mu = lo + 5; mu <= lo + 8; ++mu) {
                for (int nu = lo + 4; nu < mu; ++nu) {
                    pairs.push_back({comp, mu, nu});
                    pair_lookup[ia].push_back({comp, mu - lo, nu - lo});
                    ++comp;
                }
            }
        }
    }

    const int npts = static_cast<int>(cavity_points.rows());
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(comp, npts);
    for (int ia = 0; ia < natom; ++ia) {
        int lo = blocks[ia][0];
        int nb = blocks[ia][1] - blocks[ia][0];
        int maxkl = (nb == 1) ? 1 : (nb == 4 ? 2 : 3);
        for (int is = 0; is < npts; ++is) {
            Eigen::Vector3d d =
                coords_bohr.row(ia).transpose() - cavity_points.row(is).transpose();
            double dist = d.norm();
            double E[3] = {d[0] / dist, d[1] / dist, d[2] / dist};
            double T[9][9];
            harmtr(maxkl, E, T);
            auto M = cosmo_local_penetration_diag(Z[ia], dist, nb, p);
            for (int k = 0; k < nb; ++k) {
                B(lo + k, is) = -cosmo_v2sas_element(T, M, k, k, nb);
            }
            for (const auto& item : pair_lookup[ia]) {
                B(item[0], is) =
                    -cosmo_v2sas_element(T, M, item[1], item[2], nb);
            }
        }
    }

    MsindoCosmoBMatrix out;
    out.B = std::move(B);
    out.pairs = std::move(pairs);
    out.nsto = nsto;
    return out;
}

inline int cosmo_pair_component_count(
        int nsto, const std::vector<std::array<int, 3>>& pairs) {
    if (nsto < 0) {
        throw std::invalid_argument("MSINDO COSMO nsto must be non-negative");
    }
    int ncomp = nsto;
    for (const auto& item : pairs) {
        int comp = item[0], mu = item[1], nu = item[2];
        if (comp < nsto || mu < 0 || mu >= nsto || nu < 0 || nu >= nsto) {
            throw std::invalid_argument("invalid MSINDO COSMO pair layout");
        }
        ncomp = std::max(ncomp, comp + 1);
    }
    return ncomp;
}

inline Eigen::VectorXd cosmo_charge_vector(
        const Eigen::Ref<const Eigen::MatrixXd>& density,
        int nsto,
        const std::vector<std::array<int, 3>>& pairs) {
    if (density.rows() != nsto || density.cols() != nsto) {
        throw std::invalid_argument("MSINDO COSMO density shape does not match nsto");
    }
    int ncomp = cosmo_pair_component_count(nsto, pairs);
    Eigen::VectorXd Q = Eigen::VectorXd::Zero(ncomp);
    for (int i = 0; i < nsto; ++i) Q(i) = density(i, i);
    for (const auto& item : pairs) {
        Q(item[0]) = 2.0 * density(item[1], item[2]);
    }
    return Q;
}

inline Eigen::VectorXd cosmo_esp_at_cavity(
        const Eigen::Ref<const Eigen::MatrixXd>& B,
        const Eigen::Ref<const Eigen::MatrixXd>& density,
        int nsto,
        const std::vector<std::array<int, 3>>& pairs) {
    Eigen::VectorXd Q = cosmo_charge_vector(density, nsto, pairs);
    if (B.rows() != Q.size()) {
        throw std::invalid_argument("MSINDO COSMO B rows do not match charge vector");
    }
    return B.transpose() * Q;
}

inline Eigen::MatrixXd cosmo_fock_contribution(
        const Eigen::Ref<const Eigen::MatrixXd>& B,
        const Eigen::Ref<const Eigen::VectorXd>& charges,
        int nsto,
        const std::vector<std::array<int, 3>>& pairs) {
    int ncomp = cosmo_pair_component_count(nsto, pairs);
    if (B.rows() < ncomp || B.cols() != charges.size()) {
        throw std::invalid_argument("MSINDO COSMO B shape does not match charges");
    }
    Eigen::VectorXd psi = B * charges;
    Eigen::MatrixXd Vq = Eigen::MatrixXd::Zero(nsto, nsto);
    for (int i = 0; i < nsto; ++i) Vq(i, i) = psi(i);
    for (const auto& item : pairs) {
        Vq(item[1], item[2]) = psi(item[0]);
        Vq(item[2], item[1]) = psi(item[0]);
    }
    return Vq;
}

inline Eigen::VectorXd cosmo_core_potential_at_cavity(
        const Eigen::Ref<const Eigen::MatrixXd>& coords_bohr,
        const Eigen::Ref<const Eigen::VectorXd>& core_charges,
        const Eigen::Ref<const Eigen::MatrixXd>& cavity_points) {
    if (coords_bohr.cols() != 3 || cavity_points.cols() != 3 ||
        coords_bohr.rows() != core_charges.size()) {
        throw std::invalid_argument("MSINDO COSMO core potential shape mismatch");
    }
    Eigen::VectorXd V = Eigen::VectorXd::Zero(cavity_points.rows());
    for (int is = 0; is < cavity_points.rows(); ++is) {
        double val = 0.0;
        for (int ia = 0; ia < coords_bohr.rows(); ++ia) {
            Eigen::Vector3d d =
                coords_bohr.row(ia).transpose() - cavity_points.row(is).transpose();
            double dist = d.norm();
            if (dist == 0.0) {
                throw std::invalid_argument("MSINDO COSMO cavity point sits on atom");
            }
            val += core_charges(ia) / dist;
        }
        V(is) = val;
    }
    return V;
}

struct MsindoCosmoReactionField {
    Eigen::MatrixXd fock;
    Eigen::VectorXd q;
    Eigen::VectorXd V_total;
    double e_add = 0.0;
    double e_pol = 0.0;
};

inline MsindoCosmoReactionField cosmo_reaction_field(
        const Eigen::Ref<const Eigen::MatrixXd>& B,
        const Eigen::Ref<const Eigen::MatrixXd>& A,
        const Eigen::Ref<const Eigen::VectorXd>& core_potential,
        const Eigen::Ref<const Eigen::MatrixXd>& density,
        int nsto,
        const std::vector<std::array<int, 3>>& pairs,
        double epsilon,
        const std::string& variant = "cosmo") {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("MSINDO COSMO reaction field A must be square");
    }
    if (B.cols() != A.rows() || core_potential.size() != A.rows()) {
        throw std::invalid_argument(
            "MSINDO COSMO reaction field cavity dimensions do not match");
    }
    // Screening factor comes from the shared helper, never re-derived here.
    // This header used to write (epsilon - 1)/(epsilon + 0.5) out by hand
    // without even including solvation_cpcm.hpp: a fourth independent copy
    // of one constant, and a special case that made MSINDO COSMO the only
    // route unable to take CPCM screening. #546 is what such a copy costs
    // once it stops agreeing (#548). Rejecting epsilon <= 1 is the shared
    // helper's job too, so the message stays identical across every route.
    const double f = cpcm_dielectric_factor(epsilon, variant);

    const Eigen::VectorXd V_elec = cosmo_esp_at_cavity(B, density, nsto, pairs);
    const Eigen::VectorXd V_total = V_elec + core_potential;
    const Eigen::VectorXd q = A.partialPivLu().solve(-f * V_total);
    if (!q.allFinite()) {
        throw std::runtime_error("MSINDO COSMO reaction-field solve failed");
    }

    MsindoCosmoReactionField out;
    out.fock = cosmo_fock_contribution(B, q, nsto, pairs);
    out.q = q;
    out.V_total = V_total;
    out.e_add = 0.5 * q.dot(core_potential);
    out.e_pol = 0.5 * q.dot(V_total);
    return out;
}

inline double vfak(int z_self, int z_other, const MsindoParameterSet& p) {
    int nb = p.n_basis(z_other);
    if (nb >= 9) return 0.5;
    if (nb >= 4) return 0.75;
    return 1.0;
}
inline double vfak(int z_self, int z_other) {
    return vfak(z_self, z_other, MsindoParameterSet::embedded_hf());
}

// --------------------------------------------------------------------------- //
// Per-pair core Hamiltonian (intdrv.f flow).                                  //
// --------------------------------------------------------------------------- //
struct PairBlocks {
    Eigen::Matrix<double, 9, 9> HK1, HL1, HKL2;
    double R;
};

struct CoreH {
    double HSS=0, HPS=0, HPP=0, HDS=0, HDP=0, HDD=0;
};

// "Rumpf" overlaps: valence(z_val) with core(z_core) frozen orbitals.
struct RumpfOverlaps {
    double S1S=0, S2S=0, S2PS=0, PS1S=0, PS2S=0, PS2PS=0, PP2PP=0;
    double DS1S=0, DS2S=0, DS2PS=0, DP2PP=0;
    double S3S=0, S3PS=0, PS3S=0, PS3PS=0, PP3PP=0, DS3S=0, DS3PS=0, DP3PP=0;
    double S3DS=0, PS3DS=0, PP3DP=0, DS3DS=0, DP3DP=0, DD3DD=0;
    // [Kr] core 4s/4p (z_core>36) and [Kr]4d¹⁰ core 4d (z_core>48).
    double S4S=0, S4PS=0, PS4S=0, PS4PS=0, PP4PP=0, DS4S=0, DS4PS=0, DP4PP=0;
    double S4DS=0, PS4DS=0, PP4DP=0, DS4DS=0, DP4DP=0, DD4DD=0;
};

inline RumpfOverlaps rumpf_overlaps(int z_val, int z_core, double R, const MsindoParameterSet& p) {
    RumpfOverlaps ri{};
    if (z_core <= 2) return ri;
    bool val_p = p.n_basis(z_val) >= 4;
    bool val_d = p.n_basis(z_val) >= 9 && MsindoParameterSet::get(p.MUD, z_val) != 0.0;
    int ns = MsindoParameterSet::n_principal(z_val);
    int npr = MsindoParameterSet::n_p_principal(z_val);
    int nd = MsindoParameterSet::n_d_principal(z_val);
    double z1s = MsindoParameterSet::get(p.TAU1S, z_core);
    double zs = MsindoParameterSet::get(p.MUS, z_val);
    double zp = val_p ? MsindoParameterSet::get(p.MUP, z_val) : 0.0;
    double zd = val_d ? MsindoParameterSet::get(p.MUD, z_val) : 0.0;
    ri.S1S = s2int(ns, 0, 0, zs, 1, 0, 0, z1s, R);
    if (val_p) ri.PS1S = s2int(npr, 1, 0, zp, 1, 0, 0, z1s, R);
    if (val_d) ri.DS1S = s2int(nd, 2, 0, zd, 1, 0, 0, z1s, R);
    if (z_core > 10) {
        double z2s = MsindoParameterSet::get(p.TAU2S, z_core);
        double z2p = MsindoParameterSet::get(p.TAU2P, z_core);
        ri.S2S = s2int(ns, 0, 0, zs, 2, 0, 0, z2s, R);
        ri.S2PS = s2int(ns, 0, 0, zs, 2, 1, 0, z2p, R);
        if (val_p) {
            ri.PS2S = s2int(npr, 1, 0, zp, 2, 0, 0, z2s, R);
            ri.PS2PS = s2int(npr, 1, 0, zp, 2, 1, 0, z2p, R);
            ri.PP2PP = s2int(npr, 1, 1, zp, 2, 1, 1, z2p, R);
        }
        if (val_d) {
            ri.DS2S = s2int(nd, 2, 0, zd, 2, 0, 0, z2s, R);
            ri.DS2PS = s2int(nd, 2, 0, zd, 2, 1, 0, z2p, R);
            ri.DP2PP = s2int(nd, 2, 1, zd, 2, 1, 1, z2p, R);
        }
    }
    if (z_core > 18) {
        double z3s = MsindoParameterSet::get(p.TAU3S, z_core);
        double z3p = MsindoParameterSet::get(p.TAU3P, z_core);
        ri.S3S = s2int(ns, 0, 0, zs, 3, 0, 0, z3s, R);
        ri.S3PS = s2int(ns, 0, 0, zs, 3, 1, 0, z3p, R);
        if (val_p) {
            ri.PS3S = s2int(npr, 1, 0, zp, 3, 0, 0, z3s, R);
            ri.PS3PS = s2int(npr, 1, 0, zp, 3, 1, 0, z3p, R);
            ri.PP3PP = s2int(npr, 1, 1, zp, 3, 1, 1, z3p, R);
        }
        if (val_d) {
            ri.DS3S = s2int(nd, 2, 0, zd, 3, 0, 0, z3s, R);
            ri.DS3PS = s2int(nd, 2, 0, zd, 3, 1, 0, z3p, R);
            ri.DP3PP = s2int(nd, 2, 1, zd, 3, 1, 1, z3p, R);
        }
    }
    if (z_core > 30) {
        double z3d = MsindoParameterSet::get(p.TAU3D, z_core);
        ri.S3DS = s2int(ns, 0, 0, zs, 3, 2, 0, z3d, R);
        if (val_p) {
            ri.PS3DS = s2int(npr, 1, 0, zp, 3, 2, 0, z3d, R);
            ri.PP3DP = s2int(npr, 1, 1, zp, 3, 2, 1, z3d, R);
        }
        if (val_d) {
            ri.DS3DS = s2int(nd, 2, 0, zd, 3, 2, 0, z3d, R);
            ri.DP3DP = s2int(nd, 2, 1, zd, 3, 2, 1, z3d, R);
            ri.DD3DD = s2int(nd, 2, 2, zd, 3, 2, 2, z3d, R);
        }
    }
    if (z_core > 36) {  // [Kr] core (Rb–Xe): add the 4s, 4p core shells
        double z4s = MsindoParameterSet::get(p.TAU4S, z_core);
        double z4p = MsindoParameterSet::get(p.TAU4P, z_core);
        ri.S4S = s2int(ns, 0, 0, zs, 4, 0, 0, z4s, R);
        ri.S4PS = s2int(ns, 0, 0, zs, 4, 1, 0, z4p, R);
        if (val_p) {
            ri.PS4S = s2int(npr, 1, 0, zp, 4, 0, 0, z4s, R);
            ri.PS4PS = s2int(npr, 1, 0, zp, 4, 1, 0, z4p, R);
            ri.PP4PP = s2int(npr, 1, 1, zp, 4, 1, 1, z4p, R);
        }
        if (val_d) {
            ri.DS4S = s2int(nd, 2, 0, zd, 4, 0, 0, z4s, R);
            ri.DS4PS = s2int(nd, 2, 0, zd, 4, 1, 0, z4p, R);
            ri.DP4PP = s2int(nd, 2, 1, zd, 4, 1, 1, z4p, R);
        }
    }
    if (z_core > 48) {  // [Kr]4d¹⁰ core (In–Xe): add the 4d core shell
        double z4d = MsindoParameterSet::get(p.TAU4D, z_core);
        ri.S4DS = s2int(ns, 0, 0, zs, 4, 2, 0, z4d, R);
        if (val_p) {
            ri.PS4DS = s2int(npr, 1, 0, zp, 4, 2, 0, z4d, R);
            ri.PP4DP = s2int(npr, 1, 1, zp, 4, 2, 1, z4d, R);
        }
        if (val_d) {
            ri.DS4DS = s2int(nd, 2, 0, zd, 4, 2, 0, z4d, R);
            ri.DP4DP = s2int(nd, 2, 1, zd, 4, 2, 1, z4d, R);
            ri.DD4DD = s2int(nd, 2, 2, zd, 4, 2, 2, z4d, R);
        }
    }
    return ri;
}

// Structure for VCORR gamma cache keys.
struct GammaCache {
    double ps=0, pp=0, pd=0, ds=0, dp=0, dd=0;
};

inline CoreH v2core_one_side(int z_val, int z_core, double R, const GammaCache& gam,
                              const MsindoParameterSet& p) {
    double zc = MsindoParameterSet::eff_core_charge(z_core);
    bool val_p = p.n_basis(z_val) >= 4;
    bool val_d = p.n_basis(z_val) >= 9 && MsindoParameterSet::get(p.MUD, z_val) != 0.0;
    int ns = MsindoParameterSet::n_principal(z_val);
    int npr = MsindoParameterSet::n_p_principal(z_val);
    int nd = MsindoParameterSet::n_d_principal(z_val);
    double zs = MsindoParameterSet::get(p.MUS, z_val);
    double zp = val_p ? MsindoParameterSet::get(p.MUP, z_val) : 0.0;
    double zd = val_d ? MsindoParameterSet::get(p.MUD, z_val) : 0.0;

    RumpfOverlaps ri = rumpf_overlaps(z_val, z_core, R, p);
    CoreH h;

    auto fcp_sum = [&](double r1, double r2, double r2p, double r3, double r3p, double r3d,
                       double r4s, double r4p, double r4d) {
        double s = 0.0;
        if (z_core > 2)  s += MsindoParameterSet::get(p.FCP1S, z_core) * r1 * r1;
        if (z_core > 10) s += MsindoParameterSet::get(p.FCP2S, z_core) * r2 * r2
                            + MsindoParameterSet::get(p.FCP2P, z_core) * r2p * r2p;
        if (z_core > 18) s += MsindoParameterSet::get(p.FCP3S, z_core) * r3 * r3
                            + MsindoParameterSet::get(p.FCP3P, z_core) * r3p * r3p;
        if (z_core > 30) s += MsindoParameterSet::get(p.FCP3D, z_core) * r3d * r3d;
        if (z_core > 36) s += MsindoParameterSet::get(p.FCP4S, z_core) * r4s * r4s
                            + MsindoParameterSet::get(p.FCP4P, z_core) * r4p * r4p;
        if (z_core > 48) s += MsindoParameterSet::get(p.FCP4D, z_core) * r4d * r4d;
        return s;
    };

    h.HSS = -zc * v2int(ns, 0, 0, zs, R)
            + fcp_sum(ri.S1S, ri.S2S, ri.S2PS, ri.S3S, ri.S3PS, ri.S3DS,
                      ri.S4S, ri.S4PS, ri.S4DS);
    if (val_p) {
        h.HPS = -zc * v2int(npr, 1, 0, zp, R);
        h.HPP = -zc * v2int(npr, 1, 1, zp, R);
        if (z_core > 2)  h.HPS += MsindoParameterSet::get(p.FCP1S, z_core) * ri.PS1S * ri.PS1S;
        if (z_core > 10) {
            h.HPS += MsindoParameterSet::get(p.FCP2S, z_core) * ri.PS2S * ri.PS2S
                   + MsindoParameterSet::get(p.FCP2P, z_core) * ri.PS2PS * ri.PS2PS;
            h.HPP += MsindoParameterSet::get(p.FCP2P, z_core) * ri.PP2PP * ri.PP2PP;
        }
        if (z_core > 18) {
            h.HPS += MsindoParameterSet::get(p.FCP3S, z_core) * ri.PS3S * ri.PS3S
                   + MsindoParameterSet::get(p.FCP3P, z_core) * ri.PS3PS * ri.PS3PS;
            h.HPP += MsindoParameterSet::get(p.FCP3P, z_core) * ri.PP3PP * ri.PP3PP;
        }
        if (z_core > 30) {
            h.HPS += MsindoParameterSet::get(p.FCP3D, z_core) * ri.PS3DS * ri.PS3DS;
            h.HPP += MsindoParameterSet::get(p.FCP3D, z_core) * ri.PP3DP * ri.PP3DP;
        }
        if (z_core > 36) {
            h.HPS += MsindoParameterSet::get(p.FCP4S, z_core) * ri.PS4S * ri.PS4S
                   + MsindoParameterSet::get(p.FCP4P, z_core) * ri.PS4PS * ri.PS4PS;
            h.HPP += MsindoParameterSet::get(p.FCP4P, z_core) * ri.PP4PP * ri.PP4PP;
        }
        if (z_core > 48) {
            h.HPS += MsindoParameterSet::get(p.FCP4D, z_core) * ri.PS4DS * ri.PS4DS;
            h.HPP += MsindoParameterSet::get(p.FCP4D, z_core) * ri.PP4DP * ri.PP4DP;
        }
    }
    if (val_d) {
        h.HDS = -zc * v2int(nd, 2, 0, zd, R);
        h.HDP = -zc * v2int(nd, 2, 1, zd, R);
        h.HDD = -zc * v2int(nd, 2, 2, zd, R);
        if (z_core > 2)  h.HDS += MsindoParameterSet::get(p.FCP1S, z_core) * ri.DS1S * ri.DS1S;
        if (z_core > 10) {
            h.HDS += MsindoParameterSet::get(p.FCP2S, z_core) * ri.DS2S * ri.DS2S
                   + MsindoParameterSet::get(p.FCP2P, z_core) * ri.DS2PS * ri.DS2PS;
            h.HDP += MsindoParameterSet::get(p.FCP2P, z_core) * ri.DP2PP * ri.DP2PP;
        }
        if (z_core > 18) {
            h.HDS += MsindoParameterSet::get(p.FCP3S, z_core) * ri.DS3S * ri.DS3S
                   + MsindoParameterSet::get(p.FCP3P, z_core) * ri.DS3PS * ri.DS3PS;
            h.HDP += MsindoParameterSet::get(p.FCP3P, z_core) * ri.DP3PP * ri.DP3PP;
        }
        if (z_core > 30) {
            h.HDS += MsindoParameterSet::get(p.FCP3D, z_core) * ri.DS3DS * ri.DS3DS;
            h.HDP += MsindoParameterSet::get(p.FCP3D, z_core) * ri.DP3DP * ri.DP3DP;
            h.HDD += MsindoParameterSet::get(p.FCP3D, z_core) * ri.DD3DD * ri.DD3DD;
        }
        if (z_core > 36) {
            h.HDS += MsindoParameterSet::get(p.FCP4S, z_core) * ri.DS4S * ri.DS4S
                   + MsindoParameterSet::get(p.FCP4P, z_core) * ri.DS4PS * ri.DS4PS;
            h.HDP += MsindoParameterSet::get(p.FCP4P, z_core) * ri.DP4PP * ri.DP4PP;
        }
        if (z_core > 48) {
            h.HDS += MsindoParameterSet::get(p.FCP4D, z_core) * ri.DS4DS * ri.DS4DS;
            h.HDP += MsindoParameterSet::get(p.FCP4D, z_core) * ri.DP4DP * ri.DP4DP;
            h.HDD += MsindoParameterSet::get(p.FCP4D, z_core) * ri.DD4DD * ri.DD4DD;
        }
    }

    // VCORRK penetration: valence p/d sees z_core's valence electrons
    int npc = MsindoParameterSet::n_principal(z_core);
    double zcs = MsindoParameterSet::get(p.MUS, z_core);
    double zcp = MsindoParameterSet::get(p.MUP, z_core);
    double zcd = MsindoParameterSet::get(p.MUD, z_core);
    int LSz = MsindoParameterSet::iget(p.LS, z_core);
    int MPz = MsindoParameterSet::iget(p.MP, z_core);
    int NDz = MsindoParameterSet::iget(p.ND, z_core);
    if (val_p) {
        double coul_pss = c2int(npc, 0, 0, zcs, npr, 1, 0, zp, R);
        double coul_psp = c2int(npc, 0, 0, zcs, npr, 1, 1, zp, R);
        h.HPS += (coul_pss - gam.ps) * LSz;
        h.HPP += (coul_psp - gam.ps) * LSz;
        if (MPz != 0) {
            int npcp = MsindoParameterSet::n_p_principal(z_core);
            double coul_pps = c2int(npcp, 0, 0, zcp, npr, 1, 0, zp, R);
            double coul_ppp = c2int(npcp, 0, 0, zcp, npr, 1, 1, zp, R);
            h.HPS += (coul_pps - gam.pp) * MPz;
            h.HPP += (coul_ppp - gam.pp) * MPz;
        }
        if (NDz != 0 && zcd != 0.0) {
            int ncd = MsindoParameterSet::n_d_principal(z_core);
            double coul_pds = c2int(ncd, 0, 0, zcd, npr, 1, 0, zp, R);
            double coul_pdp = c2int(ncd, 0, 0, zcd, npr, 1, 1, zp, R);
            h.HPS += (coul_pds - gam.pd) * NDz;
            h.HPP += (coul_pdp - gam.pd) * NDz;
        }
    }
    if (val_d) {
        double coul_dss = c2int(npc, 0, 0, zcs, nd, 2, 0, zd, R);
        double coul_dsp = c2int(npc, 0, 0, zcs, nd, 2, 1, zd, R);
        double coul_dsd = c2int(npc, 0, 0, zcs, nd, 2, 2, zd, R);
        h.HDS += (coul_dss - gam.ds) * LSz;
        h.HDP += (coul_dsp - gam.ds) * LSz;
        h.HDD += (coul_dsd - gam.ds) * LSz;
        if (MPz != 0) {
            int npcp = MsindoParameterSet::n_p_principal(z_core);
            double coul_dps = c2int(npcp, 0, 0, zcp, nd, 2, 0, zd, R);
            double coul_dpp = c2int(npcp, 0, 0, zcp, nd, 2, 1, zd, R);
            double coul_dpd = c2int(npcp, 0, 0, zcp, nd, 2, 2, zd, R);
            h.HDS += (coul_dps - gam.dp) * MPz;
            h.HDP += (coul_dpp - gam.dp) * MPz;
            h.HDD += (coul_dpd - gam.dp) * MPz;
        }
        if (NDz != 0 && zcd != 0.0) {
            int ncd = MsindoParameterSet::n_d_principal(z_core);
            double coul_dds = c2int(ncd, 0, 0, zcd, nd, 2, 0, zd, R);
            double coul_ddp = c2int(ncd, 0, 0, zcd, nd, 2, 1, zd, R);
            double coul_ddd = c2int(ncd, 0, 0, zcd, nd, 2, 2, zd, R);
            h.HDS += (coul_dds - gam.dd) * NDz;
            h.HDP += (coul_ddp - gam.dd) * NDz;
            h.HDD += (coul_ddd - gam.dd) * NDz;
        }
    }
    return h;
}

// --------------------------------------------------------------------------- //
// Löwdin orthogonalization (lmunu.f)                                           //
// --------------------------------------------------------------------------- //
struct Lmunu {
    double ss=0, ps=0, sp=0, pp=0, pipi=0;
    double ds=0, sd=0, dp=0, dppi=0, pd=0, pdpi=0;
    double dd=0, ddpi=0, dddel=0;
};

// Reduced local overlaps struct.
struct LocalS {
    double ss=0, ps=0, sp=0, pp=0, pipi=0;
    double ds=0, sd=0, dp=0, dppi=0, pd=0, pdpi=0;
    double dd=0, ddpi=0, dddel=0;
};

inline Lmunu lmunu_local(int zk, int zl, double R, const LocalS& S,
                          const MsindoParameterSet& p) {
    double zsK=MsindoParameterSet::get(p.MUS,zk), zpK=MsindoParameterSet::get(p.MUP,zk);
    double zdK=MsindoParameterSet::get(p.MUD,zk);
    double zsL=MsindoParameterSet::get(p.MUS,zl), zpL=MsindoParameterSet::get(p.MUP,zl);
    double zdL=MsindoParameterSet::get(p.MUD,zl);
    bool kp=p.n_basis(zk)>=4, lp=p.n_basis(zl)>=4;
    bool kd=p.n_basis(zk)>=9 && zdK!=0.0, ld=p.n_basis(zl)>=9 && zdL!=0.0;

    auto dcor = [&](double za, double zb) {
        return -0.5*(za*za+zb*zb)/(1.0+0.5*(za+zb)*R);
    };
    Lmunu M;
    M.ss = dcor(zsK, zsL)*S.ss*(1.0-S.ss);
    if (kp && zk>2) M.ps = dcor(zpK, zsL)*S.ps*(1.0-S.ps);
    if (lp && zl>2) M.sp = dcor(zsK, zpL)*S.sp*(1.0+S.sp);
    if (kp && lp && zk>2 && zl>2) {
        M.pp = dcor(zpK, zpL)*S.pp*(1.0-std::abs(S.pp));
        M.pipi = dcor(zpK, zpL)*S.pipi*(1.0-std::abs(S.pipi));
    }
    if (kd) {
        M.ds = dcor(zdK, zsL)*S.ds*(1.0-S.ds);
        if (lp) { M.dp = dcor(zdK, zpL)*S.dp*(1.0-std::abs(S.dp)); M.dppi = dcor(zdK, zpL)*S.dppi*(1.0-std::abs(S.dppi)); }
    }
    if (ld) {
        M.sd = dcor(zsK, zdL)*S.sd*(1.0-S.sd);
        if (kp) { M.pd = dcor(zpK, zdL)*S.pd*(1.0-std::abs(S.pd)); M.pdpi = dcor(zpK, zdL)*S.pdpi*(1.0-std::abs(S.pdpi)); }
    }
    if (kd && ld) {
        M.dd = dcor(zdK, zdL)*S.dd*(1.0-std::abs(S.dd));
        M.ddpi = dcor(zdK, zdL)*S.ddpi*(1.0-std::abs(S.ddpi));
        M.dddel = dcor(zdK, zdL)*S.dddel*(1.0-std::abs(S.dddel));
    }
    if (zk>2 && zl>2) return M;

    auto avg = [&](double za, double zb, double s) {
        double dcort = 0.5*(za+zb)*R;
        double eterm = (1.0-std::exp(-dcort))/(1.0+dcort);
        return -s*eterm;
    };
    if (zk==1 && zl>2) {
        M.ss = (M.ss+avg(zsK,zsL,S.ss))/2.0;
        M.sp = (M.sp+avg(zsK,zpL,S.sp))/2.0;
        if (ld) M.sd = (M.sd+avg(zsK,zdL,S.sd))/2.0;
    } else if (zk>2 && zl==1) {
        M.ss = (M.ss+avg(zsK,zsL,S.ss))/2.0;
        M.ps = (M.ps+avg(zpK,zsL,S.ps))/2.0;
        if (kd) M.ds = (M.ds+avg(zdK,zsL,S.ds))/2.0;
    } else {
        M.ss = (M.ss+avg(zsK,zsL,S.ss))/2.0;
    }
    return M;
}

// --------------------------------------------------------------------------- //
// Pair blocks (intdrv.f/rotint.f)                                             //
// --------------------------------------------------------------------------- //
inline PairBlocks pair_blocks(int zk, int zl,
                               const std::array<double, 3>& rk,
                               const std::array<double, 3>& rl,
                               const MsindoParameterSet& p) {
    double d[3]={rl[0]-rk[0], rl[1]-rk[1], rl[2]-rk[2]};
    double R = std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
    double E[3]={d[0]/R, d[1]/R, d[2]/R};
    bool kp=p.n_basis(zk)>=4, lp=p.n_basis(zl)>=4;
    double zdK=MsindoParameterSet::get(p.MUD,zk), zdL=MsindoParameterSet::get(p.MUD,zl);
    bool kd=p.n_basis(zk)>=9 && zdK!=0.0, ld=p.n_basis(zl)>=9 && zdL!=0.0;
    int nk=MsindoParameterSet::n_principal(zk), nl=MsindoParameterSet::n_principal(zl);
    int npk=MsindoParameterSet::n_p_principal(zk), npl=MsindoParameterSet::n_p_principal(zl);
    int nkd=MsindoParameterSet::n_d_principal(zk), nld=MsindoParameterSet::n_d_principal(zl);
    double zsK=MsindoParameterSet::get(p.MUS,zk), zpK=MsindoParameterSet::get(p.MUP,zk);
    double zsL=MsindoParameterSet::get(p.MUS,zl), zpL=MsindoParameterSet::get(p.MUP,zl);

    // Reduced local overlaps
    LocalS S;
    S.ss = s2int(nk,0,0,zsK, nl,0,0,zsL, R);
    if (kp) S.ps = s2int(npk,1,0,zpK, nl,0,0,zsL, R);
    if (lp) S.sp = s2int(nk,0,0,zsK, npl,1,0,zpL, R);
    if (kp&&lp) { S.pp=s2int(npk,1,0,zpK, npl,1,0,zpL,R); S.pipi=s2int(npk,1,1,zpK, npl,1,1,zpL,R); }
    if (kd) {
        S.ds = s2int(nkd,2,0,zdK, nl,0,0,zsL, R);
        if (lp) { S.dp=s2int(nkd,2,0,zdK, npl,1,0,zpL,R); S.dppi=s2int(nkd,2,1,zdK, npl,1,1,zpL,R); }
    }
    if (ld) {
        S.sd = s2int(nk,0,0,zsK, nld,2,0,zdL, R);
        if (kp) { S.pd=s2int(npk,1,0,zpK, nld,2,0,zdL,R); S.pdpi=s2int(npk,1,1,zpK, nld,2,1,zdL,R); }
    }
    if (kd&&ld) {
        S.dd=s2int(nkd,2,0,zdK, nld,2,0,zdL,R);
        S.ddpi=s2int(nkd,2,1,zdK, nld,2,1,zdL,R);
        S.dddel=s2int(nkd,2,2,zdK, nld,2,2,zdL,R);
    }

    // VCORR gamma cache
    GammaCache gam_k{}, gam_l{};
    if (kp) { gam_k.ps = c2int(npk,0,0,zpK, nl,0,0,zsL,R); }
    if (lp) { gam_l.ps = c2int(npl,0,0,zpL, nk,0,0,zsK,R); }
    if (kp&&lp) { gam_k.pp = c2int(npk,0,0,zpK, npl,0,0,zpL,R); gam_l.pp = gam_k.pp; }
    if (kp&&ld) { gam_k.pd = c2int(npk,0,0,zpK, nld,0,0,zdL,R); }
    if (lp&&kd) { gam_l.pd = c2int(npl,0,0,zpL, nkd,0,0,zdK,R); }
    if (kd) { gam_k.ds = c2int(nkd,0,0,zdK, nl,0,0,zsL,R); }
    if (ld) { gam_l.ds = c2int(nld,0,0,zdL, nk,0,0,zsK,R); }
    if (kd&&lp) { gam_k.dp = c2int(nkd,0,0,zdK, npl,0,0,zpL,R); }
    if (ld&&kp) { gam_l.dp = c2int(nld,0,0,zdL, npk,0,0,zpK,R); }
    if (kd&&ld) { gam_k.dd = c2int(nkd,0,0,zdK, nld,0,0,zdL,R); gam_l.dd = gam_k.dd; }

    CoreH hk = v2core_one_side(zk, zl, R, gam_k, p);
    CoreH hl = v2core_one_side(zl, zk, R, gam_l, p);

    auto ek = eneg(zk, p), el = eneg(zl, p);
    double shk_ss=ek[0]+hk.HSS, shk_ps=ek[1]+hk.HPS, shk_pp=ek[1]+hk.HPP, shk_ds=ek[2]+hk.HDS, shk_dp=ek[2]+hk.HDP, shk_dd=ek[2]+hk.HDD;
    double shl_ss=el[0]+hl.HSS, shl_ps=el[1]+hl.HPS, shl_pp=el[1]+hl.HPP, shl_ds=el[2]+hl.HDS, shl_dp=el[2]+hl.HDP, shl_dd=el[2]+hl.HDD;

    Lmunu M = lmunu_local(zk, zl, R, S, p);

    // HORTH: H** -= VFAK·S·L
    double vk = vfak(zk, zl, p), vl = vfak(zl, zk, p);
    { double t=S.ss*M.ss; hk.HSS-=vk*t; hl.HSS-=vl*t; }
    if (lp) { double t=S.sp*M.sp; hk.HSS-=vk*t; hl.HPS-=vl*t; }
    if (kp) { double t=S.ps*M.ps; hk.HPS-=vk*t; hl.HSS-=vl*t; }
    if (kp&&lp) { hk.HPS-=vk*S.pp*M.pp; hl.HPS-=vl*S.pp*M.pp; hk.HPP-=vk*S.pipi*M.pipi; hl.HPP-=vl*S.pipi*M.pipi; }
    if (ld) { double t=S.sd*M.sd; hk.HSS-=vk*t; hl.HDS-=vl*t; }
    if (kd) { double t=S.ds*M.ds; hk.HDS-=vk*t; hl.HSS-=vl*t; }
    if (kp&&ld) { hk.HPS-=vk*S.pd*M.pd; hl.HDS-=vl*S.pd*M.pd; hk.HPP-=vk*S.pdpi*M.pdpi; hl.HDP-=vl*S.pdpi*M.pdpi; }
    if (kd&&lp) { hk.HDS-=vk*S.dp*M.dp; hl.HPS-=vl*S.dp*M.dp; hk.HDP-=vk*S.dppi*M.dppi; hl.HPP-=vl*S.dppi*M.dppi; }
    if (kd&&ld) { hk.HDS-=vk*S.dd*M.dd; hl.HDS-=vl*S.dd*M.dd; hk.HDP-=vk*S.ddpi*M.ddpi; hl.HDP-=vl*S.ddpi*M.ddpi; hk.HDD-=vk*S.dddel*M.dddel; hl.HDD-=vl*S.dddel*M.dddel; }

    // DELTAH resonance
    double fack=1.0-std::exp(-p.AL_val(zk,zl)*R), facl=1.0-std::exp(-p.AL_val(zl,zk)*R);
    auto reson = [&](double kk_k, double kk_l, double sval, double sh_k, double sh_l, double mval) {
        return 0.25*(kk_k+kk_l)*sval*(fack*sh_k+facl*sh_l)+mval;
    };
    Eigen::Matrix<double,9,9> core = Eigen::Matrix<double,9,9>::Zero();
    double KSSk=MsindoParameterSet::get(p.KSS,zk), KSSl=MsindoParameterSet::get(p.KSS,zl);
    double KPSk=MsindoParameterSet::get(p.KPS,zk), KPSl=MsindoParameterSet::get(p.KPS,zl);
    double KPPk=MsindoParameterSet::get(p.KPP,zk), KPPl=MsindoParameterSet::get(p.KPP,zl);
    double KDSk=MsindoParameterSet::get(p.KDS,zk), KDSl=MsindoParameterSet::get(p.KDS,zl);
    double KDPk=MsindoParameterSet::get(p.KDP,zk), KDPl=MsindoParameterSet::get(p.KDP,zl);
    double KDDk=MsindoParameterSet::get(p.KDD,zk), KDDl=MsindoParameterSet::get(p.KDD,zl);
    core(0,0)=reson(KSSk,KSSl,S.ss,shk_ss,shl_ss,M.ss);
    if (kp) core(1,0)=reson(KPSk,KSSl,S.ps,shk_ps,shl_ss,M.ps);
    if (lp) core(0,1)=reson(KSSk,KPSl,S.sp,shk_ss,shl_ps,M.sp);
    if (kp&&lp) {
        core(1,1)=reson(KPSk,KPSl,S.pp,shk_ps,shl_ps,M.pp);
        double cpp=reson(KPPk,KPPl,S.pipi,shk_pp,shl_pp,M.pipi);
        core(2,2)=cpp; core(3,3)=cpp;
    }
    if (kd) core(4,0)=reson(KDSk,KSSl,S.ds,shk_ds,shl_ss,M.ds);
    if (ld) core(0,4)=reson(KSSk,KDSl,S.sd,shk_ss,shl_ds,M.sd);
    if (kd&&lp) {
        core(4,1)=reson(KDSk,KPSl,S.dp,shk_ds,shl_ps,M.dp);
        double cdp=reson(KDPk,KPPl,S.dppi,shk_dp,shl_pp,M.dppi);
        core(5,2)=cdp; core(6,3)=cdp;
    }
    if (kp&&ld) {
        core(1,4)=reson(KPSk,KDSl,S.pd,shk_ps,shl_ds,M.pd);
        double cpd=reson(KPPk,KDPl,S.pdpi,shk_pp,shl_dp,M.pdpi);
        core(2,5)=cpd; core(3,6)=cpd;
    }
    if (kd&&ld) {
        core(4,4)=reson(KDSk,KDSl,S.dd,shk_ds,shl_ds,M.dd);
        double cddpi=reson(KDPk,KDPl,S.ddpi,shk_dp,shl_dp,M.ddpi);
        core(5,5)=cddpi; core(6,6)=cddpi;
        double cdddel=reson(KDDk,KDDl,S.dddel,shk_dd,shl_dd,M.dddel);
        core(7,7)=cdddel; core(8,8)=cdddel;
    }

    double Traw[9][9];
    Traw[0][0]=Traw[0][1]=Traw[0][2]=Traw[0][3]=Traw[0][4]=Traw[0][5]=Traw[0][6]=Traw[0][7]=Traw[0][8]=0;
    harmtr(3, E, Traw);
    Eigen::Matrix<double,9,9> T;
    for (int i=0;i<9;++i) for (int j=0;j<9;++j) T(i,j)=Traw[i][j];

    Eigen::Matrix<double,9,9> diag_k=Eigen::Matrix<double,9,9>::Zero();
    diag_k(0,0)=hk.HSS; diag_k(1,1)=hk.HPS; diag_k(2,2)=hk.HPP; diag_k(3,3)=hk.HPP;
    diag_k(4,4)=hk.HDS; diag_k(5,5)=hk.HDP; diag_k(6,6)=hk.HDP; diag_k(7,7)=hk.HDD; diag_k(8,8)=hk.HDD;
    Eigen::Matrix<double,9,9> diag_l=Eigen::Matrix<double,9,9>::Zero();
    diag_l(0,0)=hl.HSS; diag_l(1,1)=hl.HPS; diag_l(2,2)=hl.HPP; diag_l(3,3)=hl.HPP;
    diag_l(4,4)=hl.HDS; diag_l(5,5)=hl.HDP; diag_l(6,6)=hl.HDP; diag_l(7,7)=hl.HDD; diag_l(8,8)=hl.HDD;

    PairBlocks pb;
    pb.HK1 = T*diag_k*T.transpose();
    pb.HL1 = T*diag_l*T.transpose();
    pb.HKL2 = T*core*T.transpose();
    pb.R = R;
    return pb;
}

// Backward-compatible (embedded H-F set)
inline PairBlocks pair_blocks(int zk, int zl, const std::array<double,3>& rk,
                               const std::array<double,3>& rl) {
    return pair_blocks(zk, zl, rk, rl, MsindoParameterSet::embedded_hf());
}

// --------------------------------------------------------------------------- //
// Monopole gamma (coulom.f) — shell-aware.                                    //
// --------------------------------------------------------------------------- //
inline double gamma_shell(int za, int zb, int sha, int shb, double R,
                           const MsindoParameterSet& p) {
    auto na = [&](int z, int sh) {
        if (sh == 0) return MsindoParameterSet::n_principal(z);
        if (sh <= 3) return MsindoParameterSet::n_p_principal(z);
        return MsindoParameterSet::n_d_principal(z);
    };
    auto ea = [&](int z, int sh) {
        if (sh == 0) return MsindoParameterSet::get(p.MUS, z);
        if (sh <= 3) return MsindoParameterSet::get(p.MUP, z);
        return MsindoParameterSet::get(p.MUD, z);
    };
    return c2int(na(za,sha), 0, 0, ea(za,sha), na(zb,shb), 0, 0, ea(zb,shb), R);
}

// --------------------------------------------------------------------------- //
// EINZI hybrid d-block Fock terms (fockcl.f L118-210).                        //
// --------------------------------------------------------------------------- //
inline void add_einzi_dblock(Eigen::Ref<Eigen::MatrixXd> F,
                               const Eigen::Ref<const Eigen::MatrixXd>& P,
                               int lo, int z, const MsindoParameterSet& p) {
    auto h = einzi_hyb(z, p);
    double s3 = std::sqrt(3.0);
    int L=lo, L2=lo+1, L3=lo+2, L4=lo+3, L5=lo+4, L6=lo+5, L7=lo+6, L8=lo+7, L9=lo+8;
    double X1=3.0*h[1]-h[4], X2=3.0*h[3]-h[5], X3=3.0*h[2]-h[8];
    double X4=h[4]-h[1]/2.0, X5=h[5]-h[3]/2.0, X6=2.0*h[8]-h[2];
    double X7=h[13]-2.0*h[10], X8=2.0*h[11]-(h[15]+h[17])/2.0;
    double X9=2.0*h[12]-(h[14]+h[16])/2.0, X10=h[10]-3.0*h[13];
    double X11=2.0*h[14]-(h[12]+h[16])/2.0, X12=2.0*h[15]-(h[11]+h[17])/2.0;
    double X13=2.0*h[16]-(h[12]+h[14])/2.0, X14=2.0*h[17]-(h[11]+h[15])/2.0;
    double X15=3.0*h[19]-h[20], X16=h[19]/2.0-h[20];

    auto add = [&](int i, int j, double val) { F(i,j)+=val; if (i!=j) F(j,i)+=val; };

    add(L2,L,(P(L5,L2)-s3*(P(L8,L2)+P(L9,L3)+P(L6,L4)))/2.0*X1);
    add(L3,L,(P(L5,L3)-s3*(P(L9,L2)-P(L8,L3)+P(L7,L4)))/2.0*X1);
    add(L4,L,(P(L5,L4)*X2+(P(L6,L2)+P(L7,L3))*X3)/2.0);
    add(L5,L,(P(L2,L2)+P(L3,L3))*X4+P(L4,L4)*X5+(P(L5,L5)-P(L8,L8)-P(L9,L9))*h[6]/2.0+(P(L6,L6)+P(L7,L7))*h[7]/2.0);
    add(L6,L,P(L6,L5)*h[7]+P(L4,L2)*X6+(P(L8,L6)+P(L9,L7))*h[9]);
    add(L7,L,P(L4,L3)*X6+P(L7,L5)*h[7]+(P(L9,L6)-P(L8,L7))*h[9]);
    add(L8,L,(P(L2,L2)-P(L3,L3))*X6/2.0-P(L8,L5)*h[6]+(P(L6,L6)-P(L7,L7))*h[9]/2.0);
    add(L9,L,P(L3,L2)*X6-P(L9,L5)*h[6]+P(L7,L6)*h[9]);
    add(L2,L2,P(L5,L)*2.0*X4+P(L8,L)*X6-P(L8,L5)*X7);
    add(L3,L2,P(L9,L)*X6-P(L9,L5)*X7+P(L7,L6)*X8);
    add(L4,L2,P(L6,L)*X6+P(L6,L5)*X9+(P(L8,L6)+P(L9,L7))*X8);
    add(L5,L2,(P(L2,L)*X1-(P(L8,L2)+P(L9,L3))*X10)/2.0+P(L6,L4)*X11);
    add(L6,L2,P(L4,L)*X3/2.0+P(L7,L3)*X12+P(L5,L4)*X13+P(L8,L4)*X14);
    add(L7,L2,(P(L6,L3)+P(L9,L4))*X14);
    add(L8,L2,(P(L2,L)*X3-P(L5,L2)*X10+5.0*P(L9,L3)*h[18])/2.0+P(L6,L4)*X12);
    add(L9,L2,(P(L3,L)*X3-P(L5,L3)*X10-5.0*P(L8,L3)*h[18])/2.0+P(L7,L4)*X12);
    add(L3,L3,P(L5,L)*2.0*X4-P(L8,L)*X6+P(L8,L5)*X7);
    add(L4,L3,P(L7,L)*X6+(P(L9,L6)-P(L8,L7))*X8+P(L7,L5)*X9);
    add(L5,L3,(P(L3,L)*X1+(P(L8,L3)-P(L9,L2))*X10)/2.0+P(L7,L4)*X11);
    add(L6,L3,(P(L7,L2)+P(L9,L4))*X14);
    add(L7,L3,P(L4,L)*X3/2.0+P(L6,L2)*X12+P(L5,L4)*X13-P(L8,L4)*X14);
    add(L8,L3,(P(L5,L3)*X10-P(L3,L)*X3-5.0*P(L9,L2)*h[18])/2.0-P(L7,L4)*X12);
    add(L9,L3,(P(L2,L)*X3-P(L5,L2)*X10+5.0*P(L8,L2)*h[18])/2.0+P(L6,L4)*X12);
    add(L4,L4,2.0*P(L5,L)*X5);
    add(L5,L4,P(L4,L)*X2/2.0+(P(L6,L2)+P(L7,L3))*X13);
    add(L6,L4,P(L2,L)*X3/2.0+P(L5,L2)*X11+(P(L8,L2)+P(L9,L3))*X12);
    add(L7,L4,P(L3,L)*X3/2.0+(P(L9,L2)-P(L8,L3))*X12+P(L5,L3)*X11);
    add(L8,L4,(P(L6,L2)-P(L7,L3))*X14);
    add(L9,L4,(P(L7,L2)+P(L6,L3))*X14);
    add(L5,L5,P(L5,L)*h[6]);
    add(L6,L5,P(L6,L)*h[7]+P(L4,L2)*X9+(P(L8,L6)+P(L9,L7))*X15/2.0);
    add(L7,L5,P(L7,L)*h[7]+P(L4,L3)*X9+(P(L9,L6)-P(L8,L7))*X15/2.0);
    add(L8,L5,-P(L8,L)*h[6]+(P(L3,L3)-P(L2,L2))*X7/2.0+(P(L7,L7)-P(L6,L6))*X16);
    add(L9,L5,-P(L9,L)*h[6]-P(L3,L2)*X7-2.0*P(L7,L6)*X16);
    add(L6,L6,P(L5,L)*h[7]+P(L8,L)*h[9]-2.0*P(L8,L5)*X16);
    add(L7,L6,P(L9,L)*h[9]+P(L3,L2)*X8-2.0*P(L9,L5)*X16);
    add(L8,L6,P(L6,L)*h[9]+P(L4,L2)*X8+(P(L6,L5)*X15+5.0*P(L9,L7)*h[21])/2.0);
    add(L9,L6,P(L7,L)*h[9]+P(L4,L3)*X8+(P(L7,L5)*X15-5.0*P(L8,L7)*h[21])/2.0);
    add(L7,L7,P(L5,L)*h[7]-P(L8,L)*h[9]+2.0*P(L8,L5)*X16);
    add(L8,L7,-P(L7,L)*h[9]-P(L4,L3)*X8-(P(L7,L5)*X15+5.0*P(L9,L6)*h[21])/2.0);
    add(L9,L7,P(L6,L)*h[9]+P(L4,L2)*X8+(P(L6,L5)*X15+5.0*P(L8,L6)*h[21])/2.0);
    add(L8,L8,-P(L5,L)*h[6]);
    add(L9,L9,-P(L5,L)*h[6]);
}

// --------------------------------------------------------------------------- //
// EINZI hybrid d-block Fock terms, open-shell (fockop.f L124-367) — the UHF    //
// analog of add_einzi_dblock (fockcl.f).  The closed-shell X-combinations fold //
// Coulomb and exchange together (only valid when PA==PB), so the open-shell d  //
// Fock cannot be obtained per spin.  fockop.f keeps them explicit: Coulomb     //
// (GEM) and the opposite-spin cross-terms use the *total* density Pt=PA+PB;    //
// exchange uses the *same-spin* density, with fockop's own X1..X11 (from the   //
// raw HYB) and *doubled* HYB.  Reduces to add_einzi_dblock when PA==PB==P/2     //
// except for a few off-axis d-d couplings where fockop.f and fockcl.f genuinely//
// differ (the MSINDO oracle shows the same: e.g. AlCl3 RHF != UHF(M=1) by      //
// ~2.7e-5, while linear/symmetric d molecules reduce exactly).  Lockstep with  //
// python msindo._add_einzi_dblock_uhf.                                         //
// --------------------------------------------------------------------------- //
inline void add_einzi_dblock_uhf(Eigen::Ref<Eigen::MatrixXd> FA,
                                 Eigen::Ref<Eigen::MatrixXd> FB,
                                 const Eigen::Ref<const Eigen::MatrixXd>& PA,
                                 const Eigen::Ref<const Eigen::MatrixXd>& PB,
                                 int lo, int z, const MsindoParameterSet& p) {
    auto h = einzi_hyb(z, p);  // raw EINZI hybrid integrals (1-indexed)
    // fockop.f:80-90 — coefficient combinations from the *raw* HYB.
    double X1=h[1]+h[4], X2=h[2]+h[8], X3=h[3]+h[5], X4=-(h[13]+h[10]);
    double X5=h[17]+h[15], X6=h[17]+h[11], X7=h[15]+h[11], X8=h[14]+h[12];
    double X9=h[14]+h[16], X10=h[12]+h[16], X11=h[19]+h[20];
    // fockop.f:91-93 — HYB is doubled *after* the X combinations are formed.
    std::array<double,22> hyb;
    for (int i=0;i<22;++i) hyb[i]=2.0*h[i];
    int L=lo, L2=lo+1, L3=lo+2, L4=lo+3, L5=lo+4, L6=lo+5, L7=lo+6, L8=lo+7, L9=lo+8;
    Eigen::MatrixXd Pt = PA + PB;  // total density (Coulomb / GEM); spin-independent
    auto addA=[&](int i,int j,double v){FA(i,j)+=v; if(i!=j)FA(j,i)+=v;};
    auto addB=[&](int i,int j,double v){FB(i,j)+=v; if(i!=j)FB(j,i)+=v;};
    double gem;

    // p-s and d-s one-centre couplings (fockop.f:136-185)
    gem = Pt(L5,L2)*hyb[1] + (Pt(L8,L2)+Pt(L9,L3)+Pt(L6,L4))*hyb[2];
    addA(L2,L, gem - PA(L5,L2)*X1 - (PA(L8,L2)+PA(L9,L3)+PA(L6,L4))*X2);
    addB(L2,L, gem - PB(L5,L2)*X1 - (PB(L8,L2)+PB(L9,L3)+PB(L6,L4))*X2);
    gem = (Pt(L7,L4)-Pt(L8,L3)+Pt(L9,L2))*hyb[2] + Pt(L5,L3)*hyb[1];
    addA(L3,L, gem - PA(L5,L3)*X1 - (PA(L9,L2)-PA(L8,L3)+PA(L7,L4))*X2);
    addB(L3,L, gem - PB(L5,L3)*X1 - (PB(L9,L2)-PB(L8,L3)+PB(L7,L4))*X2);
    gem = (Pt(L6,L2)+Pt(L7,L3))*hyb[2] + Pt(L5,L4)*hyb[3];
    addA(L4,L, gem - (PA(L6,L2)+PA(L7,L3))*X2 - PA(L5,L4)*X3);
    addB(L4,L, gem - (PB(L6,L2)+PB(L7,L3))*X2 - PB(L5,L4)*X3);
    gem = ((Pt(L2,L2)+Pt(L3,L3))*hyb[4] + Pt(L4,L4)*hyb[5])/2.0;
    addA(L5,L, gem + ((PB(L5,L5)-PB(L8,L8)-PB(L9,L9))*hyb[6]+(PB(L6,L6)+PB(L7,L7))*hyb[7]
                      -(PA(L2,L2)+PA(L3,L3))*hyb[1]-PA(L4,L4)*hyb[3])/2.0);
    addB(L5,L, gem + ((PA(L5,L5)-PA(L8,L8)-PA(L9,L9))*hyb[6]+(PA(L6,L6)+PA(L7,L7))*hyb[7]
                      -(PB(L2,L2)+PB(L3,L3))*hyb[1]-PB(L4,L4)*hyb[3])/2.0);
    gem = Pt(L4,L2)*hyb[8];
    addA(L6,L, gem + PB(L6,L5)*hyb[7] + (PB(L8,L6)+PB(L9,L7))*hyb[9] - PA(L4,L2)*hyb[2]);
    addB(L6,L, gem + PA(L6,L5)*hyb[7] + (PA(L8,L6)+PA(L9,L7))*hyb[9] - PB(L4,L2)*hyb[2]);
    gem = Pt(L4,L3)*hyb[8];
    addA(L7,L, gem + PB(L7,L5)*hyb[7] + (PB(L9,L6)-PB(L8,L7))*hyb[9] - PA(L4,L3)*hyb[2]);
    addB(L7,L, gem + PA(L7,L5)*hyb[7] + (PA(L9,L6)-PA(L8,L7))*hyb[9] - PB(L4,L3)*hyb[2]);
    gem = (Pt(L2,L2)-Pt(L3,L3))*hyb[8]/2.0;
    addA(L8,L, gem - PB(L8,L5)*hyb[6] - ((PA(L2,L2)-PA(L3,L3))*hyb[2]-(PB(L6,L6)-PB(L7,L7))*hyb[9])/2.0);
    addB(L8,L, gem - PA(L8,L5)*hyb[6] - ((PB(L2,L2)-PB(L3,L3))*hyb[2]-(PA(L6,L6)-PA(L7,L7))*hyb[9])/2.0);
    gem = Pt(L3,L2)*hyb[8];
    addA(L9,L, gem - PB(L9,L5)*hyb[6] + PB(L7,L6)*hyb[9] - PA(L3,L2)*hyb[2]);
    addB(L9,L, gem - PA(L9,L5)*hyb[6] + PA(L7,L6)*hyb[9] - PB(L3,L2)*hyb[2]);

    // p-p and d-p one-centre couplings (fockop.f:186-292)
    gem = Pt(L5,L)*hyb[4] + Pt(L8,L)*hyb[8] + Pt(L8,L5)*hyb[10];
    addA(L2,L2, gem - PA(L8,L5)*hyb[13] - PA(L5,L)*hyb[1] - PA(L8,L)*hyb[2]);
    addB(L2,L2, gem - PB(L8,L5)*hyb[13] - PB(L5,L)*hyb[1] - PB(L8,L)*hyb[2]);
    gem = Pt(L9,L)*hyb[8] + Pt(L7,L6)*hyb[11] + Pt(L9,L5)*hyb[10];
    addA(L3,L2, gem - PA(L9,L5)*hyb[13] - PA(L7,L6)*X5 - PA(L9,L)*hyb[2]);
    addB(L3,L2, gem - PB(L9,L5)*hyb[13] - PB(L7,L6)*X5 - PB(L9,L)*hyb[2]);
    gem = Pt(L6,L)*hyb[8] + (Pt(L8,L6)+Pt(L9,L7))*hyb[11] + Pt(L6,L5)*hyb[12];
    addA(L4,L2, gem - PA(L6,L)*hyb[2] - (PA(L8,L6)+PA(L9,L7))*X5 - PA(L6,L5)*X9);
    addB(L4,L2, gem - PB(L6,L)*hyb[2] - (PB(L8,L6)+PB(L9,L7))*X5 - PB(L6,L5)*X9);
    gem = Pt(L2,L)*hyb[1] + (Pt(L8,L2)+Pt(L9,L3))*hyb[13] + Pt(L6,L4)*hyb[14];
    addA(L5,L2, gem - PA(L2,L)*X1 + (PA(L8,L2)+PA(L9,L3))*X4 - PA(L6,L4)*X10);
    addB(L5,L2, gem - PB(L2,L)*X1 + (PB(L8,L2)+PB(L9,L3))*X4 - PB(L6,L4)*X10);
    gem = Pt(L4,L)*hyb[2] + Pt(L8,L4)*hyb[17] + Pt(L7,L3)*hyb[15] + Pt(L5,L4)*hyb[16];
    addA(L6,L2, gem - PA(L4,L)*X2 - PA(L7,L3)*X6 - PA(L8,L4)*X7 - PA(L5,L4)*X8);
    addB(L6,L2, gem - PB(L4,L)*X2 - PB(L7,L3)*X6 - PB(L8,L4)*X7 - PB(L5,L4)*X8);
    gem = (Pt(L6,L3)+Pt(L9,L4))*hyb[17];
    addA(L7,L2, gem - (PA(L6,L3)+PA(L9,L4))*X7);
    addB(L7,L2, gem - (PB(L6,L3)+PB(L9,L4))*X7);
    gem = Pt(L2,L)*hyb[2] + Pt(L6,L4)*hyb[15] + Pt(L9,L3)*hyb[18] + Pt(L5,L2)*hyb[13];
    addA(L8,L2, gem - PA(L2,L)*X2 - PA(L6,L4)*X6 + PA(L9,L3)*hyb[18]/2.0 + PA(L5,L2)*X4);
    addB(L8,L2, gem - PB(L2,L)*X2 - PB(L6,L4)*X6 + PB(L9,L3)*hyb[18]/2.0 + PB(L5,L2)*X4);
    gem = Pt(L3,L)*hyb[2] + Pt(L5,L3)*hyb[13] + Pt(L7,L4)*hyb[15] - Pt(L8,L3)*hyb[18];
    addA(L9,L2, gem - PA(L3,L)*X2 + PA(L5,L3)*X4 - PA(L7,L4)*X6 - PA(L8,L3)*hyb[18]/2.0);
    addB(L9,L2, gem - PB(L3,L)*X2 + PB(L5,L3)*X4 - PB(L7,L4)*X6 - PB(L8,L3)*hyb[18]/2.0);
    gem = Pt(L5,L)*hyb[4] - Pt(L8,L)*hyb[8] - Pt(L8,L5)*hyb[10];
    addA(L3,L3, gem + PA(L8,L5)*hyb[13] - PA(L5,L)*hyb[1] + PA(L8,L)*hyb[2]);
    addB(L3,L3, gem + PB(L8,L5)*hyb[13] - PB(L5,L)*hyb[1] + PB(L8,L)*hyb[2]);
    gem = Pt(L7,L)*hyb[8] + (Pt(L9,L6)-Pt(L8,L7))*hyb[11] + Pt(L7,L5)*hyb[12];
    addA(L4,L3, gem - PA(L7,L5)*X9 - (PA(L9,L6)-PA(L8,L7))*X5 - PA(L7,L)*hyb[2]);
    addB(L4,L3, gem - PB(L7,L5)*X9 - (PB(L9,L6)-PB(L8,L7))*X5 - PB(L7,L)*hyb[2]);
    gem = Pt(L3,L)*hyb[1] - (Pt(L8,L3)-Pt(L9,L2))*hyb[13] + Pt(L7,L4)*hyb[14];
    addA(L5,L3, gem - PA(L3,L)*X1 - (PA(L8,L3)-PA(L9,L2))*X4 - PA(L7,L4)*X10);
    addB(L5,L3, gem - PB(L3,L)*X1 - (PB(L8,L3)-PB(L9,L2))*X4 - PB(L7,L4)*X10);
    gem = (Pt(L7,L2)+Pt(L9,L4))*hyb[17];
    addA(L6,L3, gem - (PA(L7,L2)+PA(L9,L4))*X7);
    addB(L6,L3, gem - (PB(L7,L2)+PB(L9,L4))*X7);
    gem = Pt(L4,L)*hyb[2] + Pt(L6,L2)*hyb[15] + Pt(L5,L4)*hyb[16] - Pt(L8,L4)*hyb[17];
    addA(L7,L3, gem - PA(L4,L)*X2 - PA(L6,L2)*X6 + PA(L8,L4)*X7 - PA(L5,L4)*X8);
    addB(L7,L3, gem - PB(L4,L)*X2 - PB(L6,L2)*X6 + PB(L8,L4)*X7 - PB(L5,L4)*X8);
    gem = -Pt(L3,L)*hyb[2] - Pt(L9,L2)*hyb[18] - Pt(L7,L4)*hyb[15] - Pt(L5,L3)*hyb[13];
    addA(L8,L3, gem + PA(L3,L)*X2 + PA(L7,L4)*X6 - PA(L5,L3)*X4 - PA(L9,L2)*hyb[18]/2.0);
    addB(L8,L3, gem + PB(L3,L)*X2 + PB(L7,L4)*X6 - PB(L5,L3)*X4 - PB(L9,L2)*hyb[18]/2.0);
    gem = Pt(L2,L)*hyb[2] + Pt(L5,L2)*hyb[13] + Pt(L8,L2)*hyb[18] + Pt(L6,L4)*hyb[15];
    addA(L9,L3, gem - PA(L2,L)*X2 + PA(L5,L2)*X4 - PA(L6,L4)*X6 + PA(L8,L2)*hyb[18]/2.0);
    addB(L9,L3, gem - PB(L2,L)*X2 + PB(L5,L2)*X4 - PB(L6,L4)*X6 + PB(L8,L2)*hyb[18]/2.0);
    gem = Pt(L5,L)*hyb[5];
    addA(L4,L4, gem - PA(L5,L)*hyb[3]);
    addB(L4,L4, gem - PB(L5,L)*hyb[3]);
    gem = Pt(L4,L)*hyb[3] + (Pt(L6,L2)+Pt(L7,L3))*hyb[16];
    addA(L5,L4, gem - PA(L4,L)*X3 - (PA(L6,L2)+PA(L7,L3))*X8);
    addB(L5,L4, gem - PB(L4,L)*X3 - (PB(L6,L2)+PB(L7,L3))*X8);
    gem = Pt(L2,L)*hyb[2] + (Pt(L8,L2)+Pt(L9,L3))*hyb[15] + Pt(L5,L2)*hyb[14];
    addA(L6,L4, gem - PA(L2,L)*X2 - PA(L5,L2)*X10 - (PA(L8,L2)+PA(L9,L3))*X6);
    addB(L6,L4, gem - PB(L2,L)*X2 - PB(L5,L2)*X10 - (PB(L8,L2)+PB(L9,L3))*X6);
    gem = Pt(L3,L)*hyb[2] + (Pt(L9,L2)-Pt(L8,L3))*hyb[15] + Pt(L5,L3)*hyb[14];
    addA(L7,L4, gem - PA(L3,L)*X2 - PA(L5,L3)*X10 - (PA(L9,L2)-PA(L8,L3))*X6);
    addB(L7,L4, gem - PB(L3,L)*X2 - PB(L5,L3)*X10 - (PB(L9,L2)-PB(L8,L3))*X6);
    gem = (Pt(L6,L2)-Pt(L7,L3))*hyb[17];
    addA(L8,L4, gem - (PA(L6,L2)-PA(L7,L3))*X7);
    addB(L8,L4, gem - (PB(L6,L2)-PB(L7,L3))*X7);
    gem = (Pt(L7,L2)+Pt(L6,L3))*hyb[17];
    addA(L9,L4, gem - (PA(L7,L2)+PA(L6,L3))*X7);
    addB(L9,L4, gem - (PB(L7,L2)+PB(L6,L3))*X7);

    // d-d one-centre couplings (fockop.f:293-367)
    addA(L5,L5, PB(L5,L)*hyb[6]);
    addB(L5,L5, PA(L5,L)*hyb[6]);
    gem = Pt(L4,L2)*hyb[12] + (Pt(L8,L6)+Pt(L9,L7))*hyb[19];
    addA(L6,L5, gem + PB(L6,L)*hyb[7] - PA(L4,L2)*X9 - (PA(L8,L6)+PA(L9,L7))*X11);
    addB(L6,L5, gem + PA(L6,L)*hyb[7] - PB(L4,L2)*X9 - (PB(L8,L6)+PB(L9,L7))*X11);
    gem = Pt(L4,L3)*hyb[12] + (Pt(L9,L6)-Pt(L8,L7))*hyb[19];
    addA(L7,L5, gem - PA(L4,L3)*X9 + (PA(L8,L7)-PA(L9,L6))*X11 + PB(L7,L)*hyb[7]);
    addB(L7,L5, gem - PB(L4,L3)*X9 + (PB(L8,L7)-PB(L9,L6))*X11 + PA(L7,L)*hyb[7]);
    gem = -((Pt(L3,L3)-Pt(L2,L2))*hyb[10]+(Pt(L7,L7)-Pt(L6,L6))*hyb[20])/2.0;
    addA(L8,L5, gem - ((PA(L6,L6)-PA(L7,L7))*hyb[19]+(PA(L2,L2)-PA(L3,L3))*hyb[13])/2.0 - PB(L8,L)*hyb[6]);
    addB(L8,L5, gem - ((PB(L6,L6)-PB(L7,L7))*hyb[19]+(PB(L2,L2)-PB(L3,L3))*hyb[13])/2.0 - PA(L8,L)*hyb[6]);
    gem = Pt(L3,L2)*hyb[10] + Pt(L7,L6)*hyb[20];
    addA(L9,L5, gem - PB(L9,L)*hyb[6] - PA(L3,L2)*hyb[13] - PA(L7,L6)*hyb[19]);
    addB(L9,L5, gem - PA(L9,L)*hyb[6] - PB(L3,L2)*hyb[13] - PB(L7,L6)*hyb[19]);
    gem = Pt(L8,L5)*hyb[20];
    addA(L6,L6, gem + PB(L5,L)*hyb[7] + PB(L8,L)*hyb[9] - PA(L8,L5)*hyb[19]);
    addB(L6,L6, gem + PA(L5,L)*hyb[7] + PA(L8,L)*hyb[9] - PB(L8,L5)*hyb[19]);
    gem = Pt(L9,L5)*hyb[20] + Pt(L3,L2)*hyb[11];
    addA(L7,L6, gem - PA(L3,L2)*X5 + PB(L9,L)*hyb[9] - PA(L9,L5)*hyb[19]);
    addB(L7,L6, gem - PB(L3,L2)*X5 + PA(L9,L)*hyb[9] - PB(L9,L5)*hyb[19]);
    gem = Pt(L9,L7)*hyb[21] + Pt(L4,L2)*hyb[11] + Pt(L6,L5)*hyb[19];
    addA(L8,L6, gem + PB(L6,L)*hyb[9] - PA(L6,L5)*X11 - PA(L4,L2)*X5 + PA(L9,L7)*hyb[21]/2.0);
    addB(L8,L6, gem + PA(L6,L)*hyb[9] - PB(L6,L5)*X11 - PB(L4,L2)*X5 + PB(L9,L7)*hyb[21]/2.0);
    gem = -Pt(L8,L7)*hyb[21] + Pt(L4,L3)*hyb[11] + Pt(L7,L5)*hyb[19];
    addA(L9,L6, gem + PB(L7,L)*hyb[9] - PA(L4,L3)*X5 - PA(L7,L5)*X11 - PA(L8,L7)*hyb[21]/2.0);
    addB(L9,L6, gem + PA(L7,L)*hyb[9] - PB(L4,L3)*X5 - PB(L7,L5)*X11 - PB(L8,L7)*hyb[21]/2.0);
    gem = -Pt(L8,L5)*hyb[20];
    addA(L7,L7, gem + PB(L5,L)*hyb[7] - PB(L8,L)*hyb[9] + PA(L8,L5)*hyb[19]);
    addB(L7,L7, gem + PA(L5,L)*hyb[7] - PA(L8,L)*hyb[9] + PB(L8,L5)*hyb[19]);
    gem = -Pt(L4,L3)*hyb[11] - Pt(L9,L6)*hyb[21] - Pt(L7,L5)*hyb[19];
    addA(L8,L7, gem - PB(L7,L)*hyb[9] + PA(L7,L5)*X11 + PA(L4,L3)*X5 - PA(L9,L6)*hyb[21]/2.0);
    addB(L8,L7, gem - PA(L7,L)*hyb[9] + PB(L7,L5)*X11 + PB(L4,L3)*X5 - PB(L9,L6)*hyb[21]/2.0);
    gem = Pt(L4,L2)*hyb[11] + Pt(L8,L6)*hyb[21] + Pt(L6,L5)*hyb[19];
    addA(L9,L7, gem + PA(L4,L2)*X5 - PA(L6,L5)*X11 + PA(L8,L6)*hyb[21]/2.0 + PB(L6,L)*hyb[9]);
    addB(L9,L7, gem + PB(L4,L2)*X5 - PB(L6,L5)*X11 + PB(L8,L6)*hyb[21]/2.0 + PA(L6,L)*hyb[9]);
    addA(L8,L8, -PB(L5,L)*hyb[6]);
    addB(L8,L8, -PA(L5,L)*hyb[6]);
    addA(L9,L9, -PB(L5,L)*hyb[6]);
    addB(L9,L9, -PA(L5,L)*hyb[6]);
}

// --------------------------------------------------------------------------- //
// Fock builders                                                                //
// --------------------------------------------------------------------------- //

// Closed-shell Fock (fockcl.f): core + two-center + one-center INDO + EINZI d.
inline Eigen::MatrixXd build_fock(const Eigen::MatrixXd& H, const Eigen::MatrixXd& G,
                                   const Eigen::MatrixXd& P,
                                   const std::vector<int>& uat,
                                   const std::vector<std::array<int, 2>>& blocks,
                                   const std::vector<int>& Z,
                                   const MsindoParameterSet& p) {
    int nsto = H.rows();
    Eigen::MatrixXd F = H;
    for (int i=0; i<nsto; ++i)
        for (int j=0; j<nsto; ++j)
            if (uat[i]!=uat[j]) {
                F(i,i) += P(j,j)*G(i,j);
                F(j,i) += -0.5*P(j,i)*G(i,j);
            }
    for (size_t ia=0; ia<blocks.size(); ++ia) {
        int lo=blocks[ia][0], hi=blocks[ia][1];
        for (int j=lo; j<hi; ++j) {
            for (int k=lo; k<hi; ++k) {
                int jmin=std::min(j,k), jmax=std::max(j,k);
                F(j,j) += P(k,k)*(G(jmin,jmax) - 0.5*G(jmax,jmin));
            }
            for (int k=j+1; k<hi; ++k) {
                double val = 0.5*P(k,j)*(3.0*G(k,j) - G(j,k));
                F(k,j) += val; F(j,k) += val;
            }
        }
        if (hi-lo >= 9) add_einzi_dblock(F, P, lo, Z[ia], p);
    }
    return F;
}

// Backward-compatible (embedded H-F set, blocks only)
inline Eigen::MatrixXd build_fock(const Eigen::MatrixXd& H, const Eigen::MatrixXd& G,
                                   const Eigen::MatrixXd& P,
                                   const std::vector<int>& uat,
                                   const std::vector<std::array<int, 2>>& blocks) {
    // Use empty Z since d-block won't fire for H-F anyway
    std::vector<int> dummyZ;
    return build_fock(H, G, P, uat, blocks, dummyZ, MsindoParameterSet::embedded_hf());
}

// UHF Fock (fockop.f), s/p/d elements.
inline void build_fock_uhf(const Eigen::MatrixXd& H, const Eigen::MatrixXd& G,
                            const Eigen::MatrixXd& PA, const Eigen::MatrixXd& PB,
                            const std::vector<std::array<int, 2>>& blocks,
                            const std::vector<int>& Z, const MsindoParameterSet& p,
                            Eigen::MatrixXd& FA, Eigen::MatrixXd& FB) {
    FA = H; FB = H;
    Eigen::MatrixXd Pt = PA + PB;
    int nsto = H.rows();
    std::vector<int> uat(nsto);
    for (size_t i=0; i<blocks.size(); ++i)
        for (int o=blocks[i][0]; o<blocks[i][1]; ++o) uat[o] = static_cast<int>(i);

    for (int i=0; i<nsto; ++i)
        for (int j=0; j<nsto; ++j)
            if (uat[i]!=uat[j]) {
                FA(i,i) += Pt(j,j)*G(i,j); FB(i,i) += Pt(j,j)*G(i,j);
                FA(j,i) += -PA(j,i)*G(i,j); FB(j,i) += -PB(j,i)*G(i,j);
            }
    for (size_t ia=0; ia<blocks.size(); ++ia) {
        int lo=blocks[ia][0], hi=blocks[ia][1];
        int nb = hi - lo;
        for (int j=lo; j<hi; ++j) {
            for (int k=lo; k<hi; ++k) {
                int jmin=std::min(j,k), jmax=std::max(j,k);
                FA(j,j) += Pt(k,k)*G(jmin,jmax) - PA(k,k)*G(jmax,jmin);
                FB(j,j) += Pt(k,k)*G(jmin,jmax) - PB(k,k)*G(jmax,jmin);
            }
            for (int k=j+1; k<hi; ++k) {
                double fa = (2.0*Pt(k,j)-PA(k,j))*G(k,j) - PA(k,j)*G(j,k);
                double fb = (2.0*Pt(k,j)-PB(k,j))*G(k,j) - PB(k,j)*G(j,k);
                FA(k,j)+=fa; FA(j,k)+=fa; FB(k,j)+=fb; FB(j,k)+=fb;
            }
        }
        if (nb >= 9) {
            // One-centre d Fock: fockop.f keeps Coulomb (total density) and
            // exchange (same-spin) explicit — the closed-shell EINZI per spin is
            // wrong for d (see add_einzi_dblock_uhf).
            add_einzi_dblock_uhf(FA, FB, PA, PB, lo, Z[ia], p);
        }
    }
}

// --------------------------------------------------------------------------- //
// NDDO integrals + Fock (nddofockcl.f + spspfockcl.f)                         //
// --------------------------------------------------------------------------- //

// s-p charge separation DA (einzentren.f).
inline double nddo_da(int z, const MsindoParameterSet& p) {
    double zp = MsindoParameterSet::get(p.MUP, z);
    if (zp == 0.0) return 0.0;
    double zs = MsindoParameterSet::get(p.MUS, z);
    double c = MsindoParameterSet::n_principal(z) + 0.5;
    return c*std::pow(zs*zp, c) / (std::sqrt(3.0)*std::pow((zs+zp)/2.0, 2*c+1));
}

inline double nddo_asp(int z, const MsindoParameterSet& p) {
    return slater_condon(z, p).G1SP / 3.0;
}

inline double nddo_rho_self(int z, const MsindoParameterSet& p) {
    return 0.5*std::pow(nddo_da(z,p)*nddo_da(z,p)/nddo_asp(z,p), 1.0/3.0);
}

inline double gss_val(int z, const MsindoParameterSet& p) {
    return one_center_2e(z, p).GSS;
}

inline double nddo_spss_si(int zi, int zj, double r, const MsindoParameterSet& p) {
    double da = nddo_da(zi, p);
    double rho = nddo_rho_self(zi, p) + 0.5/gss_val(zj, p);
    return 1.0/std::sqrt((r-da/2.0)*(r-da/2.0)+rho*rho)
         - 1.0/std::sqrt((r+da/2.0)*(r+da/2.0)+rho*rho);
}

inline double nddo_sppp_si(int zi, int zj, double r, const MsindoParameterSet& p) {
    double da = nddo_da(zi, p);
    double rho = nddo_rho_self(zi, p) + 0.5/one_center_2e(zj, p).GPP;
    return 1.0/std::sqrt((r-da/2.0)*(r-da/2.0)+rho*rho)
         - 1.0/std::sqrt((r+da/2.0)*(r+da/2.0)+rho*rho);
}

inline double nddo_gdd(int z, const MsindoParameterSet& p) {
    const SlaterCondon f = slater_condon(z, p);
    return f.F0DD + 4.0/49.0*f.F2DD + 36.0/441.0*f.F4DD;
}

// SPDD_SI is asymmetric: zi supplies the s-p dipole (DA/ASP), while zj
// supplies the spherical d-shell monopole size through GDD (spdd_si.f:26-29).
inline double nddo_spdd_si(int zi, int zj, double r, const MsindoParameterSet& p) {
    double da = nddo_da(zi, p);
    double rho = nddo_rho_self(zi, p) + 0.5/nddo_gdd(zj, p);
    return 1.0/std::sqrt((r-da/2.0)*(r-da/2.0)+rho*rho)
         - 1.0/std::sqrt((r+da/2.0)*(r+da/2.0)+rho*rho);
}

inline double nddo_spsp_si(int zi, int zj, double r, const MsindoParameterSet& p) {
    double di=nddo_da(zi,p), dj=nddo_da(zj,p);
    double rho=nddo_rho_self(zi,p)+nddo_rho_self(zj,p);
    return 0.25/std::sqrt((r+di-dj)*(r+di-dj)+rho*rho)
         - 0.25/std::sqrt((r+di+dj)*(r+di+dj)+rho*rho)
         - 0.25/std::sqrt((r-di-dj)*(r-di-dj)+rho*rho)
         + 0.25/std::sqrt((r-di+dj)*(r-di+dj)+rho*rho);
}

inline double nddo_spsp_pi(int zi, int zj, double r, const MsindoParameterSet& p) {
    double di=nddo_da(zi,p), dj=nddo_da(zj,p);
    double rho=nddo_rho_self(zi,p)+nddo_rho_self(zj,p);
    return 0.5/std::sqrt(r*r+(di-dj)*(di-dj)+rho*rho)
         - 0.5/std::sqrt(r*r+(di+dj)*(di+dj)+rho*rho);
}

// Build the NDDO two-centre multipole Fock addition as a callback-compatible
// function: f(P) -> (F_add, e_add).  Mirrors nddofockcl.f + spspfockcl.f.
// The HSP core term is added during core-and-gamma assembly (see below).
inline std::function<std::pair<Eigen::MatrixXd, double>(const Eigen::MatrixXd&)>
nddo_fock_extra_builder(const std::vector<int>& Z,
                         const std::vector<Eigen::Vector3d>& C,
                         const std::vector<std::array<int, 2>>& blocks,
                         const MsindoParameterSet& p) {
    int natom = static_cast<int>(Z.size());
    return [natom, Z, C, blocks, &p](const Eigen::MatrixXd& P)
           -> std::pair<Eigen::MatrixXd, double> {
        int nsto = static_cast<int>(P.rows());
        Eigen::MatrixXd F = Eigen::MatrixXd::Zero(nsto, nsto);
        for (int I=0; I<natom; ++I) {
            int loI=blocks[I][0], hiI=blocks[I][1];
            int zI=Z[I]; bool pI=(hiI-loI)>=4, dI=(hiI-loI)>=9;
            for (int J=0; J<natom; ++J) {
                if (I==J) continue;
                int loJ=blocks[J][0], hiJ=blocks[J][1];
                int zJ=Z[J]; bool pJ=(hiJ-loJ)>=4, dJ=(hiJ-loJ)>=9;
                Eigen::Vector3d d=C[J]-C[I];
                double R=d.norm();
                Eigen::Vector3d E=d/R;
                double SSSP=nddo_spss_si(zJ,zI,R,p), SPSS=nddo_spss_si(zI,zJ,R,p);
                double SPPP=pJ?nddo_sppp_si(zI,zJ,R,p):0.0, PPSP=pI?nddo_sppp_si(zJ,zI,R,p):0.0;
                double DDSP=(pJ&&dI)?nddo_spdd_si(zJ,zI,R,p):0.0;
                double SPDD=(pI&&dJ)?nddo_spdd_si(zI,zJ,R,p):0.0;
                double pdipJ=0.0;
                if (pJ) for (int a=0;a<3;++a) pdipJ+=P(loJ+1+a,loJ)*E[a];
                // diagonal Coulomb
                if (pJ) {
                    F(loI,loI)-=2.0*SSSP*pdipJ;
                    if (pI) for (int a=0;a<3;++a)
                        F(loI+1+a,loI+1+a)-=2.0*PPSP*pdipJ;
                    // nddofockcl.f:61-69: d(I)-diagonal from the p(J)-s(J)
                    // transition density; DDSP ordering is SPDD_SI(J,I).
                    if (dI) for (int mu=loI+4;mu<hiI;++mu)
                        F(mu,mu)-=2.0*DDSP*pdipJ;
                }
                // one-centre (s,pσ)
                if (pI) {
                    double PssJ=P(loJ,loJ), PppJ=pJ?(P(loJ+1,loJ+1)+P(loJ+2,loJ+2)+P(loJ+3,loJ+3)):0.0;
                    for (int a=0;a<3;++a) F(loI+1+a,loI)+=E[a]*(SPSS*PssJ+SPPP*PppJ);
                    // nddofockcl.f:89-96: p(I)-s(I) from the spherical d(J)
                    // population; this is Coulomb-only (no SPDD exchange).
                    if (dJ) {
                        double PddJ=0.0;
                        for (int mu=loJ+4;mu<hiJ;++mu) PddJ+=P(mu,mu);
                        for (int a=0;a<3;++a)
                            F(loI+1+a,loI)+=E[a]*SPDD*PddJ;
                    }
                }
                if (pI&&pJ) {
                    double SI=nddo_spsp_si(zJ,zI,R,p), PI=nddo_spsp_pi(zJ,zI,R,p);
                    Eigen::Matrix3d S;
                    for (int a=0;a<3;++a) for (int b=0;b<3;++b)
                        S(a,b)=E[a]*E[b]*SI+((a==b?1.0:0.0)-E[a]*E[b])*PI;
                    for (int a=0;a<3;++a) {
                        double sum=0.0; for (int b=0;b<3;++b) sum+=P(loJ+1+b,loJ)*S(a,b);
                        F(loI+1+a,loI)+=2.0*sum;
                    }
                }
                // two-centre exchange (lower triangle)
                if (I>J) {
                    if (pI) { double s=0.0; for (int a=0;a<3;++a) s+=E[a]*P(loI+1+a,loJ); F(loI,loJ)-=0.5*SPSS*s; }
                    if (pJ) {
                        F(loI,loJ)+=0.5*SSSP*(E[0]*P(loI,loJ+1)+E[1]*P(loI,loJ+2)+E[2]*P(loI,loJ+3));
                        for (int b=0;b<3;++b) F(loI,loJ+1+b)+=0.5*SSSP*E[b]*P(loI,loJ);
                    }
                    if (pI&&pJ) {
                        for (int b=0;b<3;++b) { double s=0.0; for (int a=0;a<3;++a) s+=E[a]*P(loI+1+a,loJ+1+b); F(loI,loJ+1+b)-=0.5*SPPP*s; }
                    }
                    if (pI) for (int a=0;a<3;++a) F(loI+1+a,loJ)-=0.5*SPSS*E[a]*P(loI,loJ);
                    if (pI&&pJ) {
                        for (int a=0;a<3;++a) { double s=0.0; for (int b=0;b<3;++b) s+=E[b]*P(loI+1+a,loJ+1+b); F(loI+1+a,loJ)+=0.5*PPSP*s; }
                        for (int a=0;a<3;++a) for (int b=0;b<3;++b)
                            F(loI+1+a,loJ+1+b)+=-0.5*SPPP*E[a]*P(loI,loJ+1+b)+0.5*PPSP*E[b]*P(loI+1+a,loJ);
                        // spspfockcl two-centre exchange
                        double SI=nddo_spsp_si(zJ,zI,R,p), PI=nddo_spsp_pi(zJ,zI,R,p);
                        Eigen::Matrix3d S;
                        for (int a=0;a<3;++a) for (int b=0;b<3;++b) S(a,b)=E[a]*E[b]*SI+((a==b?1.0:0.0)-E[a]*E[b])*PI;
                        { double s=0.0; for (int a=0;a<3;++a) for (int b=0;b<3;++b) s+=P(loI+1+a,loJ+1+b)*S(a,b); F(loI,loJ)-=0.5*s; }
                        for (int b=0;b<3;++b) { double s=0.0; for (int a=0;a<3;++a) s+=P(loI+1+a,loJ)*S(a,b); F(loI,loJ+1+b)-=0.5*s; }
                        for (int a=0;a<3;++a) { double s=0.0; for (int b=0;b<3;++b) s+=P(loI,loJ+1+b)*S(a,b); F(loI+1+a,loJ)-=0.5*s; }
                        for (int a=0;a<3;++a) for (int b=0;b<3;++b) F(loI+1+a,loJ+1+b)-=0.5*P(loI,loJ)*S(a,b);
                    }
                }
            }
        }
        // mirror lower -> upper
        for (int i=0;i<nsto;++i) for (int j=i+1;j<nsto;++j) F(i,j)=F(j,i);
        return std::make_pair(F, 0.0);
    };
}

// --------------------------------------------------------------------------- //
// Core + gamma assembly                                                       //
// --------------------------------------------------------------------------- //
inline void build_core_and_gamma(const std::vector<int>& Z,
                                  const std::vector<Eigen::Vector3d>& C,
                                  const std::vector<std::array<int, 2>>& blocks,
                                  int nsto, const MsindoParameterSet& p,
                                  bool nddo, Eigen::MatrixXd& H, Eigen::MatrixXd& G) {
    H = Eigen::MatrixXd::Zero(nsto, nsto);
    G = Eigen::MatrixXd::Zero(nsto, nsto);
    int natom = static_cast<int>(Z.size());
    for (int i=0; i<natom; ++i) {
        int lo=blocks[i][0], hi=blocks[i][1], z=Z[i];
        auto u = eneg(z, p);
        int nb = hi-lo;
        H(lo,lo) += u[0];
        if (nb>=4) for (int pi=1; pi<=3; ++pi) H(lo+pi,lo+pi) += u[1];
        if (nb>=9) for (int d=4; d<=8; ++d) H(lo+d,lo+d) += u[2];
        G.block(lo,lo,nb,nb) = one_center_gmunu(z, p).topLeftCorner(nb, nb);
    }
    for (int a=0; a<natom; ++a)
        for (int b=a+1; b<natom; ++b) {
            int la=blocks[a][0], ha=blocks[a][1], lb=blocks[b][0], hb=blocks[b][1];
            std::array<double,3> ra{C[a].x(),C[a].y(),C[a].z()}, rb{C[b].x(),C[b].y(),C[b].z()};
            PairBlocks pb = pair_blocks(Z[a], Z[b], ra, rb, p);
            int na_=ha-la, nb_=hb-lb;
            H.block(la,la,na_,na_) += pb.HK1.topLeftCorner(na_,na_);
            H.block(lb,lb,nb_,nb_) += pb.HL1.topLeftCorner(nb_,nb_);
            H.block(la,lb,na_,nb_) += pb.HKL2.topLeftCorner(na_,nb_);
            H.block(lb,la,nb_,na_) += pb.HKL2.topLeftCorner(na_,nb_).transpose();
            double R=pb.R;
            for (int ia=0; ia<na_; ++ia)
                for (int ib=0; ib<nb_; ++ib)
                    G(la+ia,lb+ib) = G(lb+ib,la+ia) = gamma_shell(Z[a],Z[b],ia,ib,R,p);
        }

    // NDDO HSP s-pσ core coupling (v2core.f one-centre term, mixed into H).
    if (nddo) {
        for (int a=0; a<natom; ++a) {
            int lo=blocks[a][0], hi=blocks[a][1], za=Z[a];
            if (hi-lo < 4) continue;
            for (int b=0; b<natom; ++b) {
                if (a==b) continue;
                int zb=Z[b];
                Eigen::Vector3d d=C[b]-C[a];
                double R=d.norm();
                Eigen::Vector3d E=d/R;
                double hsp = -MsindoParameterSet::eff_core_charge(zb)*nddo_spss_si(za,zb,R,p);
                for (int k=0; k<3; ++k) {
                    H(lo+1+k, lo) += hsp*E[k];
                    H(lo, lo+1+k) += hsp*E[k];
                }
            }
        }
    }
}

// --------------------------------------------------------------------------- //
// Engine result struct                                                         //
// --------------------------------------------------------------------------- //
struct MsindoResult {
    double total_energy=0, electronic_energy=0, binding_energy=0;
    Eigen::VectorXd mo_energies;
    Eigen::MatrixXd density;
    int n_iter=0;
    bool converged=false;
};

// --------------------------------------------------------------------------- //
// MSINDO-faithful molecular RHF SCF (Hückel guess + WICHT damping)            //
// --------------------------------------------------------------------------- //
// Lockstep with python msindo._scf_rhf_msindo.  Used (via the dispatch in
// run_msindo_core) for the heavier d/p-block elements where DIIS jumps SCF
// basins; reproduces MSINDO's default full-diag + WICHT-damped trajectory from
// the extended-Hückel guess.  See docs/user_guide/msindo.md.

// Elements whose molecular RHF needs this path (Nb–Pd 4d, Sb–Xe 5p).  Nb/Mo were
// added (2026-06-17) so the homonuclear dimers Nb₂/Mo₂ reach the reference state
// (DIIS jumps to a spurious basin); their well-behaved halides reproduce the
// oracle on this path too.  Lockstep with python msindo._MSINDO_TRAJECTORY_SCF.
inline bool is_msindo_trajectory_scf(int z) {
    return (z >= 41 && z <= 46) || (z >= 51 && z <= 54);
}

// Proactive strict-basin comparison is validated for the light-element
// molecular campaign (H-Ar).  Other non-pinned elements retain their converged
// Hcore/DIIS result, while the established failed-primary recovery stays
// available to every non-pinned element.
inline bool is_msindo_root_probe_element(int z) {
    return z >= 1 && z <= 18;
}

// Extended-Hückel guess Hamiltonian (huckcl.f / huckop.f): core H with the
// one-centre (same-atom) off-diagonal blocks zeroed and the diagonal set to the
// bare ionization potentials Q (IPOTS/IPOTP/IPOTD per shell), two-centre
// resonance kept.  Mirrors python msindo._huckel_guess_hamiltonian.
inline Eigen::MatrixXd huckel_guess_hamiltonian(const Eigen::MatrixXd& H,
                                                const std::vector<std::array<int, 2>>& blocks,
                                                const std::vector<int>& Z,
                                                const MsindoParameterSet& p) {
    Eigen::MatrixXd A = H;
    for (size_t i = 0; i < blocks.size(); ++i) {
        int lo = blocks[i][0], hi = blocks[i][1], nb = hi - lo, z = Z[i];
        A.block(lo, lo, nb, nb).setZero();  // zero one-centre (same-atom) block
        A(lo, lo) = MsindoParameterSet::get(p.IPOTS, z);
        if (nb >= 4)
            for (int q = 1; q < 4; ++q) A(lo + q, lo + q) = MsindoParameterSet::get(p.IPOTP, z);
        if (nb >= 9)
            for (int d = 4; d < 9; ++d) A(lo + d, lo + d) = MsindoParameterSet::get(p.IPOTD, z);
    }
    return A;
}

// Closed-shell extended-Hückel start density (huckcl.f): aufbau-fill the lowest
// nocc eigenvectors, doubly occupied.  Mirrors python msindo._huckel_guess_density.
inline Eigen::MatrixXd huckel_guess_density(const Eigen::MatrixXd& H,
                                            const std::vector<std::array<int, 2>>& blocks,
                                            const std::vector<int>& Z,
                                            const MsindoParameterSet& p, int nocc) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(huckel_guess_hamiltonian(H, blocks, Z, p));
    Eigen::MatrixXd Cmo = es.eigenvectors();
    return 2.0 * Cmo.leftCols(nocc) * Cmo.leftCols(nocc).transpose();
}

// Open-shell (UHF) extended-Hückel start densities (huckop.f): PA from the first
// nalpha and PB from the first nbeta columns of the *same* coefficient matrix
// (huckop.f sets CB=CA at the guess), each singly occupied.  With nalpha==nbeta
// returns PA==PB==P/2 of the closed-shell guess.  Mirrors python
// msindo._huckel_guess_density_uhf.
inline void huckel_guess_density_uhf(const Eigen::MatrixXd& H,
                                     const std::vector<std::array<int, 2>>& blocks,
                                     const std::vector<int>& Z,
                                     const MsindoParameterSet& p, int nalpha, int nbeta,
                                     Eigen::MatrixXd& PA, Eigen::MatrixXd& PB) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(huckel_guess_hamiltonian(H, blocks, Z, p));
    Eigen::MatrixXd Cmo = es.eigenvectors();
    PA = Cmo.leftCols(nalpha) * Cmo.leftCols(nalpha).transpose();
    if (nbeta > 0)
        PB = Cmo.leftCols(nbeta) * Cmo.leftCols(nbeta).transpose();
    else
        PB = Eigen::MatrixXd::Zero(H.rows(), H.cols());
}

// WICHT-damped full-diag RHF with energy-only convergence (DELEN=1e-8, the
// EPSI<=0 rule of eneclo.f:103).  DAMP starts at 3.0 (scfclo.f NAV=4), resets to
// 1.0 on energy rise, decays ×0.8.  Mirrors python msindo._scf_rhf_msindo.
inline MsindoResult scf_rhf_msindo_driver(const Eigen::MatrixXd& H, const Eigen::MatrixXd& G,
                                          const std::vector<std::array<int, 2>>& blocks,
                                          const std::vector<int>& Z, int nocc,
                                          const MsindoParameterSet& p, int max_iter) {
    int nsto = H.rows();
    std::vector<int> uat(nsto);
    for (size_t i = 0; i < blocks.size(); ++i)
        for (int o = blocks[i][0]; o < blocks[i][1]; ++o) uat[o] = static_cast<int>(i);

    const double DELEN = 1e-8;
    Eigen::MatrixXd P = huckel_guess_density(H, blocks, Z, p, nocc);
    double damp = 3.0, oldeng = 0.0, epsi = 0.0, e_elec = 0.0;
    Eigen::MatrixXd p_in_prev;
    bool converged = false;
    int it = 0;
    for (it = 1; it <= max_iter; ++it) {
        if (it > 1) {
            if (epsi > 0.0) damp = 1.0;  // wicht.f:31
            P = (P + damp * p_in_prev) / (1.0 + damp);
            damp *= 0.8;
        }
        Eigen::MatrixXd F = build_fock(H, G, P, uat, blocks, Z, p);
        e_elec = 0.5 * (P.array() * (H + F).array()).sum();
        epsi = e_elec - oldeng;
        if (it > 1 && (((std::abs(epsi) < DELEN) && (epsi <= 0.0)) ||
                       (std::abs(epsi) <= 0.1 * DELEN))) {
            converged = true;
            break;
        }
        oldeng = e_elec;
        p_in_prev = P;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(F);
        Eigen::MatrixXd Cmo = es.eigenvectors();
        P = 2.0 * Cmo.leftCols(nocc) * Cmo.leftCols(nocc).transpose();
    }
    Eigen::MatrixXd Ffin = build_fock(H, G, P, uat, blocks, Z, p);
    e_elec = 0.5 * (P.array() * (H + Ffin).array()).sum();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> esf(Ffin);
    MsindoResult res;
    res.electronic_energy = e_elec;
    res.mo_energies = esf.eigenvalues();
    res.density = P;
    res.n_iter = converged ? it : max_iter;
    res.converged = converged;
    return res;
}

// WICHT-damped full-diag UHF with energy-only convergence (eneopn.f) — the
// open-shell analog of scf_rhf_msindo_driver.  Open-shell extended-Hückel guess
// (huckop.f) + WICHT damping of *both* spin densities (wicht.f IF(UHF.OR.ROHF))
// + the energy-only stop (DELEN=1e-8).  Open-shell convergence is slower than
// closed-shell, so the cycle cap is higher.  Mirrors python msindo._scf_uhf_msindo.
inline MsindoResult scf_uhf_msindo_driver(const Eigen::MatrixXd& H, const Eigen::MatrixXd& G,
                                          const std::vector<std::array<int, 2>>& blocks,
                                          const std::vector<int>& Z, int nalpha, int nbeta,
                                          const MsindoParameterSet& p, int max_iter) {
    int nsto = H.rows();
    const double DELEN = 1e-8;
    Eigen::MatrixXd PA, PB;
    huckel_guess_density_uhf(H, blocks, Z, p, nalpha, nbeta, PA, PB);
    double damp = 3.0, oldeng = 0.0, epsi = 0.0, e_elec = 0.0;
    Eigen::MatrixXd pa_in_prev, pb_in_prev;
    bool converged = false;
    int it = 0;
    for (it = 1; it <= max_iter; ++it) {
        if (it > 1) {
            if (epsi > 0.0) damp = 1.0;  // wicht.f:31
            PA = (PA + damp * pa_in_prev) / (1.0 + damp);
            PB = (PB + damp * pb_in_prev) / (1.0 + damp);
            damp *= 0.8;
        }
        Eigen::MatrixXd FA, FB;
        build_fock_uhf(H, G, PA, PB, blocks, Z, p, FA, FB);
        e_elec = 0.5 * ((PA + PB).array() * H.array()).sum()
                 + 0.5 * (PA.array() * FA.array()).sum()
                 + 0.5 * (PB.array() * FB.array()).sum();
        epsi = e_elec - oldeng;
        if (it > 1 && (((std::abs(epsi) < DELEN) && (epsi <= 0.0)) ||
                       (std::abs(epsi) <= 0.1 * DELEN))) {
            converged = true;
            break;
        }
        oldeng = e_elec;
        pa_in_prev = PA;
        pb_in_prev = PB;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> esA(FA);
        PA = esA.eigenvectors().leftCols(nalpha) * esA.eigenvectors().leftCols(nalpha).transpose();
        if (nbeta > 0) {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> esB(FB);
            PB = esB.eigenvectors().leftCols(nbeta) * esB.eigenvectors().leftCols(nbeta).transpose();
        } else {
            PB = Eigen::MatrixXd::Zero(nsto, nsto);
        }
    }
    Eigen::MatrixXd FA, FB;
    build_fock_uhf(H, G, PA, PB, blocks, Z, p, FA, FB);
    e_elec = 0.5 * ((PA + PB).array() * H.array()).sum()
             + 0.5 * (PA.array() * FA.array()).sum()
             + 0.5 * (PB.array() * FB.array()).sum();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> esfA(FA);
    MsindoResult res;
    res.electronic_energy = e_elec;
    res.mo_energies = esfA.eigenvalues();
    res.density = PA + PB;
    res.n_iter = it;
    res.converged = converged;
    return res;
}

// --------------------------------------------------------------------------- //
// RHF SCF driver (shared)                                                     //
// --------------------------------------------------------------------------- //
inline bool solve_diis_coefficients(const Eigen::MatrixXd& B,
                                    const Eigen::VectorXd& rhs,
                                    Eigen::VectorXd& coeffs) {
    Eigen::PartialPivLU<Eigen::MatrixXd> lu(B);
    coeffs = lu.solve(rhs);
    if (!coeffs.allFinite()) return false;
    double scale = std::max(1.0, rhs.norm());
    return (B * coeffs - rhs).norm() <= 1e-8 * scale;
}

inline MsindoResult scf_rhf_driver(const Eigen::MatrixXd& H, const Eigen::MatrixXd& G,
                                    const std::vector<std::array<int, 2>>& blocks,
                                    const std::vector<int>& Z, int nocc,
                                    const MsindoParameterSet& p,
                                    int max_iter, double conv_tol,
                                    std::function<std::pair<Eigen::MatrixXd,double>(const Eigen::MatrixXd&)> fock_extra,
                                    const Eigen::MatrixXd* initial_density = nullptr) {
    int nsto = H.rows();
    std::vector<int> uat(nsto);
    for (size_t i=0; i<blocks.size(); ++i)
        for (int o=blocks[i][0]; o<blocks[i][1]; ++o) uat[o]=static_cast<int>(i);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H);
    Eigen::VectorXd eps=es.eigenvalues();
    Eigen::MatrixXd Cmo=es.eigenvectors();
    Eigen::MatrixXd P = initial_density == nullptr
        ? 2.0*Cmo.leftCols(nocc)*Cmo.leftCols(nocc).transpose()
        : *initial_density;

    double e_elec=0.0; bool converged=false; int it=0;
    std::vector<Eigen::MatrixXd> f_hist, e_hist;
    const int DIIS_MAX=8;
    f_hist.reserve(DIIS_MAX + 1);
    e_hist.reserve(DIIS_MAX + 1);
    for (it=1; it<=max_iter; ++it) {
        Eigen::MatrixXd F = build_fock(H, G, P, uat, blocks, Z, p);
        double e_add=0.0;
        if (fock_extra) { auto fe=fock_extra(P); F=F+fe.first; e_add=fe.second; }
        Eigen::MatrixXd err = F*P - P*F;
        double e_cur = 0.5*(P.array()*(H+F).array()).sum() + e_add;
        if (err.cwiseAbs().maxCoeff()<conv_tol && std::abs(e_cur-e_elec)<conv_tol) {
            e_elec=e_cur; converged=true; break;
        }
        e_elec=e_cur;
        f_hist.push_back(F); e_hist.push_back(err);
        if (static_cast<int>(f_hist.size())>DIIS_MAX) { f_hist.erase(f_hist.begin()); e_hist.erase(e_hist.begin()); }
        int n=static_cast<int>(f_hist.size());
        Eigen::MatrixXd F_eff=F;
        if (n>=2) {
            Eigen::MatrixXd B=Eigen::MatrixXd::Constant(n+1,n+1,-1.0); B(n,n)=0.0;
            for (int i=0;i<n;++i) for (int j=i;j<n;++j) {
                double v=(e_hist[i].array()*e_hist[j].array()).sum(); B(i,j)=v; B(j,i)=v;
            }
            Eigen::VectorXd rhs=Eigen::VectorXd::Zero(n+1); rhs(n)=-1.0;
            Eigen::VectorXd c(n+1);
            if (solve_diis_coefficients(B, rhs, c)) {
                F_eff=Eigen::MatrixXd::Zero(nsto,nsto);
                for (int i=0;i<n;++i) F_eff+=c(i)*f_hist[i];
            }
        }
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es2(F_eff);
        eps=es2.eigenvalues(); Cmo=es2.eigenvectors();
        P=2.0*Cmo.leftCols(nocc)*Cmo.leftCols(nocc).transpose();
    }
    // final energy at converged P
    {
        Eigen::MatrixXd Ffin=build_fock(H,G,P,uat,blocks,Z,p);
        double e_add=0.0;
        if (fock_extra) { auto fe=fock_extra(P); Ffin=Ffin+fe.first; e_add=fe.second; }
        e_elec=0.5*(P.array()*(H+Ffin).array()).sum()+e_add;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> esf(Ffin);
        eps=esf.eigenvalues();
    }
    MsindoResult res;
    res.electronic_energy=e_elec; res.mo_energies=eps; res.density=P;
    res.n_iter=converged ? it : max_iter; res.converged=converged;
    return res;
}

// Molecular INDO RHF root selection.  The strict Hcore/DIIS solve remains the
// primary trajectory.  In the validated H-Ar scope, also probe the reference
// program's extended-Huckel/WICHT trajectory and strict-refine every converged
// candidate before comparing stationary energies.  WICHT's energy-only
// convergence is never returned as a final result for those cases.  Every
// proactive probe retains the finite 2000-cycle recovery floor introduced for
// tetrazine in fa4af4b4f, so crossing the caller's strict-DIIS cap cannot change
// which basin is considered.  Converged jobs outside H-Ar retain their
// established route; failed-primary recovery remains available.  The
// trajectory-pinned heavy-element set keeps its direct WICHT energy path.
inline MsindoResult scf_rhf_molecular_driver(
        const Eigen::MatrixXd& H, const Eigen::MatrixXd& G,
        const std::vector<std::array<int, 2>>& blocks,
        const std::vector<int>& Z, int nocc,
        const MsindoParameterSet& p, int max_iter, double conv_tol) {
    for (int z : Z) {
        if (is_msindo_trajectory_scf(z)) {
            return scf_rhf_msindo_driver(
                H, G, blocks, Z, nocc, p, std::max(max_iter, 2000));
        }
    }

    MsindoResult primary = scf_rhf_driver(
        H, G, blocks, Z, nocc, p, max_iter, conv_tol, nullptr);
    const int primary_iters = primary.n_iter;
    bool proactive_probe = true;
    for (int z : Z) {
        if (!is_msindo_root_probe_element(z)) {
            proactive_probe = false;
            break;
        }
    }
    if (primary.converged && !proactive_probe) {
        return primary;
    }

    const int probe_cap = std::max(max_iter, 2000);
    const auto warm = scf_rhf_msindo_driver(
        H, G, blocks, Z, nocc, p, probe_cap);
    int total_iters = primary_iters + warm.n_iter;

    if (!warm.converged) {
        primary.n_iter = total_iters;
        return primary;
    }

    MsindoResult strict = scf_rhf_driver(
        H, G, blocks, Z, nocc, p, max_iter, conv_tol,
        nullptr, &warm.density);
    total_iters += strict.n_iter;

    // A failed primary keeps the exact recovery semantics of fa4af4b4f,
    // including returning the strict attempt when it also exhausts its cap.
    if (!primary.converged ||
        (strict.converged &&
         strict.electronic_energy < primary.electronic_energy - conv_tol)) {
        strict.n_iter = total_iters;
        return strict;
    }

    primary.n_iter = total_iters;
    return primary;
}

// --------------------------------------------------------------------------- //
// Main drivers — run_msindo_core (RHF) + run_msindo_core_uhf (UHF)            //
// --------------------------------------------------------------------------- //

// Forward declaration for mutual recursion.
inline MsindoResult run_msindo_core_uhf(const std::vector<int>& Z,
                                         const std::vector<std::array<double, 3>>& coords_angstrom,
                                         const MsindoParameterSet& p,
                                         int max_iter, double conv_tol,
                                         int multiplicity, bool nddo,
                                         int charge);

// Closed-shell RHF: general parameter-driven.
inline MsindoResult run_msindo_core(const std::vector<int>& Z,
                                     const std::vector<std::array<double, 3>>& coords_angstrom,
                                     const MsindoParameterSet& p,
                                     int max_iter=200, double conv_tol=1e-9,
                                     bool nddo=false, int charge = 0) {
    int natom=static_cast<int>(Z.size());
    std::vector<Eigen::Vector3d> C(natom);
    for (int i=0; i<natom; ++i)
        C[i]=Eigen::Vector3d(coords_angstrom[i][0],coords_angstrom[i][1],coords_angstrom[i][2])*ANGSTROM_TO_BOHR;

    std::vector<std::array<int,2>> blocks; int nsto=0;
    blocks.reserve(natom);
    for (int z:Z) { int nb=p.n_basis(z); blocks.push_back({nsto,nsto+nb}); nsto+=nb; }

    std::vector<int> cz(natom); int nelec=0;
    for (int i=0;i<natom;++i) { cz[i]=MsindoParameterSet::eff_core_charge(Z[i]); nelec+=cz[i]; }
    nelec -= charge;
    if (nelec < 0) {
        MsindoResult res;
        res.converged = false;
        return res;
    }
    if (nelec%2!=0) {
        // odd-electron -> auto-dispatch to UHF with multiplicity=2
        return run_msindo_core_uhf(Z, coords_angstrom, p, max_iter, conv_tol, 2, nddo, charge);
    }
    int nocc=nelec/2;

    Eigen::MatrixXd H, G;
    build_core_and_gamma(Z, C, blocks, nsto, p, nddo, H, G);

    std::function<std::pair<Eigen::MatrixXd,double>(const Eigen::MatrixXd&)> fock_extra;
    if (nddo) fock_extra = nddo_fock_extra_builder(Z, C, blocks, p);

    // NDDO has its own Fock contribution and remains on strict DIIS.  Ordinary
    // non-trajectory INDO uses the shared stationary-basin selector; the
    // trajectory-pinned heavy-element energy route remains direct WICHT.
    MsindoResult res = nddo
        ? scf_rhf_driver(H, G, blocks, Z, nocc, p, max_iter, conv_tol, fock_extra)
        : scf_rhf_molecular_driver(H, G, blocks, Z, nocc, p, max_iter, conv_tol);

    double e_core=0.0;
    for (int a=0;a<natom;++a) for (int b=a+1;b<natom;++b)
        e_core+=cz[a]*cz[b]/(C[a]-C[b]).norm();

    res.total_energy=res.electronic_energy+e_core;
    double ateng_sum=0.0;
    for (int z:Z) ateng_sum+=ateng(z,p);
    res.binding_energy=res.total_energy-ateng_sum;
    return res;
}

// Backward-compatible: without parameter set, uses embedded H-F.
inline MsindoResult run_msindo_core(const std::vector<int>& Z,
                                     const std::vector<std::array<double, 3>>& coords_angstrom,
                                     int max_iter=200, double conv_tol=1e-9,
                                     int charge = 0) {
    return run_msindo_core(Z, coords_angstrom, MsindoParameterSet::embedded_hf(),
                            max_iter, conv_tol, false, charge);
}

// Open-shell INDO UHF driver: s/p/d elements. NDDO is closed-shell only.
inline MsindoResult run_msindo_core_uhf(const std::vector<int>& Z,
                                         const std::vector<std::array<double, 3>>& coords_angstrom,
                                         const MsindoParameterSet& p,
                                         int max_iter=200, double conv_tol=1e-9,
                                         int multiplicity=2, bool nddo=false,
                                         int charge = 0) {
    if (nddo) {
        // This guard also catches run_msindo_core's odd-electron auto-dispatch.
        throw std::invalid_argument(
            "MSINDO NDDO mode is closed-shell (RHF) only; open-shell NDDO / UHF "
            "is not implemented.");
    }
    int natom=static_cast<int>(Z.size());
    std::vector<Eigen::Vector3d> C(natom);
    for (int i=0;i<natom;++i)
        C[i]=Eigen::Vector3d(coords_angstrom[i][0],coords_angstrom[i][1],coords_angstrom[i][2])*ANGSTROM_TO_BOHR;

    std::vector<std::array<int,2>> blocks; int nsto=0;
    blocks.reserve(natom);
    for (int z:Z) { int nb=p.n_basis(z); blocks.push_back({nsto,nsto+nb}); nsto+=nb; }

    std::vector<int> cz(natom); int nelec=0;
    for (int i=0;i<natom;++i) { cz[i]=MsindoParameterSet::eff_core_charge(Z[i]); nelec+=cz[i]; }
    nelec -= charge;
    if (nelec < 0 || ((nelec + multiplicity) % 2 == 0)) {
        MsindoResult res;
        res.converged=false;
        return res;
    }

    int nalpha=(nelec+multiplicity-1)/2;
    int nbeta=nelec-nalpha;
    if (nbeta<0) { MsindoResult res; res.converged=false; return res; }

    Eigen::MatrixXd H, G;
    build_core_and_gamma(Z, C, blocks, nsto, p, nddo, H, G);

    // Heavier d/p-block radicals (Tc–Pd, Sb–Xe) need the MSINDO-faithful
    // Hückel-guess + WICHT UHF SCF (the DIIS path jumps SCF basins for them, the
    // same basin-selection issue as the closed-shell heavy elements); every other
    // element keeps DIIS.  NDDO is closed-shell only.  Lockstep with python
    // msindo._run_uhf's dispatch.
    bool use_msindo_scf = !nddo;
    if (use_msindo_scf) {
        use_msindo_scf = false;
        for (int z : Z)
            if (is_msindo_trajectory_scf(z)) { use_msindo_scf = true; break; }
    }
    if (use_msindo_scf) {
        MsindoResult res = scf_uhf_msindo_driver(H, G, blocks, Z, nalpha, nbeta, p,
                                                 std::max(max_iter, 5000));
        double e_core=0.0;
        for (int a=0;a<natom;++a) for (int b=a+1;b<natom;++b)
            e_core+=cz[a]*cz[b]/(C[a]-C[b]).norm();
        res.total_energy=res.electronic_energy+e_core;
        double ateng_sum=0.0;
        for (int z:Z) ateng_sum+=ateng(z,p);
        res.binding_energy=res.total_energy-ateng_sum;
        return res;
    }

    std::function<std::pair<Eigen::MatrixXd,double>(const Eigen::MatrixXd&)> fock_extra;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H);
    Eigen::MatrixXd CA=es.eigenvectors();
    Eigen::MatrixXd PA=CA.leftCols(nalpha)*CA.leftCols(nalpha).transpose();
    Eigen::MatrixXd PB;
    if (nbeta>0) {
        PB=CA.leftCols(nbeta)*CA.leftCols(nbeta).transpose();
    } else {
        PB=Eigen::MatrixXd::Zero(nsto,nsto);
    }

    double e_elec=0.0; bool converged=false; int it=0;
    std::vector<Eigen::MatrixXd> fa_hist, fb_hist, e_hist;
    const int DIIS_MAX=8;
    fa_hist.reserve(DIIS_MAX + 1);
    fb_hist.reserve(DIIS_MAX + 1);
    e_hist.reserve(DIIS_MAX + 1);
    for (it=1;it<=max_iter;++it) {
        Eigen::MatrixXd FA, FB;
        build_fock_uhf(H, G, PA, PB, blocks, Z, p, FA, FB);
        double e_add=0.0;
        if (fock_extra) {
            auto fe=fock_extra(PA+PB);
            FA+=fe.first; FB+=fe.first; e_add=fe.second;
        }
        Eigen::MatrixXd errA=FA*PA-PA*FA, errB=FB*PB-PB*FB;
        double e_cur=0.5*((PA+PB).array()*H.array()).sum()
                      +0.5*(PA.array()*FA.array()).sum()
                      +0.5*(PB.array()*FB.array()).sum()+e_add;
        double res_err=errA.cwiseAbs().maxCoeff();
        if (nbeta>0) res_err=std::max(res_err, errB.cwiseAbs().maxCoeff());
        if (res_err<conv_tol && std::abs(e_cur-e_elec)<conv_tol) {
            e_elec=e_cur; converged=true; break;
        }
        e_elec=e_cur;
        fa_hist.push_back(FA); fb_hist.push_back(FB);
        Eigen::MatrixXd errStacked(nsto*2,1);
        for (int i=0;i<nsto;++i) { errStacked(i,0)=errA(i%nsto,i/nsto); errStacked(nsto+i,0)=errB(i%nsto,i/nsto); }
        e_hist.push_back(errStacked);
        if (static_cast<int>(fa_hist.size())>DIIS_MAX) { fa_hist.erase(fa_hist.begin()); fb_hist.erase(fb_hist.begin()); e_hist.erase(e_hist.begin()); }
        int n=static_cast<int>(fa_hist.size());
        Eigen::MatrixXd FA_eff=FA, FB_eff=FB;
        if (n>=2) {
            Eigen::MatrixXd B=Eigen::MatrixXd::Constant(n+1,n+1,-1.0); B(n,n)=0.0;
            for (int a=0;a<n;++a) for (int b=a;b<n;++b) {
                double v=(e_hist[a].array()*e_hist[b].array()).sum(); B(a,b)=v; B(b,a)=v;
            }
            Eigen::VectorXd rhs=Eigen::VectorXd::Zero(n+1); rhs(n)=-1.0;
            Eigen::VectorXd c(n+1);
            if (solve_diis_coefficients(B, rhs, c)) {
                FA_eff=Eigen::MatrixXd::Zero(nsto,nsto); FB_eff=Eigen::MatrixXd::Zero(nsto,nsto);
                for (int a=0;a<n;++a) { FA_eff+=c(a)*fa_hist[a]; FB_eff+=c(a)*fb_hist[a]; }
            }
        }
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> esA(FA_eff);
        PA=esA.eigenvectors().leftCols(nalpha)*esA.eigenvectors().leftCols(nalpha).transpose();
        if (nbeta>0) {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> esB(FB_eff);
            PB=esB.eigenvectors().leftCols(nbeta)*esB.eigenvectors().leftCols(nbeta).transpose();
        }
    }
    // Final energy
    Eigen::MatrixXd FA, FB;
    build_fock_uhf(H, G, PA, PB, blocks, Z, p, FA, FB);
    double e_add=0.0;
    if (fock_extra) { auto fe=fock_extra(PA+PB); FA+=fe.first; FB+=fe.first; e_add=fe.second; }
    e_elec=0.5*((PA+PB).array()*H.array()).sum()
           +0.5*(PA.array()*FA.array()).sum()+0.5*(PB.array()*FB.array()).sum()+e_add;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> esfA(FA);
    Eigen::VectorXd epsA=esfA.eigenvalues();

    double e_core=0.0;
    for (int a=0;a<natom;++a) for (int b=a+1;b<natom;++b)
        e_core+=cz[a]*cz[b]/(C[a]-C[b]).norm();

    MsindoResult res;
    res.electronic_energy=e_elec;
    res.total_energy=e_elec+e_core;
    double ateng_sum=0.0;
    for (int z:Z) ateng_sum+=ateng(z,p);
    res.binding_energy=res.total_energy-ateng_sum;
    res.mo_energies=epsA;
    res.density=PA+PB;
    res.n_iter=it; res.converged=converged;
    return res;
}

}  // namespace indo
}  // namespace semiempirical
}  // namespace vibeqc
