#include "vibeqc/semiempirical/parameters.hpp"

#include "vibeqc/semiempirical/core/parameter_identity.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vibeqc {
namespace semiempirical {

namespace {

constexpr int kMinPhysicalAtomicNumber = 1;
constexpr int kMaxPhysicalAtomicNumber = 118;

constexpr const char* kDftbScreeningV1Sha256 =
    "9b043a8e7a85244f5ffafc3b9f4965015db3b5e26346e0d4cb22702a8887fb40";
constexpr const char* kDftbScreeningV1Identity =
    "vibeqc-inhouse-dftb-screening-v1";

void validate_atomic_number(int Z, const char* context) {
    if (Z < kMinPhysicalAtomicNumber || Z > kMaxPhysicalAtomicNumber) {
        throw std::invalid_argument(
            std::string("SemiempiricalParameters: ") + context
            + " atomic number must be in [1, 118]");
    }
}

void validate_finite(double value, const char* field) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(
            std::string("SemiempiricalParameters: non-finite ") + field);
    }
}

void validate_finite_values(const std::vector<double>& values,
                            const char* field) {
    for (double value : values) validate_finite(value, field);
}

void validate_element_data(const ElementData& elem) {
    validate_atomic_number(elem.Z, "element");
    validate_finite_values(elem.on_site, "on-site energy");
    validate_finite_values(elem.zeta, "STO exponent");
    validate_finite(elem.hubbard_u, "Hubbard U");
}

void validate_repulsive_data(const RepulsivePairV2& rp) {
    validate_finite(rp.A, "repulsive A");
    validate_finite(rp.B, "repulsive B");

    const auto& positions = rp.spline.knot_positions();
    const auto& values = rp.spline.knot_values();
    if (positions.size() != values.size()
        || (!positions.empty() && positions.size() < 2)) {
        throw std::invalid_argument(
            "SemiempiricalParameters: invalid repulsive spline knot data");
    }
    validate_finite_values(positions, "repulsive spline knot position");
    validate_finite_values(values, "repulsive spline knot value");
    for (std::size_t i = 1; i < positions.size(); ++i) {
        if (!(positions[i] > positions[i - 1])) {
            throw std::invalid_argument(
                "SemiempiricalParameters: repulsive spline knot positions "
                "must be strictly increasing");
        }
    }
}

}  // namespace

void SemiempiricalParameters::add_element(const ElementData& elem) {
    validate_element_data(elem);
    elements_[elem.Z] = elem;
}

bool SemiempiricalParameters::has_element(int Z) const {
    return elements_.find(Z) != elements_.end();
}

const ElementData& SemiempiricalParameters::element(int Z) const {
    auto it = elements_.find(Z);
    if (it == elements_.end()) {
        throw std::runtime_error(
            "SemiempiricalParameters: element Z=" + std::to_string(Z)
            + " not found in parameter set");
    }
    return it->second;
}

double SemiempiricalParameters::on_site_energy(int Z, int l) const {
    const auto& e = element(Z);
    if (l < 0 || static_cast<std::size_t>(l) >= e.on_site.size()) return 0.0;
    return e.on_site[l];
}

double SemiempiricalParameters::sto_exponent(int Z, int l) const {
    const auto& e = element(Z);
    if (l < 0 || static_cast<std::size_t>(l) >= e.zeta.size()) return 0.0;
    return e.zeta[l];
}

double SemiempiricalParameters::average_on_site(int Z) const {
    const auto& e = element(Z);
    double sum = 0.0;
    int count = 0;
    for (double v : e.on_site) {
        if (v != 0.0) { sum += v; count++; }
    }
    return count > 0 ? sum / count : 0.0;
}

double SemiempiricalParameters::hubbard_u(int Z) const {
    return element(Z).hubbard_u;
}

int SemiempiricalParameters::valence_electrons(int Z) const {
    return element(Z).valence_electrons;
}

double SemiempiricalParameters::gamma_onsite(int Z) const {
    return element(Z).hubbard_u;
}

int SemiempiricalParameters::_pair_key(int Z1, int Z2) {
    validate_atomic_number(Z1, "repulsive-pair");
    validate_atomic_number(Z2, "repulsive-pair");
    int lo = std::min(Z1, Z2);
    int hi = std::max(Z1, Z2);
    return lo * 1000 + hi;
}

void SemiempiricalParameters::set_repulsive_pair(int Z1, int Z2,
                                                  const RepulsivePairV2& rp) {
    validate_repulsive_data(rp);
    repulsive_pairs_[_pair_key(Z1, Z2)] = rp;
}

const RepulsivePairV2* SemiempiricalParameters::repulsive_pair(
    int Z1, int Z2) const {
    auto it = repulsive_pairs_.find(_pair_key(Z1, Z2));
    return (it != repulsive_pairs_.end()) ? &it->second : nullptr;
}

void SemiempiricalParameters::set_kappa(double k) {
    validate_finite(k, "Wolfsberg-Helmholtz kappa");
    kappa_ = k;
}

std::string SemiempiricalParameters::content_sha256() const {
    detail::CanonicalParameterHasher hasher("dftb-parameters-v1");
    validate_finite(kappa_, "Wolfsberg-Helmholtz kappa");
    hasher.add_double(kappa_);

    hasher.add_size(elements_.size());
    for (const auto& item : elements_) {
        const int Z = item.first;
        const ElementData& elem = item.second;
        validate_atomic_number(Z, "stored element");
        validate_element_data(elem);
        if (elem.Z != Z) {
            throw std::invalid_argument(
                "SemiempiricalParameters: stored element key mismatch");
        }

        hasher.add_int(Z);
        hasher.add_size(elem.on_site.size());
        for (double value : elem.on_site) hasher.add_double(value);
        hasher.add_size(elem.zeta.size());
        for (double value : elem.zeta) hasher.add_double(value);
        hasher.add_double(elem.hubbard_u);
        hasher.add_int(elem.valence_electrons);
    }

    hasher.add_size(repulsive_pairs_.size());
    for (const auto& item : repulsive_pairs_) {
        const int Z1 = item.first / 1000;
        const int Z2 = item.first % 1000;
        validate_atomic_number(Z1, "stored repulsive-pair");
        validate_atomic_number(Z2, "stored repulsive-pair");
        if (Z1 > Z2 || _pair_key(Z1, Z2) != item.first) {
            throw std::invalid_argument(
                "SemiempiricalParameters: invalid stored repulsive-pair key");
        }

        const RepulsivePairV2& rp = item.second;
        validate_repulsive_data(rp);
        hasher.add_int(Z1);
        hasher.add_int(Z2);
        hasher.add_double(rp.A);
        hasher.add_double(rp.B);

        const auto& positions = rp.spline.knot_positions();
        const auto& values = rp.spline.knot_values();
        hasher.add_size(positions.size());
        for (double value : positions) hasher.add_double(value);
        hasher.add_size(values.size());
        for (double value : values) hasher.add_double(value);
    }
    return hasher.finish();
}

std::string SemiempiricalParameters::parameter_identity() const {
    const std::string sha256 = content_sha256();
    if (sha256 == kDftbScreeningV1Sha256) {
        return kDftbScreeningV1Identity;
    }
    return detail::custom_parameter_identity(sha256);
}


double SemiempiricalParameters::default_repulsive_A(int Z1, int Z2) const {
    // Estimate from Hubbard U: larger U → harder atom → larger A
    double U1 = hubbard_u(Z1);
    double U2 = hubbard_u(Z2);
    if (U1 <= 0.0) U1 = 0.3;
    if (U2 <= 0.0) U2 = 0.3;
    // A ≈ 100 * U1 * U2 gives reasonable bond lengths for main-group elements
    return 100.0 * U1 * U2;
}

double SemiempiricalParameters::repulsive_energy(int Z1, int Z2, double R) const {
    if (R < 1e-12) return 0.0;
    const auto* rp = repulsive_pair(Z1, Z2);
    if (rp != nullptr && (!rp->spline.empty() || rp->A != 0.0 || rp->B > 0.0)) {
        return eval_repulsive(*rp, R);
    }
    // Fallback: R⁻¹² with default A
    double A = default_repulsive_A(Z1, Z2);
    return A / std::pow(R, 12);
}

double SemiempiricalParameters::repulsive_derivative(int Z1, int Z2, double R) const {
    if (R < 1e-12) return 0.0;
    const auto* rp = repulsive_pair(Z1, Z2);
    if (rp != nullptr && (!rp->spline.empty() || rp->A != 0.0 || rp->B > 0.0)) {
        return eval_repulsive_derivative(*rp, R);
    }
    double A = default_repulsive_A(Z1, Z2);
    return -12.0 * A / std::pow(R, 13);
}

// ---------------------------------------------------------------------------
// Built-in DFTB0 / SCC-DFTB default parameter set (H, C, N, O)
// ---------------------------------------------------------------------------

SemiempiricalParameters SemiempiricalParameters::dftb0_default() {
    SemiempiricalParameters p;

    // Hydrogen: Z=1, 1s, U_s=0.4195, 1 valence e⁻
    p.add_element({1, {-0.2066}, {1.24}, 0.4195, 1});

    // Carbon: Z=6, 2s+2p, U_s=U_p=0.3647, 4 valence e⁻
    p.add_element({6, {-0.5040, -0.1943}, {1.608, 1.608}, 0.3647, 4});

    // Nitrogen: Z=7, 2s+2p, U_s=U_p=0.4038, 5 valence e⁻
    p.add_element({7, {-0.6809, -0.2864}, {1.924, 1.924}, 0.4038, 5});

    // Oxygen: Z=8, 2s+2p, U_s=U_p=0.4459, 6 valence e⁻
    p.add_element({8, {-0.8877, -0.3321}, {2.246, 2.246}, 0.4459, 6});

    // Fluorine: Z=9, 2s+2p, U=0.53, 7 valence e⁻
    p.add_element({9, {-1.1000, -0.4800}, {2.55, 2.55}, 0.5300, 7});

    // Phosphorus: Z=15, 3s+3p, U=0.32, 5 valence e⁻
    p.add_element({15, {-0.5000, -0.1800}, {1.60, 1.60}, 0.3200, 5});

    // Sulfur: Z=16, 3s+3p, U=0.35, 6 valence e⁻
    p.add_element({16, {-0.6000, -0.2200}, {1.80, 1.80}, 0.3500, 6});

    // Chlorine: Z=17, 3s+3p, U=0.42, 7 valence e⁻
    p.add_element({17, {-0.7500, -0.2800}, {2.00, 2.00}, 0.4200, 7});


    // --- Extended element coverage (main group + transition metals) ---
    // Li: Z=3, 2s, U=0.25, 1 valence e⁻
    p.add_element({3, {-0.1500}, {0.65}, 0.2500, 1});
    // Be: Z=4, 2s, U=0.30, 2 valence e⁻
    p.add_element({4, {-0.2500}, {0.975}, 0.3000, 2});
    // B: Z=5, 2s+2p, U=0.33, 3 valence e⁻
    p.add_element({5, {-0.3500, -0.1500}, {1.30, 1.30}, 0.3300, 3});
    // Na: Z=11, 3s, U=0.20, 1 valence e⁻
    p.add_element({11, {-0.1200}, {0.733}, 0.2000, 1});
    // Mg: Z=12, 3s, U=0.25, 2 valence e⁻
    p.add_element({12, {-0.2000}, {0.95}, 0.2500, 2});
    // Al: Z=13, 3s+3p, U=0.28, 3 valence e⁻
    p.add_element({13, {-0.2800, -0.1000}, {1.167, 1.167}, 0.2800, 3});
    // Si: Z=14, 3s+3p, U=0.30, 4 valence e⁻
    p.add_element({14, {-0.4000, -0.1500}, {1.383, 1.383}, 0.3000, 4});
    // He: Z=2, 1s, U=0.60, 2 valence e⁻
    p.add_element({2, {-0.6000}, {1.70}, 0.6000, 2});
    // Ne: Z=10, 2s+2p, U=0.65, 8 valence e⁻
    p.add_element({10, {-1.5000, -0.7000}, {2.925, 2.925}, 0.6500, 8});
    // Ar: Z=18, 3s+3p, U=0.50, 8 valence e⁻
    p.add_element({18, {-0.9000, -0.4000}, {2.20, 2.20}, 0.5000, 8});
    // K: Z=19, 4s, U=0.18, 1 valence e⁻
    p.add_element({19, {-0.1000}, {0.925}, 0.1800, 1});
    // Br: Z=35, 4s+4p, U=0.45, 7 valence e⁻
    p.add_element({35, {-0.7000, -0.2500}, {2.00, 2.00}, 0.4500, 7});
    // I: Z=53, 5s+5p, U=0.38, 7 valence e⁻
    p.add_element({53, {-0.5500, -0.2000}, {1.80, 1.80}, 0.3800, 7});
    // Fe: Z=26, 3d+4s, U=0.25, 8 valence e⁻
    p.add_element({26, {-0.4000, -0.1000, -0.2500}, {1.50, 1.50, 2.00}, 0.2500, 8});
    // Cu: Z=29, 3d+4s, U=0.28, 11 valence e⁻
    p.add_element({29, {-0.4500, -0.1200, -0.3000}, {1.80, 1.80, 2.50}, 0.2800, 11});
    // Zn: Z=30, 3d+4s+4p, U=0.30, 12 valence e⁻
    p.add_element({30, {-0.5000, -0.1400, -0.3500}, {2.00, 2.00, 2.80}, 0.3000, 12});


    // --- Row 4 transition metals (Ca–Ni) ---
    // Ca: Z=20, 4s, U=0.18, 2 valence
    p.add_element({20, {-0.1500}, {0.800}, 0.1800, 2});
    // Sc: Z=21, 3d+4s+4p, U=0.22, 3 valence
    p.add_element({21, {-0.3200, -0.2200, -0.3700}, {0.622, 0.559, 0.700}, 0.2200, 3});
    // Ti: Z=22, 3d+4s+4p, U=0.24, 4 valence
    p.add_element({22, {-0.3700, -0.2700, -0.4200}, {0.797, 0.718, 0.933}, 0.2400, 4});
    // V: Z=23, 3d+4s+4p, U=0.26, 5 valence
    p.add_element({23, {-0.4200, -0.3200, -0.4700}, {0.973, 0.876, 1.167}, 0.2600, 5});
    // Cr: Z=24, 3d+4s+4p, U=0.25, 6 valence
    p.add_element({24, {-0.4700, -0.3700, -0.5200}, {1.149, 1.034, 1.400}, 0.2500, 6});
    // Mn: Z=25, 3d+4s+4p, U=0.27, 7 valence
    p.add_element({25, {-0.5200, -0.4200, -0.5700}, {1.324, 1.192, 1.633}, 0.2700, 7});
    // Co: Z=27, 3d+4s+4p, U=0.28, 9 valence
    p.add_element({27, {-0.6200, -0.5200, -0.6700}, {1.676, 1.508, 2.100}, 0.2800, 9});
    // Ni: Z=28, 3d+4s+4p, U=0.30, 10 valence
    p.add_element({28, {-0.6700, -0.5700, -0.7200}, {1.851, 1.666, 2.333}, 0.3000, 10});
    // Ga: Z=31, 4s+4p, U=0.25, 3 valence
    p.add_element({31, {-0.2500, -0.1800}, {0.811, 0.770}, 0.2500, 3});
    // Ge: Z=32, 4s+4p, U=0.27, 4 valence
    p.add_element({32, {-0.3300, -0.2600}, {0.986, 0.937}, 0.2700, 4});
    // As: Z=33, 4s+4p, U=0.30, 5 valence
    p.add_element({33, {-0.4000, -0.3300}, {1.162, 1.104}, 0.3000, 5});
    // Se: Z=34, 4s+4p, U=0.33, 6 valence
    p.add_element({34, {-0.5000, -0.4300}, {1.338, 1.271}, 0.3300, 6});
    // Kr: Z=36, 4s+4p, U=0.45, 8 valence
    p.add_element({36, {-0.6000, -0.5300}, {1.689, 1.605}, 0.4500, 8});
    // Mo: Z=42, 4d+5s+5p, U=0.22, 6 valence
    p.add_element({42, {-0.5800, -0.5300, -0.6800}, {1.149, 0.976, 1.300}, 0.2200, 6});
    // Pd: Z=46, 4d+5s+5p, U=0.28, 10 valence
    p.add_element({46, {-0.9000, -0.8500, -1.0000}, {1.851, 1.574, 2.167}, 0.2800, 10});
    // Ag: Z=47, 4d+5s+5p, U=0.26, 11 valence
    p.add_element({47, {-0.9800, -0.9300, -1.0800}, {2.027, 1.723, 2.383}, 0.2600, 11});
    // Cd: Z=48, 4d+5s+5p, U=0.27, 12 valence
    p.add_element({48, {-1.0600, -1.0100, -1.1600}, {2.203, 1.872, 2.600}, 0.2700, 12});
    // W: Z=74, 5d+6s+6p, U=0.22, 6 valence
    p.add_element({74, {-0.4400, -0.3900, -0.5700}, {1.149, 0.919, 1.200}, 0.2200, 6});
    // Pt: Z=78, 5d+6s+6p, U=0.26, 10 valence
    p.add_element({78, {-0.6800, -0.6300, -0.8100}, {1.851, 1.481, 2.000}, 0.2600, 10});
    // Au: Z=79, 5d+6s+6p, U=0.25, 11 valence
    p.add_element({79, {-0.7400, -0.6900, -0.8700}, {2.027, 1.622, 2.200}, 0.2500, 11});
    // Hg: Z=80, 5d+6s+6p, U=0.28, 12 valence
    p.add_element({80, {-0.8000, -0.7500, -0.9300}, {2.203, 1.762, 2.400}, 0.2800, 12});


    // --- Alkali/alkaline earth (Rb, Sr, Cs, Ba) ---
    p.add_element({37, {-0.0800}, {0.650}, 0.1600, 1});  // Rb
    p.add_element({38, {-0.1300}, {0.750}, 0.1800, 2});  // Sr
    p.add_element({55, {-0.0600}, {0.550}, 0.1400, 1});  // Cs
    p.add_element({56, {-0.1100}, {0.650}, 0.1700, 2});  // Ba
    // --- 4d transition metals (Y–Rh) ---
    p.add_element({39, {-0.2800, -0.1800, -0.3300}, {0.550, 0.500, 0.600}, 0.2000, 3});  // Y
    p.add_element({40, {-0.3300, -0.2300, -0.3800}, {0.730, 0.660, 0.800}, 0.2200, 4});  // Zr
    p.add_element({41, {-0.3800, -0.2800, -0.4300}, {0.780, 0.700, 0.900}, 0.2400, 5});  // Nb
    p.add_element({43, {-0.4800, -0.3800, -0.5300}, {1.000, 0.900, 1.100}, 0.2400, 7});  // Tc
    p.add_element({44, {-0.5300, -0.4300, -0.5800}, {1.140, 1.030, 1.250}, 0.2600, 8});  // Ru
    p.add_element({45, {-0.5800, -0.4800, -0.6300}, {1.270, 1.140, 1.400}, 0.2700, 9});  // Rh
    // --- 5d transition metals (Re–Ir) ---
    p.add_element({75, {-0.3800, -0.3300, -0.5100}, {0.950, 0.760, 1.100}, 0.2200, 7});  // Re
    p.add_element({76, {-0.4300, -0.3800, -0.5600}, {1.100, 0.880, 1.270}, 0.2400, 8});  // Os
    p.add_element({77, {-0.4800, -0.4300, -0.6100}, {1.240, 0.990, 1.430}, 0.2600, 9});  // Ir


    // --- Row 5 p-block (In–Xe) ---
    p.add_element({49, {-0.2200, -0.1400}, {1.20, 1.14}, 0.24, 3});   // In
    p.add_element({50, {-0.3000, -0.2000}, {1.35, 1.28}, 0.26, 4});   // Sn
    p.add_element({51, {-0.3800, -0.2600}, {1.50, 1.42}, 0.28, 5});   // Sb
    p.add_element({52, {-0.4700, -0.3300}, {1.65, 1.57}, 0.31, 6});   // Te
    p.add_element({54, {-0.6500, -0.5000}, {1.95, 1.85}, 0.40, 8});   // Xe
    // --- 5d transition metals (Hf, Ta) ---
    p.add_element({72, {-0.3500, -0.2500, -0.4000}, {1.20, 0.96, 1.40}, 0.22, 4});  // Hf
    p.add_element({73, {-0.4000, -0.3000, -0.4500}, {1.35, 1.08, 1.60}, 0.23, 5});  // Ta
    // --- Row 6 p-block (Tl–Bi) ---
    p.add_element({81, {-0.2000, -0.1200}, {1.10, 1.05}, 0.22, 3});   // Tl
    p.add_element({82, {-0.2800, -0.1800}, {1.25, 1.19}, 0.24, 4});   // Pb
    p.add_element({83, {-0.3600, -0.2400}, {1.40, 1.33}, 0.26, 5});   // Bi

    // --- Lanthanides (Z=57-71): 4f + 5d + 6s + 6p valence ---
    // La: 5d¹6s², 3 valence
    p.add_element({57, {-0.1300, -0.0600, -0.1800, -0.2500}, {0.70, 0.60, 0.90, 1.10}, 0.20, 3});
    // Ce: 4f¹5d¹6s², 4 valence
    p.add_element({58, {-0.1400, -0.0700, -0.2000, -0.2800}, {0.75, 0.65, 0.95, 1.15}, 0.21, 4});
    // Pr: 4f³6s², 5 valence
    p.add_element({59, {-0.1500, -0.0800, -0.2200, -0.3100}, {0.80, 0.70, 1.00, 1.20}, 0.22, 5});
    // Nd: 4f⁴6s², 6 valence
    p.add_element({60, {-0.1600, -0.0900, -0.2400, -0.3400}, {0.85, 0.75, 1.05, 1.25}, 0.23, 6});
    // Pm: 4f⁵6s², 7 valence
    p.add_element({61, {-0.1700, -0.1000, -0.2600, -0.3700}, {0.90, 0.80, 1.10, 1.30}, 0.23, 7});
    // Sm: 4f⁶6s², 8 valence
    p.add_element({62, {-0.1800, -0.1100, -0.2800, -0.4000}, {0.95, 0.85, 1.15, 1.35}, 0.24, 8});
    // Eu: 4f⁷6s², 9 valence
    p.add_element({63, {-0.1900, -0.1200, -0.3000, -0.4300}, {1.00, 0.90, 1.20, 1.40}, 0.25, 9});
    // Gd: 4f⁷5d¹6s², 10 valence
    p.add_element({64, {-0.2000, -0.1300, -0.3200, -0.4600}, {1.05, 0.95, 1.25, 1.45}, 0.25, 10});
    // Tb: 4f⁹6s², 11 valence
    p.add_element({65, {-0.2100, -0.1400, -0.3400, -0.4900}, {1.10, 1.00, 1.30, 1.50}, 0.26, 11});
    // Dy: 4f¹⁰6s², 12 valence
    p.add_element({66, {-0.2200, -0.1500, -0.3600, -0.5200}, {1.15, 1.05, 1.35, 1.55}, 0.26, 12});
    // Ho: 4f¹¹6s², 13 valence
    p.add_element({67, {-0.2300, -0.1600, -0.3800, -0.5500}, {1.20, 1.10, 1.40, 1.60}, 0.27, 13});
    // Er: 4f¹²6s², 14 valence
    p.add_element({68, {-0.2400, -0.1700, -0.4000, -0.5800}, {1.25, 1.15, 1.45, 1.65}, 0.27, 14});
    // Tm: 4f¹³6s², 15 valence
    p.add_element({69, {-0.2500, -0.1800, -0.4200, -0.6100}, {1.30, 1.20, 1.50, 1.70}, 0.28, 15});
    // Yb: 4f¹⁴6s², 16 valence
    p.add_element({70, {-0.2600, -0.1900, -0.4400, -0.6400}, {1.35, 1.25, 1.55, 1.75}, 0.29, 16});
    // Lu: 4f¹⁴5d¹6s², 3 valence (counting 5d+6s)
    p.add_element({71, {-0.2700, -0.2000, -0.4600, -0.6700}, {1.40, 1.30, 1.60, 1.80}, 0.30, 3});

    // --- Actinides (Z=89-92): 5f + 6d + 7s + 7p valence ---
    // Ac: 6d¹7s², 3 valence
    p.add_element({89, {-0.1200, -0.0500, -0.1700, -0.2400}, {0.65, 0.55, 0.85, 1.05}, 0.19, 3});
    // Th: 6d²7s², 4 valence
    p.add_element({90, {-0.1300, -0.0600, -0.1900, -0.2700}, {0.70, 0.60, 0.90, 1.10}, 0.20, 4});
    // Pa: 5f²6d¹7s², 5 valence
    p.add_element({91, {-0.1400, -0.0700, -0.2100, -0.3000}, {0.75, 0.65, 0.95, 1.15}, 0.21, 5});
    // U: 5f³6d¹7s², 6 valence
    p.add_element({92, {-0.1500, -0.0800, -0.2300, -0.3300}, {0.80, 0.70, 1.00, 1.20}, 0.22, 6});

    // --- Remaining gaps (At, Rn, Fr, Ra) for full H–U coverage ---
    p.add_element({85, {-0.5500, -0.4000}, {1.85, 1.76}, 0.38, 7});   // At
    p.add_element({86, {-0.7000, -0.5500}, {2.10, 2.00}, 0.48, 8});   // Rn
    p.add_element({87, {-0.0500}, {0.50}, 0.13, 1});                  // Fr
    p.add_element({88, {-0.1000}, {0.60}, 0.16, 2});                  // Ra


    // --- Repulsive pair parameters (R^-12 form: A/R^12) ---
    // Tuned to give reasonable bond lengths for organic molecules.
    // H-X pairs
    p.set_repulsive_pair(1, 1, {5.0, 0.0});
    p.set_repulsive_pair(1, 6, {25.0, 0.0});
    p.set_repulsive_pair(1, 7, {25.0, 0.0});
    p.set_repulsive_pair(1, 8, {15.0, 0.0});
    // X-Y pairs among C, N, O
    p.set_repulsive_pair(6, 6, {60.0, 0.0});
    p.set_repulsive_pair(6, 7, {70.0, 0.0});
    p.set_repulsive_pair(6, 8, {50.0, 0.0});
    p.set_repulsive_pair(7, 7, {80.0, 0.0});
    p.set_repulsive_pair(7, 8, {60.0, 0.0});
    p.set_repulsive_pair(8, 8, {40.0, 0.0});
    // F-containing pairs
    p.set_repulsive_pair(1, 9, {20.0, 0.0});
    p.set_repulsive_pair(6, 9, {80.0, 0.0});
    p.set_repulsive_pair(7, 9, {75.0, 0.0});
    p.set_repulsive_pair(8, 9, {55.0, 0.0});
    p.set_repulsive_pair(9, 9, {25.0, 0.0});
    // P-containing pairs
    p.set_repulsive_pair(1, 15, {40.0, 0.0});
    p.set_repulsive_pair(6, 15, {120.0, 0.0});
    p.set_repulsive_pair(7, 15, {130.0, 0.0});
    p.set_repulsive_pair(8, 15, {100.0, 0.0});
    p.set_repulsive_pair(9, 15, {110.0, 0.0});
    p.set_repulsive_pair(15, 15, {200.0, 0.0});
    // S-containing pairs
    p.set_repulsive_pair(1, 16, {35.0, 0.0});
    p.set_repulsive_pair(6, 16, {100.0, 0.0});
    p.set_repulsive_pair(7, 16, {110.0, 0.0});
    p.set_repulsive_pair(8, 16, {85.0, 0.0});
    p.set_repulsive_pair(9, 16, {90.0, 0.0});
    p.set_repulsive_pair(15, 16, {170.0, 0.0});
    p.set_repulsive_pair(16, 16, {150.0, 0.0});
    // Cl-containing pairs
    p.set_repulsive_pair(1, 17, {30.0, 0.0});
    p.set_repulsive_pair(6, 17, {90.0, 0.0});
    p.set_repulsive_pair(7, 17, {95.0, 0.0});
    p.set_repulsive_pair(8, 17, {75.0, 0.0});
    p.set_repulsive_pair(9, 17, {65.0, 0.0});
    p.set_repulsive_pair(15, 17, {160.0, 0.0});
    p.set_repulsive_pair(16, 17, {140.0, 0.0});
    p.set_repulsive_pair(17, 17, {120.0, 0.0});


    return p;
}


// ---------------------------------------------------------------------------
// Production parameter set with spline-fitted repulsive potentials
// ---------------------------------------------------------------------------

SemiempiricalParameters SemiempiricalParameters::dftb0_production() {
    // Production parameters: extended element coverage (85 elements).
    // Repulsive pairs use the same tuned R^-12 defaults as dftb0_default();
    // the spline-fitted production repulsives are deferred pending a
    // validated DFT reference dataset.
    return dftb0_default();
}

}  // namespace semiempirical
}  // namespace vibeqc
