// Semiempirical parameters for DFTB0 / SCC-DFTB — production version.
//
// Stores per-element on-site energies, STO exponents, Hubbard U,
// and pairwise repulsive potentials (R⁻¹², exponential, or spline).
// Includes a default in-house 68-element parameter set and spline
// I/O for production DFT-fitted repulsives.

#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "vibeqc/semiempirical/repulsive_spline.hpp"

namespace vibeqc {
namespace semiempirical {

// ---------------------------------------------------------------------------
// Per-element data
// ---------------------------------------------------------------------------

using OnSiteEnergies = std::vector<double>;
using StoExponents = std::vector<double>;

struct ElementData {
    int Z;
    OnSiteEnergies on_site;
    StoExponents zeta;
    double hubbard_u = 0.0;
    int valence_electrons = 0;
};

// ---------------------------------------------------------------------------
// Full parameter set (production)
// ---------------------------------------------------------------------------

class SemiempiricalParameters {
public:
    SemiempiricalParameters() = default;

    // ---- Element access ----
    void add_element(const ElementData& elem);
    bool has_element(int Z) const;
    const ElementData& element(int Z) const;
    double on_site_energy(int Z, int l) const;
    double sto_exponent(int Z, int l) const;
    double average_on_site(int Z) const;
    double hubbard_u(int Z) const;
    int valence_electrons(int Z) const;
    double gamma_onsite(int Z) const;

    // ---- Repulsive pair access (production: spline-capable) ----
    void set_repulsive_pair(int Z1, int Z2, const RepulsivePairV2& rp);
    const RepulsivePairV2* repulsive_pair(int Z1, int Z2) const;

    // Evaluate V_rep(R). Uses spline if available, else R⁻¹² or exponential.
    // Falls back to default_repulsive_A estimate when no pair entry exists.
    double repulsive_energy(int Z1, int Z2, double R) const;

    // Repulsive derivative dV_rep/dR (for gradient).
    double repulsive_derivative(int Z1, int Z2, double R) const;

    // Default repulsive A from Hubbard U (R⁻¹² fallback).
    double default_repulsive_A(int Z1, int Z2) const;

    // ---- Wolfsberg-Helmholtz ----
    double kappa() const noexcept { return kappa_; }
    void set_kappa(double k);

    // Canonical identity of every stored, execution-relevant parameter.
    std::string content_sha256() const;
    std::string parameter_identity() const;

    // ---- Built-in ----
    static SemiempiricalParameters dftb0_default();
    static SemiempiricalParameters dftb0_production();

private:
    std::map<int, ElementData> elements_;
    std::map<int, RepulsivePairV2> repulsive_pairs_;
    double kappa_ = 1.75;
    static int _pair_key(int Z1, int Z2);
};

}  // namespace semiempirical
}  // namespace vibeqc
