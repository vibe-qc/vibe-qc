// NDDO-family parameter set (PM6, PM7, OMx).
//
// Implements CoreParameterSet for MNDO/AM1/PMx/OMx methods.
// These methods use an orthogonalized AO basis with parameterized
// 1-center and 2-center integrals.

#pragma once

#include <map>
#include <string>
#include <vector>

#include "vibeqc/semiempirical/core/parameters.hpp"

namespace vibeqc {
namespace semiempirical {
namespace nddo {

// ---------------------------------------------------------------------------
// NDDO one-center parameters per element
// ---------------------------------------------------------------------------

struct NDDOElementData {
    int Z = 0;
    CoreElementData base;

    // One-center one-electron energies (eV, converted to Ha)
    double uss = 0.0;      // <s|H|s>
    double upp = 0.0;      // <p|H|p>
    double udd = 0.0;      // <d|H|d> (if d orbitals)

    // One-center two-electron integrals (eV)
    double gss = 0.0;      // (ss|ss)
    double gpp = 0.0;      // (pp|pp)
    double gsp = 0.0;      // (ss|pp)
    double gp2 = 0.0;      // (pp|p'p')
    double hsp = 0.0;      // (sp|sp) exchange

    // Slater exponents (a0^-1)
    double zs = 0.0;
    double zp = 0.0;
    double zd = 0.0;

    // Neutral-atom Slater exponents (MOPAC: zsn, zpn, zdn).
    double zsn = 0.0;
    double zpn = 0.0;
    double zdn = 0.0;

    // Resonance integrals (eV)
    double betas = 0.0;    // β_s
    double betap = 0.0;    // β_p
    double betad = 0.0;    // β_d

    // Slater-Condon s-d exchange (MOPAC: f0sd, g2sd).
    double f0sd = 0.0;
    double g2sd = 0.0;

    // Core-core repulsion parameters
    double alpha = 0.0;    // α (Gaussian exponent)
    double pcore = 0.0;    // PM6/PM7 core parameter (MOPAC: poc_)
    double polvo = 0.0;    // atomic polarizability (MOPAC: polvo6)

    // Core Polarization Energy parameters (MOPAC: CPE_Zet, CPE_Z0,
    // CPE_B, CPE_Xlo, CPE_Xhi).
    double cpe_zet = 0.0;
    double cpe_z0  = 0.0;
    double cpe_b   = 0.0;
    double cpe_xlo = 0.0;
    double cpe_xhi = 0.0;
    bool has_cpe = false;

    // Multi-term Gaussian gamma expansion: Σ_i Σ_j c_i c_j / sqrt(R^2 + (d_i+d_j)^2)
    // (coeff, exponent, factor) triples.  Empty = use simple Ohno-Klopman.
    struct GammaTerm {
        double coeff = 0.0;
        double exponent = 0.0;  // d_i in Ohno-Klopman denominator
        double factor = 0.0;    // unused in basic form, available for extensions
    };
    std::vector<GammaTerm> gamma_terms;

    // Gaussian core-core correction terms (MOPAC PM6 guess1/guess2/guess3)
    // Sigma_i Z*Z/R * coeff_i * exp(-exponent_i * (R - factor_i)^2)
    // Each term: coeff (guess1), exponent (guess2), factor (guess3)
    struct GaussianTerm {
        double coeff = 0.0;     // guess1: prefactor
        double exponent = 0.0;  // guess2: Gaussian exponent
        double factor = 0.0;    // guess3: Gaussian center (distance offset)
    };
    std::vector<GaussianTerm> gaussian_terms;

    // Number of valence orbitals
    int n_orbitals = 4;    // default: s + 3*p = 4 (sp basis)
    bool has_d = false;

    // Add a Gaussian core-core correction term (convenience for pybind11)
    void add_gaussian_term(double coeff, double exponent, double factor) {
        gaussian_terms.push_back({coeff, exponent, factor});
    }
};

// ---------------------------------------------------------------------------
// NDDO diatomic parameters (per element pair)
// d1 = alpb (core-core damping exponent), d2 = xfac (core-core prefactor)

struct NDDODiatomicParams {
    double d1 = 0.0;       // alpb: pair-specific damping exponent
    double d2 = 0.0;       // xfac: pair-specific prefactor
};

// PM6 effective core charge / neutral-atom valence-electron count.
// This is the MOPAC ``tore = ios + iop + iod`` convention, not the number
// of AOs in the NDDO basis and not a generic periodic-table group count.
int pm6_core_charge(int Z);

// Whether an element record contains the finite, nonzero one-electron,
// Slater-exponent, and one-center two-electron fields required by the
// executable s-only (H/He) or s/p PM6 kernel and retained PM7 parameter
// registry. Upstream placeholder records such as At deliberately fail this
// predicate; PM7 execution remains gated on its distinct Hamiltonian.
bool pm6_has_complete_base_parameters(const NDDOElementData& element);
bool pm6_has_complete_sp_parameters(const NDDOElementData& element);

// ---------------------------------------------------------------------------
// PM6 parameter set
// ---------------------------------------------------------------------------

class PM6ParameterSet : public CoreParameterSet {
public:
    explicit PM6ParameterSet(std::string method_name = "pm6");

    // CoreParameterSet interface
    bool has_element(int Z) const override;
    const CoreElementData& element(int Z) const override;
    double repulsive_energy(int Z1, int Z2, double R) const override;
    double repulsive_derivative(int Z1, int Z2, double R) const override;
    int shell_principal_quantum_number(int Z, int l) const override;
    std::string method_name() const override { return method_name_; }
    std::string parameter_version() const override { return version_; }
    int n_elements() const override;

    // NDDO-specific access
    const NDDOElementData* element_data(int Z) const;
    // Diatomic pair access (alpb/xfac for core-core repulsion)
    const NDDODiatomicParams* diatomic(int Z1, int Z2) const;

    void add_element(const NDDOElementData& elem);
    void set_diatomic(int Z1, int Z2, const NDDODiatomicParams& dp);

    // Canonical identity of the authoritative NDDO parameter snapshot.
    // Derived CoreElementData views are deliberately excluded.
    virtual std::string content_sha256() const;
    virtual std::string parameter_identity() const;
    virtual ParameterSetMetadata metadata() const;

protected:
    // OMx owns an additional, authoritative parameter registry.  Its
    // synchronized add_omx_element() path uses this primitive while the
    // public PM6 mutator rejects cross-method use.
    void store_element(const NDDOElementData& elem);

private:
    std::map<int, NDDOElementData> elements_;
    std::map<int, NDDODiatomicParams> diatomic_;
    std::string method_name_;
    std::string version_;
    int pair_key(int Z1, int Z2) const;
};

}  // namespace nddo
}  // namespace semiempirical
}  // namespace vibeqc
