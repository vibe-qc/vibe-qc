// OMx parameter set (OM1, OM2, OM3).
//
// Orthogonalization-corrected NDDO methods from the Thiel group.
// OMx uses the same NDDO formalism as PM6 but with additional
// orthogonalization corrections that improve molecular energetics.
//
// References:
//   OM1: M. Kolb, W. Thiel, J. Comput. Chem. 1993, 14, 775.
//   OM2: W. Weber, W. Thiel, Theor. Chem. Acc. 2000, 103, 495.
//   OM3: M. Scholten, PhD thesis, Universität Düsseldorf, 2003.
//   Dral 2016: P. O. Dral et al., J. Chem. Theory Comput. 2016, 12, 1082.

#pragma once

#include <map>
#include <string>

#include "vibeqc/semiempirical/methods/nddo/pm6_parameters.hpp"

namespace vibeqc {
namespace semiempirical {
namespace nddo {

enum class OMxVariant { OM1, OM2, OM3 };

// ---------------------------------------------------------------------------
// OMx-specific element data
//
// Each OMx element carries the standard NDDO parameters (Uss, Upp, Gss, …)
// plus the additional parameters needed for the OM1/OM2/OM3 Hamiltonians:
// orbital exponent scale factor (zeta), resonance integral prefactors and
// exponents (beta_s/beta_p/beta_pi, alpha_s/alpha_p/alpha_pi),
// orthogonalization-correction factors (F1, F2, G1, G2), optional X-H
// pair-specific resonance parameters, and ECP parameters (OM2/OM3 only).
// ---------------------------------------------------------------------------

struct OMxElementData {
    // --- Core NDDO parameters (same semantics as NDDOElementData) ---
    int Z = 0;

    // One-center one-electron energies (eV)
    double uss = 0.0;
    double upp = 0.0;

    // One-center two-electron integrals (eV)
    double gss = 0.0;
    double gpp = 0.0;
    double gsp = 0.0;
    double gp2 = 0.0;
    double hsp = 0.0;

    // --- OMx-specific parameters ---

    // Orbital exponent scale factor (bohr^-1)
    // Effective zeta for overlap = zeta · ζ_standard
    double zeta = 0.0;

    // Resonance integral parameters (direct OMx form, eq 17)
    // β_type · exp(-α_type · R²)
    double beta_s  = 0.0;   // βs  (eV·bohr^{-1/2})
    double beta_p  = 0.0;   // βp  (eV·bohr^{-1/2})
    double beta_pi = 0.0;   // βπ  (eV·bohr^{-1/2})
    double alpha_s  = 0.0;  // αs  (bohr^{-2})
    double alpha_p  = 0.0;  // αp  (bohr^{-2})
    double alpha_pi = 0.0;  // απ  (bohr^{-2})

    // X-H pair resonance parameters (optional; 0.0 = use defaults)
    double beta_s_xh  = 0.0;
    double beta_p_xh  = 0.0;
    double alpha_s_xh = 0.0;
    double alpha_p_xh = 0.0;

    // Orthogonalization correction prefactors (dimensionless)
    double F1 = 0.0;  // first-order one-centre ORT (all OMx variants)
    double F2 = 0.0;  // second-order one-centre ORT (OM1/OM2)
    double G1 = 0.0;  // first-order three-centre ORT (OM2/OM3)
    double G2 = 0.0;  // second-order three-centre ORT (OM2 only)

    // ECP parameters (OM2/OM3 only)
    double zeta_alpha     = 0.0;   // Core STO exponent (a0^-1)
    double F_alpha_alpha  = 0.0;   // Fαα (eV)
    double beta_alpha     = 0.0;   // βα (eV·bohr^{-1/2})
    double alpha_alpha    = 0.0;   // αα (bohr^{-2})

    // Slater exponents for NDDO-style overlap (backward compat in a0^-1)
    double zs = 0.0;
    double zp = 0.0;

    // Core-core repulsion parameter
    double alpha = 0.0;

    // Number of valence orbitals
    int n_orbitals = 4;
};

// Whether a record contains the finite, nonzero fields required by the
// executable H or C/N/O/F OMx Hamiltonian for its fitted variant.
bool omx_has_complete_parameters(
    const OMxElementData& element,
    OMxVariant variant);

// ---------------------------------------------------------------------------
// OMx parameter set
//
// Inherits from PM6ParameterSet so shared NDDO basis machinery can use the
// synchronized element_data() view.  The authoritative OMx parameters are
// accessed via omx_data().
// ---------------------------------------------------------------------------

class OMxParameterSet : public PM6ParameterSet {
public:
    explicit OMxParameterSet(OMxVariant variant = OMxVariant::OM1);

    std::string method_name() const override;
    std::string parameter_version() const override;

    // Access OMx-specific element data
    const OMxElementData* omx_data(int Z) const;
    void add_omx_element(const OMxElementData& ed);

    std::string content_sha256() const override;
    std::string parameter_identity() const override;
    ParameterSetMetadata metadata() const override;
    OMxVariant variant() const { return variant_; }

private:
    OMxVariant variant_;
    std::map<int, OMxElementData> omx_elements_;
};

}  // namespace nddo
}  // namespace semiempirical
}  // namespace vibeqc
