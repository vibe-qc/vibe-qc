// GFN2-xTB parameter set.
//
// Reference: C. Bannwarth, S. Ehlert, S. Grimme,
// J. Chem. Theory Comput. 2019, 15, 1652-1671.
// DOI: 10.1021/acs.jctc.8b01176

#pragma once

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "vibeqc/semiempirical/core/parameters.hpp"

namespace vibeqc {
namespace semiempirical {
namespace xtb {

// ---------------------------------------------------------------------------
// GFN2-xTB per-element data
// ---------------------------------------------------------------------------

struct GFN2ElementData {
    int Z;
    CoreElementData base;

    struct ShellData {
        int n = 0;
        int l = 0;
        double en = 0.0;       // electronegativity (Ha)
        double zeta = 0.0;     // STO exponent
        double k_en = 1.0;     // EN scaling
        double kcn = 0.0;      // CN self-energy shift (KCNS/KCNP/KCND, eV→Ha)
        double poly = 0.0;     // shell distance-polynomial (POLYS/POLYP/POLYD, ×0.01)
    };
    std::vector<ShellData> shells;

    double gam = 0.5;          // chemical hardness
    double gam3 = 0.0;         // 3rd-order Hubbard (GAM3)
    double alpha = 1.0;        // charge scaling

    // --- Anisotropic electrostatics (AES) per-element parameters ---
    // Faithful GFN2 AES (Bannwarth, Ehlert & Grimme, JCTC 2019, 15, 1652,
    // doi:10.1021/acs.jctc.8b01176).  Consumed only by the experimental
    // faithful-AES path (XTBSccOptions::aes_faithful); 0 by default so the
    // shipped ad-hoc shell-resolved AES path is unaffected.
    double dpol = 0.0;    // dipole on-site XC kernel   = DPOL × 0.01  (param file)
    double qpol = 0.0;    // quadrupole on-site XC kernel= QPOL × 0.01  (param file)
    double mp_rad = 0.0;  // multipole radius  (tblite p_rad, bohr)
    double mp_vcn = 0.0;  // multipole valence coordination number (tblite p_vcn)
};

// ---------------------------------------------------------------------------
// GFN2-xTB repulsive pair data
// ---------------------------------------------------------------------------

struct GFN2RepulsivePair {
    double alpha = 1.0;
    double k_ab = 0.0;
};

// ---------------------------------------------------------------------------
// GFN2-xTB parameter set
// ---------------------------------------------------------------------------

class GFN2ParameterSet : public CoreParameterSet {
public:
    GFN2ParameterSet();

    // Dispersion parameters (GFN2's D4 global parameters, s6=1.0, s8=2.7,
    // a1=0.52, a2=5.0, s9=5.0 per Bannwarth, Ehlert & Grimme, JCTC 2019 /
    // live xtb globpar).  NOTE: the molecular energy path applies D4
    // post-SCF from Python (vibeqc.semiempirical.methods.gfn2 ->
    // vibeqc.dispersion_d4_parameters "gfn2xtb" row), so these members are
    // informational/read-write plumbing, not the live energy values.  Keep
    // them in sync with the Python table (s9=5.0, the published ATM
    // scaling) so nothing downstream can silently pick up a stale s9=1.0.
    double d4_s8 = 2.7;
    double d4_s9 = 5.0;
    double d4_a1 = 0.52;
    double d4_a2 = 5.0;

    // CoreParameterSet interface
    bool has_element(int Z) const override;
    const CoreElementData& element(int Z) const override;
    int shell_principal_quantum_number(int Z, int l) const override;
    double repulsive_energy(int Z1, int Z2, double R) const override;
    double repulsive_derivative(int Z1, int Z2, double R) const override;
    std::string method_name() const override { return "gfn2-xtb"; }
    std::string parameter_version() const override { return version_; }
    int n_elements() const override;

    // GFN2-specific
    const GFN2ElementData* element_data(int Z) const;
    const GFN2RepulsivePair* repulsive_pair(int Z1, int Z2) const;

    // Placeholder: add elements as needed during implementation
    void add_element(const GFN2ElementData& elem);
    void set_repulsive_pair(int Z1, int Z2, const GFN2RepulsivePair& rp);

    // Canonical identity of the authoritative, natively executed GFN2
    // parameter snapshot.  The derived CoreElementData projection is
    // validated but not hashed; the informational d4_* compatibility fields
    // are neither executed by the native model nor hashed here.
    std::string content_sha256() const;
    std::string parameter_identity() const;

    // Metadata
    ParameterSetMetadata metadata() const;

private:
    std::map<int, GFN2ElementData> elements_;
    std::map<int, GFN2RepulsivePair> repulsive_pairs_;
    std::string version_ = "gfn2-xtb-2019";
    int pair_key(int Z1, int Z2) const;
};

}  // namespace xtb
}  // namespace semiempirical
}  // namespace vibeqc
