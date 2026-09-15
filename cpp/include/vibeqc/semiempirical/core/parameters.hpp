// Shared semiempirical core — parameter interfaces.
//
// Phase 0, Step 2: Defines base classes for semiempirical parameters
// that method plugins (DFTB, GFN-xTB, PM6) extend with their
// specific data structures.

#pragma once

#include <map>
#include <string>
#include <vector>

namespace vibeqc {
namespace semiempirical {

// ---------------------------------------------------------------------------
// Base element data
// ---------------------------------------------------------------------------

struct CoreElementData {
    int Z = 0;                          // atomic number
    int valence_electrons = 0;          // neutral-atom valence count
    double hubbard_u = 0.0;             // chemical hardness (Ha)
    std::vector<double> on_site;        // on-site energies per shell (Ha)
    std::vector<double> zeta;           // STO exponents per shell (a0^-1)
};

// ---------------------------------------------------------------------------
// Base parameter set interface
// ---------------------------------------------------------------------------

class CoreParameterSet {
public:
    virtual ~CoreParameterSet() = default;

    // Element queries
    virtual bool has_element(int Z) const = 0;
    virtual const CoreElementData& element(int Z) const = 0;

    // Principal quantum number of a shell (n of the ao=2s2p-style record).
    // The default n = l + 1 preserves the historical l-only STO-NG row
    // selection for parameter sets that do not carry n per shell.
    virtual int shell_principal_quantum_number(int Z, int l) const {
        (void)Z;
        return l + 1;
    }

    // Repulsive pair potential — must be overridden by each method
    virtual double repulsive_energy(int Z1, int Z2, double R) const = 0;
    virtual double repulsive_derivative(int Z1, int Z2, double R) const = 0;

    // Method metadata
    virtual std::string method_name() const = 0;
    virtual std::string parameter_version() const = 0;

    // Element count for coverage reporting
    virtual int n_elements() const = 0;
};

// ---------------------------------------------------------------------------
// Parameter set metadata (for registry)
// ---------------------------------------------------------------------------

struct ParameterSetMetadata {
    std::string method_name;
    std::string version;
    std::string origin;          // "published", "in-house", "refit-from-X"
    std::string license;
    int n_elements = 0;
    std::vector<int> element_list;
    std::string doi_or_url;      // if published parameter set
    std::string fit_dataset;     // if refitted
    std::string parameter_hash;  // SHA-256 of parameter file
};

}  // namespace semiempirical
}  // namespace vibeqc
