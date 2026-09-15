#include "vibeqc/semiempirical/basis.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

#include "vibeqc/basis.hpp"
#include "vibeqc/molecule.hpp"

namespace vibeqc {
namespace semiempirical {

// ---------------------------------------------------------------------------
// SemiempiricalBasis::sto6g_table
//
// The Gaussian exponents α_i are unscaled; for a Slater exponent ζ,
// the actual exponent is α_i × ζ².
//
// PROVENANCE WARNING: these tables were long cited as Hehre, Stewart &
// Pople, J. Chem. Phys. 51, 2657 (1969) Table I (1s) / Table IV (2p),
// but they do not reproduce HSP's own published least-squares residuals
// (off by 5e4x for l=0 and 4e5x for l=1) and several exponents are
// hand-rounded numbers. The true source is unknown. DFTB0/SCC-DFTB run
// entirely on these (n_primitives=6). Tracked as
// DFTB-STO-NG-TABLES-NOT-HSP in agentic-loop/bug-claims.md; see the
// per-table notes below and the long comment in build().
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// STO-NG expansion tables from Hehre, Stewart & Pople, J. Chem. Phys. 51,
// 2657 (1969) Tables I-V (the same rows xtb's slater.f90 pAlpha/pCoeff arrays
// carry; Stewart, J. Chem. Phys. 52, 431 (1970) re-derived the same rows to
// higher precision, and the GFN2 paper cites Stewart).  All exponents are
// unscaled; for a Slater exponent ζ, the actual exponent is α_i × ζ².
//
// Rows are keyed by the shell's principal quantum number n and angular
// momentum l: xtb's slaterToGauss selects the row by n (ityp = n for s,
// 4+n for p, 7+n for d, 9+n for f, 10+n for g).  Keying by l alone (always
// the lowest-n row) silently gave the O 2s shell the 1s expansion, shifting
// the H0 spectrum by several eV (GFN2-MOL-PARITY).  Exponents stored
// ascending; coefficients are the raw contraction weights over
// unity-normalized primitives (libint renorm convention,
// coefficients_pre_normalized = false).
// ---------------------------------------------------------------------------

namespace {

struct StoNgRow {
    std::vector<double> alphas;
    std::vector<double> coeffs;
};

// (n, l) → 3G row.  Only 1s is consumed by GFN2 (H, He); the remaining rows
// mirror xtb slater.f90 for API completeness.  Exponents stored ascending;
// coefficients are the raw contraction weights in the matching ascending
// pairing (xtb's tables list both descending).
const StoNgRow& sto3g_row(int n, int l) {
    static const StoNgRow s1 = {
        {0.109818, 0.405771, 2.22766},
        {0.444635, 0.535328, 0.154329}};
    static const StoNgRow s2 = {
        {0.0601833, 0.156762, 2.58158},
        {0.458179, 0.596039, -0.0599447}};
    static const StoNgRow s3 = {
        {0.0326953, 0.0692442, 0.564149},
        {0.226184, 0.861276, -0.178258}};
    static const StoNgRow s4 = {
        {0.0219529, 0.0444818, 0.226794},
        {0.125666, 1.05674, -0.334905}};
    static const StoNgRow s5 = {
        {0.0261081, 0.0440812, 0.108020},
        {0.714649, 0.746760, -0.661740}};
    static const StoNgRow p2 = {
        {0.0800981, 0.235919, 0.919238},
        {0.422307, 0.566171, 0.162395}};
    static const StoNgRow p3 = {
        {0.0573959, 0.148936, 2.69288},
        {0.545002, 0.521856, -0.0106195}};
    static const StoNgRow p4 = {
        {0.0365334, 0.0743022, 0.485969},
        {0.393264, 0.660417, -0.0614782}};
    static const StoNgRow p5 = {
        {0.0260487, 0.0472965, 0.212748},
        {0.272603, 0.807669, -0.138953}};
    static const StoNgRow d3 = {
        {0.0638663, 0.163960, 0.522911},
        {0.405678, 0.584798, 0.168660}};
    static const StoNgRow d4 = {
        {0.0394986, 0.0804065, 0.177772},
        {0.259577, 0.604241, 0.230855}};
    static const StoNgRow d5 = {
        {0.0359421, 0.0732909, 0.491335},
        {0.465845, 0.589937, -0.0201018}};
    static const StoNgRow f4 = {
        {0.0535000, 0.124938, 0.348383},
        {0.392940, 0.597338, 0.173786}};
    static const StoNgRow f5 = {
        {0.0373579, 0.0748707, 0.164923},
        {0.305961, 0.614606, 0.190973}};
    static const StoNgRow g5 = {
        {0.0462446, 0.100654, 0.254543},
        {0.382855, 0.606376, 0.178098}};
    switch (l) {
        case 0:
            switch (n) {
                case 1: return s1; case 2: return s2; case 3: return s3;
                case 4: return s4; case 5: return s5;
            }
            break;
        case 1:
            switch (n) {
                case 2: return p2; case 3: return p3; case 4: return p4;
                case 5: return p5;
            }
            break;
        case 2:
            switch (n) {
                case 3: return d3; case 4: return d4; case 5: return d5;
            }
            break;
        case 3:
            switch (n) {
                case 4: return f4; case 5: return f5;
            }
            break;
        case 4:
            if (n == 5) return g5;
            break;
    }
    throw std::runtime_error(
        "STO-3G: no (n, l) row for n=" + std::to_string(n)
        + " l=" + std::to_string(l));
}

// (n, l) → 4G row.
const StoNgRow& sto4g_row(int n, int l) {
    static const StoNgRow s1 = {
        {0.0880186, 0.265203, 0.954618, 5.21684},
        {0.291625, 0.532846, 0.260141, 0.0567524}};
    static const StoNgRow s2 = {
        {0.0612574, 0.160728, 2.00024, 11.6153},
        {0.477008, 0.580559, -0.0547205, -0.0119841}};
    static const StoNgRow s3 = {
        {0.0376055, 0.0764332, 0.426250, 1.51327},
        {0.358963, 0.751851, -0.172452, -0.0329550}};
    static const StoNgRow s4 = {
        {0.0282907, 0.0508110, 0.166322, 0.324221},
        {0.351781, 0.890987, -0.284543, -0.112068}};
    static const StoNgRow s5 = {
        {0.0197480, 0.0344608, 0.118905, 0.860228},
        {0.173497, 1.17943, -0.560652, 0.0110366}};
    static const StoNgRow p2 = {
        {0.0654393, 0.164372, 0.466262, 1.79826},
        {0.263231, 0.551787, 0.285746, 0.0571317}};
    static const StoNgRow p3 = {
        {0.0418425, 0.0865549, 0.191508, 1.85318},
        {0.214499, 0.584675, 0.275518, -0.0143425}};
    static const StoNgRow p4 = {
        {0.0370627, 0.0755316, 0.432762, 1.49261},
        {0.411792, 0.645152, -0.0601331, -0.00603522}};
    static const StoNgRow p5 = {
        {0.0275022, 0.0494356, 0.183886, 0.396284},
        {0.340930, 0.753397, -0.136078, -0.0180146}};
    static const StoNgRow d3 = {
        {0.0528676, 0.118757, 0.292046, 0.918585},
        {0.243242, 0.560136, 0.304558, 0.0579906}};
    static const StoNgRow d4 = {
        {0.0400063, 0.0819724, 0.182346, 1.99583},
        {0.271781, 0.605805, 0.217710, -0.00281670}};
    static const StoNgRow d5 = {
        {0.0262874, 0.0459033, 0.0829386, 0.423062},
        {0.119044, 0.548952, 0.393764, -0.0242163}};
    static const StoNgRow f4 = {
        {0.0447351, 0.0929835, 0.207459, 0.569167},
        {0.228480, 0.563942, 0.319183, 0.0590273}};
    static const StoNgRow f5 = {
        {0.0303757, 0.0544701, 0.100195, 0.201783},
        {0.125400, 0.493743, 0.402350, 0.0917427}};
    static const StoNgRow g5 = {
        {0.0389870, 0.0764652, 0.158810, 0.394521},
        {0.217112, 0.565521, 0.330974, 0.0601048}};
    switch (l) {
        case 0:
            switch (n) {
                case 1: return s1; case 2: return s2; case 3: return s3;
                case 4: return s4; case 5: return s5;
            }
            break;
        case 1:
            switch (n) {
                case 2: return p2; case 3: return p3; case 4: return p4;
                case 5: return p5;
            }
            break;
        case 2:
            switch (n) {
                case 3: return d3; case 4: return d4; case 5: return d5;
            }
            break;
        case 3:
            switch (n) {
                case 4: return f4; case 5: return f5;
            }
            break;
        case 4:
            if (n == 5) return g5;
            break;
    }
    throw std::runtime_error(
        "STO-4G: no (n, l) row for n=" + std::to_string(n)
        + " l=" + std::to_string(l));
}

}  // namespace

// (n, l) → 6G rows for the GFN2 auto path (xtb setGFN2NumberOfPrimitives
// assigns 6 primitives to n > 5 shells; slater.f90 pAlpha6s/pCoeff6s,
// pAlpha6p/pCoeff6p).  Exponents ascending, coefficients in the matching
// ascending pairing.
const StoNgRow& sto6g_gfn2_row(int n, int l) {
    static const StoNgRow s6 = {
        {0.0188607, 0.0298364, 0.0497509, 0.0793852, 0.271826, 0.580029},
        {0.362252, 1.33249, -0.226980, -0.756102, 0.0528644, 0.00455436}};
    static const StoNgRow p6 = {
        {0.0188222, 0.0296131, 0.0458633, 0.0816389, 0.139509, 0.669654},
        {0.109153, 0.675205, 0.468226, -0.226626, -0.128289, 0.00278272}};
    if (n == 6 && l == 0) return s6;
    if (n == 6 && l == 1) return p6;
    throw std::runtime_error(
        "STO-6G: no GFN2 (n, l) row for n=" + std::to_string(n)
        + " l=" + std::to_string(l));
}

const SemiempiricalBasis::StoNgTable& SemiempiricalBasis::sto6g_table(int l) {
    if (l == 0) {
        // STO-6G for 1s. UNVERIFIED PROVENANCE -- cited as HSP (1969)
        // Table I, but fails HSP's own published residual by 52839x
        // (eps = 6.55e-2 vs 1.24e-6). See DFTB-STO-NG-TABLES-NOT-HSP.
        static const StoNgTable s_tbl = {
            {0.168856, 0.623913, 3.42525, 8.420, 18.0, 35.5232},
            {0.334041, 0.534205, 0.154815, 0.0203930, 0.00198418, 0.000102952}
        };
        return s_tbl;
    }
    if (l == 1) {
        // STO-6G for 2p. UNVERIFIED PROVENANCE -- cited as HSP (1969)
        // Table IV, but fails HSP's own published residual by 442028x
        // (eps = 5.39e-1 vs 1.22e-6). See DFTB-STO-NG-TABLES-NOT-HSP.
        static const StoNgTable p_tbl = {
            {0.181494, 0.663712, 3.62252, 8.404, 18.0, 35.5232},
            {0.137008, 0.504421, 0.306884, 0.0607796, 0.00650756, 0.000329759}
        };
        return p_tbl;
    }
    if (l == 2) {
        // STO-6G for 3d. UNVERIFIED PROVENANCE -- cited as HSP (1969)
        // Table V; exponents 12.553/30.0/69.864 are hand-rounded, not
        // fitted values. See DFTB-STO-NG-TABLES-NOT-HSP.
        static const StoNgTable d_tbl = {
            {0.334044, 1.22301, 6.14927, 12.553, 30.0, 69.864},
            {0.132104, 0.485118, 0.326919, 0.0747647, 0.00733498, 0.000335051}
        };
        return d_tbl;
    }
    if (l == 3) {
        // STO-6G for 4f. UNVERIFIED PROVENANCE -- exponents 18.0/40.0/
        // 90.0 are hand-rounded, not fitted values; HSP (1969) does not
        // tabulate 4f. See DFTB-STO-NG-TABLES-NOT-HSP.
        static const StoNgTable f_tbl = {
            {0.598801, 1.973182, 8.493242, 18.0, 40.0, 90.0},
            {0.159852, 0.504783, 0.319984, 0.0704716, 0.00664587, 0.000334108}
        };
        return f_tbl;
    }
    throw std::runtime_error(
        "SemiempiricalBasis: STO-6G tables only available for l=0,1,2,3. "
        "Got l=" + std::to_string(l));
}

// ---------------------------------------------------------------------------
// STO-NG expansion
// ---------------------------------------------------------------------------

std::pair<std::vector<double>, std::vector<double>>
SemiempiricalBasis::sto_ng_expansion(double zeta, int n, int l, int n_primitives) {
    const StoNgRow* row = nullptr;
    const StoNgTable* tbl6 = nullptr;
    switch (n_primitives) {
        case 3: row = &sto3g_row(n, l); break;
        case 4: row = &sto4g_row(n, l); break;
        default:
            // n == 6: GFN2 auto-path rows (HSP sixth row); n < 6: legacy
            // l-only DFTB tables (unchanged, see sto6g_table).
            row = (n == 6) ? &sto6g_gfn2_row(n, l) : nullptr;
            tbl6 = (n == 6) ? nullptr : &sto6g_table(l);
            break;
    }

    std::vector<double> exponents;
    std::vector<double> coefficients;
    if (row) {
        exponents = row->alphas;
        coefficients = row->coeffs;
    } else {
        exponents = tbl6->alphas;
        coefficients = tbl6->coeffs;
    }

    const double zeta2 = zeta * zeta;
    for (double& a : exponents) a *= zeta2;
    return {exponents, coefficients};
}

// ---------------------------------------------------------------------------
// Valence shell enumeration
// ---------------------------------------------------------------------------

std::vector<ValenceShell> SemiempiricalBasis::valence_shells(
    const Molecule& mol, const SemiempiricalParameters& params) {
    std::vector<ValenceShell> shells;

    for (std::size_t a = 0; a < mol.atoms().size(); ++a) {
        int Z = mol.atoms()[a].Z;
        if (!params.has_element(Z)) {
            throw std::runtime_error(
                "SemiempiricalBasis: element Z=" + std::to_string(Z)
                + " not in parameter set");
        }

        const auto& elem = params.element(Z);

        // Determine which angular momenta are present based on on_site list.
        for (std::size_t l = 0; l < elem.on_site.size(); ++l) {
            bool has_shell = false;
            if (elem.on_site[l] != 0.0) {
                has_shell = true;
            }
            if (l < elem.zeta.size() && elem.zeta[l] > 0.0) {
                has_shell = true;
            }
            if (has_shell) {
                ValenceShell vs;
                vs.atom_index = static_cast<int>(a);
                vs.l = static_cast<int>(l);
                vs.pure = (l > 0);  // spherical harmonics for p, d, ...
                shells.push_back(vs);
            }
        }

        // Safety: if no shells were added, add at least one s.
        if (shells.empty() || shells.back().atom_index != static_cast<int>(a)) {
            ValenceShell vs;
            vs.atom_index = static_cast<int>(a);
            vs.l = 0;
            vs.pure = false;
            shells.push_back(vs);
        }
    }

    return shells;
}

// ---------------------------------------------------------------------------
// Build ShellInfo list
// ---------------------------------------------------------------------------

std::vector<ShellInfo> SemiempiricalBasis::build_shell_info(
    const Molecule& mol, const SemiempiricalParameters& params,
    int n_primitives) {
    std::vector<ShellInfo> result;

    for (const auto& vs : valence_shells(mol, params)) {
        int Z = mol.atoms()[vs.atom_index].Z;
        double zeta = params.sto_exponent(Z, vs.l);

        // Fallback zeta for elements with undefined exponents
        if (zeta <= 0.0) {
            if (Z == 1) zeta = 1.0;         // H 1s
            else if (Z <= 2) zeta = 1.7;    // He
            else if (Z <= 10) zeta = 1.625; // Li-Ne, 2s/2p average
            else zeta = 1.0;
        }

        auto [exponents, coeffs] = sto_ng_expansion(
            zeta, vs.l + 1, vs.l, n_primitives);

        ShellInfo si;
        si.atom_index = vs.atom_index;
        si.l = vs.l;
        si.pure = vs.pure;
        si.exponents = std::move(exponents);
        si.coefficients = std::move(coeffs);
        si.origin = mol.atoms()[vs.atom_index].xyz;

        result.push_back(std::move(si));
    }

    return result;
}

// ---------------------------------------------------------------------------
// Build full BasisSet
// ---------------------------------------------------------------------------

BasisSet SemiempiricalBasis::build(const Molecule& mol,
                                    const SemiempiricalParameters& params,
                                    int n_primitives) {
    auto shells = build_shell_info(mol, params, n_primitives);
    // coefficients_pre_normalized=false: the tabulated d_i are contraction
    // weights over ALREADY-normalized primitives. Eq. (2.3) of Hehre,
    // Stewart & Pople, J. Chem. Phys. 51, 2657 (1969) defines
    //   g_1s(alpha,r) = (2*alpha/pi)^(3/4) exp(-alpha*r^2)
    //   g_2p(alpha,r) = (128*alpha^5/pi^3)^(1/4) r exp(-alpha*r^2) cos(theta)
    // (both verified unity-normalized by quadrature), and Eq. (2.2) expands
    // the STO as phi' = sum_k d_k * g(alpha_k, r) over those. That is exactly
    // libint2's documented "EMSL Gaussian Basis Set Database" convention
    // (libint2::Shell doc comment, third_party/libint/.../shell.h): such d_i
    // require Shell::renorm(), i.e. embed_normalization_into_coefficients
    // = true, which is coefficients_pre_normalized = false here.
    //
    // Do NOT flip this. fbb23f74d set it to true, which skips renorm()
    // entirely -- losing BOTH the primitive normalization AND libint's
    // enforce-unit-normalization step. Measured on H2O/dftb0_default, the
    // overlap diagonal went from exactly 1 (max|S_uu - 1| = 4.4e-16) to
    //   diag(S) = [0.647, 0.0285, 0.0285, 0.0285, 3.847, 3.847]
    // i.e. a 135x inconsistency between O 2p and H 1s.
    //
    // The STO-3G and STO-4G tables now match Stewart (1970) as
    // cross-verified against xtb slater.f90 (bdef7c15c) -- verified here
    // by quadrature: eps = 3.31e-4 / 2.69e-4 (3G s/p) and 4.38e-5 /
    // 2.90e-5 (4G s/p), self-overlap 1.000000 throughout.
    //
    // The STO-6G tables are NOT from HSP (1969) Table I/IV/V despite the
    // comment on each of them. HSP publishes the least-squares residual
    // eps per expansion, which makes that provenance claim mechanically
    // checkable, and the shipped values fail it by 52839x (l=0, eps =
    // 6.55e-2 vs HSP's 1.24e-6) and 442028x (l=1, eps = 5.39e-1 vs
    // 1.22e-6) -- i.e. worse Slater fits than STO-3G despite twice the
    // primitives. Not a zeta-convention artifact: contraction
    // self-overlap is invariant under uniform exponent scaling and is
    // 0.832 (l=0) / 0.650 (l=1) instead of 1. Not an in-house fit to
    // some other target either: Cauchy-Schwarz floors the error against
    // EVERY normalized target at >=6215x / >=30869x.
    //
    // This matters here because DFTB0/SCC-DFTB build with
    // n_primitives=6 on every path, so the DFTB stack runs entirely on
    // those two tables. Correcting them is NOT a drop-in change -- it
    // shifts every DFTB energy and the repulsive parameters may be
    // calibrated against the current values, exactly as bdef7c15c's own
    // GFN2 result showed (9 test_gfn2_xtb.py failures pending
    // recalibration). Tracked as DFTB-STO-NG-TABLES-NOT-HSP in
    // agentic-loop/bug-claims.md; needs a maintainer decision, not a
    // unilateral table swap.
    return BasisSet(mol, shells, "dftb0-sto-ng", false);
}


// ---------------------------------------------------------------------------
// CoreParameterSet overloads
// ---------------------------------------------------------------------------

BasisSet SemiempiricalBasis::build(const Molecule& mol,
                                    const CoreParameterSet& params,
                                    int n_primitives) {
    auto shells = build_shell_info(mol, params, n_primitives);
    // coefficients_pre_normalized=false — same rationale as the
    // SemiempiricalParameters::build above.  The tabulated GFN2 STO-NG
    // coefficients are contraction weights over unity-normalized primitives
    // per HSP Eq. (2.3); embedding the normalization is libint2's documented
    // convention.  GFN2 draws from the same raw HSP tables, so the same
    // normalization applies.  Empirical validation: this resolves the
    // ~1e-3 Ha/bohr analytic-vs-FD gradient residual that was causing
    // H2CO GFN2-xTB geometry optimization to stall.
    return BasisSet(mol, shells, "gfn2-sto-ng", false);
}

std::vector<ShellInfo> SemiempiricalBasis::build_shell_info(
    const Molecule& mol, const CoreParameterSet& params, int n_primitives) {
    std::vector<ShellInfo> result;
    const auto& atoms = mol.atoms();

    for (std::size_t a = 0; a < atoms.size(); ++a) {
        int Z = atoms[a].Z;
        if (!params.has_element(Z)) {
            throw std::runtime_error(
                "SemiempiricalBasis: element Z=" + std::to_string(Z)
                + " not in parameter set");
        }
        const auto& elem = params.element(Z);
        int n_on_site = static_cast<int>(elem.on_site.size());
        int n_zeta = static_cast<int>(elem.zeta.size());
        int n_shells = std::max(n_on_site, n_zeta);

        for (int l = 0; l < n_shells; ++l) {
            double zeta_val = (l < n_zeta) ? elem.zeta[l] : 0.0;
            if (zeta_val <= 0.0) continue;

            bool pure = (l > 0);
            // The shell's principal quantum number selects the STO-NG row
            // (xtb slaterToGauss keys on n).  The pre-fix l-only selection
            // gave every s shell the 1s row, e.g. O 2s.
            const int n_qn = params.shell_principal_quantum_number(Z, l);
            // n_primitives == 0 is the GFN2-xTB "auto" sentinel: xtb's
            // setGFN2NumberOfPrimitives (xtb src/xtb/gfn2.f90) assigns
            // the primitive count per element AND shell, so it must be
            // resolved here where the per-atom Z and shell n are in
            // scope. Faithful rule: every d shell expands with 3
            // primitives and f with 4, independent of the element; s/p
            // follow the element rule (H, He: s -> 3, p -> 4; heavier
            // elements: 6 when n > 5, else 4). The previous element-only
            // rule gave d shells 4 primitives (issue #43 parity family).
            int np;
            if (n_primitives == 0) {
                if (l == 2) {
                    np = 3;
                } else if (l == 3) {
                    np = 4;
                } else if (Z <= 2) {
                    np = (l == 0) ? 3 : 4;
                } else {
                    np = (n_qn > 5) ? 6 : 4;
                }
            } else {
                np = n_primitives;
            }
            auto [exponents, coefficients] =
                sto_ng_expansion(zeta_val, n_qn, l, np);
            result.push_back(
                ShellInfo{static_cast<int>(a), l, pure, exponents, coefficients,
                          {atoms[a].xyz[0], atoms[a].xyz[1], atoms[a].xyz[2]}});
        }
    }
    return result;
}

}  // namespace semiempirical
}  // namespace vibeqc
