#include "vibeqc/semiempirical/basis.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

#include "vibeqc/basis.hpp"
#include "vibeqc/molecule.hpp"

namespace vibeqc {
namespace semiempirical {

// ---------------------------------------------------------------------------
// STO-NG expansions. Every table below is keyed by the shell's principal
// quantum number n and angular momentum l, and every row is verified by
// quadrature against the least-squares residual its source publishes (see
// tests/test_semiempirical_sto_ng_tables.py). The Gaussian exponents alpha_i
// are unscaled; for a Slater exponent zeta the actual exponent is
// alpha_i * zeta^2 (Stewart 1970, Eq. 4).
//
// History: until 2026-09-19 the DFTB0 / SCC-DFTB path (n_primitives = 6)
// drew from two l-keyed "STO-6G" tables of unknown provenance that failed
// HSP's published residual by 5e4x (l = 0) and 4e5x (l = 1) and, being keyed
// by l alone, gave every 2s shell the 1s row. Measured on the shipped set:
// the C 2p-2p pi overlap at the graphene bond length was 0.016 where the
// Slater value is 0.243, the H 1s-1s overlap at 1.4 bohr 0.50 instead of
// 0.66, so hopping was almost absent and Si and h-BN came out gapless.
// Those tables are gone; the n_primitives = 6 path now uses Stewart's rows.
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

// (n, l) -> 6G row. n = 1..5: Stewart, J. Chem. Phys. 52, 431 (1970),
// Tables I-V, N = 6 (transcribed 2026-09-19 from the paper and verified
// row by row against the printed least-squares residual eps, all fifteen to
// four significant digits). n = 6: the sixth-row s and p expansions xtb
// slater.f90 carries (pAlpha6s/pCoeff6s, pAlpha6p/pCoeff6p; Stewart does
// not tabulate n = 6), used by the GFN2 auto path for n > 5 shells and by
// the DFTB path for period-6 valence s/p. Exponents ascending, coefficients
// in the matching pairing. No row exists for n = 7, so period-7 elements
// have no STO-6G expansion and the DFTB basis refuses them (see
// SemiempiricalParameters::shell_principal_quantum_number).
const StoNgRow& sto6g_row(int n, int l) {
    // 1s: Stewart Table I, eps = 1.2372e-06
    static const StoNgRow s1 = {
        {0.06510953954, 0.1580884151, 0.4070988982, 1.185056519, 4.235915534, 23.10303149},
        {0.1303340841, 0.4164915298, 0.3705627997, 0.1685383049, 0.04936149294, 0.00916359628}};
    // 2s: Stewart Table I, eps = 3.2726e-07
    static const StoNgRow s2 = {
        {0.04416183978, 0.09260298399, 0.2040335729, 1.42678605, 5.077140627, 27.68496241},
        {0.1712994697, 0.5621061301, 0.3346271174, -0.05150303337, -0.02067024148, -0.004151277819}};
    // 2p: Stewart Table I, eps = 5.4444e-07
    static const StoNgRow p2 = {
        {0.04948220127, 0.1046655969, 0.2288932733, 0.5475665231, 1.530329631, 5.868285913},
        {0.1051855189, 0.4012362861, 0.4049863191, 0.189840006, 0.05144104825, 0.007924233646}};
    // 3s: Stewart Table II, eps = 4.0662e-08
    static const StoNgRow s3 = {
        {0.02724741144, 0.04797373812, 0.08636686991, 0.3593349765, 0.9200611311, 3.273031938},
        {0.07223633674, 0.501535102, 0.5534527651, -0.1587856086, -0.05639325779, -0.006775596947}};
    // 3p: Stewart Table II, eps = 7.9285e-08
    static const StoNgRow p3 = {
        {0.03315424265, 0.06076408893, 0.1131741848, 0.2248434849, 1.34078694, 5.077973607},
        {0.07411456232, 0.390881305, 0.4485358256, 0.163939577, -0.0141948834, -0.00332992984}};
    // 3d: Stewart Table II, eps = 2.8899e-07
    static const StoNgRow d3 = {
        {0.04058484363, 0.07877734732, 0.1559114463, 0.331132749, 0.7981487853, 2.488296923},
        {0.08902827546, 0.3843100204, 0.4266269092, 0.2072139149, 0.05386799363, 0.007283828112}};
    // 4s: Stewart Table III, eps = 5.9700e-09
    static const StoNgRow s4 = {
        {0.01874093091, 0.03163400284, 0.05277666487, 0.1717905487, 0.3605788802, 3.232838646},
        {0.02487178756, 0.4389247988, 0.7812787397, -0.3130627309, -0.08666390043, 0.001374817488}};
    // 4p: Stewart Table III, eps = 1.3897e-08
    static const StoNgRow p4 = {
        {0.02816929784, 0.04958248334, 0.08847434525, 0.341554138, 0.7960947826, 2.389722618},
        {0.1204970491, 0.5433958189, 0.4053115554, -0.05958513378, -0.01657464971, -0.001665913575}};
    // 4d: Stewart Table III, eps = 8.0888e-08
    static const StoNgRow d4 = {
        {0.03232628887, 0.0590419037, 0.1101467943, 0.2209593028, 1.341648295, 4.63423942},
        {0.1081250196, 0.4621672517, 0.415964693, 0.1108670481, -0.003566777891, -0.0004749842876}};
    // 4f: Stewart Table IV, eps = 1.7590e-07
    static const StoNgRow f4 = {
        {0.03474556673, 0.06350097171, 0.1173146814, 0.2296565064, 0.5004907278, 1.357718039},
        {0.07787514504, 0.3688112625, 0.4411388883, 0.2217065797, 0.05634263745, 0.006930234381}};
    // 5s: Stewart Table IV, eps = 7.9988e-09
    static const StoNgRow s5 = {
        {0.0221355843, 0.03669584901, 0.1061070594, 0.1847926858, 0.5077878915, 1.410128298},
        {0.3103964343, 1.08761949, -0.520067356, -0.09588628125, 0.01850157487, 0.002695439582}};
    // 5p: Stewart Table IV, eps = 3.9301e-09
    static const StoNgRow p5 = {
        {0.02063815019, 0.03328911801, 0.05404070736, 0.1683175469, 0.3499121109, 3.778623374},
        {0.06021665516, 0.476880814, 0.5706134877, -0.1381051233, -0.02920771322, 0.0001163246387}};
    // 5d: Stewart Table V, eps = 1.4086e-08
    static const StoNgRow d5 = {
        {0.02108227374, 0.0339120283, 0.05472831774, 0.0920430884, 0.3410838409, 0.8820520428},
        {0.02623132212, 0.2654483467, 0.5097437054, 0.2648458555, -0.02508271857, -0.004097377019}};
    // 5f: Stewart Table V, eps = 8.4380e-09
    static const StoNgRow f5 = {
        {0.02598071843, 0.04351355997, 0.07290318381, 0.1269485744, 0.2372312347, 1.33409684},
        {0.05077063693, 0.3205010548, 0.4589112231, 0.2373699784, 0.04624275998, -0.0009486751531}};
    // 5g: Stewart Table V, eps = 1.1806e-07
    static const StoNgRow g5 = {
        {0.03057364464, 0.05340032759, 0.09373643151, 0.172791706, 0.3497184772, 0.8574668996},
        {0.06974153145, 0.3552053926, 0.4512983737, 0.2339955227, 0.0587414517, 0.006729778096}};
    static const StoNgRow s6 = {
        {0.0188607, 0.0298364, 0.0497509, 0.0793852, 0.271826, 0.580029},
        {0.362252, 1.33249, -0.226980, -0.756102, 0.0528644, 0.00455436}};
    static const StoNgRow p6 = {
        {0.0188222, 0.0296131, 0.0458633, 0.0816389, 0.139509, 0.669654},
        {0.109153, 0.675205, 0.468226, -0.226626, -0.128289, 0.00278272}};
    switch (l) {
        case 0:
            switch (n) {
                case 1: return s1; case 2: return s2; case 3: return s3;
                case 4: return s4; case 5: return s5; case 6: return s6;
            }
            break;
        case 1:
            switch (n) {
                case 2: return p2; case 3: return p3; case 4: return p4;
                case 5: return p5; case 6: return p6;
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
        "STO-6G: no (n, l) row for n=" + std::to_string(n)
        + " l=" + std::to_string(l)
        + " (Stewart 1970 tabulates n <= 5; xtb supplies 6s and 6p)");
}

// ---------------------------------------------------------------------------
// STO-NG expansion
// ---------------------------------------------------------------------------

std::pair<std::vector<double>, std::vector<double>>
SemiempiricalBasis::sto_ng_expansion(double zeta, int n, int l, int n_primitives) {
    const StoNgRow* row = nullptr;
    switch (n_primitives) {
        case 3: row = &sto3g_row(n, l); break;
        case 4: row = &sto4g_row(n, l); break;
        case 6: row = &sto6g_row(n, l); break;
        default:
            throw std::runtime_error(
                "SemiempiricalBasis: STO-NG expansions exist for 3, 4 or 6 "
                "primitives; got " + std::to_string(n_primitives));
    }

    std::vector<double> exponents = row->alphas;
    std::vector<double> coefficients = row->coeffs;

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

        // The STO-NG row is keyed by the shell's principal quantum number
        // (Stewart 1970 keys on n; xtb slaterToGauss does the same). The
        // former `vs.l + 1` gave every s shell the 1s row and every p shell
        // the 2p row whatever the period.
        auto [exponents, coeffs] = sto_ng_expansion(
            zeta, params.shell_principal_quantum_number(Z, vs.l), vs.l,
            n_primitives);

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
    // Every STO-NG row (3G, 4G, 6G) matches Stewart (1970): the 3G/4G rows
    // were cross-verified against xtb slater.f90 (bdef7c15c) and the 6G
    // rows against Stewart's own printed residuals (2026-09-19). The DFTB
    // stack builds with n_primitives = 6 on every path, so it now runs on
    // genuine STO-6G Slater fits of its tabulated exponents. Correcting the
    // former tables shifted every DFTB number (they were wrong functions,
    // not a convention); see the changelog entry and issue thread for the
    // measured before/after.
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
