#include "vibeqc/bipole_ewald_gram.hpp"
#include "vibeqc/detail/periodic_reciprocal_source.hpp"
#include "vibeqc/detail/sha256.hpp"

#include <algorithm>
#include <cfloat>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace vibeqc {
namespace {
using U = std::uint64_t;
using Z = std::complex<double>;
using Plan = BipoleEwaldGramPlan;
using Options = BipoleEwaldGramOptions;
using Selection = BipoleEwaldGramSelection;
using Inventory = BipoleEwaldGramInventory;
using Caps = BipoleEwaldGramCaps;
using Numeric = detail::PeriodicReciprocalNumericPlan;
constexpr U ShaMaximumBytes = std::numeric_limits<U>::max()/8;
constexpr std::int64_t ExactInteger = std::int64_t{1} << 53;

[[noreturn]] void invalid(const char* message) { throw std::invalid_argument(message); }
U add(U a, U b) {
    if (b > std::numeric_limits<U>::max()-a) throw std::overflow_error("BIPOLE Ewald Gram count overflow");
    return a+b;
}
U mul(U a, U b) {
    if (a && b > std::numeric_limits<U>::max()/a) throw std::overflow_error("BIPOLE Ewald Gram count overflow");
    return a*b;
}
U blocks(U count, U block) { return count/block+(count%block != 0); }
void cap(U value, U bound, const char* message) { if (value > bound) throw std::length_error(message); }
double finite(double x) {
    if (!std::isfinite(x)) throw std::overflow_error("BIPOLE Ewald Gram nonfinite numerical value");
    return x;
}
Z finite(Z z) { finite(z.real()); finite(z.imag()); return z; }
void environment() {
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0) || FLT_EVAL_METHOD != 0
    invalid("BIPOLE Ewald Gram requires strict floating point compilation");
#endif
    if (!std::numeric_limits<double>::is_iec559 || sizeof(double) != 8 ||
        std::numeric_limits<double>::digits != 53 || std::numeric_limits<double>::max_exponent != 1024 ||
        FLT_RADIX != 2 || sizeof(Z) != 16 || std::fegetround() != FE_TONEAREST)
        invalid("BIPOLE Ewald Gram requires binary64 round-to-nearest");
    volatile double tiny = std::numeric_limits<double>::denorm_min(), one = 1, zero = 0;
    volatile double normal = std::numeric_limits<double>::min(), half = 0.5;
    if (!(tiny > 0) || tiny*one != tiny || tiny+zero != tiny ||
        std::fma(tiny,one,zero) != tiny || std::nextafter(0.0,1.0) != tiny || normal*half == 0)
        invalid("BIPOLE Ewald Gram requires gradual underflow");
}
void controls(const Options& o, const Inventory& i, const Caps& c) {
    environment();
    if (!std::isfinite(o.omega) || o.omega <= 0 ||
        !std::isfinite(o.reciprocal_energy_cutoff) || o.reciprocal_energy_cutoff <= 0 ||
        o.reciprocal_energy_cutoff > std::numeric_limits<double>::max()/2 || !o.reciprocal_block_size)
        invalid("BIPOLE Ewald Gram requires explicit positive finite omega/cutoff/block controls");
    if (!i.numerical_replicas || !i.backend_margin_bytes_per_replica)
        invalid("BIPOLE Ewald Gram requires positive replica count and backend margin");
    if (!c.maximum_kpoints || !c.maximum_reciprocal_candidates || !c.maximum_accepted_vectors ||
        !c.maximum_reciprocal_blocks || !c.maximum_borrowed_numerical_bytes ||
        !c.maximum_owned_numerical_bytes || !c.maximum_control_storage_bytes ||
        !c.maximum_per_replica_inventoried_bytes || !c.maximum_node_inventoried_bytes || !c.maximum_work_units)
        invalid("BIPOLE Ewald Gram requires positive explicit resource caps");
}
void cell_shape(AOPairFourierCellView cells) {
    if (cells.element_count != mul(3,cells.cell_count) ||
        (cells.element_count && (!cells.indices || reinterpret_cast<std::uintptr_t>(cells.indices)%alignof(std::int64_t))))
        invalid("BIPOLE Ewald Gram requires aligned exact int64 cell extent");
    if (mul(cells.element_count,sizeof(std::int64_t)) > std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("BIPOLE Ewald Gram cell address extent");
}
struct SourceWorkspace {
    Numeric own, opposite;
    Eigen::Matrix3d reciprocal;
    Eigen::Vector3d q, qbar, left_k, right_k;
    std::array<std::int64_t,3> partner_shift{};
};
void center(const RegularKMesh& mesh, U index, std::array<int,3>& numerator,
            std::array<int,3>& wrap, Eigen::Vector3d& fractional) {
    const auto raw = mesh.transfer_address(index).doubled;
    for (int d = 0; d < 3; ++d) {
        const int modulus = mesh.doubled_modulus()[d];
        wrap[d] = raw[d] >= mesh.mesh()[d] ? 1 : 0;
        numerator[d] = raw[d]-modulus*wrap[d];
        fractional[d] = static_cast<double>(numerator[d])/static_cast<double>(modulus);
    }
}
SourceWorkspace prepare(const PeriodicSystem& system, const RegularKMesh& mesh,
                        const Selection& s, const Options& o, const Caps& caps, Plan& p) {
    SourceWorkspace w;
    w.reciprocal = system.reciprocal_lattice();
    if (!w.reciprocal.allFinite()) invalid("BIPOLE Ewald Gram requires finite nonsingular original lattice");
    center(mesh,s.q_index,p.centered_q_doubled,p.centered_q_wrap,w.q);
    const auto opposite = mesh.negate(mesh.transfer_address(s.q_index));
    p.opposite_q_index = mesh.transfer_index(opposite);
    std::array<int,3> opposite_numerator{}, opposite_wrap{};
    center(mesh,p.opposite_q_index,opposite_numerator,opposite_wrap,w.qbar);
    for (int d = 0; d < 3; ++d) {
        const auto sum = static_cast<std::int64_t>(p.centered_q_doubled[d])+opposite_numerator[d];
        const auto modulus = static_cast<std::int64_t>(mesh.doubled_modulus()[d]);
        if (sum%modulus || (sum/modulus != 0 && sum/modulus != -1))
            throw std::logic_error("BIPOLE Ewald Gram centered q conjugacy algebra");
        w.partner_shift[d] = -sum/modulus;
    }
    w.own = detail::prepare_periodic_reciprocal_numeric_source(w.reciprocal,w.q,
        o.reciprocal_energy_cutoff,s.q_index==0,caps.maximum_reciprocal_candidates);
    if (o.require_reciprocal_conjugacy)
        w.opposite = p.opposite_q_index==s.q_index ? w.own :
            detail::prepare_periodic_reciprocal_numeric_source(w.reciprocal,w.qbar,
                o.reciprocal_energy_cutoff,p.opposite_q_index==0,caps.maximum_reciprocal_candidates);
    w.left_k = -(w.reciprocal*mesh.fractional_at(s.left_k_index));
    w.right_k = -(w.reciprocal*mesh.fractional_at(s.right_k_index));
    if (!w.left_k.allFinite() || !w.right_k.allFinite())
        throw std::overflow_error("BIPOLE Ewald Gram nonfinite Bloch phase vector");
    return w;
}

class Hasher {
public:
    explicit Hasher(const char* domain) { text(domain); u64(1); }
    void bytes(const std::uint8_t* p, U n) {
        if (n > ShaMaximumBytes-size_) throw std::length_error("BIPOLE Ewald Gram SHA message extent");
        hash_.update(p,static_cast<std::size_t>(n)); size_ += n;
    }
    void u64(U x) {
        std::array<std::uint8_t,8> b{};
        for (unsigned j = 0; j < 8; ++j) b[j] = static_cast<std::uint8_t>(x>>(56-8*j));
        bytes(b.data(),8);
    }
    void real(double x) {
        finite(x); if (x==0) x=0;
        U bits; std::memcpy(&bits,&x,8); u64(bits);
    }
    void complex(Z z) { real(z.real()); real(z.imag()); }
    void text(const char* s) {
        const auto n = std::strlen(s); u64(n); bytes(reinterpret_cast<const std::uint8_t*>(s),n);
    }
    void text(const std::string& s) { u64(s.size()); bytes(reinterpret_cast<const std::uint8_t*>(s.data()),s.size()); }
    std::string finish() { return hash_.finish_hex(); }
private:
    detail::Sha256 hash_;
    U size_ = 0;
};
U geometry_controls(const PeriodicSystem& system) {
    U bytes = add(sizeof(PeriodicSystem),mul(system.unit_cell.size(),sizeof(Atom)));
    if (system.symmetry) {
        const auto& s = *system.symmetry;
        bytes = add(bytes,mul(s.operations.size(),sizeof(SymmetryOp)));
        bytes = add(bytes,mul(s.equivalent_atoms.size(),sizeof(int)));
        bytes = add(bytes,add(add(s.international_symbol.size(),1),add(s.point_group.size(),1)));
    }
    return bytes;
}
U fixed_controls() {
    return 65536+2*sizeof(Plan)+sizeof(BipoleEwaldGramResult)+sizeof(Options)+sizeof(Selection)+
        sizeof(Inventory)+sizeof(Caps)+sizeof(RegularKMesh)+2*sizeof(AOPairFourierPanel)+
        2*sizeof(Hasher)+5*sizeof(std::vector<Z>)+3*65;
}
std::string input_hash(const BasisSet& basis, const PeriodicSystem& system, const RegularKMesh& mesh,
                       AOPairFourierCellView cells, AOPairFourierCellView right_cells,
                       const Selection& s, const Options& o) {
    bool same = cells.cell_count == right_cells.cell_count;
    for (U j = 0; same && j < cells.element_count; ++j)
        same = cells.indices[j] == right_cells.indices[j];
    Hasher h(same ? "vibeqc.bipole.ewald-gram.input" : "vibeqc.bipole.ewald-product-gram.input");
    h.text(auxiliary_basis_content_identity_sha256(basis));
    h.u64(static_cast<U>(system.dim)); h.u64(static_cast<U>(system.charge)); h.u64(static_cast<U>(system.multiplicity));
    for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) h.real(system.lattice(a,b));
    h.u64(system.unit_cell.size());
    for (const auto& atom : system.unit_cell) {
        h.u64(static_cast<U>(atom.Z));
        for (double v : atom.xyz) h.real(v);
    }
    for (int n : mesh.mesh()) h.u64(n);
    for (int n : mesh.is_shift()) h.u64(n);
    h.u64(cells.cell_count);
    for (U j = 0; j < cells.element_count; ++j) h.u64(static_cast<U>(cells.indices[j]));
    if (!same) {
        h.u64(right_cells.cell_count);
        for (U j = 0; j < right_cells.element_count; ++j) h.u64(static_cast<U>(right_cells.indices[j]));
    }
    for (U value : {s.q_index,s.left_k_index,s.right_k_index,s.left_pair_begin,s.left_pair_count,
                   s.right_pair_begin,s.right_pair_count}) h.u64(value);
    h.real(o.omega); h.real(o.reciprocal_energy_cutoff);
    h.text(same ? "explicit-ordered-shared-cells;inverse-bloch;centered-transfer;zero-mode-omitted;no-repair;weighted-product-scaled-v1" :
        "explicit-ordered-left-right-cells;inverse-bloch;centered-transfer;zero-mode-omitted;no-repair;weighted-product-scaled-v1");
    return h.finish();
}
bool cell_payload(AOPairFourierCellView cells) {
    bool closed = true;
    for (U i = 0; i < cells.cell_count; ++i) {
        for (U d = 0; d < 3; ++d)
            if (cells.indices[3*i+d] < -ExactInteger || cells.indices[3*i+d] > ExactInteger)
                invalid("BIPOLE Ewald Gram cell index is not exactly representable in binary64");
        bool inverse_found = false;
        for (U j = 0; j < cells.cell_count; ++j) {
            bool same = true, inverse = true;
            for (U d = 0; d < 3; ++d) {
                same = same && cells.indices[3*j+d]==cells.indices[3*i+d];
                // i was bounded above; no int64 negation can overflow.
                inverse = inverse && cells.indices[3*j+d]==-cells.indices[3*i+d];
            }
            if (same && j < i) invalid("BIPOLE Ewald Gram duplicate explicit cell label");
            inverse_found = inverse_found || inverse;
        }
        closed = closed && inverse_found;
    }
    return closed;
}
U count(const Numeric& source) {
    return detail::visit_periodic_reciprocal_numeric_source(source.geometry,
        [](const std::array<std::int64_t,3>&, const std::array<double,5>&, void*) {},nullptr);
}
// Dividing sequentially avoids overflow in 4*omega^2. Overflow of the
// nonnegative quotient implies an exponentially vanishing weight. Ordinary
// exp underflow is deliberately allowed, including subnormal/zero weights.
double ewald_weight(const std::array<double,5>& lanes, double omega) {
    const double ratio = (lanes[3]/omega)/omega;
    if (std::isnan(ratio) || ratio < 0) throw std::overflow_error("BIPOLE Ewald Gram invalid screening exponent");
    const double screen = std::isinf(ratio) ? 0.0 : std::exp(-0.25*ratio);
    const double weight = finite(lanes[4]*screen);
    if (weight < 0) throw std::overflow_error("BIPOLE Ewald Gram negative reciprocal weight");
    return weight==0 ? 0.0 : weight;
}
void source_prefix(Hasher& h, const SourceWorkspace& w, const Plan& p, const Options& o, U accepted) {
    h.text("numeric-source-v1-fixed-fma-q-slack;screen=(p2/omega)/omega;zero-omitted");
    h.real(o.omega); h.real(o.reciprocal_energy_cutoff);
    for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) h.real(w.reciprocal(a,b));
    for (int n : p.centered_q_doubled) h.u64(static_cast<U>(n));
    for (int n : p.centered_q_wrap) h.u64(static_cast<U>(n));
    for (int d = 0; d < 3; ++d) h.real(w.q[d]);
    h.real(w.own.geometry.cell_volume); h.real(w.own.boundary_tolerance);
    h.real(w.own.geometry.cutoff_squared); h.real(w.own.geometry.radial_limit_squared);
    for (auto n : w.own.geometry.lower_bounds) h.u64(static_cast<U>(n));
    for (auto n : w.own.geometry.upper_bounds) h.u64(static_cast<U>(n));
    h.u64(p.reciprocal_candidates); h.u64(accepted);
}
void accumulate(double x, double& value, double& correction) {
    finite(x);
    const double sum = finite(value+x);
    correction = finite(correction+(std::abs(value)>=std::abs(x) ? (value-sum)+x : (x-sum)+value));
    value = sum;
}
void accumulate(Z x, Z& value, Z& correction) {
    double r=value.real(), i=value.imag(), rc=correction.real(), ic=correction.imag();
    accumulate(x.real(),r,rc); accumulate(x.imag(),i,ic);
    value={r,i}; correction={rc,ic};
}
bool normal_product(double a, double b) {
    if (a==0 || b==0) return true;
    int ea=0, eb=0;
    (void)std::frexp(a,&ea); (void)std::frexp(b,&eb);
    const int exponent=ea+eb;
    // Product significands have magnitude [1/4,1); leave ample room for
    // the subsequent two-term complex sum at the upper boundary.
    return exponent>=std::numeric_limits<double>::min_exponent+1 &&
        exponent<=std::numeric_limits<double>::max_exponent-4;
}
struct ScaledProduct {
    double value=0, error=0;
    int exponent=0;
};
ScaledProduct scaled_product(double a, double b, double weight) {
    if (a==0 || b==0) return {};
    int ea=0, eb=0, ew=0;
    const double ma=std::frexp(a,&ea), mb=std::frexp(b,&eb), mw=std::frexp(weight,&ew);
    const double ab=ma*mb, ab_error=std::fma(ma,mb,-ab);
    const double value=ab*mw;
    // All significand products are normal and finite. FMA captures their
    // first rounding remainders without making a certified interval claim.
    const double error=std::fma(ab,mw,-value)+ab_error*mw;
    return {value,error,ea+eb+ew};
}
bool exact_opposite_products(double a, double b, double c, double d) {
    return (a==-c && b==d) || (a==c && b==-d) ||
        (a==-d && b==c) || (a==d && b==-c);
}
bool ordinary_component(double value, double a, double b, double c, double d) {
    if (std::isnormal(value)) return true;
    if (value!=0) return false;
    const bool both_zero=(a==0 || b==0) && (c==0 || d==0);
    return both_zero || exact_opposite_products(a,b,c,d);
}
double scaled_component(double a, double b, double c, double d, double weight) {
    const auto x=scaled_product(a,b,weight), y=scaled_product(c,d,weight);
    if (x.value==0 && y.value==0) return 0;
    const int exponent=x.value==0 ? y.exponent : y.value==0 ? x.exponent : std::max(x.exponent,y.exponent);
    double sum=0, correction=0;
    accumulate(std::scalbn(x.value,x.exponent-exponent),sum,correction);
    accumulate(std::scalbn(y.value,y.exponent-exponent),sum,correction);
    accumulate(std::scalbn(x.error,x.exponent-exponent),sum,correction);
    accumulate(std::scalbn(y.error,y.exponent-exponent),sum,correction);
    const double significand=finite(sum+correction);
    if (significand==0 && x.value!=0 && y.value!=0) {
        if (!exact_opposite_products(a,b,c,d))
            throw std::overflow_error("BIPOLE Ewald Gram unresolved scaled-product cancellation");
    }
    // Only this final physical rescaling can intentionally underflow. A
    // nonfinite final component is rejected, never clipped or projected.
    const double out=finite(std::scalbn(significand,exponent));
    return out==0 ? 0.0 : out;
}
Z weighted_product(Z left, Z right, double weight) {
    finite(left); finite(right); finite(weight);
    if (weight==0) return {};
    const double a=left.real(), b=left.imag(), c=right.real(), d=right.imag();
    if (normal_product(a,c) && normal_product(b,d) && normal_product(a,d) && normal_product(b,c)) {
        const Z product=finite(std::conj(left)*right);
        // Normal terms can still cancel into a subnormal or rounded zero.
        // Only a normal component or a proven algebraic zero may bypass the
        // exponent-scaled path before the restoring weight is applied.
        if (ordinary_component(product.real(),a,c,b,d) && ordinary_component(product.imag(),a,d,-b,c))
            return finite(weight*product); // unchanged safe ordinary-range operation order
    }
    // Include weight's exponent BEFORE forming any potentially tiny or huge
    // complex product. conj(left)*right has real ac+bd and imaginary ad-bc.
    return {scaled_component(a,c,b,d,weight),scaled_component(a,d,-b,c,weight)};
}
struct Stream {
    const BasisSet& basis;
    const PeriodicSystem& system;
    AOPairFourierCellView cells, right_cells;
    const Plan& plan;
    const Options& options;
    const Caps& caps;
    const SourceWorkspace& source;
    BipoleEwaldGramDiagnostics& diagnostics;
    Hasher& hash;
    Z* values;
    Z* compensation;
    double* vectors;
    U used = 0;
    void flush() {
        if (!used) return;
        cap(add(diagnostics.evaluated_blocks,1),caps.maximum_reciprocal_blocks,"BIPOLE Ewald Gram reciprocal block cap");
        const auto& s=plan.selection;
        if (s.left_pair_count && s.right_pair_count) {
            const AuxiliaryFourierVectorView v{vectors,vectors+1,vectors+2,static_cast<std::size_t>(used),4,4,4};
            auto lc=caps.ao_panel, rc=caps.ao_panel;
            lc.maximum_output_bytes=std::min(lc.maximum_output_bytes,plan.left_panel.output_bytes);
            rc.maximum_output_bytes=std::min(rc.maximum_output_bytes,plan.right_panel.output_bytes);
            lc.maximum_work_units=std::min(lc.maximum_work_units,plan.left_panel.work_units);
            rc.maximum_work_units=std::min(rc.maximum_work_units,plan.right_panel.work_units);
            const auto left=ao_pair_gaussian_fourier_cell_panel(basis,system,v,source.left_k,
                s.left_pair_begin,s.left_pair_count,cells,lc);
            const auto right=ao_pair_gaussian_fourier_cell_panel(basis,system,v,source.right_k,
                s.right_pair_begin,s.right_pair_count,right_cells,rc);
            if (left.data.size()!=mul(s.left_pair_count,used) || right.data.size()!=mul(s.right_pair_count,used))
                throw std::logic_error("BIPOLE Ewald Gram AO panel extent changed");
            for (U l=0;l<s.left_pair_count;++l) for (U r=0;r<s.right_pair_count;++r) {
                const U index=l*s.right_pair_count+r;
                for (U g=0;g<used;++g) {
                    const double weight=vectors[4*g+3];
                    if (weight==0) continue;
                    accumulate(weighted_product(left.data[l*used+g],right.data[r*used+g],weight),
                        values[index],compensation[index]);
                }
            }
        }
        ++diagnostics.evaluated_blocks; used=0;
    }
    void record(const std::array<std::int64_t,3>& label,const std::array<double,5>& lanes) {
        cap(add(diagnostics.accepted_vectors,1),caps.maximum_accepted_vectors,"BIPOLE Ewald Gram accepted vector cap");
        const double weight=ewald_weight(lanes,options.omega);
        for (auto n:label) hash.u64(static_cast<U>(n));
        for (double v:lanes) hash.real(v);
        hash.real(weight);
        if (!weight) ++diagnostics.zero_weight_vectors;
        diagnostics.maximum_weight=std::max(diagnostics.maximum_weight,weight);
        for (U d=0;d<3;++d) vectors[4*used+d]=lanes[d];
        vectors[4*used+3]=weight;
        ++used; ++diagnostics.accepted_vectors;
        if (used==plan.block_vectors) flush();
    }
};

} // namespace

Plan plan_bipole_ewald_product_gram(const BasisSet& basis, const PeriodicSystem& system,
    const RegularKMesh& mesh, AOPairFourierCellView cells, AOPairFourierCellView right_cells,
    const Selection& selection,
    const Options& options, const Inventory& live, const Caps& caps) {
    controls(options,live,caps); cell_shape(cells); cell_shape(right_cells);
    if (system.dim!=3) invalid("BIPOLE Ewald Gram currently requires a 3D original cell");
    cap(mesh.size(),caps.maximum_kpoints,"BIPOLE Ewald Gram k-point cap");
    if (selection.q_index>=mesh.size() || selection.left_k_index>=mesh.size() || selection.right_k_index>=mesh.size())
        invalid("BIPOLE Ewald Gram reciprocal/grid index extent");
    Plan p; p.selection=selection; p.n_basis=basis.nbasis(); p.n_kpoints=mesh.size();
    p.left_cell_count=cells.cell_count; p.right_cell_count=right_cells.cell_count;
    p.shared_cell_storage=cells.indices==right_cells.indices && cells.cell_count==right_cells.cell_count;
    p.cell_count=p.shared_cell_storage ? cells.cell_count : add(cells.cell_count,right_cells.cell_count);
    // Metadata-only leaf plans precede lattice preparation and all payload scans.
    p.left_panel=plan_ao_pair_gaussian_fourier_cell_panel(basis,options.reciprocal_block_size,
        selection.left_pair_begin,selection.left_pair_count,cells.cell_count,caps.ao_panel);
    p.right_panel=plan_ao_pair_gaussian_fourier_cell_panel(basis,options.reciprocal_block_size,
        selection.right_pair_begin,selection.right_pair_count,right_cells.cell_count,caps.ao_panel);
    auto source=prepare(system,mesh,selection,options,caps,p);
    p.reciprocal_candidates=source.own.enumeration.candidate_count;
    p.opposite_reciprocal_candidates=options.require_reciprocal_conjugacy ? source.opposite.enumeration.candidate_count : 0;
    p.accepted_vectors_upper_bound=p.reciprocal_candidates;
    p.block_vectors=std::min(options.reciprocal_block_size,std::max(U{1},p.accepted_vectors_upper_bound));
    if (p.block_vectors!=options.reciprocal_block_size) {
        p.left_panel=plan_ao_pair_gaussian_fourier_cell_panel(basis,p.block_vectors,
            selection.left_pair_begin,selection.left_pair_count,cells.cell_count,caps.ao_panel);
        p.right_panel=plan_ao_pair_gaussian_fourier_cell_panel(basis,p.block_vectors,
            selection.right_pair_begin,selection.right_pair_count,right_cells.cell_count,caps.ao_panel);
    }
    p.reciprocal_blocks_upper_bound=blocks(p.accepted_vectors_upper_bound,p.block_vectors);
    cap(p.accepted_vectors_upper_bound,caps.maximum_accepted_vectors,"BIPOLE Ewald Gram candidate-based accepted-vector upper cap");
    cap(p.reciprocal_blocks_upper_bound,caps.maximum_reciprocal_blocks,"BIPOLE Ewald Gram candidate-based block upper cap");
    const U lr=mul(selection.left_pair_count,selection.right_pair_count);
    p.retained_output_bytes=mul(16,lr); p.compensation_bytes=p.retained_output_bytes;
    p.reciprocal_buffer_bytes=mul(32,p.block_vectors);
    p.maximum_ao_panel_bytes=add(p.left_panel.output_bytes,p.right_panel.output_bytes);
    p.fixed_numerical_workspace_bytes=add(sizeof(SourceWorkspace)+4096,
        std::max(add(p.left_panel.fixed_numeric_workspace_bytes,p.left_panel.fixed_scalar_numeric_bytes),
                 add(p.right_panel.fixed_numeric_workspace_bytes,p.right_panel.fixed_scalar_numeric_bytes)));
    p.borrowed_basis_numeric_bytes=p.left_panel.borrowed_basis_numeric_bytes;
    if (p.borrowed_basis_numeric_bytes!=p.right_panel.borrowed_basis_numeric_bytes)
        throw std::logic_error("BIPOLE Ewald Gram shared basis census mismatch");
    p.borrowed_cell_bytes=mul(24,p.cell_count);
    p.borrowed_geometry_numeric_bytes=add(72,mul(28,system.unit_cell.size()));
    p.borrowed_numerical_bytes=add(p.borrowed_basis_numeric_bytes,add(p.borrowed_cell_bytes,p.borrowed_geometry_numeric_bytes));
    p.peak_owned_numerical_bytes=add(add(p.retained_output_bytes,p.compensation_bytes),
        add(p.reciprocal_buffer_bytes,add(p.maximum_ao_panel_bytes,p.fixed_numerical_workspace_bytes)));
    p.control_storage_bytes=add(live.other_live_control_bytes_per_replica,
        add(fixed_controls(),add(geometry_controls(system),add(p.left_panel.basis_control_storage_bytes,
            std::max(p.left_panel.fixed_control_storage_bytes,p.right_panel.fixed_control_storage_bytes)))));
    // Own count + own streamed evaluation; optional opposite count plus the
    // conjugacy source traversal AND its one direct opposite predicate per
    // accepted source record (accepted <= own candidates). Count both roles.
    const U traversals=add(mul(2,p.reciprocal_candidates), options.require_reciprocal_conjugacy ?
        add(p.opposite_reciprocal_candidates,mul(2,p.reciprocal_candidates)) : 0);
    p.reciprocal_traversal_work_units=mul(4096,add(1024,traversals));
    p.contraction_work_units=add(mul(p.reciprocal_blocks_upper_bound,
        add(p.left_panel.work_units,p.right_panel.work_units)),mul(1024,mul(lr,p.accepted_vectors_upper_bound)));
    const U input_lanes=add(p.borrowed_numerical_bytes/4,add(basis.nshells(),system.unit_cell.size()));
    const U cell_checks=add(mul(cells.cell_count,cells.cell_count),p.shared_cell_storage ? 0 :
        mul(right_cells.cell_count,right_cells.cell_count));
    p.validation_work_units=mul(1024,add(1024,add(input_lanes,add(cell_checks,lr))));
    p.work_units_upper_bound=add(65536,add(p.reciprocal_traversal_work_units,
        add(p.contraction_work_units,add(p.validation_work_units,
            mul(4,add(p.left_panel.metadata_work_units,p.right_panel.metadata_work_units))))));
    p.per_replica_inventoried_bytes=add(p.peak_owned_numerical_bytes,add(p.borrowed_numerical_bytes,
        add(live.other_live_numerical_bytes_per_replica,add(p.control_storage_bytes,live.backend_margin_bytes_per_replica))));
    p.required_node_inventoried_bytes=add(live.external_node_bytes,mul(live.numerical_replicas,p.per_replica_inventoried_bytes));
    cap(p.borrowed_numerical_bytes,caps.maximum_borrowed_numerical_bytes,"BIPOLE Ewald Gram borrowed numerical cap");
    cap(p.peak_owned_numerical_bytes,caps.maximum_owned_numerical_bytes,"BIPOLE Ewald Gram owned numerical cap");
    cap(p.control_storage_bytes,caps.maximum_control_storage_bytes,"BIPOLE Ewald Gram control cap");
    cap(p.per_replica_inventoried_bytes,caps.maximum_per_replica_inventoried_bytes,"BIPOLE Ewald Gram worker cap");
    cap(p.required_node_inventoried_bytes,caps.maximum_node_inventoried_bytes,"BIPOLE Ewald Gram node cap");
    cap(p.work_units_upper_bound,caps.maximum_work_units,"BIPOLE Ewald Gram work cap");
    cap(p.peak_owned_numerical_bytes,std::numeric_limits<std::size_t>::max(),"BIPOLE Ewald Gram address extent");
    cap(add(4096,mul(72,p.accepted_vectors_upper_bound)),ShaMaximumBytes,"BIPOLE Ewald Gram reciprocal SHA extent");
    cap(add(4096,mul(4,p.borrowed_numerical_bytes)),ShaMaximumBytes,"BIPOLE Ewald Gram input SHA extent");
    cap(add(4096,p.retained_output_bytes),ShaMaximumBytes,"BIPOLE Ewald Gram payload SHA extent");
    return p;
}

BipoleEwaldGramResult make_bipole_ewald_product_gram(const BasisSet& basis, const PeriodicSystem& system,
    const RegularKMesh& mesh, AOPairFourierCellView cells, AOPairFourierCellView right_cells,
    const Selection& selection,
    const Options& options, const Inventory& live, const Caps& caps) {
    const auto p=plan_bipole_ewald_product_gram(basis,system,mesh,cells,right_cells,selection,options,live,caps);
    BipoleEwaldGramResult result; result.plan_=p;
    result.diagnostics_.left_cell_inversion_closed=cell_payload(cells);
    result.diagnostics_.right_cell_inversion_closed=p.shared_cell_storage ?
        result.diagnostics_.left_cell_inversion_closed : cell_payload(right_cells);
    result.diagnostics_.cell_inversion_closed=result.diagnostics_.left_cell_inversion_closed &&
        result.diagnostics_.right_cell_inversion_closed;
    if (options.require_cell_inversion_closure && !result.diagnostics_.cell_inversion_closed)
        invalid("BIPOLE Ewald Gram explicit cells lack inversion closure");
    result.input_identity_=input_hash(basis,system,mesh,cells,right_cells,selection,options);
    Plan repeated=p;
    const auto source=prepare(system,mesh,selection,options,caps,repeated);
    if (source.own.enumeration.candidate_count!=p.reciprocal_candidates ||
        repeated.centered_q_doubled!=p.centered_q_doubled || repeated.opposite_q_index!=p.opposite_q_index)
        throw std::logic_error("BIPOLE Ewald Gram reciprocal metadata changed after admission");
    const U accepted=count(source.own);
    cap(accepted,caps.maximum_accepted_vectors,"BIPOLE Ewald Gram accepted-vector cap");
    if (options.require_reciprocal_conjugacy) {
        const U opposite=p.opposite_q_index==selection.q_index ? accepted : count(source.opposite);
        if (opposite!=accepted) throw std::runtime_error("BIPOLE Ewald Gram reciprocal conjugacy count mismatch");
        detail::require_periodic_reciprocal_numeric_conjugacy(source.own.geometry,source.opposite.geometry,
            source.partner_shift,accepted);
        result.diagnostics_.reciprocal_conjugacy_audited=true;
    }
    const U elements=mul(selection.left_pair_count,selection.right_pair_count);
    result.values_.resize(elements);
    std::vector<Z> compensation(elements);
    std::vector<double> vectors(mul(4,p.block_vectors));
    Hasher reciprocal("vibeqc.bipole.ewald-gram.reciprocal");
    source_prefix(reciprocal,source,p,options,accepted);
    Stream stream{basis,system,cells,right_cells,p,options,caps,source,result.diagnostics_,reciprocal,
        result.values_.data(),compensation.data(),vectors.data()};
    const auto replay=detail::visit_periodic_reciprocal_numeric_source(source.own.geometry,
        [](const std::array<std::int64_t,3>& label,const std::array<double,5>& lanes,void* context) {
            static_cast<Stream*>(context)->record(label,lanes);
        },&stream);
    stream.flush();
    if (replay!=accepted || result.diagnostics_.accepted_vectors!=accepted ||
        result.diagnostics_.evaluated_blocks!=blocks(accepted,p.block_vectors))
        throw std::logic_error("BIPOLE Ewald Gram source replay count changed");
    result.reciprocal_identity_=reciprocal.finish();
    Hasher payload("vibeqc.bipole.ewald-gram.payload");
    payload.text(result.input_identity_); payload.text(result.reciprocal_identity_);
    payload.u64(selection.left_pair_count); payload.u64(selection.right_pair_count);
    for (U j=0;j<elements;++j) {
        result.values_[j]=finite(result.values_[j]+compensation[j]);
        result.diagnostics_.maximum_integral_magnitude=std::max(result.diagnostics_.maximum_integral_magnitude,
            finite(std::abs(result.values_[j])));
        payload.complex(result.values_[j]);
    }
    result.payload_identity_=payload.finish();
    if (input_hash(basis,system,mesh,cells,right_cells,selection,options)!=result.input_identity_)
        throw std::runtime_error("BIPOLE Ewald Gram borrowed input payload changed during evaluation");
    return result;
}

Plan plan_bipole_ewald_gram(const BasisSet& basis, const PeriodicSystem& system,
    const RegularKMesh& mesh, AOPairFourierCellView cells, const Selection& selection,
    const Options& options, const Inventory& live, const Caps& caps) {
    return plan_bipole_ewald_product_gram(basis,system,mesh,cells,cells,selection,options,live,caps);
}

BipoleEwaldGramResult make_bipole_ewald_gram(const BasisSet& basis, const PeriodicSystem& system,
    const RegularKMesh& mesh, AOPairFourierCellView cells, const Selection& selection,
    const Options& options, const Inventory& live, const Caps& caps) {
    return make_bipole_ewald_product_gram(basis,system,mesh,cells,cells,selection,options,live,caps);
}

const Z* BipoleEwaldGramResult::data() const {
    if (input_identity_.size()!=64 || reciprocal_identity_.size()!=64 || payload_identity_.size()!=64 ||
        values_.size()!=mul(plan_.selection.left_pair_count,plan_.selection.right_pair_count))
        throw std::logic_error("BIPOLE Ewald Gram result is moved from or malformed");
    return values_.data();
}
Z BipoleEwaldGramResult::element(U left,U right) const {
    const auto* p=data();
    if (left>=plan_.selection.left_pair_count || right>=plan_.selection.right_pair_count)
        throw std::out_of_range("BIPOLE Ewald Gram selected pair ordinal");
    return p[left*plan_.selection.right_pair_count+right];
}
const std::string& BipoleEwaldGramResult::input_identity_sha256() const { (void)data(); return input_identity_; }
const std::string& BipoleEwaldGramResult::reciprocal_source_identity_sha256() const { (void)data(); return reciprocal_identity_; }
const std::string& BipoleEwaldGramResult::payload_identity_sha256() const { (void)data(); return payload_identity_; }

Z detail::bipole_ewald_weighted_product(Z left, Z right, double weight) {
    environment();
    if (!std::isfinite(weight) || weight<0)
        invalid("BIPOLE Ewald weighted product requires finite nonnegative weight");
    return weighted_product(left,right,weight);
}

Z bipole_ewald_weighted_product_diagnostic(Z left, Z right, double weight) {
    return detail::bipole_ewald_weighted_product(left,right,weight);
}

} // namespace vibeqc
