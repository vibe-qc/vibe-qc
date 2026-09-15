#include "vibeqc/bipole_erfc_bloch.hpp"
#include "vibeqc/detail/sha256.hpp"

#include <algorithm>
#include <array>
#include <cfloat>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace vibeqc {
namespace {
using U = std::uint64_t;
using Z = std::complex<double>;
using Plan = BipoleErfcBlochPlan;
using Selection = BipoleEwaldGramSelection;
using Options = BipoleErfcBlochOptions;
using Inventory = BipoleErfcBlochInventory;
using Caps = BipoleErfcBlochCaps;
constexpr U ShaMaximumBytes = std::numeric_limits<U>::max()/8;
constexpr U ScalarWorkspace = 2048;
constexpr double TwoPi = 6.283185307179586476925286766559005768;

[[noreturn]] void invalid(const char* message) { throw std::invalid_argument(message); }
U add(U a, U b) {
    if (b > std::numeric_limits<U>::max()-a)
        throw std::overflow_error("BIPOLE erfc Bloch count overflow");
    return a+b;
}
U mul(U a, U b) {
    if (a && b > std::numeric_limits<U>::max()/a)
        throw std::overflow_error("BIPOLE erfc Bloch count overflow");
    return a*b;
}
void cap(U value, U maximum, const char* message) {
    if (value > maximum) throw std::length_error(message);
}
double finite(double value) {
    if (!std::isfinite(value)) throw std::overflow_error("BIPOLE erfc Bloch nonfinite numerical value");
    return value;
}
Z finite(Z value) { finite(value.real()); finite(value.imag()); return value; }
void environment() {
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0) || FLT_EVAL_METHOD != 0
    invalid("BIPOLE erfc Bloch requires strict floating point compilation");
#endif
    if (!std::numeric_limits<double>::is_iec559 || sizeof(double)!=8 || sizeof(Z)!=16 ||
        FLT_RADIX!=2 || std::numeric_limits<double>::digits!=53 ||
        std::numeric_limits<double>::max_exponent!=1024 || std::fegetround()!=FE_TONEAREST)
        invalid("BIPOLE erfc Bloch requires binary64 round-to-nearest");
    volatile double tiny=std::numeric_limits<double>::denorm_min(), one=1, zero=0;
    volatile double normal=std::numeric_limits<double>::min(), half=0.5;
    if (!(tiny>0) || tiny*one!=tiny || tiny+zero!=tiny || std::fma(tiny,one,zero)!=tiny ||
        std::nextafter(0.0,1.0)!=tiny || normal*half==0)
        invalid("BIPOLE erfc Bloch requires gradual underflow");
}
void controls(const Inventory& i, const Caps& c) {
    environment();
    if (!i.numerical_replicas || !i.backend_margin_bytes_per_replica)
        invalid("BIPOLE erfc Bloch requires positive replica count and backend allowance");
    if (!c.maximum_kpoints || !c.maximum_phase_evaluations ||
        !c.maximum_borrowed_numerical_bytes || !c.maximum_owned_numerical_bytes ||
        !c.maximum_control_storage_bytes || !c.maximum_worker_bytes ||
        !c.maximum_node_bytes || !c.maximum_work_units)
        invalid("BIPOLE erfc Bloch requires positive explicit resource caps");
}
class Hasher {
public:
    explicit Hasher(const char* domain) { text(domain); u64(1); }
    void bytes(const std::uint8_t* data, U n) {
        if (n > ShaMaximumBytes-size_)
            throw std::length_error("BIPOLE erfc Bloch SHA message extent");
        hash_.update(data,static_cast<std::size_t>(n)); size_+=n;
    }
    void u64(U value) {
        std::array<std::uint8_t,8> buffer{};
        for (unsigned d=0; d<8; ++d) buffer[d]=static_cast<std::uint8_t>(value>>(56-8*d));
        bytes(buffer.data(),8);
    }
    void real(double value) {
        finite(value); if (value==0) value=0;
        U bits; std::memcpy(&bits,&value,8); u64(bits);
    }
    void complex(Z value) { real(value.real()); real(value.imag()); }
    void text(const char* value) {
        const auto n=std::strlen(value); u64(n);
        bytes(reinterpret_cast<const std::uint8_t*>(value),n);
    }
    void text(const std::string& value) {
        u64(value.size()); bytes(reinterpret_cast<const std::uint8_t*>(value.data()),value.size());
    }
    std::string finish() { return hash_.finish_hex(); }
private:
    detail::Sha256 hash_;
    U size_=0;
};
U wrapper_controls() {
    // Fixed logical controls, not a claim about allocator capacity/RSS.
    // Includes caller/copy descriptors, both wrapper and raw result controls,
    // vector descriptors, SHA state and up to twelve live 65-byte digests.
    // Known numeric scalar temporaries have their separate 2048-byte field.
    return 16384+3*sizeof(Plan)+sizeof(BipoleErfcBlochResult)+sizeof(BipoleErfcPanelResult)+
        2*(sizeof(Options)+sizeof(Inventory)+sizeof(Caps)+sizeof(Selection)+
           sizeof(BipoleErfcImageView)+sizeof(BipoleErfcPanelInventory)+
           sizeof(BipoleErfcPanelCaps)+sizeof(BipoleErfcPanelSelection))+
        sizeof(RegularKMesh)+2*sizeof(std::vector<Z>)+4*sizeof(Hasher)+12*(65+sizeof(std::string));
}
U wrapper_resident(const Plan& p) {
    return add(add(p.retained_output_bytes,p.compensation_bytes),p.fixed_numeric_workspace_bytes);
}
U wrapper_work(const Plan& p) {
    return add(p.support_work_units,add(p.phase_work_units,add(p.folding_work_units,p.identity_work_units)));
}
BipoleErfcPanelSelection raw_selection(const Selection& s) {
    return {s.left_pair_begin,s.left_pair_count,s.right_pair_begin,s.right_pair_count};
}
BipoleErfcPanelInventory raw_inventory(const Inventory& i, const Plan& p) {
    BipoleErfcPanelInventory result;
    result.numerical_replicas=i.numerical_replicas;
    result.external_node_bytes=i.external_node_bytes;
    result.other_live_numerical_bytes_per_replica=add(i.other_live_numerical_bytes_per_replica,wrapper_resident(p));
    result.other_live_control_bytes_per_replica=add(i.other_live_control_bytes_per_replica,p.wrapper_control_storage_bytes);
    result.backend_margin_bytes_per_replica=i.backend_margin_bytes_per_replica;
    return result;
}
BipoleErfcPanelCaps raw_caps(const Caps& c, const Plan& p) {
    auto result=c.raw;
    // Count-only clamping preserves the caller's caps and scientific options.
    // It also makes the child's incremental metadata census obey the enclosing
    // envelope, instead of discovering a generous-child/strict-parent mismatch
    // only after an arbitrarily long shell walk. Final admission additionally
    // charges the extra metadata planning pass performed by make below.
    const U resident=wrapper_resident(p), work=wrapper_work(p);
    if (resident>=c.maximum_owned_numerical_bytes)
        throw std::length_error("BIPOLE erfc Bloch wrapper/raw owned numerical cap");
    if (work>=c.maximum_work_units)
        throw std::length_error("BIPOLE erfc Bloch phase/folding/identity work cap");
    result.maximum_owned_numerical_bytes=std::min(result.maximum_owned_numerical_bytes,c.maximum_owned_numerical_bytes-resident);
    result.maximum_borrowed_numerical_bytes=std::min(result.maximum_borrowed_numerical_bytes,c.maximum_borrowed_numerical_bytes);
    result.maximum_control_storage_bytes=std::min(result.maximum_control_storage_bytes,c.maximum_control_storage_bytes);
    result.maximum_worker_bytes=std::min(result.maximum_worker_bytes,c.maximum_worker_bytes);
    result.maximum_node_bytes=std::min(result.maximum_node_bytes,c.maximum_node_bytes);
    result.maximum_work_units=std::min(result.maximum_work_units,c.maximum_work_units-work);
    return result;
}
void hash_mesh_selection(Hasher& h, const RegularKMesh& mesh, const Selection& s) {
    for (int d=0; d<3; ++d) { h.u64(mesh.mesh()[d]); h.u64(mesh.is_shift()[d]); }
    for (U value : {s.q_index,s.left_k_index,s.right_k_index,s.left_pair_begin,
                    s.left_pair_count,s.right_pair_begin,s.right_pair_count}) h.u64(value);
}
std::string image_hash(BipoleErfcImageView images) {
    Hasher h("vibeqc.bipole.erfc-bloch.images"); h.u64(images.image_count);
    for (U j=0; j<images.element_count; ++j) h.u64(static_cast<U>(images.indices[j]));
    return h.finish();
}
std::string input_hash(const std::string& raw_input, const std::string& images,
                       const RegularKMesh& mesh, const Selection& s,
                       const std::string& support_identity) {
    Hasher h("vibeqc.bipole.erfc-bloch.input"); h.text(raw_input); h.text(images);
    hash_mesh_selection(h,mesh,s);
    h.text("inverse-Bloch:+kL.g+(q+kR).p-kR.s;doubled-integer-axis-characters-v1;ordered-image-slab;no-Nk-no-repair");
    if (!support_identity.empty()) {
        h.text("checked-image-permutation-support"); h.text(support_identity);
    }
    return h.finish();
}
std::string certify_image_support(BipoleErfcImageView images, const std::string& identity,
                                  U& comparisons) {
    // Sun2023 Eqs.18,45-47: real Gaussian quartets and home-cell reanchoring.
    // Differences in this exact-label domain are at most 2^54 and cannot
    // overflow int64, even for input whose transformed image is absent.
    constexpr std::int64_t exact=std::int64_t(1)<<53;
    for (U j=0; j<images.element_count; ++j)
        if (images.indices[j]<-exact || images.indices[j]>exact)
            invalid("BIPOLE erfc Bloch support requires exactly representable image labels");
    // Identity is counted through the original multiplicity. These seven
    // complete the real-ERI group, including stabilizers and repeated images.
    constexpr int permutations[7][4]={{1,0,2,3},{0,1,3,2},{1,0,3,2},
        {2,3,0,1},{3,2,0,1},{2,3,1,0},{3,2,1,0}};
    auto multiplicity=[&](const std::int64_t* target) {
        U count=0;
        for (U candidate=0; candidate<images.image_count; ++candidate) {
            bool equal=true;
            for (U d=0; d<9; ++d) equal &= images.indices[9*candidate+d]==target[d];
            if (equal) ++count;
            ++comparisons;
        }
        return count;
    };
    for (U image=0; image<images.image_count; ++image) {
        const auto* original=images.indices+9*image;
        const U expected=multiplicity(original);
        std::array<std::array<std::int64_t,3>,4> cells{};
        for (U center=1; center<4; ++center)
            for (U d=0; d<3; ++d) cells[center][d]=original[3*(center-1)+d];
        for (const auto& permutation : permutations) {
            std::array<std::int64_t,9> transformed{};
            for (U center=1; center<4; ++center)
                for (U d=0; d<3; ++d)
                    transformed[3*(center-1)+d]=cells[permutation[center]][d]-cells[permutation[0]][d];
            if (multiplicity(transformed.data())!=expected)
                invalid("BIPOLE erfc Bloch image multiset is not closed under quartet permutation/reanchoring");
        }
    }
    Hasher h("vibeqc.bipole.erfc-bloch.image-permutation-support");
    h.text(identity); h.u64(images.image_count);
    h.text("real-ERI-eight-permutations;first-cell-reanchoring;exact-integer-multiplicity;no-completion;image-only-v1");
    return h.finish();
}
U residue(std::int64_t label, U modulus) {
    // Modulus is at most INT_MAX-1. The signed remainder is safe even for
    // INT64_MIN; no negation/abs of the original label is performed.
    auto r=label%static_cast<std::int64_t>(modulus);
    if (r<0) r+=static_cast<std::int64_t>(modulus);
    return static_cast<U>(r);
}
Z character(U numerator, U modulus) {
    if (numerator==0) return {1,0};
    if (mul(2,numerator)==modulus) return {-1,0};
    if (mul(4,numerator)==modulus) return {0,1};
    if (mul(4,numerator)==mul(3,modulus)) return {0,-1};
    // Opposite residues use the same positive angle then exact conjugation.
    // Zero, half and quarter turns never pass through libm. No unit-modulus
    // projection is applied to general phases, whose residual is reported.
    const bool conjugate=numerator>modulus/2;
    const U reduced=conjugate ? modulus-numerator : numerator;
    const double angle=TwoPi*(static_cast<double>(reduced)/static_cast<double>(modulus));
    const Z positive{std::cos(angle),std::sin(angle)};
    return finite(conjugate ? std::conj(positive) : positive);
}
struct PhaseAddresses {
    std::array<int,3> left{}, right{}, transfer{}, modulus{};
};
PhaseAddresses addresses(const RegularKMesh& mesh, const Selection& s) {
    return {mesh.address(s.left_k_index).doubled,mesh.address(s.right_k_index).doubled,
            mesh.transfer_address(s.q_index).doubled,mesh.doubled_modulus()};
}
Z phase(const PhaseAddresses& a, const std::int64_t* images) {
    Z result{1,0};
    for (int d=0; d<3; ++d) {
        const U m=static_cast<U>(a.modulus[d]), left=static_cast<U>(a.left[d]);
        const U right=static_cast<U>(a.right[d]);
        const U shifted=add(static_cast<U>(a.transfer[d]),right)%m;
        const U g=mul(left,residue(images[d],m))%m;
        const U p=mul(shifted,residue(images[3+d],m))%m;
        const U s=mul(right,residue(images[6+d],m))%m;
        const U gp=add(g,p)%m;
        const U numerator=gp>=s ? gp-s : m-(s-gp);
        result=finite(result*character(numerator,m));
    }
    return result;
}
void accumulate(double value, double& sum, double& correction) {
    const double next=finite(sum+value);
    const double residual=std::abs(sum)>=std::abs(value)
        ? finite(finite(sum-next)+value) : finite(finite(value-next)+sum);
    correction=finite(correction+residual); sum=next;
}
void accumulate(Z value, Z& sum, Z& correction) {
    double real=sum.real(), imag=sum.imag(), real_c=correction.real(), imag_c=correction.imag();
    accumulate(value.real(),real,real_c); accumulate(value.imag(),imag,imag_c);
    sum={real,imag}; correction={real_c,imag_c};
}
} // namespace

BipoleErfcBlochPlan plan_bipole_erfc_bloch(const BasisSet& basis, const PeriodicSystem& system,
        const RegularKMesh& mesh, BipoleErfcImageView images, const Selection& s,
        const Options& o, const Inventory& inventory, const Caps& caps) {
    controls(inventory,caps);
    Plan p; p.n_basis=basis.nbasis(); p.n_kpoints=mesh.size(); p.image_count=images.image_count; p.selection=s;
    cap(p.n_kpoints,caps.maximum_kpoints,"BIPOLE erfc Bloch k-point cap");
    if (s.q_index>=p.n_kpoints || s.left_k_index>=p.n_kpoints || s.right_k_index>=p.n_kpoints)
        invalid("BIPOLE erfc Bloch q/grid index extent");
    p.output_elements=mul(s.left_pair_count,s.right_pair_count);
    p.retained_output_bytes=mul(16,p.output_elements); p.compensation_bytes=p.retained_output_bytes;
    p.fixed_numeric_workspace_bytes=ScalarWorkspace;
    if (o.require_image_permutation_closure) {
        if (!caps.maximum_support_image_comparisons)
            invalid("BIPOLE erfc Bloch requires a positive support image comparison cap");
        p.support_image_comparisons_upper_bound=mul(8,mul(images.image_count,images.image_count));
        cap(p.support_image_comparisons_upper_bound,caps.maximum_support_image_comparisons,
            "BIPOLE erfc Bloch support image comparison cap");
        p.support_work_units=add(4096,add(mul(128,images.image_count),
            mul(256,p.support_image_comparisons_upper_bound)));
        // Local quartet positions and one transformed image; no orbit table.
        p.fixed_numeric_workspace_bytes=add(p.fixed_numeric_workspace_bytes,256);
    }
    p.wrapper_control_storage_bytes=wrapper_controls();
    if (p.output_elements>std::vector<Z>().max_size() ||
        p.retained_output_bytes>static_cast<U>(std::numeric_limits<std::ptrdiff_t>::max()))
        throw std::length_error("BIPOLE erfc Bloch output address extent");
    p.phase_evaluations=images.image_count;
    p.folded_scalar_terms=mul(images.image_count,p.output_elements);
    cap(p.phase_evaluations,caps.maximum_phase_evaluations,"BIPOLE erfc Bloch phase evaluation cap");
    p.phase_work_units=add(4096,mul(4096,p.phase_evaluations));
    p.folding_work_units=add(mul(128,p.folded_scalar_terms),mul(128,p.output_elements));
    // Two full image receipt passes; output codec; per-image phase receipt;
    // constant owner/descriptor hashes. This is an abstract bound, not timing
    // or a claim that compensated complex sums are universally correctly rounded.
    const U image_bytes=mul(72,images.image_count), output_bytes=p.retained_output_bytes;
    cap(add(4096,image_bytes),ShaMaximumBytes,"BIPOLE erfc Bloch image SHA extent");
    cap(add(4096,mul(16,images.image_count)),ShaMaximumBytes,"BIPOLE erfc Bloch phase SHA extent");
    cap(add(4096,output_bytes),ShaMaximumBytes,"BIPOLE erfc Bloch output SHA extent");
    p.identity_work_units=mul(256,add(4096,add(mul(2,image_bytes),add(mul(16,images.image_count),output_bytes))));
    cap(wrapper_work(p),caps.maximum_work_units,"BIPOLE erfc Bloch phase/folding/identity work cap");
    cap(wrapper_resident(p),caps.maximum_owned_numerical_bytes,"BIPOLE erfc Bloch output/compensation owned cap");
    const auto child_inventory=raw_inventory(inventory,p);
    const auto child_caps=raw_caps(caps,p);
    p.raw=plan_bipole_erfc_panel(basis,system,images,raw_selection(s),o.raw,child_inventory,child_caps);
    if (p.raw.output_elements!=p.folded_scalar_terms || p.raw.n_basis!=p.n_basis)
        throw std::logic_error("BIPOLE erfc Bloch raw plan shape mismatch");
    p.raw_phase_owned_numerical_bytes=add(wrapper_resident(p),p.raw.peak_owned_numerical_bytes);
    p.fold_phase_owned_numerical_bytes=add(wrapper_resident(p),p.raw.retained_output_bytes);
    p.peak_owned_numerical_bytes=std::max(p.raw_phase_owned_numerical_bytes,p.fold_phase_owned_numerical_bytes);
    p.borrowed_numerical_bytes=p.raw.borrowed_numerical_bytes;
    p.control_storage_bytes=p.raw.control_storage_bytes;
    p.per_replica_inventoried_bytes=add(p.peak_owned_numerical_bytes,add(p.borrowed_numerical_bytes,
        add(p.control_storage_bytes,add(inventory.other_live_numerical_bytes_per_replica,inventory.backend_margin_bytes_per_replica))));
    p.required_node_inventoried_bytes=add(inventory.external_node_bytes,mul(inventory.numerical_replicas,p.per_replica_inventoried_bytes));
    // Native make first calls this planner, then raw make repeats its own
    // count-only plan. The metadata walk is charged for BOTH executions.
    p.work_units_upper_bound=add(wrapper_work(p),add(p.raw.work_units_upper_bound,p.raw.metadata_work_units));
    if (p.per_replica_inventoried_bytes!=p.raw.per_replica_inventoried_bytes ||
        p.required_node_inventoried_bytes!=p.raw.required_node_inventoried_bytes)
        throw std::logic_error("BIPOLE erfc Bloch raw/wrapper live union mismatch");
    cap(p.borrowed_numerical_bytes,caps.maximum_borrowed_numerical_bytes,"BIPOLE erfc Bloch borrowed numerical cap");
    cap(p.peak_owned_numerical_bytes,caps.maximum_owned_numerical_bytes,"BIPOLE erfc Bloch owned numerical cap");
    cap(p.control_storage_bytes,caps.maximum_control_storage_bytes,"BIPOLE erfc Bloch control cap");
    cap(p.per_replica_inventoried_bytes,caps.maximum_worker_bytes,"BIPOLE erfc Bloch worker cap");
    cap(p.required_node_inventoried_bytes,caps.maximum_node_bytes,"BIPOLE erfc Bloch node cap");
    cap(p.work_units_upper_bound,caps.maximum_work_units,"BIPOLE erfc Bloch work cap");
    return p;
}

BipoleErfcBlochResult make_bipole_erfc_bloch(const BasisSet& basis, const PeriodicSystem& system,
        const RegularKMesh& mesh, BipoleErfcImageView images, const Selection& selection,
        const Options& options, const Inventory& inventory, const Caps& caps) {
    const auto s=selection; const auto o=options; const auto i=inventory; const auto c=caps;
    BipoleErfcBlochResult result;
    result.plan_=plan_bipole_erfc_bloch(basis,system,mesh,images,s,o,i,c);
    const auto& p=result.plan_;
    // Full enclosing and nested admission is complete before the first label
    // read, numerical allocation, primitive scan or libint engine creation.
    const auto original_images=image_hash(images);
    if (o.require_image_permutation_closure) {
        result.image_permutation_support_identity_=certify_image_support(
            images,original_images,result.diagnostics_.support_image_comparisons);
        if (result.diagnostics_.support_image_comparisons!=p.support_image_comparisons_upper_bound)
            throw std::logic_error("BIPOLE erfc Bloch support comparison census mismatch");
    }
    const auto a=addresses(mesh,s);
    result.values_.assign(static_cast<std::size_t>(p.output_elements),Z{});
    std::vector<Z> correction(static_cast<std::size_t>(p.output_elements),Z{});
    {
        const auto raw=make_bipole_erfc_panel(basis,system,images,raw_selection(s),o.raw,raw_inventory(i,p),raw_caps(c,p));
        if (raw.memory().output_elements!=p.folded_scalar_terms ||
            raw.memory().peak_owned_numerical_bytes!=p.raw.peak_owned_numerical_bytes ||
            raw.memory().per_replica_inventoried_bytes!=p.per_replica_inventoried_bytes ||
            raw.memory().work_units_upper_bound!=p.raw.work_units_upper_bound)
            throw std::runtime_error("BIPOLE erfc Bloch raw source metadata changed during evaluation");
        result.diagnostics_.raw=raw.diagnostics();
        result.raw_input_identity_=raw.input_identity_sha256();
        result.raw_source_identity_=raw.source_identity_sha256();
        result.raw_payload_identity_=raw.payload_identity_sha256();
        result.input_identity_=input_hash(result.raw_input_identity_,original_images,mesh,s,
                                          result.image_permutation_support_identity_);
        Hasher source("vibeqc.bipole.erfc-bloch.source");
        source.text(result.input_identity_); source.text(result.raw_source_identity_); source.text(result.raw_payload_identity_);
        source.u64(p.image_count);
        const double* values=raw.data();
        for (U image=0; image<p.image_count; ++image) {
            const Z factor=phase(a,images.indices+9*image);
            source.complex(factor);
            result.diagnostics_.maximum_phase_modulus_residual=std::max(
                result.diagnostics_.maximum_phase_modulus_residual,finite(std::abs(std::abs(factor)-1)));
            ++result.diagnostics_.evaluated_images;
            for (U j=0; j<p.output_elements; ++j) {
                const double value=finite(values[image*p.output_elements+j]);
                const Z term=finite(Z{finite(factor.real()*value),finite(factor.imag()*value)});
                accumulate(term,result.values_[j],correction[j]);
                ++result.diagnostics_.folded_scalar_terms;
            }
        }
        result.source_identity_=source.finish();
    } // Raw real output is released; no engine, raw panel or child owner escapes.
    if (result.diagnostics_.evaluated_images!=p.phase_evaluations ||
        result.diagnostics_.folded_scalar_terms!=p.folded_scalar_terms)
        throw std::logic_error("BIPOLE erfc Bloch final work census mismatch");
    for (U j=0; j<p.output_elements; ++j) {
        result.values_[j]=finite(result.values_[j]+correction[j]);
        result.diagnostics_.maximum_integral_magnitude=std::max(
            result.diagnostics_.maximum_integral_magnitude,finite(std::abs(result.values_[j])));
    }
    if (image_hash(images)!=original_images)
        throw std::runtime_error("BIPOLE erfc Bloch original images changed during evaluation");
    Hasher payload("vibeqc.bipole.erfc-bloch.payload");
    payload.text(result.input_identity_); payload.text(result.source_identity_);
    payload.u64(s.left_pair_count); payload.u64(s.right_pair_count); payload.u64(p.output_elements);
    for (Z value : result.values_) payload.complex(value);
    result.payload_identity_=payload.finish();
    return result;
}

BipoleErfcBlochResult::BipoleErfcBlochResult(BipoleErfcBlochResult&& other) noexcept
    : plan_(other.plan_), diagnostics_(other.diagnostics_), values_(std::move(other.values_)),
      input_identity_(std::move(other.input_identity_)), source_identity_(std::move(other.source_identity_)),
      payload_identity_(std::move(other.payload_identity_)), raw_input_identity_(std::move(other.raw_input_identity_)),
      raw_source_identity_(std::move(other.raw_source_identity_)), raw_payload_identity_(std::move(other.raw_payload_identity_)),
      image_permutation_support_identity_(std::move(other.image_permutation_support_identity_)) {
    other.input_identity_.clear();
}
BipoleErfcBlochResult& BipoleErfcBlochResult::operator=(BipoleErfcBlochResult&& other) noexcept {
    if (this!=&other) {
        plan_=other.plan_; diagnostics_=other.diagnostics_; values_=std::move(other.values_);
        input_identity_=std::move(other.input_identity_); source_identity_=std::move(other.source_identity_);
        payload_identity_=std::move(other.payload_identity_); raw_input_identity_=std::move(other.raw_input_identity_);
        raw_source_identity_=std::move(other.raw_source_identity_); raw_payload_identity_=std::move(other.raw_payload_identity_);
        image_permutation_support_identity_=std::move(other.image_permutation_support_identity_);
        other.input_identity_.clear();
    }
    return *this;
}
const Z* BipoleErfcBlochResult::data() const {
    if (values_.size()!=plan_.output_elements || input_identity_.size()!=64 || source_identity_.size()!=64 ||
        payload_identity_.size()!=64 || raw_input_identity_.size()!=64 || raw_source_identity_.size()!=64 ||
        raw_payload_identity_.size()!=64 ||
        (!image_permutation_support_identity_.empty() && image_permutation_support_identity_.size()!=64))
        throw std::logic_error("BIPOLE erfc Bloch result is consumed or malformed");
    return values_.data();
}
Z BipoleErfcBlochResult::element(U left, U right) const {
    (void)data();
    if (left>=plan_.selection.left_pair_count || right>=plan_.selection.right_pair_count)
        throw std::out_of_range("BIPOLE erfc Bloch element index");
    return values_[left*plan_.selection.right_pair_count+right];
}
const std::string& BipoleErfcBlochResult::input_identity_sha256() const { (void)data(); return input_identity_; }
const std::string& BipoleErfcBlochResult::source_identity_sha256() const { (void)data(); return source_identity_; }
const std::string& BipoleErfcBlochResult::payload_identity_sha256() const { (void)data(); return payload_identity_; }
const std::string& BipoleErfcBlochResult::raw_input_identity_sha256() const { (void)data(); return raw_input_identity_; }
const std::string& BipoleErfcBlochResult::raw_source_identity_sha256() const { (void)data(); return raw_source_identity_; }
const std::string& BipoleErfcBlochResult::raw_payload_identity_sha256() const { (void)data(); return raw_payload_identity_; }
bool BipoleErfcBlochResult::image_permutation_support_certified() const {
    (void)data(); return !image_permutation_support_identity_.empty();
}
const std::string& BipoleErfcBlochResult::image_permutation_support_identity_sha256() const {
    (void)data(); return image_permutation_support_identity_;
}

} // namespace vibeqc
