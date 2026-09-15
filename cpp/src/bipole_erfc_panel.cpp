#include "vibeqc/bipole_erfc_panel.hpp"
#include "vibeqc/detail/sha256.hpp"
#include "vibeqc/init.hpp"
#include "vibeqc/periodic_auxiliary_fourier.hpp"

#include <libint2/engine.h>
#include <algorithm>
#include <array>
#include <cfloat>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace vibeqc {
namespace {
using U = std::uint64_t;
using Plan = BipoleErfcPanelPlan;
using Selection = BipoleErfcPanelSelection;
using Options = BipoleErfcPanelOptions;
using Inventory = BipoleErfcPanelInventory;
using Caps = BipoleErfcPanelCaps;
constexpr std::int64_t ExactInteger = std::int64_t{1} << 53;
// The reviewed libint2.13 erfc backend has a lazily allocated shared Boys
// table, including transient coexistence during table growth. Its private
// table/layout and allocator capacity are NOT an exact public memory API.
// Require this explicit floor in the caller's backend allowance; known
// Engine vectors/stack/scratch are charged independently below.
constexpr U MinimumBackendAllowance = U{8} << 20;
constexpr U ShaMaximumBytes = std::numeric_limits<U>::max()/8;

[[noreturn]] void invalid(const char* message) { throw std::invalid_argument(message); }
U add(U a, U b) {
    if (b > std::numeric_limits<U>::max()-a) throw std::overflow_error("BIPOLE erfc panel count overflow");
    return a+b;
}
U mul(U a, U b) {
    if (a && b > std::numeric_limits<U>::max()/a) throw std::overflow_error("BIPOLE erfc panel count overflow");
    return a*b;
}
U fourth(U a) { const U a2=mul(a,a); return mul(a2,a2); }
void cap(U value, U maximum, const char* message) {
    if (value > maximum) throw std::length_error(message);
}
double finite(double value) {
    if (!std::isfinite(value)) throw std::overflow_error("BIPOLE erfc panel nonfinite numerical value");
    return value;
}
void environment() {
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0) || FLT_EVAL_METHOD != 0
    invalid("BIPOLE erfc panel requires strict floating point compilation");
#endif
    if (!std::numeric_limits<double>::is_iec559 || sizeof(double)!=8 || FLT_RADIX!=2 ||
        std::numeric_limits<double>::digits!=53 || std::numeric_limits<double>::max_exponent!=1024 ||
        std::fegetround()!=FE_TONEAREST)
        invalid("BIPOLE erfc panel requires binary64 round-to-nearest");
    volatile double tiny=std::numeric_limits<double>::denorm_min(), one=1, zero=0;
    volatile double normal=std::numeric_limits<double>::min(), half=0.5;
    if (!(tiny>0) || tiny*one!=tiny || tiny+zero!=tiny || std::fma(tiny,one,zero)!=tiny ||
        std::nextafter(0.0,1.0)!=tiny || normal*half==0)
        invalid("BIPOLE erfc panel requires gradual underflow");
}
void controls(const Options& o, const Inventory& i, const Caps& c) {
    environment();
    // Libint forms omega^2 directly. Do not allow its finite positive input
    // to silently become zero or infinity before evaluating erfc.
    if (!std::isfinite(o.omega) || o.omega<=0 || !std::isfinite(o.omega*o.omega) || o.omega*o.omega==0)
        invalid("BIPOLE erfc panel requires positive finite representable omega squared");
    if (!i.numerical_replicas || i.backend_margin_bytes_per_replica<MinimumBackendAllowance)
        invalid("BIPOLE erfc panel requires replicas and an explicit backend allowance of at least 8 MiB");
    if (!c.maximum_images || !c.maximum_shell_quartet_calls || !c.maximum_primitive_quartet_visits ||
        !c.maximum_borrowed_numerical_bytes || !c.maximum_owned_numerical_bytes ||
        !c.maximum_control_storage_bytes || !c.maximum_worker_bytes || !c.maximum_node_bytes || !c.maximum_work_units)
        invalid("BIPOLE erfc panel requires positive explicit resource caps");
}
void image_shape(BipoleErfcImageView v) {
    if (v.element_count!=mul(9,v.image_count) || (v.element_count &&
        (!v.indices || reinterpret_cast<std::uintptr_t>(v.indices)%alignof(std::int64_t))))
        invalid("BIPOLE erfc panel image extent/alignment must be int64[image,3,3]");
    const U bytes=mul(8,v.element_count);
    if (bytes>static_cast<U>(std::numeric_limits<std::ptrdiff_t>::max()) ||
        (bytes && reinterpret_cast<std::uintptr_t>(v.indices)>std::numeric_limits<std::uintptr_t>::max()-bytes))
        throw std::overflow_error("BIPOLE erfc panel image address extent");
}
U geometry_controls(const PeriodicSystem& system) {
    U bytes=add(sizeof(PeriodicSystem),mul(system.unit_cell.size(),sizeof(Atom)));
    if (system.symmetry) {
        const auto& s=*system.symmetry;
        bytes=add(bytes,mul(s.operations.size(),sizeof(SymmetryOp)));
        bytes=add(bytes,mul(s.equivalent_atoms.size(),sizeof(int)));
        bytes=add(bytes,add(add(s.international_symbol.size(),1),add(s.point_group.size(),1)));
    }
    return bytes;
}
U fixed_controls() {
    return 65536+2*sizeof(Plan)+sizeof(BipoleErfcPanelResult)+sizeof(Selection)+sizeof(Options)+
        sizeof(Inventory)+sizeof(Caps)+sizeof(BipoleErfcImageView)+sizeof(libint2::Engine)+
        4*(sizeof(libint2::Shell)+sizeof(libint2::Shell::Contraction))+3*65;
}
class Hasher {
public:
    explicit Hasher(const char* domain) { text(domain); u64(1); }
    void bytes(const std::uint8_t* p, U n) {
        if (n>ShaMaximumBytes-size_) throw std::length_error("BIPOLE erfc panel SHA message extent");
        hash_.update(p,static_cast<std::size_t>(n)); size_+=n;
    }
    void u64(U x) {
        std::array<std::uint8_t,8> b{};
        for (unsigned d=0; d<8; ++d) b[d]=static_cast<std::uint8_t>(x>>(56-8*d));
        bytes(b.data(),8);
    }
    void bits(double x) {
        if (x==0) x=0;
        U value; std::memcpy(&value,&x,8); u64(value);
    }
    void real(double x) { bits(finite(x)); }
    void text(const char* value) { text(std::string(value)); }
    void text(const std::string& value) {
        u64(value.size()); bytes(reinterpret_cast<const std::uint8_t*>(value.data()),value.size());
    }
    std::string finish() { return hash_.finish_hex(); }
private:
    detail::Sha256 hash_;
    U size_=0;
};
std::string source_hash(const BasisSet& basis,const PeriodicSystem& system,const Options& options) {
    Hasher h("vibeqc.bipole.erfc-panel.source");
    h.text(auxiliary_basis_content_identity_sha256(basis));
    // max_ln_coeff is derived but consumed by libint screening machinery;
    // preserve its actual payload, including valid log(0)=-infinity entries.
    for (const auto& shell : basis.libint()) {
        h.u64(shell.max_ln_coeff.size());
        for (double value : shell.max_ln_coeff) h.bits(value);
    }
    h.u64(static_cast<U>(system.dim)); h.u64(static_cast<U>(system.charge)); h.u64(static_cast<U>(system.multiplicity));
    for (int a=0; a<3; ++a) for (int b=0; b<3; ++b) h.real(system.lattice(a,b));
    h.u64(system.unit_cell.size());
    for (const auto& atom : system.unit_cell) {
        h.u64(static_cast<U>(atom.Z)); for (double x : atom.xyz) h.real(x);
    }
    h.real(options.omega); h.text(LIBINT_VERSION);
    h.u64(LIBINT2_MAX_AM_eri); h.u64(LIBINT2_MAX_VECLEN);
    h.text("raw-erfc;xx_xx;screening-method-original;zero-primitive-screening;original-normalized-segmented-shells;no-domain-no-phase-no-repair;fma-original-lattice-v1");
    return h.finish();
}
std::string input_hash(const std::string& source,BipoleErfcImageView images,const Selection& s) {
    Hasher h("vibeqc.bipole.erfc-panel.input"); h.text(source); h.u64(images.image_count);
    for (U j=0; j<images.element_count; ++j) h.u64(static_cast<U>(images.indices[j]));
    for (U value : {s.left_pair_begin,s.left_pair_count,s.right_pair_begin,s.right_pair_count}) h.u64(value);
    return h.finish();
}
std::array<double,3> translated_origin(const libint2::Shell& shell,const PeriodicSystem& system,
                                     const std::int64_t* cell) {
    std::array<double,3> result{};
    for (int a=0; a<3; ++a) {
        double x=shell.O[a];
        for (int b=0; b<3; ++b) x=finite(std::fma(system.lattice(a,b),static_cast<double>(cell[b]),x));
        result[a]=x;
    }
    return result;
}
void validate_payload(const BasisSet& basis,const PeriodicSystem& system,BipoleErfcImageView images) {
    if (system.dim!=3 || !system.lattice.allFinite()) invalid("BIPOLE erfc panel requires finite original 3D lattice");
    // A scaled determinant avoids spurious over/underflow in a nonsingular
    // lattice test. The original, unscaled entries form every translation.
    const double scale=system.lattice.cwiseAbs().maxCoeff();
    if (!(scale>0)) invalid("BIPOLE erfc panel requires nonsingular original lattice");
    const Eigen::Matrix3d scaled=system.lattice/scale;
    const double determinant=scaled.determinant();
    if (!std::isfinite(determinant) || determinant==0) invalid("BIPOLE erfc panel requires numerically nonsingular original lattice");
    for (const auto& atom : system.unit_cell) for (double x : atom.xyz) finite(x);
    for (const auto& shell : basis.libint()) {
        for (double x : shell.O) finite(x);
        for (double exponent : shell.alpha)
            if (!std::isfinite(exponent) || exponent<=0) invalid("BIPOLE erfc panel requires positive finite primitive exponents");
        for (double coefficient : shell.contr[0].coeff) finite(coefficient);
        for (double x : shell.max_ln_coeff)
            if (std::isnan(x) || x==std::numeric_limits<double>::infinity())
                invalid("BIPOLE erfc panel invalid derived primitive screening payload");
    }
    for (U j=0; j<images.element_count; ++j)
        if (images.indices[j]<-ExactInteger || images.indices[j]>ExactInteger)
            invalid("BIPOLE erfc panel image label is not exactly representable in binary64");
    // Validate ALL requested translated original shell centers before output
    // or backend allocations, including empty selected AO intervals.
    for (U image=0; image<images.image_count; ++image)
        for (int role=0; role<3; ++role)
            for (const auto& shell : basis.libint())
                (void)translated_origin(shell,system,images.indices+9*image+3*role);
}
struct Location { U shell=0, local=0; };
Location locate(const BasisSet& basis,U ao) {
    U offset=0, index=0;
    for (const auto& shell : basis.libint()) {
        const U size=shell.size();
        if (ao-offset<size) return {index,ao-offset};
        offset+=size; ++index;
    }
    throw std::logic_error("BIPOLE erfc panel admitted AO lookup failed");
}
void result_shape(const Plan& p,const std::vector<double>& values,
                  const std::string& input,const std::string& source,const std::string& payload) {
    if (values.size()!=p.output_elements || input.size()!=64 || source.size()!=64 || payload.size()!=64)
        throw std::logic_error("BIPOLE erfc panel result is consumed or malformed");
}
} // namespace

BipoleErfcPanelPlan plan_bipole_erfc_panel(const BasisSet& basis,const PeriodicSystem& system,
        BipoleErfcImageView images,const Selection& s,const Options& o,const Inventory& inventory,const Caps& caps) {
    controls(o,inventory,caps); image_shape(images);
    Plan p; p.n_basis=basis.nbasis(); p.n_shells=basis.nshells(); p.image_count=images.image_count; p.selection=s;
    p.backend_maximum_angular_momentum=std::min<U>(6,LIBINT2_MAX_AM_eri);
    const U pairs=mul(p.n_basis,p.n_basis);
    if (!p.n_basis || !p.n_shells || s.left_pair_begin>pairs || s.left_pair_count>pairs-s.left_pair_begin ||
        s.right_pair_begin>pairs || s.right_pair_count>pairs-s.right_pair_begin)
        invalid("BIPOLE erfc panel AO pair interval/basis extent");
    cap(images.image_count,caps.maximum_images,"BIPOLE erfc panel image cap");
    p.output_elements=mul(images.image_count,mul(s.left_pair_count,s.right_pair_count));
    p.retained_output_bytes=mul(8,p.output_elements);
    cap(p.retained_output_bytes,caps.maximum_owned_numerical_bytes,"BIPOLE erfc panel output/owned cap");
    if (p.output_elements>std::vector<double>().max_size()) throw std::length_error("BIPOLE erfc panel output address extent");
    p.shell_quartet_calls_upper_bound=p.output_elements;
    cap(p.shell_quartet_calls_upper_bound,caps.maximum_shell_quartet_calls,"BIPOLE erfc panel shell call cap");
    p.borrowed_image_bytes=mul(72,images.image_count);
    p.borrowed_geometry_numeric_bytes=add(72,mul(28,system.unit_cell.size()));
    p.metadata_work_units=add(4096,mul(512,add(p.n_shells,system.unit_cell.size())));
    const U base_validation=add(4096,add(mul(256,images.element_count),
        add(mul(512,mul(images.image_count,p.n_shells)),mul(128,system.unit_cell.size()))));
    U lanes=0, counted_ao=0, max_shell_lanes=0;
    U basis_controls=add(sizeof(BasisSet),add(basis.name().size()+1,
        mul(p.n_shells,sizeof(libint2::Shell)+sizeof(libint2::Shell::Contraction)+sizeof(int)+sizeof(std::size_t))));
    auto admit_census=[&]() {
        cap(add(p.metadata_work_units,add(base_validation,mul(256,lanes))),caps.maximum_work_units,
            "BIPOLE erfc panel metadata/validation work cap");
        cap(add(mul(8,lanes),add(p.borrowed_image_bytes,p.borrowed_geometry_numeric_bytes)),
            caps.maximum_borrowed_numerical_bytes,"BIPOLE erfc panel borrowed numerical cap");
        cap(add(fixed_controls(),add(basis_controls,add(geometry_controls(system),inventory.other_live_control_bytes_per_replica))),
            caps.maximum_control_storage_bytes,"BIPOLE erfc panel control cap");
    };
    admit_census();
    for (const auto& shell : basis.libint()) {
        if (shell.alpha.empty() || shell.contr.size()!=1 || shell.max_ln_coeff.size()!=shell.alpha.size())
            invalid("BIPOLE erfc panel requires nonempty segmented shells and matching primitive metadata");
        const auto& c=shell.contr[0];
        if (c.l<0 || static_cast<U>(c.l)>p.backend_maximum_angular_momentum || (!c.pure && c.l!=0) ||
            c.coeff.size()!=shell.alpha.size())
            invalid("BIPOLE erfc panel supports segmented pure shells within generated ERI limit or Cartesian s");
        const U shell_lanes=add(3,add(shell.alpha.size(),add(c.coeff.size(),shell.max_ln_coeff.size())));
        lanes=add(lanes,shell_lanes); max_shell_lanes=std::max(max_shell_lanes,shell_lanes);
        p.maximum_primitives=std::max<U>(p.maximum_primitives,shell.alpha.size());
        p.maximum_angular_momentum=std::max<U>(p.maximum_angular_momentum,c.l);
        counted_ao=add(counted_ao,static_cast<U>(2*c.l+1));
        admit_census();
    }
    if (counted_ao!=p.n_basis) invalid("BIPOLE erfc panel AO/contraction census mismatch");
    p.borrowed_basis_numeric_bytes=mul(8,lanes);
    p.borrowed_numerical_bytes=add(p.borrowed_basis_numeric_bytes,add(p.borrowed_image_bytes,p.borrowed_geometry_numeric_bytes));
    p.validation_work_units=add(base_validation,mul(256,lanes));
    const U primitives=fourth(p.maximum_primitives);
    p.primitive_quartet_visits_upper_bound=mul(p.shell_quartet_calls_upper_bound,primitives);
    cap(p.primitive_quartet_visits_upper_bound,caps.maximum_primitive_quartet_visits,"BIPOLE erfc panel primitive quartet cap");
    const U cart=(p.maximum_angular_momentum+1)*(p.maximum_angular_momentum+2)/2;
    p.maximum_shell_quartet_elements=fourth(2*p.maximum_angular_momentum+1);
    p.fixed_numeric_workspace_bytes=4096;
    if (p.output_elements) {
        // Engine::initialize uses pow(pmax,4). Restrict its integer conversion
        // to an exactly represented/admitted binary64 range before calling it.
        if (primitives>static_cast<U>(ExactInteger) || primitives>std::vector<Libint_t>().max_size())
            throw std::length_error("BIPOLE erfc panel primitive engine address extent");
        p.engine_primitive_bytes=mul(primitives,sizeof(Libint_t));
        p.engine_pair_bytes=mul(mul(2,mul(p.maximum_primitives,p.maximum_primitives)),sizeof(libint2::ShellPair::PrimPairData));
        p.engine_stack_bytes=mul(sizeof(double),LIBINT2_PREFIXED_NAME(libint2_need_memory_eri)(static_cast<int>(p.maximum_angular_momentum)));
        // reset_scratch can own TWO full Cartesian quartet buffers, including
        // spherical transformation/reordering. Engine's borrowed result block
        // lives in these arrays or the generated stack, never an extra copy.
        p.engine_scratch_bytes=mul(16,fourth(cart));
        // Generic erfc core evaluator's owned Fm, Engine pack and the
        // per-evaluation copied functor: conservative three columns.
        p.engine_core_numeric_bytes=mul(24,4*p.maximum_angular_momentum+1);
        p.shell_copy_numeric_bytes=mul(32,max_shell_lanes);
    }
    p.peak_owned_numerical_bytes=p.retained_output_bytes;
    for (U bytes : {p.engine_primitive_bytes,p.engine_pair_bytes,p.engine_stack_bytes,p.engine_scratch_bytes,
                   p.engine_core_numeric_bytes,p.shell_copy_numeric_bytes,p.fixed_numeric_workspace_bytes})
        p.peak_owned_numerical_bytes=add(p.peak_owned_numerical_bytes,bytes);
    p.control_storage_bytes=add(fixed_controls(),add(basis_controls,
        add(geometry_controls(system),inventory.other_live_control_bytes_per_replica)));
    p.per_replica_inventoried_bytes=add(p.peak_owned_numerical_bytes,add(p.borrowed_numerical_bytes,
        add(p.control_storage_bytes,add(inventory.other_live_numerical_bytes_per_replica,inventory.backend_margin_bytes_per_replica))));
    p.required_node_inventoried_bytes=add(inventory.external_node_bytes,mul(inventory.numerical_replicas,p.per_replica_inventoried_bytes));
    // Charges four AO lookups, four shell copies, full returned-block finite
    // scans and a conservative angular recurrence/transform upper per
    // primitive. No density screening or skipped-call work discount.
    const U scalar_call=add(4096,add(mul(256,p.n_shells),add(mul(128,max_shell_lanes),mul(64,p.maximum_shell_quartet_elements))));
    const U primitive_work=mul(4096,mul(fourth(cart),4*p.maximum_angular_momentum+1));
    p.contraction_work_units=add(mul(p.shell_quartet_calls_upper_bound,scalar_call),mul(p.primitive_quartet_visits_upper_bound,primitive_work));
    p.work_units_upper_bound=add(p.metadata_work_units,add(p.validation_work_units,p.contraction_work_units));
    cap(p.work_units_upper_bound,caps.maximum_work_units,"BIPOLE erfc panel work cap");
    cap(p.peak_owned_numerical_bytes,caps.maximum_owned_numerical_bytes,"BIPOLE erfc panel owned numerical cap");
    cap(p.per_replica_inventoried_bytes,caps.maximum_worker_bytes,"BIPOLE erfc panel worker cap");
    cap(p.required_node_inventoried_bytes,caps.maximum_node_bytes,"BIPOLE erfc panel node cap");
    return p;
}

BipoleErfcPanelResult make_bipole_erfc_panel(const BasisSet& basis,const PeriodicSystem& system,
        BipoleErfcImageView images,const Selection& selection,const Options& options,
        const Inventory& inventory,const Caps& caps) {
    const auto o=options; const auto s=selection; const auto i=inventory; const auto c=caps;
    BipoleErfcPanelResult result;
    result.plan_=plan_bipole_erfc_panel(basis,system,images,s,o,i,c);
    const auto& p=result.plan_;
    validate_payload(basis,system,images);
    result.source_identity_=source_hash(basis,system,o);
    result.input_identity_=input_hash(result.source_identity_,images,s);
    result.values_.assign(static_cast<std::size_t>(p.output_elements),0.0);
    if (p.output_elements) {
        ensure_libint_initialized();
        libint2::Engine engine(libint2::Operator::erfc_coulomb,static_cast<std::size_t>(p.maximum_primitives),
            static_cast<int>(p.maximum_angular_momentum),0,0.0,o.omega,
            libint2::BraKet::xx_xx,libint2::ScreeningMethod::Original);
        if (engine.precision()!=0 || engine.screening_method()!=libint2::ScreeningMethod::Original ||
            engine.braket()!=libint2::BraKet::xx_xx)
            throw std::logic_error("BIPOLE erfc panel primitive screening policy mismatch");
        const auto& shells=basis.libint();
        for (U image=0; image<images.image_count; ++image)
            for (U left=0; left<s.left_pair_count; ++left)
                for (U right=0; right<s.right_pair_count; ++right) {
                    if (result.diagnostics_.shell_quartet_calls>=p.shell_quartet_calls_upper_bound)
                        throw std::logic_error("BIPOLE erfc panel admitted shell call census exceeded");
                    const U ab=s.left_pair_begin+left, cd=s.right_pair_begin+right;
                    const std::array<Location,4> at={locate(basis,ab/p.n_basis),locate(basis,ab%p.n_basis),
                        locate(basis,cd/p.n_basis),locate(basis,cd%p.n_basis)};
                    std::array<libint2::Shell,4> local={shells[at[0].shell],shells[at[1].shell],
                        shells[at[2].shell],shells[at[3].shell]};
                    for (int role=1; role<4; ++role)
                        local[role].O=translated_origin(shells[at[role].shell],system,images.indices+9*image+3*(role-1));
                    engine.compute(local[0],local[1],local[2],local[3]);
                    ++result.diagnostics_.shell_quartet_calls;
                    const double* block=engine.results()[0];
                    double value=0;
                    if (!block) ++result.diagnostics_.null_shell_quartet_buffers;
                    else {
                        const U n1=local[0].size(), n2=local[1].size(), n3=local[2].size(), n4=local[3].size();
                        const U elements=mul(mul(n1,n2),mul(n3,n4));
                        if (elements>p.maximum_shell_quartet_elements) throw std::logic_error("BIPOLE erfc panel returned block extent");
                        for (U x=0; x<elements; ++x) finite(block[x]);
                        value=block[((at[0].local*n2+at[1].local)*n3+at[2].local)*n4+at[3].local];
                    }
                    result.values_[(image*s.left_pair_count+left)*s.right_pair_count+right]=value;
                    result.diagnostics_.maximum_integral_magnitude=std::max(result.diagnostics_.maximum_integral_magnitude,std::abs(value));
                }
    }
    if (result.diagnostics_.shell_quartet_calls!=p.shell_quartet_calls_upper_bound)
        throw std::logic_error("BIPOLE erfc panel final shell call census mismatch");
    validate_payload(basis,system,images);
    const auto final_source=source_hash(basis,system,o);
    if (final_source!=result.source_identity_ || input_hash(final_source,images,s)!=result.input_identity_)
        throw std::runtime_error("BIPOLE erfc panel original input changed during evaluation");
    Hasher payload("vibeqc.bipole.erfc-panel.payload");
    payload.text(result.input_identity_); payload.u64(p.output_elements);
    for (double value : result.values_) payload.real(value);
    result.payload_identity_=payload.finish();
    return result;
}

const double* BipoleErfcPanelResult::data() const {
    result_shape(plan_,values_,input_identity_,source_identity_,payload_identity_); return values_.data();
}
double BipoleErfcPanelResult::element(U image,U left,U right) const {
    (void)data();
    if (image>=plan_.image_count || left>=plan_.selection.left_pair_count || right>=plan_.selection.right_pair_count)
        throw std::out_of_range("BIPOLE erfc panel element index");
    return values_[(image*plan_.selection.left_pair_count+left)*plan_.selection.right_pair_count+right];
}
const std::string& BipoleErfcPanelResult::input_identity_sha256() const { (void)data(); return input_identity_; }
const std::string& BipoleErfcPanelResult::source_identity_sha256() const { (void)data(); return source_identity_; }
const std::string& BipoleErfcPanelResult::payload_identity_sha256() const { (void)data(); return payload_identity_; }

} // namespace vibeqc
