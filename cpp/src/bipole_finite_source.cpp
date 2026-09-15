#include "vibeqc/bipole_finite_source.hpp"
#include "vibeqc/detail/sha256.hpp"
#include "vibeqc/detail/periodic_reciprocal_source.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace vibeqc {
namespace {
using U = std::uint64_t;
using Z = std::complex<double>;
using Selection = BipoleEwaldGramSelection;
using Inventory = BipoleFinitePanelInventory;
using Caps = BipoleFinitePanelCaps;
using Plan = BipoleFinitePanelPlan;
constexpr double Pi = 3.141592653589793238462643383279502884;
constexpr U ShaMaximumBytes = std::numeric_limits<U>::max()/8;
constexpr std::int64_t ExactLabel = std::int64_t{1}<<53;

[[noreturn]] void invalid(const char* message) { throw std::invalid_argument(message); }
U add(U a, U b) {
    if (b>std::numeric_limits<U>::max()-a) throw std::overflow_error("BIPOLE finite source count overflow");
    return a+b;
}
U mul(U a, U b) {
    if (a && b>std::numeric_limits<U>::max()/a) throw std::overflow_error("BIPOLE finite source count overflow");
    return a*b;
}
void cap(U value, U maximum, const char* message) {
    if (value>maximum) throw std::length_error(message);
}
double finite(double value) {
    if (!std::isfinite(value)) throw std::overflow_error("BIPOLE finite source nonfinite value");
    return value;
}
Z finite(Z value) { finite(value.real()); finite(value.imag()); return value; }
U residual(U maximum, U occupied, const char* message) {
    if (occupied>=maximum) throw std::length_error(message);
    return maximum-occupied;
}
class Hasher {
public:
    explicit Hasher(const char* domain) { text(domain); u64(1); }
    void bytes(const std::uint8_t* data, U n) {
        cap(n,ShaMaximumBytes-size_,"BIPOLE finite source SHA extent");
        hash_.update(data,static_cast<std::size_t>(n)); size_+=n;
    }
    void u64(U n) {
        std::array<std::uint8_t,8> wire{};
        for (unsigned j=0; j<8; ++j) wire[j]=static_cast<std::uint8_t>(n>>(56-8*j));
        bytes(wire.data(),8);
    }
    void bits(double value) { U wire; std::memcpy(&wire,&value,8); u64(wire); }
    void real(double value) { finite(value); bits(value==0 ? 0.0 : value); }
    void complex(Z value) { real(value.real()); real(value.imag()); }
    void text(const char* value) {
        const auto n=std::strlen(value); u64(n); bytes(reinterpret_cast<const std::uint8_t*>(value),n);
    }
    void text(const std::string& value) {
        u64(value.size()); bytes(reinterpret_cast<const std::uint8_t*>(value.data()),value.size());
    }
    std::string finish() { return hash_.finish_hex(); }
private:
    detail::Sha256 hash_;
    U size_=0;
};
void options(const BipoleFiniteSourceOptions& o) {
    // Reuse the strict binary64/rounding/underflow check of the shared scalar.
    (void)detail::bipole_ewald_weighted_product({0,0},{0,0},0);
    if (!std::isfinite(o.omega) || o.omega<=0 || !std::isfinite(o.reciprocal_energy_cutoff) ||
        o.reciprocal_energy_cutoff<=0 || o.reciprocal_energy_cutoff>std::numeric_limits<double>::max()/2)
        invalid("BIPOLE finite source requires positive finite omega/cutoff");
    if (o.zero_mode!=BipoleFiniteZeroMode::G0Omitted)
        invalid("BIPOLE finite source requires explicit G0Omitted zero-mode convention");
}
void shape(const std::int64_t* data, U rows, U width, U count) {
    if (count!=mul(rows,width) || (count && (!data || reinterpret_cast<std::uintptr_t>(data)%alignof(std::int64_t))))
        invalid("BIPOLE finite source requires aligned exact image/cell extents");
    cap(mul(count,8),std::numeric_limits<std::size_t>::max(),"BIPOLE finite source label address extent");
}
std::string labels_hash(const char* domain, const std::int64_t* data, U rows, U width) {
    Hasher h(domain); h.u64(rows); h.u64(width);
    for (U j=0; j<mul(rows,width); ++j) {
        if (data[j]<-ExactLabel || data[j]>ExactLabel)
            invalid("BIPOLE finite source requires exactly representable image/cell labels");
        h.u64(static_cast<U>(data[j]));
    }
    return h.finish();
}
std::string basis_hash(const BasisSet& basis) {
    Hasher h("vibeqc.bipole.finite-source.basis");
    h.u64(basis.nbasis()); h.u64(basis.nshells());
    for (const auto& shell : basis.libint()) {
        for (double x : shell.O) h.real(x);
        h.u64(shell.alpha.size()); h.u64(shell.contr.size());
        for (double x : shell.alpha) {
            if (!std::isfinite(x) || x<=0) invalid("BIPOLE finite source requires positive finite exponents");
            h.real(x);
        }
        for (const auto& c : shell.contr) {
            h.u64(c.l); h.u64(c.pure); h.u64(c.coeff.size());
            for (double x : c.coeff) h.real(x);
        }
        h.u64(shell.max_ln_coeff.size());
        for (double x : shell.max_ln_coeff) {
            if (std::isnan(x) || x==std::numeric_limits<double>::infinity())
                invalid("BIPOLE finite source invalid primitive screening metadata");
            h.bits(x==0 ? 0.0 : x); // log(0)=-infinity is valid backend metadata.
        }
    }
    return h.finish();
}
void require_unique_cells(AOPairFourierCellView cells) {
    for (U j=0; j<cells.cell_count; ++j) for (U k=0; k<j; ++k) {
        bool equal=true;
        for (U d=0; d<3; ++d) equal &= cells.indices[3*j+d]==cells.indices[3*k+d];
        if (equal) invalid("BIPOLE finite source requires unique explicit cells");
    }
}
PeriodicSystem system_for(const BipoleFiniteSource& source) {
    PeriodicSystem system; system.lattice=source.direct_lattice();
    return system; // Empty non-electronic atom/symmetry owners by policy.
}
BipoleErfcBlochOptions sr_options(const BipoleFiniteSource& source) {
    BipoleErfcBlochOptions o; o.raw.omega=source.options().omega;
    o.require_image_permutation_closure=source.options().require_image_permutation_closure;
    return o;
}
BipoleEwaldGramOptions lr_options(const BipoleFiniteSource& source, const Inventory& i) {
    BipoleEwaldGramOptions o; o.omega=source.options().omega;
    o.reciprocal_energy_cutoff=source.options().reciprocal_energy_cutoff;
    o.reciprocal_block_size=i.reciprocal_block_size;
    o.require_cell_inversion_closure=source.options().require_cell_inversion_closure;
    o.require_reciprocal_conjugacy=source.options().require_reciprocal_conjugacy;
    return o;
}
U resident(const Plan& p) {
    return add(p.fixed_numeric_workspace_bytes,add(p.retained_output_bytes,p.compensation_bytes));
}
U base_work(const Plan& p) {
    return add(p.wrapper_work_units,mul(2,add(p.source.work_units_upper_bound,
        add(p.zero_left.work_units,p.zero_right.work_units))));
}
BipoleErfcBlochInventory sr_inventory(const Inventory& i, const Plan& p) {
    BipoleErfcBlochInventory out=i;
    out.other_live_numerical_bytes_per_replica=add(i.other_live_numerical_bytes_per_replica,
        add(resident(p),p.source.borrowed_cell_bytes));
    out.other_live_control_bytes_per_replica=add(i.other_live_control_bytes_per_replica,p.wrapper_control_storage_bytes);
    return out;
}
BipoleEwaldGramInventory lr_inventory(const Inventory& i, const Plan& p) {
    BipoleEwaldGramInventory out;
    out.numerical_replicas=i.numerical_replicas; out.external_node_bytes=i.external_node_bytes;
    out.backend_margin_bytes_per_replica=i.backend_margin_bytes_per_replica;
    out.other_live_numerical_bytes_per_replica=add(i.other_live_numerical_bytes_per_replica,
        add(resident(p),p.source.borrowed_image_bytes));
    out.other_live_control_bytes_per_replica=add(i.other_live_control_bytes_per_replica,p.wrapper_control_storage_bytes);
    return out;
}
BipoleErfcBlochCaps sr_caps(const Caps& c, const Plan& p) {
    auto out=c.short_range;
    out.maximum_owned_numerical_bytes=std::min(out.maximum_owned_numerical_bytes,
        residual(c.maximum_owned_numerical_bytes,resident(p),"BIPOLE finite panel SR owned cap"));
    out.maximum_borrowed_numerical_bytes=std::min(out.maximum_borrowed_numerical_bytes,c.maximum_borrowed_numerical_bytes);
    out.maximum_control_storage_bytes=std::min(out.maximum_control_storage_bytes,c.maximum_control_storage_bytes);
    out.maximum_worker_bytes=std::min(out.maximum_worker_bytes,c.maximum_worker_bytes);
    out.maximum_node_bytes=std::min(out.maximum_node_bytes,c.maximum_node_bytes);
    out.maximum_work_units=std::min(out.maximum_work_units,
        residual(c.maximum_work_units,base_work(p),"BIPOLE finite panel SR work cap")/2);
    return out;
}
BipoleEwaldGramCaps lr_caps(const Caps& c, const Plan& p) {
    auto out=c.long_range;
    out.maximum_owned_numerical_bytes=std::min(out.maximum_owned_numerical_bytes,
        residual(c.maximum_owned_numerical_bytes,resident(p),"BIPOLE finite panel LR owned cap"));
    out.maximum_borrowed_numerical_bytes=std::min(out.maximum_borrowed_numerical_bytes,c.maximum_borrowed_numerical_bytes);
    out.maximum_control_storage_bytes=std::min(out.maximum_control_storage_bytes,c.maximum_control_storage_bytes);
    out.maximum_per_replica_inventoried_bytes=std::min(out.maximum_per_replica_inventoried_bytes,c.maximum_worker_bytes);
    out.maximum_node_inventoried_bytes=std::min(out.maximum_node_inventoried_bytes,c.maximum_node_bytes);
    out.maximum_work_units=std::min(out.maximum_work_units,residual(c.maximum_work_units,
        add(base_work(p),mul(2,p.short_range.work_units_upper_bound)),"BIPOLE finite panel LR work cap")/2);
    return out;
}
void accumulate(double x, double& sum, double& correction) {
    const double next=finite(sum+x);
    const double error=std::abs(sum)>=std::abs(x) ? finite(finite(sum-next)+x) : finite(finite(x-next)+sum);
    correction=finite(correction+error); sum=next;
}
void accumulate(Z x, Z& sum, Z& correction) {
    finite(x); double r=sum.real(), i=sum.imag(), cr=correction.real(), ci=correction.imag();
    accumulate(x.real(),r,cr); accumulate(x.imag(),i,ci); sum={r,i}; correction={cr,ci};
}
void selection_hash(Hasher& h, const Selection& s) {
    for (U x : {s.q_index,s.left_k_index,s.right_k_index,s.left_pair_begin,
                 s.left_pair_count,s.right_pair_begin,s.right_pair_count}) h.u64(x);
}
} // namespace

BipoleFiniteSourcePlan plan_bipole_finite_source(const BasisSet& basis, const RegularKMesh& mesh,
        BipoleErfcImageView images, AOPairFourierCellView cells,
        const BipoleFiniteSourceOptions& o, const BipoleFiniteSourceCaps& c) {
    options(o); shape(images.indices,images.image_count,9,images.element_count);
    shape(cells.indices,cells.cell_count,3,cells.element_count);
    if (!c.maximum_kpoints || !c.maximum_images || !c.maximum_cells || !c.maximum_context_storage_bytes ||
        !c.maximum_borrowed_numerical_bytes || !c.maximum_work_units)
        invalid("BIPOLE finite source requires positive explicit caps");
    BipoleFiniteSourcePlan p;
    p.n_basis=basis.nbasis(); p.n_shells=basis.nshells(); p.n_kpoints=mesh.size();
    p.image_count=images.image_count; p.cell_count=cells.cell_count;
    p.left_cell_count=p.right_cell_count=cells.cell_count;
    p.context_storage_bytes=sizeof(BipoleFiniteSource)+5*65+256; // fixed digest/shared-owner allowance
    cap(p.context_storage_bytes,c.maximum_context_storage_bytes,"BIPOLE finite source context storage cap");
    cap(p.n_kpoints,c.maximum_kpoints,"BIPOLE finite source k-point cap");
    cap(p.image_count,c.maximum_images,"BIPOLE finite source image cap");
    cap(p.cell_count,c.maximum_cells,"BIPOLE finite source cell cap");
    p.borrowed_image_bytes=mul(72,p.image_count); p.borrowed_cell_bytes=mul(24,p.cell_count);
    const U labels=add(p.borrowed_image_bytes,p.borrowed_cell_bytes);
    cap(add(72,labels),c.maximum_borrowed_numerical_bytes,"BIPOLE finite source borrowed cap");
    const U label_work=add(8192,mul(256,add(labels,mul(p.cell_count,p.cell_count))));
    AOPairFourierCellPanelCaps census;
    census.maximum_cells=c.maximum_cells; census.maximum_pair_cell_visits=1; census.maximum_output_bytes=1;
    census.maximum_work_units=residual(c.maximum_work_units,label_work,"BIPOLE finite source work cap")/4;
    p.basis_census=plan_ao_pair_gaussian_fourier_cell_panel(basis,0,0,0,p.cell_count,census);
    p.borrowed_basis_numeric_bytes=p.basis_census.borrowed_basis_numeric_bytes;
    p.borrowed_numerical_bytes=add(72,add(p.borrowed_basis_numeric_bytes,labels));
    p.work_units_upper_bound=add(label_work,mul(4,p.basis_census.work_units));
    cap(p.borrowed_numerical_bytes,c.maximum_borrowed_numerical_bytes,"BIPOLE finite source borrowed cap");
    cap(p.work_units_upper_bound,c.maximum_work_units,"BIPOLE finite source work cap");
    cap(add(8192,mul(16,p.borrowed_numerical_bytes)),ShaMaximumBytes,"BIPOLE finite source SHA cap");
    return p;
}

std::shared_ptr<BipoleFiniteSource> make_bipole_finite_source(const BasisSet& basis,
        const PeriodicSystem& system, const RegularKMesh& mesh, BipoleErfcImageView images,
        AOPairFourierCellView cells, const BipoleFiniteSourceOptions& o, const BipoleFiniteSourceCaps& c) {
    const auto p=plan_bipole_finite_source(basis,mesh,images,cells,o,c);
    if (system.dim!=3 || !system.lattice.allFinite()) invalid("BIPOLE finite source requires finite original 3D lattice");
    const Eigen::Matrix3d reciprocal=system.reciprocal_lattice();
    if (!reciprocal.allFinite()) invalid("BIPOLE finite source requires finite nonsingular lattice");
    const double scale=reciprocal.cwiseAbs().maxCoeff();
    if (!(scale>0)) invalid("BIPOLE finite source requires nonsingular reciprocal lattice");
    // O(1) geometry preparation only: no reciprocal records are visited here.
    // Use the exact same fixed-FMA cell volume as every LR numerical tile,
    // rather than introduce a second determinant/rounding convention.
    const auto reciprocal_geometry=detail::prepare_periodic_reciprocal_numeric_source(
        reciprocal,Eigen::Vector3d::Zero(),o.reciprocal_energy_cutoff,true,std::numeric_limits<U>::max());
    const double volume=reciprocal_geometry.geometry.cell_volume;
    const Eigen::Matrix3d reciprocal_check=system.lattice.transpose()*reciprocal;
    const double conditioning=system.lattice.cwiseAbs().maxCoeff()*scale;
    if (!reciprocal_check.allFinite() || !std::isfinite(conditioning) ||
        (reciprocal_check-2*Pi*Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff()>
            128*std::numeric_limits<double>::epsilon()*std::max(1.0,conditioning))
        invalid("BIPOLE finite source inconsistent original reciprocal lattice");
    // Avoid overflow of omega^2 or a transient reciprocal volume. The final
    // positive binary64 coefficient itself must be representable; fail closed
    // rather than silently erase a zero-mode term before its overlap product.
    int ev=0, ew=0;
    const double mv=std::frexp(volume,&ev), mw=std::frexp(o.omega,&ew);
    const double coefficient=finite(std::scalbn(Pi/(mv*mw*mw),-ev-2*ew));
    if (!(coefficient>0)) invalid("BIPOLE finite source zero-mode coefficient is not representable");
    const auto bd=basis_hash(basis);
    const auto id=labels_hash("vibeqc.bipole.finite-source.images",images.indices,p.image_count,9);
    const auto cd=labels_hash("vibeqc.bipole.finite-source.cells",cells.indices,p.cell_count,3);
    require_unique_cells(cells);
    Hasher h("vibeqc.bipole.finite-source.context");
    h.text(bd); h.text(id); h.text(cd);
    for (int a=0; a<3; ++a) for (int b=0; b<3; ++b) { h.real(system.lattice(a,b)); h.real(reciprocal(a,b)); }
    for (int d=0; d<3; ++d) { h.u64(mesh.mesh()[d]); h.u64(mesh.is_shift()[d]); }
    h.real(o.omega); h.real(o.reciprocal_energy_cutoff); h.real(volume); h.real(coefficient);
    h.u64(static_cast<U>(o.zero_mode)); h.u64(o.require_image_permutation_closure);
    h.u64(o.require_cell_inversion_closure); h.u64(o.require_reciprocal_conjugacy);
    h.text(LIBINT_VERSION); h.u64(LIBINT2_MAX_AM_eri); h.u64(LIBINT2_MAX_VECLEN);
    h.text("original-AO;3d;inverse-Bloch;ordered-common-SR-images;explicit-LR-overlap-cells;"
           "native-raw-erfc-original-screening-zero-precision;native-reciprocal-fma-q-slack-v1;"
           "G0-omitted;subtract-pi/Omega/omega^2;no-Nk-spin-probe-charge-no-repair;"
           "no-atom-charge-symmetry-source-inputs;selected-finite-reference-v1");
    const auto identity=h.finish();
    auto out=std::shared_ptr<BipoleFiniteSource>(new BipoleFiniteSource);
    out->direct_=system.lattice; out->reciprocal_=reciprocal; out->mesh_=mesh;
    out->options_=o; out->plan_=p; out->volume_=volume; out->zero_coefficient_=coefficient;
    out->basis_identity_=bd; out->images_identity_=id; out->cells_identity_=cd; out->source_identity_=identity;
    return out;
}

void BipoleFiniteSource::verify_inputs(const BasisSet& basis, BipoleErfcImageView images,
        AOPairFourierCellView cells, const BipoleFiniteSourceCaps& caps) const {
    if (product_resolved()) invalid("BIPOLE product source requires both product supports");
    (void)plan_bipole_finite_source(basis,mesh_,images,cells,options_,caps);
    if (basis_hash(basis)!=basis_identity_ ||
        labels_hash("vibeqc.bipole.finite-source.images",images.indices,images.image_count,9)!=images_identity_ ||
        labels_hash("vibeqc.bipole.finite-source.cells",cells.indices,cells.cell_count,3)!=cells_identity_)
        invalid("BIPOLE finite source borrowed basis/image/cell content does not match immutable source");
}

namespace {
void product_domain_shape(const BipoleFiniteProductDomain& d, U n) {
    const U n2=mul(n,n);
    for (const auto& axis : {std::array<U,2>{d.left_pair_begin,d.left_pair_count},
                             std::array<U,2>{d.right_pair_begin,d.right_pair_count}})
        if (!axis[1] || axis[0]>=n2 || axis[1]>n2-axis[0])
            invalid("BIPOLE product source requires positive AO-pair domains inside the basis");
}
void product_selection(const BipoleFiniteSource& source, const Selection& s,
                       AOPairFourierCellView left, AOPairFourierCellView right) {
    if (!source.product_resolved()) {
        if (left.indices!=right.indices || left.cell_count!=right.cell_count || left.element_count!=right.element_count)
            invalid("BIPOLE shared source requires the same cell view on both axes");
        return;
    }
    const auto& d=source.product_domain();
    for (const auto& axis : {std::array<U,4>{s.left_pair_begin,s.left_pair_count,d.left_pair_begin,d.left_pair_count},
                             std::array<U,4>{s.right_pair_begin,s.right_pair_count,d.right_pair_begin,d.right_pair_count}})
        if (axis[0]<axis[2] || axis[0]-axis[2]>axis[3] || axis[1]>axis[3]-(axis[0]-axis[2]))
            invalid("BIPOLE finite panel selection outside immutable product domain");
}
}

BipoleFiniteSourcePlan plan_bipole_finite_product_source(const BasisSet& basis,
        const RegularKMesh& mesh, BipoleErfcImageView images,
        AOPairFourierCellView left, AOPairFourierCellView right,
        const BipoleFiniteProductDomain& domain, const BipoleFiniteSourceOptions& o,
        const BipoleFiniteSourceCaps& c) {
    product_domain_shape(domain,basis.nbasis());
    auto p=plan_bipole_finite_source(basis,mesh,images,left,o,c);
    const auto r=plan_bipole_finite_source(basis,mesh,images,right,o,c);
    p.product_resolved=true; p.product_domain=domain;
    p.left_cell_count=left.cell_count; p.right_cell_count=right.cell_count;
    const bool shared=left.indices==right.indices && left.cell_count==right.cell_count;
    p.cell_count=shared ? left.cell_count : add(left.cell_count,right.cell_count);
    p.borrowed_cell_bytes=mul(24,p.cell_count);
    p.borrowed_numerical_bytes=add(72,add(p.borrowed_basis_numeric_bytes,
        add(p.borrowed_image_bytes,p.borrowed_cell_bytes)));
    // Both full validation/census envelopes plus fixed domain/hash work.
    // Repeated shared roles are deliberately conservative in work, not bytes.
    p.work_units_upper_bound=add(65536,add(p.work_units_upper_bound,r.work_units_upper_bound));
    cap(p.borrowed_numerical_bytes,c.maximum_borrowed_numerical_bytes,"BIPOLE product source borrowed cap");
    cap(p.work_units_upper_bound,c.maximum_work_units,"BIPOLE product source work cap");
    cap(add(8192,mul(16,p.borrowed_numerical_bytes)),ShaMaximumBytes,"BIPOLE product source SHA cap");
    return p;
}

std::shared_ptr<BipoleFiniteSource> make_bipole_finite_product_source(const BasisSet& basis,
        const PeriodicSystem& system, const RegularKMesh& mesh, BipoleErfcImageView images,
        AOPairFourierCellView left, AOPairFourierCellView right,
        const BipoleFiniteProductDomain& domain, const BipoleFiniteSourceOptions& o,
        const BipoleFiniteSourceCaps& c) {
    const auto p=plan_bipole_finite_product_source(basis,mesh,images,left,right,domain,o,c);
    const auto rd=labels_hash("vibeqc.bipole.finite-source.cells",right.indices,right.cell_count,3);
    require_unique_cells(right);
    auto out=make_bipole_finite_source(basis,system,mesh,images,left,o,c);
    Hasher h("vibeqc.bipole.finite-product-source.context");
    h.text(out->source_identity_sha256()); h.text(rd);
    for (U x : {domain.left_pair_begin,domain.left_pair_count,domain.right_pair_begin,domain.right_pair_count}) h.u64(x);
    h.text("source-owned-AO-pair-ranges;ordered-left-right-LR-and-zero-cells;declared-SR-images;no-cutoff-certificate-v1");
    out->source_identity_=h.finish(); out->right_cells_identity_=rd; out->plan_=p;
    return out;
}

void BipoleFiniteSource::verify_product_inputs(const BasisSet& basis, BipoleErfcImageView images,
        AOPairFourierCellView left, AOPairFourierCellView right, const BipoleFiniteSourceCaps& caps) const {
    if (!product_resolved()) invalid("BIPOLE product input verification requires a product source");
    (void)plan_bipole_finite_product_source(basis,mesh_,images,left,right,product_domain(),options_,caps);
    if (basis_hash(basis)!=basis_identity_ ||
        labels_hash("vibeqc.bipole.finite-source.images",images.indices,images.image_count,9)!=images_identity_ ||
        labels_hash("vibeqc.bipole.finite-source.cells",left.indices,left.cell_count,3)!=cells_identity_ ||
        labels_hash("vibeqc.bipole.finite-source.cells",right.indices,right.cell_count,3)!=right_cells_identity_)
        invalid("BIPOLE product source borrowed basis/image/left/right content does not match immutable source");
}

BipoleFiniteSupportPlan plan_bipole_finite_support_action(const BipoleFiniteSource& source,
        const BasisSet& basis, BipoleErfcImageView images, AOPairFourierCellView cells,
        const BipoleFiniteSupportCaps& caps) {
    if (source.product_resolved()) invalid("BIPOLE common-source consumer requires a common source, not a product domain");
    if (!caps.maximum_label_comparisons || !caps.maximum_inventoried_bytes || !caps.maximum_work_units)
        invalid("BIPOLE finite support requires positive explicit caps");
    shape(images.indices,images.image_count,9,images.element_count);
    shape(cells.indices,cells.cell_count,3,cells.element_count);
    BipoleFiniteSupportPlan p;
    // Two multiplicity counts per source row, for two 3-label pair views
    // and one 9-label quartet view. No early exit hides later bad labels.
    p.label_comparisons=add(mul(12,mul(cells.cell_count,cells.cell_count)),
                            mul(18,mul(images.image_count,images.image_count)));
    cap(p.label_comparisons,caps.maximum_label_comparisons,"BIPOLE finite support comparison cap");
    p.fixed_workspace_bytes=32768+2*sizeof(BipoleFiniteSupportPlan)+
        sizeof(BipoleFiniteSupportDiagnostics)+2*sizeof(BipoleFiniteSupportAction)+
        2*sizeof(BipoleFiniteSupportCaps);
    cap(p.fixed_workspace_bytes,caps.maximum_inventoried_bytes,"BIPOLE finite support workspace cap");
    const U work=add(65536,add(mul(64,p.label_comparisons),
                              mul(4096,add(cells.cell_count,images.image_count))));
    auto sc=caps.source;
    sc.maximum_borrowed_numerical_bytes=std::min(sc.maximum_borrowed_numerical_bytes,
        residual(caps.maximum_inventoried_bytes,p.fixed_workspace_bytes,"BIPOLE finite support borrowed cap"));
    sc.maximum_work_units=std::min(sc.maximum_work_units,
        residual(caps.maximum_work_units,work,"BIPOLE finite support work cap")/2);
    p.source=plan_bipole_finite_source(basis,source.mesh(),images,cells,source.options(),sc);
    p.inventoried_bytes=add(p.fixed_workspace_bytes,add(source.memory().context_storage_bytes,
        add(p.source.borrowed_numerical_bytes,p.source.basis_census.basis_control_storage_bytes)));
    p.work_units_upper_bound=add(work,mul(2,p.source.work_units_upper_bound));
    cap(p.inventoried_bytes,caps.maximum_inventoried_bytes,"BIPOLE finite support enclosing memory cap");
    cap(p.work_units_upper_bound,caps.maximum_work_units,"BIPOLE finite support enclosing work cap");
    return p;
}

BipoleFiniteSupportDiagnostics audit_bipole_finite_support_action(const BipoleFiniteSource& source,
        const BasisSet& basis, BipoleErfcImageView images, AOPairFourierCellView cells,
        const BipoleFiniteSupportAction& action, const BipoleFiniteSupportCaps& caps) {
    BipoleFiniteSupportDiagnostics out;
    out.memory=plan_bipole_finite_support_action(source,basis,images,cells,caps);
    source.verify_inputs(basis,images,cells,caps.source);
    // int32 W: a determinant needs at most 96 signed bits. int64 labels
    // times int32 W likewise fit in 128 bits; only the final cell is cast.
    // In particular, neither abs(INT64_MIN) nor overflowing int64 products
    // are used, and large cancelling terms retain their exact value.
    using Wide = __int128;
    const auto& w=action.rotation;
    Wide determinant=0;
    for (int j=0; j<3; ++j) {
        const int a=(j+1)%3, b=(j+2)%3;
        determinant+=Wide(w[j])*(Wide(w[3+a])*w[6+b]-Wide(w[3+b])*w[6+a]);
    }
    if (determinant!=1 && determinant!=-1)
        invalid("BIPOLE finite support requires a unimodular integer rotation");
    for (auto ell : action.atom_shifts)
        if (ell < -ExactLabel || ell > ExactLabel)
            invalid("BIPOLE finite support requires exactly representable atom shifts");
    auto transform=[&](const std::int64_t* original, int anchor, int center, std::int64_t* target) {
        for (int d=0; d<3; ++d) {
            Wide value=Wide(action.atom_shifts[3*anchor+d])-action.atom_shifts[3*center+d];
            for (int e=0; e<3; ++e) value+=Wide(w[3*d+e])*original[e];
            if (value<std::numeric_limits<std::int64_t>::min() || value>std::numeric_limits<std::int64_t>::max())
                throw std::overflow_error("BIPOLE finite support transformed cell exceeds int64");
            target[d]=static_cast<std::int64_t>(value);
        }
    };
    auto multiplicity=[&](const std::int64_t* data, U count, U width, const std::int64_t* label) {
        U found=0;
        for (U row=0; row<count; ++row) {
            bool equal=true;
            for (U d=0; d<width; ++d) {
                equal &= data[row*width+d]==label[d];
                ++out.label_comparisons;
            }
            found+=equal;
        }
        return found;
    };
    auto pair=[&](int anchor, int center, std::array<std::int64_t,3>& missing) {
        bool closed=true;
        for (U row=0; row<cells.cell_count; ++row) {
            const auto* original=cells.indices+3*row;
            std::array<std::int64_t,3> target{};
            transform(original,anchor,center,target.data());
            const auto before=multiplicity(cells.indices,cells.cell_count,3,original);
            const auto after=multiplicity(cells.indices,cells.cell_count,3,target.data());
            if (before!=after) { if (closed) missing=target; closed=false; }
        }
        return closed;
    };
    out.left_cells_closed=pair(0,1,out.first_left_cell);
    out.right_cells_closed=pair(2,3,out.first_right_cell);
    out.images_closed=true;
    for (U row=0; row<images.image_count; ++row) {
        const auto* original=images.indices+9*row;
        std::array<std::int64_t,9> target{};
        for (int center=1; center<4; ++center)
            transform(original+3*(center-1),0,center,target.data()+3*(center-1));
        const auto before=multiplicity(images.indices,images.image_count,9,original);
        const auto after=multiplicity(images.indices,images.image_count,9,target.data());
        if (before!=after) { if (out.images_closed) out.first_image=target; out.images_closed=false; }
    }
    if (out.label_comparisons!=out.memory.label_comparisons)
        throw std::logic_error("BIPOLE finite support comparison census mismatch");
    out.source_identity_sha256=source.source_identity_sha256();
    Hasher hash("vibeqc.bipole.finite-source.declared-support-action");
    hash.text(out.source_identity_sha256);
    for (auto value : w) hash.u64(static_cast<U>(static_cast<std::int64_t>(value)));
    for (auto ell : action.atom_shifts) hash.u64(static_cast<U>(ell));
    hash.text("W.R+ell_anchor-ell_center;exact-multisets;no-torus-wrap;declared-map-only-v1");
    out.action_identity_sha256=hash.finish();
    return out;
}

BipoleFiniteMappedSupportPlan plan_bipole_finite_mapped_support(
        const BipoleFiniteSource& source, const BasisSet& basis, const PeriodicSystem& system,
        const SymmetryOp& op, BipoleErfcImageView images, AOPairFourierCellView cells,
        const BipoleFiniteMappedSupportSelection& selection,
        const PeriodicAOBlochTransportOptions& options,
        const PeriodicAOBlochTransportInventory& inventory,
        const BipoleFiniteMappedSupportCaps& caps) {
    if (!caps.maximum_per_replica_inventoried_bytes || !caps.maximum_node_inventoried_bytes ||
        !caps.maximum_work_units)
        invalid("BIPOLE mapped support requires positive enclosing caps");
    for (auto shell : selection.source_shells)
        if (shell >= basis.nshells()) throw std::out_of_range("BIPOLE mapped support selected shell index");
    BipoleFiniteMappedSupportPlan p;
    p.fixed_workspace_bytes=32768+4*(sizeof(p)+sizeof(BipoleFiniteMappedSupportDiagnostics)+
        sizeof(selection)+sizeof(options)+sizeof(inventory)+sizeof(caps));
    cap(p.fixed_workspace_bytes,caps.maximum_per_replica_inventoried_bytes,
        "BIPOLE mapped support fixed workspace cap");
    p.support=plan_bipole_finite_support_action(source,basis,images,cells,caps.support);
    p.transport=plan_periodic_ao_bloch_transport(basis,system,op,source.mesh(),
        selection.source_k_index,selection.time_reversal,0,options,inventory,caps.transport);
    // Retained mappings coexist with the support audit. Summing full child
    // envelopes also charges shared borrowed basis/geometry twice by role.
    p.per_replica_inventoried_bytes=add(p.fixed_workspace_bytes,
        add(p.support.inventoried_bytes,p.transport.per_replica_inventoried_bytes));
    p.required_node_inventoried_bytes=add(inventory.external_node_bytes,
        mul(inventory.numerical_replicas,p.per_replica_inventoried_bytes));
    const U hash_work=add(65536,mul(256,add(p.transport.n_atoms,p.transport.n_shells)));
    p.work_units_upper_bound=add(hash_work,mul(2,
        add(p.support.source.work_units_upper_bound,
            add(p.support.work_units_upper_bound,p.transport.work_units_upper_bound))));
    cap(p.per_replica_inventoried_bytes,caps.maximum_per_replica_inventoried_bytes,
        "BIPOLE mapped support enclosing per-replica cap");
    cap(p.required_node_inventoried_bytes,caps.maximum_node_inventoried_bytes,
        "BIPOLE mapped support enclosing node cap");
    cap(p.work_units_upper_bound,caps.maximum_work_units,"BIPOLE mapped support enclosing work cap");
    return p;
}

BipoleFiniteMappedSupportDiagnostics audit_bipole_finite_mapped_support(
        const BipoleFiniteSource& source, const BasisSet& basis, const PeriodicSystem& system,
        const SymmetryOp& op, BipoleErfcImageView images, AOPairFourierCellView cells,
        const BipoleFiniteMappedSupportSelection& selection,
        const PeriodicAOBlochTransportOptions& options,
        const PeriodicAOBlochTransportInventory& inventory,
        const BipoleFiniteMappedSupportCaps& caps) {
    BipoleFiniteMappedSupportDiagnostics out;
    out.memory=plan_bipole_finite_mapped_support(source,basis,system,op,images,cells,
        selection,options,inventory,caps);
    if (system.dim!=3 || !(system.lattice.array()==source.direct_lattice().array()).all())
        invalid("BIPOLE mapped support geometry must use the exact finite-source 3D lattice");
    source.verify_inputs(basis,images,cells,caps.support.source);
    const auto transport=apply_periodic_ao_bloch_operation(basis,system,op,source.mesh(),
        selection.source_k_index,selection.time_reversal,nullptr,0,0,options,inventory,caps.transport);
    out.transport=transport.diagnostics();
    BipoleFiniteSupportAction action;
    for (int d=0; d<3; ++d) for (int e=0; e<3; ++e) action.rotation[3*d+e]=op.rotation(d,e);
    out.source_shells=selection.source_shells;
    for (U center=0; center<4; ++center) {
        const auto shell=selection.source_shells[center];
        const auto atom=static_cast<U>(basis.shell_atom_index(shell));
        out.source_atoms[center]=atom;
        out.destination_atoms[center]=transport.atom_destination(atom);
        out.destination_shells[center]=transport.shell_destination(shell);
        const auto ell=transport.atom_lattice_shift(atom);
        for (U d=0; d<3; ++d) action.atom_shifts[3*center+d]=ell[d];
    }
    out.atom_shifts=action.atom_shifts;
    out.support=audit_bipole_finite_support_action(source,basis,images,cells,action,caps.support);
    Hasher hash("vibeqc.bipole.finite-source.geometry-mapped-support");
    hash.text(out.support.action_identity_sha256);
    hash.u64(system.unit_cell.size());
    for (const auto& atom : system.unit_cell) {
        hash.u64(static_cast<U>(atom.Z));
        for (double x : atom.xyz) hash.real(x);
    }
    hash.u64(basis.nshells());
    for (U s=0; s<basis.nshells(); ++s) hash.u64(static_cast<U>(basis.shell_atom_index(s)));
    for (double x : {op.translation[0],op.translation[1],op.translation[2]}) hash.real(x);
    for (auto shell : out.source_shells) hash.u64(shell);
    for (auto shell : out.destination_shells) hash.u64(shell);
    hash.u64(selection.source_k_index); hash.u64(selection.time_reversal);
    for (double x : {options.maximum_atom_mapping_residual_bohr,
        options.maximum_basis_origin_residual_bohr,options.maximum_rotation_orthogonality_residual,
        options.maximum_polynomial_reconstruction_residual,options.maximum_pure_rotation_unitarity_residual,
        options.minimum_relative_lattice_volume}) hash.real(x);
    hash.text("single-operation;selected-shell-quartet;no-Hamiltonian-or-group-certificate-v1");
    out.mapping_identity_sha256=hash.finish();
    return out;
}

BipoleFiniteOverlapPlan plan_bipole_finite_overlap(
        const BipoleFiniteSource& source, const BasisSet& basis,
        BipoleErfcImageView images, AOPairFourierCellView cells,
        const Z* overlap, U count, U k, double tolerance,
        const PeriodicAOBlochTransportInventory& i, const BipoleFiniteOverlapCaps& c) {
    if (source.product_resolved()) invalid("BIPOLE common-source consumer requires a common source, not a product domain");
    if (!i.numerical_replicas || !i.backend_margin_bytes_per_replica ||
        !c.maximum_per_replica_inventoried_bytes || !c.maximum_node_inventoried_bytes ||
        !c.maximum_work_units)
        invalid("BIPOLE overlap requires positive explicit inventory/caps");
    if (!std::isfinite(tolerance) || tolerance<0)
        invalid("BIPOLE overlap requires finite nonnegative absolute tolerance");
    if (k>=source.mesh().size()) throw std::out_of_range("BIPOLE overlap k index");
    BipoleFiniteOverlapPlan p;
    p.k_index=k; p.overlap_elements=mul(basis.nbasis(),basis.nbasis());
    if (basis.nbasis()!=source.memory().n_basis || count!=p.overlap_elements ||
        (count && (!overlap || reinterpret_cast<std::uintptr_t>(overlap)%alignof(Z))))
        invalid("BIPOLE overlap requires aligned row-major Nao squared extent");
    p.borrowed_overlap_bytes=mul(16,count);
    cap(p.borrowed_overlap_bytes,std::numeric_limits<std::size_t>::max(),"BIPOLE overlap address extent");
    p.fixed_workspace_bytes=32768+4*(sizeof(p)+sizeof(BipoleFiniteOverlapDiagnostics)+sizeof(i)+sizeof(c));
    cap(add(p.fixed_workspace_bytes,p.borrowed_overlap_bytes),c.maximum_per_replica_inventoried_bytes,
        "BIPOLE overlap fixed/borrowed cap");
    p.source=plan_bipole_finite_source(basis,source.mesh(),images,cells,source.options(),c.source);
    p.fourier=plan_ao_pair_gaussian_fourier_cell_panel(basis,1,0,count,cells.cell_count,c.fourier);
    U bytes=add(p.fixed_workspace_bytes,add(p.borrowed_overlap_bytes,p.source.context_storage_bytes));
    bytes=add(bytes,add(p.source.borrowed_numerical_bytes,p.fourier.output_bytes));
    bytes=add(bytes,add(p.fourier.basis_control_storage_bytes,p.fourier.fixed_control_storage_bytes));
    bytes=add(bytes,add(p.fourier.fixed_numeric_workspace_bytes,p.fourier.fixed_scalar_numeric_bytes));
    bytes=add(bytes,add(i.other_live_numerical_bytes_per_replica,i.other_live_control_bytes_per_replica));
    p.per_replica_inventoried_bytes=add(bytes,i.backend_margin_bytes_per_replica);
    p.required_node_inventoried_bytes=add(i.external_node_bytes,mul(i.numerical_replicas,p.per_replica_inventoried_bytes));
    p.work_units_upper_bound=add(add(65536,mul(512,count)),
        mul(4,add(p.source.work_units_upper_bound,p.fourier.work_units)));
    cap(p.per_replica_inventoried_bytes,c.maximum_per_replica_inventoried_bytes,"BIPOLE overlap per-replica cap");
    cap(p.required_node_inventoried_bytes,c.maximum_node_inventoried_bytes,"BIPOLE overlap node cap");
    cap(p.work_units_upper_bound,c.maximum_work_units,"BIPOLE overlap work cap");
    return p;
}

BipoleFiniteOverlapDiagnostics audit_bipole_finite_overlap(
        const BipoleFiniteSource& source, const BasisSet& basis,
        BipoleErfcImageView images, AOPairFourierCellView cells,
        const Z* overlap, U count, U k, double tolerance,
        const PeriodicAOBlochTransportInventory& i, const BipoleFiniteOverlapCaps& c) {
    BipoleFiniteOverlapDiagnostics out;
    out.memory=plan_bipole_finite_overlap(source,basis,images,cells,overlap,count,k,tolerance,i,c);
    source.verify_inputs(basis,images,cells,c.source);
    // Validate every borrowed value before the Fourier panel allocation.
    Hasher hash("vibeqc.bipole.finite-source.zero-mode-overlap");
    hash.text(source.source_identity_sha256()); hash.u64(k); hash.real(tolerance); hash.u64(count);
    for (U j=0; j<count; ++j) hash.complex(overlap[j]);
    const auto system=system_for(source);
    const std::array<double,3> g{0,0,0};
    const AuxiliaryFourierVectorView vector{g.data(),g.data()+1,g.data()+2,1,1,1,1};
    const Eigen::Vector3d minus_k=-(source.reciprocal_lattice()*source.mesh().fractional_at(k));
    const auto b=ao_pair_gaussian_fourier_cell_panel(basis,system,vector,minus_k,0,count,cells,c.fourier);
    const U n=out.memory.source.n_basis;
    for (U a=0; a<n; ++a) for (U d=0; d<n; ++d) {
        const U index=a*n+d;
        out.maximum_direct_residual=std::max(out.maximum_direct_residual,
            finite(std::abs(finite(overlap[index]-std::conj(b.data[index])))));
        out.maximum_dual_residual=std::max(out.maximum_dual_residual,
            finite(std::abs(finite(overlap[index]-b.data[d*n+a]))));
        hash.complex(b.data[index]);
    }
    out.absolute_tolerance=tolerance;
    out.direct_matches=out.maximum_direct_residual<=tolerance;
    out.dual_matches=out.maximum_dual_residual<=tolerance;
    out.source_identity_sha256=source.source_identity_sha256();
    out.overlap_identity_sha256=hash.finish();
    return out;
}

Plan plan_bipole_finite_product_panel(const BipoleFiniteSource& source, const BasisSet& basis,
        BipoleErfcImageView images, AOPairFourierCellView cells, AOPairFourierCellView right_cells, const Selection& s,
        const Inventory& i, const Caps& c) {
    if (!i.numerical_replicas || !i.backend_margin_bytes_per_replica || !i.reciprocal_block_size ||
        !c.maximum_borrowed_numerical_bytes || !c.maximum_owned_numerical_bytes ||
        !c.maximum_control_storage_bytes || !c.maximum_worker_bytes || !c.maximum_node_bytes || !c.maximum_work_units)
        invalid("BIPOLE finite panel requires positive explicit inventory and caps");
    product_selection(source,s,cells,right_cells);
    Plan p; p.selection=s; p.subtract_zero_mode=s.q_index==0;
    p.output_elements=mul(s.left_pair_count,s.right_pair_count);
    p.retained_output_bytes=mul(16,p.output_elements); p.compensation_bytes=p.retained_output_bytes;
    p.fixed_numeric_workspace_bytes=8192;
    p.wrapper_control_storage_bytes=32768+4*sizeof(Plan)+sizeof(BipoleFinitePanelResult)+
        4*(sizeof(Caps)+sizeof(Inventory))+source.memory().context_storage_bytes;
    p.wrapper_work_units=add(65536,mul(4096,p.output_elements));
    cap(p.output_elements,std::vector<Z>().max_size(),"BIPOLE finite panel output address cap");
    cap(resident(p),c.maximum_owned_numerical_bytes,"BIPOLE finite panel outer output cap");
    auto source_caps=c.source;
    source_caps.maximum_borrowed_numerical_bytes=std::min(source_caps.maximum_borrowed_numerical_bytes,c.maximum_borrowed_numerical_bytes);
    source_caps.maximum_work_units=std::min(source_caps.maximum_work_units,
        residual(c.maximum_work_units,p.wrapper_work_units,"BIPOLE finite panel context work cap")/2);
    p.source=source.product_resolved() ?
        plan_bipole_finite_product_source(basis,source.mesh(),images,cells,right_cells,source.product_domain(),source.options(),source_caps) :
        plan_bipole_finite_source(basis,source.mesh(),images,cells,source.options(),source_caps);
    const auto system=system_for(source);
    if (p.subtract_zero_mode) {
        p.zero_left=plan_ao_pair_gaussian_fourier_cell_panel(basis,1,s.left_pair_begin,s.left_pair_count,cells.cell_count,c.zero_mode);
        p.zero_right=plan_ao_pair_gaussian_fourier_cell_panel(basis,1,s.right_pair_begin,s.right_pair_count,right_cells.cell_count,c.zero_mode);
        p.zero_mode_workspace_bytes=add(add(p.zero_left.output_bytes,p.zero_right.output_bytes),
            std::max(add(p.zero_left.fixed_numeric_workspace_bytes,p.zero_left.fixed_scalar_numeric_bytes),
                     add(p.zero_right.fixed_numeric_workspace_bytes,p.zero_right.fixed_scalar_numeric_bytes)));
    }
    p.short_range=plan_bipole_erfc_bloch(basis,system,source.mesh(),images,s,sr_options(source),sr_inventory(i,p),sr_caps(c,p));
    p.long_range=plan_bipole_ewald_product_gram(basis,system,source.mesh(),cells,right_cells,s,lr_options(source,i),lr_inventory(i,p),lr_caps(c,p));
    p.borrowed_numerical_bytes=p.source.borrowed_numerical_bytes;
    if (p.short_range.borrowed_numerical_bytes!=p.borrowed_numerical_bytes-p.source.borrowed_cell_bytes ||
        p.long_range.borrowed_numerical_bytes!=p.borrowed_numerical_bytes-p.source.borrowed_image_bytes)
        throw std::logic_error("BIPOLE finite panel common borrowed-owner census mismatch");
    p.peak_owned_numerical_bytes=add(resident(p),std::max({p.short_range.peak_owned_numerical_bytes,
        p.long_range.peak_owned_numerical_bytes,p.zero_mode_workspace_bytes}));
    const U zero_controls=add(p.wrapper_control_storage_bytes,add(i.other_live_control_bytes_per_replica,
        add(p.source.basis_census.basis_control_storage_bytes,
            std::max(p.zero_left.fixed_control_storage_bytes,p.zero_right.fixed_control_storage_bytes))));
    p.control_storage_bytes=std::max({p.short_range.control_storage_bytes,p.long_range.control_storage_bytes,zero_controls});
    p.per_replica_inventoried_bytes=add(p.peak_owned_numerical_bytes,add(p.borrowed_numerical_bytes,
        add(p.control_storage_bytes,add(i.other_live_numerical_bytes_per_replica,i.backend_margin_bytes_per_replica))));
    p.required_node_inventoried_bytes=add(i.external_node_bytes,mul(i.numerical_replicas,p.per_replica_inventoried_bytes));
    // Charge two complete child work envelopes, conservatively including the
    // separate outer planning pass. Only one numerical execution occurs.
    p.work_units_upper_bound=add(base_work(p),mul(2,add(p.short_range.work_units_upper_bound,p.long_range.work_units_upper_bound)));
    cap(p.peak_owned_numerical_bytes,c.maximum_owned_numerical_bytes,"BIPOLE finite panel owned cap");
    cap(p.borrowed_numerical_bytes,c.maximum_borrowed_numerical_bytes,"BIPOLE finite panel borrowed cap");
    cap(p.control_storage_bytes,c.maximum_control_storage_bytes,"BIPOLE finite panel control cap");
    cap(p.per_replica_inventoried_bytes,c.maximum_worker_bytes,"BIPOLE finite panel worker cap");
    cap(p.required_node_inventoried_bytes,c.maximum_node_bytes,"BIPOLE finite panel node cap");
    cap(p.work_units_upper_bound,c.maximum_work_units,"BIPOLE finite panel work cap");
    return p;
}

BipoleFinitePanelResult make_bipole_finite_product_panel(std::shared_ptr<const BipoleFiniteSource> source,
        const BasisSet& basis, BipoleErfcImageView images, AOPairFourierCellView cells, AOPairFourierCellView right_cells,
        const Selection& s, const Inventory& i, const Caps& c) {
    if (!source) invalid("BIPOLE finite panel requires a live immutable source owner");
    BipoleFinitePanelResult out;
    out.plan_=plan_bipole_finite_product_panel(*source,basis,images,cells,right_cells,s,i,c);
    const auto& p=out.plan_;
    if (source->product_resolved()) source->verify_product_inputs(basis,images,cells,right_cells,c.source);
    else source->verify_inputs(basis,images,cells,c.source);
    out.source_=std::move(source);
    const auto system=system_for(*out.source_);
    Hasher input("vibeqc.bipole.finite-panel.input");
    input.text(out.source_->source_identity_sha256()); selection_hash(input,s);
    out.input_identity_=input.finish();
    out.values_.assign(p.output_elements,Z{});
    std::vector<Z> correction(p.output_elements,Z{});
    {
        const auto sr=make_bipole_erfc_bloch(basis,system,out.source_->mesh(),images,s,
            sr_options(*out.source_),sr_inventory(i,p),sr_caps(c,p));
        out.sr_identity_=sr.payload_identity_sha256(); out.diagnostics_.short_range=sr.diagnostics();
        out.diagnostics_.image_permutation_support_certified=sr.image_permutation_support_certified();
        const auto* data=sr.data();
        for (U j=0; j<p.output_elements; ++j) accumulate(data[j],out.values_[j],correction[j]);
    }
    {
        const auto lr=make_bipole_ewald_product_gram(basis,system,out.source_->mesh(),cells,right_cells,s,
            lr_options(*out.source_,i),lr_inventory(i,p),lr_caps(c,p));
        out.lr_identity_=lr.payload_identity_sha256(); out.diagnostics_.long_range=lr.diagnostics();
        const auto* data=lr.data();
        for (U j=0; j<p.output_elements; ++j) accumulate(data[j],out.values_[j],correction[j]);
    }
    Hasher zero("vibeqc.bipole.finite-panel.zero-mode");
    zero.text(out.input_identity_); zero.u64(p.subtract_zero_mode);
    if (p.subtract_zero_mode) {
        const std::array<double,3> g{0,0,0};
        const AuxiliaryFourierVectorView vector{g.data(),g.data()+1,g.data()+2,1,1,1,1};
        const Eigen::Vector3d kl=-(out.source_->reciprocal_lattice()*out.source_->mesh().fractional_at(s.left_k_index));
        const Eigen::Vector3d kr=-(out.source_->reciprocal_lattice()*out.source_->mesh().fractional_at(s.right_k_index));
        const auto left=ao_pair_gaussian_fourier_cell_panel(basis,system,vector,kl,s.left_pair_begin,s.left_pair_count,cells,c.zero_mode);
        const auto right=ao_pair_gaussian_fourier_cell_panel(basis,system,vector,kr,s.right_pair_begin,s.right_pair_count,right_cells,c.zero_mode);
        const double coefficient=out.source_->zero_mode_coefficient();
        zero.real(coefficient); zero.u64(left.data.size()); zero.u64(right.data.size());
        for (Z value : left.data) zero.complex(value);
        for (Z value : right.data) zero.complex(value);
        out.diagnostics_.zero_mode_pair_values=add(left.data.size(),right.data.size());
        for (U l=0; l<s.left_pair_count; ++l) for (U r=0; r<s.right_pair_count; ++r) {
            const Z term=detail::bipole_ewald_weighted_product(left.data[l],right.data[r],coefficient);
            accumulate(-term,out.values_[l*s.right_pair_count+r],correction[l*s.right_pair_count+r]);
            out.diagnostics_.maximum_zero_mode_magnitude=std::max(out.diagnostics_.maximum_zero_mode_magnitude,finite(std::abs(term)));
        }
    }
    out.zero_identity_=zero.finish();
    Hasher payload("vibeqc.bipole.finite-panel.payload");
    payload.text(out.input_identity_); payload.text(out.sr_identity_); payload.text(out.lr_identity_); payload.text(out.zero_identity_);
    for (U j=0; j<p.output_elements; ++j) {
        out.values_[j]=finite(out.values_[j]+correction[j]); payload.complex(out.values_[j]);
        out.diagnostics_.maximum_integral_magnitude=std::max(out.diagnostics_.maximum_integral_magnitude,finite(std::abs(out.values_[j])));
    }
    out.payload_identity_=payload.finish();
    return out;
}

Plan plan_bipole_finite_panel(const BipoleFiniteSource& source, const BasisSet& basis,
        BipoleErfcImageView images, AOPairFourierCellView cells, const Selection& s,
        const Inventory& i, const Caps& c) {
    if (source.product_resolved()) invalid("BIPOLE product source requires the product panel entry point");
    return plan_bipole_finite_product_panel(source,basis,images,cells,cells,s,i,c);
}

BipoleFinitePanelResult make_bipole_finite_panel(std::shared_ptr<const BipoleFiniteSource> source,
        const BasisSet& basis, BipoleErfcImageView images, AOPairFourierCellView cells,
        const Selection& s, const Inventory& i, const Caps& c) {
    if (!source) invalid("BIPOLE finite panel requires a live immutable source owner");
    if (source->product_resolved()) invalid("BIPOLE product source requires the product panel entry point");
    return make_bipole_finite_product_panel(std::move(source),basis,images,cells,cells,s,i,c);
}

void BipoleFinitePanelResult::require_live() const {
    if (!source_ || values_.size()!=plan_.output_elements || input_identity_.size()!=64 ||
        sr_identity_.size()!=64 || lr_identity_.size()!=64 || zero_identity_.size()!=64 || payload_identity_.size()!=64)
        throw std::logic_error("BIPOLE finite panel result is consumed or malformed");
}
const BipoleFiniteSource& BipoleFinitePanelResult::source() const { require_live(); return *source_; }
const Z* BipoleFinitePanelResult::data() const { require_live(); return values_.data(); }
Z BipoleFinitePanelResult::element(U left, U right) const {
    require_live();
    if (left>=plan_.selection.left_pair_count || right>=plan_.selection.right_pair_count)
        throw std::out_of_range("BIPOLE finite panel selected pair ordinal");
    return values_[left*plan_.selection.right_pair_count+right];
}
const std::string& BipoleFinitePanelResult::input_identity_sha256() const { require_live(); return input_identity_; }
const std::string& BipoleFinitePanelResult::short_range_payload_identity_sha256() const { require_live(); return sr_identity_; }
const std::string& BipoleFinitePanelResult::long_range_payload_identity_sha256() const { require_live(); return lr_identity_; }
const std::string& BipoleFinitePanelResult::zero_mode_identity_sha256() const { require_live(); return zero_identity_; }
const std::string& BipoleFinitePanelResult::payload_identity_sha256() const { require_live(); return payload_identity_; }

namespace {
using JKPlan = BipoleFiniteJKPlan;
using JKCaps = BipoleFiniteJKCaps;
using JKSelection = BipoleFiniteJKSelection;
U jk_resident(const JKPlan& p) {
    return add(p.fixed_numeric_workspace_bytes,add(p.retained_output_bytes,p.compensation_bytes));
}
Inventory jk_inventory(const Inventory& i, const JKPlan& p) {
    auto out=i;
    out.other_live_numerical_bytes_per_replica=add(i.other_live_numerical_bytes_per_replica,
        add(jk_resident(p),p.density_bytes));
    out.other_live_control_bytes_per_replica=add(i.other_live_control_bytes_per_replica,p.wrapper_control_storage_bytes);
    return out;
}
Caps jk_caps(const JKCaps& c, const JKPlan& p) {
    auto out=c.panel;
    out.maximum_owned_numerical_bytes=residual(out.maximum_owned_numerical_bytes,jk_resident(p),
        "BIPOLE finite J/K owned cap");
    out.maximum_borrowed_numerical_bytes=residual(out.maximum_borrowed_numerical_bytes,p.density_bytes,
        "BIPOLE finite J/K borrowed cap");
    // Two complete child envelopes cover this preflight and the repeated
    // child admission during execution. No unbounded planning loop occurs.
    out.maximum_work_units=std::min(out.maximum_work_units,
        residual(c.maximum_work_units,p.wrapper_work_units,"BIPOLE finite J/K work cap")/mul(2,p.panel_calls));
    return out;
}
template<class Visitor>
void jk_schedule(const RegularKMesh& mesh, const JKPlan& p, Visitor&& visit) {
    const auto& s=p.selection;
    const U n=p.n_basis, n2=mul(n,n), block=s.density_pair_block_size;
    // Per-output accumulation order is k,c,d, independent of tile boundaries.
    for (U k=0; k<p.n_kpoints; ++k) for (U out=0; out<s.pair_count; ++out) {
        const U pair=s.pair_begin+out, a=pair/n, b=pair%n;
        for (U begin=0; begin<n2;) {
            const U count=std::min(block,n2-begin);
            const Selection child{0,s.target_k_index,k,pair,1,begin,count};
            visit(child,k,out,begin); begin+=count;
        }
        const U q=mesh.transfer_index(k,s.target_k_index);
        for (U c=0; c<n; ++c) for (U d=0; d<n;) {
            const U count=std::min(block,n-d);
            const Selection child{q,k,k,a*n+c,1,b*n+d,count};
            visit(child,k,s.pair_count+out,c*n+d); d+=count;
        }
    }
}
} // namespace

JKPlan plan_bipole_finite_jk(const BipoleFiniteSource& source, const BasisSet& basis,
        BipoleErfcImageView images, AOPairFourierCellView cells, BipoleFiniteDensityView density,
        const JKSelection& s, const Inventory& i, const JKCaps& c) {
    if (source.product_resolved()) invalid("BIPOLE common-source consumer requires a common source, not a product domain");
    if (!c.maximum_panel_calls || !c.maximum_work_units || !s.density_pair_block_size || !s.pair_count)
        invalid("BIPOLE finite J/K requires positive explicit selection, block and caps");
    JKPlan p; p.selection=s; p.n_basis=source.memory().n_basis; p.n_kpoints=source.mesh().size();
    cap(p.n_kpoints,U{1}<<53,"BIPOLE finite J/K exact full-mesh weight count");
    const U n=p.n_basis, n2=mul(n,n), block=s.density_pair_block_size;
    if (!n || basis.nbasis()!=n || s.target_k_index>=p.n_kpoints || s.pair_begin>=n2 || s.pair_count>n2-s.pair_begin)
        invalid("BIPOLE finite J/K selected k/AO pair outside source");
    const U count=mul(p.n_kpoints,n2);
    if (density.n_kpoints!=p.n_kpoints || density.n_basis!=n || density.value_count!=count || !density.values ||
        reinterpret_cast<std::uintptr_t>(density.values)%alignof(Z))
        invalid("BIPOLE finite J/K needs aligned exact full-grid density extents");
    p.density_bytes=mul(16,count);
    cap(p.density_bytes,std::numeric_limits<std::size_t>::max(),"BIPOLE finite J/K density address cap");
    p.output_elements=mul(2,s.pair_count);
    cap(p.output_elements,std::vector<Z>().max_size(),"BIPOLE finite J/K output address cap");
    p.retained_output_bytes=mul(16,p.output_elements); p.compensation_bytes=p.retained_output_bytes;
    p.fixed_numeric_workspace_bytes=8192;
    p.wrapper_control_storage_bytes=16384+4*sizeof(JKPlan)+sizeof(BipoleFiniteJKResult)+
        4*(sizeof(Plan)+sizeof(JKCaps)+sizeof(Inventory));
    const U per_output=add(1+(n2-1)/block,mul(n,1+(n-1)/block));
    p.panel_calls=mul(mul(p.n_kpoints,s.pair_count),per_output);
    cap(p.panel_calls,c.maximum_panel_calls,"BIPOLE finite J/K panel-call cap");
    p.contracted_terms=mul(mul(p.n_kpoints,p.output_elements),n2);
    p.wrapper_work_units=add(mul(2,source.memory().work_units_upper_bound),
        add(65536,add(mul(256,count),add(mul(4096,p.panel_calls),mul(256,p.contracted_terms)))));
    cap(add(4096,mul(32,add(count,p.output_elements))),ShaMaximumBytes,"BIPOLE finite J/K SHA extent");
    cap(jk_resident(p),c.panel.maximum_owned_numerical_bytes,"BIPOLE finite J/K outer output cap");
    const auto inventory=jk_inventory(i,p);
    const auto caps=jk_caps(c,p);
    U maximum_child_work=0;
    // No descriptor table, density reads, image-label scans or engines here.
    jk_schedule(source.mesh(),p,[&](const Selection& child, U, U, U) {
        const auto cp=plan_bipole_finite_panel(source,basis,images,cells,child,inventory,caps);
        p.borrowed_numerical_bytes=std::max(p.borrowed_numerical_bytes,add(cp.borrowed_numerical_bytes,p.density_bytes));
        p.peak_owned_numerical_bytes=std::max(p.peak_owned_numerical_bytes,add(cp.peak_owned_numerical_bytes,jk_resident(p)));
        p.control_storage_bytes=std::max(p.control_storage_bytes,cp.control_storage_bytes);
        p.per_replica_inventoried_bytes=std::max(p.per_replica_inventoried_bytes,cp.per_replica_inventoried_bytes);
        p.required_node_inventoried_bytes=std::max(p.required_node_inventoried_bytes,cp.required_node_inventoried_bytes);
        maximum_child_work=std::max(maximum_child_work,cp.work_units_upper_bound);
    });
    // Seal the same uniform child envelope used on replay, so the returned
    // total is itself an admissible exact cap, not an average child budget.
    p.work_units_upper_bound=add(p.wrapper_work_units,mul(mul(2,p.panel_calls),maximum_child_work));
    cap(p.borrowed_numerical_bytes,c.panel.maximum_borrowed_numerical_bytes,"BIPOLE finite J/K borrowed cap");
    cap(p.peak_owned_numerical_bytes,c.panel.maximum_owned_numerical_bytes,"BIPOLE finite J/K owned cap");
    cap(p.control_storage_bytes,c.panel.maximum_control_storage_bytes,"BIPOLE finite J/K control cap");
    cap(p.per_replica_inventoried_bytes,c.panel.maximum_worker_bytes,"BIPOLE finite J/K worker cap");
    cap(p.required_node_inventoried_bytes,c.panel.maximum_node_bytes,"BIPOLE finite J/K node cap");
    cap(p.work_units_upper_bound,c.maximum_work_units,"BIPOLE finite J/K work cap");
    return p;
}

BipoleFiniteJKResult make_bipole_finite_jk(std::shared_ptr<const BipoleFiniteSource> source,
        const BasisSet& basis, BipoleErfcImageView images, AOPairFourierCellView cells,
        BipoleFiniteDensityView density, const JKSelection& s, const Inventory& i, const JKCaps& c) {
    if (!source) invalid("BIPOLE finite J/K requires a source owner");
    const auto p=plan_bipole_finite_jk(*source,basis,images,cells,density,s,i,c);
    const auto inventory=jk_inventory(i,p);
    const auto caps=jk_caps(c,p);
    // Bind source and EVERY density entry before any numerical allocation.
    // Child calls repeat source verification, under the admitted work budget.
    source->verify_inputs(basis,images,cells,caps.source);
    Hasher input("vibeqc.bipole.finite-jk.input");
    input.text(source->source_identity_sha256());
    input.text("full-grid-C-Cdagger;J-ab-cd;K-ac-bd;q=target-source;uniform-1/Nk;"
               "no-spin-no-probe-charge-no-repair;selected-finite-action-v1");
    input.u64(s.target_k_index); input.u64(s.pair_begin); input.u64(s.pair_count);
    input.u64(density.n_kpoints); input.u64(density.n_basis);
    for (U j=0; j<density.value_count; ++j) input.complex(density.values[j]);
    BipoleFiniteJKResult out; out.source_=std::move(source); out.plan_=p;
    out.input_identity_=input.finish();
    out.values_.resize(p.output_elements); std::vector<Z> correction(p.output_elements);
    const double weight=1.0/static_cast<double>(p.n_kpoints);
    const U n2=mul(p.n_basis,p.n_basis);
    jk_schedule(out.source_->mesh(),p,[&](const Selection& child, U k, U output, U offset) {
        const auto panel=make_bipole_finite_panel(out.source_,basis,images,cells,child,inventory,caps);
        for (U r=0; r<child.right_pair_count; ++r) {
            // Shared scaled multiplication avoids overflow of an unweighted
            // product whose correctly weighted result is representable.
            const Z term=detail::bipole_ewald_weighted_product(
                std::conj(panel.element(0,r)),density.values[k*n2+offset+r],weight);
            accumulate(term,out.values_[output],correction[output]);
        }
    }); // Each child dies before the next integral panel is allocated.
    Hasher payload("vibeqc.bipole.finite-jk.payload"); payload.text(out.input_identity_);
    for (U j=0; j<p.output_elements; ++j) {
        out.values_[j]=finite(out.values_[j]+correction[j]); payload.complex(out.values_[j]);
    }
    out.payload_identity_=payload.finish();
    return out;
}

void BipoleFiniteJKResult::require_live() const {
    if (!source_ || values_.size()!=plan_.output_elements || input_identity_.size()!=64 || payload_identity_.size()!=64)
        throw std::logic_error("BIPOLE finite J/K result is consumed or malformed");
}
const BipoleFiniteSource& BipoleFiniteJKResult::source() const { require_live(); return *source_; }
const Z* BipoleFiniteJKResult::data() const { require_live(); return values_.data(); }
Z BipoleFiniteJKResult::element(U kind, U pair) const {
    require_live();
    if (kind>=2 || pair>=plan_.selection.pair_count) throw std::out_of_range("BIPOLE finite J/K selected ordinal");
    return values_[kind*plan_.selection.pair_count+pair];
}
const std::string& BipoleFiniteJKResult::input_identity_sha256() const { require_live(); return input_identity_; }
const std::string& BipoleFiniteJKResult::payload_identity_sha256() const { require_live(); return payload_identity_; }


BipoleProductJKStreamPlan plan_bipole_product_jk_stream(const BipoleFiniteSource& source,
        BipoleFiniteDensityView density, U target, const Inventory& i,
        const BipoleProductJKStreamCaps& c) {
    if (source.product_resolved() || source.memory().image_count || source.memory().cell_count)
        invalid("BIPOLE product J/K stream needs an empty common declaration");
    if (!c.maximum_panel_calls || !c.maximum_density_bytes || !c.maximum_state_bytes ||
        !c.maximum_panel_inventoried_bytes || !c.maximum_panel_work_units ||
        !c.maximum_node_bytes || !c.maximum_work_units || !i.numerical_replicas)
        invalid("BIPOLE product J/K stream needs positive explicit caps and replicas");
    BipoleProductJKStreamPlan p;
    p.n_basis=source.memory().n_basis; p.n_kpoints=source.mesh().size(); p.target_k_index=target;
    const U n2=mul(p.n_basis,p.n_basis), count=mul(p.n_kpoints,n2);
    if (!n2 || target>=p.n_kpoints || p.n_kpoints>(U{1}<<53) ||
        density.n_basis!=p.n_basis || density.n_kpoints!=p.n_kpoints || density.value_count!=count ||
        !density.values || reinterpret_cast<std::uintptr_t>(density.values)%alignof(Z))
        invalid("BIPOLE product J/K stream requires exact aligned full-grid density extents");
    p.quartet_count=mul(n2,n2); p.panel_calls=mul(2,mul(p.quartet_count,p.n_kpoints));
    cap(p.panel_calls,c.maximum_panel_calls,"BIPOLE product J/K stream panel-call cap");
    p.density_bytes=mul(16,count); p.output_elements=mul(2,n2);
    cap(p.density_bytes,c.maximum_density_bytes,"BIPOLE product J/K stream density cap");
    cap(count,std::vector<Z>().max_size(),"BIPOLE product J/K stream density address cap");
    cap(p.output_elements,std::vector<Z>().max_size(),"BIPOLE product J/K stream output address cap");
    cap(n2,std::vector<std::array<char,65>>().max_size(),"BIPOLE product J/K stream receipt address cap");
    const U control=add(65536+sizeof(BipoleProductJKStream),
        add(source.memory().context_storage_bytes,mul(65,n2)));
    p.state_bytes=add(control,add(p.density_bytes,mul(32,p.output_elements)));
    cap(p.state_bytes,c.maximum_state_bytes,"BIPOLE product J/K stream state cap");
    // Retained density plus its caller-owned input are both inventoried.
    // The complete incoming panel node envelope is conservatively charged
    // for each stream replica, including its backend/external allowances.
    p.per_replica_inventoried_bytes=add(p.state_bytes,add(p.density_bytes,
        add(source.memory().borrowed_numerical_bytes,
        add(i.other_live_numerical_bytes_per_replica,
        add(i.other_live_control_bytes_per_replica,i.backend_margin_bytes_per_replica)))));
    p.required_node_inventoried_bytes=add(i.external_node_bytes,
        mul(i.numerical_replicas,add(p.per_replica_inventoried_bytes,c.maximum_panel_inventoried_bytes)));
    p.wrapper_work_units=add(65536,add(mul(512,count),mul(16384,p.panel_calls)));
    // Numerical children are supplied externally. Reserve two complete
    // child envelopes (preflight/replay) rather than claiming only summation.
    p.work_units_upper_bound=add(p.wrapper_work_units,mul(2,mul(p.panel_calls,c.maximum_panel_work_units)));
    cap(p.required_node_inventoried_bytes,c.maximum_node_bytes,"BIPOLE product J/K stream node cap");
    cap(p.work_units_upper_bound,c.maximum_work_units,"BIPOLE product J/K stream work cap");
    cap(add(4096,mul(32,add(count,p.output_elements))),ShaMaximumBytes,"BIPOLE product J/K stream SHA cap");
    return p;
}

std::shared_ptr<BipoleProductJKStream> make_bipole_product_jk_stream(
        std::shared_ptr<const BipoleFiniteSource> source, BipoleFiniteDensityView density,
        U target, const std::string& policy, const Inventory& i, const BipoleProductJKStreamCaps& c) {
    if (!source) invalid("BIPOLE product J/K stream requires a declaration");
    const auto p=plan_bipole_product_jk_stream(*source,density,target,i,c);
    if (policy.size()!=64 || !std::all_of(policy.begin(),policy.end(),[](char x) {
            return (x>='0' && x<='9') || (x>='a' && x<='f'); }))
        invalid("BIPOLE product J/K stream requires a lowercase SHA256 policy declaration");
    Hasher h("vibeqc.bipole.product-jk-stream.input");
    h.text(source->source_identity_sha256()); h.text(policy); h.u64(target);
    h.text("quartet-a-b-c-d;k;J-then-K;J-ab-Dcd;K-ac-Dbd;q=target-k;1/Nk;"
           "owned-density;no-spin-no-probe-no-repair;declared-policy-v1");
    for (U j=0; j<density.value_count; ++j) h.complex(density.values[j]);
    auto out=std::shared_ptr<BipoleProductJKStream>(new BipoleProductJKStream);
    out->declaration_=std::move(source); out->plan_=p; out->caps_=c;
    out->policy_identity_=policy; out->chain_identity_=h.finish();
    out->density_.assign(density.values,density.values+density.value_count);
    out->values_.resize(p.output_elements); out->correction_.resize(p.output_elements);
    out->pair_identities_.resize(p.n_basis*p.n_basis);
    return out;
}

Selection BipoleProductJKStream::next_selection() const {
    if (complete()) throw std::logic_error("BIPOLE product J/K stream already complete");
    const U n2=plan_.n_basis*plan_.n_basis, quartet=accepted_/(2*plan_.n_kpoints);
    const U k=(accepted_/2)%plan_.n_kpoints;
    return Selection{accepted_%2 ? declaration_->mesh().transfer_index(k,plan_.target_k_index) : 0,
        accepted_%2 ? k : plan_.target_k_index,k,quartet/n2,1,quartet%n2,1};
}

void BipoleProductJKStream::consume(const BipoleFinitePanelResult& panel) {
    const auto expected=next_selection();
    const auto& p=panel.memory(); const auto& actual=p.selection;
    if (actual.q_index!=expected.q_index || actual.left_k_index!=expected.left_k_index ||
        actual.right_k_index!=expected.right_k_index || actual.left_pair_begin!=expected.left_pair_begin ||
        actual.right_pair_begin!=expected.right_pair_begin || actual.left_pair_count!=1 || actual.right_pair_count!=1)
        invalid("BIPOLE product J/K stream panel is out of schedule");
    cap(std::max(p.required_node_inventoried_bytes,p.per_replica_inventoried_bytes),
        caps_.maximum_panel_inventoried_bytes,"BIPOLE product J/K stream incoming panel memory cap");
    cap(p.work_units_upper_bound,caps_.maximum_panel_work_units,"BIPOLE product J/K stream incoming panel work cap");
    const auto& source=panel.source(); const auto& root=*declaration_;
    const auto& a=source.options(); const auto& b=root.options();
    if (!source.product_resolved() || source.basis_identity_sha256()!=root.basis_identity_sha256() ||
        source.mesh().mesh()!=root.mesh().mesh() || source.mesh().is_shift()!=root.mesh().is_shift() ||
        !(source.direct_lattice().array()==root.direct_lattice().array()).all() ||
        a.omega!=b.omega || a.reciprocal_energy_cutoff!=b.reciprocal_energy_cutoff || a.zero_mode!=b.zero_mode ||
        a.require_image_permutation_closure!=b.require_image_permutation_closure ||
        a.require_cell_inversion_closure!=b.require_cell_inversion_closure ||
        a.require_reciprocal_conjugacy!=b.require_reciprocal_conjugacy)
        invalid("BIPOLE product J/K stream panel differs from common declaration");
    const auto& d=source.product_domain();
    if (d.left_pair_begin!=expected.left_pair_begin || d.right_pair_begin!=expected.right_pair_begin ||
        d.left_pair_count!=1 || d.right_pair_count!=1)
        invalid("BIPOLE product J/K stream requires singleton source domains");
    const bool first=accepted_%(2*plan_.n_kpoints)==0;
    if (!first && source.source_identity_sha256()!=domain_identity_)
        invalid("BIPOLE product J/K stream source changes within its quartet");
    const auto& left=source.cells_identity_sha256(); const auto& right=source.right_cells_identity_sha256();
    auto check_pair=[&](U pair,const std::string& identity) {
        const auto& known=pair_identities_[pair];
        if (known[0] && std::strncmp(known.data(),identity.c_str(),65)!=0)
            invalid("BIPOLE product J/K stream product support changes across quartets");
    };
    check_pair(expected.left_pair_begin,left); check_pair(expected.right_pair_begin,right);
    if (expected.left_pair_begin==expected.right_pair_begin && left!=right)
        invalid("BIPOLE product J/K stream same product has inconsistent left/right supports");
    const U n=plan_.n_basis, n2=n*n, k=(accepted_/2)%plan_.n_kpoints;
    const U ai=expected.left_pair_begin/n, bi=expected.left_pair_begin%n;
    const U ci=expected.right_pair_begin/n, di=expected.right_pair_begin%n;
    const bool exchange=accepted_%2;
    const U output=exchange ? n2+ai*n+ci : ai*n+bi;
    const U density_pair=exchange ? bi*n+di : ci*n+di;
    const Z term=detail::bipole_ewald_weighted_product(std::conj(panel.element(0,0)),
        density_[k*n2+density_pair],1.0/static_cast<double>(plan_.n_kpoints));
    Z sum=values_[output], correction=correction_[output];
    accumulate(term,sum,correction);
    Hasher chain("vibeqc.bipole.product-jk-stream.step");
    chain.text(chain_identity_); chain.text(panel.input_identity_sha256());
    auto next_chain=chain.finish(); auto next_domain=source.source_identity_sha256();
    // No potentially throwing operation after this commit point.
    values_[output]=sum; correction_[output]=correction;
    std::memcpy(pair_identities_[expected.left_pair_begin].data(),left.c_str(),65);
    std::memcpy(pair_identities_[expected.right_pair_begin].data(),right.c_str(),65);
    chain_identity_.swap(next_chain); domain_identity_.swap(next_domain); ++accepted_;
}

void BipoleProductJKStream::finalize() {
    if (finalized_) throw std::logic_error("BIPOLE product J/K stream is already finalized");
    if (!complete()) throw std::logic_error("BIPOLE product J/K stream is incomplete");
    Hasher h("vibeqc.bipole.product-jk-stream.payload"); h.text(chain_identity_);
    for (U j=0; j<plan_.output_elements; ++j) h.complex(finite(values_[j]+correction_[j]));
    auto identity=h.finish();
    for (U j=0; j<plan_.output_elements; ++j) values_[j]+=correction_[j];
    payload_identity_.swap(identity); finalized_=true;
}
void BipoleProductJKStream::require_finalized() const {
    if (!finalized_) throw std::logic_error("BIPOLE product J/K stream has no finalized result");
}
const Z* BipoleProductJKStream::data() const { require_finalized(); return values_.data(); }
Z BipoleProductJKStream::element(U kind, U pair) const {
    require_finalized(); const U n2=plan_.n_basis*plan_.n_basis;
    if (kind>=2 || pair>=n2) throw std::out_of_range("BIPOLE product J/K stream output ordinal");
    return values_[kind*n2+pair];
}
const std::string& BipoleProductJKStream::input_identity_sha256() const { require_finalized(); return chain_identity_; }
const std::string& BipoleProductJKStream::payload_identity_sha256() const { require_finalized(); return payload_identity_; }

} // namespace vibeqc
