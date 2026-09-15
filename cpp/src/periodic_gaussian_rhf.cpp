#include "vibeqc/periodic_gaussian_rhf.hpp"

#include <algorithm>
#include <cfenv>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include "vibeqc/detail/sha256.hpp"

namespace vibeqc {
namespace {
using I = std::uint64_t;
using C = std::complex<double>;
using Live = PeriodicGaussianMetricLiveInventory;
using Plan = PeriodicGaussianRHFPlan;
using Stage = PeriodicGaussianRHFStage;
constexpr I control_reservation = 65536;
static_assert(sizeof(C) == 16 && sizeof(double) == 8 && sizeof(int) == 4
              && sizeof(void*) == 8 && std::numeric_limits<double>::is_iec559,
              "Gaussian RHF inventory requires binary64 and 64-bit pointers");
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error "Gaussian RHF forbids fast/finite-only math"
#endif
#if FLT_EVAL_METHOD != 0
#error "Gaussian RHF requires binary64 evaluation"
#endif
I add(I a, I b) {
    if (b > std::numeric_limits<I>::max()-a) throw std::overflow_error("Gaussian RHF count sum overflow");
    return a+b;
}
I mul(I a, I b) {
    if (a && b > std::numeric_limits<I>::max()/a) throw std::overflow_error("Gaussian RHF count product overflow");
    return a*b;
}
void limit(I n, I cap, const char* message) { if (n > cap) throw std::length_error(message); }
void positive(I n) { if (!n) throw std::invalid_argument("Gaussian RHF requires explicit positive resource caps"); }
void finite(double x) { if (!std::isfinite(x)) throw std::invalid_argument("Gaussian RHF nonfinite input or arithmetic"); }
void ieee() {
    if (std::fegetround() != FE_TONEAREST) throw std::invalid_argument("Gaussian RHF requires round-to-nearest arithmetic");
    volatile double tiny=std::numeric_limits<double>::min(), half=0.5;
    if (tiny*half == 0.0) throw std::invalid_argument("Gaussian RHF requires gradual underflow");
}
class Digest {
public:
    explicit Digest(const char* domain) { text(domain); u64(kPeriodicGaussianRHFVersion); }
    void bytes(const std::uint8_t* p, I n) {
        limit(add(extent_,n),std::numeric_limits<I>::max()/8,"Gaussian RHF SHA extent overflow");
        hash_.update(p,static_cast<std::size_t>(n)); extent_+=n;
    }
    void u64(I n) {
        std::array<std::uint8_t,8> b{};
        for (unsigned j=0; j<8; ++j) b[j]=n>>(56-8*j);
        bytes(b.data(),8);
    }
    void real(double x) { finite(x); if (x==0) x=0; I n; std::memcpy(&n,&x,8); u64(n); }
    void text(const std::string& s) { u64(s.size()); bytes(reinterpret_cast<const std::uint8_t*>(s.data()),s.size()); }
    std::string finish() { return hash_.finish_hex(); }
private: detail::Sha256 hash_; I extent_=0;
};
std::array<char,64> ascii(const std::string& s) {
    if (s.size()!=64) throw std::logic_error("Gaussian RHF native SHA size mismatch");
    std::array<char,64> out{}; std::copy(s.begin(),s.end(),out.begin()); return out;
}
std::string string(const std::array<char,64>& a) { return {a.begin(),a.end()}; }
I extras(const Live& l) { return add(l.other_retained_bytes_per_replica,l.other_transient_bytes_per_replica); }
Live extended(Live l, I n) { l.other_retained_bytes_per_replica=add(l.other_retained_bytes_per_replica,n); return l; }
PeriodicGaussianOneElectronLiveInventory one_live(const Live& l) {
    return {l.replicas_per_node,l.other_retained_bytes_per_replica,l.other_transient_bytes_per_replica,
            l.fixed_backend_margin_bytes_per_replica,l.external_node_bytes};
}
PeriodicGaussianNuclearLiveInventory nuclear_live(const Live& l) {
    return {l.replicas_per_node,l.other_retained_bytes_per_replica,l.other_transient_bytes_per_replica,
            l.fixed_backend_margin_bytes_per_replica,l.external_node_bytes};
}
PeriodicGaussianSourceCaps exact_basis_caps(const PeriodicGaussianSourceContext& context) {
    const auto& p=context.inventory(); const auto& a=p.ao; const auto& b=p.auxiliary;
    return {sizeof(PeriodicGaussianSourceContext),context.mesh().size(),add(a.shell_count,b.shell_count),
        add(a.contraction_count,b.contraction_count),
        add(add(a.exponent_count,a.coefficient_count),add(b.exponent_count,b.coefficient_count)),
        add(a.content_wire_bytes,b.content_wire_bytes),p.combined_borrowed_active_numeric_bytes,p.work_units_upper_bound};
}
// Bounds on the original metadata precede any atom/mask content scan.
I check_system(const PeriodicGaussianSourceContext& context, const PeriodicSystem& system, I atom_cap) {
    if (system.dim!=3 || system.multiplicity!=1 || system.unit_cell.empty())
        throw std::invalid_argument("Gaussian RHF requires an explicit all-electron singlet bulk cell");
    limit(system.unit_cell.size(),atom_cap,"Gaussian RHF atom count cap exceeded");
    limit(system.unit_cell.size(),(std::numeric_limits<std::int64_t>::max()-I{2147483648})/118,
          "Gaussian RHF electron count extent overflow");
    for (unsigned row=0; row<3; ++row) for (unsigned col=0; col<3; ++col) {
        finite(system.lattice(row,col));
        if (system.lattice(row,col)!=context.direct_lattice()(row,col))
            throw std::invalid_argument("Gaussian RHF original lattice differs from source context");
    }
    std::int64_t electrons=-static_cast<std::int64_t>(system.charge);
    for (const auto& atom : system.unit_cell) {
        if (atom.Z<1 || atom.Z>118) throw std::invalid_argument("Gaussian RHF requires actual all-electron nuclear charges");
        for (double x : atom.xyz) finite(x);
        electrons+=atom.Z;
    }
    const I n=context.inventory().ao.function_count;
    if (electrons<=0 || electrons%2 || n<2 || static_cast<I>(electrons)/2>=n)
        throw std::invalid_argument("Gaussian RHF requires positive even electron count and a virtual orbital");
    return static_cast<I>(electrons);
}
void check_selection(PeriodicGaussianRHFFrozenSelection mask, I k, I occupied) {
    if (mask.elements!=mul(k,occupied) || !mask.data)
        throw std::invalid_argument("Gaussian RHF frozen selection must have exact [Nk,nocc] extent");
    I first=0;
    for (I point=0; point<k; ++point) {
        I count=0;
        for (I i=0; i<occupied; ++i) {
            const auto value=mask.data[point*occupied+i];
            if (value>1) throw std::invalid_argument("Gaussian RHF frozen selection must be binary");
            count+=value;
        }
        if (count>=occupied || (point && count!=first))
            throw std::invalid_argument("Gaussian RHF requires equal frozen rank and at least one active occupied band");
        first=count;
    }
}
std::string input_identity(const PeriodicGaussianSourceContext& context, const BasisSet& ao,
                           const BasisSet& auxiliary, const PeriodicSystem& system,
                           PeriodicGaussianRHFFrozenSelection mask, const Plan& plan) {
    if (system.unit_cell.size()!=plan.atom_count || mask.elements!=plan.frozen_selection_bytes)
        throw std::invalid_argument("Gaussian RHF borrowed metadata changed during callback");
    context.verify_bases(ao,auxiliary,exact_basis_caps(context));
    if (check_system(context,system,plan.atom_count)!=plan.electrons_per_cell)
        throw std::invalid_argument("Gaussian RHF electron count changed during callback");
    check_selection(mask,plan.n_kpoints,plan.electrons_per_cell/2);
    Digest h("vibeqc.periodic.gaussian-rhf.original-input");
    h.text(context.source_context_identity_sha256());
    h.u64(plan.atom_count); h.u64(static_cast<I>(static_cast<std::int64_t>(system.charge)));
    h.u64(system.multiplicity); h.u64(plan.electrons_per_cell);
    for (unsigned r=0; r<3; ++r) for (unsigned c=0; c<3; ++c) h.real(system.lattice(r,c));
    for (const auto& atom : system.unit_cell) { h.u64(atom.Z); for (double x : atom.xyz) h.real(x); }
    h.u64(mask.elements); h.bytes(mask.data,mask.elements);
    return h.finish();
}
void basis_cap_controls(const PeriodicGaussianSourceContext& context, const PeriodicGaussianSourceCaps& c) {
    const auto actual=exact_basis_caps(context);
#define GRHF_BASIS_CAP(field) positive(c.field); limit(actual.field,c.field,"Gaussian RHF basis verification cap exceeded")
    GRHF_BASIS_CAP(maximum_context_storage_bytes); GRHF_BASIS_CAP(maximum_kpoint_count);
    GRHF_BASIS_CAP(maximum_shell_count); GRHF_BASIS_CAP(maximum_contraction_count);
    GRHF_BASIS_CAP(maximum_primitive_numeric_lanes); GRHF_BASIS_CAP(maximum_basis_content_wire_bytes);
    GRHF_BASIS_CAP(maximum_borrowed_active_numeric_bytes); GRHF_BASIS_CAP(maximum_work_units);
#undef GRHF_BASIS_CAP
}
void structural_controls(double a,double r) {
    if (!std::isfinite(a)||!std::isfinite(r)||a<0||r<0||a>=1||r>=1||(a==0&&r==0))
        throw std::invalid_argument("Gaussian RHF one-electron structural controls must be finite in [0,1), not both zero");
}
void hash_options(Digest& h, const PeriodicGaussianRHFOptions& o) {
    h.u64(o.one_electron_pair_block);
    h.real(o.one_electron.structural_absolute_tolerance); h.real(o.one_electron.structural_relative_tolerance);
    h.real(o.nuclear.alpha); h.real(o.nuclear.real_cutoff_bohr); h.real(o.nuclear.reciprocal_cutoff_bohr_inverse);
    h.real(o.nuclear.structural_absolute_tolerance); h.real(o.nuclear.structural_relative_tolerance);
    const auto& s=o.scf;
    h.u64(s.maximum_iterations); h.u64(s.maximum_diis_history); h.u64(s.jacobi_max_sweeps);
    for (double x : {s.jacobi_relative_tolerance,s.overlap_rank_absolute_floor,s.overlap_rank_relative_floor,
        s.overlap_negative_absolute_tolerance,s.overlap_negative_relative_tolerance,
        s.hermitian_absolute_tolerance,s.hermitian_relative_tolerance,s.maximum_overlap_projection_error,
        s.maximum_hcore_projection_error,s.maximum_fock_projection_error,s.algebra_absolute_tolerance,
        s.algebra_relative_tolerance,s.eigen_relative_tolerance,s.commutator_tolerance,
        s.density_closure_absolute_tolerance,s.density_closure_relative_tolerance,s.energy_change_tolerance,
        s.minimum_band_gap_hartree}) h.real(x);
}
void unused_provider(const C*,std::size_t,C*,std::size_t,void*) {
    throw std::logic_error("Gaussian RHF count-only provider was invoked");
}
double average(double a, double b) {
    finite(a); finite(b);
    if (a==b) return a;
    int exponent=0; std::frexp(std::max(std::abs(a),std::abs(b)),&exponent);
    const double x=std::scalbn(a,-exponent),y=std::scalbn(b,-exponent);
    if (std::scalbn(x,exponent)!=a || std::scalbn(y,exponent)!=b)
        throw std::overflow_error("Gaussian RHF Hermitian projection loses input range");
    const double result=std::scalbn(x+y,exponent-1); finite(result); return result;
}
C average(C a, C b) { return {average(a.real(),b.real()),average(a.imag(),b.imag())}; }
C projected(const C* matrix, I base, I n, I i, I j) {
    if (i==j) return {matrix[base+i*n+j].real(),0.0};
    return average(matrix[base+i*n+j],std::conj(matrix[base+j*n+i]));
}
struct Sum {
    double value=0, correction=0;
    void append(double x) {
        finite(x); const double next=value+x; finite(next);
        correction+=std::abs(value)>=std::abs(x) ? (value-next)+x : (x-next)+value;
        finite(correction); value=next;
    }
    double get() const { const double result=value+correction; finite(result); return result; }
};
struct ComplexSum {
    Sum real,imag;
    void append(C z) { real.append(z.real()); imag.append(z.imag()); }
    C get() const { return {real.get(),imag.get()}; }
};
// Independent final SAME-D physical audits in the metric-orthonormal C
// basis. Four n*r panels, no full-supercell or four-index work. Frobenius
// commutator/eigen norms are invariant under a retained-space unitary.
void capture_audit(const PeriodicMeanFieldComplexMatrix& s, const PeriodicMeanFieldComplexMatrix& f,
                   const PeriodicMeanFieldComplexMatrix& c, const Eigen::VectorXd& eps,
                   const C* density, const BoundedPeriodicRHFOptions& options,
                   PeriodicGaussianRHFDiagnostics& d, double& all_commutator) {
    const I n=s.rows(),r=c.cols(),nr=mul(n,r);
    std::vector<C> workspace(mul(4,nr));
    C* fc=workspace.data(); C* sc=fc+nr; C* dfc=sc+nr; C* dsc=dfc+nr;
    for (I i=0; i<n; ++i) for (I a=0; a<r; ++a) {
        ComplexSum fs,ss;
        for (I j=0; j<n; ++j) { fs.append(f(i,j)*c(j,a)); ss.append(s(i,j)*c(j,a)); }
        fc[i*r+a]=fs.get(); sc[i*r+a]=ss.get();
    }
    for (I i=0; i<n; ++i) for (I a=0; a<r; ++a) {
        ComplexSum fs,ss;
        for (I j=0; j<n; ++j) { fs.append(density[i*n+j]*fc[j*r+a]); ss.append(density[i*n+j]*sc[j*r+a]); }
        dfc[i*r+a]=fs.get(); dsc[i*r+a]=ss.get();
    }
    double eigen_error=0,eigen_scale=0;
    for (I a=0; a<r; ++a) for (I b=0; b<r; ++b) {
        ComplexSum comm,gram,op;
        for (I i=0; i<n; ++i) {
            comm.append(std::conj(fc[i*r+a])*dsc[i*r+b]-std::conj(sc[i*r+a])*dfc[i*r+b]);
            gram.append(std::conj(c(i,a))*sc[i*r+b]); op.append(std::conj(c(i,a))*fc[i*r+b]);
        }
        const C g=gram.get(),operator_value=op.get();
        const double metric_error=std::abs(g-C(a==b ? 1.0 : 0.0,0)); finite(metric_error);
        d.maximum_capture_coefficient_metric_error=std::max(d.maximum_capture_coefficient_metric_error,metric_error);
        if (metric_error>options.algebra_absolute_tolerance+options.algebra_relative_tolerance*std::max(std::abs(g),a==b?1.0:0.0))
            throw std::invalid_argument("Gaussian RHF final coefficient metric gate failed");
        all_commutator=std::hypot(all_commutator,std::abs(comm.get())); finite(all_commutator);
        eigen_error=std::hypot(eigen_error,std::abs(operator_value-g*eps(b)));
        eigen_scale=std::hypot(eigen_scale,std::abs(operator_value)); finite(eigen_error); finite(eigen_scale);
    }
    const double relative=eigen_scale==0 ? eigen_error : eigen_error/eigen_scale; finite(relative);
    d.maximum_capture_projected_eigen_relative_residual=std::max(d.maximum_capture_projected_eigen_relative_residual,relative);
    if (relative>options.eigen_relative_tolerance)
        throw std::invalid_argument("Gaussian RHF rebuilt physical Fock exceeds requested eigen residual tolerance");
}
struct Adapter {
    std::shared_ptr<const PeriodicGaussianSourceContext> context;
    const BasisSet& ao; const BasisSet& auxiliary; const PeriodicSystem& system;
    PeriodicGaussianRHFFrozenSelection mask;
    const PeriodicGaussianRHFOptions& options; const PeriodicGaussianFockConfig& fock_config;
    const PeriodicGaussianRHFCaps& caps; const Plan& plan;
    Live fock_live;
    PeriodicGaussianRHFDiagnostics& diagnostics;
    std::string original, last_fock;
    PeriodicGaussianRHFCallback callback; void* callback_context;
    void check() const {
        if (input_identity(*context,ao,auxiliary,system,mask,plan)!=original)
            throw std::invalid_argument("Gaussian RHF borrowed physical input changed during callback");
    }
    void emit(PeriodicGaussianRHFProgress event) {
        if (!callback) return;
        limit(add(diagnostics.completed_progress_callbacks,1),caps.maximum_progress_callbacks,
              "Gaussian RHF progress callback cap exceeded");
        event.completed_panel_calls=diagnostics.completed_panel_calls;
        event.fock_call=diagnostics.completed_fock_calls;
        callback(event,callback_context); check(); ++diagnostics.completed_progress_callbacks;
    }
    static void fock_progress(const PeriodicGaussianFockProgress& event, void* opaque) {
        auto& self=*static_cast<Adapter*>(opaque);
        PeriodicGaussianRHFProgress p; p.stage=Stage::TwoElectron; p.fock=event; self.emit(p);
    }
    PeriodicGaussianFockResult response(const C* density, I count) {
        auto result=build_periodic_gaussian_fock(context,ao,auxiliary,{density,count},fock_config,fock_live,caps.fock,
                                                callback ? fock_progress : nullptr,this);
        if (result.context_handle().get()!=context.get()) throw std::logic_error("Gaussian RHF Fock source owner mismatch");
        Digest h("vibeqc.periodic.gaussian-rhf.evaluated-fock");
        h.text(result.density_identity_sha256()); h.text(result.consumed_factor_identity_sha256());
        h.text(result.payload_identity_sha256()); last_fock=h.finish();
        ++diagnostics.completed_fock_calls; return result;
    }
    static void evaluate(const C* density,std::size_t n,C* output,std::size_t output_n,void* opaque) {
        auto& self=*static_cast<Adapter*>(opaque);
        if (n!=output_n || n!=self.plan.fock.density_element_count)
            throw std::logic_error("Gaussian RHF native density/output extent mismatch");
        const auto response=self.response(density,n);
        std::copy(response.matrix_row_major().begin(),response.matrix_row_major().end(),output);
    }
    static bool scf_progress(const BoundedPeriodicRHFSnapshot& snapshot, void* opaque) {
        auto& self=*static_cast<Adapter*>(opaque);
        PeriodicGaussianRHFProgress p; p.stage=Stage::SCF; p.scf=snapshot; self.emit(p); return true;
    }
};
} // namespace

PeriodicGaussianRHFPlan plan_periodic_gaussian_rhf(
    const PeriodicGaussianSourceContext& context, const PeriodicSystem& system,
    PeriodicGaussianRHFFrozenSelection mask, const PeriodicGaussianRHFOptions& options,
    const PeriodicGaussianFockConfig& config, const Live& live, const PeriodicGaussianRHFCaps& caps) {
    ieee();
    for (I c : {caps.maximum_owned_numeric_bytes,caps.maximum_per_replica_inventoried_bytes,
        caps.maximum_node_inventoried_bytes,caps.maximum_state_numeric_bytes,
        caps.maximum_one_electron_panel_calls,caps.maximum_progress_callbacks,caps.maximum_work_units,
        caps.one_electron.maximum_owned_numeric_bytes,caps.one_electron.maximum_work_units,
        caps.nuclear.maximum_owned_numeric_bytes,caps.nuclear.maximum_work_units,caps.nuclear.maximum_atom_count,
        caps.one_electron.maximum_per_replica_inventoried_bytes,caps.one_electron.maximum_node_inventoried_bytes,
        caps.one_electron.maximum_pair_count,caps.one_electron.maximum_candidate_evaluations,
        caps.nuclear.maximum_per_replica_inventoried_bytes,caps.nuclear.maximum_node_inventoried_bytes,
        caps.nuclear.maximum_pair_count,caps.nuclear.maximum_ao_image_candidates,caps.nuclear.maximum_nuclear_image_candidates,
        caps.nuclear.maximum_reciprocal_candidates,caps.nuclear.maximum_ewald_pair_candidates,
        caps.scf.maximum_owned_numerical_bytes,caps.scf.maximum_total_numerical_bytes,
        caps.scf.maximum_work_units,caps.scf.maximum_fock_calls,live.replicas_per_node,
        live.fixed_backend_margin_bytes_per_replica,options.one_electron_pair_block}) positive(c);
    if (caps.scf.other_live_numerical_bytes)
        throw std::invalid_argument("Gaussian RHF declares other live arrays only through the outer live inventory");
    basis_cap_controls(context,options.one_electron.basis_verification_caps);
    basis_cap_controls(context,options.nuclear.basis_verification_caps);
    structural_controls(options.one_electron.structural_absolute_tolerance,options.one_electron.structural_relative_tolerance);
    structural_controls(options.nuclear.structural_absolute_tolerance,options.nuclear.structural_relative_tolerance);
    for (double x : {options.nuclear.alpha,options.nuclear.real_cutoff_bohr,options.nuclear.reciprocal_cutoff_bohr_inverse})
        if (!std::isfinite(x)||x<=0||!std::isfinite(x*x)||x*x<=0)
            throw std::invalid_argument("Gaussian RHF requires representable positive nuclear alpha and cutoffs");
    Plan p; p.n_kpoints=context.mesh().size(); p.n_basis=context.inventory().ao.function_count;
    p.atom_count=system.unit_cell.size();
    limit(p.atom_count,caps.nuclear.maximum_atom_count,"Gaussian RHF atom count cap exceeded");
    limit(add(mul(512,p.atom_count),4096),caps.maximum_work_units,"Gaussian RHF metadata work cap exceeded");
    p.electrons_per_cell=check_system(context,system,caps.nuclear.maximum_atom_count);
    p.frozen_selection_bytes=mul(p.n_kpoints,p.electrons_per_cell/2);
    p.input_check_work_units=add(context.inventory().work_units_upper_bound,
        add(mul(512,p.atom_count),add(mul(512,p.frozen_selection_bytes),4096)));
    limit(p.input_check_work_units,caps.maximum_work_units,"Gaussian RHF input check work cap exceeded");
    check_selection(mask,p.n_kpoints,p.electrons_per_cell/2);
    const I nn=mul(p.n_basis,p.n_basis), elements=mul(p.n_kpoints,nn), density_bytes=mul(16,elements);
    limit(elements,std::vector<C>().max_size(),"Gaussian RHF AO matrix extent exceeds addressable vector");
    limit(p.n_basis,static_cast<I>(std::numeric_limits<Eigen::Index>::max()),"Gaussian RHF Eigen index extent overflow");
    limit(options.one_electron_pair_block,nn,"Gaussian RHF pair block exceeds AO pair extent");
    limit(options.one_electron_pair_block,caps.one_electron.maximum_pair_count,"Gaussian RHF S/T pair block cap exceeded");
    limit(options.one_electron_pair_block,caps.nuclear.maximum_pair_count,"Gaussian RHF nuclear pair block cap exceeded");
    p.overlap_hcore_bytes=mul(32,elements);
    p.state_bytes_upper_bound=estimate_periodic_restricted_mean_field_resident_bytes(context.mesh().mesh(),p.n_basis,p.n_basis);
    limit(p.state_bytes_upper_bound,caps.maximum_state_numeric_bytes,"Gaussian RHF state payload cap exceeded");
    // Conservative reservation for existing Eigen-based v1 validation, not
    // an exact allocator/RSS assertion. No full-supercell matrix is formed.
    p.state_validation_workspace_reservation_bytes=add(mul(256,nn),mul(128,p.n_basis));
    const I panels=mul(p.n_kpoints,add(nn/options.one_electron_pair_block,nn%options.one_electron_pair_block!=0));
    p.one_electron_panel_calls=mul(2,panels);
    limit(p.one_electron_panel_calls,caps.maximum_one_electron_panel_calls,"Gaussian RHF one-electron panel call cap exceeded");
    const I system_bytes=mul(28,p.atom_count), basis_bytes=context.inventory().combined_borrowed_active_numeric_bytes;
    p.borrowed_input_numeric_bytes=add(basis_bytes,add(system_bytes,p.frozen_selection_bytes));
    // Per-point owner controls (including both assembly/state block vectors)
    // and atom padding are explicit non-numerical reservations, not free.
    p.control_storage_reservation_bytes=add(control_reservation,
        add(mul(1024,p.n_kpoints),mul(sizeof(Atom)-28,p.atom_count)));
    p.fock=plan_periodic_gaussian_fock(context,config,live,caps.fock);
    p.control_storage_reservation_bytes=add(p.control_storage_reservation_bytes,
        add(p.fock.macro_fixed_object_bytes,p.fock.maximum_leaf_fixed_inventory_bytes));
    BoundedPeriodicRHFTwoElectronProvider provider{unused_provider,nullptr,0,p.fock.owned_numeric_upper_bound,
                                                  p.fock.work_units_upper_bound};
    p.scf=plan_bounded_periodic_rhf_solver(p.n_kpoints,p.n_basis,p.electrons_per_cell,p.n_basis,
        provider,options.scf,add(p.borrowed_input_numeric_bytes,extras(live)));
    limit(p.scf.peak_owned_numerical_bytes,caps.scf.maximum_owned_numerical_bytes,"Gaussian RHF SCF owned memory cap exceeded");
    limit(p.scf.total_numerical_bytes,caps.scf.maximum_total_numerical_bytes,"Gaussian RHF SCF total memory cap exceeded");
    limit(p.scf.maximum_work_units,caps.scf.maximum_work_units,"Gaussian RHF SCF work cap exceeded");
    limit(options.scf.maximum_iterations,caps.scf.maximum_fock_calls,"Gaussian RHF SCF call ceiling exceeds cap");
    p.one_electron_owned_upper_bound=add(p.overlap_hcore_bytes,
        std::max(caps.one_electron.maximum_owned_numeric_bytes,caps.nuclear.maximum_owned_numeric_bytes));
    p.scf_owned_upper_bound=add(p.overlap_hcore_bytes,add(p.scf.peak_owned_numerical_bytes,p.fock.owned_numeric_upper_bound));
    const I capture_phase=std::max({add(density_bytes,p.fock.owned_numeric_upper_bound),
        add(mul(2,density_bytes),add(p.state_bytes_upper_bound,mul(64,nn))),
        add(p.state_bytes_upper_bound,p.state_validation_workspace_reservation_bytes)});
    p.capture_owned_upper_bound=add(p.overlap_hcore_bytes,add(p.scf.output_numerical_bytes,capture_phase));
    p.owned_numeric_upper_bound=std::max({p.one_electron_owned_upper_bound,p.scf_owned_upper_bound,p.capture_owned_upper_bound});
    p.per_replica_inventoried_bytes=add(p.owned_numeric_upper_bound,add(p.borrowed_input_numeric_bytes,
        add(extras(live),add(p.control_storage_reservation_bytes,live.fixed_backend_margin_bytes_per_replica))));
    p.node_inventoried_bytes=add(live.external_node_bytes,mul(live.replicas_per_node,p.per_replica_inventoried_bytes));
    limit(p.owned_numeric_upper_bound,caps.maximum_owned_numeric_bytes,"Gaussian RHF enclosing owned memory cap exceeded");
    limit(p.per_replica_inventoried_bytes,caps.maximum_per_replica_inventoried_bytes,"Gaussian RHF enclosing replica memory cap exceeded");
    limit(p.node_inventoried_bytes,caps.maximum_node_inventoried_bytes,"Gaussian RHF enclosing node memory cap exceeded");
    const I fock_calls=add(options.scf.maximum_iterations,1);
    p.progress_callback_upper_bound=add(8,add(p.one_electron_panel_calls,
        add(options.scf.maximum_iterations,mul(fock_calls,p.fock.progress_callback_upper_bound))));
    limit(p.progress_callback_upper_bound,caps.maximum_progress_callbacks,"Gaussian RHF enclosing callback cap exceeded");
    p.work_units_upper_bound=add(mul(panels,add(caps.one_electron.maximum_work_units,caps.nuclear.maximum_work_units)),
        add(caps.nuclear.maximum_work_units,add(p.scf.maximum_work_units,p.fock.work_units_upper_bound)));
    p.work_units_upper_bound=add(p.work_units_upper_bound,
        add(mul(8192,mul(p.n_kpoints,mul(nn,p.n_basis))),
            mul(add(p.progress_callback_upper_bound,2),p.input_check_work_units)));
    limit(p.work_units_upper_bound,caps.maximum_work_units,"Gaussian RHF enclosing work cap exceeded");
    // Re-admit the Fock producer with the largest surrounding SCF/capture
    // allocation BEFORE any all-k S/H or solver arrays are allocated.
    const I surrounding=std::max(p.scf.peak_owned_numerical_bytes-density_bytes,p.scf.output_numerical_bytes);
    const Live fock_live=extended(live,add(p.overlap_hcore_bytes,add(surrounding,
        add(system_bytes,add(p.frozen_selection_bytes,p.control_storage_reservation_bytes)))));
    plan_periodic_gaussian_fock(context,config,fock_live,caps.fock);
    return p;
}

PeriodicGaussianRHFResult run_periodic_gaussian_rhf(
    std::shared_ptr<const PeriodicGaussianSourceContext> context, const BasisSet& ao, const BasisSet& auxiliary,
    const PeriodicSystem& system, PeriodicGaussianRHFFrozenSelection mask,
    const PeriodicGaussianRHFOptions& options_in, const PeriodicGaussianFockConfig& config_in,
    const Live& live_in, const PeriodicGaussianRHFCaps& caps_in,
    PeriodicGaussianRHFCallback progress, void* progress_context) {
    if (!context) throw std::invalid_argument("Gaussian RHF requires a live Gaussian source context");
    // These bounded-size controls may be Python-visible/mutable; callbacks
    // cannot change the admitted controls halfway through a calculation.
    const auto options=options_in; const auto config=config_in; const auto live=live_in; const auto caps=caps_in;
    PeriodicGaussianRHFResult result; result.context_=context;
    result.plan_=plan_periodic_gaussian_rhf(*context,system,mask,options,config,live,caps);
    const auto& p=result.plan_; auto& d=result.diagnostics_;
    const I k=p.n_kpoints,n=p.n_basis,nn=mul(n,n),elements=mul(k,nn),occupied=p.electrons_per_cell/2;
    const I system_bytes=mul(28,p.atom_count), basis_bytes=context->inventory().combined_borrowed_active_numeric_bytes;
    const I one_surround=add(p.overlap_hcore_bytes,add(p.frozen_selection_bytes,p.control_storage_reservation_bytes));
    Adapter adapter{context,ao,auxiliary,system,mask,options,config,caps,p,{},d,
        input_identity(*context,ao,auxiliary,system,mask,p),{},progress,progress_context};
    result.input_=ascii(adapter.original);
    PeriodicGaussianRHFProgress event; event.stage=Stage::Begin; adapter.emit(event);
    std::vector<C> overlap(elements), hcore(elements);
    Digest one("vibeqc.periodic.gaussian-rhf.one-electron"); one.text(adapter.original); hash_options(one,options);
    std::string nuclei;
    {
        const auto energy=build_periodic_gaussian_nuclear_ewald(context,system,options.nuclear,
            nuclear_live(extended(live,add(one_surround,basis_bytes))),caps.nuclear);
        nuclei=energy.nuclei_policy_identity_sha256(); d.nuclear_energy_per_cell=energy.energy();
        one.text(nuclei); one.text(energy.scalar_source_identity_sha256()); one.text(energy.payload_identity_sha256());
        one.real(energy.energy());
    }
    event.stage=Stage::NuclearEnergy; adapter.emit(event);
    for (I point=0; point<k; ++point) for (I begin=0; begin<nn; begin+=std::min(options.one_electron_pair_block,nn-begin)) {
        const I count=std::min(options.one_electron_pair_block,nn-begin);
        {
            const auto panel=build_periodic_gaussian_one_electron_panel(context,ao,auxiliary,system,point,begin,count,
                options.one_electron,one_live(extended(live,add(one_surround,system_bytes))),caps.one_electron);
            if (panel.context_handle().get()!=context.get()) throw std::logic_error("Gaussian RHF S/T owner mismatch");
            one.u64(point); one.u64(begin); one.u64(count);
            one.text(panel.operator_source_identity_sha256()); one.text(panel.payload_identity_sha256());
            for (I pair=0; pair<count; ++pair) {
                overlap[point*nn+begin+pair]=panel.element(0,pair);
                hcore[point*nn+begin+pair]=panel.element(1,pair);
            }
        }
        ++d.completed_panel_calls; event.stage=Stage::OneElectron; event.k_index=point; event.pair_begin=begin; adapter.emit(event);
        {
            const auto panel=build_periodic_gaussian_nuclear_panel(context,ao,auxiliary,system,point,begin,count,
                options.nuclear,nuclear_live(extended(live,one_surround)),caps.nuclear);
            if (panel.context_handle().get()!=context.get() || panel.nuclei_policy_identity_sha256()!=nuclei)
                throw std::logic_error("Gaussian RHF nuclear operator/energy source mismatch");
            one.text(panel.panel_source_identity_sha256()); one.text(panel.payload_identity_sha256());
            for (I pair=0; pair<count; ++pair) hcore[point*nn+begin+pair]+=panel.element(3,pair);
        }
        ++d.completed_panel_calls; adapter.emit(event);
    }
    if (d.completed_panel_calls!=p.one_electron_panel_calls) throw std::logic_error("Gaussian RHF panel receipt mismatch");
    result.one_electron_=ascii(one.finish());
    const I density_bytes=mul(16,elements);
    adapter.fock_live=extended(live,add(one_surround,add(system_bytes,p.scf.peak_owned_numerical_bytes-density_bytes)));
    BoundedPeriodicRHFTwoElectronProvider provider{Adapter::evaluate,&adapter,0,p.fock.owned_numeric_upper_bound,p.fock.work_units_upper_bound};
    auto scf_caps=caps.scf; scf_caps.other_live_numerical_bytes=add(p.borrowed_input_numeric_bytes,extras(live));
    std::optional<BoundedPeriodicRHFResult> numerical;
    numerical.emplace(solve_bounded_periodic_rhf(
        {k,n,p.electrons_per_cell,d.nuclear_energy_per_cell,overlap.data(),overlap.size(),hcore.data(),hcore.size()},
        provider,options.scf,scf_caps,progress ? Adapter::scf_progress : nullptr,&adapter));
    d.last_scf_snapshot=numerical->final_snapshot(); d.scf=numerical->diagnostics();
    if (!numerical->converged()) {
        result.unfinished_.emplace(std::move(*numerical));
        event.stage=Stage::Finished; event.scf=d.last_scf_snapshot; adapter.emit(event); adapter.check(); return result;
    }
    const I rank=numerical->memory().retained_rank;
    event.stage=Stage::CaptureRebuild; event.scf=d.last_scf_snapshot; adapter.emit(event);
    PeriodicRestrictedMeanFieldInput input;
    input.periodic_dimension=3; input.mesh=context->mesh().mesh(); input.is_shift=context->mesh().is_shift();
    input.reciprocal_lattice=context->reciprocal_lattice(); input.n_basis=n; input.n_effective_orbitals=rank;
    input.electrons_per_cell=p.electrons_per_cell; input.minimum_band_gap_hartree=options.scf.minimum_band_gap_hartree;
    {
        // Reconstruct EXACT Hermitian D from the final physical orbitals.
        // No last-iteration density is relabelled as this new density.
        std::vector<C> density(elements);
        const auto* coefficients=numerical->coefficients_data(); const auto* energies=numerical->orbital_energies_data();
        for (I point=0; point<k; ++point) for (I mu=0; mu<n; ++mu) for (I nu=mu; nu<n; ++nu) {
            ComplexSum sum;
            for (I i=0; i<occupied; ++i) {
                sum.append(coefficients[(point*n+mu)*rank+i]*std::conj(coefficients[(point*n+nu)*rank+i]));
            }
            const C raw=2.0*sum.get(); finite(raw.real()); finite(raw.imag());
            const C z(raw.real(),mu==nu ? 0.0 : raw.imag());
            density[point*nn+mu*n+nu]=z; density[point*nn+nu*n+mu]=std::conj(z);
        }
        adapter.fock_live=extended(live,add(one_surround,add(system_bytes,numerical->memory().output_numerical_bytes)));
        const auto response=adapter.response(density.data(),elements);
        result.final_fock_=ascii(adapter.last_fock);
        const auto& g=response.matrix_row_major();
        Digest calculation("vibeqc.periodic.gaussian-rhf.calculation");
        calculation.text(adapter.original); calculation.text(string(result.one_electron_));
        calculation.text(adapter.last_fock); hash_options(calculation,options);
        input.calculation_identity=calculation.finish();
        Sum electronic; double all_commutator=0;
        const PeriodicMeanFieldValidationTolerances fixed;
        for (I point=0; point<k; ++point) {
            PeriodicMeanFieldComplexMatrix s(n,n),f(n,n),c(n,rank);
            Eigen::VectorXd eps(rank), occupations(rank);
            std::vector<std::uint8_t> frozen(rank,0),active(rank,0),virtuals(rank,0);
            double projection=0;
            for (I mu=0; mu<n; ++mu) for (I nu=0; nu<n; ++nu) {
                const I at=point*nn+mu*n+nu, reverse=point*nn+nu*n+mu;
                const C h=projected(hcore.data(),point*nn,n,mu,nu);
                // Match the numerical leaf: add raw H+G first, then audit
                // and project that physical sum. H is projected separately
                // only in the energy expression.
                const C raw=hcore[at]+g[at], adjoint=std::conj(hcore[reverse]+g[reverse]);
                finite(raw.real()); finite(raw.imag());
                const double defect=std::abs(raw-adjoint), scale=std::max(std::abs(raw),std::abs(adjoint));
                finite(defect); finite(scale);
                d.maximum_capture_fock_hermiticity_defect=std::max(d.maximum_capture_fock_hermiticity_defect,defect);
                if (defect>fixed.matrix_absolute+fixed.matrix_relative*std::max(1.0,scale)
                    || defect>options.scf.hermitian_absolute_tolerance+options.scf.hermitian_relative_tolerance*scale)
                    throw std::invalid_argument("Gaussian RHF final raw Fock Hermiticity gate failed");
                const C value=mu==nu ? C(raw.real(),0.0) : average(raw,adjoint);
                projection=std::hypot(projection,std::abs(raw-value)); finite(projection);
                f(mu,nu)=value; s(mu,nu)=projected(overlap.data(),point*nn,n,mu,nu);
                d.maximum_capture_fock_change=std::max(d.maximum_capture_fock_change,std::abs(value-numerical->fock_data()[at]));
                const C e=density[reverse]*(h+value);
                electronic.append(e.real()/(2.0*static_cast<double>(k)));
            }
            if (projection>options.scf.maximum_fock_projection_error)
                throw std::invalid_argument("Gaussian RHF final Fock projection budget exceeded");
            for (I i=0; i<rank; ++i) {
                eps(i)=energies[point*rank+i]; occupations(i)=i<occupied ? 2.0 : 0.0;
                if (i<occupied) { frozen[i]=mask.data[point*occupied+i]; active[i]=1-frozen[i]; }
                else virtuals[i]=1;
                for (I mu=0; mu<n; ++mu) c(mu,i)=coefficients[(point*n+mu)*rank+i];
            }
            capture_audit(s,f,c,eps,density.data()+point*nn,options.scf,d,all_commutator);
            const auto coordinate=context->k_record(point).cartesian;
            input.add_kpoint(Eigen::Vector3d(coordinate[0],coordinate[1],coordinate[2]),1.0/static_cast<double>(k),
                std::move(s),std::move(f),std::move(c),std::move(eps),std::move(occupations),
                std::move(frozen),std::move(active),std::move(virtuals));
        }
        d.capture_commutator_frobenius_rms=all_commutator/std::sqrt(static_cast<double>(k));
        finite(d.capture_commutator_frobenius_rms);
        if (d.capture_commutator_frobenius_rms>options.scf.commutator_tolerance)
            throw std::invalid_argument("Gaussian RHF rebuilt physical Fock exceeds requested SCF commutator tolerance");
        d.captured_energy_per_cell=d.nuclear_energy_per_cell+electronic.get(); finite(d.captured_energy_per_cell);
        d.capture_energy_change=d.captured_energy_per_cell-d.last_scf_snapshot.energy_per_cell; finite(d.capture_energy_change);
        if (std::abs(d.capture_energy_change)>options.scf.energy_change_tolerance)
            throw std::invalid_argument("Gaussian RHF final density reconstruction energy gate failed");
        input.reference_energy_per_cell=d.captured_energy_per_cell;
    }
    // D_C and transient response are gone before the existing Eigen validator
    // runs. Its fixed full-AO/overlap/gap gates cannot be relaxed by this API.
    input.converged=true;
    result.state_=make_periodic_restricted_mean_field_state(std::move(input));
    numerical.reset();
    Digest reference("vibeqc.periodic.gaussian-rhf.reference");
    reference.text(adapter.original); reference.text(string(result.one_electron_)); reference.text(string(result.final_fock_));
    reference.text(result.state_->state_identity_sha256()); reference.text(result.state_->numerical_payload_sha256());
    result.reference_=ascii(reference.finish());
    event.stage=Stage::Captured; adapter.emit(event); event.stage=Stage::Finished; adapter.emit(event); adapter.check();
    return result;
}

const std::shared_ptr<const PeriodicRestrictedMeanFieldState>& PeriodicGaussianRHFResult::state_handle() const {
    if (!state_) throw std::logic_error("Gaussian RHF has no converged matched mean-field state"); return state_;
}
const BoundedPeriodicRHFResult& PeriodicGaussianRHFResult::unconverged_numerical_result() const {
    if (!unfinished_) throw std::logic_error("Gaussian RHF has no unfinished numerical result"); return *unfinished_;
}
std::string PeriodicGaussianRHFResult::original_input_identity_sha256() const { return string(input_); }
std::string PeriodicGaussianRHFResult::one_electron_source_identity_sha256() const { return string(one_electron_); }
std::string PeriodicGaussianRHFResult::final_fock_source_identity_sha256() const {
    state_handle(); return string(final_fock_);
}
std::string PeriodicGaussianRHFResult::reference_source_identity_sha256() const {
    state_handle(); return string(reference_);
}
void PeriodicGaussianRHFResult::verify_physical_inputs(const BasisSet& ao,const BasisSet& auxiliary,
                                                      const PeriodicSystem& system) const {
    ieee(); state_handle();
    if (!context_ || system.unit_cell.size()!=plan_.atom_count)
        throw std::invalid_argument("Gaussian RHF original physical input extent mismatch");
    context_->verify_bases(ao,auxiliary,exact_basis_caps(*context_));
    if (check_system(*context_,system,plan_.atom_count)!=plan_.electrons_per_cell)
        throw std::invalid_argument("Gaussian RHF original physical electron count mismatch");
    Digest h("vibeqc.periodic.gaussian-rhf.original-input");
    h.text(context_->source_context_identity_sha256());
    h.u64(plan_.atom_count); h.u64(static_cast<I>(static_cast<std::int64_t>(system.charge)));
    h.u64(system.multiplicity); h.u64(plan_.electrons_per_cell);
    for (unsigned r=0; r<3; ++r) for (unsigned c=0; c<3; ++c) h.real(system.lattice(r,c));
    for (const auto& atom : system.unit_cell) { h.u64(atom.Z); for (double x : atom.xyz) h.real(x); }
    h.u64(plan_.frozen_selection_bytes);
    for (I k=0; k<plan_.n_kpoints; ++k) {
        const auto& mask=state_->frozen_core_mask(k);
        h.bytes(mask.data(),plan_.electrons_per_cell/2);
    }
    if (ascii(h.finish())!=input_)
        throw std::invalid_argument("Gaussian RHF original physical input content mismatch");
}
} // namespace vibeqc
