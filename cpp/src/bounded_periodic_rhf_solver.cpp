#include "vibeqc/bounded_periodic_rhf_solver.hpp"

#include <algorithm>
#include <cfenv>
#include <cfloat>
#include <cmath>
#include <limits>
#include <stdexcept>
#include "vibeqc/hermitian_jacobi.hpp"

namespace vibeqc {
namespace {
using I=std::uint64_t;
using Z=std::complex<double>;
using Options=BoundedPeriodicRHFOptions;
using Memory=BoundedPeriodicRHFMemoryPlan;
using Diagnostics=BoundedPeriodicRHFDiagnostics;
using Snapshot=BoundedPeriodicRHFSnapshot;
using Status=BoundedPeriodicRHFStatus;
static_assert(sizeof(double)==8 && sizeof(Z)==16 && std::numeric_limits<double>::is_iec559
              && std::numeric_limits<double>::digits==53,"bounded periodic RHF requires binary64");
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error "bounded periodic RHF forbids fast/finite-only math"
#endif
#if FLT_EVAL_METHOD != 0
#error "bounded periodic RHF requires binary64 evaluation"
#endif
I add(I a,I b) {
    if (b>std::numeric_limits<I>::max()-a) throw std::overflow_error("bounded periodic RHF count sum overflow");
    return a+b;
}
I mul(I a,I b) {
    if (a && b>std::numeric_limits<I>::max()/a) throw std::overflow_error("bounded periodic RHF count product overflow");
    return a*b;
}
void limit(I a,I b,const char* text) { if (a>b) throw std::length_error(text); }
double finite(double x) {
    if (!std::isfinite(x)) throw std::overflow_error("bounded periodic RHF arithmetic is nonfinite");
    return x;
}
Z finite(Z z) { finite(z.real()); finite(z.imag()); return z; }
double abs(Z z) { return finite(std::abs(finite(z))); }
double norm_add(double a,double b) { return finite(std::hypot(a,b)); }
struct RealSum {
    double value=0,correction=0;
    void push(double x) {
        finite(x); const double next=finite(value+x);
        correction=finite(correction+(std::abs(value)>=std::abs(x)?(value-next)+x:(x-next)+value));
        value=next;
    }
    double get() const { return finite(value+correction); }
};
struct Sum {
    RealSum real,imag;
    void push(Z x) { real.push(x.real()); imag.push(x.imag()); }
    Z get() const { return Z(real.get(),imag.get()); }
};
double average(double a,double b) {
    if (a==b) return a;
    int e=0; std::frexp(std::max(std::abs(a),std::abs(b)),&e);
    const double x=std::scalbn(a,-e),y=std::scalbn(b,-e);
    if (std::scalbn(x,e)!=a || std::scalbn(y,e)!=b)
        throw std::overflow_error("bounded periodic RHF Hermitian projection loses input range");
    return finite(std::scalbn(x+y,e-1));
}
Z hermitian(const Z* matrix,I n,I i,I j) {
    const Z a=matrix[i*n+j];
    if (i==j) return Z(a.real(),0);
    const Z b=matrix[j*n+i];
    return Z(average(a.real(),b.real()),average(a.imag(),-b.imag()));
}
void pair_control(double a,double r,const char* message) {
    if (!std::isfinite(a) || a<0 || !std::isfinite(r) || r<0 || r>=1 || (a==0 && r==0))
        throw std::invalid_argument(message);
}
void controls(const Options& o) {
    if (o.maximum_diis_history!=0) throw std::invalid_argument("bounded periodic RHF v1 requires maximum_diis_history=0; DIIS is not implemented here");
    if (!o.maximum_iterations || !o.jacobi_max_sweeps)
        throw std::invalid_argument("bounded periodic RHF requires explicit iteration and Jacobi sweep limits");
    for (double x:{o.jacobi_relative_tolerance,o.eigen_relative_tolerance})
        if (!std::isfinite(x) || x<=0 || x>=1)
            throw std::invalid_argument("bounded periodic RHF requires positive numerical eigensolver tolerances below one");
    pair_control(o.overlap_rank_absolute_floor,o.overlap_rank_relative_floor,"bounded periodic RHF requires an explicit overlap rank cutoff");
    pair_control(o.overlap_negative_absolute_tolerance,o.overlap_negative_relative_tolerance,"bounded periodic RHF requires explicit overlap negative controls");
    pair_control(o.hermitian_absolute_tolerance,o.hermitian_relative_tolerance,"bounded periodic RHF requires explicit Hermitian controls");
    pair_control(o.algebra_absolute_tolerance,o.algebra_relative_tolerance,"bounded periodic RHF requires explicit algebra controls");
    pair_control(o.density_closure_absolute_tolerance,o.density_closure_relative_tolerance,"bounded periodic RHF requires explicit density closure controls");
    for (double x:{o.maximum_overlap_projection_error,o.maximum_hcore_projection_error,
            o.maximum_fock_projection_error,o.minimum_band_gap_hartree})
        if (!std::isfinite(x) || x<0) throw std::invalid_argument("bounded periodic RHF projection/gap controls must be finite nonnegative");
    for (double x:{o.commutator_tolerance,o.energy_change_tolerance})
        if (!std::isfinite(x) || x<=0) throw std::invalid_argument("bounded periodic RHF convergence tolerances must be finite positive");
    volatile double tiny=std::numeric_limits<double>::denorm_min(),one=1.0,zero=0.0;
    if (std::fegetround()!=FE_TONEAREST || !(tiny>0) || std::fma(tiny,one,zero)!=tiny)
        throw std::invalid_argument("bounded periodic RHF requires round-to-nearest and gradual underflow");
}
void extent(I count,I bytes) {
    limit(bytes,static_cast<I>(std::numeric_limits<std::ptrdiff_t>::max()),"bounded periodic RHF address extent exceeded");
    limit(count,static_cast<I>(std::vector<Z>().max_size()),"bounded periodic RHF vector extent exceeded");
}
void view(const Z* data,std::size_t accessible,I count) {
    const I bytes=mul(count,sizeof(Z));
    if (!data || accessible<count || reinterpret_cast<std::uintptr_t>(data)%alignof(Z)!=0)
        throw std::invalid_argument("bounded periodic RHF invalid input pointer/alignment/extent");
    if (bytes>std::numeric_limits<std::uintptr_t>::max()-reinterpret_cast<std::uintptr_t>(data))
        throw std::overflow_error("bounded periodic RHF input pointer range overflow");
}
double boundary(double a,double r,double scale) { return finite(a+finite(r*scale)); }
void check(Z a,Z b,const Options& o,double& maximum,const char* message) {
    const double error=abs(a-b); maximum=std::max(maximum,error);
    if (error>boundary(o.algebra_absolute_tolerance,o.algebra_relative_tolerance,std::max(abs(a),abs(b))))
        throw std::runtime_error(message);
}
// Audit both raw orientations BEFORE any projection. The Frobenius correction
// includes both triangle entries and the full diagonal imaginary lane.
void audit(const Z* raw,I n,double at,double rt,double cap,double& defect,double& correction) {
    double change=0;
    for (I i=0;i<n;++i) for (I j=i;j<n;++j) {
        const Z a=finite(raw[i*n+j]),b=finite(raw[j*n+i]);
        const double error=abs(a-std::conj(b)); defect=std::max(defect,error);
        if (error>boundary(at,rt,std::max(abs(a),abs(b))))
            throw std::invalid_argument("bounded periodic RHF raw operator failed Hermitian defect gate");
        const Z projected=hermitian(raw,n,i,j);
        change=norm_add(change,abs(projected-a));
        if (i!=j) change=norm_add(change,abs(std::conj(projected)-b));
    }
    correction=std::max(correction,change);
    if (change>cap) throw std::invalid_argument("bounded periodic RHF operator projection correction exceeds explicit cap");
}
void project_in_place(Z* raw,I n) {
    for (I i=0;i<n;++i) for (I j=i;j<n;++j) {
        const Z value=hermitian(raw,n,i,j);
        raw[i*n+j]=value;
        if (i!=j) raw[j*n+i]=std::conj(value);
    }
}
template<class Entry>
void apply(I n,I r,Entry entry,const Z* x,Z* out) {
    for (I i=0;i<n;++i) for (I a=0;a<r;++a) {
        Sum value; for (I j=0;j<n;++j) value.push(entry(i,j)*x[j*r+a]);
        out[i*r+a]=value.get();
    }
}
void left_product(I n,I r,const Z* x,const Z* applied,Z* out) {
    for (I a=0;a<r;++a) for (I b=0;b<r;++b) {
        Sum value; for (I i=0;i<n;++i) value.push(std::conj(x[i*r+a])*applied[i*r+b]);
        out[a*r+b]=value.get();
    }
}
template<class Entry>
double eigen_audit(I n,Entry entry,const Z* u,const double* values,const Options& o) {
    double residual=0,scale=0,unitary=0;
    for (I i=0;i<n;++i) for (I j=0;j<n;++j) scale=norm_add(scale,abs(entry(i,j)));
    for (I i=0;i<n;++i) for (I a=0;a<n;++a) {
        Sum value; for (I j=0;j<n;++j) value.push(entry(i,j)*u[j*n+a]);
        residual=norm_add(residual,abs(value.get()-values[a]*u[i*n+a]));
    }
    for (I a=0;a<n;++a) for (I b=0;b<n;++b) {
        Sum value; for (I i=0;i<n;++i) value.push(std::conj(u[i*n+a])*u[i*n+b]);
        check(value.get(),double(a==b),o,unitary,"bounded periodic RHF eigenvectors failed independent unitarity audit");
    }
    const double relative=scale==0?residual:finite(residual/scale);
    if (relative>o.eigen_relative_tolerance)
        throw std::runtime_error("bounded periodic RHF original-operator eigen residual failed");
    return relative;
}
void jacobi(Z* m,Z* u,double* values,I n,const Options& o,Diagnostics& d) {
    const auto result=hermitian_jacobi_in_place(m,n*n,u,n*n,values,n,n,
        HermitianJacobiOptions{o.jacobi_max_sweeps,o.jacobi_relative_tolerance});
    d.jacobi_sweeps=add(d.jacobi_sweeps,result.sweeps);
    if (result.status!=HermitianJacobiStatus::Success || result.scaling_underflow_components!=0)
        throw std::runtime_error("bounded periodic RHF Jacobi failed or lost a scaled input lane");
}
I overlap_eigen(const Z* s,I n,Z* m,Z* u,double* values,const Options& o,Diagnostics& d) {
    for (I i=0;i<n;++i) for (I j=0;j<n;++j) m[i*n+j]=hermitian(s,n,i,j);
    jacobi(m,u,values,n,o,d);
    d.maximum_overlap_eigen_relative_residual=std::max(d.maximum_overlap_eigen_relative_residual,
        eigen_audit(n,[s,n](I i,I j){return hermitian(s,n,i,j);},u,values,o));
    d.minimum_overlap_eigenvalue=std::min(d.minimum_overlap_eigenvalue,values[0]);
    const double scale=std::max(std::abs(values[0]),std::abs(values[n-1]));
    if (values[0]<-boundary(o.overlap_negative_absolute_tolerance,o.overlap_negative_relative_tolerance,scale))
        throw std::invalid_argument("bounded periodic RHF overlap has a materially negative eigenvalue");
    const double cutoff=std::max(o.overlap_rank_absolute_floor,finite(o.overlap_rank_relative_floor*values[n-1]));
    I r=0; for (I a=0;a<n;++a) if (values[a]>cutoff) ++r;
    return r;
}
void density_from_coefficients(I n,I r,I occupied,const Z* c,Z* d) {
    for (I i=0;i<n;++i) for (I j=i;j<n;++j) {
        Sum sum; for (I a=0;a<occupied;++a) sum.push(c[i*r+a]*std::conj(c[j*r+a]));
        Z value=finite(2.0*sum.get());
        if (i==j) value=Z(value.real(),0);
        d[i*n+j]=value; if (i!=j) d[j*n+i]=std::conj(value);
    }
}
// The n*r applied panel is retained until eigen_audit reconstructs the
// ORIGINAL projected operator X^H F X, independently of Jacobi's work array.
template<class Entry>
double diagonalize(I n,I r,Entry entry,const Z* x,Z* m,Z* u,Z* applied,Z* reconstruction,
                   double* values,Z* c,double* energies,const Options& o,Diagnostics& d) {
    apply(n,r,entry,x,applied); left_product(n,r,x,applied,m);
    double matrix_norm=0; for (I a=0;a<r*r;++a) matrix_norm=norm_add(matrix_norm,abs(m[a]));
    const double cap=boundary(o.algebra_absolute_tolerance,o.algebra_relative_tolerance,matrix_norm);
    audit(m,r,o.algebra_absolute_tolerance,o.algebra_relative_tolerance,cap,
        d.maximum_reduced_operator_hermitian_defect,d.maximum_reduced_operator_projection_frobenius);
    project_in_place(m,r); jacobi(m,u,values,r,o,d);
    // Evaluate X^H[(F X)U]-U epsilon in O(n*r^2), retaining the
    // original F X panel. Reconstructing each reduced element separately
    // inside a cubic eigenaudit would need O(n*r^3) unplanned work.
    for (I i=0;i<n;++i) for (I col=0;col<r;++col) {
        Sum value; for (I j=0;j<r;++j) value.push(applied[i*r+j]*u[j*r+col]);
        reconstruction[i*r+col]=value.get();
    }
    double error=0,unitary=0;
    for (I row=0;row<r;++row) for (I col=0;col<r;++col) {
        Sum value,gram;
        for (I i=0;i<n;++i) value.push(std::conj(x[i*r+row])*reconstruction[i*r+col]);
        error=norm_add(error,abs(value.get()-u[row*r+col]*values[col]));
        for (I i=0;i<r;++i) gram.push(std::conj(u[i*r+row])*u[i*r+col]);
        check(gram.get(),double(row==col),o,unitary,"bounded periodic RHF reduced eigenvectors failed unitarity audit");
    }
    const double residual=matrix_norm==0?error:finite(error/matrix_norm);
    if (residual>o.eigen_relative_tolerance)
        throw std::runtime_error("bounded periodic RHF original projected-operator eigen residual failed");
    for (I i=0;i<n;++i) for (I a=0;a<r;++a) {
        Sum value; for (I b=0;b<r;++b) value.push(x[i*r+b]*u[b*r+a]);
        c[i*r+a]=value.get();
    }
    for (I a=0;a<r;++a) energies[a]=values[a];
    return residual;
}
void density_audits(I n,I r,I occupied,const Z* s,const Z* f,const Z* d,const Z* x,
                    const Z* c,const double* eps,Z* m,Z* u,Z* a,Z* b,
                    const Options& o,Snapshot& snap,double& commutator_norm,bool& closed) {
    const auto se=[s,n](I i,I j){return hermitian(s,n,i,j);};
    apply(n,r,se,c,a); left_product(n,r,c,a,m);
    for (I i=0;i<r;++i) for (I j=0;j<r;++j)
        check(m[i*r+j],double(i==j),o,snap.maximum_coefficient_metric_error,
              "bounded periodic RHF physical coefficients failed metric orthonormality");
    apply(n,r,[f,n](I i,I j){return f[i*n+j];},c,b);
    for (I i=0;i<n;++i) for (I j=0;j<r;++j)
        snap.maximum_full_ao_eigen_residual=std::max(snap.maximum_full_ao_eigen_residual,abs(b[i*r+j]-a[i*r+j]*eps[j]));
    double closure=0,dnorm=0,cnorm=0;
    Sum electron;
    for (I i=0;i<n;++i) for (I j=0;j<n;++j) {
        Sum value; for (I l=0;l<occupied;++l) value.push(c[i*r+l]*std::conj(c[j*r+l]));
        const Z dc=finite(2.0*value.get());
        closure=norm_add(closure,abs(d[i*n+j]-dc)); dnorm=norm_add(dnorm,abs(d[i*n+j])); cnorm=norm_add(cnorm,abs(dc));
        electron.push(d[i*n+j]*se(j,i));
    }
    const double scale=std::max({1.0,dnorm,cnorm});
    snap.maximum_density_closure_frobenius=std::max(snap.maximum_density_closure_frobenius,closure);
    snap.maximum_density_closure_relative=std::max(snap.maximum_density_closure_relative,finite(closure/scale));
    closed &= closure<=boundary(o.density_closure_absolute_tolerance,o.density_closure_relative_tolerance,scale);
    check(electron.get(),2.0*static_cast<double>(occupied),o,snap.maximum_electron_count_error,
          "bounded periodic RHF physical density has the wrong electron count");
    // P'=X^H S D S X, and physical F'=X^H F X. These use the actual
    // evaluated density, not its newly diagonalized canonical projector.
    apply(n,r,se,x,a);
    apply(n,r,[d,n](I i,I j){return d[i*n+j];},a,b); left_product(n,r,a,b,m);
    apply(n,r,[f,n](I i,I j){return f[i*n+j];},x,a); left_product(n,r,x,a,u);
    double point_norm=0;
    for (I i=0;i<r;++i) for (I j=0;j<r;++j) {
        Sum fp,pf,pp;
        for (I l=0;l<r;++l) {
            fp.push(u[i*r+l]*m[l*r+j]); pf.push(m[i*r+l]*u[l*r+j]); pp.push(m[i*r+l]*m[l*r+j]);
        }
        const double error=abs(fp.get()-pf.get()); point_norm=norm_add(point_norm,error);
        snap.maximum_commutator_element=std::max(snap.maximum_commutator_element,error);
        check(pp.get(),2.0*m[i*r+j],o,snap.maximum_metric_idempotency_error,
              "bounded periodic RHF physical density failed metric idempotency");
    }
    commutator_norm=norm_add(commutator_norm,point_norm);
}
void admit(const Memory& m,const BoundedPeriodicRHFCaps& c) {
    limit(m.peak_owned_numerical_bytes,c.maximum_owned_numerical_bytes,"bounded periodic RHF owned numerical byte cap exceeded");
    limit(m.total_numerical_bytes,c.maximum_total_numerical_bytes,"bounded periodic RHF total numerical byte cap exceeded");
}
} // namespace

BoundedPeriodicRHFMemoryPlan plan_bounded_periodic_rhf_solver(
    I k,I n,I electrons,I r,const BoundedPeriodicRHFTwoElectronProvider& provider,const Options& o,I other) {
    controls(o);
    if (!k || !n || !electrons || electrons%2!=0 || electrons/2>=n || r>n
        || (r && r<=electrons/2)) throw std::invalid_argument("bounded periodic RHF requires closed-shell dimensions and at least one virtual band");
    if (!provider.evaluate || !provider.maximum_work_units_per_call)
        throw std::invalid_argument("bounded periodic RHF requires a bounded native two-electron callback");
    Memory m; m.n_kpoints=k; m.n_basis=n; m.n_occupied=electrons/2; m.retained_rank=r;
    const I nn=mul(n,n),knn=mul(k,nn),nr=mul(n,r),knr=mul(k,nr),kr=mul(k,r);
    m.jacobi_workspace_bytes=add(mul(32,nn),mul(8,n));
    m.overlap_discovery_owned_bytes=m.jacobi_workspace_bytes;
    m.borrowed_input_bytes=mul(32,knn); m.provider_retained_bytes=provider.retained_numerical_bytes;
    m.provider_workspace_bytes=provider.peak_workspace_numerical_bytes; m.other_live_numerical_bytes=other;
    m.input_validation_work_units=mul(512,add(knn,1));
    const I n3=mul(nn,n),r3=mul(mul(r,r),r);
    m.overlap_discovery_work_units=mul(1024,mul(k,add(mul(add(o.jacobi_max_sweeps,2),n3),add(nn,1))));
    m.peak_owned_numerical_bytes=m.overlap_discovery_owned_bytes;
    m.maximum_work_units=add(m.input_validation_work_units,m.overlap_discovery_work_units);
    if (r) {
        m.orthogonalizer_bytes=mul(16,knr); m.density_bytes=mul(16,knn); m.fock_bytes=m.density_bytes;
        m.coefficient_bytes=mul(16,knr); m.orbital_energy_bytes=mul(8,kr);
        m.contraction_workspace_bytes=mul(32,nr);
        m.output_numerical_bytes=add(add(m.density_bytes,m.fock_bytes),add(m.coefficient_bytes,m.orbital_energy_bytes));
        m.peak_owned_numerical_bytes=add(m.output_numerical_bytes,
            add(m.orthogonalizer_bytes,add(m.jacobi_workspace_bytes,m.contraction_workspace_bytes)));
        const I contraction=add(mul(nn,r),mul(n,mul(r,r)));
        m.orthogonalizer_work_units=add(m.overlap_discovery_work_units,mul(1024,mul(k,add(contraction,1))));
        const I iteration_math=add(mul(add(o.jacobi_max_sweeps,2),r3),
            add(mul(8,contraction),add(mul(nn,m.n_occupied),add(nn,add(nr,1)))));
        m.initial_density_work_units=mul(2048,mul(k,iteration_math));
        m.work_units_per_evaluated_iteration=add(m.initial_density_work_units,provider.maximum_work_units_per_call);
        m.maximum_work_units=add(m.maximum_work_units,add(m.orthogonalizer_work_units,
            add(m.initial_density_work_units,mul(o.maximum_iterations,m.work_units_per_evaluated_iteration))));
    }
    m.total_numerical_bytes=add(add(m.borrowed_input_bytes,m.provider_retained_bytes),
        add(other,add(m.peak_owned_numerical_bytes,r?m.provider_workspace_bytes:0)));
    extent(knn,mul(16,knn)); extent(knr,mul(16,knr)); extent(nn,mul(16,nn)); extent(nr,mul(16,nr));
    extent(kr,mul(8,kr)); extent(n,mul(8,n));
    return m;
}

void BoundedPeriodicRHFResult::require_live() const {
    if (!snapshot_.evaluated_iteration || density_.size()!=memory_.density_bytes/16
        || fock_.size()!=memory_.fock_bytes/16 || coefficients_.size()!=memory_.coefficient_bytes/16
        || energies_.size()!=memory_.orbital_energy_bytes/8)
        throw std::logic_error("bounded periodic RHF result is consumed or not evaluated");
}
const Z* BoundedPeriodicRHFResult::density_data() const { require_live(); return density_.data(); }
const Z* BoundedPeriodicRHFResult::fock_data() const { require_live(); return fock_.data(); }
const Z* BoundedPeriodicRHFResult::coefficients_data() const { require_live(); return coefficients_.data(); }
const double* BoundedPeriodicRHFResult::orbital_energies_data() const { require_live(); return energies_.data(); }

BoundedPeriodicRHFResult solve_bounded_periodic_rhf(
    const BoundedPeriodicRHFInput& input,const BoundedPeriodicRHFTwoElectronProvider& provider,
    const Options& o,const BoundedPeriodicRHFCaps& caps,
    BoundedPeriodicRHFProgressCallback progress,void* progress_context) {
    for (I x:{caps.maximum_owned_numerical_bytes,caps.maximum_total_numerical_bytes,caps.maximum_work_units,caps.maximum_fock_calls})
        if (!x) throw std::invalid_argument("bounded periodic RHF caps must be positive");
    const I k=input.n_kpoints,n=input.n_basis;
    auto memory=plan_bounded_periodic_rhf_solver(k,n,input.electrons_per_cell,0,provider,o,caps.other_live_numerical_bytes);
    admit(memory,caps); limit(memory.maximum_work_units,caps.maximum_work_units,"bounded periodic RHF overlap preflight work cap exceeded");
    const I nn=mul(n,n),knn=mul(k,nn); view(input.overlap,input.overlap_elements,knn); view(input.hcore,input.hcore_elements,knn);
    finite(input.nuclear_repulsion_energy_per_cell);
    BoundedPeriodicRHFResult out; auto& d=out.diagnostics_;
    d.minimum_overlap_eigenvalue=std::numeric_limits<double>::infinity();
    d.minimum_retained_overlap_eigenvalue=std::numeric_limits<double>::infinity();
    I work=memory.input_validation_work_units;
    for (I point=0;point<k;++point) {
        audit(input.overlap+point*nn,n,o.hermitian_absolute_tolerance,o.hermitian_relative_tolerance,
            o.maximum_overlap_projection_error,d.maximum_raw_overlap_hermitian_defect,d.maximum_overlap_projection_frobenius);
        audit(input.hcore+point*nn,n,o.hermitian_absolute_tolerance,o.hermitian_relative_tolerance,
            o.maximum_hcore_projection_error,d.maximum_raw_hcore_hermitian_defect,d.maximum_hcore_projection_frobenius);
    }
    work=add(work,memory.overlap_discovery_work_units);
    std::vector<Z> matrix(nn),vectors(nn);
    std::vector<double> values(n);
    I r=0;
    for (I point=0;point<k;++point) {
        const I rank=overlap_eigen(input.overlap+point*nn,n,matrix.data(),vectors.data(),values.data(),o,d);
        if (rank<=memory.n_occupied) throw std::invalid_argument("bounded periodic RHF retained overlap rank lacks an occupied-plus-virtual space");
        if (point && rank!=r) throw std::invalid_argument("bounded periodic RHF overlap retained rank differs across k points");
        r=rank;
    }
    memory=plan_bounded_periodic_rhf_solver(k,n,input.electrons_per_cell,r,provider,o,caps.other_live_numerical_bytes);
    admit(memory,caps);
    const I preparation=add(memory.orthogonalizer_work_units,memory.initial_density_work_units);
    limit(add(work,add(preparation,memory.work_units_per_evaluated_iteration)),caps.maximum_work_units,
          "bounded periodic RHF initial complete evaluation work cap exceeded");
    out.memory_=memory;
    const I nr=mul(n,r),knr=mul(k,nr);
    std::vector<Z> x(knr),a(nr),b(nr);
    out.density_.resize(knn); out.fock_.resize(knn); out.coefficients_.resize(knr); out.energies_.resize(k*r);
    work=add(work,preparation);
    for (I point=0;point<k;++point) {
        const Z* s=input.overlap+point*nn; const Z* h=input.hcore+point*nn; Z* xp=x.data()+point*nr;
        const I rank=overlap_eigen(s,n,matrix.data(),vectors.data(),values.data(),o,d);
        if (rank!=r) throw std::logic_error("bounded periodic RHF overlap rank changed between immutable-input passes");
        for (I col=0;col<r;++col) {
            const double lambda=values[n-r+col];
            d.minimum_retained_overlap_eigenvalue=std::min(d.minimum_retained_overlap_eigenvalue,lambda);
            const double root=finite(std::sqrt(lambda));
            for (I row=0;row<n;++row) xp[row*r+col]=finite(vectors[row*n+n-r+col]/root);
        }
        apply(n,r,[s,n](I i,I j){return hermitian(s,n,i,j);},xp,a.data());
        left_product(n,r,xp,a.data(),matrix.data());
        for (I i=0;i<r;++i) for (I j=0;j<r;++j)
            check(matrix[i*r+j],double(i==j),o,d.maximum_orthogonalizer_metric_error,
                  "bounded periodic RHF canonical orthogonalizer failed original metric audit");
        diagonalize(n,r,[h,n](I i,I j){return hermitian(h,n,i,j);},xp,matrix.data(),vectors.data(),
            a.data(),b.data(),values.data(),out.coefficients_.data()+point*nr,out.energies_.data()+point*r,o,d);
        density_from_coefficients(n,r,memory.n_occupied,out.coefficients_.data()+point*nr,out.density_.data()+point*nn);
    }
    double previous_energy=0;
    for (I iteration=1;iteration<=o.maximum_iterations;++iteration) {
        work=add(work,memory.work_units_per_evaluated_iteration);
        limit(work,caps.maximum_work_units,"bounded periodic RHF iteration work reservation failed");
        std::fill(out.fock_.begin(),out.fock_.end(),Z(std::numeric_limits<double>::quiet_NaN(),0));
        provider.evaluate(out.density_.data(),out.density_.size(),out.fock_.data(),out.fock_.size(),provider.context);
        Snapshot snap; snap.evaluated_iteration=iteration; snap.fock_calls=iteration; snap.charged_work_units=work;
        snap.global_homo=-std::numeric_limits<double>::infinity(); snap.global_lumo=std::numeric_limits<double>::infinity();
        RealSum energy,raw_energy; double commutator=0; bool density_closed=true;
        const double weight=1.0/static_cast<double>(k);
        for (I point=0;point<k;++point) {
            const Z* s=input.overlap+point*nn; const Z* h=input.hcore+point*nn;
            Z* f=out.fock_.data()+point*nn; const Z* density=out.density_.data()+point*nn;
            for (I ij=0;ij<nn;++ij) f[ij]=finite(finite(f[ij])+h[ij]);
            for (I i=0;i<n;++i) for (I j=0;j<n;++j)
                raw_energy.push(finite(0.5*weight*finite(density[i*n+j]*(h[j*n+i]+f[j*n+i])).real()));
            audit(f,n,o.hermitian_absolute_tolerance,o.hermitian_relative_tolerance,o.maximum_fock_projection_error,
                d.maximum_raw_fock_hermitian_defect,d.maximum_fock_projection_frobenius);
            project_in_place(f,n);
            for (I i=0;i<n;++i) for (I j=0;j<n;++j)
                energy.push(finite(0.5*weight*finite(density[i*n+j]*(hermitian(h,n,j,i)+f[j*n+i])).real()));
            Z* c=out.coefficients_.data()+point*nr; double* eps=out.energies_.data()+point*r;
            snap.maximum_projected_eigen_relative_residual=std::max(snap.maximum_projected_eigen_relative_residual,
                diagonalize(n,r,[f,n](I i,I j){return f[i*n+j];},x.data()+point*nr,matrix.data(),vectors.data(),
                    a.data(),b.data(),values.data(),c,eps,o,d));
            density_audits(n,r,memory.n_occupied,s,f,density,x.data()+point*nr,c,eps,
                matrix.data(),vectors.data(),a.data(),b.data(),o,snap,commutator,density_closed);
            snap.global_homo=std::max(snap.global_homo,eps[memory.n_occupied-1]);
            snap.global_lumo=std::min(snap.global_lumo,eps[memory.n_occupied]);
        }
        snap.electronic_energy_per_cell=energy.get();
        snap.energy_per_cell=finite(snap.electronic_energy_per_cell+input.nuclear_repulsion_energy_per_cell);
        snap.raw_energy_per_cell=finite(raw_energy.get()+input.nuclear_repulsion_energy_per_cell);
        snap.energy_projection_change=finite(std::abs(snap.energy_per_cell-snap.raw_energy_per_cell));
        snap.commutator_frobenius_rms=finite(commutator/std::sqrt(static_cast<double>(k)));
        snap.global_band_gap=finite(snap.global_lumo-snap.global_homo);
        snap.has_energy_change=iteration>1;
        snap.energy_change=iteration>1?finite(snap.energy_per_cell-previous_energy):0.0;
        const bool stationary=snap.has_energy_change && std::abs(snap.energy_change)<=o.energy_change_tolerance
            && snap.commutator_frobenius_rms<=o.commutator_tolerance && density_closed;
        bool stop=false;
        if (stationary) {
            snap.converged=snap.global_band_gap>o.minimum_band_gap_hartree;
            snap.status=snap.converged?Status::Converged:Status::NonInsulating; stop=true;
        } else if (iteration==o.maximum_iterations) { snap.status=Status::IterationLimit; stop=true; }
        else if (iteration==caps.maximum_fock_calls) { snap.status=Status::FockCallLimit; stop=true; }
        else if (memory.work_units_per_evaluated_iteration>caps.maximum_work_units-work) { snap.status=Status::WorkLimit; stop=true; }
        if (progress && !progress(snap,progress_context)) { snap.status=Status::Cancelled; snap.converged=false; stop=true; }
        out.snapshot_=snap;
        if (stop) break;
        previous_energy=snap.energy_per_cell;
        // This next density is never returned without a complete new F[D]
        // evaluation, eigen/closure audit and same-D energy calculation.
        for (I point=0;point<k;++point)
            density_from_coefficients(n,r,memory.n_occupied,out.coefficients_.data()+point*nr,out.density_.data()+point*nn);
    }
    return out;
}
} // namespace vibeqc
