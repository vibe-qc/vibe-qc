#pragma once
// Private numerical helpers shared by the bounded RHF and UHF leaves.
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
void density_from_coefficients(I n,I r,I occupied,const Z* c,Z* d,double occupation=2.0) {
    for (I i=0;i<n;++i) for (I j=i;j<n;++j) {
        Sum sum; for (I a=0;a<occupied;++a) sum.push(c[i*r+a]*std::conj(c[j*r+a]));
        Z value=finite(occupation*sum.get());
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
                    const Options& o,Snapshot& snap,double& commutator_norm,bool& closed,double occupation=2.0) {
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
        const Z dc=finite(occupation*value.get());
        closure=norm_add(closure,abs(d[i*n+j]-dc)); dnorm=norm_add(dnorm,abs(d[i*n+j])); cnorm=norm_add(cnorm,abs(dc));
        electron.push(d[i*n+j]*se(j,i));
    }
    const double scale=std::max({1.0,dnorm,cnorm});
    snap.maximum_density_closure_frobenius=std::max(snap.maximum_density_closure_frobenius,closure);
    snap.maximum_density_closure_relative=std::max(snap.maximum_density_closure_relative,finite(closure/scale));
    closed &= closure<=boundary(o.density_closure_absolute_tolerance,o.density_closure_relative_tolerance,scale);
    check(electron.get(),occupation*static_cast<double>(occupied),o,snap.maximum_electron_count_error,
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
        check(pp.get(),occupation*m[i*r+j],o,snap.maximum_metric_idempotency_error,
              "bounded periodic RHF physical density failed metric idempotency");
    }
    commutator_norm=norm_add(commutator_norm,point_norm);
}
void admit(const Memory& m,const BoundedPeriodicRHFCaps& c) {
    limit(m.peak_owned_numerical_bytes,c.maximum_owned_numerical_bytes,"bounded periodic RHF owned numerical byte cap exceeded");
    limit(m.total_numerical_bytes,c.maximum_total_numerical_bytes,"bounded periodic RHF total numerical byte cap exceeded");
}
} // namespace

} // namespace vibeqc
