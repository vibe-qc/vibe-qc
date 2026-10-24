#include "vibeqc/manta_periodic_exchange.hpp"
#include "vibeqc/manta_admission.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>

namespace py=pybind11;
using namespace vibeqc::manta;
namespace {
void check(const py::array& a,const py::dtype& dtype,int dimensions) {
    if(!a.dtype().is(dtype) || a.ndim()!=dimensions)
        throw py::value_error("MANTA periodic exchange requires float64 vectors/weights, int64 offsets and complex128 densities with the documented shapes");
    if(!(a.flags() & (py::array::c_style | py::array::f_style)))
        throw py::value_error("MANTA periodic exchange inputs must be C- or F-contiguous");
}
Count vectors(const py::array& a) {
    check(a,py::dtype::of<double>(),2);
    if(a.shape(1)!=3) throw py::value_error("MANTA periodic exchange vectors must have shape (count,3)");
    return a.shape(0);
}
template<class T> T item(const py::array& a,py::ssize_t offset) {
    T out; std::memcpy(&out,static_cast<const char*>(a.data())+offset,sizeof(T)); return out;
}
md::Matrix copy(const py::array& a) {
    md::Matrix out(a.shape(0),a.shape(1));
    for(Eigen::Index i=0;i<out.rows();++i) for(Eigen::Index j=0;j<out.cols();++j)
        out(i,j)=item<double>(a,i*a.strides(0)+j*a.strides(1));
    return out;
}
} // namespace
void bind_manta_periodic_exchange(py::module_& m) {
    auto p=py::class_<PeriodicExchangePlan>(m,"PeriodicExchangePlan");
#define P(name) p.def_readonly(#name,&PeriodicExchangePlan::name)
    P(roundoff_requested);P(n_basis); P(momenta); P(densities); P(cells); P(vectors_per_pair); P(transfer_pairs); P(reciprocal_batch);
    P(shell_calls); P(primitive_pairs); P(vector_primitive_products); P(phase_products); P(phase_accumulations);
    P(matrix_products); P(contraction_terms); P(scalar_terms); P(kernel_terms); P(output_bytes); P(driver_live_bytes); P(modeled_peak_bytes);
#undef P
    auto r=py::class_<PeriodicExchangeReceipt>(m,"PeriodicExchangeReceipt");
#define R(name) r.def_readonly(#name,&PeriodicExchangeReceipt::name)
    R(roundoff_evaluated);R(roundoff_scope);R(roundoff_bound);R(energy_roundoff_bounds);R(provider); R(provider_version); R(normalization); R(algorithm); R(boundary); R(convention);
    R(kernel); R(omega); R(zero_vectors);
    R(source_identity); R(density_identity); R(density_identities); R(energies); R(transfer_vectors);
    R(panel_calls); R(shell_calls); R(skipped_shell_pairs); R(primitive_pairs); R(vector_primitive_products);
    R(phase_products); R(phase_accumulations); R(matrix_products); R(contraction_terms); R(modeled_peak_bytes);
    R(cell_volume); R(density_hermiticity_residual); R(response_hermiticity_residual);
#undef R
    py::class_<PeriodicExchangeResult>(m,"PeriodicExchangeResult")
        .def_readonly("plan",&PeriodicExchangeResult::plan)
        .def_readonly("receipt",&PeriodicExchangeResult::receipt)
        .def_property_readonly("roundoff_bounds",[](const PeriodicExchangeResult& r) {
            py::array_t<double> a({static_cast<py::ssize_t>(r.receipt.roundoff_evaluated?r.plan.densities:0),static_cast<py::ssize_t>(r.plan.momenta),
                static_cast<py::ssize_t>(r.plan.n_basis),static_cast<py::ssize_t>(r.plan.n_basis)});
            if(!r.roundoff_bounds.empty())std::memcpy(a.mutable_data(),r.roundoff_bounds.data(),r.roundoff_bounds.size()*sizeof(double));return a;
        })
        .def_property_readonly("values",[](const PeriodicExchangeResult& r) {
            py::array_t<std::complex<double>> a({static_cast<py::ssize_t>(r.plan.densities),static_cast<py::ssize_t>(r.plan.momenta),
                static_cast<py::ssize_t>(r.plan.n_basis),static_cast<py::ssize_t>(r.plan.n_basis)});
            if(!r.values.empty()) std::memcpy(a.mutable_data(),r.values.data(),r.values.size()*sizeof(std::complex<double>));
            return a;
        });
    m.def("plan_periodic_exchange",&plan_periodic_exchange,py::arg("basis"),py::arg("vectors_per_pair"),py::arg("cells"),
        py::arg("momenta"),py::arg("densities"),py::arg("reciprocal_batch")=64,
        py::arg("pair_cutoff_bohr")=py::none(),py::arg("budget")=ResourceBudget{},py::arg("kernel")="coulomb",py::arg("omega")=0.,py::arg("audit_roundoff")=false);
    m.def("compute_periodic_exchange",[](const vibeqc::BasisSet& basis,const py::array& lattice,const py::array& q,
        const py::array& offsets,const py::array& cells,const py::array& k,const py::array& weights,const py::array& d,
        Count batch,std::optional<double> cutoff,const ResourceBudget& budget,const std::string& kernel,double omega,bool audit_roundoff) {
        using namespace admission;
        if(vectors(lattice)!=3) throw py::value_error("MANTA periodic exchange lattice must have shape (3,3)");
        const Count nq=vectors(q),nc=vectors(cells),nk=vectors(k),n=basis.nbasis(),pairs=square(nk);
        check(offsets,py::dtype::of<std::int64_t>(),1); check(weights,py::dtype::of<double>(),1);
        check(d,py::dtype::of<std::complex<double>>(),4);
        if(static_cast<Count>(offsets.shape(0))!=add(pairs,1) || weights.shape(0)!=static_cast<py::ssize_t>(nk)
           || d.shape(1)!=static_cast<py::ssize_t>(nk) || d.shape(2)!=static_cast<py::ssize_t>(n)
           || d.shape(3)!=static_cast<py::ssize_t>(n))
            throw py::value_error("MANTA periodic exchange offsets/weights/densities have incompatible shapes");
        const Count nd=d.shape(0);
        // Admit the pair-metadata scan and fixed-shape owners before scanning
        // offsets. Only counts are read; numerical arrays are copied after
        // the complete maximum-channel work and memory plan is admitted.
        plan_periodic_exchange(basis,0,nc,nk,nd,batch,cutoff,budget,kernel,omega,audit_roundoff);
        Count maximum=0,previous=0;
        if(item<std::int64_t>(offsets,0)!=0) throw py::value_error("MANTA periodic exchange offsets must start at zero");
        for(Count i=1;i<=pairs;++i) {
            const auto next=item<std::int64_t>(offsets,i*offsets.strides(0));
            if(next<0 || static_cast<Count>(next)<previous || static_cast<Count>(next)>nq)
                throw py::value_error("MANTA periodic exchange offsets must be monotone and bounded by the vector count");
            maximum=std::max(maximum,static_cast<Count>(next)-previous); previous=next;
        }
        if(previous!=nq) throw py::value_error("MANTA periodic exchange offsets must end at the vector count");
        plan_periodic_exchange(basis,maximum,nc,nk,nd,batch,cutoff,budget,kernel,omega,audit_roundoff);
        const auto a=copy(lattice),wave=copy(q),images=copy(cells),momenta=copy(k);
        std::vector<Count> o(pairs+1); std::vector<double> w(nk);
        std::vector<std::complex<double>> density(nd*nk*n*n);
        for(Count i=0;i<=pairs;++i) o[i]=item<std::int64_t>(offsets,i*offsets.strides(0));
        for(Count ik=0;ik<nk;++ik) w[ik]=item<double>(weights,ik*weights.strides(0));
        for(Count state=0;state<nd;++state) for(Count ik=0;ik<nk;++ik) for(Count i=0;i<n;++i) for(Count j=0;j<n;++j)
            density[((state*nk+ik)*n+i)*n+j]=item<std::complex<double>>(d,state*d.strides(0)+ik*d.strides(1)+i*d.strides(2)+j*d.strides(3));
        return compute_periodic_exchange(basis,a,wave,o,images,momenta,w,density,nd,batch,cutoff,budget,kernel,omega,audit_roundoff);
    },py::arg("basis"),py::arg("lattice").noconvert(),py::arg("wave_vectors").noconvert(),py::arg("transfer_offsets").noconvert(),
      py::arg("translations").noconvert(),py::arg("momenta").noconvert(),py::arg("weights").noconvert(),py::arg("densities").noconvert(),
      py::arg("reciprocal_batch")=64,py::arg("pair_cutoff_bohr")=py::none(),py::arg("budget")=ResourceBudget{},
      py::arg("kernel")="coulomb",py::arg("omega")=0.,py::arg("audit_roundoff")=false);
}
