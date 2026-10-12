#include "vibeqc/manta_periodic_fock.hpp"
#include "vibeqc/manta_admission.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>

namespace py=pybind11;
using namespace vibeqc::manta;
namespace {
using Complex=std::complex<double>;
void check(const py::array& a,const py::dtype& dtype,int dimensions) {
    if(!a.dtype().is(dtype) || a.ndim()!=dimensions || !(a.flags()&(py::array::c_style|py::array::f_style)))
        throw py::value_error("MANTA periodic Fock requires contiguous float64 vectors/weights, int64 offsets and complex128 H/D arrays");
}
Count vectors(const py::array& a) {
    check(a,py::dtype::of<double>(),2);
    if(a.shape(1)!=3) throw py::value_error("MANTA periodic Fock vectors require shape (count,3)");
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
py::array_t<Complex> owned(const std::vector<Complex>& values,const std::vector<py::ssize_t>& shape) {
    py::array_t<Complex> out(shape);
    if(!values.empty()) std::memcpy(out.mutable_data(),values.data(),values.size()*sizeof(Complex));
    return out;
}
} // namespace
void bind_manta_periodic_fock(py::module_& m) {
    auto p=py::class_<PeriodicFockPlan>(m,"PeriodicFockPlan");
#define P(name) p.def_readonly(#name,&PeriodicFockPlan::name)
    P(hartree); P(exchange); P(n_basis); P(momenta); P(driver_live_bytes); P(modeled_peak_bytes);
    P(shell_calls); P(primitive_pairs); P(scalar_terms); P(kernel_terms);
#undef P
    auto r=py::class_<PeriodicFockReceipt>(m,"PeriodicFockReceipt");
#define R(name) r.def_readonly(#name,&PeriodicFockReceipt::name)
    R(provider); R(algorithm); R(convention); R(source_identity); R(density_identity); R(one_electron_identity); R(snapshot_identity);
    R(exchange_scale); R(one_electron_energy); R(coulomb_energy); R(exchange_energy); R(electronic_energy);
    R(hcore_hermiticity_residual); R(fock_hermiticity_residual); R(modeled_peak_bytes);
#undef R
    py::class_<PeriodicFockResult>(m,"PeriodicFockResult")
        .def_readonly("plan",&PeriodicFockResult::plan)
        .def_readonly("receipt",&PeriodicFockResult::receipt)
        .def_property_readonly("hartree_receipt",[](const PeriodicFockResult& r){return r.hartree.receipt;})
        .def_property_readonly("exchange_receipt",[](const PeriodicFockResult& r){return r.exchange.receipt;})
        .def_property_readonly("values",[](const PeriodicFockResult& r){
            const auto nk=static_cast<py::ssize_t>(r.plan.momenta),n=static_cast<py::ssize_t>(r.plan.n_basis);
            return owned(r.values,{2,nk,n,n});
        })
        .def_property_readonly("exchange_values",[](const PeriodicFockResult& r){
            const auto nk=static_cast<py::ssize_t>(r.plan.momenta),n=static_cast<py::ssize_t>(r.plan.n_basis);
            return owned(r.exchange.values,{2,nk,n,n});
        })
        .def_property_readonly("hartree_values",[](const PeriodicFockResult& r){
            const auto nk=static_cast<py::ssize_t>(r.plan.momenta),n=static_cast<py::ssize_t>(r.plan.n_basis);
            return owned(r.hartree.momenta,{nk,n,n});
        });
    m.def("plan_periodic_fock",&plan_periodic_fock,py::arg("basis"),py::arg("hartree_vectors"),py::arg("exchange_vectors_per_pair"),
        py::arg("cells"),py::arg("momenta"),py::arg("reciprocal_batch")=64,py::arg("pair_cutoff_bohr")=py::none(),
        py::arg("budget")=ResourceBudget{},py::arg("kernel")="coulomb",py::arg("omega")=0.,py::arg("exchange_scale")=1.);
    m.def("compute_periodic_fock",[](const vibeqc::BasisSet& basis,const py::array& lattice,
        const py::array& g,const py::array& q,const py::array& offsets,const py::array& cells,
        const py::array& k,const py::array& weights,const py::array& h,const py::array& d,
        Count batch,std::optional<double> cutoff,const ResourceBudget& budget,const std::string& kernel,double omega,double scale) {
        using namespace admission;
        if(vectors(lattice)!=3) throw py::value_error("MANTA periodic Fock lattice must have shape (3,3)");
        const Count ng=vectors(g),nq=vectors(q),nc=vectors(cells),nk=vectors(k),n=basis.nbasis(),pairs=square(nk);
        check(offsets,py::dtype::of<std::int64_t>(),1); check(weights,py::dtype::of<double>(),1);
        check(h,py::dtype::of<Complex>(),3); check(d,py::dtype::of<Complex>(),4);
        if(static_cast<Count>(offsets.shape(0))!=add(pairs,1) || weights.shape(0)!=static_cast<py::ssize_t>(nk)
           || h.shape(0)!=static_cast<py::ssize_t>(nk) || h.shape(1)!=static_cast<py::ssize_t>(n) || h.shape(2)!=static_cast<py::ssize_t>(n)
           || d.shape(0)!=2 || d.shape(1)!=static_cast<py::ssize_t>(nk) || d.shape(2)!=static_cast<py::ssize_t>(n) || d.shape(3)!=static_cast<py::ssize_t>(n))
            throw py::value_error("MANTA periodic Fock requires H[k,AO,AO] and D[spin=2,k,AO,AO] with matching offsets/weights");
        plan_periodic_fock(basis,ng,0,nc,nk,batch,cutoff,budget,kernel,omega,scale);
        Count maximum=0,previous=0;
        if(item<std::int64_t>(offsets,0)!=0) throw py::value_error("MANTA periodic Fock offsets must start at zero");
        for(Count i=1;i<=pairs;++i) {
            const auto next=item<std::int64_t>(offsets,i*offsets.strides(0));
            if(next<0 || static_cast<Count>(next)<previous || static_cast<Count>(next)>nq)
                throw py::value_error("MANTA periodic Fock offsets must be monotone and within the vector extent");
            maximum=std::max(maximum,static_cast<Count>(next)-previous); previous=next;
        }
        if(previous!=nq) throw py::value_error("MANTA periodic Fock offsets must end at the vector count");
        plan_periodic_fock(basis,ng,maximum,nc,nk,batch,cutoff,budget,kernel,omega,scale);
        const auto a=copy(lattice),coulomb=copy(g),exchange=copy(q),translations=copy(cells),momenta=copy(k);
        std::vector<Count> o(pairs+1); std::vector<double> w(nk);
        const Count elements=mul(nk,square(n)); std::vector<Complex> hcore(elements),density(mul(2,elements));
        for(Count i=0;i<=pairs;++i) o[i]=item<std::int64_t>(offsets,i*offsets.strides(0));
        for(Count ik=0;ik<nk;++ik) {
            w[ik]=item<double>(weights,ik*weights.strides(0));
            for(Count i=0;i<n;++i) for(Count j=0;j<n;++j) {
                hcore[(ik*n+i)*n+j]=item<Complex>(h,ik*h.strides(0)+i*h.strides(1)+j*h.strides(2));
                for(Count spin=0;spin<2;++spin)
                    density[((spin*nk+ik)*n+i)*n+j]=item<Complex>(d,spin*d.strides(0)+ik*d.strides(1)+i*d.strides(2)+j*d.strides(3));
            }
        }
        return compute_periodic_fock(basis,a,coulomb,exchange,o,translations,momenta,w,hcore,density,batch,cutoff,budget,kernel,omega,scale);
    },py::arg("basis"),py::arg("lattice").noconvert(),py::arg("hartree_vectors").noconvert(),py::arg("exchange_vectors").noconvert(),
      py::arg("transfer_offsets").noconvert(),py::arg("translations").noconvert(),py::arg("momenta").noconvert(),
      py::arg("weights").noconvert(),py::arg("hcore").noconvert(),py::arg("spin_densities").noconvert(),
      py::arg("reciprocal_batch")=64,py::arg("pair_cutoff_bohr")=py::none(),py::arg("budget")=ResourceBudget{},
      py::arg("kernel")="coulomb",py::arg("omega")=0.,py::arg("exchange_scale")=1.);
}
