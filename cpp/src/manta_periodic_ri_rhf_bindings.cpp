#include "vibeqc/manta_periodic_ri_rhf.hpp"
#include "vibeqc/manta_periodic_ri_domains.hpp"
#include "vibeqc/manta_admission.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>

namespace py=pybind11;
using namespace vibeqc::manta;
namespace {
using Complex=std::complex<double>;
Count vectors(const py::array& a) {
    if(!a.dtype().is(py::dtype::of<double>()) || a.ndim()!=2 || a.shape(1)!=3
       || !(a.flags()&(py::array::c_style|py::array::f_style)))
        throw py::value_error("MANTA periodic RI RHF vectors must be contiguous float64 (count,3)");
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
template<class T> py::array_t<T> owned(const T* values,Count count,std::vector<py::ssize_t> shape) {
    py::array_t<T> out(shape); if(count) std::memcpy(out.mutable_data(),values,count*sizeof(T)); return out;
}
} // namespace
void bind_manta_periodic_ri_rhf(py::module_& m) {
    py::class_<PeriodicRIRHFOptions,PeriodicRHFOptions>(m,"PeriodicRIRHFOptions").def(py::init<>())
        .def_readwrite("fitting",&PeriodicRIRHFOptions::fitting).def_readwrite("ri",&PeriodicRIRHFOptions::ri)
        .def_readwrite("ri_accuracy",&PeriodicRIRHFOptions::ri_accuracy)
        .def_readwrite("coulomb_algorithm",&PeriodicRIRHFOptions::coulomb_algorithm)
        .def_readwrite("exchange_algorithm",&PeriodicRIRHFOptions::exchange_algorithm);
    auto p=py::class_<PeriodicRIRHFPlan>(m,"PeriodicRIRHFPlan");
#define P(name) p.def_readonly(#name,&PeriodicRIRHFPlan::name)
    P(overlap); P(kinetic); P(nuclear); P(fock); P(solver); P(density_envelope); P(n_basis); P(momenta); P(electrons);
    P(driver_live_bytes); P(modeled_peak_bytes); P(fock_calls_upper_bound); P(shell_calls); P(primitive_pairs);
    P(scalar_terms); P(kernel_terms); P(nuclear_image_candidates_upper_bound);
    P(madelung); P(exchange_correction_kernel_terms_per_call);
    P(domain_live_bytes); P(domain_scalar_terms); P(domain_kernel_terms);
    P(history_capacity); P(history_live_bytes); P(exact_hartree); P(exact_exchange);
    P(mixed_driver_live_bytes); P(two_electron_peak_bytes); P(two_electron_kernel_terms_per_call);
#undef P
    py::class_<PeriodicRIRHFReceipt,PeriodicRHFReceipt>(m,"PeriodicRIRHFReceipt")
        .def_readonly("accuracy_density",&PeriodicRIRHFReceipt::accuracy_density)
        .def_readonly("fitting_info",&PeriodicRIRHFReceipt::fitting_info)
        .def_readonly("fitted_fock",&PeriodicRIRHFReceipt::fitted_fock)
        .def_readonly("coulomb_algorithm",&PeriodicRIRHFReceipt::coulomb_algorithm)
        .def_readonly("exchange_algorithm",&PeriodicRIRHFReceipt::exchange_algorithm)
        .def_readonly("coulomb_identity",&PeriodicRIRHFReceipt::coulomb_identity)
        .def_readonly("exchange_identity",&PeriodicRIRHFReceipt::exchange_identity)
        .def_readonly("exact_hartree",&PeriodicRIRHFReceipt::exact_hartree)
        .def_readonly("exact_exchange",&PeriodicRIRHFReceipt::exact_exchange);
    py::class_<PeriodicRIRHFResult>(m,"PeriodicRIRHFResult")
        .def_readonly("plan",&PeriodicRIRHFResult::plan).def_readonly("receipt",&PeriodicRIRHFResult::receipt)
        .def_property_readonly("converged",&PeriodicRIRHFResult::converged)
        .def_property_readonly("exchange_q0",[](const PeriodicRIRHFResult& r){return r.receipt.exchange_q0;})
        .def_property_readonly("solver_snapshot",[](const PeriodicRIRHFResult& r){return r.numerical->final_snapshot();})
        .def_property_readonly("solver_diagnostics",[](const PeriodicRIRHFResult& r){return r.numerical->diagnostics();})
        .def_property_readonly("solver_memory",[](const PeriodicRIRHFResult& r){return r.numerical->memory();})
        .def_property_readonly("scf_history",[](const PeriodicRIRHFResult& r){return r.scf_history;})
        .def_property_readonly("momenta",[](const PeriodicRIRHFResult& r){
            py::array_t<double> out({r.momenta.rows(),Eigen::Index{3}});
            for(Eigen::Index i=0;i<r.momenta.rows();++i)for(Eigen::Index j=0;j<3;++j)
                out.mutable_at(i,j)=r.momenta(i,j);
            return out;
        })
        .def_property_readonly("density",[](const PeriodicRIRHFResult& r){
            const auto& p=r.numerical->memory(); return owned(r.numerical->density_data(),p.n_kpoints*p.n_basis*p.n_basis,
                {static_cast<py::ssize_t>(p.n_kpoints),static_cast<py::ssize_t>(p.n_basis),static_cast<py::ssize_t>(p.n_basis)});
        })
        .def_property_readonly("fock",[](const PeriodicRIRHFResult& r){
            const auto& p=r.numerical->memory(); return owned(r.numerical->fock_data(),p.n_kpoints*p.n_basis*p.n_basis,
                {static_cast<py::ssize_t>(p.n_kpoints),static_cast<py::ssize_t>(p.n_basis),static_cast<py::ssize_t>(p.n_basis)});
        })
        .def_property_readonly("coefficients",[](const PeriodicRIRHFResult& r){
            const auto& p=r.numerical->memory(); return owned(r.numerical->coefficients_data(),p.n_kpoints*p.n_basis*p.retained_rank,
                {static_cast<py::ssize_t>(p.n_kpoints),static_cast<py::ssize_t>(p.n_basis),static_cast<py::ssize_t>(p.retained_rank)});
        })
        .def_property_readonly("orbital_energies",[](const PeriodicRIRHFResult& r){
            const auto& p=r.numerical->memory(); return owned(r.numerical->orbital_energies_data(),p.n_kpoints*p.retained_rank,
                {static_cast<py::ssize_t>(p.n_kpoints),static_cast<py::ssize_t>(p.retained_rank)});
        })
        .def_property_readonly("overlap",[](const PeriodicRIRHFResult& r){const auto& p=r.plan; return owned(r.overlap.data(),r.overlap.size(),
                {static_cast<py::ssize_t>(p.momenta),static_cast<py::ssize_t>(p.n_basis),static_cast<py::ssize_t>(p.n_basis)});})
        .def_property_readonly("hcore",[](const PeriodicRIRHFResult& r){const auto& p=r.plan; return owned(r.hcore.data(),r.hcore.size(),
                {static_cast<py::ssize_t>(p.momenta),static_cast<py::ssize_t>(p.n_basis),static_cast<py::ssize_t>(p.n_basis)});});
    m.def("plan_periodic_ri_rhf",&plan_periodic_ri_rhf,py::arg("basis"),py::arg("auxiliary"),py::arg("system"),py::arg("mesh"),
        py::arg("reciprocal_vectors"),py::arg("exchange_vectors_per_pair"),py::arg("cells"),
        py::arg("options")=PeriodicRIRHFOptions{},py::arg("budget")=ResourceBudget{});
    m.def("run_periodic_ri_rhf",[](const vibeqc::BasisSet& basis,const vibeqc::BasisSet& auxiliary,const vibeqc::PeriodicSystem& system,const vibeqc::RegularKMesh& mesh,
        const py::array& g,const py::array& q,const py::array& offsets,const py::array& cells,PeriodicRIRHFOptions o,ResourceBudget budget) {
        using namespace admission;
        const Count ng=vectors(g),nq=vectors(q),nc=vectors(cells),pairs=square(mesh.size());
        if(!offsets.dtype().is(py::dtype::of<std::int64_t>()) || offsets.ndim()!=1 || static_cast<Count>(offsets.shape(0))!=add(pairs,1)
           || !(offsets.flags()&(py::array::c_style|py::array::f_style)))
            throw py::value_error("MANTA periodic RI RHF offsets require contiguous int64 (nk*nk+1)");
        plan_periodic_ri_rhf(basis,auxiliary,system,mesh,ng,o.exchange_algorithm=="exact" || mesh.size()>1?1:0,nc,o,budget);
        Count maximum=0,previous=0;
        if(item<std::int64_t>(offsets,0)!=0) throw py::value_error("MANTA periodic RI RHF offsets must start at zero");
        for(Count i=1;i<=pairs;++i) {
            const auto next=item<std::int64_t>(offsets,i*offsets.strides(0));
            if(next<0 || static_cast<Count>(next)<previous || static_cast<Count>(next)>nq)
                throw py::value_error("MANTA periodic RI RHF offsets must be monotone and bounded");
            const Count length=static_cast<Count>(next)-previous,pair=i-1,nk=mesh.size();
            if(o.exchange_algorithm=="ri" ? ((pair/nk==pair%nk && length) || (pair/nk!=pair%nk && !length)) : (pair/nk==pair%nk && !length))
                throw py::value_error("MANTA periodic RI RHF pair segments must match the selected exchange source");
            maximum=std::max(maximum,length); previous=next;
        }
        if(previous!=nq) throw py::value_error("MANTA periodic RI RHF offsets must end at the vector count");
        plan_periodic_ri_rhf(basis,auxiliary,system,mesh,ng,maximum,nc,o,budget);
        const auto reciprocal=copy(g),exchange=copy(q),translations=copy(cells);
        std::vector<Count> boundaries(pairs+1);
        for(Count i=0;i<=pairs;++i) boundaries[i]=item<std::int64_t>(offsets,i*offsets.strides(0));
        return run_periodic_ri_rhf(basis,auxiliary,system,mesh,reciprocal,exchange,boundaries,translations,o,budget);
    },py::arg("basis"),py::arg("auxiliary"),py::arg("system"),py::arg("mesh"),py::arg("reciprocal_vectors").noconvert(),
      py::arg("pair_vectors").noconvert(),py::arg("transfer_offsets").noconvert(),py::arg("translations").noconvert(),
      py::arg("options")=PeriodicRIRHFOptions{},py::arg("budget")=ResourceBudget{});
    auto dp=py::class_<PeriodicRIRHFDomainPlan>(m,"PeriodicRIRHFDomainPlan");
#define D(name) dp.def_readonly(#name,&PeriodicRIRHFDomainPlan::name)
    D(rhf); D(reciprocal_energy_cutoff); D(pair_cutoff_bohr); D(translation_radius_bohr); D(momenta);
    D(translation_candidates); D(hartree_candidates); D(maximum_exchange_candidates); D(exchange_candidates); D(candidate_visits);
    D(domain_live_bytes); D(domain_scalar_terms); D(domain_kernel_terms);
    D(modeled_peak_bytes); D(shell_calls); D(primitive_pairs); D(scalar_terms); D(kernel_terms);
#undef D
    auto domains=py::class_<PeriodicRIRHFDomains>(m,"PeriodicRIRHFDomains");
    domains.def_readonly("plan",&PeriodicRIRHFDomains::plan).def_readonly("receipt",&PeriodicRIRHFDomains::receipt);
    auto matrix=[](const md::Matrix& a) {
        py::array_t<double> out({static_cast<py::ssize_t>(a.rows()),py::ssize_t{3}});
        for(Eigen::Index i=0;i<a.rows();++i) for(int j=0;j<3;++j) out.mutable_data()[3*i+j]=a(i,j);
        return out;
    };
    domains.def_property_readonly("translations",[matrix](const PeriodicRIRHFDomains& d){return matrix(d.translations);})
        .def_property_readonly("reciprocal_vectors",[matrix](const PeriodicRIRHFDomains& d){return matrix(d.reciprocal_vectors);})
        .def_property_readonly("pair_vectors",[matrix](const PeriodicRIRHFDomains& d){return matrix(d.exchange_vectors);})
        .def_property_readonly("transfer_offsets",[](const PeriodicRIRHFDomains& d){
            py::array_t<std::int64_t> out(d.transfer_offsets.size());
            for(Count i=0;i<d.transfer_offsets.size();++i) out.mutable_data()[i]=d.transfer_offsets[i];
            return out;
        });
    m.def("plan_periodic_ri_rhf_from_cutoffs",&plan_periodic_ri_rhf_from_cutoffs,py::arg("basis"),py::arg("auxiliary"),
        py::arg("system"),py::arg("mesh"),py::arg("reciprocal_energy_cutoff"),py::arg("options"),py::arg("budget")=ResourceBudget{});
    m.def("build_periodic_ri_rhf_domains",&build_periodic_ri_rhf_domains,py::arg("basis"),py::arg("auxiliary"),
        py::arg("system"),py::arg("mesh"),py::arg("reciprocal_energy_cutoff"),py::arg("options"),py::arg("budget")=ResourceBudget{});
    m.def("run_periodic_ri_rhf_from_cutoffs",&run_periodic_ri_rhf_from_cutoffs,py::arg("basis"),py::arg("auxiliary"),
        py::arg("system"),py::arg("mesh"),py::arg("reciprocal_energy_cutoff"),py::arg("options"),py::arg("budget")=ResourceBudget{});
}
