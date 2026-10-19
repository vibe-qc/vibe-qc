#include "vibeqc/manta_periodic_ri_uhf.hpp"
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
        throw py::value_error("MANTA periodic RI UHF vectors must be contiguous float64 (count,3)");
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
void bind_manta_periodic_ri_uhf(py::module_& m) {
    py::class_<PeriodicRIUHFOptions,PeriodicUHFOptions>(m,"PeriodicRIUHFOptions").def(py::init<>())
        .def_readwrite("fitting",&PeriodicRIUHFOptions::fitting).def_readwrite("ri",&PeriodicRIUHFOptions::ri)
        .def_readwrite("ri_accuracy",&PeriodicRIUHFOptions::ri_accuracy)
        .def_readwrite("one_electron_accuracy",&PeriodicRIUHFOptions::one_electron_accuracy)
        .def_readwrite("exchange_accuracy",&PeriodicRIUHFOptions::exchange_accuracy)
        .def_readwrite("coulomb_algorithm",&PeriodicRIUHFOptions::coulomb_algorithm)
        .def_readwrite("exchange_algorithm",&PeriodicRIUHFOptions::exchange_algorithm);
    auto p=py::class_<PeriodicRIUHFPlan>(m,"PeriodicRIUHFPlan");
#define P(name) p.def_readonly(#name,&PeriodicRIUHFPlan::name)
    P(overlap); P(kinetic); P(nuclear); P(fock); P(solver); P(density_envelope); P(one_electron_accuracy); P(exchange_accuracy); P(n_basis); P(momenta); P(electrons); P(n_alpha); P(n_beta);
    P(driver_live_bytes); P(modeled_peak_bytes); P(fock_calls_upper_bound); P(shell_calls); P(primitive_pairs);
    P(scalar_terms); P(kernel_terms); P(nuclear_image_candidates_upper_bound);
    P(madelung); P(exchange_correction_kernel_terms_per_call);
    P(domain_live_bytes); P(domain_scalar_terms); P(domain_kernel_terms);
    P(history_capacity); P(history_live_bytes); P(exact_hartree); P(exact_exchange);
    P(mixed_driver_live_bytes); P(two_electron_peak_bytes); P(two_electron_kernel_terms_per_call);
#undef P
    py::class_<PeriodicRIUHFReceipt,PeriodicUHFReceipt>(m,"PeriodicRIUHFReceipt")
        .def_readonly("accuracy_density",&PeriodicRIUHFReceipt::accuracy_density)
        .def_readonly("one_electron_accuracy",&PeriodicRIUHFReceipt::one_electron_accuracy)
        .def_readonly("exchange_accuracy",&PeriodicRIUHFReceipt::exchange_accuracy)
        .def_readonly("fitting_info",&PeriodicRIUHFReceipt::fitting_info)
        .def_readonly("fitted_fock",&PeriodicRIUHFReceipt::fitted_fock)
        .def_readonly("coulomb_algorithm",&PeriodicRIUHFReceipt::coulomb_algorithm)
        .def_readonly("exchange_algorithm",&PeriodicRIUHFReceipt::exchange_algorithm)
        .def_readonly("coulomb_identity",&PeriodicRIUHFReceipt::coulomb_identity)
        .def_readonly("exchange_identity",&PeriodicRIUHFReceipt::exchange_identity)
        .def_readonly("exact_hartree",&PeriodicRIUHFReceipt::exact_hartree)
        .def_readonly("exact_exchange",&PeriodicRIUHFReceipt::exact_exchange);
    py::class_<PeriodicRIUHFResult>(m,"PeriodicRIUHFResult")
        .def_readonly("plan",&PeriodicRIUHFResult::plan).def_readonly("receipt",&PeriodicRIUHFResult::receipt)
        .def_property_readonly("converged",&PeriodicRIUHFResult::converged)
        .def_property_readonly("exchange_q0",[](const PeriodicRIUHFResult& r){return r.receipt.exchange_q0;})
        .def_property_readonly("solver_snapshot",[](const PeriodicRIUHFResult& r){return r.numerical->final_snapshot();})
        .def_property_readonly("solver_diagnostics",[](const PeriodicRIUHFResult& r){return r.numerical->diagnostics();})
        .def_property_readonly("solver_memory",[](const PeriodicRIUHFResult& r){return r.numerical->memory();})
        .def_property_readonly("scf_history",[](const PeriodicRIUHFResult& r){return r.scf_history;})
        .def_property_readonly("momenta",[](const PeriodicRIUHFResult& r){
            py::array_t<double> out({r.momenta.rows(),Eigen::Index{3}});
            for(Eigen::Index i=0;i<r.momenta.rows();++i)for(Eigen::Index j=0;j<3;++j)
                out.mutable_at(i,j)=r.momenta(i,j);
            return out;
        })
        .def_property_readonly("density",[](const PeriodicRIUHFResult& r){
            const auto& p=r.numerical->memory(); return owned(r.numerical->density_data(),2*p.n_kpoints*p.n_basis*p.n_basis,
                {py::ssize_t{2},static_cast<py::ssize_t>(p.n_kpoints),static_cast<py::ssize_t>(p.n_basis),static_cast<py::ssize_t>(p.n_basis)});
        })
        .def_property_readonly("fock",[](const PeriodicRIUHFResult& r){
            const auto& p=r.numerical->memory(); return owned(r.numerical->fock_data(),2*p.n_kpoints*p.n_basis*p.n_basis,
                {py::ssize_t{2},static_cast<py::ssize_t>(p.n_kpoints),static_cast<py::ssize_t>(p.n_basis),static_cast<py::ssize_t>(p.n_basis)});
        })
        .def_property_readonly("coefficients",[](const PeriodicRIUHFResult& r){
            const auto& p=r.numerical->memory(); return owned(r.numerical->coefficients_data(),2*p.n_kpoints*p.n_basis*p.retained_rank,
                {py::ssize_t{2},static_cast<py::ssize_t>(p.n_kpoints),static_cast<py::ssize_t>(p.n_basis),static_cast<py::ssize_t>(p.retained_rank)});
        })
        .def_property_readonly("orbital_energies",[](const PeriodicRIUHFResult& r){
            const auto& p=r.numerical->memory(); return owned(r.numerical->orbital_energies_data(),2*p.n_kpoints*p.retained_rank,
                {py::ssize_t{2},static_cast<py::ssize_t>(p.n_kpoints),static_cast<py::ssize_t>(p.retained_rank)});
        })
        .def_property_readonly("overlap",[](const PeriodicRIUHFResult& r){const auto& p=r.plan; return owned(r.overlap.data(),r.overlap.size(),
                {static_cast<py::ssize_t>(p.momenta),static_cast<py::ssize_t>(p.n_basis),static_cast<py::ssize_t>(p.n_basis)});})
        .def_property_readonly("hcore",[](const PeriodicRIUHFResult& r){const auto& p=r.plan; return owned(r.hcore.data(),r.hcore.size(),
                {static_cast<py::ssize_t>(p.momenta),static_cast<py::ssize_t>(p.n_basis),static_cast<py::ssize_t>(p.n_basis)});});
    m.def("plan_periodic_ri_uhf",&plan_periodic_ri_uhf,py::arg("basis"),py::arg("auxiliary"),py::arg("system"),py::arg("mesh"),
        py::arg("reciprocal_vectors"),py::arg("exchange_vectors_per_pair"),py::arg("cells"),
        py::arg("options")=PeriodicRIUHFOptions{},py::arg("budget")=ResourceBudget{});
    m.def("run_periodic_ri_uhf",[](const vibeqc::BasisSet& basis,const vibeqc::BasisSet& auxiliary,const vibeqc::PeriodicSystem& system,const vibeqc::RegularKMesh& mesh,
        const py::array& g,const py::array& q,const py::array& offsets,const py::array& cells,PeriodicRIUHFOptions o,ResourceBudget budget) {
        using namespace admission;
        const Count ng=vectors(g),nq=vectors(q),nc=vectors(cells),pairs=square(mesh.size());
        if(!offsets.dtype().is(py::dtype::of<std::int64_t>()) || offsets.ndim()!=1 || static_cast<Count>(offsets.shape(0))!=add(pairs,1)
           || !(offsets.flags()&(py::array::c_style|py::array::f_style)))
            throw py::value_error("MANTA periodic RI UHF offsets require contiguous int64 (nk*nk+1)");
        plan_periodic_ri_uhf(basis,auxiliary,system,mesh,ng,o.exchange_algorithm=="exact" || mesh.size()>1?1:0,nc,o,budget);
        Count maximum=0,previous=0;
        if(item<std::int64_t>(offsets,0)!=0) throw py::value_error("MANTA periodic RI UHF offsets must start at zero");
        for(Count i=1;i<=pairs;++i) {
            const auto next=item<std::int64_t>(offsets,i*offsets.strides(0));
            if(next<0 || static_cast<Count>(next)<previous || static_cast<Count>(next)>nq)
                throw py::value_error("MANTA periodic RI UHF offsets must be monotone and bounded");
            const Count length=static_cast<Count>(next)-previous,pair=i-1,nk=mesh.size();
            if(o.exchange_algorithm=="ri" ? ((pair/nk==pair%nk && length) || (pair/nk!=pair%nk && !length)) : (pair/nk==pair%nk && !length))
                throw py::value_error("MANTA periodic RI UHF pair segments must match the selected exchange source");
            maximum=std::max(maximum,length); previous=next;
        }
        if(previous!=nq) throw py::value_error("MANTA periodic RI UHF offsets must end at the vector count");
        plan_periodic_ri_uhf(basis,auxiliary,system,mesh,ng,maximum,nc,o,budget);
        const auto reciprocal=copy(g),exchange=copy(q),translations=copy(cells);
        std::vector<Count> boundaries(pairs+1);
        for(Count i=0;i<=pairs;++i) boundaries[i]=item<std::int64_t>(offsets,i*offsets.strides(0));
        return run_periodic_ri_uhf(basis,auxiliary,system,mesh,reciprocal,exchange,boundaries,translations,o,budget);
    },py::arg("basis"),py::arg("auxiliary"),py::arg("system"),py::arg("mesh"),py::arg("reciprocal_vectors").noconvert(),
      py::arg("pair_vectors").noconvert(),py::arg("transfer_offsets").noconvert(),py::arg("translations").noconvert(),
      py::arg("options")=PeriodicRIUHFOptions{},py::arg("budget")=ResourceBudget{});
    auto dp=py::class_<PeriodicRIUHFDomainPlan>(m,"PeriodicRIUHFDomainPlan");
#define D(name) dp.def_readonly(#name,&PeriodicRIUHFDomainPlan::name)
    D(uhf); D(reciprocal_energy_cutoff); D(pair_cutoff_bohr); D(translation_radius_bohr); D(momenta);
    D(translation_candidates); D(hartree_candidates); D(maximum_exchange_candidates); D(exchange_candidates); D(candidate_visits);
    D(domain_live_bytes); D(domain_scalar_terms); D(domain_kernel_terms);
    D(modeled_peak_bytes); D(shell_calls); D(primitive_pairs); D(scalar_terms); D(kernel_terms);
#undef D
    auto domains=py::class_<PeriodicRIUHFDomains>(m,"PeriodicRIUHFDomains");
    domains.def_readonly("plan",&PeriodicRIUHFDomains::plan).def_readonly("receipt",&PeriodicRIUHFDomains::receipt);
    auto matrix=[](const md::Matrix& a) {
        py::array_t<double> out({static_cast<py::ssize_t>(a.rows()),py::ssize_t{3}});
        for(Eigen::Index i=0;i<a.rows();++i) for(int j=0;j<3;++j) out.mutable_data()[3*i+j]=a(i,j);
        return out;
    };
    domains.def_property_readonly("translations",[matrix](const PeriodicRIUHFDomains& d){return matrix(d.translations);})
        .def_property_readonly("reciprocal_vectors",[matrix](const PeriodicRIUHFDomains& d){return matrix(d.reciprocal_vectors);})
        .def_property_readonly("pair_vectors",[matrix](const PeriodicRIUHFDomains& d){return matrix(d.exchange_vectors);})
        .def_property_readonly("transfer_offsets",[](const PeriodicRIUHFDomains& d){
            py::array_t<std::int64_t> out(d.transfer_offsets.size());
            for(Count i=0;i<d.transfer_offsets.size();++i) out.mutable_data()[i]=d.transfer_offsets[i];
            return out;
        });
    m.def("plan_periodic_ri_uhf_from_cutoffs",&plan_periodic_ri_uhf_from_cutoffs,py::arg("basis"),py::arg("auxiliary"),
        py::arg("system"),py::arg("mesh"),py::arg("reciprocal_energy_cutoff"),py::arg("options"),py::arg("budget")=ResourceBudget{});
    m.def("build_periodic_ri_uhf_domains",&build_periodic_ri_uhf_domains,py::arg("basis"),py::arg("auxiliary"),
        py::arg("system"),py::arg("mesh"),py::arg("reciprocal_energy_cutoff"),py::arg("options"),py::arg("budget")=ResourceBudget{});
    m.def("run_periodic_ri_uhf_from_cutoffs",&run_periodic_ri_uhf_from_cutoffs,py::arg("basis"),py::arg("auxiliary"),
        py::arg("system"),py::arg("mesh"),py::arg("reciprocal_energy_cutoff"),py::arg("options"),py::arg("budget")=ResourceBudget{});
}
