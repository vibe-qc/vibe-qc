#include "vibeqc/manta_periodic_bupo_hf.hpp"
#include "vibeqc/manta_admission.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>
namespace py=pybind11;
namespace {
using namespace vibeqc::manta;
template<class T> T item(const py::array& a,py::ssize_t offset) {T x;std::memcpy(&x,static_cast<const char*>(a.data())+offset,sizeof(T));return x;}
template<class T> py::array_t<T> owned(const T* data,std::vector<py::ssize_t> shape) {
    py::array_t<T> out(shape);if(out.size())std::memcpy(out.mutable_data(),data,out.size()*sizeof(T));return out;
}
}
void bind_manta_periodic_bupo_hf(py::module_& m) {
    using namespace vibeqc::manta;
    auto o=py::class_<PeriodicBupoHFOptions>(m,"PeriodicBupoHFOptions").def(py::init<>());
#define O(n) o.def_readwrite(#n,&PeriodicBupoHFOptions::n)
    O(scf);O(spin_mode);O(exchange_q0);O(madelung_alpha);O(madelung_tolerance);O(reciprocal_batch);O(charge_tolerance);
    O(maximum_electrostatic_energy_error);O(maximum_electrostatic_matrix_error);O(confirmation_fock_tolerance);O(confirmation_energy_tolerance);
    O(maximum_overlap_image_error);O(maximum_kinetic_image_error);O(maximum_kinetic_energy_error);
    O(bound_exchange);O(exchange_bound_cube_radius);O(maximum_exchange_matrix_error);O(maximum_exchange_energy_error);
#undef O
    auto p=py::class_<PeriodicBupoHFPlan>(m,"PeriodicBupoHFPlan");
#define P(n) p.def_readonly(#n,&PeriodicBupoHFPlan::n)
    P(electrostatic);P(kinetic);P(exchange);P(madelung);P(solver);P(n_basis);P(momenta);P(spins);P(electrons);P(n_alpha);P(n_beta);P(cells);
    P(one_electron_images);
    P(exchange_bounds);
    P(fock_calls_upper_bound);P(provider_kernel_terms_per_call);P(driver_live_bytes);P(modeled_peak_bytes);P(shell_calls);P(primitive_pairs);
    P(scalar_terms);P(kernel_terms);P(history_capacity);P(history_live_bytes);
#undef P
    auto r=py::class_<PeriodicBupoHFReceipt,PeriodicUHFReceipt>(m,"PeriodicBupoHFReceipt");
#define R(n) r.def_readonly(#n,&PeriodicBupoHFReceipt::n)
    R(bupo_source_identity);R(accuracy_qualification);R(electrostatic);R(exchange);R(maximum_electrostatic_energy_error);
    R(maximum_electrostatic_matrix_error);R(electrostatic_accuracy_passed);R(coulomb);
    R(maximum_observed_electrostatic_energy_error);R(maximum_observed_electrostatic_matrix_error);
    R(maximum_overlap_image_error);R(maximum_kinetic_image_error);R(maximum_kinetic_energy_error);
    R(kinetic_energy_error_bound);R(maximum_observed_kinetic_energy_error);R(one_electron_accuracy_passed);
    R(exchange_bounds);R(maximum_exchange_matrix_error);R(maximum_exchange_energy_error);R(exchange_matrix_error_bound);R(exchange_energy_error_bound);
    R(maximum_observed_exchange_matrix_error);R(maximum_observed_exchange_energy_error);R(exchange_correction_matrix_error);R(exchange_correction_energy_error);R(exchange_accuracy_passed);
#undef R
    py::class_<PeriodicBupoHFResult>(m,"PeriodicBupoHFResult")
        .def_readonly("plan",&PeriodicBupoHFResult::plan).def_readonly("receipt",&PeriodicBupoHFResult::receipt)
        .def_readonly("one_electron_images",&PeriodicBupoHFResult::one_electron_images)
        .def_property_readonly("converged",&PeriodicBupoHFResult::converged)
        .def_property_readonly("solver_snapshot",[](const PeriodicBupoHFResult& r){return r.rhf?py::cast(r.rhf->final_snapshot(),py::return_value_policy::copy):py::cast(r.uhf->final_snapshot(),py::return_value_policy::copy);})
        .def_property_readonly("solver_diagnostics",[](const PeriodicBupoHFResult& r){return r.rhf?r.rhf->diagnostics():r.uhf->diagnostics();})
        .def_property_readonly("solver_memory",[](const PeriodicBupoHFResult& r){return r.rhf?py::cast(r.rhf->memory(),py::return_value_policy::copy):py::cast(r.uhf->memory(),py::return_value_policy::copy);})
        .def_property_readonly("scf_history",[](const PeriodicBupoHFResult& r){return r.rhf?py::cast(r.rhf_history,py::return_value_policy::copy):py::cast(r.uhf_history,py::return_value_policy::copy);})
        .def_property_readonly("density",[](const PeriodicBupoHFResult& r){const auto& p=r.plan;return owned(r.rhf?r.rhf->density_data():r.uhf->density_data(),
            {py::ssize_t(p.spins),py::ssize_t(p.momenta),py::ssize_t(p.n_basis),py::ssize_t(p.n_basis)});})
        .def_property_readonly("fock",[](const PeriodicBupoHFResult& r){const auto& p=r.plan;return owned(r.rhf?r.rhf->fock_data():r.uhf->fock_data(),
            {py::ssize_t(p.spins),py::ssize_t(p.momenta),py::ssize_t(p.n_basis),py::ssize_t(p.n_basis)});})
        .def_property_readonly("coefficients",[](const PeriodicBupoHFResult& r){const auto& p=r.plan;const auto rank=r.rhf?r.rhf->memory().retained_rank:r.uhf->memory().retained_rank;
            return owned(r.rhf?r.rhf->coefficients_data():r.uhf->coefficients_data(),{py::ssize_t(p.spins),py::ssize_t(p.momenta),py::ssize_t(p.n_basis),py::ssize_t(rank)});})
        .def_property_readonly("orbital_energies",[](const PeriodicBupoHFResult& r){const auto& p=r.plan;const auto rank=r.rhf?r.rhf->memory().retained_rank:r.uhf->memory().retained_rank;
            return owned(r.rhf?r.rhf->orbital_energies_data():r.uhf->orbital_energies_data(),{py::ssize_t(p.spins),py::ssize_t(p.momenta),py::ssize_t(rank)});})
        .def_property_readonly("overlap",[](const PeriodicBupoHFResult& r){const auto& p=r.plan;return owned(r.overlap.data(),{py::ssize_t(p.momenta),py::ssize_t(p.n_basis),py::ssize_t(p.n_basis)});})
        .def_property_readonly("hcore",[](const PeriodicBupoHFResult& r){const auto& p=r.plan;return owned(r.hcore.data(),{py::ssize_t(p.momenta),py::ssize_t(p.n_basis),py::ssize_t(p.n_basis)});});
    m.def("plan_periodic_bupo_hf",&plan_periodic_bupo_hf,py::arg("electronic"),py::arg("system"),py::arg("mesh"),py::arg("exchange_vectors_per_pair"),
        py::arg("options")=PeriodicBupoHFOptions{},py::arg("budget")=ResourceBudget{});
    m.def("run_periodic_bupo_hf",[](std::shared_ptr<const PeriodicBupoSource> source,const vibeqc::PeriodicSystem& system,const vibeqc::RegularKMesh& mesh,
        const py::array& q,const py::array& offsets,PeriodicBupoHFOptions options,ResourceBudget budget) {
        using namespace admission;
        if(!source)throw py::value_error("MANTA BUPO HF requires a native electronic source");
        if(!q.dtype().is(py::dtype::of<double>()) || q.ndim()!=2 || q.shape(1)!=3 || !(q.flags()&(py::array::c_style|py::array::f_style)))
            throw py::value_error("MANTA BUPO HF exchange vectors require contiguous float64 (count,3)");
        const Count pairs=square(mesh.size()),nq=q.shape(0);
        if(!offsets.dtype().is(py::dtype::of<std::int64_t>()) || offsets.ndim()!=1 || Count(offsets.shape(0))!=add(pairs,1)
           || !(offsets.flags()&(py::array::c_style|py::array::f_style)))throw py::value_error("MANTA BUPO HF offsets require contiguous int64 (nk*nk+1)");
        plan_periodic_bupo_hf(*source,system,mesh,0,options,budget);
        if(item<std::int64_t>(offsets,0)!=0)throw py::value_error("MANTA BUPO HF offsets must start at zero");
        Count previous=0,maximum=0;
        for(Count i=1;i<=pairs;++i) {
            const auto next=item<std::int64_t>(offsets,i*offsets.strides(0));
            if(next<0 || Count(next)<previous || Count(next)>nq)throw py::value_error("MANTA BUPO HF offsets must be monotone and bounded");
            maximum=std::max(maximum,Count(next)-previous);previous=next;
        }
        if(previous!=nq)throw py::value_error("MANTA BUPO HF offsets must end at the vector count");
        plan_periodic_bupo_hf(*source,system,mesh,maximum,options,budget);
        md::Matrix vectors(nq,3);for(Count i=0;i<nq;++i)for(int j=0;j<3;++j)vectors(i,j)=item<double>(q,i*q.strides(0)+j*q.strides(1));
        std::vector<Count> boundaries(pairs+1);for(Count i=0;i<=pairs;++i)boundaries[i]=item<std::int64_t>(offsets,i*offsets.strides(0));
        const auto physical=system;const auto mesh_copy=mesh;py::gil_scoped_release release;
        return run_periodic_bupo_hf(std::move(source),physical,mesh_copy,vectors,boundaries,options,budget);
    },py::arg("electronic"),py::arg("system"),py::arg("mesh"),py::arg("exchange_vectors").noconvert(),py::arg("transfer_offsets").noconvert(),
       py::arg("options")=PeriodicBupoHFOptions{},py::arg("budget")=ResourceBudget{});
}
