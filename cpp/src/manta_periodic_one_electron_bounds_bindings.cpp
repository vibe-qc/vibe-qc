#include "vibeqc/manta_periodic_one_electron_bounds.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py=pybind11;
namespace {
py::array_t<double> owned(const vibeqc::manta::md::Matrix& m) {
    py::array_t<double> out({py::ssize_t(m.rows()),py::ssize_t(m.cols())});auto a=out.mutable_unchecked<2>();
    for(Eigen::Index i=0;i<m.rows();++i)for(Eigen::Index j=0;j<m.cols();++j)a(i,j)=m(i,j);return out;
}
}
void bind_manta_periodic_one_electron_bounds(py::module_& m) {
    using namespace vibeqc::manta;
    auto p=py::class_<PeriodicOneElectronImagePlan>(m,"PeriodicOneElectronImagePlan");
#define P(n) p.def_readonly(#n,&PeriodicOneElectronImagePlan::n)
    P(n_basis);P(momenta);P(cube_radius);P(cube_cells);P(shell_calls);P(primitive_pairs);P(scalar_terms);P(kernel_terms);P(modeled_peak_bytes);
#undef P
    auto r=py::class_<PeriodicOneElectronImageReceipt>(m,"PeriodicOneElectronImageReceipt");
#define R(n) r.def_readonly(#n,&PeriodicOneElectronImageReceipt::n)
    R(qualification);R(source_identity);R(pair_source_identity);R(cube_radius);R(cube_cells);R(shell_pairs_evaluated);R(primitive_pairs_evaluated);
    R(retained_ordered_pairs);R(omitted_ordered_pairs_in_cube);R(real_metric_lower);R(maximum_overlap_error);R(maximum_kinetic_error);R(roundoff_included);
#undef R
    py::class_<PeriodicOneElectronImageResult>(m,"PeriodicOneElectronImageResult")
        .def_readonly("plan",&PeriodicOneElectronImageResult::plan).def_readonly("receipt",&PeriodicOneElectronImageResult::receipt)
        .def_property_readonly("overlap_error_bound",[](const PeriodicOneElectronImageResult& r){return owned(r.overlap_error_bound);})
        .def_property_readonly("kinetic_error_bound",[](const PeriodicOneElectronImageResult& r){return owned(r.kinetic_error_bound);});
    m.def("plan_periodic_one_electron_images",py::overload_cast<const PeriodicPairSource&,const ResourceBudget&>(&plan_periodic_one_electron_images),py::arg("source"),py::arg("budget")=ResourceBudget{});
    m.def("bound_periodic_one_electron_images",py::overload_cast<const PeriodicPairSource&,const ResourceBudget&>(&bound_periodic_one_electron_images),py::arg("source"),py::arg("budget")=ResourceBudget{},py::call_guard<py::gil_scoped_release>());
    m.def("plan_periodic_one_electron_images",py::overload_cast<const PeriodicPairSource&,Count,const ResourceBudget&>(&plan_periodic_one_electron_images),py::arg("source"),py::arg("completion_cube_radius"),py::arg("budget")=ResourceBudget{});
    m.def("bound_periodic_one_electron_images",py::overload_cast<const PeriodicPairSource&,Count,const ResourceBudget&>(&bound_periodic_one_electron_images),py::arg("source"),py::arg("completion_cube_radius"),py::arg("budget")=ResourceBudget{},py::call_guard<py::gil_scoped_release>());
    auto rank=py::class_<PeriodicOverlapImageRankResult>(m,"PeriodicOverlapImageRankResult");
#define R(n) rank.def_readonly(#n,&PeriodicOverlapImageRankResult::n)
    R(qualified);R(maximum_perturbation_bound);R(minimum_stability_radius);R(perturbation_bounds);R(projection_bounds);R(spectra);R(certificate_identity);
#undef R
    m.def("plan_periodic_overlap_image_rank",&plan_periodic_overlap_image_rank,py::arg("n"),py::arg("nk"),
        py::arg("scf"),py::arg("budget")=ResourceBudget{});
    m.def("bound_periodic_overlap_image_rank",[](const py::array& overlap,const py::array& bounds,
        Count expected_rank,const vibeqc::BoundedPeriodicRHFOptions& scf,const ResourceBudget& budget) {
        if(overlap.ndim()!=3 || overlap.shape(0)<=0 || overlap.shape(1)<=0 || overlap.shape(1)!=overlap.shape(2) ||
           !overlap.dtype().is(py::dtype::of<std::complex<double>>()) || !(overlap.flags() & py::array::c_style))
            throw py::value_error("MANTA overlap image rank requires contiguous complex128 [k,n,n]");
        const Count nk=overlap.shape(0),n=overlap.shape(1);
        if(bounds.ndim()!=2 || bounds.shape(0)!=n || bounds.shape(1)!=n ||
           !bounds.dtype().is(py::dtype::of<double>()) || !(bounds.flags() & py::array::c_style))
            throw py::value_error("MANTA overlap image rank requires contiguous float64 [n,n] bounds");
        plan_periodic_overlap_image_rank(n,nk,scf,budget); // Before payload copies/inspection.
        const auto* raw=static_cast<const std::complex<double>*>(overlap.data());
        std::vector<std::complex<double>> copied(raw,raw+overlap.size());
        md::Matrix errors(n,n);const auto* b=static_cast<const double*>(bounds.data());
        for(Count i=0;i<n;++i)for(Count j=0;j<n;++j)errors(i,j)=b[i*n+j];
        return bound_periodic_overlap_image_rank(n,nk,copied,errors,expected_rank,scf,budget);
    },py::arg("overlap").noconvert(),py::arg("bounds").noconvert(),py::arg("expected_rank"),
      py::arg("scf"),py::arg("budget")=ResourceBudget{});
    auto o=py::class_<PeriodicOneElectronAccuracyOptions>(m,"PeriodicOneElectronAccuracyOptions").def(py::init<>());
#define O(n) o.def_readwrite(#n,&PeriodicOneElectronAccuracyOptions::n)
    O(overlap_error);O(kinetic_error);O(kinetic_energy_error);O(completion_cube_radius);O(overlap_image_rank);
#undef O
    auto ap=py::class_<PeriodicOneElectronAccuracyPlan>(m,"PeriodicOneElectronAccuracyPlan");
#define A(n) ap.def_readonly(#n,&PeriodicOneElectronAccuracyPlan::n)
    A(pairs);A(images);A(overlap_rank);A(modeled_peak_bytes);A(shell_calls);A(primitive_pairs);A(scalar_terms);A(kernel_terms);
#undef A
    auto ar=py::class_<PeriodicOneElectronAccuracyResult>(m,"PeriodicOneElectronAccuracyResult");
#define A(n) ar.def_readonly(#n,&PeriodicOneElectronAccuracyResult::n)
    A(images);A(qualified);A(overlap_rank);A(kinetic_energy_error_bound);A(density_l1_limit);
    A(density_envelope_identity);A(certificate_identity);A(shell_calls);A(primitive_pairs);
    ar.def_property_readonly("targets",[](const PeriodicOneElectronAccuracyResult& r){return r.targets;});
#undef A
}
