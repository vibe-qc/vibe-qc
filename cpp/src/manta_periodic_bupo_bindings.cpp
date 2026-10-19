#include "vibeqc/manta_periodic_bupo.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring>
namespace py=pybind11;
namespace {
using namespace vibeqc::manta;
void shape(const py::array& a,const py::dtype& type,int ndim) {
    if(!a.dtype().is(type) || a.ndim()!=ndim || !(a.flags()&(py::array::c_style|py::array::f_style)))
        throw py::value_error("MANTA periodic BUPO needs contiguous arrays with the documented dtype and shape");
}
template<class T> T item(const py::array& a,py::ssize_t offset) {
    T value; std::memcpy(&value,static_cast<const char*>(a.data())+offset,sizeof(T)); return value;
}
py::object optional_matrix(const std::optional<md::Matrix>& value) {
    if(!value) return py::none();
    py::array_t<double> a({value->rows(),value->cols()});
    std::memcpy(a.mutable_data(),value->data(),value->size()*sizeof(double)); return a;
}
} // namespace
void bind_manta_periodic_bupo(py::module_& m) {
    using namespace vibeqc::manta;
    py::class_<PeriodicBupoOptions>(m,"PeriodicBupoOptions").def(py::init<>())
        .def_readwrite("opening",&PeriodicBupoOptions::opening)
        .def_readwrite("integral_tolerance",&PeriodicBupoOptions::integral_tolerance)
        .def_readwrite("bound_infinite_domain",&PeriodicBupoOptions::bound_infinite_domain)
        .def_readwrite("bound_pair_images",&PeriodicBupoOptions::bound_pair_images);
    auto p=py::class_<PeriodicBupoPlan>(m,"PeriodicBupoPlan");
#define P(n) p.def_readonly(#n,&PeriodicBupoPlan::n)
    P(order); P(n_basis); P(momenta); P(pairs); P(nodes); P(moments); P(maximum_queries);
    P(shell_calls); P(primitive_products); P(scalar_terms); P(kernel_terms); P(modeled_peak_bytes);
    P(domain_majorant_terms); P(domain_pair_interactions);
    P(image_cube_radius); P(image_cells); P(image_primitive_products);
#undef P
    auto b=py::class_<PeriodicBupoPartition>(m,"PeriodicBupoPartition");
#define B(n) b.def_readonly(#n,&PeriodicBupoPartition::n)
    B(traversal); B(approximate_blocks); B(direct_blocks); B(approximate_pairs); B(direct_pairs); B(maximum_integral_error); B(identity);
#undef B
    auto domain=py::class_<PeriodicBupoDomainReceipt>(m,"PeriodicBupoDomainReceipt");
#define D(n) domain.def_readonly(#n,&PeriodicBupoDomainReceipt::n)
    D(qualification); D(source_identity); D(pair_source_identity); D(kernel_identity); D(envelope_identity); D(density_identity);
    D(real_domain_identity); D(reciprocal_domain_identity); D(real_cube_radius); D(reciprocal_cube_radius);
    D(majorant_terms); D(pair_interactions); D(majorant_pairs_evaluated); D(majorant_shell_calls);
    D(real_energy_error_bound); D(reciprocal_energy_error_bound); D(energy_error_bound);
    D(maximum_j_error_bound); D(maximum_integral_error); D(minimum_effective_eta); D(minimum_omitted_real_distance);
    D(roundoff_included); D(pair_image_omission_included);
#undef D
    auto image=py::class_<PeriodicBupoPairImageReceipt>(m,"PeriodicBupoPairImageReceipt");
#define I(n) image.def_readonly(#n,&PeriodicBupoPairImageReceipt::n)
    I(qualification); I(source_identity); I(pair_source_identity); I(kernel_identity); I(density_identity);
    I(cube_radius); I(cube_cells); I(shell_pairs_evaluated); I(primitive_pairs_evaluated);
    I(retained_ordered_pairs); I(omitted_ordered_pairs_in_cube);
    I(real_metric_lower); I(reciprocal_metric_lower); I(volume_lower); I(maximum_gaussian_exponent);
    I(real_kernel_bound); I(reciprocal_kernel_bound); I(gauge_kernel_bound); I(kernel_bound);
    I(retained_density_mass_bound); I(omitted_density_mass_bound);
    I(maximum_retained_pair_mass); I(maximum_omitted_pair_mass);
    I(energy_error_bound); I(maximum_j_error_bound); I(roundoff_included);
#undef I
    auto r=py::class_<PeriodicBupoReceipt>(m,"PeriodicBupoReceipt");
#define R(n) r.def_readonly(#n,&PeriodicBupoReceipt::n)
    R(direct); R(partition); R(source_identity); R(pair_source_identity); R(tree_identity); R(kernel_identity); R(density_identity);
    R(qualification); R(direct_energy); R(far_energy); R(energy); R(trace_energy); R(energy_error_bound);
    R(maximum_j_error_bound); R(charge); R(hermiticity_residual); R(forward_shell_calls); R(adjoint_shell_calls);
    R(translation_calls); R(far_matrix_terms); R(modeled_peak_bytes); R(roundoff_included);
    R(finite_energy_error_bound); R(finite_maximum_j_error_bound); R(domain); R(pair_images);
#undef R
    py::class_<PeriodicBupoResult>(m,"PeriodicBupoResult")
        .def_readonly("plan",&PeriodicBupoResult::plan).def_readonly("receipt",&PeriodicBupoResult::receipt)
        .def_property_readonly("values",[](const PeriodicBupoResult& r) {
            const auto n=static_cast<py::ssize_t>(r.plan.n_basis);
            py::array_t<std::complex<double>> a({static_cast<py::ssize_t>(r.plan.momenta),n,n});
            if(!r.values.empty()) std::memcpy(a.mutable_data(),r.values.data(),r.values.size()*sizeof(std::complex<double>)); return a;
        })
        .def_property_readonly("j_error_bound",[](const PeriodicBupoResult& r) {
            const auto n=static_cast<py::ssize_t>(r.plan.n_basis); py::array_t<double> a({n,n});
            std::memcpy(a.mutable_data(),r.j_error_bound.data(),r.j_error_bound.size()*sizeof(double)); return a;
        })
        .def_property_readonly("domain_j_error_bound",[](const PeriodicBupoResult& r)->py::object {
            if(!r.domain_j_error_bound) return py::none();
            const auto& value=*r.domain_j_error_bound; const auto n=static_cast<py::ssize_t>(r.plan.n_basis);
            py::array_t<double> a({n,n}); std::memcpy(a.mutable_data(),value.data(),value.size()*sizeof(double)); return a;
        })
        .def_property_readonly("pair_image_j_error_bound",[](const PeriodicBupoResult& r) { return optional_matrix(r.pair_image_j_error_bound); })
        .def_property_readonly("retained_pair_mass_bound",[](const PeriodicBupoResult& r) { return optional_matrix(r.retained_pair_mass_bound); })
        .def_property_readonly("omitted_pair_mass_bound",[](const PeriodicBupoResult& r) { return optional_matrix(r.omitted_pair_mass_bound); });
    m.def("plan_periodic_bupo",&PeriodicBupoSource::plan_source,py::arg("tree"),py::arg("direct"),py::arg("near_cells"),
        py::arg("options")=PeriodicBupoOptions{},py::arg("budget")=ResourceBudget{});
    py::class_<PeriodicBupoSource,std::shared_ptr<PeriodicBupoSource>>(m,"PeriodicBupoSource")
        .def(py::init([](std::shared_ptr<const PeriodicBubbleTree> tree,std::shared_ptr<const PeriodicEwaldSource> direct,
            const py::array& near,const PeriodicBupoOptions& options,const ResourceBudget& budget) {
            if(!tree || !direct) throw py::value_error("MANTA periodic BUPO requires native hierarchy and direct Ewald sources");
            shape(near,py::dtype::of<std::int64_t>(),2);
            if(near.shape(1)!=3) throw py::value_error("MANTA periodic BUPO near indices require (count,3) shape");
            PeriodicBupoSource::plan_source(*tree,*direct,near.shape(0),options,budget);
            std::vector<LatticeIndex> indices(near.shape(0));
            for(py::ssize_t i=0;i<near.shape(0);++i) for(int j=0;j<3;++j) indices[i][j]=item<std::int64_t>(near,i*near.strides(0)+j*near.strides(1));
            const auto opts=options; const auto limits=budget;
            py::gil_scoped_release release; return std::make_shared<PeriodicBupoSource>(std::move(tree),std::move(direct),indices,opts,limits);
        }),py::arg("tree"),py::arg("direct"),py::arg("near_indices").noconvert(),
           py::arg("options")=PeriodicBupoOptions{},py::arg("budget")=ResourceBudget{})
        .def_property_readonly("plan",[](const PeriodicBupoSource& s) { return PeriodicBupoPlan(s.plan()); })
        .def_property_readonly("partition",[](const PeriodicBupoSource& s) { return PeriodicBupoPartition(s.partition()); })
        .def_property_readonly("source_identity",&PeriodicBupoSource::source_identity)
        .def("build",[](const PeriodicBupoSource& s,const py::array& d) {
            shape(d,py::dtype::of<std::complex<double>>(),3); const auto& p=s.plan();
            if(d.shape(0)!=static_cast<py::ssize_t>(p.momenta) || d.shape(1)!=static_cast<py::ssize_t>(p.n_basis) || d.shape(2)!=static_cast<py::ssize_t>(p.n_basis))
                throw py::value_error("MANTA periodic BUPO density requires (momenta,AO,AO) shape");
            std::vector<std::complex<double>> values(p.momenta*p.n_basis*p.n_basis);
            for(Count k=0;k<p.momenta;++k) for(Count i=0;i<p.n_basis;++i) for(Count j=0;j<p.n_basis;++j)
                values[(k*p.n_basis+i)*p.n_basis+j]=item<std::complex<double>>(d,k*d.strides(0)+i*d.strides(1)+j*d.strides(2));
            py::gil_scoped_release release; return s.build(values);
        },py::arg("densities").noconvert());
}
