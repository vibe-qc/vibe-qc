#include "vibeqc/manta_periodic_fitting.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>
namespace py=pybind11;
using namespace vibeqc::manta;
namespace {
void check(const py::array& a,int dimensions) {
    if(!a.dtype().is(py::dtype::of<double>()) || a.ndim()!=dimensions ||
       !(a.flags() & (py::array::c_style|py::array::f_style)))
        throw py::value_error("MANTA periodic fitting requires C- or F-contiguous float64 q and ket momenta");
}
double item(const py::array& a,py::ssize_t offset) {
    double value;std::memcpy(&value,static_cast<const char*>(a.data())+offset,sizeof(double));return value;
}
}
void bind_manta_periodic_fitting(py::module_& m) {
    auto o=py::class_<PeriodicGDFShortRangeOptions>(m,"PeriodicGDFShortRangeOptions").def(py::init<>());
#define O(name) o.def_readwrite(#name,&PeriodicGDFShortRangeOptions::name)
    O(omega);O(pair_cutoff_bohr);O(auxiliary_cutoff_bohr);O(integral_screen_error);O(image_candidate_cap);O(compute_metric);
#undef O
    auto p=py::class_<PeriodicGDFShortRangePlan>(m,"PeriodicGDFShortRangePlan");
#define P(name) p.def_readonly(#name,&PeriodicGDFShortRangePlan::name)
    P(n_aux);P(n_orb);P(momenta);P(compute_metric);P(metric_image_candidates);P(pair_image_candidates);P(auxiliary_image_candidates);
    P(metric_shell_calls);P(three_center_shell_calls);P(shell_calls);P(primitive_products);P(scalar_terms);P(kernel_terms);
    P(output_bytes);P(driver_live_bytes);P(modeled_peak_bytes);P(metric);P(three_center);
#undef P
    auto r=py::class_<PeriodicGDFShortRangeReceipt>(m,"PeriodicGDFShortRangeReceipt");
#define R(name) r.def_readonly(#name,&PeriodicGDFShortRangeReceipt::name)
    R(provider);R(provider_version);R(normalization);R(algorithm);R(kernel);R(convention);
    R(orbital_basis_identity);R(auxiliary_basis_identity);R(metric_source_identity);R(three_center_source_identity);R(source_identity);
    R(omega);R(pair_cutoff_bohr);R(auxiliary_cutoff_bohr);R(integral_screen_error);R(metric_shell_calls);R(three_center_shell_calls);
    R(pair_shell_images);R(primitive_products);R(phase_products);R(phase_accumulations);R(modeled_peak_bytes);R(threads);
#undef R
    py::class_<PeriodicGDFShortRangeResult>(m,"PeriodicGDFShortRangeResult")
        .def_readonly("plan",&PeriodicGDFShortRangeResult::plan)
        .def_readonly("receipt",&PeriodicGDFShortRangeResult::receipt)
        .def_property_readonly("metric",[](const PeriodicGDFShortRangeResult& r) {
            const auto n=static_cast<py::ssize_t>(r.plan.compute_metric?r.plan.n_aux:0);
            py::array_t<std::complex<double>> out({n,n});
            if(n) std::memcpy(out.mutable_data(),r.metric_values.data(),r.metric_values.size()*sizeof(std::complex<double>));return out;
        })
        .def_property_readonly("three_center",[](const PeriodicGDFShortRangeResult& r) {
            py::array_t<std::complex<double>> out({static_cast<py::ssize_t>(r.plan.momenta),static_cast<py::ssize_t>(r.plan.n_aux),
                static_cast<py::ssize_t>(r.plan.n_orb),static_cast<py::ssize_t>(r.plan.n_orb)});
            std::memcpy(out.mutable_data(),r.three_center_values.data(),r.three_center_values.size()*sizeof(std::complex<double>));return out;
        });
    m.def("plan_periodic_gdf_short_range",&plan_periodic_gdf_short_range,
        py::arg("orbital"),py::arg("auxiliary"),py::arg("system"),py::arg("momenta"),
        py::arg("options")=PeriodicGDFShortRangeOptions{},py::arg("budget")=ResourceBudget{});
    m.def("compute_periodic_gdf_short_range",[](const vibeqc::BasisSet& orbital,const vibeqc::BasisSet& auxiliary,
        const vibeqc::PeriodicSystem& system,const py::array& q,const py::array& k,
        const PeriodicGDFShortRangeOptions& o,const ResourceBudget& budget) {
        check(q,1);check(k,2);
        if(q.shape(0)!=3 || k.shape(1)!=3) throw py::value_error("MANTA periodic fitting q must have shape (3,) and ket momenta (nk,3)");
        // Complete joint admission precedes either numerical input copy. Keep
        // the GIL; the borrowed bases/system/options must remain unchanged.
        plan_periodic_gdf_short_range(orbital,auxiliary,system,k.shape(0),o,budget);
        Eigen::Vector3d transfer;md::Matrix momenta(k.shape(0),3);
        for(int j=0;j<3;++j) transfer[j]=item(q,j*q.strides(0));
        for(Eigen::Index i=0;i<momenta.rows();++i)for(int j=0;j<3;++j)momenta(i,j)=item(k,i*k.strides(0)+j*k.strides(1));
        return compute_periodic_gdf_short_range(orbital,auxiliary,system,transfer,momenta,o,budget);
    },py::arg("orbital"),py::arg("auxiliary"),py::arg("system"),py::arg("q").noconvert(),py::arg("ket_kpoints").noconvert(),
        py::arg("options")=PeriodicGDFShortRangeOptions{},py::arg("budget")=ResourceBudget{});

    py::class_<PeriodicGDFKernelOptions>(m,"PeriodicGDFKernelOptions").def(py::init<>())
        .def_readwrite("kernel",&PeriodicGDFKernelOptions::kernel)
        .def_readwrite("omega",&PeriodicGDFKernelOptions::omega);
    auto gp=py::class_<PeriodicGDFPlan>(m,"PeriodicGDFPlan");
#define P(name) gp.def_readonly(#name,&PeriodicGDFPlan::name)
    P(n_aux);P(n_orb);P(momenta);P(vectors);P(panel_vectors);P(panel_count);P(compute_metric);
    P(shell_calls);P(primitive_products);P(vector_primitive_products);P(scalar_terms);P(kernel_terms);
    P(output_bytes);P(modeled_peak_bytes);P(short_range);P(auxiliary_fourier);P(ao_pair_fourier);
#undef P
    auto gr=py::class_<PeriodicGDFReceipt>(m,"PeriodicGDFReceipt");
#define R(name) gr.def_readonly(#name,&PeriodicGDFReceipt::name)
    R(provider);R(provider_version);R(normalization);R(algorithm);R(kernel);R(convention);
    R(metric_source_identity);R(three_center_source_identity);R(source_identity);R(short_range);
    R(zero_transfer);R(physical_zero_included);R(real_short_range);R(physical_omega);R(finite_zero_removed);R(reciprocal_vectors);R(auxiliary_fourier_shell_calls);R(pair_fourier_shell_calls);
    R(shell_calls);R(primitive_products);R(vector_primitive_products);R(phase_products);R(phase_accumulations);
    R(contraction_terms);R(modeled_peak_bytes);R(threads);
#undef R
    py::class_<PeriodicGDFResult>(m,"PeriodicGDFResult")
        .def_readonly("plan",&PeriodicGDFResult::plan)
        .def_readonly("receipt",&PeriodicGDFResult::receipt)
        .def_property_readonly("metric",[](const PeriodicGDFResult& r) {
            const auto n=static_cast<py::ssize_t>(r.plan.compute_metric?r.plan.n_aux:0);
            py::array_t<std::complex<double>> out({n,n});
            if(n) std::memcpy(out.mutable_data(),r.metric_values.data(),r.metric_values.size()*sizeof(std::complex<double>));return out;
        })
        .def_property_readonly("three_center",[](const PeriodicGDFResult& r) {
            py::array_t<std::complex<double>> out({static_cast<py::ssize_t>(r.plan.momenta),static_cast<py::ssize_t>(r.plan.n_aux),
                static_cast<py::ssize_t>(r.plan.n_orb),static_cast<py::ssize_t>(r.plan.n_orb)});
            std::memcpy(out.mutable_data(),r.three_center_values.data(),r.three_center_values.size()*sizeof(std::complex<double>));return out;
        });
    m.def("plan_periodic_gdf",&plan_periodic_gdf,py::arg("orbital"),py::arg("auxiliary"),py::arg("system"),
        py::arg("vectors"),py::arg("momenta"),py::arg("options")=PeriodicGDFShortRangeOptions{},py::arg("budget")=ResourceBudget{},py::arg("physical_kernel")=PeriodicGDFKernelOptions{});
    m.def("compute_periodic_gdf",[](const vibeqc::BasisSet& orbital,const vibeqc::BasisSet& auxiliary,
        const vibeqc::PeriodicSystem& system,const py::array& q,const py::array& k,const py::array& v,
        const PeriodicGDFShortRangeOptions& o,const ResourceBudget& budget,const PeriodicGDFKernelOptions& physical) {
        check(q,1);check(k,2);check(v,2);
        if(q.shape(0)!=3 || k.shape(1)!=3 || v.shape(1)!=3)
            throw py::value_error("MANTA GDF q must have shape (3,), momenta and vectors (n,3)");
        plan_periodic_gdf(orbital,auxiliary,system,v.shape(0),k.shape(0),o,budget,physical);
        Eigen::Vector3d transfer;md::Matrix momenta(k.shape(0),3),vectors(v.shape(0),3);
        for(int j=0;j<3;++j) transfer[j]=item(q,j*q.strides(0));
        for(Eigen::Index i=0;i<momenta.rows();++i)for(int j=0;j<3;++j)momenta(i,j)=item(k,i*k.strides(0)+j*k.strides(1));
        for(Eigen::Index i=0;i<vectors.rows();++i)for(int j=0;j<3;++j)vectors(i,j)=item(v,i*v.strides(0)+j*v.strides(1));
        return compute_periodic_gdf(orbital,auxiliary,system,transfer,momenta,vectors,o,budget,physical);
    },py::arg("orbital"),py::arg("auxiliary"),py::arg("system"),py::arg("q").noconvert(),
        py::arg("ket_kpoints").noconvert(),py::arg("vectors").noconvert(),
        py::arg("options")=PeriodicGDFShortRangeOptions{},py::arg("budget")=ResourceBudget{},py::arg("physical_kernel")=PeriodicGDFKernelOptions{});

    auto tp=py::class_<PeriodicGDFTilesPlan>(m,"PeriodicGDFTilesPlan");
#define P(name) tp.def_readonly(#name,&PeriodicGDFTilesPlan::name)
    P(n_aux);P(n_orb);P(momenta);P(vectors);P(panel_vectors);P(compute_metric);
    P(metric_shell_calls);P(coupling_shell_calls);P(metric_primitive_products);P(coupling_primitive_products);
    P(metric_scalar_terms);P(coupling_scalar_terms);P(metric_kernel_terms);P(coupling_kernel_terms);
    P(shell_calls);P(primitive_products);P(scalar_terms);P(kernel_terms);P(maximum_tile_elements);
    P(retained_bytes);P(metric_peak_bytes);P(coupling_peak_bytes);P(modeled_peak_bytes);P(external_live_bytes);
#undef P
    py::class_<PeriodicGDFTile>(m,"PeriodicGDFTile")
        .def_readonly("shape",&PeriodicGDFTile::shape).def_readonly("offsets",&PeriodicGDFTile::offsets)
        .def_readonly("receipt",&PeriodicGDFTile::receipt)
        .def_property_readonly("values",[](const PeriodicGDFTile& tile) {
            std::vector<py::ssize_t> shape(tile.shape.begin(),tile.shape.end());py::array_t<std::complex<double>> out(shape);
            if(!tile.values.empty()) std::memcpy(out.mutable_data(),tile.values.data(),tile.values.size()*sizeof(std::complex<double>));return out;
        });
    m.def("plan_periodic_gdf_tiles",&plan_periodic_gdf_tiles,py::arg("orbital"),py::arg("auxiliary"),py::arg("system"),
        py::arg("vectors"),py::arg("momenta"),py::arg("options")=PeriodicGDFShortRangeOptions{},py::arg("budget")=ResourceBudget{},py::arg("physical_kernel")=PeriodicGDFKernelOptions{});
    py::class_<PeriodicGDFTiles,std::shared_ptr<PeriodicGDFTiles>>(m,"PeriodicGDFTiles")
        .def(py::init([](const vibeqc::BasisSet& orbital,const vibeqc::BasisSet& auxiliary,
            const vibeqc::PeriodicSystem& system,const py::array& q,const py::array& k,const py::array& v,
            const PeriodicGDFShortRangeOptions& o,const ResourceBudget& budget,const PeriodicGDFKernelOptions& physical) {
            check(q,1);check(k,2);check(v,2);
            if(q.shape(0)!=3 || k.shape(1)!=3 || v.shape(1)!=3) throw py::value_error("MANTA GDF tile vectors require three components");
            plan_periodic_gdf_tiles(orbital,auxiliary,system,v.shape(0),k.shape(0),o,budget,physical);
            Eigen::Vector3d transfer;md::Matrix momenta(k.shape(0),3),vectors(v.shape(0),3);
            for(int j=0;j<3;++j) transfer[j]=item(q,j*q.strides(0));
            for(Eigen::Index i=0;i<momenta.rows();++i)for(int j=0;j<3;++j)momenta(i,j)=item(k,i*k.strides(0)+j*k.strides(1));
            for(Eigen::Index i=0;i<vectors.rows();++i)for(int j=0;j<3;++j)vectors(i,j)=item(v,i*v.strides(0)+j*v.strides(1));
            return std::make_shared<PeriodicGDFTiles>(orbital,auxiliary,system,transfer,momenta,vectors,o,budget,physical);
        }),py::arg("orbital"),py::arg("auxiliary"),py::arg("system"),py::arg("q").noconvert(),
            py::arg("ket_kpoints").noconvert(),py::arg("vectors").noconvert(),
            py::arg("options")=PeriodicGDFShortRangeOptions{},py::arg("budget")=ResourceBudget{},py::arg("physical_kernel")=PeriodicGDFKernelOptions{})
        .def_property_readonly("plan",&PeriodicGDFTiles::plan)
        .def_property_readonly("receipt",[](const PeriodicGDFTiles& s){return s.source();})
        .def_property_readonly("source_identity",[](const PeriodicGDFTiles& s){return s.source().source_identity;})
        .def_property_readonly("metric_source_identity",[](const PeriodicGDFTiles& s){return s.source().metric_source_identity;})
        .def_property_readonly("three_center_source_identity",[](const PeriodicGDFTiles& s){return s.source().three_center_source_identity;})
        .def_property_readonly("orbital_basis_identity",[](const PeriodicGDFTiles& s){return s.source().short_range.orbital_basis_identity;})
        .def_property_readonly("auxiliary_basis_identity",[](const PeriodicGDFTiles& s){return s.source().short_range.auxiliary_basis_identity;})
        .def_property_readonly("shell_extents",[](const PeriodicGDFTiles& s){return std::vector<Count>{s.auxiliary().nshells(),s.orbital().nshells(),s.orbital().nshells()};})
        .def("metric",&PeriodicGDFTiles::metric)
        .def("evaluate",&PeriodicGDFTiles::evaluate,py::arg("auxiliary_shell"),py::arg("bra_shell"),py::arg("ket_shell"));
}
