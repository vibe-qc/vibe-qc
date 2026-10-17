#include "vibeqc/manta_periodic_semilocal_xc.hpp"
#include "vibeqc/manta_periodic_xc_refinement.hpp"
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
        throw py::value_error("MANTA semilocal XC requires contiguous float64 source arrays and complex128 D[spin,k,AO,AO]");
}
Count vectors(const py::array& a) {
    check(a,py::dtype::of<double>(),2);
    if(a.shape(1)!=3) throw py::value_error("MANTA semilocal XC vectors require shape (count,3)");
    return a.shape(0);
}
template<class T> T item(const py::array& a,py::ssize_t offset) {
    T out; std::memcpy(&out,static_cast<const char*>(a.data())+offset,sizeof(T)); return out;
}
Eigen::MatrixXd copy(const py::array& a) {
    Eigen::MatrixXd out(a.shape(0),a.shape(1));
    for(Eigen::Index i=0;i<out.rows();++i) for(Eigen::Index j=0;j<out.cols();++j)
        out(i,j)=item<double>(a,i*a.strides(0)+j*a.strides(1));
    return out;
}
std::vector<PeriodicQuadratureOptions> grid_list(const py::sequence& values) {
    if((!py::isinstance<py::list>(values) && !py::isinstance<py::tuple>(values)) || values.size()<2 || values.size()>32)
        throw py::value_error("MANTA XC refinement requires a list/tuple of 2..32 grid options");
    std::vector<PeriodicQuadratureOptions> out;out.reserve(values.size());
    for(const auto& value:values) out.push_back(py::cast<PeriodicQuadratureOptions>(value));
    return out;
}
}
void bind_manta_periodic_semilocal_xc(py::module_& m) {
    py::class_<PeriodicSemilocalXCPlan,PeriodicLocalExchangePlan>(m,"PeriodicSemilocalXCPlan")
        .def_readonly("gradients",&PeriodicSemilocalXCPlan::gradients)
        .def_readonly("functional_components",&PeriodicSemilocalXCPlan::functional_components)
        .def_readonly("functional_workspace_bytes",&PeriodicSemilocalXCPlan::functional_workspace_bytes);
    py::class_<PeriodicSemilocalXCReceipt,PeriodicLocalExchangeReceipt>(m,"PeriodicSemilocalXCReceipt")
        .def_readonly("semilocal_component_only",&PeriodicSemilocalXCReceipt::semilocal_component_only)
        .def_readonly("required_exact_exchange_fraction",&PeriodicSemilocalXCReceipt::required_exact_exchange_fraction)
        .def_readonly("required_exchange_components",&PeriodicSemilocalXCReceipt::required_exchange_components)
        .def_readonly("backend_version",&PeriodicSemilocalXCReceipt::backend_version)
        .def_readonly("component_ids",&PeriodicSemilocalXCReceipt::component_ids)
        .def_readonly("component_weights",&PeriodicSemilocalXCReceipt::component_weights)
        .def_readonly("functional_batches",&PeriodicSemilocalXCReceipt::functional_batches);
    py::class_<PeriodicSemilocalXCResult>(m,"PeriodicSemilocalXCResult")
        .def_readonly("plan",&PeriodicSemilocalXCResult::plan)
        .def_readonly("receipt",&PeriodicSemilocalXCResult::receipt)
        .def_property_readonly("values",[](const PeriodicSemilocalXCResult& r) {
            py::array_t<Complex> out({static_cast<py::ssize_t>(r.plan.spins),static_cast<py::ssize_t>(r.plan.momenta),
                static_cast<py::ssize_t>(r.plan.n_basis),static_cast<py::ssize_t>(r.plan.n_basis)});
            std::memcpy(out.mutable_data(),r.values.data(),r.values.size()*sizeof(Complex)); return out;
        });
    m.def("plan_periodic_semilocal_xc",&plan_periodic_semilocal_xc,
        py::arg("basis"),py::arg("functional"),py::arg("images"),py::arg("momenta"),py::arg("grid_points"),
        py::arg("grid_batch")=64,py::arg("budget")=ResourceBudget{});
    m.def("compute_periodic_semilocal_xc",[](const vibeqc::BasisSet& basis,const vibeqc::Functional& functional,const py::array& lattice,
        const py::array& images,const py::array& k,const py::array& weights,const py::array& points,
        const py::array& qw,const py::array& d,Count batch,const ResourceBudget& budget) {
        using namespace admission;
        if(vectors(lattice)!=3) throw py::value_error("MANTA semilocal XC lattice requires shape (3,3)");
        const Count ni=vectors(images),nk=vectors(k),ng=vectors(points),n=basis.nbasis();
        check(weights,py::dtype::of<double>(),1); check(qw,py::dtype::of<double>(),1); check(d,py::dtype::of<Complex>(),4);
        if(weights.shape(0)!=static_cast<py::ssize_t>(nk) || qw.shape(0)!=static_cast<py::ssize_t>(ng)
           || d.shape(1)!=static_cast<py::ssize_t>(nk) || d.shape(2)!=static_cast<py::ssize_t>(n) || d.shape(3)!=static_cast<py::ssize_t>(n))
            throw py::value_error("MANTA semilocal XC source extents disagree");
        const Count ns=d.shape(0);
        if(ns!=static_cast<Count>(functional.spin())) throw py::value_error("MANTA semilocal XC density spin and functional disagree");
        plan_periodic_semilocal_xc(basis,functional,ni,nk,ng,batch,budget);
        // Count-only admission precedes every numerical copy. Keep the GIL:
        // these borrowed Python owners cannot be cooperatively mutated mid-call.
        const auto a=copy(lattice),t=copy(images),momenta=copy(k),grid=copy(points);
        std::vector<double> w(nk),q(ng); std::vector<Complex> density(mul(mul(ns,nk),square(n)));
        for(Count i=0;i<nk;++i) w[i]=item<double>(weights,i*weights.strides(0));
        for(Count i=0;i<ng;++i) q[i]=item<double>(qw,i*qw.strides(0));
        for(Count spin=0;spin<ns;++spin) for(Count ik=0;ik<nk;++ik) for(Count i=0;i<n;++i) for(Count j=0;j<n;++j)
            density[((spin*nk+ik)*n+i)*n+j]=item<Complex>(d,spin*d.strides(0)+ik*d.strides(1)+i*d.strides(2)+j*d.strides(3));
        return compute_periodic_semilocal_xc(basis,functional,a,t,momenta,w,grid,q,density,batch,budget);
    },py::arg("basis"),py::arg("functional"),py::arg("lattice").noconvert(),py::arg("images").noconvert(),py::arg("momenta").noconvert(),
      py::arg("weights").noconvert(),py::arg("points").noconvert(),py::arg("quadrature_weights").noconvert(),
      py::arg("densities").noconvert(),py::arg("grid_batch")=64,py::arg("budget")=ResourceBudget{});
}

void bind_manta_periodic_xc_refinement(py::module_& m) {
    auto p=py::class_<PeriodicXCRefinementPlan>(m,"PeriodicXCRefinementPlan");
#define P(name) p.def_readonly(#name,&PeriodicXCRefinementPlan::name)
    P(quadrature);P(xc);P(spins);P(momenta);P(n_basis);P(reference_index);P(driver_live_bytes);
    P(modeled_peak_bytes);P(shell_calls);P(primitive_points);P(scalar_terms);P(kernel_terms);
    P(semilocal_component_only);
#undef P
    auto s=py::class_<PeriodicXCRefinementStep>(m,"PeriodicXCRefinementStep");
    s.def_property_readonly("options",[](const PeriodicXCRefinementStep& s){return s.options;});
#define S(name) s.def_readonly(#name,&PeriodicXCRefinementStep::name)
    S(quadrature);S(xc);S(energy_difference);S(potential_max_abs);S(potential_frobenius);S(population_differences);
#undef S
    auto r=py::class_<PeriodicXCRefinementResult>(m,"PeriodicXCRefinementResult");
    r.def_property_readonly("reference_potential",[](const PeriodicXCRefinementResult& r) {
        py::array_t<Complex> out({static_cast<py::ssize_t>(r.plan.spins),static_cast<py::ssize_t>(r.plan.momenta),
            static_cast<py::ssize_t>(r.plan.n_basis),static_cast<py::ssize_t>(r.plan.n_basis)});
        std::memcpy(out.mutable_data(),r.reference_values.data(),r.reference_values.size()*sizeof(Complex));return out;
    });
#define R(name) r.def_readonly(#name,&PeriodicXCRefinementResult::name)
    R(plan);R(steps);R(source_identity);R(density_identity);R(comparison);R(shell_calls);R(primitive_points);
#undef R
    for(bool hybrid:{false,true}) {
    m.def(hybrid?"plan_periodic_hybrid_xc_refinement":"plan_periodic_xc_refinement",[hybrid](const vibeqc::BasisSet& basis,const vibeqc::PeriodicSystem& system,
        const vibeqc::Functional& f,Count images,Count momenta,const py::sequence& grids,Count reference,Count batch,const ResourceBudget& budget) {
        const auto planner=hybrid?plan_periodic_hybrid_xc_refinement:plan_periodic_xc_refinement;
        return planner(basis,system,f,images,momenta,grid_list(grids),reference,batch,budget);
    },py::arg("basis"),py::arg("system"),py::arg("functional"),py::arg("images"),py::arg("momenta"),
      py::arg("grid_options").noconvert(),py::arg("reference_index"),py::arg("grid_batch")=64,py::arg("budget")=ResourceBudget{});
    m.def(hybrid?"assess_periodic_hybrid_xc_refinement":"assess_periodic_xc_refinement",[hybrid](const vibeqc::BasisSet& basis,const vibeqc::PeriodicSystem& system,
        const vibeqc::Functional& f,const py::array& images,const py::array& k,const py::array& weights,
        const py::array& d,const py::sequence& grids,Count reference,Count batch,const ResourceBudget& budget) {
        using namespace admission;
        const Count ni=vectors(images),nk=vectors(k),n=basis.nbasis(),ns=f.spin();
        check(weights,py::dtype::of<double>(),1);check(d,py::dtype::of<Complex>(),4);
        if(weights.shape(0)!=static_cast<py::ssize_t>(nk) || d.shape(0)!=static_cast<py::ssize_t>(ns)
           || d.shape(1)!=static_cast<py::ssize_t>(nk) || d.shape(2)!=static_cast<py::ssize_t>(n) || d.shape(3)!=static_cast<py::ssize_t>(n))
            throw py::value_error("MANTA XC refinement spin/k/density extents disagree");
        const auto controls=grid_list(grids);
        const auto planner=hybrid?plan_periodic_hybrid_xc_refinement:plan_periodic_xc_refinement;
        planner(basis,system,f,ni,nk,controls,reference,batch,budget);
        // Joint count-only admission precedes every numerical array copy.
        // Keep the GIL; input mutation during this synchronous call is invalid.
        const auto t=copy(images),momenta=copy(k);std::vector<double> w(nk);
        std::vector<Complex> density(mul(mul(ns,nk),square(n)));
        for(Count ik=0;ik<nk;++ik) w[ik]=item<double>(weights,ik*weights.strides(0));
        for(Count spin=0;spin<ns;++spin) for(Count ik=0;ik<nk;++ik) for(Count i=0;i<n;++i) for(Count j=0;j<n;++j)
            density[((spin*nk+ik)*n+i)*n+j]=item<Complex>(d,spin*d.strides(0)+ik*d.strides(1)+i*d.strides(2)+j*d.strides(3));
        const auto assess=hybrid?assess_periodic_hybrid_xc_refinement:assess_periodic_xc_refinement;
        return assess(basis,system,f,t,momenta,w,density,controls,reference,batch,budget);
    },py::arg("basis"),py::arg("system"),py::arg("functional"),py::arg("images").noconvert(),
      py::arg("momenta").noconvert(),py::arg("weights").noconvert(),py::arg("densities").noconvert(),
      py::arg("grid_options").noconvert(),py::arg("reference_index"),py::arg("grid_batch")=64,py::arg("budget")=ResourceBudget{});
    }
}
