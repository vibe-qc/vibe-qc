#include "vibeqc/manta_periodic_bubbles.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring>
namespace py=pybind11;
namespace {
using namespace vibeqc::manta;
void shape(const py::array& a,const py::dtype& dtype,int ndim) {
    if(!a.dtype().is(dtype) || a.ndim()!=ndim || !(a.flags()&(py::array::c_style|py::array::f_style)))
        throw py::value_error("MANTA periodic bubbles require contiguous arrays with the documented dtype and shape");
}
template<class T> T item(const py::array& a,py::ssize_t offset) {
    T x; std::memcpy(&x,static_cast<const char*>(a.data())+offset,sizeof(T)); return x;
}
py::array_t<double> owned(const md::Matrix& m) {
    py::array_t<double> out({static_cast<py::ssize_t>(m.rows()),static_cast<py::ssize_t>(m.cols())});
    auto v=out.mutable_unchecked<2>();
    for(Eigen::Index i=0;i<m.rows();++i) for(Eigen::Index j=0;j<m.cols();++j) v(i,j)=m(i,j);
    return out;
}
} // namespace
void bind_manta_periodic_bubbles(py::module_& m) {
    using namespace vibeqc::manta;
    auto p=py::class_<PeriodicBubblePlan>(m,"PeriodicBubblePlan");
#define P(n) p.def_readonly(#n,&PeriodicBubblePlan::n)
    P(geometry); P(pairs); P(maximum_nodes); P(moments); P(maximum_translation_calls);
    P(shell_calls); P(primitive_pairs); P(scalar_terms); P(kernel_terms); P(modeled_peak_bytes);
#undef P
    auto f=py::class_<PeriodicBubbleDensity>(m,"PeriodicBubbleDensity");
#define F(n) f.def_readonly(#n,&PeriodicBubbleDensity::n)
    F(pair_receipt); F(source_identity); F(density_identity); F(qualification);
    F(translation_calls); F(translation_terms); F(modeled_peak_bytes);
#undef F
    f.def_property_readonly("moments",[](const PeriodicBubbleDensity& d) { return owned(d.moments); })
        .def_property_readonly("full_moment_bounds",[](const PeriodicBubbleDensity& d) { return owned(d.full_moment_bounds); })
        .def_property_readonly("tail_moment_bounds",[](const PeriodicBubbleDensity& d) { return owned(d.tail_moment_bounds); })
        .def_property_readonly("coefficient_l1",[](const PeriodicBubbleDensity& d) { return std::vector<double>(d.coefficient_l1); });
    auto a=py::class_<PeriodicBubbleAdjoint>(m,"PeriodicBubbleAdjoint");
#define A(n) a.def_readonly(#n,&PeriodicBubbleAdjoint::n)
    A(pair); A(source_identity); A(potential_identity); A(translation_calls); A(translation_terms); A(modeled_peak_bytes);
#undef A
    a.def_property_readonly("values",[](const PeriodicBubbleAdjoint& a) {
        const auto n=static_cast<py::ssize_t>(a.pair.n_basis);
        py::array_t<std::complex<double>> out({static_cast<py::ssize_t>(a.pair.momenta),n,n});
        if(!a.pair.values.empty()) std::memcpy(out.mutable_data(),a.pair.values.data(),a.pair.values.size()*sizeof(std::complex<double>));
        return out;
    });
    m.def("plan_periodic_bubbles",&PeriodicBubbleTree::plan_source,py::arg("source"),
        py::arg("options")=BubbleOptions{},py::arg("budget")=ResourceBudget{});
    py::class_<PeriodicBubbleTree,std::shared_ptr<PeriodicBubbleTree>>(m,"PeriodicBubbleTree")
        .def(py::init([](std::shared_ptr<const PeriodicPairSource> source,const BubbleOptions& options,const ResourceBudget& budget) {
            const auto opts=options; const auto limits=budget;
            py::gil_scoped_release release; return std::make_shared<PeriodicBubbleTree>(std::move(source),opts,limits);
        }),py::arg("source"),py::arg("options")=BubbleOptions{},py::arg("budget")=ResourceBudget{})
        .def_property_readonly("plan",[](const PeriodicBubbleTree& t) { return PeriodicBubblePlan(t.plan()); })
        .def_property_readonly("nodes",[](const PeriodicBubbleTree& t) { return std::vector<BubbleNode>(t.nodes()); })
        .def_property_readonly("pair_order",[](const PeriodicBubbleTree& t) { return std::vector<Count>(t.pair_order()); })
        .def_property_readonly("source_identity",&PeriodicBubbleTree::source_identity)
        .def_property_readonly("pair_source_identity",&PeriodicBubbleTree::pair_source_identity)
        .def_property_readonly("depth",&PeriodicBubbleTree::depth)
        .def("traversal",&PeriodicBubbleTree::traversal,py::arg("opening")=.5,py::arg("budget")=ResourceBudget{})
        .def("contract",[](const PeriodicBubbleTree& t,const py::array& d) {
            shape(d,py::dtype::of<std::complex<double>>(),3); const auto& p=t.pair_plan();
            if(d.shape(0)!=static_cast<py::ssize_t>(p.momenta) || d.shape(1)!=static_cast<py::ssize_t>(p.n_basis) || d.shape(2)!=static_cast<py::ssize_t>(p.n_basis))
                throw py::value_error("MANTA periodic bubble density requires (momenta,AO,AO) shape");
            std::vector<std::complex<double>> values(p.momenta*p.n_basis*p.n_basis);
            for(Count k=0;k<p.momenta;++k) for(Count i=0;i<p.n_basis;++i) for(Count j=0;j<p.n_basis;++j)
                values[(k*p.n_basis+i)*p.n_basis+j]=item<std::complex<double>>(d,k*d.strides(0)+i*d.strides(1)+j*d.strides(2));
            py::gil_scoped_release release; return t.contract(values);
        },py::arg("densities").noconvert())
        .def("adjoint",[](const PeriodicBubbleTree& t,const py::array& a) {
            shape(a,py::dtype::of<double>(),2);
            if(a.shape(0)!=static_cast<py::ssize_t>(t.nodes().size()) || a.shape(1)!=static_cast<py::ssize_t>(t.plan().moments))
                throw py::value_error("MANTA periodic bubble potentials require (nodes,moments) shape");
            md::Matrix values(a.shape(0),a.shape(1));
            for(Eigen::Index i=0;i<values.rows();++i) for(Eigen::Index j=0;j<values.cols();++j) values(i,j)=item<double>(a,i*a.strides(0)+j*a.strides(1));
            py::gil_scoped_release release; return t.adjoint(values);
        },py::arg("node_potentials").noconvert());
}
