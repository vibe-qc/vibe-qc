#include "vibeqc/manta.hpp"
#include "vibeqc/manta_provider.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py=pybind11;
namespace {
using namespace vibeqc::manta;
std::shared_ptr<const IntegralProvider> provider(const vibeqc::BasisSet& basis,const std::string& backend,
                                              const KernelRequest& request,const ResourceBudget& budget) {
    if(backend=="manta") return std::make_shared<MantaProvider>(basis,request,budget,1,true);
    if(backend=="libint") return std::make_shared<LibintProvider>(basis,request,budget,true);
    throw std::invalid_argument("MANTA tile backend must be manta or libint");
}
struct ShellTileSession {
    std::shared_ptr<const IntegralProvider> source;
    Workspace workspace;
    Count calls=0;
    ShellTileSession(const vibeqc::BasisSet& basis,const std::string& backend,const KernelRequest& request,const ResourceBudget& budget)
        :source(provider(basis,backend,request,budget)),workspace(source->plan()) {}
    py::dict evaluate(const std::array<std::size_t,4>& indices) {
        const auto& shells=source->basis().libint(); std::vector<py::ssize_t> shape;
        Count primitives=1;
        for(auto i:indices) {
            if(i>=shells.size()) throw py::index_error("MANTA shell-tile index");
            shape.push_back(shells[i].size()); primitives*=shells[i].alpha.size();
        }
        const auto tile=source->evaluate(indices,workspace); ++calls;
        py::array_t<double> values(shape);
        if(tile.values) std::copy_n(tile.values,values.size(),values.mutable_data());
        else std::fill_n(values.mutable_data(),values.size(),0.);
        py::dict result;
        result["values"]=std::move(values); result["provider"]=source->name();
        result["source_identity"]=source->source_identity(); result["shell_calls"]=1;
        result["null_tiles"]=tile.values?0:1; result["primitive_quartets_requested"]=primitives;
        result["primitive_evaluations_measured"]=source->name()=="manta";
        result["primitive_quartets_evaluated"]=source->name()=="manta"?primitives:0;
        result["modeled_peak_bytes"]=source->plan().modeled_peak_bytes;
        result["kernel"]=source->kernel_request().kernel; result["omega"]=source->kernel_request().omega;
        return result;
    }
};
} // namespace
void bind_manta_tiles(py::module_& m) {
    using namespace vibeqc::manta;
    m.def("plan_shell_tiles",[](const vibeqc::BasisSet& b,const std::string& backend,const KernelRequest& request,const ResourceBudget& budget) {
        if(backend=="manta") return plan_md_tiles(b,request,budget);
        if(backend=="libint") return plan_libint_tiles(b,request,budget);
        throw std::invalid_argument("MANTA tile backend must be manta or libint");
    },py::arg("basis"),py::arg("backend")="manta",py::arg("request")=KernelRequest{},py::arg("budget")=ResourceBudget{});
    py::class_<ShellTileSession>(m,"ShellTileSession")
        .def(py::init<const vibeqc::BasisSet&,const std::string&,const KernelRequest&,const ResourceBudget&>(),
            py::arg("basis"),py::arg("backend")="manta",py::arg("request")=KernelRequest{},py::arg("budget")=ResourceBudget{})
        .def("evaluate",&ShellTileSession::evaluate)
        .def_readonly("calls",&ShellTileSession::calls)
        .def_property_readonly("plan",[](const ShellTileSession& s) { return BuildPlan(s.source->plan()); })
        .def_property_readonly("source_identity",[](const ShellTileSession& s) { return s.source->source_identity(); });
}
