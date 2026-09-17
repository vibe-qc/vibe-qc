#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "vibeqc/opentrustregion.hpp"
#include "vibeqc/orbital_objective.hpp"
namespace py = pybind11;
using namespace vibeqc;
namespace {
class CallbackObjective final : public OrbitalObjective {
    int n_;
    py::function update_, trial_, hessian_;
    double tol_, rms_ = 1;
public:
    CallbackObjective(int n, py::function update, py::function trial, py::function hessian, double tol)
      : n_(n), update_(std::move(update)), trial_(std::move(trial)), hessian_(std::move(hessian)), tol_(tol) {}
    int size() const override { return n_; }
    OrbitalModel update(const Eigen::VectorXd& k) override {
        py::gil_scoped_acquire gil;
        const auto t = update_(k).cast<py::tuple>();
        OrbitalModel m{t[0].cast<double>(),t[1].cast<Eigen::VectorXd>(),t[2].cast<Eigen::VectorXd>()};
        rms_ = n_ ? m.gradient.norm()/std::sqrt(double(n_)) : 0;
        return m;
    }
    double trial(const Eigen::VectorXd& k) override {
        py::gil_scoped_acquire gil; return trial_(k).cast<double>();
    }
    Eigen::VectorXd hessian(const Eigen::VectorXd& k) override {
        py::gil_scoped_acquire gil; return hessian_(k).cast<Eigen::VectorXd>();
    }
    bool converged() const override { return rms_ < tol_; }
};
}
void bind_opentrustregion(py::module_& m) {
    py::class_<OpenTrustRegionOptions>(m,"OpenTrustRegionOptions")
        .def(py::init<>())
        .def_readwrite("subsystem_solver",&OpenTrustRegionOptions::subsystem_solver)
        .def_readwrite("gradient_rms_tolerance",&OpenTrustRegionOptions::gradient_rms_tolerance)
        .def_readwrite("initial_trust_radius",&OpenTrustRegionOptions::initial_trust_radius)
        .def_readwrite("max_micro_iterations",&OpenTrustRegionOptions::max_micro_iterations)
        .def_readwrite("seed",&OpenTrustRegionOptions::seed)
        .def_readwrite("line_search",&OpenTrustRegionOptions::line_search)
        .def_readwrite("stability",&OpenTrustRegionOptions::stability)
        .def_readwrite("stability_max_iterations",&OpenTrustRegionOptions::stability_max_iterations)
        .def_readwrite("stability_tolerance",&OpenTrustRegionOptions::stability_tolerance)
        .def_readwrite("stability_residual_tolerance",&OpenTrustRegionOptions::stability_residual_tolerance);
    py::class_<OpenTrustRegionReport>(m,"OpenTrustRegionReport")
        .def_readonly("backend",&OpenTrustRegionReport::backend)
        .def_readonly("version",&OpenTrustRegionReport::version)
        .def_readonly("subsystem_solver",&OpenTrustRegionReport::subsystem_solver)
        .def_readonly("stability_policy",&OpenTrustRegionReport::stability_policy)
        .def_readonly("manifold",&OpenTrustRegionReport::manifold)
        .def_readonly("termination",&OpenTrustRegionReport::termination)
        .def_readonly("callback_error",&OpenTrustRegionReport::callback_error)
        .def_readonly("error_code",&OpenTrustRegionReport::error_code)
        .def_readonly("accepted_evaluations",&OpenTrustRegionReport::accepted_evaluations)
        .def_readonly("trial_evaluations",&OpenTrustRegionReport::trial_evaluations)
        .def_readonly("response_evaluations",&OpenTrustRegionReport::response_evaluations)
        .def_readonly("fock_evaluations",&OpenTrustRegionReport::fock_evaluations)
        .def_readonly("macro_iterations",&OpenTrustRegionReport::macro_iterations)
        .def_readonly("micro_iterations",&OpenTrustRegionReport::micro_iterations)
        .def_readonly("gradient_rms",&OpenTrustRegionReport::gradient_rms)
        .def_readonly("final_residual",&OpenTrustRegionReport::final_residual)
        .def_readonly("energy_converged",&OpenTrustRegionReport::energy_converged)
        .def_readonly("gradient_converged",&OpenTrustRegionReport::gradient_converged)
        .def_readonly("stability_checked",&OpenTrustRegionReport::stability_checked)
        .def_readonly("stability_converged",&OpenTrustRegionReport::stability_converged)
        .def_readonly("stable",&OpenTrustRegionReport::stable)
        .def_readonly("stability_threshold",&OpenTrustRegionReport::stability_threshold)
        .def_readonly("log",&OpenTrustRegionReport::log);
    m.def("has_opentrustregion",&has_opentrustregion);
    // Private validation surface: the production C++ adapter can be exercised
    // with independent analytic objectives, exceptions and concurrent calls.
    m.def("_otr_minimize",[](int n, py::function update, py::function trial,
        py::function hessian, const OpenTrustRegionOptions& o, int max_iter) {
        CallbackObjective objective(n,std::move(update),std::move(trial),std::move(hessian),o.gradient_rms_tolerance);
        OpenTrustRegionReport report;
        { py::gil_scoped_release release; report = optimize_orbitals(objective,o,max_iter); }
        return report;
    });
    py::class_<OrbitalModel>(m,"_OrbitalModel")
        .def_readonly("energy",&OrbitalModel::energy)
        .def_readonly("gradient",&OrbitalModel::gradient)
        .def_readonly("diagonal",&OrbitalModel::diagonal);
    py::class_<MolecularOrbitalObjective>(m,"_MolecularOrbitalObjective")
        .def(py::init([](const Eigen::MatrixXd& s, const Eigen::MatrixXd& h,
            double nuclear, const JKBuilder& jk, const Eigen::MatrixXd& ca,
            const Eigen::MatrixXd& cb, int na, int nb, bool restricted,
            double exchange, const BasisSet& basis, const Grid& grid, const std::string& functional) {
            OrbitalXCFunction xc;
            if (!functional.empty()) xc = restricted ? make_orbital_rks_xc(basis,grid,functional)
                                                     : make_orbital_uks_xc(basis,grid,functional);
            return std::make_unique<MolecularOrbitalObjective>(s,h,nuclear,jk,ca,cb,na,nb,restricted,
                exchange,std::move(xc),1e-8,1e-6,1e-8);
        }),py::keep_alive<1,5>())
        .def("update",&MolecularOrbitalObjective::update)
        .def("trial",&MolecularOrbitalObjective::trial)
        .def("hessian",&MolecularOrbitalObjective::hessian)
        .def("size",&MolecularOrbitalObjective::size);
}
