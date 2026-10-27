#include "vibeqc/translation_mp2.hpp"
#include "vibeqc/dlpno_ccsd.hpp"
#include <pybind11/functional.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_translation_mp2(py::module_& module) {
    py::class_<vibeqc::TranslationMP2Transport>(module, "_TranslationMP2Transport")
        .def_property_readonly("n_links", [](const vibeqc::TranslationMP2Transport& value) { return value.links.size(); })
        .def_property_readonly("n_couplings", [](const vibeqc::TranslationMP2Transport& value) { return value.couplings.size(); })
        .def_property_readonly("n_representatives", [](const vibeqc::TranslationMP2Transport& value) { return value.virtual_fock.size(); })
        .def_property_readonly("bounded", [](const vibeqc::TranslationMP2Transport& value) { return bool(value.recipes); })
        .def_readonly("numerical_link_bytes", &vibeqc::TranslationMP2Transport::numerical_link_bytes)
        .def_readonly("recipe_bytes", &vibeqc::TranslationMP2Transport::recipe_bytes)
        .def_readonly("metadata_bytes", &vibeqc::TranslationMP2Transport::metadata_bytes)
        .def_readonly("target_metric_cache_bytes", &vibeqc::TranslationMP2Transport::target_metric_cache_bytes)
        .def("link", &vibeqc::translation_transport_link)
        .def("cross_link", &vibeqc::translation_transport_cross_link)
        .def_readonly("compact", &vibeqc::TranslationMP2Transport::compact)
        .def_readonly("recipe_only", &vibeqc::TranslationMP2Transport::recipe_only)
        .def_readonly("cyclic_metric", &vibeqc::TranslationMP2Transport::cyclic_metric)
        .def_readonly("stable_links", &vibeqc::TranslationMP2Transport::stable_links)
        .def_readonly("retained_bytes", &vibeqc::TranslationMP2Transport::retained_bytes)
        .def_readonly("workspace_bytes", &vibeqc::TranslationMP2Transport::workspace_bytes)
        .def_readonly("required_bytes", &vibeqc::TranslationMP2Transport::required_bytes);
    const auto residual_record=[](const vibeqc::DLPNOCCSDResidualResult& r) {
        py::dict out;out["R1"]=r.R1;out["R2"]=r.R2;out["energy"]=r.energy;
        out["singles_terms"]=r.singles_terms;out["doubles_terms"]=r.doubles_terms;return out;
    };
    module.def("_translation_ccsd_workspace",&vibeqc::translation_ccsd_workspace);
    module.def("_translation_ccsd_evaluate",[residual_record](const vibeqc::TranslationMP2Transport& transport,
        const Eigen::MatrixXd& foo,int naux,int rank,const vibeqc::DLPNOCCOperand& operand,
        std::uint64_t maximum_bytes,bool terms) {
        const auto layout=vibeqc::translation_transport_layout(transport);
        const auto required=vibeqc::translation_ccsd_workspace(transport.n_occupied,naux,rank,layout.first.size(),terms);
        if(required>maximum_bytes) throw std::length_error("translation CCSD workspace exceeds maximum_bytes");
        for(const auto& f:transport.virtual_fock) if(f.rows()>rank)
            throw std::invalid_argument("CCSD rank bound smaller than admitted pair rank");
        vibeqc::translation_transport_validate_fock(transport,foo);
        // Operand callbacks reacquire Python; scientific contractions stay native.
        return residual_record(vibeqc::translation_ccsd_evaluate(foo,naux,rank,layout.bands,layout.first,layout.second,
            transport.energy_weights,operand,maximum_bytes,terms));
    });
    module.def("_dlpno_ccsd_pno_evaluate",[residual_record](const vibeqc::DLPNORowMatCRef& bov,
        const vibeqc::DLPNORowMatCRef& boo,const vibeqc::DLPNORowMatCRef& bvv,
        const Eigen::MatrixXd& foo,const Eigen::MatrixXd& fov,const Eigen::MatrixXd& fvv,
        const std::vector<int>& first,const std::vector<int>& second,
        const std::vector<Eigen::MatrixXd>& U,const std::vector<Eigen::VectorXd>& eps,
        const std::vector<Eigen::MatrixXd>& K,const std::vector<Eigen::MatrixXd>& T,
        const std::vector<std::vector<int>>& L,const std::vector<Eigen::VectorXd>& singles,bool terms) {
        const auto n=first.size();
        if(second.size()!=n || U.size()!=n || eps.size()!=n || K.size()!=n || T.size()!=n || L.size()!=n)
            throw std::invalid_argument("CCSD residual pair inventory");
        std::vector<vibeqc::DLPNOPairInput> pairs(n);
        for(std::size_t p=0;p<n;++p) pairs[p]={first[p],second[p],U[p],eps[p],K[p],T[p],L[p]};
        vibeqc::DLPNOCCSDResidualResult r;
        {py::gil_scoped_release release;
         r=vibeqc::dlpno_ccsd_pno_evaluate(bov,boo,bvv,foo,fov,fvv,std::move(pairs),singles,terms);}
        return residual_record(r);
    });
    const auto real_matrix = [](py::handle object) {
        const auto array = py::array::ensure(object);
        if (!array || array.ndim() != 2
            || (array.dtype().kind() != 'f' && array.dtype().kind() != 'i' && array.dtype().kind() != 'u'))
            throw std::invalid_argument("translation MP2 requires real numeric matrices");
        return array.cast<Eigen::MatrixXd>();
    };
    const auto real_matrices = [real_matrix](py::sequence objects) {
        std::vector<Eigen::MatrixXd> result;
        for (const auto object : objects) result.push_back(real_matrix(object));
        return result;
    };
    const auto preflight = [](bool bounded, std::uint64_t cap, py::sequence fock,
            py::sequence local, py::sequence projected, py::object home,
            py::object foo, py::sequence metrics) {
        if (!bounded) return;
        std::uint64_t bytes = 0;
        const auto add = [&](py::handle object) {
            if (!py::isinstance<py::array>(object))
                throw std::invalid_argument("bounded transport requires array inputs before conversion");
            const auto a = py::reinterpret_borrow<py::array>(object);
            if (!a || a.ndim()!=2) throw std::invalid_argument("invalid transport matrix");
            const auto entries = static_cast<std::uint64_t>(a.size());
            if (entries > (cap-std::min(cap,bytes))/8)
                throw std::length_error("bounded transport conversion exceeds maximum_bytes");
            bytes += entries*8;
        };
        for (auto x : fock) add(x);
        for (auto x : local) add(x);
        for (auto x : projected) add(x);
        for (auto x : metrics) add(x);
        if (!home.is_none()) add(home);
        add(foo);
        if (bytes > cap) throw std::length_error("bounded transport conversion exceeds maximum_bytes");
    };
    module.def("dlpno_prepare_translation_transport",
        [real_matrix, real_matrices, preflight](const std::vector<int>& mesh, int bands,
           const std::vector<int>& pair_i, const std::vector<int>& pair_j,
           py::sequence fock, py::sequence coefficients, py::object occupied_fock,
           py::object overlap, const std::vector<double>& weights, double tolerance,
           std::uint64_t maximum_bytes, const std::vector<std::vector<int>>& domains,
           py::sequence local, py::sequence projected, py::object all_occupied, bool occupied_home, bool bounded) {
            if (bounded && (!occupied_home || py::len(coefficients)))
                throw std::invalid_argument("bounded transport requires empty AO input and occupied home recipes");
            preflight(bounded,maximum_bytes,fock,local,projected,all_occupied,occupied_fock,py::make_tuple(overlap));
            const auto f = real_matrices(fock), v = real_matrices(coefficients);
            const auto foo = real_matrix(occupied_fock), s = real_matrix(overlap);
            const auto a = real_matrices(local), b = real_matrices(projected);
            const auto c = all_occupied.is_none() ? Eigen::MatrixXd() : real_matrix(all_occupied);
            py::gil_scoped_release release;
            return vibeqc::prepare_translation_mp2_transport(mesh, bands, pair_i, pair_j,
                f, v, foo, s, weights, tolerance, maximum_bytes, domains, a, b, c, occupied_home, bounded);
        },
        py::arg("mesh"), py::arg("n_bands"), py::arg("pair_i"), py::arg("pair_j"),
        py::arg("F_vv"), py::arg("V"), py::arg("F_oo"), py::arg("S_ao"),
        py::arg("energy_weights"), py::arg("symmetry_tolerance"), py::arg("maximum_bytes"),
        py::arg("domains") = std::vector<std::vector<int>>{},
        py::arg("local_coefficients") = py::tuple(), py::arg("occupied_coefficients") = py::tuple(),
        py::arg("all_occupied") = py::none(), py::arg("occupied_home") = false, py::arg("bounded") = false,
        "Prepare unique translated pair-space overlaps under an explicit transport byte budget.");
    module.def("dlpno_prepare_translation_transport_blocks",
        [real_matrix, real_matrices, preflight](const std::vector<int>& mesh, int bands,
           const std::vector<int>& pair_i, const std::vector<int>& pair_j,
           py::sequence fock, py::sequence coefficients, py::object occupied_fock,
           py::sequence overlap_blocks, const std::vector<double>& weights, double tolerance,
           std::uint64_t maximum_bytes, const std::vector<std::vector<int>>& domains,
           py::sequence local, py::sequence projected, py::object all_occupied, bool occupied_home, bool bounded) {
            if (bounded && (!occupied_home || py::len(coefficients)))
                throw std::invalid_argument("bounded transport requires empty AO input and occupied home recipes");
            preflight(bounded,maximum_bytes,fock,local,projected,all_occupied,occupied_fock,overlap_blocks);
            const auto f = real_matrices(fock), v = real_matrices(coefficients);
            const auto foo = real_matrix(occupied_fock);
            const auto s = real_matrices(overlap_blocks);
            const auto a = real_matrices(local), b = real_matrices(projected);
            const auto c = all_occupied.is_none() ? Eigen::MatrixXd() : real_matrix(all_occupied);
            py::gil_scoped_release release;
            return vibeqc::prepare_translation_mp2_transport_blocks(mesh, bands, pair_i, pair_j,
                f, v, foo, s, weights, tolerance, maximum_bytes, domains, a, b, c, occupied_home, bounded);
        },
        py::arg("mesh"), py::arg("n_bands"), py::arg("pair_i"), py::arg("pair_j"),
        py::arg("F_vv"), py::arg("V"), py::arg("F_oo"), py::arg("S_home_blocks"),
        py::arg("energy_weights"), py::arg("symmetry_tolerance"), py::arg("maximum_bytes"),
        py::arg("domains") = std::vector<std::vector<int>>{},
        py::arg("local_coefficients") = py::tuple(), py::arg("occupied_coefficients") = py::tuple(),
        py::arg("all_occupied") = py::none(), py::arg("occupied_home") = false, py::arg("bounded") = false,
        "Prepare translated pair overlaps from cyclic metric blocks without dense AO expansion.");
    module.def("dlpno_translation_mp2_solve_prepared",
        [real_matrices](const vibeqc::TranslationMP2Transport& transport,
           py::sequence integrals, py::sequence seeds, int maximum_iterations,
           double energy_tolerance, double residual_tolerance, double damping, bool work) -> py::tuple {
            const auto k = real_matrices(integrals), t = real_matrices(seeds);
            vibeqc::TranslationMP2Result result;
            {
                py::gil_scoped_release release;
                result = vibeqc::translation_mp2_solve_prepared(transport, k, t,
                    maximum_iterations, energy_tolerance, residual_tolerance, damping);
            }
            py::list trace;
            for (const auto& step : result.trace)
                trace.append(py::make_tuple(step.iteration, step.energy, step.energy_change, step.maximum_residual));
            if (work) return py::make_tuple(result.amplitudes, result.residuals, result.converged, trace,
                py::make_tuple(result.link_constructions,result.link_cache_hits,result.link_evictions,
                    result.target_metric_constructions,result.target_metric_cache_hits,result.target_metric_evictions));
            return py::make_tuple(result.amplitudes, result.residuals, result.converged, trace);
        },
        py::arg("transport"), py::arg("K"), py::arg("T"), py::arg("maximum_iterations"),
        py::arg("energy_tolerance"), py::arg("residual_tolerance"), py::arg("damping"), py::arg("work") = false,
        "Solve with prepared links or bounded recipe access using the same compiled iteration.");
    module.def("dlpno_translation_mp2_solve",
        [](const std::vector<int>& mesh, int bands,
           const std::vector<int>& pair_i, const std::vector<int>& pair_j,
           py::sequence integrals, py::sequence fock, py::sequence seeds,
           py::sequence coefficients, py::object occupied_fock, py::object overlap,
           const std::vector<double>& weights, int maximum_iterations,
           double energy_tolerance, double residual_tolerance, double damping,
           double symmetry_tolerance) {
            const auto matrix = [](py::handle object) {
                const auto array = py::array::ensure(object);
                if (!array || array.ndim() != 2
                    || (array.dtype().kind() != 'f' && array.dtype().kind() != 'i'
                        && array.dtype().kind() != 'u'))
                    throw std::invalid_argument("translation MP2 requires real numeric matrices");
                return array.cast<Eigen::MatrixXd>();
            };
            const auto matrices = [&](py::sequence objects) {
                std::vector<Eigen::MatrixXd> result;
                for (const auto object : objects) result.push_back(matrix(object));
                return result;
            };
            const auto k = matrices(integrals), f = matrices(fock), t = matrices(seeds), v = matrices(coefficients);
            const auto foo = matrix(occupied_fock), s = matrix(overlap);
            vibeqc::TranslationMP2Result result;
            {
                py::gil_scoped_release release;
                result = vibeqc::translation_mp2_solve(mesh, bands, pair_i, pair_j,
                    k, f, t, v, foo, s, weights, maximum_iterations,
                    energy_tolerance, residual_tolerance, damping, symmetry_tolerance);
            }
            py::list trace;
            for (const auto& step : result.trace)
                trace.append(py::make_tuple(step.iteration, step.energy,
                                             step.energy_change, step.maximum_residual));
            return py::make_tuple(result.amplitudes, result.residuals, result.converged, trace);
        },
        py::arg("mesh"), py::arg("n_bands"), py::arg("pair_i"), py::arg("pair_j"),
        py::arg("K"), py::arg("F_vv"), py::arg("T"), py::arg("V"),
        py::arg("F_oo"), py::arg("S_ao"), py::arg("energy_weights"),
        py::arg("maximum_iterations"), py::arg("energy_tolerance"),
        py::arg("residual_tolerance"), py::arg("damping"), py::arg("symmetry_tolerance"),
        "Experimental representative finite-torus MP2 Jacobi solve; raw snapshot convergence.");
}
