#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "vibeqc/periodic_gdf_short_range.hpp"

namespace py = pybind11;

namespace {
py::array_t<std::complex<double>> gdf_metric_array(Eigen::MatrixXcd value) {
    using Complex = std::complex<double>;
    auto* owner = new Eigen::MatrixXcd(std::move(value));
    py::capsule lifetime(owner, [](void* p) { delete static_cast<Eigen::MatrixXcd*>(p); });
    const auto na = static_cast<py::ssize_t>(owner->rows());
    const auto item = static_cast<py::ssize_t>(sizeof(Complex));
    return py::array_t<Complex>({na, na}, {item, na * item}, owner->data(), lifetime);
}

py::array_t<std::complex<double>> gdf_sr_batch_array(vibeqc::GDFShortRangeBatch batch) {
    using Complex = std::complex<double>;
    const auto nk = static_cast<py::ssize_t>(batch.n_k);
    const auto na = static_cast<py::ssize_t>(batch.n_aux);
    const auto no = static_cast<py::ssize_t>(batch.n_orb);
    const auto item = static_cast<py::ssize_t>(sizeof(Complex));
    auto* owner = new std::vector<Complex>(std::move(batch.data));
    py::capsule lifetime(owner, [](void* p) { delete static_cast<std::vector<Complex>*>(p); });
    return py::array_t<Complex>({nk, na, no, no},
        {na * no * no * item, no * no * item, no * item, item}, owner->data(), lifetime);
}
}

void bind_periodic_gdf_short_range(py::module_& m) {
    m.def("_libint_boys_values", [](double argument, int maximum_order) {
        if (!std::isfinite(argument) || argument < 0.0
            || maximum_order < 0 || maximum_order > 24)
            throw std::invalid_argument("Boys diagnostic requires finite T>=0 and order 0..24");
        std::vector<double> values(static_cast<std::size_t>(maximum_order + 1));
        libint2::FmEval_Chebyshev7<double>::instance(maximum_order)->eval(
            values.data(), argument, maximum_order);
        return values;
    }, py::arg("argument"), py::arg("maximum_order"),
       "Bounded diagnostic of the linked libint C++ Boys evaluator.");
    m.def("gdf_short_range_workspace_bytes", &vibeqc::gdf_short_range_workspace_bytes,
          py::arg("orbital"), py::arg("aux"), py::arg("n_threads"),
          py::arg("derivative_order") = 0, py::arg("n_atoms") = 0,
          py::call_guard<py::gil_scoped_release>(),
          "Linked-kernel SR engine workspace for the requested task-limited worker team.");
    m.def("compute_gdf_sr_metric", &vibeqc::compute_gdf_sr_metric,
          py::arg("aux"), py::arg("system"), py::arg("q"),
          py::arg("omega"), py::arg("cutoff"),
          py::arg("output_byte_cap"), py::arg("image_candidate_cap"),
          py::arg("workspace_byte_cap") = 256U * 1024U * 1024U,
          py::arg("integral_screen_error") = 0.0,
          py::call_guard<py::gil_scoped_release>(),
          "Raw erfc periodic metric; includes its finite zero mode.");
    m.def("compute_gdf_sr_three_center",
          [](const vibeqc::BasisSet& orbital, const vibeqc::BasisSet& aux,
             const vibeqc::PeriodicSystem& system, const Eigen::Vector3d& q,
             const Eigen::MatrixXd& ket_kpoints, double omega,
             double pair_cutoff, double auxiliary_cutoff,
             std::size_t output_byte_cap, std::size_t image_candidate_cap,
             std::size_t workspace_byte_cap, double integral_screen_error) {
              vibeqc::GDFShortRangeBatch batch;
              {
                  py::gil_scoped_release release;
                  batch = vibeqc::compute_gdf_sr_three_center(
                      orbital, aux, system, q, ket_kpoints, omega,
                      pair_cutoff, auxiliary_cutoff, output_byte_cap,
                      image_candidate_cap, workspace_byte_cap, integral_screen_error);
              }
              return gdf_sr_batch_array(std::move(batch));
          }, py::arg("orbital"), py::arg("aux"), py::arg("system"),
          py::arg("q"), py::arg("ket_kpoints"), py::arg("omega"),
          py::arg("pair_cutoff"), py::arg("auxiliary_cutoff"),
          py::arg("output_byte_cap"), py::arg("image_candidate_cap"),
          py::arg("workspace_byte_cap") = 256U * 1024U * 1024U,
          py::arg("integral_screen_error") = 0.0,
          "Double-image erfc integrals in shared-q batches, shape (nk,naux,nao,nao).");
    m.def("compute_gdf_sr_metric_gradient_weighted",
          &vibeqc::compute_gdf_sr_metric_gradient_weighted,
          py::arg("aux"), py::arg("system"), py::arg("q"),
          py::arg("omega"), py::arg("cutoff"), py::arg("weight").noconvert(),
          py::arg("output_byte_cap"), py::arg("image_candidate_cap"),
          py::arg("workspace_byte_cap"), py::arg("integral_screen_error") = 0.0,
          py::call_guard<py::gil_scoped_release>(),
          "Atomic derivative of Re sum(weight * SR metric); borrow complex128 column-major weights.");
    m.def("compute_gdf_sr_three_center_gradient_weighted",
          [](const vibeqc::BasisSet& orbital, const vibeqc::BasisSet& aux,
             const vibeqc::PeriodicSystem& system, const Eigen::Vector3d& q,
             const Eigen::MatrixXd& ket_kpoints, double omega,
             double pair_cutoff, double auxiliary_cutoff,
             const py::array_t<std::complex<double>, py::array::c_style>& weight,
             std::size_t output_byte_cap, std::size_t image_candidate_cap,
             std::size_t workspace_byte_cap, double integral_screen_error) {
              if (weight.ndim() != 4 || weight.shape(2) != weight.shape(3))
                  throw std::invalid_argument("GDF SR derivative weight must have shape (nk,naux,nao,nao)");
              const vibeqc::GDFShortRangeWeightView view{
                  weight.data(), static_cast<std::size_t>(weight.shape(0)),
                  static_cast<std::size_t>(weight.shape(1)),
                  static_cast<std::size_t>(weight.shape(2))};
              py::gil_scoped_release release;
              return vibeqc::compute_gdf_sr_three_center_gradient_weighted(
                  orbital, aux, system, q, ket_kpoints, omega,
                  pair_cutoff, auxiliary_cutoff, view, output_byte_cap,
                  image_candidate_cap, workspace_byte_cap, integral_screen_error);
          }, py::arg("orbital"), py::arg("aux"), py::arg("system"),
          py::arg("q"), py::arg("ket_kpoints"), py::arg("omega"),
          py::arg("pair_cutoff"), py::arg("auxiliary_cutoff"),
          py::arg("weight").noconvert(), py::arg("output_byte_cap"),
          py::arg("image_candidate_cap"), py::arg("workspace_byte_cap"),
          py::arg("integral_screen_error") = 0.0,
          "Atomic derivative of Re sum(weight * SR tensor); borrow complex128 row-major weights.");
    m.def("compute_gdf_range_separated_gradient_weighted",
          [](const vibeqc::BasisSet& orbital, const vibeqc::BasisSet& aux,
             const vibeqc::PeriodicSystem& system, const Eigen::Vector3d& q,
             const Eigen::MatrixXd& ket_kpoints,
             const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& vectors,
             double omega, double pair_cutoff, double auxiliary_cutoff,
             const Eigen::Ref<const Eigen::MatrixXcd>& metric_weight,
             const py::array_t<std::complex<double>, py::array::c_style>& three_center_weight,
             std::size_t output_byte_cap, std::size_t transient_byte_cap,
             std::size_t image_candidate_cap, double integral_screen_error) {
              if (three_center_weight.ndim() != 4
                  || three_center_weight.shape(2) != three_center_weight.shape(3))
                  throw std::invalid_argument("RSGDF derivative weight must have shape (nk,naux,nao,nao)");
              const vibeqc::GDFShortRangeWeightView view{
                  three_center_weight.data(), static_cast<std::size_t>(three_center_weight.shape(0)),
                  static_cast<std::size_t>(three_center_weight.shape(1)),
                  static_cast<std::size_t>(three_center_weight.shape(2))};
              py::gil_scoped_release release;
              return vibeqc::compute_gdf_range_separated_gradient_weighted(
                  orbital, aux, system, q, ket_kpoints, vectors, omega,
                  pair_cutoff, auxiliary_cutoff, metric_weight, view,
                  output_byte_cap, transient_byte_cap, image_candidate_cap, integral_screen_error);
          }, py::arg("orbital"), py::arg("aux"), py::arg("system"),
          py::arg("q"), py::arg("ket_kpoints"), py::arg("vectors"),
          py::arg("omega"), py::arg("pair_cutoff"), py::arg("auxiliary_cutoff"),
          py::arg("metric_weight").noconvert(), py::arg("three_center_weight").noconvert(),
          py::arg("output_byte_cap"), py::arg("transient_byte_cap"), py::arg("image_candidate_cap"),
          py::arg("integral_screen_error") = 0.0,
          "Atomic derivative of weighted SR/LR Coulomb integrals, including the finite zero-mode overlap.");
    m.def("compute_gdf_range_separated_integrals",
          [](const vibeqc::BasisSet& orbital, const vibeqc::BasisSet& aux,
             const vibeqc::PeriodicSystem& system, const Eigen::Vector3d& q,
             const Eigen::MatrixXd& ket_kpoints,
             const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& vectors,
             double omega, double pair_cutoff, double auxiliary_cutoff,
             std::size_t output_byte_cap, std::size_t transient_byte_cap,
             std::size_t image_candidate_cap, double integral_screen_error, bool compute_metric) {
              vibeqc::GDFRangeSeparatedIntegrals result;
              {
                  py::gil_scoped_release release;
                  result = vibeqc::compute_gdf_range_separated_integrals(
                      orbital, aux, system, q, ket_kpoints, vectors, omega,
                      pair_cutoff, auxiliary_cutoff, output_byte_cap,
                      transient_byte_cap, image_candidate_cap, integral_screen_error, compute_metric);
              }
              return py::make_tuple(gdf_metric_array(std::move(result.metric)),
                  gdf_sr_batch_array(std::move(result.three_center)), result.reciprocal_vector_count);
          }, py::arg("orbital"), py::arg("aux"), py::arg("system"),
          py::arg("q"), py::arg("ket_kpoints"), py::arg("vectors"),
          py::arg("omega"), py::arg("pair_cutoff"), py::arg("auxiliary_cutoff"),
          py::arg("output_byte_cap"), py::arg("transient_byte_cap"), py::arg("image_candidate_cap"),
          py::arg("integral_screen_error") = 0.0,
          py::arg("compute_metric") = true,
          "Periodic Coulomb (M,T,nG) from real-space erfc plus compact reciprocal erf integrals. "
          "compute_metric=False returns an empty M when the caller already owns this q metric.");
    m.def("_compute_gdf_plane_wave_projection",
          [](const vibeqc::BasisSet& orbital, const vibeqc::BasisSet& aux,
             const vibeqc::PeriodicSystem& system, const Eigen::Vector3d& q,
             const Eigen::MatrixXd& ket_kpoints,
             const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& vectors,
             double pair_cutoff, std::size_t output_byte_cap,
             std::size_t transient_byte_cap, std::size_t image_candidate_cap,
             bool compute_metric) {
              vibeqc::GDFPlaneWaveProjection result;
              {
                  py::gil_scoped_release release;
                  result = vibeqc::compute_gdf_plane_wave_projection(
                      orbital, aux, system, q, ket_kpoints, vectors, pair_cutoff,
                      output_byte_cap, transient_byte_cap, image_candidate_cap, compute_metric);
              }
              return py::make_tuple(gdf_metric_array(std::move(result.metric)),
                  gdf_sr_batch_array(std::move(result.three_center)),
                  gdf_sr_batch_array(std::move(result.factors)), result.reciprocal_vector_count);
          }, py::arg("orbital"), py::arg("aux"), py::arg("system"),
          py::arg("q"), py::arg("ket_kpoints"), py::arg("vectors"),
          py::arg("pair_cutoff"), py::arg("output_byte_cap"),
          py::arg("transient_byte_cap"), py::arg("image_candidate_cap"),
          py::arg("compute_metric") = true,
          "Private MDF PW Gram blocks and factors on the SR/LR AO-pair image domain. "
          "No zero-mode or exchange correction; production MDF remains separately gated.");
    m.def("_compute_gdf_plane_wave_projection_gradient_weighted",
          [](const vibeqc::BasisSet& orbital, const vibeqc::BasisSet& aux,
             const vibeqc::PeriodicSystem& system, const Eigen::Vector3d& q,
             const Eigen::MatrixXd& ket_kpoints,
             const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& vectors,
             double pair_cutoff, const Eigen::Ref<const Eigen::MatrixXcd>& metric_weight,
             const py::array_t<std::complex<double>, py::array::c_style>& three_center_weight,
             const py::array_t<std::complex<double>, py::array::c_style>& factor_weight,
             std::size_t output_byte_cap, std::size_t transient_byte_cap,
             std::size_t image_candidate_cap) {
              auto weight_view = [](const auto& weight) {
                  if (weight.ndim() != 4 || weight.shape(2) != weight.shape(3))
                      throw std::invalid_argument("MDF derivative weights require shape (nk,nfit,nao,nao)");
                  return vibeqc::GDFShortRangeWeightView{
                      weight.data(), static_cast<std::size_t>(weight.shape(0)),
                      static_cast<std::size_t>(weight.shape(1)),
                      static_cast<std::size_t>(weight.shape(2))};
              };
              const auto tensor_view = weight_view(three_center_weight);
              const auto factor_view = weight_view(factor_weight);
              py::gil_scoped_release release;
              return vibeqc::compute_gdf_plane_wave_projection_gradient_weighted(
                  orbital, aux, system, q, ket_kpoints, vectors, pair_cutoff,
                  metric_weight, tensor_view, factor_view, output_byte_cap,
                  transient_byte_cap, image_candidate_cap);
          }, py::arg("orbital"), py::arg("aux"), py::arg("system"), py::arg("q"),
          py::arg("ket_kpoints"), py::arg("vectors"), py::arg("pair_cutoff"),
          py::arg("metric_weight").noconvert(), py::arg("three_center_weight").noconvert(),
          py::arg("factor_weight").noconvert(), py::arg("output_byte_cap"),
          py::arg("transient_byte_cap"), py::arg("image_candidate_cap"),
          "Private atomic derivative of Re(sum(WM*M_PW)+sum(WT*T_PW)+sum(WF*F_PW)). "
          "Borrow unconjugated complex128 weights; fixed cell/momenta/image membership. "
          "No singular-mode restoration or complete MDF force claim.");
}
