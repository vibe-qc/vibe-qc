# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
find_package(pybind11 CONFIG REQUIRED)

# Homebrew currently ships Eigen 5.x; we only use dense matrix ops and
# SelfAdjointEigenSolver, which have been stable since Eigen 3.4. Don't pin a
# version.
find_package(Eigen3 CONFIG REQUIRED NO_MODULE)

# Libint installs CMake config as "libint2-config.cmake".
find_package(Libint2 2.7 CONFIG REQUIRED)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/RequireLibintOneBodyDerivatives.cmake")
vibeqc_require_libint_onebody_derivatives()

find_package(OpenMP REQUIRED)

# libxc: exchange-correlation functional library for DFT.
find_package(Libxc 7.0 CONFIG REQUIRED)

# spglib: space-group analysis for periodic systems (towards the cyclic
# cluster model). Target name: Spglib::symspg.
find_package(Spglib 2.0 CONFIG REQUIRED)

# libecpint: effective-core-potential integrals (Phase 14). Built and
# vendored under third_party/libecpint/install/ alongside its own
# pugixml + libcerf builds; see scripts/build_libecpint.sh. Target
# name: ECPINT::ecpint.
find_package(ecpint 1.0 CONFIG REQUIRED)

# FFTW3: FFT library for the reciprocal-space long-range Hartree piece
# of Ewald-decomposed periodic Coulomb (Phase 12e-c-3). Homebrew and
# Ubuntu's libfftw3-dev both ship an old-style CMake config that sets
# FFTW3_LIBRARIES / FFTW3_INCLUDE_DIRS but no imported target. Bridge
# to a modern FFTW3::fftw3 target so target_link_libraries stays
# uniform across dependencies.
find_package(FFTW3 CONFIG REQUIRED)
if(NOT TARGET FFTW3::fftw3)
    add_library(FFTW3::fftw3 INTERFACE IMPORTED)
    target_include_directories(FFTW3::fftw3
        INTERFACE ${FFTW3_INCLUDE_DIRS})
    target_link_directories(FFTW3::fftw3
        INTERFACE ${FFTW3_LIBRARY_DIRS})
    target_link_libraries(FFTW3::fftw3
        INTERFACE ${FFTW3_LIBRARIES})
endif()

# -----------------------------------------------------------------------------
# Core library (separate from the pybind11 module so we can unit-test it
# without Python in the loop).
# -----------------------------------------------------------------------------
add_library(vibeqc_core STATIC
    src/init.cpp
    src/molecule.cpp
    src/basis.cpp
    src/degenerate_frontier.cpp
    src/integrals.cpp
    src/df.cpp
    src/aux_eri.cpp
    src/periodic_gdf_short_range.cpp
    src/aopair_ft.cpp
    src/cosx.cpp
    src/cosx_cell_pair.cpp
    src/cosx_kernel.cpp
    src/cosx_one_center.cpp
    src/cosx_robust.cpp
    src/jk_builder.cpp
    src/jk_direct.cpp
    src/periodic_jk_builder.cpp
    src/periodic_cosx_jk_builder.cpp
    src/fock.cpp
    src/diis.cpp
    src/ediis.cpp
    src/kdiis.cpp
    src/soscf.cpp
    src/scf_restart.cpp
    src/trah.cpp
    src/guess.cpp
    src/rhf.cpp
    src/uhf.cpp
    src/gradient.cpp
    src/basis_param_gradient.cpp
    src/periodic_gradient.cpp
    src/hessian_integrals.cpp
    src/grid.cpp
    src/grid_batch.cpp
    src/lebedev_data.cpp
    src/ao_eval.cpp
    src/xc.cpp
    src/vv10.cpp
    src/b97m_energy.cpp
    src/xc_kernel.cpp
    src/rks.cpp
    src/uks.cpp
    src/mp2.cpp
    src/ump2.cpp
    src/ccsd.cpp
    src/uccsd.cpp
    src/crystal.cpp
    src/periodic.cpp
    src/lattice_sum.cpp
    src/lattice_integrals.cpp
    src/multipole_moments_lattice.cpp
    src/adjoined_pair_moments.cpp
    src/kmesh_address.cpp
    src/symmetry_shared.cpp
    src/periodic_ao_bloch_transport.cpp
    src/periodic_orbital_sewing.cpp
    src/bipole_ewald_gram.cpp
    src/bipole_erfc_panel.cpp
    src/bipole_erfc_bloch.cpp
    src/bipole_finite_source.cpp
    src/bloch.cpp
    src/periodic_bloch.cpp
    src/periodic_fock.cpp
    src/bipole_dispatch.cpp
    src/bipole_far_field_kernel.cpp
    src/bipole_contractor.cpp
    src/bipole_spherical_buffer.cpp
    src/bipole_pair_moments.cpp
    src/bipole_moment_derivatives.cpp
    src/bipole_far_field_gradient.cpp
    src/bipole_fock_kernel_builder.cpp
    src/polipo_moments.cpp
    src/bipole_multipole.cpp
    src/schwarz.cpp
    src/periodic_rhf.cpp
    src/periodic_scf.cpp
    src/periodic_correlation_resource.cpp
    src/periodic_correlation_admitted_reference.cpp
    src/periodic_correlation_factor_stream.cpp
    src/periodic_correlation_factor_build_admission.cpp
    src/periodic_correlation_reciprocal_metric.cpp
    src/periodic_correlation_metric_factorization.cpp
    src/periodic_correlation_three_center.cpp
    src/periodic_correlation_factor_store.cpp
    src/periodic_correlation_wannier.cpp
    src/periodic_correlation_diabatic_seed.cpp
    src/periodic_correlation_bloch_iao.cpp
    src/periodic_correlation_iao_pm.cpp
    src/periodic_correlation_iao_optimizer.cpp
    src/periodic_correlation_occupied_fock.cpp
    src/periodic_correlation_pao_domain.cpp
    src/periodic_correlation_pao_space.cpp
    src/periodic_correlation_real_pao_space.cpp
    src/periodic_correlation_real_pao_embedding.cpp
    src/periodic_correlation_occupied_pao_domain.cpp
    src/periodic_correlation_pair_pao_domain.cpp
    src/periodic_correlation_local_factors.cpp
    src/periodic_correlation_density_factors.cpp
    src/periodic_correlation_local_orbital_factors.cpp
    src/periodic_correlation_real_local_basis.cpp
    src/periodic_correlation_real_local_provider.cpp
    src/periodic_gaussian_source_context.cpp
    src/periodic_gaussian_reciprocal_source.cpp
    src/periodic_gaussian_metric.cpp
    src/periodic_gaussian_three_center.cpp
    src/periodic_gaussian_bloch_iao.cpp
    src/periodic_gaussian_localization.cpp
    src/periodic_gaussian_occupied_pao_domain.cpp
    src/periodic_gaussian_pair_pao_domain.cpp
    src/periodic_gaussian_pair_domain_builder.cpp
    src/periodic_gaussian_one_electron.cpp
    src/periodic_gaussian_nuclear.cpp
    src/periodic_gaussian_fock.cpp
    src/bounded_periodic_rhf_solver.cpp
    src/periodic_gaussian_rhf.cpp
    src/periodic_gaussian_local_orbital_factors.cpp
    src/periodic_gaussian_density_gram.cpp
    src/periodic_gaussian_mixed_pair_factors.cpp
    src/periodic_gaussian_pair_interaction.cpp
    src/periodic_gaussian_pair_particle_hole.cpp
    src/periodic_gaussian_pair_ccsd_physical_particle_hole.cpp
    src/periodic_gaussian_real_local_provider.cpp
    src/periodic_gaussian_selected_local_ccsd_t.cpp
    src/periodic_gaussian_pair_pnos.cpp
    src/periodic_gaussian_embedded_pair_pnos.cpp
    src/periodic_gaussian_gram_pair_pnos.cpp
    src/periodic_gaussian_pair_space.cpp
    src/bounded_restricted_pair_mp2_solver.cpp
    src/bounded_restricted_pair_ccsd_amplitudes.cpp
    src/bounded_restricted_ccsd_energy.cpp
    src/bounded_restricted_pair_ccsd_solver.cpp
    src/bounded_restricted_pair_ccsd_interaction.cpp
    src/bounded_restricted_pair_ccsd_particle_hole.cpp
    src/periodic_gaussian_pair_mp2.cpp
    src/periodic_gaussian_domain_pair_mp2.cpp
    src/periodic_gaussian_pair_ccsd.cpp
    src/periodic_gaussian_triple_spaces.cpp
    src/periodic_gaussian_local_triples.cpp
    src/periodic_correlation_pao_overlap.cpp
    src/periodic_correlation_pair_integrals.cpp
    src/periodic_correlation_pair_residual.cpp
    src/bounded_restricted_ccsd_target.cpp
    src/bounded_restricted_ccsd_solver.cpp
    src/bounded_restricted_triples_target.cpp
    src/bounded_restricted_triples_moments.cpp
    src/bounded_restricted_triples_solver.cpp
    src/bounded_restricted_local_triples_solver.cpp
    src/bounded_restricted_local_triples_moments.cpp
    src/bounded_restricted_triple_natural_orbitals.cpp
    src/periodic_correlation_real_local_ccsd_t.cpp
    src/hermitian_jacobi.cpp
    src/pair_natural_orbitals.cpp
    src/periodic_auxiliary_fourier.cpp
    src/periodic_correlation_pair_topology.cpp
    src/periodic_mean_field_state.cpp
    src/periodic_rhf_state_capture.cpp
    src/periodic_xc.cpp
    src/ewald.cpp
    src/dft_plus_u.cpp
    src/dispersion.cpp
    src/dispersion_data.cpp
    src/dispersion_params.cpp
    src/eeq_charges.cpp
    src/eeq_charges_data.cpp
    src/semiempirical/parameters.cpp
    src/semiempirical/basis.cpp
    src/semiempirical/hamiltonian.cpp
    src/semiempirical/dftb0.cpp
    src/semiempirical/gradient.cpp
    src/semiempirical/repulsive_spline.cpp
    src/semiempirical/scc_dftb.cpp
    src/semiempirical/native_facade.cpp
    src/semiempirical/periodic_dftb0.cpp
    src/semiempirical/periodic_scc_dftb.cpp
    src/semiempirical/kpoints_occupations.cpp
    src/semiempirical/kpoints_dftb0.cpp
    src/semiempirical/kpoints_scc_dftb.cpp
    src/semiempirical/kpoints_dftb_fd_batch.cpp
    src/semiempirical/seccm/dftb0.cpp
    src/semiempirical/seccm/scc_dftb.cpp
    src/semiempirical/seccm/pm6.cpp
    src/semiempirical/seccm/omx.cpp
    src/semiempirical/seccm/gfn2.cpp
    src/semiempirical/core/method_registry.cpp
    src/semiempirical/methods/xtb/gfn2_parameters.cpp
    src/semiempirical/methods/xtb/gfn2_driver.cpp
    src/semiempirical/methods/xtb/gfn2_multipole.cpp
    src/semiempirical/methods/xtb/gfn2_aes.cpp
    src/semiempirical/methods/xtb/periodic_gfn2.cpp
    src/semiempirical/methods/xtb/ugfn2.cpp
    src/semiempirical/methods/nddo/pm6_parameters.cpp
    src/semiempirical/methods/nddo/pm6_fock.cpp
    src/semiempirical/methods/nddo/periodic_pm6.cpp
    src/semiempirical/methods/nddo/periodic_omx.cpp
    src/semiempirical/methods/nddo/periodic_fd_batch.cpp
    src/semiempirical/methods/nddo/omx_fock.cpp
    src/semiempirical/methods/nddo/omx_parameters.cpp
    src/semiempirical/core/hamiltonian_builders.cpp
    src/semiempirical/core/charge_mixer.cpp
    src/semiempirical/core/periodic_gamma.cpp
    src/semiempirical/methods/xtb/kpoints_gfn2.cpp
    src/fft_poisson.cpp
    src/linear_dependence.cpp
    src/brent.cpp
    src/davidson.cpp
    src/jd.cpp
    src/gplhr.cpp
    src/geomopt_kernels.cpp
    src/lobpcg.cpp
    src/ecp.cpp
    src/quadratic_scf.cpp
    src/opentrustregion.cpp
    src/orbital_objective.cpp
    src/newton.cpp
    src/casci.cpp
    src/coop_cohp.cpp
    src/nevpt2_ops.cpp
    src/selected_ci.cpp
    src/casscf_gradient.cpp
    src/pt2_transform.cpp
    src/ccsd_triples.cpp
    src/dlpno_mp2.cpp
)

if(VIBEQC_ENABLE_OPENTRUSTREGION)
    target_link_libraries(vibeqc_core PUBLIC OpenTrustRegion::opentrustregion)
    set_source_files_properties(src/opentrustregion.cpp PROPERTIES
        COMPILE_DEFINITIONS VIBEQC_HAS_OPENTRUSTREGION=1)
endif()

target_include_directories(vibeqc_core PUBLIC include)

target_link_libraries(vibeqc_core PUBLIC
    Eigen3::Eigen
    Libint2::cxx          # header-only C++11 interface to libint
    Libxc::xc             # C library for exchange-correlation functionals
    Spglib::symspg        # C library for space-group analysis
    FFTW3::fftw3          # double-precision real/complex FFTs
    ECPINT::ecpint        # effective-core-potential integrals (Phase 14)
    OpenMP::OpenMP_CXX
)

# -----------------------------------------------------------------------------
# Optimised BLAS + LAPACK backend (default ON, opt-out via -DVIBEQC_USE_BLAS=OFF)
#
# Without this, every dense matrix product, eigendecomposition and Cholesky
# inside Eigen runs Eigen's generic-C++ kernels. Those are fine for tiny
# matrices but easily an order of magnitude slower than an optimised BLAS for
# the SCF Fock-build / diagonalisation / DF Cholesky chain at >100 basis
# functions, and they account for the bulk of vibe-qc's wall-time gap vs
# ORCA on RIJCOSX / HF / def2-TZVP. Defining EIGEN_USE_BLAS (and, when
# possible, EIGEN_USE_LAPACKE) makes Eigen delegate to the linked
# BLAS+LAPACK with **no source-code change** — the speedup applies
# everywhere a `*`, `.noalias()`, `Eigen::LLT`, `Eigen::LDLT`,
# `Eigen::SelfAdjointEigenSolver`, etc. appears in the C++ core.
#
# Backend selection:
#   - macOS: Apple Accelerate framework (ships with the OS; no install).
#     Picked automatically via FindBLAS(BLA_VENDOR=Apple).
#   - Linux: CMake's FindBLAS scans well-known vendors. Override with
#     -DVIBEQC_BLAS_VENDOR=<name> to pin (e.g. OpenBLAS, Intel10_64lp).
#     Falls back to whatever liblapack/libblas the system provides
#     (reference netlib BLAS works but is slow; install openblas for
#     the production win).
#   - Disable: -DVIBEQC_USE_BLAS=OFF reverts to Eigen-generic kernels.
#
# LAPACKE (C interface to LAPACK) is required for EIGEN_USE_LAPACKE.
# Distros split it from liblapack: Arch has it via `openblas` /
# `blas-openblas`, Debian/Ubuntu has `liblapacke-dev`, reference
# netlib lapack typically lacks it. When LAPACKE isn't found,
# EIGEN_USE_BLAS is still set — the GEMM/GEMV chain (the dominant
# SCF cost) accelerates regardless, only dsyev / Cholesky delegation
# requires LAPACKE.
#
# Threading: vibe-qc parallelises with OpenMP. To avoid nested
# oversubscription with a threaded BLAS (OpenBLAS-pthreads, MKL),
# python/vibeqc/__init__.py sets OPENBLAS_NUM_THREADS=1 /
# MKL_NUM_THREADS=1 / VECLIB_MAXIMUM_THREADS=1 by default; user-set
# values are respected via os.environ.setdefault.
# -----------------------------------------------------------------------------
option(VIBEQC_USE_BLAS
    "Link an optimised BLAS+LAPACK so Eigen delegates dgemm/dsyev/dpotrf"
    ON)
set(VIBEQC_BLAS_VENDOR "" CACHE STRING
    "BLA_VENDOR pin for FindBLAS/FindLAPACK (empty = auto, Apple on macOS)")

set(_VIBEQC_BLAS_BACKEND "")
set(_VIBEQC_BLAS_FLAG 0)
set(_VIBEQC_LAPACKE_FLAG 0)

if(VIBEQC_USE_BLAS)
    if(VIBEQC_BLAS_VENDOR)
        set(BLA_VENDOR "${VIBEQC_BLAS_VENDOR}")
    elseif(APPLE)
        # macOS default: Apple Accelerate. Users who want OpenBLAS
        # on macOS pass -DVIBEQC_BLAS_VENDOR=OpenBLAS explicitly
        # (and call scripts/build_openblas.sh first to make the
        # vendored install available).
        set(BLA_VENDOR "Apple")
    elseif(EXISTS "${CMAKE_SOURCE_DIR}/third_party/openblas/install/lib")
        # Vendored OpenBLAS is present (built via
        # scripts/build_openblas.sh, opted into via
        # WITH_OPENBLAS=1 ./scripts/setup_native_deps.sh). Pin
        # FindBLAS to the OpenBLAS vendor so it resolves the
        # libopenblas.so under third_party/openblas/install/lib/
        # (already prepended to CMAKE_PREFIX_PATH by the top-level
        # CMakeLists.txt). This makes the vendored build win
        # deterministically over any system BLAS, including a slow
        # netlib BLAS that find_package(BLAS) might otherwise pick
        # first.
        set(BLA_VENDOR "OpenBLAS")
        message(STATUS
            "vibe-qc: vendored OpenBLAS detected at "
            "third_party/openblas/install — pinning BLA_VENDOR=OpenBLAS")
    endif()

    find_package(BLAS QUIET)
    find_package(LAPACK QUIET)

    # If the targeted vendor isn't installed on this box, retry once
    # with no pin so the build still picks up whatever BLAS *is* there.
    if((NOT BLAS_FOUND OR NOT LAPACK_FOUND) AND BLA_VENDOR)
        message(STATUS
            "vibe-qc: BLA_VENDOR=${BLA_VENDOR} not found — retrying auto-detect")
        unset(BLA_VENDOR)
        find_package(BLAS QUIET)
        find_package(LAPACK QUIET)
    endif()

    if(BLAS_FOUND AND LAPACK_FOUND)
        message(STATUS "vibe-qc: BLAS_LIBRARIES   = ${BLAS_LIBRARIES}")
        message(STATUS "vibe-qc: LAPACK_LIBRARIES = ${LAPACK_LIBRARIES}")

        # LAPACK_LIBRARIES on macOS reproduces "-framework Accelerate"
        # exactly — duplicates are harmless. On Linux they're distinct
        # paths (/usr/lib/libblas.so + /usr/lib/liblapack.so).
        target_link_libraries(vibeqc_core PUBLIC
            ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
        target_compile_definitions(vibeqc_core PUBLIC EIGEN_USE_BLAS)
        set(_VIBEQC_BLAS_BACKEND "${BLAS_LIBRARIES}")
        set(_VIBEQC_BLAS_FLAG 1)

        # LAPACKE detection. Apple is handled separately — Accelerate
        # ships LAPACKE but Eigen's bridge has version-specific quirks
        # vs the macOS-13.3+ "new LAPACK" ABI; we keep dense solvers
        # on Eigen-generic on Apple and let Accelerate's BLAS soak up
        # the GEMM cost, which is what matters for SCF.
        if(NOT APPLE)
            find_path(LAPACKE_INCLUDE_DIR
                NAMES lapacke.h
                PATH_SUFFIXES openblas)
            find_library(LAPACKE_LIBRARY
                NAMES lapacke openblas)
            if(LAPACKE_INCLUDE_DIR AND LAPACKE_LIBRARY)
                target_include_directories(vibeqc_core SYSTEM PUBLIC
                    "${LAPACKE_INCLUDE_DIR}")
                target_link_libraries(vibeqc_core PUBLIC
                    "${LAPACKE_LIBRARY}")
                target_compile_definitions(vibeqc_core PUBLIC
                    EIGEN_USE_LAPACKE)
                set(_VIBEQC_LAPACKE_FLAG 1)
                message(STATUS
                    "vibe-qc: LAPACKE_LIBRARY  = ${LAPACKE_LIBRARY} "
                    "(EIGEN_USE_LAPACKE enabled)")
            else()
                message(STATUS
                    "vibe-qc: LAPACKE not found — dense solvers stay on "
                    "Eigen-generic path (install lapacke / openblas-dev "
                    "to enable EIGEN_USE_LAPACKE)")
            endif()
        endif()
    else()
        message(WARNING
            "vibe-qc: VIBEQC_USE_BLAS=ON but no BLAS/LAPACK found — Eigen "
            "will run generic kernels (slow at SCF size). Install OpenBLAS "
            "(Linux) / build with Apple Accelerate (macOS), or pass "
            "-DVIBEQC_USE_BLAS=OFF to silence this warning.")
    endif()
endif()

# Surface the chosen backend to the C++ layer for runtime reporting via
# vibeqc.banner.library_versions()["blas"]. The generated header is
# read by cpp/src/bindings.cpp (blas_info()).
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/vibeqc/build_config.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/vibeqc/build_config.hpp"
    @ONLY)
target_include_directories(vibeqc_core PUBLIC
    "${CMAKE_CURRENT_BINARY_DIR}/generated")

target_compile_options(vibeqc_core PRIVATE
    -Wall -Wextra -Wpedantic
    # GCC 15.x tightened -Wmaybe-uninitialized and -Warray-bounds
    # enough that libint2's solid-harmonics + g94 reader headers
    # generate noisy diagnostics through inlined stl_vector.h /
    # array.h paths. Those are not our code (they're inside
    # third_party/libint/install/include/libint2/), and we can't
    # fix them — suppress at our compile site so the build stays
    # clean on Arch / Fedora / Ubuntu running GCC 15.
    $<$<CXX_COMPILER_ID:GNU>:-Wno-maybe-uninitialized -Wno-array-bounds>
)

# -----------------------------------------------------------------------------
# Optional host-CPU tuning (default OFF — portable wheels stay bit-identical).
#
# -march=native (x86) / -mcpu=native (arm64) lets the compiler emit the host's
# full SIMD instruction set for the integral / XC / linear-algebra kernels,
# typically ~1.1–1.4x on compute-bound paths for a local/dev build. OFF by
# default because the resulting binary is NOT portable across CPU generations
# (and may select FMA, shifting energies at the ~1e-14 level — generally more
# accurate, but not bit-identical to the portable build). Enable with
# -DVIBEQC_NATIVE_ARCH=ON or the matching env var. Applied to the kernel
# library only (bindings are glue). NOT -ffast-math — that would perturb the
# energies the regression suite pins.
# -----------------------------------------------------------------------------
option(VIBEQC_NATIVE_ARCH
    "Tune the build for the host CPU (-march/-mcpu=native + unrolling). OFF for portable wheels."
    OFF)
if(NOT VIBEQC_NATIVE_ARCH AND DEFINED ENV{VIBEQC_NATIVE_ARCH})
    set(VIBEQC_NATIVE_ARCH $ENV{VIBEQC_NATIVE_ARCH})
endif()
if(VIBEQC_NATIVE_ARCH)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-march=native" _vibeqc_has_march_native)
    if(_vibeqc_has_march_native)
        target_compile_options(vibeqc_core PRIVATE -march=native)
        set(_vibeqc_arch_flag "-march=native")
    else()
        check_cxx_compiler_flag("-mcpu=native" _vibeqc_has_mcpu_native)
        if(_vibeqc_has_mcpu_native)
            target_compile_options(vibeqc_core PRIVATE -mcpu=native)
            set(_vibeqc_arch_flag "-mcpu=native")
        else()
            set(_vibeqc_arch_flag "(none — compiler rejected -march/-mcpu=native)")
        endif()
    endif()
    target_compile_options(vibeqc_core PRIVATE -funroll-loops)
    message(STATUS
        "vibe-qc: VIBEQC_NATIVE_ARCH=ON — host-CPU tuning ${_vibeqc_arch_flag} "
        "-funroll-loops (non-portable binary)")
endif()

# Force third-party include paths to be treated as SYSTEM so headers
# from libint / Eigen / libxc / spglib / libecpint can't trip our
# warnings when they get stricter (cf. GCC 15's diagnostics, the
# Eigen3 + libint2 templates, etc.). Has no effect when the dependency
# was pulled in via a CMake config that already marked itself SYSTEM,
# but the libint2 ConfigVersion.cmake we vendor currently does not.
set(_VIBEQC_THIRDPARTY_TARGETS
    Eigen3::Eigen Libint2::cxx Libxc::xc
    Spglib::symspg FFTW3::fftw3 ECPINT::ecpint
)
foreach(_tgt IN LISTS _VIBEQC_THIRDPARTY_TARGETS)
    if(TARGET ${_tgt})
        get_target_property(_inc_dirs ${_tgt} INTERFACE_INCLUDE_DIRECTORIES)
        if(_inc_dirs)
            set_target_properties(${_tgt} PROPERTIES
                INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_inc_dirs}"
            )
        endif()
    endif()
endforeach()
unset(_VIBEQC_THIRDPARTY_TARGETS)

# Bake the path to libecpint's XML basis library (Stuttgart-Köln
# ecpNNmdf, LANL2DZ) so ``compute_ecp_matrix`` can find it without
# the user passing share_dir explicitly. Resolution order:
#
#  1. The repo-local third_party build, if scripts/build_libecpint.sh
#     populated it (vendored layout — preferred for development).
#  2. PACKAGE_PREFIX_DIR set by libecpint's CMake config when we
#     find_package(ecpint) above (Homebrew, conda-forge, system).
#
# Subtle path convention: libecpint's
# ``ECPIntegrator::set_ecp_basis_from_library`` internally appends
# ``"/xml/" + name + ".xml"`` to its share_dir argument — so we pass
# the directory CONTAINING xml/ (i.e. ``.../share/libecpint``), NOT
# the xml/ directory itself. The header doc in libecpint/api.hpp is
# misleading on this point; the source (src/lib/api.cpp:73) is the
# authority.
set(_VIBEQC_VENDORED_ECPINT_SHARE
    "${CMAKE_SOURCE_DIR}/third_party/libecpint/install/share/libecpint")
if(EXISTS "${_VIBEQC_VENDORED_ECPINT_SHARE}/xml/ecp10mdf.xml")
    set(_VIBEQC_LIBECPINT_SHARE_DIR "${_VIBEQC_VENDORED_ECPINT_SHARE}")
elseif(DEFINED PACKAGE_PREFIX_DIR
       AND EXISTS "${PACKAGE_PREFIX_DIR}/share/libecpint/xml/ecp10mdf.xml")
    set(_VIBEQC_LIBECPINT_SHARE_DIR "${PACKAGE_PREFIX_DIR}/share/libecpint")
else()
    # Last-ditch: search common install layouts via find_path() for
    # the directory containing xml/ecp10mdf.xml.
    find_path(_VIBEQC_LIBECPINT_SHARE_DIR
              NAMES xml/ecp10mdf.xml
              PATH_SUFFIXES share/libecpint
              PATHS ${CMAKE_PREFIX_PATH})
    if(NOT _VIBEQC_LIBECPINT_SHARE_DIR)
        set(_VIBEQC_LIBECPINT_SHARE_DIR "")
        message(WARNING "vibeqc: could not auto-locate libecpint's XML share "
                        "directory. compute_ecp_matrix will require the "
                        "share_dir argument at every call site.")
    endif()
endif()
message(STATUS "vibeqc: libecpint share dir = ${_VIBEQC_LIBECPINT_SHARE_DIR}")
target_compile_definitions(vibeqc_core PRIVATE
    VIBEQC_LIBECPINT_SHARE_DIR="${_VIBEQC_LIBECPINT_SHARE_DIR}"
    HAS_PUGIXML=1
)

# -fPIC so the static lib can be linked into the shared pybind11 module.
set_target_properties(vibeqc_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# -----------------------------------------------------------------------------
# pybind11 module: _vibeqc_core
# -----------------------------------------------------------------------------
pybind11_add_module(_vibeqc_core MODULE
    src/opentrustregion_bindings.cpp
    src/bindings.cpp
)

target_link_libraries(_vibeqc_core PRIVATE
    vibeqc_core
    Libint2::cxx          # brings header path for libint2/util/configuration.h
)

target_compile_options(_vibeqc_core PRIVATE
    -Wall -Wextra -Wpedantic
    # See note on the matching block for ``vibeqc_core`` above:
    # GCC 15 emits new -Wmaybe-uninitialized / -Warray-bounds
    # diagnostics through libint2's inlined headers that we can't
    # fix at the source. Suppress them here so binders compile clean
    # on Linux GCC 15 (Arch / Fedora / Ubuntu 25.10+).
    $<$<CXX_COMPILER_ID:GNU>:-Wno-maybe-uninitialized -Wno-array-bounds>
)

# Bake RPATH so the compiled module finds the vendored shared
# libraries (libint, libxc, spglib, FFTW, libecpint) without the user
# having to set ``LD_LIBRARY_PATH`` (Linux) or ``DYLD_LIBRARY_PATH``
# (macOS). Absolute paths under ``CMAKE_SOURCE_DIR`` tie the build to
# this checkout — fine for editable-mode development; wheel builds
# need a different scheme (auditwheel-style bundling) which we'll
# handle when we publish wheels.
#
# Linux note: macOS happens to find some of these via DYLD's fallback
# search even without explicit rpath, but Linux's loader is strict —
# every vendored .so MUST be reachable through RPATH or a system path.
# Pre-fix, ``import vibeqc`` failed on Manjaro / glibc with
# ``ImportError: libxc.so.15: cannot open shared object file`` because
# only libint + libecpint were listed in the rpath; libxc / spglib /
# FFTW were silently expected to be on the system path. This loop
# covers all vendored deps explicitly.
#
# ``openblas`` only appears in the rpath when its install dir is
# present (the vendored OpenBLAS is opt-in via WITH_OPENBLAS=1 in
# scripts/setup_native_deps.sh; system-installed OpenBLAS lives on
# the linker's default search path and doesn't need rpath baking).
#
# Both ``lib`` and ``lib64``: CMake's GNUInstallDirs sends the vendored
# .so to ``lib64`` on openSUSE / Tumbleweed
# (see docs/cluster_setup.md) but to ``lib`` on Arch / Ubuntu.
# Baking both keeps the rpath correct on either layout; a non-existent
# directory in the rpath is harmlessly ignored by the dynamic loader, so
# the extra entries cost nothing on distros that use ``lib``.
set(_VIBEQC_RPATH_DIRS "")
foreach(_dep IN ITEMS libint libxc spglib fftw libecpint)
    list(APPEND _VIBEQC_RPATH_DIRS
         "${CMAKE_SOURCE_DIR}/third_party/${_dep}/install/lib"
         "${CMAKE_SOURCE_DIR}/third_party/${_dep}/install/lib64")
endforeach()
foreach(_obdir lib lib64)
    if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/openblas/install/${_obdir}")
        list(APPEND _VIBEQC_RPATH_DIRS
             "${CMAKE_SOURCE_DIR}/third_party/openblas/install/${_obdir}")
    endif()
endforeach()

set_target_properties(_vibeqc_core PROPERTIES
    INSTALL_RPATH "${_VIBEQC_RPATH_DIRS}"
    BUILD_RPATH   "${_VIBEQC_RPATH_DIRS}"
    MACOSX_RPATH ON
)

# Linux DT_RPATH vs DT_RUNPATH gotcha: modern GNU ld defaults to
# emitting DT_RUNPATH, which is **non-transitive** — the rpath in
# vibeqc_core.so applies only to its DIRECT dependencies. When a
# direct dep (libecpint.so) then needs ITS own dep (libcerfcpp.so.3),
# the loader does NOT search vibeqc_core.so's rpath; libcerfcpp must
# come from libecpint's own rpath, $LD_LIBRARY_PATH, or a system path.
#
# Pre-fix, ``import vibeqc`` on Linux failed at libecpint's libcerfcpp
# resolution. Switching to DT_RPATH (the legacy, transitive form) via
# ``-Wl,--disable-new-dtags`` makes vibeqc_core.so's rpath visible to
# libecpint.so when *it* looks for libcerf. macOS uses @rpath, which
# is transitive by default; the flag is silently ignored by ld64 on
# macOS, so guarding by Linux is just for clarity.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(_vibeqc_core PRIVATE
        "LINKER:--disable-new-dtags")
endif()

# Install the compiled module into the vibeqc package directory inside the wheel
install(TARGETS _vibeqc_core
    LIBRARY DESTINATION vibeqc
)
