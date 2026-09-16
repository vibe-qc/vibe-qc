# Version and installation origin do not specify libint's generated kernels.
# Check the target used by both vibeqc_core and the Python module (#271).
function(vibeqc_require_libint_onebody_derivatives)
    if(NOT TARGET Libint2::cxx)
        message(FATAL_ERROR "The selected Libint2 package has no Libint2::cxx target.")
    endif()
    # Libint's installed package normally exposes cxx as an alias. Passing an
    # alias to try_compile requires CMake 3.29, while vibe-qc supports 3.20.
    # Resolve it without reconstructing (or losing) transitive usage requirements.
    get_target_property(_vibeqc_libint_target Libint2::cxx ALIASED_TARGET)
    if(NOT _vibeqc_libint_target)
        set(_vibeqc_libint_target Libint2::cxx)
    endif()

    # A static-library probe would never resolve the derivative symbols. Keep
    # this local to the function, including when a toolchain requests otherwise.
    # Compile/link only: do not execute libint, also when cross-compiling.
    set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
    # Recheck even after an in-place dependency replacement, not only a changed
    # Libint2_DIR. Neither a cached success nor a cached failure is authoritative.
    unset(_vibeqc_libint_onebody_ok CACHE)
    try_compile(_vibeqc_libint_onebody_ok
        "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/vibeqc-libint-onebody"
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/libint_onebody_witness.cpp"
        LINK_LIBRARIES "${_vibeqc_libint_target}"
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        OUTPUT_VARIABLE _vibeqc_libint_onebody_output)
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/vibeqc-libint-onebody.log"
        "Libint2_DIR=${Libint2_DIR}\nTarget=${_vibeqc_libint_target}\n"
        "${_vibeqc_libint_onebody_output}")
    set(_vibeqc_libint_onebody_result "${_vibeqc_libint_onebody_ok}")
    unset(_vibeqc_libint_onebody_ok CACHE)
    if(NOT _vibeqc_libint_onebody_result)
        message(FATAL_ERROR
            "The selected Libint2::cxx target cannot provide the ordinary "
            "one-body first derivatives required by vibe-qc analytic gradients.\n"
            "Required: LIBINT2_SUPPORT_ONEBODY=1 and "
            "LIBINT2_DERIV_ONEBODY_ORDER>=1, with matching overlap1, kinetic1 "
            "and elecpot1 build tables and memory functions in the library.\n"
            "Selected package: ${Libint2_DIR}\n"
            "Build a compatible dependency with scripts/build_libint.sh and "
            "reconfigure with Libint2_DIR pointing to its installed CMake package. "
            "VIBEQC_REQUIRE_VENDORED=OFF does not bypass this requirement.\n"
            "Compiler/linker details (including available header capabilities):\n"
            "${_vibeqc_libint_onebody_output}")
    endif()
    message(STATUS "Libint2::cxx ordinary one-body first derivatives: compile/link check passed")
endfunction()
