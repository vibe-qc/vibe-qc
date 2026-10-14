# scripts/build_libecpint.sh patches libecpint and records its patch version
# in the install. A plain `pip install -e .` does not rerun the script, and a
# patch that changes only libecpint's sources (the #358 buffer fix) still
# compiles against an older install, so the stale library was linked without
# a word (#460). Compare the recorded version with the script's: an error
# under VIBEQC_REQUIRE_VENDORED, a warning otherwise. Nothing is checked when
# libecpint is not vendored.
function(vibeqc_require_current_libecpint_patch)
    set(_install "${CMAKE_SOURCE_DIR}/third_party/libecpint/install")
    set(_script "${CMAKE_SOURCE_DIR}/scripts/build_libecpint.sh")
    set(_marker "${_install}/.vibeqc-radial-quadrature")
    if(NOT EXISTS "${_install}" OR NOT EXISTS "${_script}")
        return()
    endif()
    # Reconfigure when either side changes, e.g. after a pull or a rebuild.
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_script}")

    file(STRINGS "${_script}" _want REGEX "^RADIAL_PATCH_VERSION=\"[^\"]+\"$"
        LIMIT_COUNT 1)
    if(NOT _want)
        message(FATAL_ERROR
            "scripts/build_libecpint.sh defines no RADIAL_PATCH_VERSION=\"...\" "
            "line, so the vendored libecpint cannot be checked; update "
            "cpp/cmake/RequireCurrentLibecpintPatch.cmake with the script.")
    endif()
    string(REGEX REPLACE "^RADIAL_PATCH_VERSION=\"([^\"]+)\"$" "\\1" _want "${_want}")

    set(_have "(no patch marker)")
    if(EXISTS "${_marker}")
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_marker}")
        file(STRINGS "${_marker}" _have LIMIT_COUNT 1)
    endif()
    if(_have STREQUAL _want)
        message(STATUS "Vendored libecpint patch: ${_have}")
        return()
    endif()

    if(VIBEQC_REQUIRE_VENDORED)
        set(_severity FATAL_ERROR)
    else()
        set(_severity WARNING)
    endif()
    message(${_severity}
        "third_party/libecpint/install carries libecpint patch '${_have}', "
        "but scripts/build_libecpint.sh applies '${_want}'. vibe-qc would be "
        "linked against a libecpint without the current fixes. Run "
        "scripts/build_libecpint.sh, which rebuilds only libecpint, and "
        "configure again.")
endfunction()
