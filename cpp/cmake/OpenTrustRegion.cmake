# An opt-in pinned source build keeps the C header, Fortran settings layout,
# LP64 BLAS and runtime from drifting independently. No Fortran without opt-in.
option(VIBEQC_ENABLE_OPENTRUSTREGION "Build the optional OpenTrustRegion orbital optimizer" OFF)
function(vibeqc_build_opentrustregion)
    include(FetchContent)
    if(DEFINED INTEGER_SIZE AND NOT INTEGER_SIZE STREQUAL "4" AND NOT INTEGER_SIZE STREQUAL "")
        message(FATAL_ERROR "vibe-qc OpenTrustRegion requires INTEGER_SIZE=4 and LP64 BLAS/LAPACK")
    endif()
    set(INTEGER_SIZE 4 CACHE STRING "OpenTrustRegion integer width" FORCE)
    set(BLA_SIZEOF_INTEGER 4)
    if(VIBEQC_BLAS_VENDOR)
        set(BLA_VENDOR "${VIBEQC_BLAS_VENDOR}")
    elseif(APPLE)
        set(BLA_VENDOR "Apple")
    endif()
    set(OpenTrustRegion_ENABLE_XHOST OFF CACHE BOOL "Portable OpenTrustRegion build" FORCE)
    set(OpenTrustRegion_BUILD_TESTING OFF CACHE BOOL "Build only the numerical library" FORCE)
    set(OpenTrustRegion_HOST_PROVIDES_BLAS OFF CACHE BOOL "Link matching BLAS runtime" FORCE)
    # Isolate upstream BUILD_SHARED_LIBS from the host; static PIC is linked
    # into the extension, with CMake supplying the Fortran runtime libraries.
    set(BUILD_SHARED_LIBS OFF)
    FetchContent_Declare(opentrustregion
        GIT_REPOSITORY https://github.com/eriksen-lab/opentrustregion.git
        GIT_TAG 8fa7769ae66233a566868a6bf03cdcbdb1ee69d0
        GIT_SHALLOW FALSE)
    FetchContent_MakeAvailable(opentrustregion)
    find_package(Git REQUIRED)
    execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${opentrustregion_SOURCE_DIR}" rev-parse HEAD
        OUTPUT_VARIABLE _otr_revision OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _otr_git_status)
    if(NOT _otr_git_status EQUAL 0 OR NOT _otr_revision STREQUAL "8fa7769ae66233a566868a6bf03cdcbdb1ee69d0")
        message(FATAL_ERROR "OpenTrustRegion source override must be the pinned revision 8fa7769ae66233a566868a6bf03cdcbdb1ee69d0")
    endif()
    execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${opentrustregion_SOURCE_DIR}" diff HEAD --quiet
        RESULT_VARIABLE _otr_dirty)
    if(NOT _otr_dirty EQUAL 0)
        message(FATAL_ERROR "OpenTrustRegion source must have no tracked changes from the pinned revision")
    endif()
    install(FILES "${opentrustregion_SOURCE_DIR}/LICENSE"
        DESTINATION share/vibeqc/licenses RENAME opentrustregion-MPL-2.0.txt)
    if(NOT TARGET OpenTrustRegion::opentrustregion)
        message(FATAL_ERROR "Pinned OpenTrustRegion target missing")
    endif()
endfunction()
if(VIBEQC_ENABLE_OPENTRUSTREGION)
    enable_language(C)
    enable_language(Fortran)
    vibeqc_build_opentrustregion()
endif()
