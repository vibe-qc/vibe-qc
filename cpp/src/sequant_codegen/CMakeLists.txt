# CMakeLists.txt — SeQuant → vibe-qc codegen library
#
# Conditionally builds libvibeqc_sequant_codegen when SeQuant is available.
# The library is optional; the rest of vibe-qc builds without it.
#
# To enable:
#   cmake -B build -DVIBEQC_USE_SEQUANT=ON -DCMAKE_PREFIX_PATH=/path/to/sequant/install
#
# Without SeQuant, a stub library with no-op emitters is built so that
# downstream code using the header compiles (emission returns false).

option(VIBEQC_USE_SEQUANT "Enable SeQuant-based equation derivation and code generation" OFF)

add_library(vibeqc_sequant_codegen STATIC
    vibe_qc_codegen.cpp
    serialized_expr.cpp
)

target_include_directories(vibeqc_sequant_codegen PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(vibeqc_sequant_codegen PUBLIC
    vibeqc_core  # for Mat, RowMat, IntegralBlocks definitions
)

target_compile_features(vibeqc_sequant_codegen PUBLIC cxx_std_17)

if(VIBEQC_USE_SEQUANT)
    find_package(SeQuant CONFIG REQUIRED)
    target_link_libraries(vibeqc_sequant_codegen PUBLIC
        SeQuant::SeQuant
    )
    target_compile_definitions(vibeqc_sequant_codegen PUBLIC
        VIBEQC_HAS_SEQUANT
    )

    message(STATUS "SeQuant → vibe-qc codegen ENABLED (SeQuant ${SeQuant_VERSION})")
else()
    message(STATUS "SeQuant → vibe-qc codegen DISABLED (set -DVIBEQC_USE_SEQUANT=ON to enable)")
endif()

# ---- Tests (only when SeQuant is available) ----
if(VIBEQC_USE_SEQUANT AND BUILD_TESTING)
    add_executable(test_sequant_codegen_ccsd
        ${CMAKE_CURRENT_SOURCE_DIR}/../tests/test_sequant_codegen_ccsd.cpp
    )
    target_link_libraries(test_sequant_codegen_ccsd PRIVATE
        vibeqc_sequant_codegen
        vibeqc_core
    )
    add_test(NAME sequant_codegen_ccsd COMMAND test_sequant_codegen_ccsd)
endif()
