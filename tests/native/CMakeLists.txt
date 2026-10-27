cmake_minimum_required(VERSION 3.18)
project(vibeqc_native_regressions LANGUAGES CXX)
enable_testing()
find_package(Eigen3 CONFIG REQUIRED NO_MODULE)
find_package(Libint2 2.7 CONFIG REQUIRED)
get_filename_component(VIBEQC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
add_executable(charge_diis
    ../test_charge_diis.cpp
    ${VIBEQC_ROOT}/cpp/src/semiempirical/core/charge_mixer.cpp)
target_compile_features(charge_diis PRIVATE cxx_std_17)
target_include_directories(charge_diis PRIVATE ${VIBEQC_ROOT}/cpp/include)
target_link_libraries(charge_diis PRIVATE Eigen3::Eigen)
add_test(NAME charge_diis COMMAND charge_diis)
set_tests_properties(charge_diis PROPERTIES TIMEOUT 30)
add_executable(trah_noncanonical
    ../test_trah_noncanonical.cpp
    ${VIBEQC_ROOT}/cpp/src/trah.cpp
    ${VIBEQC_ROOT}/cpp/src/quadratic_scf.cpp)
target_compile_features(trah_noncanonical PRIVATE cxx_std_17)
target_include_directories(trah_noncanonical PRIVATE ${VIBEQC_ROOT}/cpp/include)
target_link_libraries(trah_noncanonical PRIVATE Eigen3::Eigen Libint2::cxx)
add_test(NAME trah_noncanonical COMMAND trah_noncanonical)
set_tests_properties(trah_noncanonical PROPERTIES TIMEOUT 30)
add_test(NAME trah_hard_case_424 COMMAND trah_noncanonical --hard-case)
set_tests_properties(trah_hard_case_424 PROPERTIES TIMEOUT 30 LABELS "issue-424")
