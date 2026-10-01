cmake_minimum_required(VERSION 3.18)
project(vibeqc_native_regressions LANGUAGES CXX)
enable_testing()
find_package(Eigen3 CONFIG REQUIRED NO_MODULE)
find_package(Libint2 2.7 CONFIG REQUIRED)
get_filename_component(VIBEQC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
add_executable(trah_noncanonical
    ../test_trah_noncanonical.cpp
    ${VIBEQC_ROOT}/cpp/src/trah.cpp
    ${VIBEQC_ROOT}/cpp/src/quadratic_scf.cpp)
target_compile_features(trah_noncanonical PRIVATE cxx_std_17)
target_include_directories(trah_noncanonical PRIVATE ${VIBEQC_ROOT}/cpp/include)
target_link_libraries(trah_noncanonical PRIVATE Eigen3::Eigen Libint2::cxx)
add_test(NAME trah_noncanonical COMMAND trah_noncanonical)
set_tests_properties(trah_noncanonical PROPERTIES TIMEOUT 30)
add_test(NAME trah_hard_case_424_expected_failure
    COMMAND ${CMAKE_COMMAND} -DTRAH_PROGRAM=$<TARGET_FILE:trah_noncanonical>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_hard_case_failure.cmake)
set_tests_properties(trah_hard_case_424_expected_failure PROPERTIES
    TIMEOUT 30 LABELS "known-failure;issue-424")
