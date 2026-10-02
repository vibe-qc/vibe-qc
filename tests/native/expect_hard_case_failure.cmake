# Strict expected failure for #424: only its explicit scientific check may fail.
execute_process(COMMAND "${TRAH_PROGRAM}" --hard-case
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
message("${output}${error}")
if(result EQUAL 0)
    message(FATAL_ERROR "Unexpected pass for #424: remove the expected-failure wrapper")
elseif(NOT "${result}" STREQUAL "86" OR NOT error MATCHES "KNOWN_FAILURE_424:")
    message(FATAL_ERROR "Unexpected failure running #424 regression: ${result}")
endif()
