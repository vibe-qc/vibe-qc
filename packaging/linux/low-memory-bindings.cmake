# Keep the calculation core optimized; bound memory for the monolithic Python
# binding translation unit. Mixing this native object with the other LTO binding
# objects is supported by the linker. Product source remains unchanged.
function(vibe_package_binding_options)
  set_property(SOURCE "${CMAKE_SOURCE_DIR}/cpp/src/bindings.cpp"
    TARGET_DIRECTORY _vibeqc_core APPEND PROPERTY COMPILE_OPTIONS "-O1;-fno-lto;-Xclang;-clear-ast-before-backend")
endfunction()
cmake_language(DEFER CALL vibe_package_binding_options)
