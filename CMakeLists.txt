cmake_minimum_required(VERSION 3.24)

project(cuda_convolution_optimization LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures")

option(CUDA_CONV_BUILD_TESTS "Build GPU correctness tests" ON)

add_library(cuda_conv STATIC
  src/cpu_reference.cpp
  src/cuda_conv.cu
)
target_include_directories(cuda_conv PUBLIC include)
if(MSVC)
  target_compile_options(cuda_conv PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/O2>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=/O2>
  )
else()
  target_compile_options(cuda_conv PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  )
endif()
set_target_properties(cuda_conv PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

add_executable(cuda-conv-bench app/benchmark.cpp)
target_link_libraries(cuda-conv-bench PRIVATE cuda_conv)

if(CUDA_CONV_BUILD_TESTS)
  include(CTest)
  add_executable(cuda-conv-correctness tests/correctness.cpp)
  target_link_libraries(cuda-conv-correctness PRIVATE cuda_conv)
  add_test(NAME cuda-conv-correctness COMMAND cuda-conv-correctness)
endif()
