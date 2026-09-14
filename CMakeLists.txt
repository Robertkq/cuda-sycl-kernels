cmake_minimum_required(VERSION 3.24)

set(SYCL_CUDA_ROOT "$ENV{HOME}/projects/llvm/build" CACHE PATH
    "Root of an intel/llvm build configured with --cuda")
set(_sycl_cuda_clangxx "${SYCL_CUDA_ROOT}/bin/clang++")
if(EXISTS "${_sycl_cuda_clangxx}")
  if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "${_sycl_cuda_clangxx}" CACHE FILEPATH "" FORCE)
  endif()
  if(CMAKE_CXX_COMPILER STREQUAL "${_sycl_cuda_clangxx}")
    set(CMAKE_BUILD_RPATH "${SYCL_CUDA_ROOT}/lib")
    set(CMAKE_INSTALL_RPATH "${SYCL_CUDA_ROOT}/lib")
    set(SYCL_CUDA_AVAILABLE TRUE)
  else()
    set(SYCL_CUDA_AVAILABLE FALSE)
  endif()
else()
  set(SYCL_CUDA_AVAILABLE FALSE)
endif()

project(cuda_sycl_kernels CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  enable_language(CUDA)
else()
  message(WARNING "No CUDA compiler found -- kernels will skip CUDA targets.")
endif()

if(SYCL_CUDA_AVAILABLE)
  set(SYCL_CUDA_TARGET "nvptx64-nvidia-cuda")
  message(STATUS "SYCL CUDA backend enabled: ${CMAKE_CXX_COMPILER} (-fsycl-targets=${SYCL_CUDA_TARGET})")
  get_filename_component(_sycl_cuda_bindir "${CMAKE_CXX_COMPILER}" DIRECTORY)
  find_program(NVPTX_ARCH_TOOL NAMES nvptx-arch HINTS "${_sycl_cuda_bindir}")
  if(NVPTX_ARCH_TOOL)
    execute_process(
      COMMAND "${NVPTX_ARCH_TOOL}"
      OUTPUT_VARIABLE SYCL_CUDA_GPU_ARCH
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
  endif()
  set(SYCL_CUDA_FLAGS -fsycl -fsycl-targets=${SYCL_CUDA_TARGET})
  if(SYCL_CUDA_GPU_ARCH)
    list(APPEND SYCL_CUDA_FLAGS -Xsycl-target-backend=${SYCL_CUDA_TARGET} --cuda-gpu-arch=${SYCL_CUDA_GPU_ARCH})
  endif()
else()
  message(WARNING
    "No SYCL CUDA compiler found at ${_sycl_cuda_clangxx} -- kernels will skip SYCL targets. "
    "Set -DSYCL_CUDA_ROOT to override.")
endif()

add_subdirectory(src/benchmark)
add_subdirectory(src/vector_add)
