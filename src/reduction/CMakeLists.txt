
if(CMAKE_CUDA_COMPILER)
endif()

if(CMAKE_CUDA_COMPILER)
endif()

if(SYCL_CUDA_AVAILABLE)
  add_executable(reduction-naive-sycl sycl/naive.cpp)
  target_compile_options(reduction-naive-sycl PRIVATE ${SYCL_CUDA_FLAGS})
  target_link_options(reduction-naive-sycl PRIVATE ${SYCL_CUDA_FLAGS} -Wl,--disable-new-dtags)
  target_link_libraries(reduction-naive-sycl PUBLIC benchmark)
endif()

if(SYCL_CUDA_AVAILABLE)
endif()
