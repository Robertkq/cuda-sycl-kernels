
if(CMAKE_CUDA_COMPILER)
  add_executable(vector_add_naive_cuda cuda/naive.cpp)
  set_source_files_properties(cuda/naive.cpp PROPERTIES LANGUAGE CUDA)
  set_target_properties(vector_add_naive_cuda PROPERTIES LINKER_LANGUAGE CUDA)
  target_link_libraries(vector_add_naive_cuda PUBLIC benchmark)
endif()

if(SYCL_CUDA_AVAILABLE)
  add_executable(vector_add_naive_sycl sycl/naive.cpp)
  target_compile_options(vector_add_naive_sycl PRIVATE ${SYCL_CUDA_FLAGS})
  target_link_options(vector_add_naive_sycl PRIVATE ${SYCL_CUDA_FLAGS} -Wl,--disable-new-dtags)
  target_link_libraries(vector_add_naive_sycl PUBLIC benchmark)
endif()
