#pragma once

#include <cuda_runtime.h>

#include <iostream>
#include <string>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t err = (expr);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::cerr << cudaGetErrorString(err) << "\n";                            \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

inline std::string getCudaDeviceName() {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  return std::string(prop.name);
}
