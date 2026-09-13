#include <cuda_runtime.h>

#include <benchmark.hpp>

#include <cstdio>
#include <format>
#include <iostream>
#include <vector>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t err = (expr);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::cerr << cudaGetErrorString(err) << "\n";                            \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

void work(Benchmark &bench, bool record, bool verify) {
  constexpr float lhsValue = 1.0f;
  constexpr float rhsValue = 2.0f;
  constexpr float expectedValue = lhsValue + rhsValue;
  float *lhs = nullptr;
  float *rhs = nullptr;
  float *out = nullptr;
  CUDA_CHECK(cudaMalloc(&lhs, bench.count() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&rhs, bench.count() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&out, bench.count() * sizeof(float)));
}

int main(int argc, char **argv) {
  Benchmark bench(argc, argv);

  for (uint32_t i = 0; i < bench.warmups(); ++i) {
    work(bench, false, bench.verify());
  }

  for (uint32_t i = 0; i < bench.iterations(); ++i) {
    work(bench, true, bench.verify());
  }

  std::cout << std::format("Hello, CUDA!\n");
}