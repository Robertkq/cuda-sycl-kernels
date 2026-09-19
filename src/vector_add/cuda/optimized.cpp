#include <cuda_runtime.h>

#include <benchmark.hpp>
#include <cuda_commons.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

__global__ void fill(float4 *data, float value, size_t count) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    data[idx] = make_float4(value, value, value, value);
  }
}

__global__ void vectorAdd(float4 *lhs, float4 *rhs, float4 *out, size_t count) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    float4 a = lhs[idx];
    float4 b = rhs[idx];
    out[idx] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
  }
}

int main(int argc, char **argv) {
  Benchmark bench(argc, argv, getCudaDeviceName());

  if (bench.count() % 4 != 0) {
    std::cerr << "Count must be a multiple of 4 for the vectorized kernel\n";
    std::exit(EXIT_FAILURE);
  }

  constexpr int threads = 256;
  const int blocks = static_cast<int>(
      (static_cast<uint64_t>(bench.count() / 4) + threads - 1) / threads);

  constexpr float lhsValue = 1.0f;
  constexpr float rhsValue = 2.0f;
  constexpr float expectedValue = lhsValue + rhsValue;
  float *lhs = nullptr;
  float *rhs = nullptr;
  float *out = nullptr;
  CUDA_CHECK(cudaMalloc(&lhs, bench.count() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&rhs, bench.count() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&out, bench.count() * sizeof(float)));
  fill<<<blocks, threads>>>(reinterpret_cast<float4 *>(lhs), lhsValue,
                            bench.count() / 4);
  fill<<<blocks, threads>>>(reinterpret_cast<float4 *>(rhs), rhsValue,
                            bench.count() / 4);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  auto work = [&](bool verify) -> uint64_t {
    CUDA_CHECK(cudaEventRecord(start));
    vectorAdd<<<blocks, threads>>>(
        reinterpret_cast<float4 *>(lhs), reinterpret_cast<float4 *>(rhs),
        reinterpret_cast<float4 *>(out), bench.count() / 4);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    if (verify) {
      std::vector<float> hostOut(bench.count());
      CUDA_CHECK(cudaMemcpy(hostOut.data(), out, bench.count() * sizeof(float),
                            cudaMemcpyDeviceToHost));
      if (!std::all_of(
              hostOut.begin(), hostOut.end(),
              [expectedValue](float v) { return v == expectedValue; })) {
        std::cerr << "Verification failed!\n";
        std::exit(EXIT_FAILURE);
      }
    }

    return static_cast<uint64_t>(milliseconds * 1e6);
  };
  const uint64_t bytesPerIteration =
      3 * static_cast<uint64_t>(bench.count()) * sizeof(float);
  bench.run(work, bytesPerIteration);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(lhs));
  CUDA_CHECK(cudaFree(rhs));
  CUDA_CHECK(cudaFree(out));
}
