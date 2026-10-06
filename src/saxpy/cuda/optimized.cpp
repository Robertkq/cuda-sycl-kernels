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

__global__ void saxpy(float a, const float4 *x, const float4 *y, float4 *out,
                      size_t count) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    float4 xv = x[idx];
    float4 yv = y[idx];
    out[idx] = make_float4(a * xv.x + yv.x, a * xv.y + yv.y, a * xv.z + yv.z,
                           a * xv.w + yv.w);
  }
}

int main(int argc, char **argv) {
  uint32_t count = 1 << 27;
  float alpha = 2.0f;
  Benchmark bench(argc, argv, getCudaDeviceName(),
                  [&](BenchmarkOptions &options) {
                    options.add("-c,--count", count, "Number of elements");
                    options.add("-a,--alpha", alpha, "Scalar a in a * x + y");
                  });

  if (count % 4 != 0) {
    std::cerr << "Count must be a multiple of 4 for the vectorized kernel\n";
    std::exit(EXIT_FAILURE);
  }

  constexpr int threads = 256;
  const int blocks = static_cast<int>(
      (static_cast<uint64_t>(count / 4) + threads - 1) / threads);

  const float a = alpha;
  constexpr float xValue = 1.0f;
  constexpr float yValue = 2.0f;
  const float expectedValue = a * xValue + yValue;
  float *x = nullptr;
  float *y = nullptr;
  float *out = nullptr;
  CUDA_CHECK(cudaMalloc(&x, count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&y, count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&out, count * sizeof(float)));
  fill<<<blocks, threads>>>(reinterpret_cast<float4 *>(x), xValue, count / 4);
  fill<<<blocks, threads>>>(reinterpret_cast<float4 *>(y), yValue, count / 4);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  auto work = [&](bool verify) -> uint64_t {
    CUDA_CHECK(cudaEventRecord(start));
    saxpy<<<blocks, threads>>>(a, reinterpret_cast<const float4 *>(x),
                               reinterpret_cast<const float4 *>(y),
                               reinterpret_cast<float4 *>(out), count / 4);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    if (verify) {
      std::vector<float> hostOut(count);
      CUDA_CHECK(cudaMemcpy(hostOut.data(), out, count * sizeof(float),
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
      3 * static_cast<uint64_t>(count) * sizeof(float);
  // one multiply and one add per element
  const uint64_t flopsPerIteration = 2 * static_cast<uint64_t>(count);
  bench.run(work, bytesPerIteration, flopsPerIteration);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(x));
  CUDA_CHECK(cudaFree(y));
  CUDA_CHECK(cudaFree(out));
}
