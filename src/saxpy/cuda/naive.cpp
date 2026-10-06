#include <cuda_runtime.h>

#include <benchmark.hpp>
#include <cuda_commons.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

__global__ void fill(float *data, float value, size_t count) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    data[idx] = value;
  }
}

__global__ void saxpy(float a, const float *x, const float *y, float *out,
                      size_t count) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    out[idx] = a * x[idx] + y[idx];
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

  constexpr int threads = 256;
  const int blocks =
      static_cast<int>((static_cast<uint64_t>(count) + threads - 1) / threads);

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
  fill<<<blocks, threads>>>(x, xValue, count);
  fill<<<blocks, threads>>>(y, yValue, count);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  auto work = [&](bool verify) -> uint64_t {
    CUDA_CHECK(cudaEventRecord(start));
    saxpy<<<blocks, threads>>>(a, x, y, out, count);
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
