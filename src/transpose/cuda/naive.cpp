#include <cuda_runtime.h>

#include <benchmark.hpp>
#include <cuda_commons.h>

#include <cmath>
#include <iostream>
#include <ranges>
#include <string>
#include <vector>

__global__ void transpose(const float *input, float *output, uint32_t rows,
                          uint32_t cols) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t rowIndex = idx / cols;
  const size_t colIndex = idx % cols;
  // output is cols x rows, so its row width is rows
  if (idx < static_cast<size_t>(rows) * cols)
    output[colIndex * rows + rowIndex] = input[rowIndex * cols + colIndex];
}

int main(int argc, char **argv) {

  Benchmark bench(argc, argv, getCudaDeviceName());

  const uint32_t count = bench.count();

  const std::vector<float> input = bench.generateWith(count, []() {
    static uint32_t index = 0;
    return index++;
  });

  const uint32_t rows = sqrt(count);
  const uint32_t cols = sqrt(count);

  float *deviceInputVector = nullptr;
  float *deviceOutputVector = nullptr;
  CUDA_CHECK(cudaMalloc(&deviceInputVector, bench.count() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&deviceOutputVector, bench.count() * sizeof(float)));

  constexpr int threads = 256;
  const int blocks = (rows * cols + threads - 1) / threads;

  CUDA_CHECK(cudaMemcpy(deviceInputVector, input.data(),
                        rows * cols * sizeof(float), cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  auto work = [&](bool verify) -> uint64_t {
    CUDA_CHECK(cudaEventRecord(start));
    transpose<<<blocks, threads>>>(deviceInputVector, deviceOutputVector, rows,
                                   cols);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    if (verify) {
      std::vector<float> hostOutputVector(count);
      CUDA_CHECK(cudaMemcpy(hostOutputVector.data(), deviceOutputVector,
                            rows * cols * sizeof(float),
                            cudaMemcpyDeviceToHost));
      for (auto row : std::views::iota(0u, rows)) {
        for (auto col : std::views::iota(0u, cols)) {
          float expected = col * cols + row;
          if (hostOutputVector[row * cols + col] != expected) {
            std::cerr << "Verification failed, expected value at " << row
                      << ", " << col << "(" << row * cols + col << " ) is "
                      << expected << " but got "
                      << hostOutputVector[row * cols + col] << '\n';
            std::exit(EXIT_FAILURE);
          }
        }
      }
    }

    return static_cast<uint64_t>(milliseconds * 1e6);
  };

  const uint64_t bytesPerIteration = 2 * rows * cols * sizeof(float);
  bench.run(work, bytesPerIteration);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(deviceInputVector));
  CUDA_CHECK(cudaFree(deviceOutputVector));

  return 0;
}