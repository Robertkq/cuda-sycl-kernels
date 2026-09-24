#include <cuda_runtime.h>

#include <benchmark.hpp>
#include <cuda_commons.h>

#include <cmath>
#include <iostream>
#include <ranges>
#include <string>
#include <vector>

constexpr uint32_t tileLength = 16;

__global__ void transpose(const float *input, float *output, uint32_t rows,
                          uint32_t cols) {
  const size_t groupCol = blockIdx.x;
  const size_t groupRow = blockIdx.y;
  const size_t localCol = threadIdx.x;
  const size_t localRow = threadIdx.y;
  const size_t globalCol = groupCol * tileLength + localCol;
  const size_t globalRow = groupRow * tileLength + localRow;

  // +1 pad: avoids bank conflicts on the column read below
  __shared__ float tile[tileLength][tileLength + 1];

  if (globalRow < rows && globalCol < cols)
    tile[localRow][localCol] = input[globalRow * cols + globalCol];

  // no early return above: every thread must reach the barrier
  __syncthreads();

  // mirrored tile position; output (cols x rows) has row width rows
  const size_t outRow = groupCol * tileLength + localRow;
  const size_t outCol = groupRow * tileLength + localCol;
  if (outRow < cols && outCol < rows)
    output[outRow * rows + outCol] = tile[localCol][localRow];
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
  const dim3 threads(tileLength, tileLength);
  const dim3 blocks((cols + tileLength - 1) / tileLength,
                    (rows + tileLength - 1) / tileLength);

  float *deviceInputVector = nullptr;
  float *deviceOutputVector = nullptr;
  CUDA_CHECK(cudaMalloc(&deviceInputVector, count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&deviceOutputVector, count * sizeof(float)));

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