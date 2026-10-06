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

  uint32_t rows = 8192;
  uint32_t cols = 8192;
  Benchmark bench(argc, argv, getCudaDeviceName(),
                  [&](BenchmarkOptions &options) {
                    options.add("--rows", rows, "Rows of the input matrix");
                    options.add("--cols", cols, "Columns of the input matrix");
                  });

  const size_t count = static_cast<size_t>(rows) * cols;

  const std::vector<float> input = bench.generateWith(count, []() {
    static uint32_t index = 0;
    return index++;
  });

  const dim3 threads(tileLength, tileLength);
  const dim3 blocks((cols + tileLength - 1) / tileLength,
                    (rows + tileLength - 1) / tileLength);

  float *deviceInputVector = nullptr;
  float *deviceOutputVector = nullptr;
  CUDA_CHECK(cudaMalloc(&deviceInputVector, count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&deviceOutputVector, count * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(deviceInputVector, input.data(), count * sizeof(float),
                        cudaMemcpyHostToDevice));

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
                            count * sizeof(float), cudaMemcpyDeviceToHost));
      // input[row][col] must end up at output[col][row]; the output is
      // cols x rows, so its row width is rows
      for (auto row : std::views::iota(0u, rows)) {
        for (auto col : std::views::iota(0u, cols)) {
          const float expected = input[static_cast<size_t>(row) * cols + col];
          const float got =
              hostOutputVector[static_cast<size_t>(col) * rows + row];
          if (got != expected) {
            std::cerr << "Verification failed at input (" << row << ", " << col
                      << "): expected " << expected << " but got " << got
                      << '\n';
            std::exit(EXIT_FAILURE);
          }
        }
      }
    }

    return static_cast<uint64_t>(milliseconds * 1e6);
  };

  const uint64_t bytesPerIteration = 2 * count * sizeof(float);
  bench.run(work, bytesPerIteration, 0);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(deviceInputVector));
  CUDA_CHECK(cudaFree(deviceOutputVector));

  return 0;
}