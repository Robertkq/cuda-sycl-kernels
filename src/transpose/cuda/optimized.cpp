#include <cuda_runtime.h>

#include <benchmark.hpp>
#include <cuda_commons.h>

#include <cmath>
#include <iostream>
#include <ranges>
#include <string>
#include <vector>

// Each block handles one tileLength x tileLength tile: one thread per element.
constexpr uint32_t tileLength = 16;

__global__ void transpose(const float *input, float *output, uint32_t rows,
                          uint32_t cols) {
  // CUDA x is the fast (column) index, y the slow (row) index.
  const size_t groupCol = blockIdx.x;
  const size_t groupRow = blockIdx.y;
  const size_t localCol = threadIdx.x;
  const size_t localRow = threadIdx.y;
  const size_t globalCol = groupCol * tileLength + localCol;
  const size_t globalRow = groupRow * tileLength + localRow;

  // +1 column of padding: without it, every element of a tile column sits in
  // the same shared-memory bank, and the column read below would make threads
  // queue up (bank conflict). The extra float shifts each row by one bank.
  __shared__ float tile[tileLength][tileLength + 1];

  // Load: neighbouring threads (localCol) read neighbouring floats of one
  // input row, so the global read is coalesced.
  if (globalRow < rows && globalCol < cols)
    tile[localRow][localCol] = input[globalRow * cols + globalCol];

  // Every thread must reach this, so the bounds checks above and below stay
  // around the memory access instead of returning early.
  __syncthreads();

  // Store: tile (groupRow, groupCol) goes to tile (groupCol, groupRow) of the
  // output. Neighbouring threads (localCol) write neighbouring floats of one
  // output row, so the global write is coalesced too; the column-wise access
  // happens in shared memory instead (tile[localCol][localRow]).
  // Output is cols x rows, so its row width is rows.
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
  // dim3 is (x, y): x = columns (fast), y = rows (slow). The grid counts
  // blocks, rounded up so partial edge tiles are covered.
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