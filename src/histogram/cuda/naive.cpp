#include <cuda_runtime.h>

#include <benchmark.hpp>
#include <cuda_commons.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

__global__ void histogram(const uint8_t *input, uint32_t *output,
                          size_t count) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    const uint8_t value = input[idx];
    atomicAdd(&output[value], 1u);
  }
}

int main(int argc, char **argv) {

  uint32_t count = 1 << 27;
  Benchmark bench(argc, argv, getCudaDeviceName(),
                  [&](BenchmarkOptions &options) {
                    options.add("-c,--count", count, "Number of elements");
                  });

  const auto binCount = 256;
  const auto workGroupSize = 256;
  const auto numGroups = (count + workGroupSize - 1) / workGroupSize;

  uint8_t *deviceInputVec = nullptr;
  uint32_t *deviceOutputVec = nullptr;
  CUDA_CHECK(cudaMalloc(&deviceInputVec, count * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc(&deviceOutputVec, binCount * sizeof(uint32_t)));

  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, 255);
  const auto hostInput = bench.generateWith<uint8_t>(
      count, [&] { return static_cast<uint8_t>(dist(rng)); });

  CUDA_CHECK(cudaMemcpy(deviceInputVec, hostInput.data(),
                        count * sizeof(uint8_t), cudaMemcpyHostToDevice));

  // expected bins, computed once on the CPU
  std::vector<uint32_t> expectedBins(binCount, 0);
  for (const uint8_t value : hostInput) {
    ++expectedBins[value];
  }

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  auto work = [&](bool verify) -> uint64_t {
    CUDA_CHECK(cudaMemset(deviceOutputVec, 0, binCount * sizeof(uint32_t)));

    CUDA_CHECK(cudaEventRecord(start));
    histogram<<<numGroups, workGroupSize>>>(deviceInputVec, deviceOutputVec,
                                            count);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    if (verify) {
      std::vector<uint32_t> hostBins(binCount);
      CUDA_CHECK(cudaMemcpy(hostBins.data(), deviceOutputVec,
                            binCount * sizeof(uint32_t),
                            cudaMemcpyDeviceToHost));

      for (size_t bin = 0; bin < binCount; ++bin) {
        if (hostBins[bin] != expectedBins[bin]) {
          std::cerr << "Verification failed at bin " << bin << ": expected "
                    << expectedBins[bin] << " but got " << hostBins[bin]
                    << '\n';
          std::exit(EXIT_FAILURE);
        }
      }
    }

    return static_cast<uint64_t>(milliseconds * 1e6);
  };

  // read every input byte once, write the bins once
  const uint64_t bytesPerIteration =
      static_cast<uint64_t>(count) * sizeof(uint8_t) +
      binCount * sizeof(uint32_t);
  // one add per input element
  const uint64_t flopsPerIteration = count;
  bench.run(work, bytesPerIteration, flopsPerIteration);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(deviceInputVec));
  CUDA_CHECK(cudaFree(deviceOutputVec));

  return 0;
}
