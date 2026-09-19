#include <cuda_runtime.h>

#include <benchmark.hpp>
#include <cuda_commons.h>

__global__ void fill(float *data, float value, size_t count) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    data[idx] = value;
  }
}

__global__ void reduction(const float *vector, float *sum, size_t count) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float groupSum;
  if (threadIdx.x == 0) {
    groupSum = 0.0f;
  }
  __syncthreads();

  if (idx < count) {
    atomicAdd(&groupSum, vector[idx]);
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(sum, groupSum);
  }
}

int main(int argc, char **argv) {

  Benchmark bench(argc, argv, getCudaDeviceName());

  float *deviceVector = nullptr;
  float *deviceSum = nullptr;
  CUDA_CHECK(cudaMalloc(&deviceVector, bench.count() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&deviceSum, sizeof(float)));

  constexpr int threads = 256;
  const int blocks = static_cast<int>(
      (static_cast<uint64_t>(bench.count()) + threads - 1) / threads);
  fill<<<blocks, threads>>>(deviceVector, 1.0f, bench.count());

  constexpr float elementValue = 1.0f;
  const float expectedSum = elementValue * static_cast<float>(bench.count());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  auto work = [&](bool verify) -> uint64_t {
    CUDA_CHECK(cudaMemset(deviceSum, 0, sizeof(float)));

    CUDA_CHECK(cudaEventRecord(start));
    reduction<<<blocks, threads>>>(deviceVector, deviceSum, bench.count());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    if (verify) {
      float hostSum = 0.0f;
      CUDA_CHECK(cudaMemcpy(&hostSum, deviceSum, sizeof(float),
                            cudaMemcpyDeviceToHost));
      if (hostSum != expectedSum) {
        std::cerr << "Verification failed! Expected: " << expectedSum
                  << ", Got: " << hostSum << "\n";
        std::exit(EXIT_FAILURE);
      }
    }

    return static_cast<uint64_t>(milliseconds * 1e6);
  };

  const uint64_t bytesPerIteration =
      bench.count() * sizeof(float) + sizeof(float);
  bench.run(work, bytesPerIteration);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(deviceVector));
  CUDA_CHECK(cudaFree(deviceSum));

  return 0;
}