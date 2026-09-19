#include <cuda_runtime.h>

#include <benchmark.hpp>
#include <cuda_commons.h>

constexpr int blockSize = 256;

__global__ void fill(float *data, float value, size_t count) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    data[idx] = value;
  }
}

__global__ void firstTreeReduction(const float *deviceInputVector,
                                   float *deviceOutputVector, size_t count) {
  __shared__ float sdata[blockSize];

  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  sdata[threadIdx.x] = (idx < count) ? deviceInputVector[idx] : 0.0f;
  __syncthreads();

  for (int stride = blockSize / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      sdata[threadIdx.x] += sdata[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    deviceOutputVector[blockIdx.x] = sdata[0];
  }
}

__global__ void secondTreeReduction(const float *deviceOutputVector,
                                    float *deviceSum, size_t blocks) {
  __shared__ float sdata[blockSize];

  float partial = 0.0f;
  for (size_t i = threadIdx.x; i < blocks; i += blockSize) {
    partial += deviceOutputVector[i];
  }
  sdata[threadIdx.x] = partial;
  __syncthreads();

  for (int stride = blockSize / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      sdata[threadIdx.x] += sdata[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    *deviceSum = sdata[0];
  }
}

int main(int argc, char **argv) {

  Benchmark bench(argc, argv, getCudaDeviceName());

  const int blocks = static_cast<int>(
      (static_cast<uint64_t>(bench.count()) + blockSize - 1) / blockSize);

  float *deviceInputVector = nullptr;
  float *deviceOutputVector = nullptr;
  float *deviceSum = nullptr;
  CUDA_CHECK(cudaMalloc(&deviceInputVector, bench.count() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&deviceOutputVector, blocks * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&deviceSum, sizeof(float)));

  fill<<<blocks, blockSize>>>(deviceInputVector, 1.0f, bench.count());

  constexpr float elementValue = 1.0f;
  const float expectedSum = elementValue * static_cast<float>(bench.count());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  auto work = [&](bool verify) -> uint64_t {
    CUDA_CHECK(cudaMemset(deviceSum, 0, sizeof(float)));

    CUDA_CHECK(cudaEventRecord(start));
    firstTreeReduction<<<blocks, blockSize>>>(
        deviceInputVector, deviceOutputVector, bench.count());
    secondTreeReduction<<<1, blockSize>>>(deviceOutputVector, deviceSum,
                                          static_cast<size_t>(blocks));
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

  const uint64_t bytesPerIteration = (static_cast<uint64_t>(bench.count()) +
                                      2 * static_cast<uint64_t>(blocks) + 1) *
                                     sizeof(float);
  bench.run(work, bytesPerIteration);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(deviceInputVector));
  CUDA_CHECK(cudaFree(deviceOutputVector));
  CUDA_CHECK(cudaFree(deviceSum));

  return 0;
}
