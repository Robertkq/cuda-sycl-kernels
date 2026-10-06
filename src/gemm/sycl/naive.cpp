#include <sycl/sycl.hpp>

#include <benchmark.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <ranges>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  sycl::queue q({sycl::property::queue::enable_profiling()});
  // C (rows x cols) = A (rows x inner) * B (inner x cols)
  uint32_t rows = 4096;
  uint32_t cols = 4096;
  uint32_t inner = 4096;
  Benchmark bench(argc, argv,
                  q.get_device().get_info<sycl::info::device::name>(),
                  [&](BenchmarkOptions &options) {
                    options.add("--rows", rows, "Rows of A and C");
                    options.add("--cols", cols, "Columns of B and C");
                    options.add("--inner", inner, "Columns of A, rows of B");
                  });

  // values are at most 7, so every sum stays below 49 * inner; it has to stay
  // below 2^24 for float results to be exact
  if (49ull * inner >= (1ull << 24)) {
    std::cerr << "--inner must be below " << (1ull << 24) / 49
              << " for exact verification\n";
    std::exit(EXIT_FAILURE);
  }

  const size_t sizeA = static_cast<size_t>(rows) * inner;
  const size_t sizeB = static_cast<size_t>(inner) * cols;
  const size_t sizeC = static_cast<size_t>(rows) * cols;

  // small values (0..7) keep every sum below 2^24, so float results are exact;
  // A and B use different patterns so mixing them up fails verification
  const std::vector<float> hostA = bench.generateWith(sizeA, []() {
    static uint32_t index = 0;
    return index++ % 8;
  });
  const std::vector<float> hostB = bench.generateWith(sizeB, []() {
    static uint32_t index = 0;
    return (index++ * 3) % 8;
  });

  const size_t workGroupSize = 16; // 2D so it's actually 256
  const size_t rowsRoundedToWorkGroupSize =
      (rows + workGroupSize - 1) / workGroupSize * workGroupSize;
  const size_t colsRoundedToWorkGroupSize =
      (cols + workGroupSize - 1) / workGroupSize * workGroupSize;

  float *deviceA = sycl::malloc_device<float>(sizeA, q);
  float *deviceB = sycl::malloc_device<float>(sizeB, q);
  float *deviceC = sycl::malloc_device<float>(sizeC, q);
  if (!deviceA || !deviceB || !deviceC) {
    std::cerr << "Device allocation failed\n";
    sycl::free(deviceA, q);
    sycl::free(deviceB, q);
    sycl::free(deviceC, q);
    std::exit(EXIT_FAILURE);
  }

  q.memcpy(deviceA, hostA.data(), sizeA * sizeof(float)).wait_and_throw();
  q.memcpy(deviceB, hostB.data(), sizeB * sizeof(float)).wait_and_throw();

  auto work = [&](bool verify) -> uint64_t {
    auto event = q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::nd_range<2>({rowsRoundedToWorkGroupSize,
                                        colsRoundedToWorkGroupSize},
                                       {workGroupSize, workGroupSize}),
                     [=](sycl::nd_item<2> item) {
                       const size_t globalRow = item.get_global_id(0);
                       const size_t globalCol = item.get_global_id(1);
                       const size_t localRow = item.get_local_id(0);
                       const size_t localCol = item.get_local_id(1);
                       const size_t groupX = item.get_group(0);
                       const size_t groupY = item.get_group(1);

                       if (globalRow < rows && globalCol < cols) {
                         float sum = 0.0f;
                         for (size_t k = 0; k < inner; ++k) {
                           sum += deviceA[globalRow * inner + k] *
                                  deviceB[k * cols + globalCol];
                         }
                         const size_t globalOutputIndex =
                             globalRow * cols + globalCol;
                         deviceC[globalOutputIndex] = sum;
                       }
                     });
    });
    event.wait_and_throw();
    if (verify) {
      std::vector<float> hostC(sizeC);
      q.memcpy(hostC.data(), deviceC, sizeC * sizeof(float)).wait_and_throw();

      // a full CPU GEMM would take minutes, so check a fixed sample:
      // the four corners (partial edge tiles) plus random positions
      std::vector<std::pair<size_t, size_t>> samples = {
          {0, 0}, {0, cols - 1}, {rows - 1, 0}, {rows - 1, cols - 1}};
      std::mt19937 rng(42);
      std::uniform_int_distribution<size_t> pickRow(0, rows - 1);
      std::uniform_int_distribution<size_t> pickCol(0, cols - 1);
      for (int i = 0; i < 256; ++i) {
        samples.emplace_back(pickRow(rng), pickCol(rng));
      }

      for (const auto [row, col] : samples) {
        int64_t expected = 0;
        for (size_t k = 0; k < inner; ++k) {
          expected += static_cast<int64_t>(hostA[row * inner + k]) *
                      static_cast<int64_t>(hostB[k * cols + col]);
        }
        const float got = hostC[row * cols + col];
        if (got != static_cast<float>(expected)) {
          std::cerr << "Verification failed at (" << row << ", " << col
                    << "): expected " << expected << " but got " << got << '\n';
          std::exit(EXIT_FAILURE);
        }
      }
    }
    return event
               .get_profiling_info<sycl::info::event_profiling::command_end>() -
           event.get_profiling_info<
               sycl::info::event_profiling::command_start>();
  };

  // read A and B once, write C once (the rounded-up launch grid moves no data)
  const uint64_t bytesPerIteration = (sizeA + sizeB + sizeC) * sizeof(float);
  // one multiply and one add per inner step, for each of the rows * cols
  // outputs
  const uint64_t flopsPerIteration = 2 * sizeC * inner;
  bench.run(work, bytesPerIteration, flopsPerIteration);

  sycl::free(deviceA, q);
  sycl::free(deviceB, q);
  sycl::free(deviceC, q);
}
