#include <sycl/sycl.hpp>

#include <benchmark.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <ranges>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  sycl::queue q({sycl::property::queue::enable_profiling()});
  uint32_t rows = 8192;
  uint32_t cols = 8192;
  Benchmark bench(argc, argv,
                  q.get_device().get_info<sycl::info::device::name>(),
                  [&](BenchmarkOptions &options) {
                    options.add("--rows", rows, "Rows of the input matrix");
                    options.add("--cols", cols, "Columns of the input matrix");
                  });

  const size_t count = static_cast<size_t>(rows) * cols;

  const std::vector<float> input = bench.generateWith(count, []() {
    static uint32_t index = 0;
    return index++;
  });

  constexpr size_t groupSize = 256;
  const size_t numGroups = (rows * cols + groupSize - 1) / groupSize;
  const size_t totalItems = numGroups * groupSize;

  float *deviceInputVector = sycl::malloc_device<float>(count, q);
  float *deviceOutputVector = sycl::malloc_device<float>(count, q);
  if (!deviceInputVector || !deviceOutputVector) {
    std::cerr << "Device allocation failed\n";
    sycl::free(deviceInputVector, q);
    sycl::free(deviceOutputVector, q);
    std::exit(EXIT_FAILURE);
  }

  q.memcpy(deviceInputVector, input.data(), count * sizeof(float))
      .wait_and_throw();

  auto work = [&](bool verify) {
    auto event = q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::nd_range<1>(totalItems, groupSize),
                     [=](sycl::nd_item<1> item) {
                       const size_t globalId = item.get_global_id(0);
                       const size_t rowIndex = globalId / cols;
                       const size_t colIndex = globalId % cols;

                       // output is cols x rows, so its row width is rows
                       if (globalId < rows * cols)
                         deviceOutputVector[colIndex * rows + rowIndex] =
                             deviceInputVector[rowIndex * cols + colIndex];
                     });
    });
    event.wait_and_throw();
    if (verify) {
      std::vector<float> hostOutputVector(count);
      q.memcpy(hostOutputVector.data(), deviceOutputVector,
               count * sizeof(float))
          .wait_and_throw();
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
    return event
               .get_profiling_info<sycl::info::event_profiling::command_end>() -
           event.get_profiling_info<
               sycl::info::event_profiling::command_start>();
  };

  const uint64_t bytesPerIteration = 2 * count * sizeof(float);
  bench.run(work, bytesPerIteration, 0);
  sycl::free(deviceInputVector, q);
  sycl::free(deviceOutputVector, q);
}
