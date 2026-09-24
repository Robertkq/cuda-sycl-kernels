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
  Benchmark bench(argc, argv,
                  q.get_device().get_info<sycl::info::device::name>());

  const uint32_t count = bench.count();

  const std::vector<float> input = bench.generateWith(count, []() {
    static uint32_t index = 0;
    return index++;
  });

  const uint32_t rows = sqrt(count);
  const uint32_t cols = sqrt(count);

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
    return event
               .get_profiling_info<sycl::info::event_profiling::command_end>() -
           event.get_profiling_info<
               sycl::info::event_profiling::command_start>();
  };

  const uint64_t bytesPerIteration = 2 * rows * cols * sizeof(float);
  bench.run(work, bytesPerIteration);
  sycl::free(deviceInputVector, q);
  sycl::free(deviceOutputVector, q);
}
