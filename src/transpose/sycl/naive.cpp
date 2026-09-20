#include <sycl/sycl.hpp>

#include <benchmark.hpp>

#include <iostream>
#include <math.h>
#include <ranges>
#include <string>
#include <vector>

// Playground only -- not part of the transpose kernel. Just here to make
// work-item / work-group ids concrete by printing them.
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

  q.memcpy(deviceInputVector, input.data(), count * sizeof(float))
      .wait_and_throw();

  auto work = [&](bool verify) {
    auto event = q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::nd_range<1>(totalItems, groupSize),
                     [=](sycl::nd_item<1> item) {
                       auto localId = item.get_local_id(0);
                       auto globalId = item.get_global_id(0);
                       auto rowIndex = globalId / cols;
                       auto colIndex = globalId % cols;

                       if (globalId < rows * cols)
                         deviceOutputVector[colIndex * cols + rowIndex] =
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
}
