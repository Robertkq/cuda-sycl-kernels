#include <sycl/sycl.hpp>

#include <benchmark.hpp>

#include <iostream>
#include <math.h>
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

  constexpr size_t tileLength = 16;

  float *deviceInputVector = sycl::malloc_device<float>(count, q);
  float *deviceOutputVector = sycl::malloc_device<float>(count, q);

  q.memcpy(deviceInputVector, input.data(), count * sizeof(float))
      .wait_and_throw();

  auto work = [&](bool verify) {
    auto event = q.submit([&](sycl::handler &h) {
      sycl::local_accessor<float, 2> tile(
          sycl::range<2>(tileLength, tileLength + 1), h);

      const size_t paddedRows =
          (rows + tileLength - 1) / tileLength * tileLength;
      const size_t paddedCols =
          (cols + tileLength - 1) / tileLength * tileLength;

      h.parallel_for(sycl::nd_range<2>(sycl::range<2>(paddedRows, paddedCols),
                                       sycl::range<2>(tileLength, tileLength)),
                     [=](sycl::nd_item<2> item) {
                       auto localRow = item.get_local_id(0);
                       auto localCol = item.get_local_id(1);
                       auto globalRow = item.get_global_id(0);
                       auto globalCol = item.get_global_id(1);
                       auto groupRow = item.get_group(0);
                       auto groupCol = item.get_group(1);

                       if (globalRow < rows && globalCol < cols) {
                         tile[localRow][localCol] =
                             deviceInputVector[globalRow * cols + globalCol];
                       }

                       sycl::group_barrier(item.get_group());

                       auto outRow = groupCol * tileLength + localRow;
                       auto outCol = groupRow * tileLength + localCol;

                       if (outRow < cols && outCol < rows) {
                         deviceOutputVector[outRow * rows + outCol] =
                             tile[localCol][localRow];
                       }
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
