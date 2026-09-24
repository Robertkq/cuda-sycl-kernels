#include <sycl/sycl.hpp>

#include <benchmark.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <ranges>
#include <string>
#include <vector>

constexpr uint32_t tileLength = 16;

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

  // {rows, cols}: dim 1 is the fast index (CUDA x)
  const size_t paddedRows = (rows + tileLength - 1) / tileLength * tileLength;
  const size_t paddedCols = (cols + tileLength - 1) / tileLength * tileLength;
  const sycl::nd_range<2> ndRange(sycl::range<2>(paddedRows, paddedCols),
                                  sycl::range<2>(tileLength, tileLength));

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
      // +1 pad: avoids bank conflicts on the column read below
      sycl::local_accessor<float, 2> tile(
          sycl::range<2>(tileLength, tileLength + 1), h);

      h.parallel_for(ndRange, [=](sycl::nd_item<2> item) {
        const size_t localRow = item.get_local_id(0);
        const size_t localCol = item.get_local_id(1);
        const size_t globalRow = item.get_global_id(0);
        const size_t globalCol = item.get_global_id(1);
        const size_t groupRow = item.get_group(0);
        const size_t groupCol = item.get_group(1);

        if (globalRow < rows && globalCol < cols) {
          tile[localRow][localCol] =
              deviceInputVector[globalRow * cols + globalCol];
        }

        // no early return above: every work-item must reach the barrier
        sycl::group_barrier(item.get_group());

        // mirrored tile position; output (cols x rows) has row width rows
        const size_t outRow = groupCol * tileLength + localRow;
        const size_t outCol = groupRow * tileLength + localCol;
        if (outRow < cols && outCol < rows) {
          deviceOutputVector[outRow * rows + outCol] = tile[localCol][localRow];
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
