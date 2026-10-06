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
