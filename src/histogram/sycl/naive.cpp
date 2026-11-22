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

  uint32_t count = 1 << 27;
  Benchmark bench(argc, argv,
                  q.get_device().get_info<sycl::info::device::name>(),
                  [&](BenchmarkOptions &options) {
                    options.add("-c,--count", count, "Number of elements");
                  });

  const auto binCount = 256;
  const auto workGroupSize = 256;
  const auto numGroups = (count + workGroupSize - 1) / workGroupSize;

  auto deviceInputVec = sycl::malloc_device<uint8_t>(count, q);
  auto deviceOutputVec = sycl::malloc_device<uint32_t>(binCount, q);
  if (!deviceInputVec || !deviceOutputVec) {
    std::cerr << "Device allocation failed\n";
    sycl::free(deviceInputVec, q);
    sycl::free(deviceOutputVec, q);
    std::exit(EXIT_FAILURE);
  }

  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, 255);
  const auto hostInput = bench.generateWith<uint8_t>(
      count, [&] { return static_cast<uint8_t>(dist(rng)); });

  q.memcpy(deviceInputVec, hostInput.data(), count * sizeof(uint8_t))
      .wait_and_throw();

  // expected bins, computed once on the CPU
  std::vector<uint32_t> expectedBins(binCount, 0);
  for (const uint8_t value : hostInput) {
    ++expectedBins[value];
  }

  auto work = [&](bool verify) -> uint64_t {
    q.fill<uint32_t>(deviceOutputVec, 0, binCount).wait_and_throw();

    auto event = q.submit([&](sycl::handler &h) {
      h.parallel_for(
          sycl::nd_range<1>(numGroups * workGroupSize, workGroupSize),
          [=](sycl::nd_item<1> item) {
            const auto globalId = item.get_global_id(0);

            if (globalId < count) {
              const uint8_t value = deviceInputVec[globalId];

              sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                               sycl::memory_scope::device,
                               sycl::access::address_space::global_space>
                  bin(deviceOutputVec[value]);

              bin.fetch_add(1);
            }
          });
    });

    event.wait_and_throw();

    if (verify) {
      std::vector<uint32_t> hostBins(binCount);
      q.memcpy(hostBins.data(), deviceOutputVec, binCount * sizeof(uint32_t))
          .wait_and_throw();

      for (size_t bin = 0; bin < binCount; ++bin) {
        if (hostBins[bin] != expectedBins[bin]) {
          std::cerr << "Verification failed at bin " << bin << ": expected "
                    << expectedBins[bin] << " but got " << hostBins[bin]
                    << '\n';
          std::exit(EXIT_FAILURE);
        }
      }
    }

    return event
               .get_profiling_info<sycl::info::event_profiling::command_end>() -
           event.get_profiling_info<
               sycl::info::event_profiling::command_start>();
  };

  // read every input byte once, write the bins once
  const uint64_t bytesPerIteration =
      static_cast<uint64_t>(count) * sizeof(uint8_t) +
      binCount * sizeof(uint32_t);
  // one add per input element
  const uint64_t flopsPerIteration = count;
  bench.run(work, bytesPerIteration, flopsPerIteration);

  sycl::free(deviceInputVec, q);
  sycl::free(deviceOutputVec, q);
}
