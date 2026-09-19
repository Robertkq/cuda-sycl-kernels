#include <sycl/sycl.hpp>

#include <benchmark.hpp>

#include <cstdlib>
#include <iostream>

int main(int argc, char **argv) {
  sycl::queue q({sycl::property::queue::enable_profiling()});
  Benchmark bench(argc, argv,
                  q.get_device().get_info<sycl::info::device::name>());
  constexpr float elementValue = 1.0f;
  const float expectedSum = elementValue * static_cast<float>(bench.count());

  float *deviceVector = sycl::malloc_device<float>(bench.count(), q);
  float *deviceSum = sycl::malloc_device<float>(1, q);
  if (!deviceVector || !deviceSum) {
    std::cerr << "Device allocation failed\n";
    sycl::free(deviceVector, q);
    sycl::free(deviceSum, q);
    std::exit(EXIT_FAILURE);
  }

  q.fill<float>(deviceVector, elementValue, bench.count()).wait_and_throw();

  constexpr uint32_t workGroupSize = 256;
  const uint32_t count = bench.count();
  const uint32_t numGroups = (count + workGroupSize - 1) / workGroupSize;
  const sycl::nd_range<1> ndRange(sycl::range<1>(numGroups * workGroupSize),
                                  sycl::range<1>(workGroupSize));

  auto work = [&](bool verify) -> uint64_t {
    // Reset the accumulator before each timed run
    q.fill<float>(deviceSum, 0.0f, 1).wait_and_throw();

    auto event = q.submit([&](sycl::handler &h) {
      sycl::local_accessor<float, 1> groupSum(sycl::range<1>(1), h);

      h.parallel_for(ndRange, [=](sycl::nd_item<1> item) {
        const size_t globalId = item.get_global_id(0);
        const size_t localId = item.get_local_id(0);

        if (localId == 0) {
          groupSum[0] = 0.0f;
        }
        sycl::group_barrier(item.get_group());

        if (globalId < count) {
          sycl::atomic_ref<float, sycl::memory_order::relaxed,
                           sycl::memory_scope::work_group,
                           sycl::access::address_space::local_space>
              atomicGroupSum(groupSum[0]);
          atomicGroupSum += deviceVector[globalId];
        }
        sycl::group_barrier(item.get_group());

        if (localId == 0) {
          sycl::atomic_ref<float, sycl::memory_order::relaxed,
                           sycl::memory_scope::device,
                           sycl::access::address_space::global_space>
              atomicSum(*deviceSum);
          atomicSum += groupSum[0];
        }
      });
    });

    event.wait_and_throw();

    if (verify) {
      float hostSum = 0.0f;
      q.memcpy(&hostSum, deviceSum, sizeof(float)).wait_and_throw();
      if (hostSum != expectedSum) {
        std::cerr << "Verification failed! Expected " << expectedSum << ", got "
                  << hostSum << "\n";
        std::exit(EXIT_FAILURE);
      }
    }

    return event
               .get_profiling_info<sycl::info::event_profiling::command_end>() -
           event.get_profiling_info<
               sycl::info::event_profiling::command_start>();
  };

  const uint64_t bytesPerIteration =
      (static_cast<uint64_t>(bench.count()) + 1) * sizeof(float);
  bench.run(work, bytesPerIteration);

  sycl::free(deviceVector, q);
  sycl::free(deviceSum, q);
}
