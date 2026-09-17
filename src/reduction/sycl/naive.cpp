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

  auto work = [&](bool verify) -> uint64_t {
    // Reset the accumulator before each timed run; kept outside the timed
    // event since it's setup, not part of the reduction itself.
    q.fill<float>(deviceSum, 0.0f, 1).wait_and_throw();

    auto event = q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::range<1>(bench.count()), [=](sycl::id<1> idx) {
        // Naive baseline: every work-item atomically adds directly into a
        // single global accumulator. Correct, but all N adds serialize on
        // one address, so this is contention-bound rather than
        // bandwidth-bound.
        sycl::atomic_ref<float, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            atomicSum(*deviceSum);
        atomicSum += deviceVector[idx];
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
