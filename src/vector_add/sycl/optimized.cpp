#include <sycl/sycl.hpp>

#include <benchmark.hpp>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

int main(int argc, char **argv) {
  Benchmark bench(argc, argv);

  if (bench.count() % 4 != 0) {
    std::cerr << "Count must be a multiple of 4 for the vectorized kernel\n";
    std::exit(EXIT_FAILURE);
  }

  sycl::queue q({sycl::property::queue::enable_profiling()});
  constexpr float lhsValue = 1.0f;
  constexpr float rhsValue = 2.0f;
  constexpr float expectedValue = lhsValue + rhsValue;
  float *lhs = sycl::malloc_device<float>(bench.count(), q);
  float *rhs = sycl::malloc_device<float>(bench.count(), q);
  float *out = sycl::malloc_device<float>(bench.count(), q);
  if (!lhs || !rhs || !out) {
    std::cerr << "Device allocation failed\n";
    sycl::free(lhs, q);
    sycl::free(rhs, q);
    sycl::free(out, q);
    std::exit(EXIT_FAILURE);
  }
  auto eventLhs = q.fill<float>(lhs, lhsValue, bench.count());
  auto eventRhs = q.fill<float>(rhs, rhsValue, bench.count());

  eventLhs.wait_and_throw();
  eventRhs.wait_and_throw();

  class VectorAddKernel;

  auto work = [&](bool verify) -> uint64_t {
    auto event = q.submit([&](sycl::handler &h) {
      h.parallel_for<VectorAddKernel>(
          sycl::range<1>(bench.count() / 4), [=](sycl::id<1> idx) {
            auto lhs4 = reinterpret_cast<sycl::vec<float, 4> *>(lhs);
            auto rhs4 = reinterpret_cast<sycl::vec<float, 4> *>(rhs);
            auto out4 = reinterpret_cast<sycl::vec<float, 4> *>(out);
            out4[idx] = lhs4[idx] + rhs4[idx];
          });
    });

    event.wait_and_throw();

    if (verify) {
      std::vector<float> hostOut(bench.count());
      q.memcpy(hostOut.data(), out, bench.count() * sizeof(float))
          .wait_and_throw();
      if (!std::all_of(
              hostOut.begin(), hostOut.end(),
              [expectedValue](float v) { return v == expectedValue; })) {
        std::cerr << "Verification failed!\n";
        std::exit(EXIT_FAILURE);
      }
    }

    return event
               .get_profiling_info<sycl::info::event_profiling::command_end>() -
           event.get_profiling_info<
               sycl::info::event_profiling::command_start>();
  };

  const uint64_t bytesPerIteration =
      3 * static_cast<uint64_t>(bench.count()) * sizeof(float);
  bench.run(work, bytesPerIteration);

  sycl::free(lhs, q);
  sycl::free(rhs, q);
  sycl::free(out, q);
}
