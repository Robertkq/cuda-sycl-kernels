#include <sycl/sycl.hpp>

#include <benchmark.hpp>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

class SaxpyKernel;

int main(int argc, char **argv) {
  sycl::queue q({sycl::property::queue::enable_profiling()});
  Benchmark bench(argc, argv,
                  q.get_device().get_info<sycl::info::device::name>());

  if (bench.count() % 4 != 0) {
    std::cerr << "Count must be a multiple of 4 for the vectorized kernel\n";
    std::exit(EXIT_FAILURE);
  }

  constexpr float a = 2.0f;
  constexpr float xValue = 1.0f;
  constexpr float yValue = 2.0f;
  constexpr float expectedValue = a * xValue + yValue;
  float *x = sycl::aligned_alloc_device<float>(16, bench.count(), q);
  float *y = sycl::aligned_alloc_device<float>(16, bench.count(), q);
  float *out = sycl::aligned_alloc_device<float>(16, bench.count(), q);
  if (!x || !y || !out) {
    std::cerr << "Device allocation failed\n";
    sycl::free(x, q);
    sycl::free(y, q);
    sycl::free(out, q);
    std::exit(EXIT_FAILURE);
  }
  auto eventX = q.fill<float>(x, xValue, bench.count());
  auto eventY = q.fill<float>(y, yValue, bench.count());

  eventX.wait_and_throw();
  eventY.wait_and_throw();

  auto work = [&](bool verify) -> uint64_t {
    auto event = q.submit([&](sycl::handler &h) {
      h.parallel_for<SaxpyKernel>(
          sycl::range<1>(bench.count() / 4), [=](sycl::id<1> idx) {
            auto x4 = reinterpret_cast<sycl::vec<float, 4> *>(x);
            auto y4 = reinterpret_cast<sycl::vec<float, 4> *>(y);
            auto out4 = reinterpret_cast<sycl::vec<float, 4> *>(out);
            out4[idx] = a * x4[idx] + y4[idx];
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

  sycl::free(x, q);
  sycl::free(y, q);
  sycl::free(out, q);
}
