#include <sycl/sycl.hpp>

#include <benchmark.hpp>

#include <algorithm>
#include <execution>
#include <format>
#include <iostream>

void work(Benchmark &bench, sycl::queue &q, bool record, bool verify) {
  constexpr float lhsValue = 1.0f;
  constexpr float rhsValue = 2.0f;
  constexpr float expectedValue = lhsValue + rhsValue;
  float *lhs = sycl::malloc_device<float>(bench.count(), q);
  float *rhs = sycl::malloc_device<float>(bench.count(), q);
  float *out = sycl::malloc_device<float>(bench.count(), q);
  auto eventLhs = q.fill<float>(lhs, 1.0f, bench.count());
  auto eventRhs = q.fill<float>(rhs, 2.0f, bench.count());
  auto eventOut = q.fill<float>(out, 0.0f, bench.count());

  auto event = q.submit([&](sycl::handler &h) {
    h.depends_on({eventLhs, eventRhs, eventOut});
    h.parallel_for(sycl::range<1>(bench.count()),
                   [=](sycl::id<1> idx) { out[idx] = lhs[idx] + rhs[idx]; });
  });

  event.wait();

  if (record) {
    uint64_t ns =
        event.get_profiling_info<sycl::info::event_profiling::command_end>() -
        event.get_profiling_info<sycl::info::event_profiling::command_start>();
    bench.add_record(ns);
  }

  if (verify) {
    std::vector<float> hostOut(bench.count());
    q.memcpy(hostOut.data(), out, bench.count() * sizeof(float)).wait();
    if (!std::all_of(hostOut.begin(), hostOut.end(),
                     [expectedValue](float v) { return v == expectedValue; })) {
      std::cerr << "Verification failed!\n";
      std::exit(EXIT_FAILURE);
    }
  }

  sycl::free(lhs, q);
  sycl::free(rhs, q);
  sycl::free(out, q);
}

int main(int argc, char **argv) {
  Benchmark bench(argc, argv);

  sycl::queue q({sycl::property::queue::enable_profiling()});

  for (uint32_t i = 0; i < bench.warmups(); ++i) {
    work(bench, q, false, bench.verify());
  }

  for (uint32_t i = 0; i < bench.iterations(); ++i) {
    work(bench, q, true, bench.verify());
  }
}
