#include <sycl/sycl.hpp>

#include <benchmark.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>

int main(int argc, char **argv) {
  sycl::queue q({sycl::property::queue::enable_profiling()});
  Benchmark bench(argc, argv,
                  q.get_device().get_info<sycl::info::device::name>());
  constexpr float elementValue = 1.0f;
  const float expectedSum = elementValue * static_cast<float>(bench.count());

  constexpr uint32_t workGroupSize = 256;
  const uint32_t count = bench.count();
  const uint32_t numGroups = (count + workGroupSize - 1) / workGroupSize;
  const sycl::nd_range<1> ndRange(sycl::range<1>(numGroups * workGroupSize),
                                  sycl::range<1>(workGroupSize));

  float *deviceInputVector = sycl::malloc_device<float>(bench.count(), q);
  float *deviceOutputVector = sycl::malloc_device<float>(numGroups, q);
  float *deviceSum = sycl::malloc_device<float>(1, q);
  if (!deviceInputVector || !deviceOutputVector || !deviceSum) {
    std::cerr << "Device allocation failed\n";
    sycl::free(deviceInputVector, q);
    sycl::free(deviceOutputVector, q);
    sycl::free(deviceSum, q);
    std::exit(EXIT_FAILURE);
  }

  q.fill<float>(deviceInputVector, elementValue, bench.count())
      .wait_and_throw();
  q.fill<float>(deviceOutputVector, 0.0f, numGroups).wait_and_throw();

  auto work = [&](bool verify) -> uint64_t {
    // Reset the accumulator before each timed run
    q.fill<float>(deviceSum, 0.0f, 1).wait_and_throw();

    auto firstTreeReduction = q.submit([&](sycl::handler &h) {
      sycl::local_accessor<float, 1> sdata(sycl::range<1>(workGroupSize), h);

      h.parallel_for(ndRange, [=](sycl::nd_item<1> item) {
        const size_t globalId = item.get_global_id(0);
        const size_t localId = item.get_local_id(0);
        if (globalId < count) {
          sdata[localId] = deviceInputVector[globalId];
        } else {
          sdata[localId] = 0.0f;
        }
        sycl::group_barrier(item.get_group());

        for (uint32_t stride = workGroupSize / 2; stride > 0; stride /= 2) {
          if (localId < stride) {
            sdata[localId] += sdata[localId + stride];
          }
          sycl::group_barrier(item.get_group());
        }
        if (localId == 0) {
          deviceOutputVector[item.get_group(0)] = sdata[0];
        }
      });
    });

    auto secondTreeReduction = q.submit([&](sycl::handler &h) {
      h.depends_on(firstTreeReduction);

      auto ndRange = sycl::nd_range<1>(sycl::range<1>(workGroupSize),
                                       sycl::range<1>(workGroupSize));
      sycl::local_accessor<float, 1> sdata(sycl::range<1>(workGroupSize), h);
      h.parallel_for(ndRange, [=](sycl::nd_item<1> item) {
        const size_t localId = item.get_local_id(0);

        float partial = 0.0f;
        for (size_t i = localId; i < numGroups; i += workGroupSize) {
          partial += deviceOutputVector[i];
        }

        sdata[localId] = partial;
        sycl::group_barrier(item.get_group());

        for (uint32_t stride = workGroupSize / 2; stride > 0; stride /= 2) {
          if (localId < stride) {
            sdata[localId] += sdata[localId + stride];
          }
          sycl::group_barrier(item.get_group());
        }
        if (localId == 0) {
          *deviceSum = sdata[0];
        }
      });
    });

    secondTreeReduction.wait_and_throw();

    if (verify) {
      float hostSum = 0.0f;
      q.memcpy(&hostSum, deviceSum, sizeof(float)).wait_and_throw();
      if (hostSum != expectedSum) {
        std::cerr << "Verification failed! Expected " << expectedSum << ", got "
                  << hostSum << "\n";
        std::exit(EXIT_FAILURE);
      }
    }

    return secondTreeReduction
               .get_profiling_info<sycl::info::event_profiling::command_end>() -
           firstTreeReduction.get_profiling_info<
               sycl::info::event_profiling::command_start>();
  };

  const uint64_t bytesPerIteration =
      (static_cast<uint64_t>(bench.count()) + 1) * sizeof(float);
  bench.run(work, bytesPerIteration);

  sycl::free(deviceInputVector, q);
  sycl::free(deviceOutputVector, q);
  sycl::free(deviceSum, q);
}
