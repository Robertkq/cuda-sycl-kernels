#include <sycl/sycl.hpp>

#include <iostream>

// Playground only -- not part of the transpose kernel. Just here to make
// work-item / work-group ids concrete by printing them.
int main() {
  sycl::queue q;

  constexpr size_t totalItems = 16; // total threads, CUDA calls this the grid
  constexpr size_t groupSize = 4;   // threads per group, CUDA calls this the block

  // malloc_shared: memory both host and device can touch directly, no
  // explicit copy needed. Slower than malloc_device, fine for a playground.
  int *globalIds = sycl::malloc_shared<int>(totalItems, q);
  int *localIds = sycl::malloc_shared<int>(totalItems, q);
  int *groupIds = sycl::malloc_shared<int>(totalItems, q);

  q.submit([&](sycl::handler &h) {
     h.parallel_for(sycl::nd_range<1>(totalItems, groupSize),
                     [=](sycl::nd_item<1> item) {
                       size_t g = item.get_global_id(0);
                       globalIds[g] = static_cast<int>(g);
                       localIds[g] = static_cast<int>(item.get_local_id(0));
                       groupIds[g] = static_cast<int>(item.get_group(0));
                     });
   }).wait();

  std::cout << "global | local | group   (CUDA: global==idx, local==threadIdx.x, group==blockIdx.x)\n";
  for (size_t i = 0; i < totalItems; ++i) {
    std::cout << "  " << globalIds[i] << "    |   " << localIds[i] << "   |   "
              << groupIds[i] << "\n";
  }

  sycl::free(globalIds, q);
  sycl::free(localIds, q);
  sycl::free(groupIds, q);
}
