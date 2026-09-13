#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

class Benchmark {
public:
  Benchmark(int argc, char **argv);
  ~Benchmark();

  void print_info() const;
  void print_summary();

  void add_record(uint64_t ns);

  uint32_t iterations() const;
  uint32_t warmups() const;
  uint32_t count() const;
  bool verify() const;

  template <typename T = float, typename Gen>
  std::vector<T> generate_with(uint32_t count, Gen gen) {
    std::vector<T> data(count);
    std::generate(data.begin(), data.end(), gen);
    return data;
  }

  template <typename T = float>
  std::vector<T> generate(uint32_t count, T min = 0, T max = 1) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<T> dis(min, max);
    return generate_with<T>(count, [&] { return dis(mt); });
  }

  template <typename T = float>
  std::vector<T> constant(uint32_t count, T value) {
    return generate_with<T>(count, [value] { return value; });
  }

private:
  struct Impl;
  std::unique_ptr<Impl> _impl;
};
