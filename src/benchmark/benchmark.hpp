#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

enum class VerifyMode : uint8_t { None, Semi, Full };

class Benchmark {
public:
  Benchmark(int argc, char **argv);
  ~Benchmark();

  uint32_t iterations() const;
  uint32_t warmups() const;
  uint32_t count() const;
  VerifyMode verify() const;

  template <typename Func> void run(Func &&func) {
    for (uint32_t i = 0; i < warmups(); ++i) {
      func(false);
    }

    for (uint32_t i = 0; i < iterations(); ++i) {
      addRecord(func(shouldVerify(i)));
    }
  }

  template <typename T = float, typename Gen>
  std::vector<T> generateWith(uint32_t count, Gen gen) {
    std::vector<T> data(count);
    std::generate(data.begin(), data.end(), gen);
    return data;
  }

  template <typename T = float>
  std::vector<T> generate(uint32_t count, T min = 0, T max = 1) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<T> dis(min, max);
    return generateWith<T>(count, [&] { return dis(mt); });
  }

  template <typename T = float>
  std::vector<T> constant(uint32_t count, T value) {
    return generateWith<T>(count, [value] { return value; });
  }

private:
  void printInfo() const;
  void printSummary();

  void addRecord(uint64_t ns);
  std::string verifyModeToString(VerifyMode mode) const;
  bool shouldVerify(uint32_t i) const;

private:
  struct Impl;
  std::unique_ptr<Impl> _impl;
};
