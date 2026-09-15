#pragma once

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

template <typename Func>
concept Benchmarkable = requires(Func f) {
  { f(bool{}) } -> std::same_as<uint64_t>;
};

enum class VerifyMode : uint8_t { None, Semi, Full };
enum class TimeUnit : uint8_t { Ns, Us, Ms };
enum class Color : uint8_t { Reset, Red, Green };

class Benchmark {
public:
  Benchmark(int argc, char **argv);
  ~Benchmark();

  uint32_t iterations() const;
  uint32_t warmups() const;
  uint32_t count() const;
  VerifyMode verify() const;

  template <Benchmarkable Func>
  void run(Func &&func, uint64_t bytesPerIteration) {
    setBytesPerIteration(bytesPerIteration);

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
  std::string timeUnitToString(TimeUnit unit) const;
  bool shouldVerify(uint32_t i) const;
  void setBytesPerIteration(uint64_t bytes);
  double bandwidthGbps(uint64_t ns) const;
  std::string formatTime(uint64_t ns) const;
  std::string color(Color c) const;

private:
  struct Impl;
  std::unique_ptr<Impl> _impl;
};
