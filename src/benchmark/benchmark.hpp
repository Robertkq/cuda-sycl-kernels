#pragma once

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

template <typename Func>
concept Benchmarkable = requires(Func f) {
  { f(bool{}) } -> std::same_as<uint64_t>;
};

namespace CLI {
class App;
}

// Lets a kernel add its own command-line options (sizes, parameters). Each
// value is written to the bound variable when argv is parsed. CLI11 stays
// inside the benchmark library: CUDA host code is compiled by a different
// compiler, and mixing two compilers' copies of a header-only library breaks.
class BenchmarkOptions {
public:
  // integer options must be positive
  void add(const std::string &flags, uint32_t &value,
           const std::string &description);
  void add(const std::string &flags, uint64_t &value,
           const std::string &description);
  void add(const std::string &flags, float &value,
           const std::string &description);

private:
  friend class Benchmark;
  explicit BenchmarkOptions(CLI::App &app) : _app(app) {}
  CLI::App &_app;
};

enum class VerifyMode : uint8_t { None, Semi, Full };
enum class TimeUnit : uint8_t { Ns, Us, Ms };
enum class Color : uint8_t { Reset, Red, Green, Yellow };

class Benchmark {
public:
  // hardware must be non-empty; throws std::invalid_argument otherwise.
  Benchmark(int argc, char **argv, const std::string &hardware);
  // addOptions registers kernel-specific options (sizes, parameters) before
  // argv is parsed; their final values are printed and written to the JSON.
  Benchmark(int argc, char **argv, const std::string &hardware,
            const std::function<void(BenchmarkOptions &)> &addOptions);
  ~Benchmark();

  uint32_t iterations() const;
  uint32_t warmups() const;
  VerifyMode verify() const;

  template <Benchmarkable Func>
  void run(Func &&func, uint64_t bytesPerIteration,
           uint64_t flopsPerIteration) {
    setBytesPerIteration(bytesPerIteration);
    setFlopsPerIteration(flopsPerIteration);

    for (uint32_t i = 0; i < warmups(); ++i) {
      func(false);
    }

    for (uint32_t i = 0; i < iterations(); ++i) {
      addRecord(func(shouldVerify(i)));
    }
  }

  template <typename T = float, typename Gen>
  std::vector<T> generateWith(size_t count, Gen gen) {
    std::vector<T> data(count);
    std::generate(data.begin(), data.end(), gen);
    return data;
  }

  template <typename T = float>
  std::vector<T> generate(size_t count, T min = 0, T max = 1) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<T> dis(min, max);
    return generateWith<T>(count, [&] { return dis(mt); });
  }

  template <typename T = float> std::vector<T> constant(size_t count, T value) {
    return generateWith<T>(count, [value] { return value; });
  }

private:
  void printInfo() const;
  void printSummary();
  void printJson(uint64_t minTime, uint64_t maxTime, uint64_t avgTime,
                 uint64_t medianTime, double maxBandwidth, double minBandwidth,
                 double meanBandwidth, double medianBandwidth, double maxGflops,
                 double minGflops, double meanGflops,
                 double medianGflops) const;

  void addRecord(uint64_t ns);
  std::string verifyModeToString(VerifyMode mode) const;
  std::string timeUnitToString(TimeUnit unit) const;
  bool shouldVerify(uint32_t i) const;
  void setBytesPerIteration(uint64_t bytes);
  void setFlopsPerIteration(uint64_t flops);
  double bandwidthGbps(uint64_t ns) const;
  double gflops(uint64_t ns) const;
  std::string formatTime(uint64_t ns) const;
  std::string color(Color c) const;

private:
  struct Impl;
  std::unique_ptr<Impl> _impl;
};
