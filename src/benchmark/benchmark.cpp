#include "benchmark.hpp"

#include <CLI/CLI.hpp>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <unistd.h>

struct Benchmark::Impl {
  CLI::App app{"Benchmark"};
  std::vector<uint64_t> records;
  uint32_t iterations = 100;
  uint32_t warmups = 5;
  uint32_t count = 1 << 27;
  VerifyMode verify = VerifyMode::None;
  TimeUnit timeUnit = TimeUnit::Ns;
  uint64_t bytesPerIteration = 0;
  bool noColor = false;
};

Benchmark::Benchmark(int argc, char **argv) : _impl(std::make_unique<Impl>()) {
  _impl->app
      .add_option("-i,--iterations", _impl->iterations, "Number of iterations")
      ->default_val(10);
  _impl->app
      .add_option("-w,--warmups", _impl->warmups, "Number of warmup iterations")
      ->default_val(2);
  _impl->app
      .add_option("-c,--count", _impl->count, "Number of elements to process")
      ->default_val(1 << 27)
      ->check(CLI::PositiveNumber);
  const std::map<std::string, VerifyMode> verifyModeChoices{
      {"None", VerifyMode::None},
      {"Semi", VerifyMode::Semi},
      {"Full", VerifyMode::Full},
  };
  _impl->app
      .add_option("-v,--verify", _impl->verify,
                  "Verify results: None, Semi (every 10th iteration), Full "
                  "(every iteration)")
      ->transform(CLI::CheckedTransformer(verifyModeChoices, CLI::ignore_case));
  const std::map<std::string, TimeUnit> timeUnitChoices{
      {"ns", TimeUnit::Ns},
      {"us", TimeUnit::Us},
      {"ms", TimeUnit::Ms},
  };
  _impl->app
      .add_option("-t,--time-unit", _impl->timeUnit,
                  "Unit for reported times: ns, us, ms")
      ->transform(CLI::CheckedTransformer(timeUnitChoices, CLI::ignore_case));
  _impl->app.add_flag("--no-color", _impl->noColor, "Disable colored output");

  try {
    _impl->app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    std::exit(_impl->app.exit(e));
  }

  printInfo();
}

Benchmark::~Benchmark() { printSummary(); }

void Benchmark::addRecord(uint64_t ns) { _impl->records.push_back(ns); }

void Benchmark::setBytesPerIteration(uint64_t bytes) {
  _impl->bytesPerIteration = bytes;
}

uint32_t Benchmark::iterations() const { return _impl->iterations; }
uint32_t Benchmark::warmups() const { return _impl->warmups; }
uint32_t Benchmark::count() const { return _impl->count; }
VerifyMode Benchmark::verify() const { return _impl->verify; }

std::string Benchmark::verifyModeToString(VerifyMode mode) const {
  switch (mode) {
  case VerifyMode::None:
    return "None";
  case VerifyMode::Semi:
    return "Semi";
  case VerifyMode::Full:
    return "Full";
  }
  return "Unknown";
}

std::string Benchmark::timeUnitToString(TimeUnit unit) const {
  switch (unit) {
  case TimeUnit::Ns:
    return "ns";
  case TimeUnit::Us:
    return "us";
  case TimeUnit::Ms:
    return "ms";
  }
  return "unknown";
}

double Benchmark::bandwidthGbps(uint64_t ns) const {
  return static_cast<double>(_impl->bytesPerIteration) /
         static_cast<double>(ns);
}

std::string Benchmark::formatTime(uint64_t ns) const {
  std::ostringstream oss;
  if (_impl->timeUnit == TimeUnit::Ns) {
    oss << ns << " ns";
  } else {
    double divisor = _impl->timeUnit == TimeUnit::Us ? 1e3 : 1e6;
    oss << std::fixed << std::setprecision(3)
        << static_cast<double>(ns) / divisor << " "
        << timeUnitToString(_impl->timeUnit);
  }
  return oss.str();
}

std::string Benchmark::color(Color c) const {
  if (_impl->noColor || !isatty(STDOUT_FILENO)) {
    return "";
  }
  switch (c) {
  case Color::Reset:
    return "\033[0m";
  case Color::Red:
    return "\033[31m";
  case Color::Green:
    return "\033[32m";
  }
  return "";
}

bool Benchmark::shouldVerify(uint32_t i) const {
  switch (_impl->verify) {
  case VerifyMode::None:
    return false;
  case VerifyMode::Semi:
    return i % 10 == 0;
  case VerifyMode::Full:
    return true;
  }
  return false;
}

void Benchmark::printInfo() const {
  std::cout << "Benchmark info:\n";
  std::cout << "  Iterations: " << _impl->iterations << "\n";
  std::cout << "  Warmups: " << _impl->warmups << "\n";
  std::cout << "  Count: " << _impl->count << "\n";
  std::cout << "  Verify: " << verifyModeToString(_impl->verify) << "\n";
  std::cout << "  Time unit: " << timeUnitToString(_impl->timeUnit) << "\n";
}

void Benchmark::printSummary() {
  auto &records = _impl->records;
  if (records.empty()) {
    std::cout << "No records to summarize.\n";
    return;
  }

  std::sort(records.begin(), records.end());

  uint64_t minTime = records.front();
  uint64_t maxTime = records.back();
  uint64_t sum = std::reduce(records.begin(), records.end(), uint64_t(0));
  uint64_t avgTime = sum / records.size();

  size_t mid = records.size() / 2;
  uint64_t medianTime = records.size() % 2 == 0
                            ? std::midpoint(records[mid - 1], records[mid])
                            : records[mid];

  std::cout << "Benchmark summary:\n";
  std::cout << color(Color::Green) << "(Best)\tMin time:\t"
            << formatTime(minTime) << color(Color::Reset) << "\n";
  std::cout << color(Color::Red) << "(Worst)\tMax time:\t"
            << formatTime(maxTime) << color(Color::Reset) << "\n";
  std::cout << "()\tAvg time:\t" << formatTime(avgTime) << "\n";
  std::cout << "()\tMedian time:\t" << formatTime(medianTime) << "\n";

  if (_impl->bytesPerIteration > 0) {
    double maxBandwidth = bandwidthGbps(minTime);
    double minBandwidth = bandwidthGbps(maxTime);
    double medianBandwidth = bandwidthGbps(medianTime);

    double meanBandwidth = 0.0;
    for (uint64_t t : records) {
      meanBandwidth += bandwidthGbps(t);
    }
    meanBandwidth /= static_cast<double>(records.size());

    std::cout << "\n";
    std::cout << std::fixed << std::setprecision(1);
    std::cout << color(Color::Green) << "(Best)\tBandwidth (max):\t"
              << maxBandwidth << " GB/s" << color(Color::Reset) << "\n";
    std::cout << color(Color::Red) << "(Worst)\tBandwidth (min):\t"
              << minBandwidth << " GB/s" << color(Color::Reset) << "\n";
    std::cout << "()\tBandwidth (mean):\t" << meanBandwidth << " GB/s\n";
    std::cout << "()\tBandwidth (median):\t" << medianBandwidth << " GB/s\n";
  }
}
