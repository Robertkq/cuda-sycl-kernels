#include "benchmark.hpp"

#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
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
  bool json = false;
  std::string programName;
  std::string hardware;
};

Benchmark::Benchmark(int argc, char **argv, const std::string &hardware)
    : _impl(std::make_unique<Impl>()) {
  if (hardware.empty()) {
    throw std::invalid_argument("Benchmark: hardware must not be empty");
  }
  _impl->hardware = hardware;
  _impl->programName = std::filesystem::path(argv[0]).filename().string();

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
  _impl->app.add_flag("--json", _impl->json,
                       "Write a JSON file containing the results");

  try {
    _impl->app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    std::exit(_impl->app.exit(e));
  }

  if (_impl->json && _impl->timeUnit != TimeUnit::Ns) {
    std::cerr << "Warning: --time-unit is ignored with --json; JSON output "
                  "always reports times in ns\n";
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
  case Color::Yellow:
    return "\033[33m";
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
  if (_impl->json) {
    return;
  }
  std::cout << "Benchmark info:\n";
  std::cout << "  Program: " << _impl->programName << "\n";
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

  const bool hasBandwidth = _impl->bytesPerIteration > 0;
  double maxBandwidth = 0.0;
  double minBandwidth = 0.0;
  double meanBandwidth = 0.0;
  double medianBandwidth = 0.0;
  if (hasBandwidth) {
    maxBandwidth = bandwidthGbps(minTime);
    minBandwidth = bandwidthGbps(maxTime);
    medianBandwidth = bandwidthGbps(medianTime);
    for (uint64_t t : records) {
      meanBandwidth += bandwidthGbps(t);
    }
    meanBandwidth /= static_cast<double>(records.size());
  }

  if (_impl->json) {
    printJson(minTime, maxTime, avgTime, medianTime, maxBandwidth, minBandwidth,
              meanBandwidth, medianBandwidth);
    return;
  }

  std::cout << "Benchmark summary:\n";
  std::cout << "HW executed on: " << _impl->hardware << "\n";
  std::cout << color(Color::Green) << "(Best)\tMin time:\t"
            << formatTime(minTime) << color(Color::Reset) << "\n";
  std::cout << color(Color::Yellow) << "()\tMean time:\t" << formatTime(avgTime)
            << color(Color::Reset) << "\n";
  std::cout << color(Color::Yellow) << "()\tMedian time:\t"
            << formatTime(medianTime) << color(Color::Reset) << "\n";
  std::cout << color(Color::Red) << "(Worst)\tMax time:\t"
            << formatTime(maxTime) << color(Color::Reset) << "\n";

  if (hasBandwidth) {
    std::cout << "\n";
    std::cout << std::fixed << std::setprecision(1);
    std::cout << color(Color::Green) << "(Best)\tBandwidth (max):\t"
              << maxBandwidth << " GB/s" << color(Color::Reset) << "\n";
    std::cout << color(Color::Yellow) << "()\tBandwidth (mean):\t"
              << meanBandwidth << " GB/s" << color(Color::Reset) << "\n";
    std::cout << color(Color::Yellow) << "()\tBandwidth (median):\t"
              << medianBandwidth << " GB/s" << color(Color::Reset) << "\n";
    std::cout << color(Color::Red) << "(Worst)\tBandwidth (min):\t"
              << minBandwidth << " GB/s" << color(Color::Reset) << "\n";
  }
}

void Benchmark::printJson(uint64_t minTime, uint64_t maxTime, uint64_t avgTime,
                          uint64_t medianTime, double maxBandwidth,
                          double minBandwidth, double meanBandwidth,
                          double medianBandwidth) const {
  nlohmann::json j;
  j["program"] = _impl->programName;
  j["hardware"] = _impl->hardware;
  j["count"] = _impl->count;
  j["iterations"] = _impl->iterations;
  j["warmups"] = _impl->warmups;
  j["verify"] = verifyModeToString(_impl->verify);
  j["min_ns"] = minTime;
  j["max_ns"] = maxTime;
  j["avg_ns"] = avgTime;
  j["median_ns"] = medianTime;
  j["max_gbps"] = maxBandwidth;
  j["min_gbps"] = minBandwidth;
  j["mean_gbps"] = meanBandwidth;
  j["median_gbps"] = medianBandwidth;

  const std::string filename = _impl->programName + ".json";
  std::ofstream file(filename);
  if (!file) {
    std::cerr << "Failed to open " << filename << " for JSON output\n";
    return;
  }
  file << j.dump(2) << '\n';

  std::cerr << "Wrote JSON summary to " << filename << "\n";
}
