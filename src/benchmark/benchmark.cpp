#include "benchmark.hpp"

#include <CLI/CLI.hpp>
#include <cstdlib>
#include <execution>
#include <iostream>
#include <map>
#include <numeric>

struct Benchmark::Impl {
  CLI::App app{"Benchmark"};
  std::vector<uint64_t> records;
  uint32_t iterations = 100;
  uint32_t warmups = 5;
  uint32_t count = 1 << 27;
  VerifyMode verify = VerifyMode::None;
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
      ->default_val(1 << 27);
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

  try {
    _impl->app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    std::exit(_impl->app.exit(e));
  }

  printInfo();
}

Benchmark::~Benchmark() { printSummary(); }

void Benchmark::addRecord(uint64_t ns) { _impl->records.push_back(ns); }

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
}

void Benchmark::printSummary() {
  auto &records = _impl->records;
  if (records.empty()) {
    std::cout << "No records to summarize.\n";
    return;
  }

  std::sort(std::execution::par, records.begin(), records.end());
  std::cout << "Benchmark summary:\n";
  std::cout << "  Min time: " << records.front() << " ns\n";
  std::cout << "  Max time: " << records.back() << " ns\n";
  uint64_t sum = std::reduce(std::execution::par, records.begin(),
                             records.end(), uint64_t(0));
  std::cout << "  Avg time: " << sum / records.size() << " ns\n";
  std::cout << " Median time: " << records[records.size() / 2] << " ns\n";
}
