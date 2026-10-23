// SPDX-License-Identifier: GPL-3.0-only
// Causal residual-calibration screening on observation-only coded-bit traces.
// Build: c++ -O3 -std=c++17 research/replay.cpp -o /YOUR/temp/replay
#include "trace.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

double logit(double p) {
  p = std::clamp(p, 1.0 / 65536, 65535.0 / 65536);
  return std::log(p / (1 - p));
}
double logistic(double z) { return 1 / (1 + std::exp(-z)); }
double cost(double p, int bit) { return -std::log2(bit ? p : 1 - p); }

struct Cell { double delta = 0, information; };
struct Calibrator {
  const char* name;
  int kind;
  std::vector<Cell> cells;
  double loss = 0;
  Calibrator(const char* name, int kind, int count, double prior)
      : name(name), kind(kind), cells(count, Cell{0, prior}) {}

  void observe(const HutterTraceRecord& r, std::uint64_t i, double p, double z) {
    const int bit = r.flags & 1;
    if (r.flags & 2) { loss += cost(p, bit); return; }
    const int bitpos = i & 7;
    const int bin = std::clamp(static_cast<int>((z + 10) * 0.8), 0, 15);
    int key = kind == 0 ? 0 : (kind == 1 ? bitpos : bitpos * 16 + bin);
    if (kind >= 3) {
      const double gap = (kind == 3 ? logit(r.neural) : r.mixer) - z;
      key = key * 3 + (gap < -1 ? 0 : (gap > 1 ? 2 : 1));
    }
    auto& c = cells[key];
    const double raw = logistic(z + c.delta);
    const auto coded = std::clamp(std::lround(raw * 65536), 1L, 65535L);
    const double q = coded / 65536.0;
    loss += cost(q, bit);  // Score BEFORE observing the current label.
    c.information += q * (1 - q);
    // ponytail: one bounded intercept per cell; no hidden head or optimizer.
    // A richer model is justified by stable residual gain on longer traces.
    c.delta = std::clamp(c.delta + (bit - q) / c.information, -1.0, 1.0);
  }
};

struct Replay {
  std::uint64_t bits = 0, overrides = 0;
  double base_loss = 0, neural_loss = 0, fxcm_loss = 0, mixer_loss = 0;
  float fxcm_min = 1, fxcm_max = 0;
  std::vector<Calibrator> models{
      {"global_64", 0, 1, 64}, {"bitpos_64", 1, 8, 64},
      {"confidence_64", 2, 128, 64},
      {"neural_gap_64", 3, 384, 64}, {"neural_gap_256", 3, 384, 256},
      {"pre_sse_gap_256", 4, 384, 256}};
  void observe(const HutterTraceRecord& r) {
    if (!r.probability || r.flags > 3 || !std::isfinite(r.neural) ||
        !std::isfinite(r.fxcm) || !std::isfinite(r.mixer) ||
        r.neural < 0 || r.neural > 1 || r.fxcm < 0 || r.fxcm > 1)
      throw std::runtime_error("invalid trace record");
    const int bit = r.flags & 1;
    const double p = r.probability / 65536.0, z = logit(p);
    base_loss += cost(p, bit);
    neural_loss += cost(logistic(logit(r.neural)), bit);
    fxcm_loss += cost(logistic(logit(r.fxcm)), bit);
    mixer_loss += cost(logistic(std::clamp(double(r.mixer), -11.09, 11.09)), bit);
    fxcm_min = std::min(fxcm_min, r.fxcm);
    fxcm_max = std::max(fxcm_max, r.fxcm);
    for (auto& model : models) model.observe(r, bits, p, z);
    overrides += (r.flags >> 1) & 1;
    ++bits;
  }
  void report(const char* stage) const {
    std::cout << std::setprecision(12) << "{\"stage\":\"" << stage
        << "\",\"bits\":" << bits << ",\"base_bits\":" << base_loss
        << ",\"override_bits\":" << overrides
        << ",\"neural_bits\":" << neural_loss << ",\"fxcm_bits\":" << fxcm_loss
        << ",\"pre_sse_bits\":" << mixer_loss
        << ",\"fxcm_min\":" << fxcm_min << ",\"fxcm_max\":" << fxcm_max
        << ",\"models\":[";
    for (std::size_t i = 0; i < models.size(); ++i) {
      const auto& m = models[i];
      if (i) std::cout << ',';
      std::cout << "{\"name\":\"" << m.name << "\",\"loss_bits\":" << m.loss
          << ",\"saved_bits\":" << base_loss - m.loss
          << ",\"saved_fraction\":" << (base_loss - m.loss) / base_loss << '}';
    }
    std::cout << "]}\n";
  }
};

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::runtime_error("usage: replay TRACE.bin | --self-test");
    Replay replay;
    if (std::string(argv[1]) == "--self-test") {
      // A known intercept error must improve; overrides must remain untouched.
      for (unsigned i = 0; i < 80000; ++i) {
        HutterTraceRecord r{26214, std::uint16_t(i % 10 < 6), 0.6f, 0.5f, 0.0f};
        replay.observe(r);
      }
      assert(replay.base_loss - replay.models[0].loss > 1000);
      Replay forced;
      for (unsigned i = 0; i < 80; ++i) forced.observe({65535, 3, 1, 0.5f, 0});
      for (const auto& m : forced.models) assert(m.loss == forced.base_loss);
      assert(logistic(logit(0.5)) == 0.5);
      replay.report("self_test_passed");
      return 0;
    }
    std::ifstream input(argv[1], std::ios::binary);
    if (!input) throw std::runtime_error("cannot open trace");
    std::array<HutterTraceRecord, 8192> block;
    while (input.read(reinterpret_cast<char*>(block.data()), sizeof(block)) || input.gcount()) {
      const auto bytes = input.gcount();
      if (bytes % sizeof(HutterTraceRecord)) throw std::runtime_error("truncated trace record");
      for (std::size_t i = 0; i < bytes / sizeof(HutterTraceRecord); ++i) {
        replay.observe(block[i]);
        if (replay.bits % (8 * 262144) == 0) replay.report("block");
      }
    }
    if (!input.eof()) throw std::runtime_error("trace read failed");
    if (!replay.bits || replay.bits % 8) throw std::runtime_error("trace must contain whole bytes");
    replay.report("complete");
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
