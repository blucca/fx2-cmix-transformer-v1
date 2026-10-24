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

// Fixed ablation prompted by measured SSE regressions. This defines a new
// strict integer boundary, rather than reproducing the original float API.
std::uint32_t pre_sse_frequency(const HutterTraceRecord& r) {
  if (r.flags & 2) return r.probability;
  const double limit = std::log(65535.0);
  return std::clamp(std::lround(65536 * logistic(
      std::clamp(double(r.mixer), -limit, limit))), 1L, 65535L);
}

void validate(const HutterTraceRecord& r) {
  if (!r.probability || r.flags > 3 || !std::isfinite(r.neural) ||
      !std::isfinite(r.fxcm) || !std::isfinite(r.mixer) ||
      r.neural < 0 || r.neural > 1 || r.fxcm < 0 || r.fxcm > 1)
    throw std::runtime_error("invalid trace record");
}

// src/coder/encoder.cpp's inclusive intervals, without retaining output bytes.
struct RangeCounter {
  std::uint32_t low = 0, high = 0xffffffff;
  std::uint64_t bytes = 0;
  void normalize() {
    while (((low ^ high) & 0xff000000) == 0) {
      ++bytes;
      low <<= 8;
      high = (high << 8) + 255;
    }
  }
  void encode(std::uint32_t p, int bit) {
    const std::uint32_t mid = low + ((high - low) >> 16) * p +
        (((high - low) & 0xffff) * p >> 16);
    if (bit) high = mid;
    else low = mid + 1;
    normalize();
  }
  std::uint64_t flushed_bytes() const {
    // A block report closes a copy; the live interval continues unchanged.
    auto copy = *this;
    copy.normalize();
    return copy.bytes + 1;  // Encoder::Flush writes the final high byte.
  }
};

struct Cell { double delta = 0, information; };
struct Calibrator {
  const char* name;
  int kind;
  std::vector<Cell> cells;
  double loss = 0;
  RangeCounter coder;
  Calibrator(const char* name, int kind, int count, double prior)
      : name(name), kind(kind), cells(count, Cell{0, prior}) {}

  std::uint32_t observe(const HutterTraceRecord& r, std::uint64_t i, double p, double z) {
    const int bit = r.flags & 1;
    if (r.flags & 2) {
      coder.encode(r.probability, bit);
      loss += cost(p, bit);
      return r.probability;
    }
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
    coder.encode(coded, bit);
    loss += cost(q, bit);  // Score BEFORE observing the current label.
    c.information += q * (1 - q);
    // ponytail: one bounded intercept per cell; no hidden head or optimizer.
    // A richer model is justified by stable residual gain on longer traces.
    c.delta = std::clamp(c.delta + (bit - q) / c.information, -1.0, 1.0);
    return coded;
  }
};

struct Replay {
  std::uint64_t bits = 0, overrides = 0;
  double base_loss = 0, neural_loss = 0, fxcm_loss = 0, mixer_loss = 0;
  RangeCounter base_coder;
  RangeCounter pre_sse_coder;
  float fxcm_min = 1, fxcm_max = 0;
  std::vector<Calibrator> models{
      {"global_64", 0, 1, 64}, {"bitpos_64", 1, 8, 64},
      {"confidence_64", 2, 128, 64},
      {"neural_gap_64", 3, 384, 64}, {"neural_gap_256", 3, 384, 256},
      {"pre_sse_gap_256", 4, 384, 256}};
  void observe(const HutterTraceRecord& r) {
    validate(r);
    const int bit = r.flags & 1;
    const double p = r.probability / 65536.0, z = logit(p);
    base_coder.encode(r.probability, bit);
    pre_sse_coder.encode(pre_sse_frequency(r), bit);
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
        << ",\"base_coded_bytes\":" << base_coder.flushed_bytes()
        << ",\"pre_sse_coded_bytes\":" << pre_sse_coder.flushed_bytes()
        << ",\"pre_sse_saved_bytes\":" << std::int64_t(base_coder.flushed_bytes()) -
            std::int64_t(pre_sse_coder.flushed_bytes())
        << ",\"override_bits\":" << overrides
        << ",\"neural_bits\":" << neural_loss << ",\"fxcm_bits\":" << fxcm_loss
        << ",\"pre_sse_bits\":" << mixer_loss
        << ",\"fxcm_min\":" << fxcm_min << ",\"fxcm_max\":" << fxcm_max
        << ",\"models\":[";
    for (std::size_t i = 0; i < models.size(); ++i) {
      const auto& m = models[i];
      if (i) std::cout << ',';
      std::cout << "{\"name\":\"" << m.name << "\",\"loss_bits\":" << m.loss
          << ",\"coded_bytes\":" << m.coder.flushed_bytes()
          << ",\"saved_bytes\":" << static_cast<std::int64_t>(base_coder.flushed_bytes()) -
              static_cast<std::int64_t>(m.coder.flushed_bytes())
          << ",\"saved_bits\":" << base_loss - m.loss
          << ",\"saved_fraction\":" << (base_loss - m.loss) / base_loss << '}';
    }
    std::cout << "]}\n";
  }
};

// A research-only frozen head. All history uses base predictions, so its
// features remain causal and unchanged when the final coder probability moves.
struct HistoryHead {
  static constexpr int width = 18;
  struct Ema {
    double alpha, numerator = 0, denominator = 0;
    explicit Ema(double window) : alpha(1 / window) {}
    double value() const { return denominator ? numerator / denominator : 0; }
    void update(double x) {
      numerator = (1 - alpha) * numerator + alpha * x;
      denominator = (1 - alpha) * denominator + alpha;
    }
  };
  Ema r32{32}, r512{512}, r8192{8192}, gap_neural{512}, gap_mixer{512},
      loss64{64}, loss4096{4096};
  std::uint64_t bits = 0;
  using Row = std::array<float, width>;

  Row features(const HutterTraceRecord& r) const {
    const double scale = std::log(65535.0);
    const double z = logit(r.probability / 65536.0);
    const double t = z / scale;
    const double gn = (logit(r.neural) - z) / scale;
    const double gm = (std::clamp(double(r.mixer), -scale, scale) - z) / scale;
    const double surprise = (loss64.value() - loss4096.value()) / 4;
    Row row{};
    row[bits & 7] = 1;
    row[8] = t; row[9] = gn; row[10] = gm;
    row[11] = gn * gap_neural.value() / 4;
    row[12] = gm * gap_mixer.value() / 4;
    row[13] = 4 * r32.value(); row[14] = 4 * r512.value();
    row[15] = 4 * r8192.value();
    row[16] = t * surprise; row[17] = gn * surprise;
    return row;  // One shared f32 feature boundary for fitting and scoring.
  }
  static std::uint32_t predict(const HutterTraceRecord& r, const Row& row,
                               const Row& weights) {
    if (r.flags & 2) return r.probability;
    double delta = 0;
    for (int j = 0; j < width; ++j) delta += double(row[j]) * double(weights[j]);
    return std::clamp(std::lround(65536 * logistic(
        logit(r.probability / 65536.0) + delta)), 1L, 65535L);
  }
  void update(const HutterTraceRecord& r) {
    const int bit = r.flags & 1;
    const double p = r.probability / 65536.0;
    const double scale = std::log(65535.0);
    const double loss = cost(p, bit);
    r32.update(bit - p); r512.update(bit - p); r8192.update(bit - p);
    gap_neural.update(loss - cost(logistic(logit(r.neural)), bit));
    gap_mixer.update(loss - cost(logistic(
        std::clamp(double(r.mixer), -scale, scale)), bit));
    loss64.update(loss); loss4096.update(loss);
    ++bits;  // Overrides enter history too, after their unchanged prediction.
  }
};

int history_trace(bool export_features, const char* path, const char* artifact) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open trace");
  HistoryHead state;
  HistoryHead::Row weights{};
  std::ofstream output;
  if (export_features) {
    if (std::ifstream(artifact).good()) throw std::runtime_error("features already exist");
    output.open(artifact, std::ios::binary);
    if (!output) throw std::runtime_error("cannot create features");
  } else {
    std::ifstream model(artifact, std::ios::binary);
    if (!model.read(reinterpret_cast<char*>(weights.data()), sizeof(weights)) ||
        model.peek() != std::char_traits<char>::eof())
      throw std::runtime_error("weights must be exactly 18 little-endian f32 values");
    for (float w : weights) if (!std::isfinite(w) || std::abs(w) > 2)
      throw std::runtime_error("invalid head weight");
  }
  // Held-out input-byte blocks 3 and 7 of every eight 128 KiB blocks.
  // Reset only their diagnostic range counters, never the feature history or
  // the six online controls. The whole-stream counters always keep running.
  constexpr std::uint64_t block_bits = 8 * 131072;
  RangeCounter base, head, block_base, block_head;
  auto controls = Replay{}.models;
  std::array<RangeCounter, 6> block_controls;
  std::array<std::uint64_t, 6> held_controls{};
  std::uint64_t held_bits = 0, held_base = 0, held_head = 0;
  auto close_block = [&]() {
    const auto block = (state.bits - 1) / block_bits;
    if ((block & 7) == 3 || (block & 7) == 7) {
      const auto n = (state.bits - 1) % block_bits + 1;
      held_bits += n;
      held_base += block_base.flushed_bytes();
      held_head += block_head.flushed_bytes();
      for (std::size_t j = 0; j < controls.size(); ++j)
        held_controls[j] += block_controls[j].flushed_bytes();
    }
    std::cout << "{\"stage\":\"history_block\",\"bits\":" << state.bits
        << ",\"base_coded_bytes\":" << base.flushed_bytes()
        << ",\"head_coded_bytes\":" << head.flushed_bytes() << "}\n";
    block_base = {}; block_head = {}; block_controls = {};
  };
  std::array<HutterTraceRecord, 8192> buffer;
  while (input.read(reinterpret_cast<char*>(buffer.data()), sizeof(buffer)) || input.gcount()) {
    const auto bytes = input.gcount();
    if (bytes % sizeof(HutterTraceRecord)) throw std::runtime_error("truncated trace record");
    for (std::size_t i = 0; i < bytes / sizeof(HutterTraceRecord); ++i) {
      const auto& r = buffer[i];
      validate(r);
      const auto row = state.features(r);
      if (export_features) {
        output.write(reinterpret_cast<const char*>(row.data()), sizeof(row));
        if (!output) throw std::runtime_error("feature write failed");
      } else {
        const int bit = r.flags & 1;
        const auto q = HistoryHead::predict(r, row, weights);
        base.encode(r.probability, bit); head.encode(q, bit);
        block_base.encode(r.probability, bit); block_head.encode(q, bit);
        const double p = r.probability / 65536.0, z = logit(p);
        for (std::size_t j = 0; j < controls.size(); ++j)
          block_controls[j].encode(controls[j].observe(r, state.bits, p, z), bit);
      }
      state.update(r);
      if (!export_features && state.bits % block_bits == 0) close_block();
    }
  }
  if (!input.eof()) throw std::runtime_error("trace read failed");
  if (!state.bits || state.bits % 8) throw std::runtime_error("trace must contain whole bytes");
  if (export_features) {
    output.flush();
    if (!output) throw std::runtime_error("feature flush failed");
    std::cout << "{\"stage\":\"history_features\",\"bits\":" << state.bits
        << ",\"columns\":18,\"row_bytes\":72}\n";
    return 0;
  }
  if (state.bits % block_bits) close_block();
  std::cout << "{\"stage\":\"history_complete\",\"bits\":" << state.bits
      << ",\"base_coded_bytes\":" << base.flushed_bytes()
      << ",\"head_coded_bytes\":" << head.flushed_bytes()
      << ",\"saved_bytes\":" << std::int64_t(base.flushed_bytes()) - std::int64_t(head.flushed_bytes())
      << ",\"holdout_input_bytes\":" << held_bits / 8
      << ",\"holdout_base_coded_bytes\":" << held_base
      << ",\"holdout_head_coded_bytes\":" << held_head
      << ",\"holdout_saved_bytes\":" << std::int64_t(held_base) - std::int64_t(held_head)
      << ",\"controls\":[";
  for (std::size_t j = 0; j < controls.size(); ++j) {
    if (j) std::cout << ',';
    std::cout << "{\"name\":\"" << controls[j].name << "\",\"coded_bytes\":"
        << controls[j].coder.flushed_bytes() << ",\"holdout_coded_bytes\":" << held_controls[j]
        << ",\"holdout_saved_bytes\":" << std::int64_t(held_base) - std::int64_t(held_controls[j]) << '}';
  }
  std::cout << "]}\n";
  return 0;
}

int main(int argc, char** argv) {
  try {
    if (argc == 4 && std::string(argv[1]) == "--history-features")
      return history_trace(true, argv[2], argv[3]);
    if (argc == 4 && std::string(argv[1]) == "--history-score")
      return history_trace(false, argv[2], argv[3]);
    if (argc != 2) throw std::runtime_error("usage: replay TRACE.bin | --self-test");
    Replay replay;
    if (std::string(argv[1]) == "--self-test") {
      // Zero correction preserves every legal coder frequency. Compare the
      // counter with byte-emitting Encoder arithmetic using a 64-bit product.
      RangeCounter counted;
      HistoryHead history;
      HistoryHead::Row zero{};
      std::uint32_t low = 0, high = 0xffffffff;
      std::vector<unsigned char> reference;
      for (std::uint32_t p = 1; p <= 65535; ++p) {
        assert(std::lround(logistic(logit(p / 65536.0)) * 65536) == p);
        const HutterTraceRecord r{std::uint16_t(p), 0, 0.6f, 0.5f, 0.1f};
        assert(HistoryHead::predict(r, history.features(r), zero) == p);
        const int bit = (p * 2654435761u) >> 31;
        const std::uint32_t mid = low + (std::uint64_t(high - low) * p >> 16);
        if (bit) high = mid;
        else low = mid + 1;
        while (((low ^ high) & 0xff000000) == 0) {
          reference.push_back(high >> 24);
          low <<= 8;
          high = (high << 8) + 255;
        }
        counted.encode(p, bit);
        assert(counted.low == low && counted.high == high);
        assert(counted.bytes == reference.size());
      }
      reference.push_back(high >> 24);
      assert(counted.flushed_bytes() == reference.size());
      // A known intercept error must improve; overrides must remain untouched.
      for (unsigned i = 0; i < 80000; ++i) {
        HutterTraceRecord r{26214, std::uint16_t(i % 10 < 6), 0.6f, 0.5f, 0.0f};
        replay.observe(r);
      }
      assert(replay.base_loss - replay.models[0].loss > 1000);
      assert(replay.base_coder.flushed_bytes() > replay.models[0].coder.flushed_bytes());
      Replay forced;
      for (unsigned i = 0; i < 80; ++i) {
        forced.observe(i & 1 ? HutterTraceRecord{65535, 2, 1, 0.5f, 0} :
                              HutterTraceRecord{1, 3, 0, 0.5f, 0});
      }
      assert(forced.base_coder.flushed_bytes() == 161);
      assert(forced.pre_sse_coder.flushed_bytes() == forced.base_coder.flushed_bytes());
      assert(pre_sse_frequency({1, 0, 0.5f, 0.5f, 0}) == 32768);
      assert(pre_sse_frequency({1, 0, 0.5f, 0.5f, -100}) == 1);
      assert(pre_sse_frequency({1, 0, 0.5f, 0.5f, 100}) == 65535);
      for (const auto& m : forced.models) {
        assert(m.loss == forced.base_loss);
        assert(m.coder.flushed_bytes() == forced.base_coder.flushed_bytes());
        for (const auto& c : m.cells) assert(c.delta == 0);
      }
      assert(logistic(logit(0.5)) == 0.5);
      // Current labels never enter current features; an observed label changes
      // subsequent history. Overrides retain their frequency and advance time.
      HutterTraceRecord r{32768, 0, 0.6f, 0.5f, 0.1f};
      const auto before = history.features(r);
      r.flags = 1;
      assert(history.features(r) == before);
      history.update(r);
      assert(history.bits == 1 && history.features(r)[13] == 2);
      auto nonzero = zero; nonzero.fill(2);
      r.flags = 3;
      assert(HistoryHead::predict(r, history.features(r), nonzero) == r.probability);
      history.update(r);
      assert(history.bits == 2);
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
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
