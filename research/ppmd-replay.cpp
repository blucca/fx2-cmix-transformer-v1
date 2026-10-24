// SPDX-License-Identifier: GPL-3.0-only
// Fixed, causal PPMd-information ablation. Build without fast-math:
// c++ -O3 -std=c++17 research/ppmd-replay.cpp -o /YOUR/temp/ppmd-replay
// Reuse the exact range counter, integer boundary, and existing control.
// ponytail: this research executable includes the existing implementation;
// extract a shared header when a selected head is integrated into the coder.
#define main hutter_original_replay_main
#include "replay.cpp"
#undef main

namespace ppmd_replay {
constexpr std::uint64_t block_rows = 8 * 131072;
constexpr int width = 9;
constexpr double prior = 64;
constexpr double weight_limit = 16;  // Allows a unit logit mixture with features scaled by T.
using Row = std::array<double, width>;
using Matrix = std::array<Row, width>;
constexpr const char* names[] = {"base", "neural_gap_64", "old_info_64",
    "ppmd_gap_64", "ppmd_quality_64", "ppmd_reliability_64"};

struct Scores {
  RangeCounter all, block;
  double loss = 0, block_loss = 0;
  void observe(std::uint32_t probability, int bit) {
    const double nll = cost(probability / 65536.0, bit);
    all.encode(probability, bit); block.encode(probability, bit);
    loss += nll; block_loss += nll;
  }
  void close_block() { block = {}; block_loss = 0; }
};

struct Reliability {
  double numerator = 0, denominator = 0;
  double value() const { return denominator ? numerator / denominator : 0; }
  void update(double base_loss_minus_ppmd_loss) {
    constexpr double alpha = 1.0 / 512;
    numerator = (1 - alpha) * numerator + alpha * base_loss_minus_ppmd_loss;
    denominator = (1 - alpha) * denominator + alpha;
  }
};

Row features(const HutterTraceRecord& r, const HutterPpmdTraceRecord& s,
             const Reliability& history) {
  if (!s.flags) return {};
  const double scale = std::log(65535.0);
  const double z = logit(r.probability / 65536.0);
  const double gap = (logit(s.ppmd) - z) / scale;
  return {1, z / scale, (logit(r.neural) - z) / scale,
      (std::clamp(double(r.mixer), -scale, scale) - z) / scale,
      gap, gap * (1 - s.entropy / 4.0), gap * (1 - 2.0 * s.escape),
      gap * (2 * s.suffix_depth / 25.0 - 1),
      gap * std::clamp(history.value() / 4, -1.0, 1.0)};
}

struct Newton {
  int dimensions;
  Row weights{};
  Matrix inverse{};
  explicit Newton(int dimensions) : dimensions(dimensions) {
    for (int j = 0; j < dimensions; ++j) inverse[j][j] = 1 / prior;
  }
  std::uint32_t predict(const HutterTraceRecord& r,
                        const HutterPpmdTraceRecord& s, const Row& row) const {
    if (!s.flags || (r.flags & 2)) return r.probability;
    double delta = 0;
    for (int j = 0; j < dimensions; ++j) delta += weights[j] * row[j];
    return std::clamp(std::lround(65536 * logistic(
        logit(r.probability / 65536.0) + std::clamp(delta, -1.0, 1.0))),
        1L, 65535L);
  }
  void update(const Row& row, std::uint32_t frequency, int bit) {
    // Accumulated full observed-Fisher matrix with a rank-one inverse update.
    // All quantities here use the already-scored, integer-boundary prediction.
    const double q = frequency / 65536.0, curvature = q * (1 - q);
    Row projected{};
    double quadratic = 0;
    for (int j = 0; j < dimensions; ++j) {
      for (int k = 0; k < dimensions; ++k)
        projected[j] += inverse[j][k] * row[k];
      quadratic += row[j] * projected[j];
    }
    const double denominator = 1 + curvature * quadratic;
    const double multiplier = curvature / denominator;
    for (int j = 0; j < dimensions; ++j) {
      weights[j] = std::clamp(weights[j] + (bit - q) * projected[j] / denominator,
          -weight_limit, weight_limit);
      for (int k = 0; k <= j; ++k) {
        inverse[j][k] -= multiplier * projected[j] * projected[k];
        inverse[k][j] = inverse[j][k];
      }
    }
  }
};

struct PpmdReplay {
  std::uint64_t rows = 0, active_rows = 0, override_rows = 0, cold_rows = 0;
  double ppmd_loss = 0;
  HutterPpmdTraceRecord byte_prior{};
  Reliability history;
  Calibrator control{"neural_gap_64", 3, 384, prior};
  std::array<Newton, 4> models{Newton{4}, Newton{5}, Newton{8}, Newton{9}};
  std::array<Scores, 6> scores;

  void validate_pair(const HutterTraceRecord& r, const HutterPpmdTraceRecord& s) {
    validate(r);
    if (!std::isfinite(s.ppmd) || s.ppmd < 0 || s.ppmd > 1 ||
        !std::isfinite(s.entropy) || s.entropy < 0 || s.entropy > 8 ||
        !std::isfinite(s.escape) || s.suffix_depth > 25 ||
        s.flags != (rows < 8 ? 0 : 1))
      throw std::runtime_error("invalid PPMd record at row " + std::to_string(rows));
    if (!s.flags) {
      if (s.ppmd != 0.5f || s.entropy != 8 || s.escape != -1 || s.suffix_depth)
        throw std::runtime_error("invalid initial PPMd byte");
    } else if (s.escape < 0 || s.escape > 1) {
      throw std::runtime_error("invalid PPMd escape probability");
    }
    if (!(rows & 7)) byte_prior = s;
    else if (s.entropy != byte_prior.entropy || s.escape != byte_prior.escape ||
        s.suffix_depth != byte_prior.suffix_depth || s.flags != byte_prior.flags)
      throw std::runtime_error("PPMd byte features changed within byte at row " +
          std::to_string(rows));
  }

  void observe(const HutterTraceRecord& r, const HutterPpmdTraceRecord& s) {
    validate_pair(r, s);
    const auto row = features(r, s, history);  // Only earlier labels enter history.
    const int bit = r.flags & 1;
    const bool learn = s.flags && !(r.flags & 2);
    const double p = r.probability / 65536.0;
    scores[0].observe(r.probability, bit);
    // Cold byte follows the same bypass as every new head; existing control
    // otherwise retains its exact cells, prior, update, and bit-position key.
    const auto control_frequency = learn ? control.observe(r, rows, p, logit(p))
                                        : r.probability;
    scores[1].observe(control_frequency, bit);
    for (std::size_t j = 0; j < models.size(); ++j) {
      const auto frequency = models[j].predict(r, s, row);
      scores[j + 2].observe(frequency, bit);  // Score before updating this head.
      if (learn) models[j].update(row, frequency, bit);
    }
    const double raw_ppmd_loss = cost(logistic(logit(s.ppmd)), bit);
    ppmd_loss += raw_ppmd_loss;
    if (learn) history.update(cost(p, bit) - raw_ppmd_loss);
    active_rows += learn; override_rows += (r.flags >> 1) & 1;
    cold_rows += !s.flags;
    ++rows;
  }

  void report(const char* stage, bool block = false) const {
    const auto base_bytes = scores[0].all.flushed_bytes();
    const auto block_base = scores[0].block.flushed_bytes();
    std::cout << std::setprecision(15)
        << "{\"schema\":\"hutter.ppmd-replay.v1\",\"stage\":\"" << stage
        << "\",\"rows\":" << rows << ",\"input_bytes\":" << rows / 8
        << ",\"active_rows\":" << active_rows
        << ",\"override_rows\":" << override_rows << ",\"cold_rows\":" << cold_rows
        << ",\"base_loss_bits\":" << scores[0].loss
        << ",\"base_coded_bytes\":" << base_bytes
        << ",\"raw_ppmd_loss_bits\":" << ppmd_loss;
    if (block) std::cout << ",\"block_index\":" << (rows - 1) / block_rows
        << ",\"block_start_input_byte\":" << ((rows - 1) / block_rows) * (block_rows / 8)
        << ",\"block_rows\":" << (rows - 1) % block_rows + 1
        << ",\"block_base_loss_bits\":" << scores[0].block_loss
        << ",\"block_base_coded_bytes\":" << block_base;
    std::cout << ",\"models\":[";
    for (std::size_t j = 1; j < scores.size(); ++j) {
      const auto& s = scores[j];
      if (j > 1) std::cout << ',';
      std::cout << "{\"name\":\"" << names[j] << "\",\"loss_bits\":" << s.loss
          << ",\"coded_bytes\":" << s.all.flushed_bytes()
          << ",\"saved_bits\":" << scores[0].loss - s.loss
          << ",\"saved_bytes\":" << std::int64_t(base_bytes) -
              std::int64_t(s.all.flushed_bytes());
      if (block) std::cout << ",\"block_loss_bits\":" << s.block_loss
          << ",\"block_coded_bytes\":" << s.block.flushed_bytes()
          << ",\"block_saved_bits\":" << scores[0].block_loss - s.block_loss
          << ",\"block_saved_bytes\":" << std::int64_t(block_base) -
              std::int64_t(s.block.flushed_bytes());
      std::cout << '}';
    }
    std::cout << "]}\n" << std::flush;
  }

  void close_block() {
    report("block", true);
    for (auto& s : scores) s.close_block();  // Heads and live intervals keep going.
  }
};

void protocol() {
  std::cout << R"JSON({"schema":"hutter.ppmd-replay.v1","stage":"protocol","base_record_bytes":16,"sidecar_record_bytes":16,"sidecar_format":"<fffHH","order":"MSB-first, one record per coded bit","cold_start":true,"block_input_bytes":131072,"prior":64,"weight_limit":16,"delta_limit":1,"ema_window_active_bits":512,"suffix_depth_scale":25,"optimizer":"full online observed-Fisher Newton, score then rank-one update, no forgetting","quantization":"clamp(lround(65536*sigmoid(logit(base/65536)+delta)),1,65535)","bypass":"cold flags=0 or neural vocabulary override: exact base, no learning or EMA update","feature_scale":"T=log(65535)","features":["1","z/T","(logit(neural)-z)/T","(clamp(mixer,-T,T)-z)/T","g=(logit(ppmd)-z)/T","g*(1-entropy/4)","g*(1-2*escape)","g*(2*suffix_depth/25-1)","g*clamp(EMA512(base_loss-ppmd_loss)/4,-1,1)"],"trajectories":[{"name":"base","dimensions":0},{"name":"neural_gap_64","dimensions":384,"source":"existing replay.cpp control, cold byte bypassed"},{"name":"old_info_64","dimensions":4},{"name":"ppmd_gap_64","dimensions":5},{"name":"ppmd_quality_64","dimensions":8},{"name":"ppmd_reliability_64","dimensions":9}]})JSON" << '\n';
}

void self_test() {
  char old_program[] = "replay", old_option[] = "--self-test";
  char* old_args[] = {old_program, old_option};
  if (hutter_original_replay_main(2, old_args))
    throw std::runtime_error("existing replay self-test failed");

  HutterTraceRecord r{32768, 0, 0.5f, 0.5f, 0};
  HutterPpmdTraceRecord s{0.8f, 2, 0.1f, 10, 1};
  Reliability history;
  Newton model{9};
  for (std::uint32_t p = 1; p <= 65535; ++p) {
    r.probability = p;
    assert(model.predict(r, s, features(r, s, history)) == p);
  }
  r.probability = 32768;
  model.weights[4] = 0.3;
  const auto row = features(r, s, history);
  const auto before = model.predict(r, s, row);
  r.flags = 1;
  assert(features(r, s, history) == row);
  assert(model.predict(r, s, row) == before);  // Current label changes no prediction.
  auto zero_label = model, one_label = model;
  zero_label.update(row, before, 0); one_label.update(row, before, 1);
  assert(zero_label.predict(r, s, row) < one_label.predict(r, s, row));
  r.flags = 3;
  assert(model.predict(r, s, row) == r.probability);
  assert(model.predict(r, {0.5f, 8, -1, 0, 0}, {}) == r.probability);

  PpmdReplay bypass;
  for (int i = 0; i < 16; ++i) {
    const HutterPpmdTraceRecord side = i < 8 ? HutterPpmdTraceRecord{0.5f, 8, -1, 0, 0} : s;
    bypass.observe({std::uint16_t(i & 1 ? 65535 : 1), std::uint16_t(i < 8 ? i & 1 : (i & 1) | 2),
        0.5f, 0.5f, 0}, side);
  }
  for (const auto& m : bypass.models) {
    for (double w : m.weights) assert(w == 0);
    for (int j = 0; j < m.dimensions; ++j) assert(m.inverse[j][j] == 1 / prior);
  }
  for (const auto& c : bypass.control.cells) assert(c.delta == 0 && c.information == prior);
  assert(bypass.history.denominator == 0 && bypass.rows == 16 && bypass.active_rows == 0);
  for (const auto& score : bypass.scores) {
    assert(score.loss == bypass.scores[0].loss);
    assert(score.all.flushed_bytes() == bypass.scores[0].all.flushed_bytes());
  }
  bool rejected = false;
  try { bypass.observe(r, {0.8f, 2, 0.1f, 10, 0}); }
  catch (const std::runtime_error&) { rejected = true; }
  assert(rejected);

  PpmdReplay synthetic;
  std::uint32_t random = 923;
  auto draw = [&]() {
    random ^= random << 13; random ^= random >> 17; random ^= random << 5;
    return random;
  };
  for (int byte = 0; byte < 32768; ++byte) {
    s = byte ? HutterPpmdTraceRecord{draw() & 1 ? 0.8f : 0.2f, 2, 0.1f, 10, 1}
             : HutterPpmdTraceRecord{0.5f, 8, -1, 0, 0};
    for (int bit = 0; bit < 8; ++bit) {
      const bool label = (draw() / 4294967296.0) < s.ppmd;
      synthetic.observe({32768, std::uint16_t(label), 0.5f, 0.5f, 0}, s);
    }
  }
  assert(synthetic.scores[2].loss - synthetic.scores[3].loss > 1000);
  assert(synthetic.scores[2].all.flushed_bytes() > synthetic.scores[3].all.flushed_bytes());
  synthetic.report("ppmd_self_test_passed");
}
}  // namespace ppmd_replay

int main(int argc, char** argv) {
  try {
    using namespace ppmd_replay;
    if (argc == 2 && std::string(argv[1]) == "--self-test") {
      self_test(); return 0;
    }
    if (argc != 3) throw std::runtime_error(
        "usage: ppmd-replay BASE_TRACE.bin PPMD_SIDECAR.bin | --self-test");
    std::ifstream base(argv[1], std::ios::binary), side(argv[2], std::ios::binary);
    if (!base || !side) throw std::runtime_error("cannot open paired traces");
    protocol();
    PpmdReplay replay;
    std::array<HutterTraceRecord, 8192> base_buffer;
    std::array<HutterPpmdTraceRecord, 8192> side_buffer;
    while (true) {
      base.read(reinterpret_cast<char*>(base_buffer.data()), sizeof(base_buffer));
      side.read(reinterpret_cast<char*>(side_buffer.data()), sizeof(side_buffer));
      const auto bytes = base.gcount();
      if (bytes != side.gcount()) throw std::runtime_error("paired trace lengths differ");
      if (base.bad() || side.bad()) throw std::runtime_error("trace read failed");
      if (!bytes) break;
      if (bytes % sizeof(HutterTraceRecord)) throw std::runtime_error("truncated trace record");
      for (std::size_t i = 0; i < bytes / sizeof(HutterTraceRecord); ++i) {
        replay.observe(base_buffer[i], side_buffer[i]);
        if (replay.rows % block_rows == 0) replay.close_block();
      }
    }
    if (!base.eof() || !side.eof()) throw std::runtime_error("trace read failed");
    if (!replay.rows || (replay.rows & 7))
      throw std::runtime_error("paired traces must contain nonempty whole bytes");
    if (replay.rows % block_rows) replay.close_block();
    replay.report("complete");
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
