// SPDX-License-Identifier: GPL-3.0-only
// Frozen first-layer reliability ablation; build without fast-math:
// c++ -O3 -ffp-contract=off -std=c++17 research/mixer-support-replay.cpp -o /YOUR/temp/mixer-replay
// Reuse the original integer contract, range counter and its runnable check.
#define main hutter_original_replay_main
#include "replay.cpp"
#undef main
#include <filesystem>

namespace mixer_support {
constexpr int mixers = 23, control_width = 27, width = 73;
constexpr std::uint64_t block_rows = 8 * 131072;
using Row = std::array<float, width>;
const double scale = std::log(65535.0);

double half_value(std::uint16_t h) {
  const unsigned exponent = (h >> 10) & 31, fraction = h & 1023;
  if (exponent == 31) throw std::runtime_error("nonfinite mixer logit");
  const double value = exponent ? std::ldexp(1024 + fraction, int(exponent) - 25)
                                : std::ldexp(fraction, -24);
  return h & 0x8000 ? -value : value;
}

struct Features {
  std::array<double, 65536> logits{};
  std::array<double, 33> decay{};
  Features() {
    for (int p = 1; p <= 65535; ++p) logits[p] = logit(p / 65536.0);
    for (int k = 0; k <= 32; ++k) decay[k] = std::exp2(-0.5 * k);
  }
  Row operator()(const HutterTraceRecord& r, const HutterMixerTraceRecord& s) const {
    validate(r);
    if (s.flags != 1 || s.reserved[0] || s.reserved[1])
      throw std::runtime_error("invalid mixer flags/reserved bytes");
    const double z = logits[r.probability];
    Row row{};
    row[0] = 1;
    row[1] = z / scale;
    row[2] = (logit(r.neural) - z) / scale;
    row[3] = (std::clamp(double(r.mixer), -scale, scale) - z) / scale;
    for (int j = 0; j < mixers; ++j) {
      const auto support = s.support[j];
      if ((support & 64) || (support & 63) > 32)
        throw std::runtime_error("invalid mixer support bucket");
      const float gap = (half_value(s.logits[j]) - z) / scale;
      const bool fallback = support & 128;
      row[4 + j] = gap;
      row[27 + j] = fallback ? 0 : double(gap) * decay[support & 63];
      row[50 + j] = fallback ? gap : 0;
    }
    return row;  // Shared f32 boundary for exported training and exact replay.
  }
};

std::uint32_t frequency(const HutterTraceRecord& r, const Row& row,
                        const Row& weights, int dimensions, double z) {
  if (r.flags & 2) return r.probability;
  double delta = 0;
  for (int j = 0; j < dimensions; ++j) delta += double(row[j]) * weights[j];
  // No delta clipping: the fixed fit uses a convex offset-logistic objective.
  return std::clamp(std::lround(65536 * logistic(z + delta)), 1L, 65535L);
}

Row read_weights(const char* path, int dimensions) {
  std::ifstream f(path, std::ios::binary);
  Row weights{};
  if (!f.read(reinterpret_cast<char*>(weights.data()), dimensions * sizeof(float)) ||
      f.peek() != std::char_traits<char>::eof())
    throw std::runtime_error("wrong frozen weight size");
  for (float w : weights)
    if (!std::isfinite(w) || std::abs(w) > 16)
      throw std::runtime_error("invalid frozen coefficient");
  return weights;
}

struct Score {
  RangeCounter all, block, tail;
  double loss = 0, block_loss = 0, tail_loss = 0;
  void observe(std::uint32_t p, int bit, bool in_tail) {
    const double nll = cost(p / 65536.0, bit);
    all.encode(p, bit); block.encode(p, bit);
    loss += nll; block_loss += nll;
    if (in_tail) { tail.encode(p, bit); tail_loss += nll; }
  }
  void next_block() { block = {}; block_loss = 0; }
};

void report(const char* stage, std::uint64_t rows, std::uint64_t tail_rows,
            std::uint64_t overrides, const std::array<Score, 3>& scores) {
  const char* names[] = {"base", "logit27", "support73"};
  std::cout << std::setprecision(15)
      << "{\"schema\":\"hutter.mixer-support.v1\",\"stage\":\"" << stage
      << "\",\"rows\":" << rows << ",\"input_bytes\":" << rows / 8
      << ",\"override_rows\":" << overrides
      << ",\"tail_input_bytes\":" << tail_rows / 8
      << ",\"base_coded_bytes\":" << scores[0].all.flushed_bytes()
      << ",\"block_index\":" << (rows - 1) / block_rows
      << ",\"models\":[";
  for (int j = 0; j < 3; ++j) {
    const auto& s = scores[j];
    if (j) std::cout << ',';
    std::cout << "{\"name\":\"" << names[j] << "\",\"coded_bytes\":"
        << s.all.flushed_bytes() << ",\"loss_bits\":" << s.loss
        << ",\"saved_bytes\":" << std::int64_t(scores[0].all.flushed_bytes()) -
            std::int64_t(s.all.flushed_bytes())
        << ",\"block_coded_bytes\":" << s.block.flushed_bytes()
        << ",\"block_saved_bytes\":" << std::int64_t(scores[0].block.flushed_bytes()) -
            std::int64_t(s.block.flushed_bytes())
        << ",\"block_saved_bits\":" << scores[0].block_loss - s.block_loss
        << ",\"tail_coded_bytes\":" << (tail_rows ? s.tail.flushed_bytes() : 0)
        << ",\"tail_saved_bytes\":" << (tail_rows ?
            std::int64_t(scores[0].tail.flushed_bytes()) -
            std::int64_t(s.tail.flushed_bytes()) : 0)
        << ",\"tail_saved_bits\":" << scores[0].tail_loss - s.tail_loss << '}';
  }
  std::cout << "]}\n" << std::flush;
}

void protocol() {
  std::cout << R"JSON({"schema":"hutter.mixer-support.v1","stage":"protocol","features":73,"control_dimensions":27,"support_dimensions":73,"feature_dtype":"little-endian f32","weight_dtype":"little-endian f32","dot_product":"strict double, feature order","T":"log(65535)","old_features":["1","z/T","(logit(neural)-z)/T","(clamp(preSSE,-T,T)-z)/T"],"per_mixer_features":["g=f32((half_decode(logit)-z)/T)","f32((1-fallback)*g*2^(-bucket/2))","fallback*g"],"feature_order":"old4, gaps23, independent-support23, fallback23","warmup_rows":0,"override":"exact base; excluded from fitting; underlying model continues","quantization":"clamp(lround(65536*sigmoid(z+dot)),1,65535)","delta_clip":false,"block_input_bytes":131072,"tail":"last quarter of each slice; screening fits only first three quarters"})JSON" << '\n';
}

void process(bool exporting, const char* base_path, const char* side_path,
             const char* artifact, const char* support_weights = nullptr) {
  const auto base_size = std::filesystem::file_size(base_path);
  if (!base_size || base_size % (8 * sizeof(HutterTraceRecord)))
    throw std::runtime_error("base trace needs whole input bytes");
  const auto total = base_size / sizeof(HutterTraceRecord);
  if (std::filesystem::file_size(side_path) != total * sizeof(HutterMixerTraceRecord))
    throw std::runtime_error("paired trace row counts differ");
  std::ifstream base(base_path, std::ios::binary), side(side_path, std::ios::binary);
  if (!base || !side) throw std::runtime_error("cannot open paired traces");
  const auto tail_start = (total / 8 * 3 / 4) * 8;
  std::ofstream output;
  std::array<Row, 2> weights{};
  if (exporting) {
    if (std::filesystem::exists(artifact)) throw std::runtime_error("features already exist");
    output.open(artifact, std::ios::binary);
    if (!output) throw std::runtime_error("cannot create features");
  } else {
    weights[0] = read_weights(artifact, control_width);
    weights[1] = read_weights(support_weights, width);
  }
  Features features;
  // Keep buffers on the heap; the paired streaming working set is under 4 MiB.
  std::vector<HutterTraceRecord> rb(8192);
  std::vector<HutterMixerTraceRecord> sb(8192);
  std::vector<Row> xb(8192);
  std::array<Score, 3> scores;
  std::uint64_t rows = 0, overrides = 0;
  protocol();
  while (rows < total) {
    const auto n = std::min<std::uint64_t>(rb.size(), total - rows);
    if (!base.read(reinterpret_cast<char*>(rb.data()), n * sizeof(rb[0])) ||
        !side.read(reinterpret_cast<char*>(sb.data()), n * sizeof(sb[0])))
      throw std::runtime_error("truncated paired trace");
    for (std::uint64_t i = 0; i < n; ++i, ++rows) {
      if (!rows)
        for (auto count : sb[i].support)
          if (count) throw std::runtime_error("initial mixer support must be zero");
      const auto x = features(rb[i], sb[i]);
      if (exporting) { xb[i] = x; continue; }
      const auto& r = rb[i];
      const int bit = r.flags & 1;
      const bool in_tail = rows >= tail_start;
      scores[0].observe(r.probability, bit, in_tail);
      scores[1].observe(frequency(r, x, weights[0], control_width,
          features.logits[r.probability]), bit, in_tail);
      scores[2].observe(frequency(r, x, weights[1], width,
          features.logits[r.probability]), bit, in_tail);
      overrides += (r.flags >> 1) & 1;
      if ((rows + 1) % block_rows == 0 || rows + 1 == total) {
        report("block", rows + 1, rows + 1 > tail_start ? rows + 1 - tail_start : 0,
               overrides, scores);
        if (rows + 1 < total)
          for (auto& s : scores) s.next_block();
      }
    }
    if (exporting && !output.write(reinterpret_cast<const char*>(xb.data()), n * sizeof(Row)))
      throw std::runtime_error("feature write failed");
  }
  if (exporting) {
    output.flush();
    if (!output) throw std::runtime_error("feature flush failed");
    std::cout << "{\"stage\":\"features_complete\",\"rows\":" << rows
        << ",\"dimensions\":73,\"feature_bytes\":" << rows * sizeof(Row) << "}\n";
  } else report("complete", rows, rows - tail_start, overrides, scores);
}

void self_test() {
  char program[] = "replay", option[] = "--self-test";
  char* original[] = {program, option};
  if (hutter_original_replay_main(2, original))
    throw std::runtime_error("original integer/range check failed");
  assert(half_value(0x0001) == std::ldexp(1., -24));
  assert(half_value(0x3c00) == 1 && half_value(0xbc00) == -1);
  Features features;
  HutterTraceRecord r{32768, 0, .5f, .5f, 0};
  HutterMixerTraceRecord s{}; s.flags = 1; s.logits[0] = 0x4000;
  Row weights{};
  for (unsigned p = 1; p <= 65535; ++p) {
    r.probability = p;
    const auto x = features(r, s);
    assert(frequency(r, x, weights, width, features.logits[p]) == p);
  }
  r.probability = 32768;
  auto x = features(r, s);
  r.flags = 1; assert(features(r, s) == x);  // Current label enters no feature.
  assert(x[4] == x[27] && x[50] == 0);
  s.support[0] = 8; x = features(r, s);
  assert(x[27] == float(double(x[4]) / 16));
  s.support[0] = 128 | 8; x = features(r, s);
  assert(x[27] == 0 && x[50] == x[4]);
  weights[50] = float(scale);
  r.flags = 2; assert(frequency(r, x, weights, width, 0) == 32768);
  bool rejected = false;
  s.support[0] = 64;
  try { features(r, s); } catch (const std::runtime_error&) { rejected = true; }
  assert(rejected);

  // A support-dependent signal has identical logits in both regimes.
  // Its correct frozen interaction must beat the zero/global-logit control.
  std::uint32_t rng = 923;
  auto draw = [&]() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; };
  Row known{}; known[27] = float(scale); known[50] = -float(scale);
  Score base, candidate;
  for (int i = 0; i < 65536; ++i) {
    s.support[0] = draw() & 1 ? 0 : 128;
    s.logits[0] = draw() & 1 ? 0x4000 : 0xc000;
    r.flags = 0;
    x = features(r, s);
    const auto q = frequency(r, x, known, width, 0);
    const int bit = draw() / 4294967296.0 < q / 65536.0;
    base.observe(32768, bit, false); candidate.observe(q, bit, false);
  }
  assert(base.loss - candidate.loss > 10000);
  assert(base.all.flushed_bytes() > candidate.all.flushed_bytes());
  std::cout << "{\"stage\":\"mixer_support_self_test_passed\",\"features\":73,"
               "\"all_zero_frequencies_exact\":65535,\"synthetic_saved_bytes\":"
            << base.all.flushed_bytes() - candidate.all.flushed_bytes() << "}\n";
}
}  // namespace mixer_support

int main(int argc, char** argv) try {
  using namespace mixer_support;
  if (argc == 2 && std::string(argv[1]) == "--self-test") { self_test(); return 0; }
  if (argc == 5 && std::string(argv[1]) == "--export")
    process(true, argv[2], argv[3], argv[4]);
  else if (argc == 6 && std::string(argv[1]) == "--score")
    process(false, argv[2], argv[3], argv[4], argv[5]);
  else throw std::runtime_error(
      "usage: mixer-replay --export BASE SIDE FEATURES | --score BASE SIDE CONTROL27.f32 SUPPORT73.f32 | --self-test");
  return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
