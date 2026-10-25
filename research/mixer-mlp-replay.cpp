// SPDX-License-Identifier: GPL-3.0-only
// Frozen 16-unit nonlinear residual above the unchanged 27-feature control.
// c++ -O3 -ffp-contract=off -std=c++17 research/mixer-mlp-replay.cpp -o /YOUR/temp/mixer-mlp-replay
#define HUTTER_MIXER_SUPPORT_LIBRARY
#include "mixer-support-replay.cpp"
#undef HUTTER_MIXER_SUPPORT_LIBRARY

namespace mixer_mlp {
constexpr int width = 88, hidden = 16;
constexpr int bias_offset = hidden * width, output_offset = bias_offset + hidden;
constexpr int weight_count = output_offset + hidden + 1;
constexpr std::uint64_t tail_limit = 8 * 262144;
using Row = std::array<float, width>;
using Weights = std::array<float, weight_count>;
using mixer_support::Score;

struct Inputs {
  std::array<std::uint8_t, 8> history{};
  unsigned partial = 0, bitpos = 0;

  Row before(const mixer_support::Row& phi, const HutterMixerTraceRecord& s) const {
    Row x{};
    for (int j = 0; j < 26; ++j) x[j] = phi[j + 1];
    for (int j = 0; j < mixer_support::mixers; ++j) {
      x[26 + j] = (s.support[j] & 63) / 32.0f;
      x[49 + j] = (s.support[j] & 128) ? 1.0f : 0.0f;
    }
    x[72 + bitpos] = 1;
    for (int j = 0; j < 8; ++j) x[80 + j] = float(history[j] / 255.0);
    return x;  // Current label and incomplete byte never enter these f32 inputs.
  }

  void observe(const HutterTraceRecord& r) {
    partial = (partial << 1) | (r.flags & 1);
    if (++bitpos == 8) {
      for (int j = 0; j < 7; ++j) history[j] = history[j + 1];
      history[7] = partial;
      partial = bitpos = 0;
    }
  }
};

Weights read_head(const char* path) {
  std::ifstream input(path, std::ios::binary);
  Weights weights{};
  if (!input.read(reinterpret_cast<char*>(weights.data()), sizeof(weights)) ||
      input.peek() != std::char_traits<char>::eof())
    throw std::runtime_error("MLP weights must be exactly 1441 little-endian f32 values");
  for (float w : weights)
    if (!std::isfinite(w) || std::abs(w) > 16)
      throw std::runtime_error("invalid frozen MLP coefficient");
  return weights;
}

std::uint32_t frequency(const HutterTraceRecord& r, const mixer_support::Row& phi,
                        const Row& x, const mixer_support::Row& theta,
                        const Weights& weights, double z) {
  if (r.flags & 2) return r.probability;
  double linear = 0;
  for (int j = 0; j < mixer_support::control_width; ++j)
    linear += double(phi[j]) * theta[j];
  double residual = 0;
  for (int h = 0; h < hidden; ++h) {
    double activation = 0;
    for (int j = 0; j < width; ++j)
      activation += double(weights[h * width + j]) * x[j];
    activation += weights[bias_offset + h];
    residual += double(weights[output_offset + h]) * std::tanh(activation);
  }
  return std::clamp(std::lround(65536 * logistic(
      ((z + linear) + residual) + weights.back())), 1L, 65535L);
}

void report(const char* stage, std::uint64_t rows, std::uint64_t tail_rows,
            std::uint64_t overrides, const std::array<Score, 3>& scores) {
  const char* names[] = {"base", "linear27", "mlp16"};
  std::cout << std::setprecision(15)
      << "{\"schema\":\"hutter.mixer-mlp.v1\",\"stage\":\"" << stage
      << "\",\"rows\":" << rows << ",\"input_bytes\":" << rows / 8
      << ",\"override_rows\":" << overrides
      << ",\"tail_input_bytes\":" << tail_rows / 8
      << ",\"base_coded_bytes\":" << scores[0].all.flushed_bytes()
      << ",\"block_index\":" << (rows - 1) / mixer_support::block_rows
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
  std::cout << R"JSON({"schema":"hutter.mixer-mlp.v1","stage":"protocol","features":88,"control_dimensions":27,"hidden_units":16,"new_weight_count":1441,"feature_dtype":"little-endian f32","weight_dtype":"little-endian f32","weight_order":"W[16,88] row-major, b[16], v[16], c","feature_order":"phi27[1:27], bucket23/32, fallback23, bitpos8 one-hot, completed-byte-history8/255 oldest-to-newest","initial_history":"eight zero bytes","byte_order":"MSB-first; history advances only after the eighth label","arithmetic":"strict double: theta dot in index order; each W dot in index order then b; hidden sum in index order; ((z+linear)+hidden_sum)+c; tanh double","override":"exact base; excluded from fitting; bit position and byte history still advance","feature_feedback":"original base only; frozen head never changes inputs","quantization":"clamp(lround(65536*sigmoid(logit)),1,65535)","delta_clip":false,"coefficient_bound":16,"warmup_rows":0,"block_input_bytes":131072,"tail":"continuous last min(262144,input_bytes); counter and feature history continue across diagnostic blocks"})JSON" << '\n';
}

void process(bool exporting, const char* base_path, const char* side_path,
             const char* artifact, const char* head_path = nullptr) {
  const auto base_size = std::filesystem::file_size(base_path);
  if (!base_size || base_size % (8 * sizeof(HutterTraceRecord)))
    throw std::runtime_error("base trace needs whole input bytes");
  const auto total = base_size / sizeof(HutterTraceRecord);
  if (std::filesystem::file_size(side_path) != total * sizeof(HutterMixerTraceRecord))
    throw std::runtime_error("paired trace row counts differ");
  std::ifstream base(base_path, std::ios::binary), side(side_path, std::ios::binary);
  if (!base || !side) throw std::runtime_error("cannot open paired traces");
  const auto tail_start = total - std::min<std::uint64_t>(total, tail_limit);
  std::ofstream output;
  mixer_support::Row theta{};
  Weights weights{};
  if (exporting) {
    if (std::filesystem::exists(artifact)) throw std::runtime_error("features already exist");
    output.open(artifact, std::ios::binary);
    if (!output) throw std::runtime_error("cannot create features");
  } else {
    theta = mixer_support::read_weights(artifact, mixer_support::control_width);
    weights = read_head(head_path);
  }
  mixer_support::Features features;
  Inputs inputs;
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
      const auto& r = rb[i];
      const auto phi = features(r, sb[i]);
      const auto x = inputs.before(phi, sb[i]);
      if (exporting) xb[i] = x;
      else {
        const int bit = r.flags & 1;
        const bool in_tail = rows >= tail_start;
        const double z = features.logits[r.probability];
        scores[0].observe(r.probability, bit, in_tail);
        scores[1].observe(mixer_support::frequency(r, phi, theta,
            mixer_support::control_width, z), bit, in_tail);
        scores[2].observe(frequency(r, phi, x, theta, weights, z), bit, in_tail);
        overrides += (r.flags >> 1) & 1;
        if ((rows + 1) % mixer_support::block_rows == 0 || rows + 1 == total) {
          mixer_mlp::report("block", rows + 1,
              rows + 1 > tail_start ? rows + 1 - tail_start : 0, overrides, scores);
          if (rows + 1 < total)
            for (auto& s : scores) s.next_block();
        }
      }
      inputs.observe(r);  // Overrides also advance the completed-byte history.
    }
    if (exporting && !output.write(reinterpret_cast<const char*>(xb.data()), n * sizeof(Row)))
      throw std::runtime_error("feature write failed");
  }
  if (exporting) {
    output.flush();
    if (!output) throw std::runtime_error("feature flush failed");
    std::cout << "{\"stage\":\"features_complete\",\"rows\":" << rows
        << ",\"dimensions\":88,\"feature_bytes\":" << rows * sizeof(Row) << "}\n";
  } else mixer_mlp::report("complete", rows, rows - tail_start, overrides, scores);
}

void self_test() {
  mixer_support::self_test();  // Existing half, phi27, override and exact range checks.
  mixer_support::Features features;
  mixer_support::Row theta{};
  theta[0] = .125f; theta[2] = -.25f; theta[7] = .0625f;
  HutterTraceRecord r{32768, 0, .625f, .5f, .2f};
  HutterMixerTraceRecord s{}; s.flags = 1;
  for (int j = 0; j < mixer_support::mixers; ++j) {
    s.logits[j] = j & 1 ? 0x3c00 : 0xbc00;
    s.support[j] = (j % 2 ? 128 : 0) | j;
  }
  Inputs inputs;
  auto phi = features(r, s);
  auto x = inputs.before(phi, s);
  for (int j = 0; j < 26; ++j) assert(x[j] == phi[j + 1]);
  for (int j = 0; j < mixer_support::mixers; ++j) {
    assert(x[26 + j] == j / 32.0f);
    assert(x[49 + j] == (j % 2 ? 1 : 0));
  }
  r.flags = 1; assert(inputs.before(features(r, s), s) == x);
  r.flags = 3; assert(inputs.before(features(r, s), s) == x);
  for (int i = 0; i < 8; ++i) {
    x = inputs.before(phi, s);
    for (int j = 0; j < 8; ++j) {
      assert(x[72 + j] == (i == j ? 1 : 0));
      assert(x[80 + j] == 0);  // Even observed partial bits stay out of history.
    }
    r.flags = 2 | ((0xa5 >> (7 - i)) & 1);
    inputs.observe(r);
  }
  x = inputs.before(phi, s);
  assert(inputs.bitpos == 0 && x[72] == 1 && x[87] == float(165 / 255.0));
  for (int j = 0; j < 7; ++j) assert(x[80 + j] == 0);
  for (int byte = 1; byte <= 8; ++byte)
    for (int i = 0; i < 8; ++i) {
      r.flags = (byte >> (7 - i)) & 1;
      inputs.observe(r);
    }
  x = inputs.before(phi, s);
  for (int j = 0; j < 8; ++j) assert(x[80 + j] == float((j + 1) / 255.0));

  Weights zero{};
  RangeCounter linear, nonlinear;
  for (unsigned p = 1; p <= 65535; ++p) {
    r.probability = p; r.flags = p % 7 == 0 ? 3 : p & 1;
    phi = features(r, s); x = inputs.before(phi, s);
    const auto expected = mixer_support::frequency(r, phi, theta,
        mixer_support::control_width, features.logits[p]);
    const auto actual = frequency(r, phi, x, theta, zero, features.logits[p]);
    assert(actual == expected);
    linear.encode(expected, r.flags & 1); nonlinear.encode(actual, r.flags & 1);
    assert(linear.low == nonlinear.low && linear.high == nonlinear.high &&
           linear.bytes == nonlinear.bytes);
    inputs.observe(r);
  }
  assert(linear.flushed_bytes() == nonlinear.flushed_bytes());

  Weights one{};
  one[87] = .5f; one[bias_offset] = .25f;
  one[output_offset] = .125f; one.back() = -.0625f;
  one[15 * width + 79] = -.5f; one[bias_offset + 15] = -.125f;
  one[output_offset + 15] = .25f;
  r.flags = 0; r.probability = 32768;
  phi = features(r, s); x = inputs.before(phi, s);
  double offset = 0;
  for (int j = 0; j < mixer_support::control_width; ++j)
    offset += double(phi[j]) * theta[j];
  const double expected_logit = (offset + (.125 * std::tanh(.5 * double(x[87]) + .25) +
      .25 * std::tanh(-.5 * double(x[79]) - .125))) - .0625;
  assert(frequency(r, phi, x, theta, one, 0) ==
      std::clamp(std::lround(65536 * logistic(expected_logit)), 1L, 65535L));
  r.flags = 3;
  assert(frequency(r, phi, x, theta, one, 0) == r.probability);
  std::cout << "{\"stage\":\"mixer_mlp_self_test_passed\",\"features\":88,"
               "\"new_weights\":1441,\"zero_head_matches_linear_frequencies\":65535,"
               "\"history\":\"causal, MSB-first, includes overrides\"}\n";
}
}  // namespace mixer_mlp

int main(int argc, char** argv) try {
  using namespace mixer_mlp;
  if (argc == 2 && std::string(argv[1]) == "--self-test") { self_test(); return 0; }
  if (argc == 5 && std::string(argv[1]) == "--features")
    process(true, argv[2], argv[3], argv[4]);
  else if (argc == 6 && std::string(argv[1]) == "--score")
    process(false, argv[2], argv[3], argv[4], argv[5]);
  else throw std::runtime_error(
      "usage: mixer-mlp-replay --features BASE SIDE FEATURES | --score BASE SIDE THETA27.f32 MLP16.f32 | --self-test");
  return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
