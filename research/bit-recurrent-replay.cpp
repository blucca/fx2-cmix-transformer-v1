// SPDX-License-Identifier: GPL-3.0-only
// Content/history features and seq64 LSTM residual, inspired by cmix-obias
// BITLSTM32_SPEC (dfreelan, 51488a0c). Uses this fork's integer contract.
// c++ -O3 -ffp-contract=off -std=c++17 research/bit-recurrent-replay.cpp -o /YOUR/temp/bit-recurrent
#define HUTTER_TRACE
#define HUTTER_MIXER_TRACE
#define HUTTER_MIXER_SUPPORT_LIBRARY
#include "mixer-support-replay.cpp"
#undef HUTTER_MIXER_SUPPORT_LIBRARY

namespace bit_recurrent {
constexpr int width = 78, hidden = 32, sequence = 64, weights_count = 11009;
constexpr double scale = 12.203;
constexpr int res_windows[] = {32, 256, 2048, 16384, 131072};
constexpr int sur_windows[] = {16, 64, 512, 4096, 65536, 1048576};
constexpr int experts[] = {0, 5, 11, 17, 22, 23, 24};
using Row = std::array<float, width>;
using Weights = std::array<float, weights_count>;
using mixer_support::Score;

// One shared implementation generates training rows and frozen replay rows.
// All ring entries are f32, sums are double, and the current label is observed
// only after the current row has been returned. Independent traces start empty.
struct Features {
  std::vector<float> residual = std::vector<float>(131072);
  std::vector<float> surprise = std::vector<float>(1048576);
  std::array<float, 256> distance{};
  std::array<std::array<float, 7>, 512> expert_loss{};
  std::array<double, 5> residual_sum{};
  std::array<double, 6> surprise_sum{};
  std::array<double, 7> expert_sum{};
  std::array<std::uint8_t, 8> history{};
  std::array<double, 7> pending_expert{};
  std::uint64_t index = 0, last_shock = 0;
  double distance_sum = 0, pending_p = .5, pending_distance = 0;
  unsigned shock_count = 0, partial = 0;
  float previous = .5f;
  mixer_support::Features original;

  static double sign(double x) { return (x > 0) - (x < 0); }
  double denominator(int window) const {
    return std::max<std::uint64_t>(1, std::min<std::uint64_t>(index, window));
  }
  Row before(const HutterTraceRecord& r, const HutterMixerTraceRecord& s) {
    original(r, s);  // Reuse paired-record, finite and support validation.
    const double z = original.logits[r.probability];
    std::array<double, 25> e;
    for (int j = 0; j < 23; ++j) e[j] = mixer_support::half_value(s.logits[j]);
    // Explicitly our raw-neural logit, not the champion's table-stretched input.
    e[23] = mixer_support::half_value(HutterMixerLogitHalf(float(logit(r.neural))));
    e[24] = mixer_support::half_value(HutterMixerLogitHalf(r.mixer));
    Row x{};
    for (int j = 0; j < 25; ++j) x[j] = e[j];
    x[25 + (index & 7)] = 1;
    x[33] = z / scale; x[34] = x[33] * x[33]; x[35] = sign(z);
    for (int j = 0; j < 5; ++j)
      x[36 + j] = 4 * residual_sum[j] / denominator(res_windows[j]);
    for (int j = 0; j < 6; ++j)
      x[41 + j] = surprise_sum[j] / denominator(sur_windows[j]);
    x[47] = 4 * (x[42] - x[44]); x[48] = 4 * (x[43] - x[46]);
    x[49] = std::log1p(double(index - last_shock)) / 14;
    x[50] = 8 * shock_count / denominator(256);
    x[51] = (z - e[24]) / scale;
    x[52] = distance_sum / denominator(256) / scale;
    for (int j = 0; j < 7; ++j) x[53 + j] = expert_sum[j] / denominator(512);
    double mean = 0, variance = 0;
    int agree = 0;
    for (double value : e) { mean += value; agree += sign(value) == sign(z); }
    mean /= e.size();
    for (double value : e) variance += (value - mean) * (value - mean);
    auto sorted = e; std::sort(sorted.begin(), sorted.end());
    x[60] = mean; x[61] = std::sqrt(variance / 24);
    x[62] = sorted.back() - sorted.front(); x[63] = agree / 25.0;
    for (int j = 0; j < 5; ++j) x[64 + j] = sorted[j * 6];
    for (int j = 0; j < 8; ++j) x[69 + j] = history[j] / 255.0;
    x[77] = previous;
    pending_p = r.probability / 65536.0;
    pending_distance = std::abs(z - e[24]);
    for (int j = 0; j < 7; ++j)
      pending_expert[j] = std::clamp(logistic(e[experts[j]]), 1e-6, 1 - 1e-6);
    for (float value : x)
      if (!std::isfinite(value)) throw std::runtime_error("nonfinite recurrent feature");
    return x;
  }
  void observe(int bit) {
    if (bit != 0 && bit != 1) throw std::runtime_error("invalid observed bit");
    const float r = bit - pending_p;
    const float s = cost(pending_p, bit);
    for (int j = 0; j < 5; ++j) {
      if (index >= std::uint64_t(res_windows[j]))
        residual_sum[j] -= residual[(index - res_windows[j]) % residual.size()];
      residual_sum[j] += r;
    }
    residual[index % residual.size()] = r;
    for (int j = 0; j < 6; ++j) {
      if (index >= std::uint64_t(sur_windows[j]))
        surprise_sum[j] -= surprise[(index - sur_windows[j]) % surprise.size()];
      surprise_sum[j] += s;
    }
    if (index >= 256) shock_count -= surprise[(index - 256) % surprise.size()] > 4;
    shock_count += s > 4;
    if (s > 4) last_shock = index;
    surprise[index % surprise.size()] = s;
    const float d = pending_distance;
    if (index >= 256) distance_sum -= distance[index % 256];
    distance_sum += d; distance[index % 256] = d;
    for (int j = 0; j < 7; ++j) {
      const float loss = cost(pending_expert[j], bit);
      if (index >= 512) expert_sum[j] -= expert_loss[index % 512][j];
      expert_sum[j] += loss; expert_loss[index % 512][j] = loss;
    }
    previous = bit;
    partial = (partial << 1) | bit;
    if ((index & 7) == 7) {
      for (int j = 0; j < 7; ++j) history[j] = history[j + 1];
      history[7] = partial; partial = 0;
    }
    ++index;
  }
};

// Strict double inference of the exact stored f32 weights. Float Torch training
// is an optimizer; this path defines the candidate frequencies and future coder.
struct Network {
  Weights weights{};
  std::array<double, hidden> h{}, c{};
  std::uint64_t index = 0;
  static constexpr int bin = hidden * width;
  static constexpr int wih = bin + hidden, whh = wih + 4 * hidden * hidden;
  static constexpr int bih = whh + 4 * hidden * hidden, bhh = bih + 4 * hidden;
  static constexpr int wout = bhh + 4 * hidden, bout = wout + hidden;
  static_assert(bout + 1 == weights_count);

  double next(const Row& x) {
    if (index++ % sequence == 0) { h.fill(0); c.fill(0); }
    std::array<double, hidden> projected;
    for (int j = 0; j < hidden; ++j) {
      double a = 0;
      for (int k = 0; k < width; ++k) a += double(weights[j * width + k]) * x[k];
      a += weights[bin + j]; projected[j] = a * logistic(a);
    }
    std::array<double, 4 * hidden> gates;
    for (int j = 0; j < 4 * hidden; ++j) {
      double a = 0, b = 0;
      for (int k = 0; k < hidden; ++k) {
        a += double(weights[wih + j * hidden + k]) * projected[k];
        b += double(weights[whh + j * hidden + k]) * h[k];
      }
      gates[j] = ((a + weights[bih + j]) + b) + weights[bhh + j];
    }
    double delta = 0;
    for (int j = 0; j < hidden; ++j) {
      c[j] = logistic(gates[hidden + j]) * c[j] +
          logistic(gates[j]) * std::tanh(gates[2 * hidden + j]);
      h[j] = logistic(gates[3 * hidden + j]) * std::tanh(c[j]);
      delta += double(weights[wout + j]) * h[j];
    }
    delta += weights[bout];
    if (!std::isfinite(delta)) throw std::runtime_error("nonfinite recurrent output");
    return delta;
  }
  void load(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.read(reinterpret_cast<char*>(weights.data()), sizeof(weights)) ||
        f.peek() != std::char_traits<char>::eof())
      throw std::runtime_error("recurrent weights require 11009 little-endian f32 values");
    for (float value : weights)
      if (!std::isfinite(value)) throw std::runtime_error("nonfinite recurrent weight");
  }
};

std::uint32_t frequency(const HutterTraceRecord& r, double z, double delta) {
  if (r.flags & 2) return r.probability;
  return std::clamp(std::lround(65536 * logistic(z + delta)), 1L, 65535L);
}

void report(const char* stage, std::uint64_t rows, std::uint64_t tail_rows,
            std::uint64_t overrides, const std::array<Score, 2>& scores) {
  std::cout << std::setprecision(15) << "{\"schema\":\"hutter.bit-recurrent.v1\",\"stage\":\""
      << stage << "\",\"rows\":" << rows << ",\"input_bytes\":" << rows / 8
      << ",\"override_rows\":" << overrides << ",\"tail_input_bytes\":" << tail_rows / 8
      << ",\"base_coded_bytes\":" << scores[0].all.flushed_bytes() << ",\"models\":[";
  for (int j = 0; j < 2; ++j) {
    const auto& s = scores[j];
    if (j) std::cout << ',';
    std::cout << "{\"name\":\"" << (j ? "lstm32" : "base") << "\",\"coded_bytes\":"
        << s.all.flushed_bytes() << ",\"saved_bytes\":"
        << std::int64_t(scores[0].all.flushed_bytes()) - std::int64_t(s.all.flushed_bytes())
        << ",\"loss_bits\":" << s.loss << ",\"block_saved_bytes\":"
        << std::int64_t(scores[0].block.flushed_bytes()) - std::int64_t(s.block.flushed_bytes())
        << ",\"block_saved_bits\":" << scores[0].block_loss - s.block_loss
        << ",\"tail_saved_bytes\":" << (tail_rows ?
            std::int64_t(scores[0].tail.flushed_bytes()) - std::int64_t(s.tail.flushed_bytes()) : 0)
        << ",\"tail_saved_bits\":" << scores[0].tail_loss - s.tail_loss << '}';
  }
  std::cout << "]}\n" << std::flush;
}

void process(bool exporting, const char* base_path, const char* side_path,
             const char* artifact, std::uint64_t score_start_byte = UINT64_MAX) {
  const auto base_size = std::filesystem::file_size(base_path);
  if (!base_size || base_size % (8 * sizeof(HutterTraceRecord)))
    throw std::runtime_error("base trace requires whole input bytes");
  const auto total = base_size / sizeof(HutterTraceRecord);
  if (std::filesystem::file_size(side_path) != total * sizeof(HutterMixerTraceRecord))
    throw std::runtime_error("paired trace row counts differ");
  const auto tail_start = score_start_byte == UINT64_MAX ?
      total - std::min<std::uint64_t>(total, 8 * 262144) : 8 * score_start_byte;
  if (tail_start > total) throw std::runtime_error("score start beyond trace");
  std::ifstream base(base_path, std::ios::binary), side(side_path, std::ios::binary);
  std::ofstream output;
  Network net;
  if (exporting) {
    if (std::filesystem::exists(artifact)) throw std::runtime_error("features already exist");
    output.open(artifact, std::ios::binary);
    if (!output) throw std::runtime_error("cannot create features");
  } else net.load(artifact);
  Features features;
  std::vector<HutterTraceRecord> rb(8192);
  std::vector<HutterMixerTraceRecord> sb(8192);
  std::vector<Row> xb(8192);
  std::array<Score, 2> scores;
  std::uint64_t rows = 0, overrides = 0;
  while (rows < total) {
    const auto n = std::min<std::uint64_t>(rb.size(), total - rows);
    if (!base.read(reinterpret_cast<char*>(rb.data()), n * sizeof(rb[0])) ||
        !side.read(reinterpret_cast<char*>(sb.data()), n * sizeof(sb[0])))
      throw std::runtime_error("truncated paired trace");
    for (std::uint64_t k = 0; k < n; ++k, ++rows) {
      const auto& r = rb[k];
      const auto x = features.before(r, sb[k]);
      if (exporting) xb[k] = x;
      else {
        const double delta = net.next(x);  // Overrides still advance the network.
        const double z = features.original.logits[r.probability];
        scores[0].observe(r.probability, r.flags & 1, rows >= tail_start);
        scores[1].observe(frequency(r, z, delta), r.flags & 1, rows >= tail_start);
        overrides += (r.flags >> 1) & 1;
        if ((rows + 1) % mixer_support::block_rows == 0 || rows + 1 == total) {
          report("block", rows + 1, rows + 1 > tail_start ? rows + 1 - tail_start : 0,
                 overrides, scores);
          if (rows + 1 < total) for (auto& s : scores) s.next_block();
        }
      }
      features.observe(r.flags & 1);
    }
    if (exporting && !output.write(reinterpret_cast<const char*>(xb.data()), n * sizeof(Row)))
      throw std::runtime_error("feature write failed");
  }
  if (exporting) {
    output.flush();
    if (!output) throw std::runtime_error("feature flush failed");
    std::cout << "{\"stage\":\"features-complete\",\"rows\":" << rows
        << ",\"dimensions\":78,\"feature_bytes\":" << rows * sizeof(Row) << "}\n";
  } else report("complete", rows, total - tail_start, overrides, scores);
}

void infer(const char* features_path, const char* weights_path, const char* output_path) {
  if (std::filesystem::exists(output_path)) throw std::runtime_error("infer output already exists");
  if (std::filesystem::file_size(features_path) % sizeof(Row))
    throw std::runtime_error("invalid feature size");
  std::ifstream f(features_path, std::ios::binary);
  std::ofstream out(output_path, std::ios::binary);
  Network net; net.load(weights_path); Row x;
  while (f.read(reinterpret_cast<char*>(x.data()), sizeof(x))) {
    for (float value : x)
      if (!std::isfinite(value)) throw std::runtime_error("nonfinite inference feature");
    const double delta = net.next(x);
    if (!out.write(reinterpret_cast<const char*>(&delta), sizeof(delta)))
      throw std::runtime_error("inference output write failed");
  }
  out.flush();
  if (!f.eof() || !out) throw std::runtime_error("inference I/O failed");
}

void self_test() {
  mixer_support::self_test();
  Features features;
  HutterTraceRecord r{32768, 0, .625f, .5f, .2f};
  HutterMixerTraceRecord side{}; side.flags = 1;
  for (auto& x : side.logits) x = 0x3c00;
  const auto initial = features.before(r, side);
  r.flags = 3; assert(features.before(r, side) == initial);
  assert(initial[25] == 1 && initial[77] == .5f);
  for (int j = 36; j <= 46; ++j) assert(initial[j] == 0);
  for (int j = 69; j < 77; ++j) assert(initial[j] == 0);
  for (int i = 0; i < 8; ++i) {
    auto x = features.before(r, side);
    assert(x[25 + i] == 1);
    for (int j = 69; j < 77; ++j) assert(x[j] == 0);
    features.observe((0xa5 >> (7 - i)) & 1);
  }
  const auto next = features.before(r, side);
  assert(next[76] == float(165 / 255.0) && next[77] == 1);
  assert(next[36] == 0 && next[41] == 1);
  // An exact ring reference for residual means after saturation and wrap.
  std::vector<float> values;
  for (int k = 0; k < 2000; ++k) {
    const auto x = features.before(r, side);
    if (k > 512) {
      double sum = 0;
      for (int j = k - 32; j < k; ++j) sum += values[j];
      assert(x[36] == float(4 * sum / 32));
    }
    const int bit = (k * 13 + k / 7) & 1;
    values.push_back(bit - .5f); features.observe(bit);
  }
  for (unsigned p = 1; p <= 65535; ++p) {
    r.probability = p; r.flags = p % 7 == 0 ? 3 : p & 1;
    assert(frequency(r, features.original.logits[p], 0) == p);
    if (r.flags & 2) assert(frequency(r, features.original.logits[p], 20) == p);
  }
  Network net; Row x{};
  for (int i = 0; i < 128; ++i) assert(net.next(x) == 0);
  net.weights[Network::bih + 2 * hidden] = .5f;
  net.weights[Network::wout] = 1;
  double first = 0, later = 0;
  for (int i = 0; i < 65; ++i) {
    const double d = net.next(x);
    if (i == 0) first = d;
    if (i == 1) later = d;
    if (i == 64) assert(d == first);
  }
  assert(later > first && first > 0);
  std::cout << "{\"stage\":\"self-test\",\"passed\":true,\"features\":78,\"weights\":11009,"
      "\"checks\":\"causal-label/history/ring,all-integer-frequencies,override,seq64-reset\"}\n";
}
}  // namespace bit_recurrent

int main(int argc, char** argv) {
  try {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "--self-test" && argc == 2) bit_recurrent::self_test();
    else if (mode == "--export" && argc == 5)
      bit_recurrent::process(true, argv[2], argv[3], argv[4]);
    else if (mode == "--score" && (argc == 5 || argc == 6))
      bit_recurrent::process(false, argv[2], argv[3], argv[4],
          argc == 6 ? std::stoull(argv[5]) : UINT64_MAX);
    else if (mode == "--infer" && argc == 5)
      bit_recurrent::infer(argv[2], argv[3], argv[4]);
    else throw std::runtime_error("--self-test | --export BASE SIDE FEATURES | "
        "--score BASE SIDE WEIGHTS [SCORE_START_INPUT_BYTE] | --infer FEATURES WEIGHTS DELTA_F64");
  } catch (const std::exception& e) {
    std::cerr << "bit-recurrent: " << e.what() << '\n'; return 1;
  }
}
