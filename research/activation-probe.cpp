// Fixed-weight Transformer probe. Capture uses the coder's actual PPMd path;
// eval scores raw fp32 outputs, separately from the coder's f16/guarded output.
// Compile this main without fast-math; reuse the production model objects.
#include "predictor.h"
#ifdef FX2_WINDOW_PROBE
#include "../cpp_infer/src/opt/attn.h"
#endif
#ifdef FX2_MLP_UNIT_STATS
#include "../cpp_infer/src/weights_io.h"
#endif

#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <sys/resource.h>
#include <sys/stat.h>
#include <vector>

#ifdef __FAST_MATH__
#error "activation-probe requires strict floating point (no fast-math)"
#endif
#ifndef FX2_POSITIVE_U8
#define FX2_POSITIVE_U8 0
#endif
#ifndef FX2_DYNAMIC_HEAD
#define FX2_DYNAMIC_HEAD 0
#endif
#if FX2_DYNAMIC_HEAD
#include "dynamic-head.h"
#endif
#ifndef FX2_NEURAL_CACHE
#define FX2_NEURAL_CACHE 0
#endif
#if FX2_NEURAL_CACHE
#include "neural-cache.h"
#endif

namespace {
using U64 = unsigned long long;
constexpr unsigned V = 205;
constexpr U64 MAX_PIECE = 1ULL << 17;
constexpr uint8_t SEPARATOR[15] = {
    0x08, 0x08, 0x25, 0xac, 0x65, 0x27, 0x05,
    0x08, 0x08, 0x08, 0x08, 0x25, 0xac, 0x68, 0x27};
using Clock = std::chrono::steady_clock;

double Seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

double CpuSeconds() {
  rusage u{};
  if (getrusage(RUSAGE_SELF, &u)) Fail("getrusage failed");
  return u.ru_utime.tv_sec + u.ru_utime.tv_usec * 1e-6 +
         u.ru_stime.tv_sec + u.ru_stime.tv_usec * 1e-6;
}

U64 Size(const char* path) {
  struct stat s{};
  if (stat(path, &s) || !S_ISREG(s.st_mode) || s.st_size < 0)
    Fail("cannot stat regular file %s", path);
  return static_cast<U64>(s.st_size);
}

void Fresh(const char* path) {
  struct stat s{};
  if (lstat(path, &s) == 0) Fail("output already exists: %s", path);
  if (errno != ENOENT) Fail("cannot check output path: %s", path);
}

U64 Number(const char* value) {
  if (!*value || *value < '0' || *value > '9') Fail("invalid integer: %s", value);
  errno = 0;
  char* end = nullptr;
  U64 n = std::strtoull(value, &end, 10);
  if (errno || *end) Fail("invalid integer: %s", value);
  return n;
}

std::vector<uint8_t> ReadSlice(const char* path, U64 offset, U64 length) {
  U64 size = Size(path);
  if (offset > size || length > size - offset ||
      length > static_cast<U64>(std::numeric_limits<std::streamsize>::max()))
    Fail("slice exceeds input: %s", path);
  std::ifstream in(path, std::ios::binary);
  in.seekg(static_cast<std::streamoff>(offset));
  std::vector<uint8_t> bytes(length);
  if (!in.read(reinterpret_cast<char*>(bytes.data()), length))
    Fail("short input: %s", path);
  return bytes;
}

struct Pieces {
  std::vector<U64> bounds{0};  // Includes final EOF once, never an empty piece.
  U64 separators = 0, length_cuts = 0;
};

Pieces Split(const std::vector<uint8_t>& tokens) {
  // Exactly Predictor::TransformerByteUpdate: keep the separator window
  // across forced 2^17 cuts; the next piece starts AFTER the separator.
  std::array<uint8_t, sizeof(SEPARATOR)> window;
  window.fill(0xff);
  U64 in_piece = 0;
  Pieces p;
  for (U64 i = 0; i < tokens.size(); ++i) {
    if (tokens[i] >= V) Fail("invalid token %u at %llu", tokens[i], i);
    std::memmove(window.data(), window.data() + 1, window.size() - 1);
    window.back() = tokens[i];
    ++in_piece;
    bool sep = std::memcmp(window.data(), SEPARATOR, window.size()) == 0;
    bool cut = in_piece >= MAX_PIECE;
    if (sep || cut) {
      p.separators += sep;
      p.length_cuts += cut;
      p.bounds.push_back(i + 1);
      in_piece = 0;
    }
  }
  if (p.bounds.back() != tokens.size()) p.bounds.push_back(tokens.size());
  return p;
}

void PrintPieces(const Pieces& p) {
  std::printf("\"pieces\":%zu,\"separator_ends\":%llu,\"length_cuts\":%llu,"
              "\"piece_bounds\":[", p.bounds.size() - 1, p.separators, p.length_cuts);
  for (size_t i = 0; i < p.bounds.size(); ++i)
    std::printf("%s%llu", i ? "," : "", p.bounds[i]);
  std::printf("]");
}

struct Loss {
  U64 count = 0;
  double logits = 0, probs = 0;
  void Add(double l, double p) { ++count; logits += l; probs += p; }
};

double LogitNll(const float* logits, unsigned target) {
  // Same double logsumexp as cpp_infer/src/opt/test_e2e_opt.cpp.
  float m = logits[0];
  for (unsigned i = 1; i < V; ++i) if (logits[i] > m) m = logits[i];
  double den = 0;
  for (unsigned i = 0; i < V; ++i) den += std::exp(double(logits[i]) - double(m));
  return -(double(logits[target]) - (double(m) + std::log(den)));
}

U64 HashBytes(U64 hash, const void* data, size_t size) {
  const auto* bytes = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < size; ++i) hash = (hash ^ bytes[i]) * 1099511628211ULL;
  return hash;
}

void PrintLoss(const char* name, const Loss& loss) {
  std::printf("\"%s\":{\"count\":%llu,\"logits_sum\":%.17g,\"probs_sum\":%.17g,"
              "\"logits_mean\":%.17g,\"probs_mean\":%.17g}", name, loss.count,
              loss.logits, loss.probs, loss.count ? loss.logits / loss.count : 0,
              loss.count ? loss.probs / loss.count : 0);
}

void PrintResources(Clock::time_point started, double cpu_start) {
  rusage u{};
  if (getrusage(RUSAGE_SELF, &u)) Fail("getrusage failed");
  std::printf(",\"seconds\":%.6f,\"cpu_seconds\":%.6f,\"max_rss_kib\":%ld",
              Seconds(started), CpuSeconds() - cpu_start, u.ru_maxrss);
}

int Capture(char** argv) {
  // capture INPUT TOKENS PRIORS VOCAB OFFSET LENGTH
  const char *input_path = argv[2], *token_path = argv[3],
             *prior_path = argv[4], *vocab_path = argv[5];
  const U64 offset = Number(argv[6]), length = Number(argv[7]);
  if (length < 2 || length > std::numeric_limits<U64>::max() / (V * 2))
    Fail("capture length must be at least two and fit the prior file size");
  Fresh(token_path);
  Fresh(prior_path);
  if (std::string(token_path) == prior_path) Fail("token/prior outputs must differ");
  if (Size(vocab_path) != 32) Fail("vocab must contain exactly 32 bytes");
  auto packed_vocab = ReadSlice(vocab_path, 0, 32);
  std::vector<bool> vocab(256, false);
  int byte_to_index[256];
  unsigned count = 0;
  for (unsigned i = 0; i < 256; ++i) {
    vocab[i] = (packed_vocab[i / 8] >> (i % 8)) & 1;
    byte_to_index[i] = vocab[i] ? count++ : -1;
  }
  if (count != V) Fail("expected fixed full-stream 205-byte vocabulary");
  auto tokens = ReadSlice(input_path, offset, length);
  // Validate the entire input before constructing PPMd or creating outputs.
  for (U64 i = 0; i < length; ++i)
    if (byte_to_index[tokens[i]] < 0) Fail("input byte at %llu is outside vocab", offset + i);
  std::ofstream out(token_path, std::ios::binary);
  if (!out) Fail("cannot create token output");
  const auto started = Clock::now();
  const double cpu_start = CpuSeconds();
  double capture_seconds, capture_cpu_seconds;
  {
    PredictorOptions options;
    options.ppmd_only = true;
    options.save_ppmd_probs = prior_path;
    options.num_input_bytes = length;
    std::srand(923);
    Predictor p(vocab, options);
    // Pretrain does not touch PPMd. No dictionary warmup, as in runner's
    // --ppmd-only path. Prediction/perception order matches Encoder::Encode.
    const auto loop_start = Clock::now();
    const double loop_cpu = CpuSeconds();
    for (U64 i = 0; i < length; ++i) {
      const uint8_t byte = tokens[i];
      for (int j = 7; j >= 0; --j) {
        p.Predict();
        p.Perceive((byte >> j) & 1);
      }
      tokens[i] = static_cast<uint8_t>(byte_to_index[byte]);
    }
    capture_seconds = Seconds(loop_start);
    capture_cpu_seconds = CpuSeconds() - loop_cpu;
  }  // Flush the original HalfFileWriter before checking its exact size.
  out.write(reinterpret_cast<const char*>(tokens.data()), length);
  out.close();
  if (!out || Size(token_path) != length || Size(prior_path) != length * V * 2)
    Fail("capture output length mismatch");
  Pieces pieces = Split(tokens);
  std::printf("ACTIVATION_RESULT {\"mode\":\"capture\",\"offset\":%llu,"
              "\"input_bytes\":%llu,\"vocab\":205,\"prior_bytes\":%llu,"
              "\"capture_seconds\":%.6f,\"capture_cpu_seconds\":%.6f,",
              offset, length, length * V * 2, capture_seconds, capture_cpu_seconds);
  PrintPieces(pieces);
  PrintResources(started, cpu_start);
  std::printf("}\n");
  return 0;
}

int Eval(char** argv) {
  // eval TOKENS PRIORS WEIGHTS
  const char *token_path = argv[2], *prior_path = argv[3], *weights_path = argv[4];
  const U64 length = Size(token_path);
  if (length < 2 || length > std::numeric_limits<U64>::max() / (V * 2) ||
      Size(prior_path) != length * V * 2)
    Fail("expected one token and exactly 205 f16 priors per input byte");
  auto tokens = ReadSlice(token_path, 0, length);
  Pieces pieces = Split(tokens);
  std::ifstream prior(prior_path, std::ios::binary);
  if (!prior) Fail("cannot open priors");
  std::array<uint16_t, V> row;
  std::array<float, V> probs;
  Loss total, halves[2];
#if FX2_NEURAL_CACHE
  NeuralCache cache;
  Fresh(argv[5]);
  std::ofstream pairs(argv[5], std::ios::binary);
  if (!pairs) Fail("cannot create cache probability pairs");
#endif
#if FX2_DYNAMIC_HEAD
  DynamicHead head;  // Zero once per independent slice; retained across pieces.
  Loss shadow, shadow_halves[2];
  struct Block { Loss base, shadow; };
  constexpr U64 BLOCK = 32768;
  std::vector<Block> blocks((length + BLOCK - 1) / BLOCK);
#endif
  U64 probability_hash = 14695981039346656037ULL;
#ifdef FX2_WINDOW_PROBE
  U64 short_context_hash = 14695981039346656037ULL, short_context_count = 0;
#endif
  struct Sample { U64 target_offset; unsigned target; float prob, logit; };
  std::vector<Sample> samples;
  const auto started = Clock::now();
  const double cpu_start = CpuSeconds();
  fx2::opt::TransformerOpt model(weights_path, fx2::opt::AttnKind::KVI8);
  const auto loop_start = Clock::now();
  const double loop_cpu = CpuSeconds();
  for (size_t piece = 0; piece + 1 < pieces.bounds.size(); ++piece) {
    const U64 begin = pieces.bounds[piece], end = pieces.bounds[piece + 1];
    model.begin_article(0);
    for (U64 i = begin; i < end; ++i) {
      if (!prior.read(reinterpret_cast<char*>(row.data()), sizeof(row)))
        Fail("short prior row at %llu", i);
      // Piece-last rows cannot predict the next piece's first token. The
      // final EOF row has no target; skip it like the upstream e2e test.
      if (i + 1 == end) continue;
#ifdef FX2_MLP_UNIT_STATS
      model.unit_stats_half((i + 1) >= length / 2);
#endif
      model.step(tokens[i], row.data(), probs.data());
      // Every raw fp32 probability, in step/vocabulary order; hash work is
      // included in eval time. x86-64 emits little-endian float bytes.
      probability_hash = HashBytes(probability_hash, probs.data(), sizeof(probs));
#ifdef FX2_WINDOW_PROBE
      // Both variants use the same variable-length kernel before position 1023.
      if (i - begin < 1023) {
        short_context_hash = HashBytes(short_context_hash, probs.data(), sizeof(probs));
        ++short_context_count;
      }
#endif
      const unsigned target = tokens[i + 1];
      const float* logits = model.last_logits();
      double ln = LogitNll(logits, target);
      double pn = -std::log(double(probs[target]));
      if (!std::isfinite(ln) || !std::isfinite(pn)) Fail("nonfinite NLL at %llu", i + 1);
      total.Add(ln, pn);
      halves[(i + 1) >= length / 2].Add(ln, pn);
#if FX2_NEURAL_CACHE
      cache.Predict(tokens[i], model.last_hidden(), probs.data());
      const float pair[2] = {probs[target], cache.probs[target]};
      if (!std::isfinite(pair[1]) || pair[1] < 0 || pair[1] > 1)
        Fail("invalid cache probability at %llu", i+1);
      pairs.write(reinterpret_cast<const char*>(pair), sizeof(pair));
      cache.Observe(target);
#endif
#if FX2_DYNAMIC_HEAD
      head.Predict(model.last_hidden(), logits);
      const double sn = head.Nll(target), sp = -std::log(double(head.probs[target]));
      if (!std::isfinite(sn) || !std::isfinite(sp)) Fail("nonfinite shadow NLL at %llu", i + 1);
      shadow.Add(sn, sp);
      shadow_halves[(i + 1) >= length / 2].Add(sn, sp);
      blocks[(i + 1) / BLOCK].base.Add(ln, pn);
      blocks[(i + 1) / BLOCK].shadow.Add(sn, sp);
      // The piece's last token is a target here. The next piece's first
      // token has no preceding neural prediction and is never observed.
      head.Observe(target);
#endif
      if (samples.size() < 4) samples.push_back({i + 1, target, probs[target], logits[target]});
    }
  }
  const double eval_seconds = Seconds(loop_start);
  const double eval_cpu_seconds = CpuSeconds() - loop_cpu;
  if (total.count != length - (pieces.bounds.size() - 1))
    Fail("predicted-token count mismatch");
  if (!total.count) Fail("no predicted tokens");
#ifdef FX2_MLP_UNIT_STATS
  Fresh(argv[5]);
  FILE* units = std::fopen(argv[5], "w");
  if (!units) Fail("cannot create unit statistics");
  std::fprintf(units, "layer,unit,first_hits,first_q2,second_hits,second_q2,norm2\n");
  const auto wf = fx2::WeightsFile::load_compressed(weights_path);
  const auto& us = model.activation_stats();
  for (unsigned l = 0; l < 12; ++l) {
    const std::string prefix = "blocks." + std::to_string(l) + ".mlp.down";
    const auto& w = wf.get(prefix + ".weight.q", fx2::DT_I8, {192, 768});
    const auto& s = wf.get(prefix + ".weight.scale", fx2::DT_BF16, {192});
    const auto& a = wf.get(prefix + ".quantize_activation.scale", fx2::DT_BF16, {1});
    U64 hits = 0;
    for (unsigned u = 0; u < 768; ++u) {
      double norm = 0;
      for (unsigned o = 0; o < 192; ++o) {
        const float fold = fx2::bf16_to_f32(a.bf16_bits()[0]) *
                           fx2::bf16_to_f32(s.bf16_bits()[o]);
        const double v = double(w.i8()[o * 768 + u]) * double(fold);
        norm += v * v;
      }
      if (!std::isfinite(norm) || norm < 0) Fail("invalid column norm");
      for (unsigned h = 0; h < 2; ++h) {
        const U64 n = us.hit[h][l][u], q2 = us.q2[h][l][u];
        if (n > halves[h].count || q2 < n || q2 > n * 127 * 127)
          Fail("unit count/q2 mismatch");
        hits += n;
      }
      std::fprintf(units, "%u,%u,%llu,%llu,%llu,%llu,%.17g\n", l, u,
          U64(us.hit[0][l][u]), U64(us.q2[0][l][u]),
          U64(us.hit[1][l][u]), U64(us.q2[1][l][u]), norm);
    }
    if (hits != us.down_nnz[l]) Fail("unit hit total mismatch");
  }
  if (std::ferror(units) || std::fclose(units)) Fail("unit output write failed");
#endif
#if FX2_NEURAL_CACHE
  pairs.close();
  if (!pairs || Size(argv[5]) != total.count * 2 * sizeof(float) ||
      cache.pending || cache.updates != total.count)
    Fail("cache pair/count mismatch");
#endif
  std::printf("ACTIVATION_RESULT {\"mode\":\"eval\",\"positive_u8\":%d,"
              "\"input_bytes\":%llu,\"prior_bytes\":%llu,\"vocab\":205,"
              "\"attention\":\"KVI8\",\"rope_offset\":0,\"half_split\":%llu,"
              "\"eval_seconds\":%.6f,\"eval_cpu_seconds\":%.6f,",
              FX2_POSITIVE_U8, length, length * V * 2, length / 2,
              eval_seconds, eval_cpu_seconds);
  PrintPieces(pieces);
  std::printf(",\"probability_hash\":\"%016llx\",\"probability_hash_format\":\"fnv1a64-f32le\","
              "\"probability_bytes\":%llu", probability_hash, total.count * V * sizeof(float));
  std::printf(","); PrintLoss("total", total);
  std::printf(","); PrintLoss("first_half", halves[0]);
  std::printf(","); PrintLoss("second_half", halves[1]);
#ifdef FX2_WINDOW_PROBE
  std::printf(",\"training_window\":1024,\"inference_window\":%d,"
              "\"short_context_hash\":\"%016llx\",\"short_context_count\":%llu",
              fx2::opt::ATTN_WIN, short_context_hash, short_context_count);
#endif
#if FX2_NEURAL_CACHE
  std::printf(",\"neural_cache\":true,\"cache_capacity_per_token\":%u,"
              "\"cache_ram_bytes\":%zu,\"cache_updates\":%llu,"
              "\"cache_comparisons\":%llu,\"empty_cache_queries\":%llu,"
              "\"pair_format\":\"base-cache-f32le\"", NeuralCache::K,
              cache.entries.size()*sizeof(NeuralCache::Entry), cache.updates,
              cache.comparisons, cache.empty_queries);
#endif
#if FX2_DYNAMIC_HEAD
  if (head.pending || head.updates != total.count || shadow.count != total.count)
    Fail("dynamic head prediction/update count mismatch");
  std::printf(",\"dynamic_head\":true,\"updates\":%llu,\"initial_weight_bytes\":0,"
              "\"weight_ram_bytes\":%zu,\"rate\":%.17g,\"keep\":%.17g,",
              head.updates, sizeof(head.weights), double(head.RATE), double(head.KEEP));
  PrintLoss("shadow_total", shadow);
  std::printf(","); PrintLoss("shadow_first_half", shadow_halves[0]);
  std::printf(","); PrintLoss("shadow_second_half", shadow_halves[1]);
  std::printf(",\"blocks\":[");
  for (size_t b = 0; b < blocks.size(); ++b) {
    std::printf("%s{\"begin\":%llu,\"end\":%llu,", b ? "," : "", U64(b) * BLOCK,
                std::min(length, U64(b + 1) * BLOCK));
    PrintLoss("base", blocks[b].base);
    std::printf(","); PrintLoss("shadow", blocks[b].shadow);
    std::printf("}");
  }
  std::printf("]");
#endif
#ifdef FX2_ACTIVATION_STATS
  const auto& stats = model.activation_stats();
  std::printf(",\"mlp_steps\":%llu,\"down_nnz_sum\":[", static_cast<U64>(stats.steps));
  for (unsigned l = 0; l < 12; ++l)
    std::printf("%s%llu", l ? "," : "", static_cast<U64>(stats.down_nnz[l]));
  std::printf("]");
#endif
  std::printf(",\"samples\":[");
  for (size_t i = 0; i < samples.size(); ++i) {
    const auto& s = samples[i];
    std::printf("%s{\"target_offset\":%llu,\"target\":%u,\"prob\":%.9g,\"logit\":%.9g}",
                i ? "," : "", s.target_offset, s.target, double(s.prob), double(s.logit));
  }
  std::printf("]");
  PrintResources(started, cpu_start);
  std::printf("}\n");
  return 0;
}

int SelfTest() {
  auto check = [](bool ok) { if (!ok) Fail("activation probe self-test failed"); };
  check(HashBytes(14695981039346656037ULL, "hello", 5) == 0xa430d84680aabd0bULL);
  auto counts = [&](const std::vector<uint8_t>& t, const std::vector<U64>& expected) {
    Pieces p = Split(t);
    check(p.bounds == expected);
    U64 scored = 0;
    for (size_t k = 0; k + 1 < p.bounds.size(); ++k)
      for (U64 i = p.bounds[k]; i + 1 < p.bounds[k + 1]; ++i) ++scored;
    check(scored == t.size() - (p.bounds.size() - 1));
  };
  counts({}, {0});
  counts({0}, {0, 1});
  counts({0, 1, 2, 3}, {0, 4});
  std::vector<uint8_t> t(SEPARATOR, SEPARATOR + sizeof(SEPARATOR));
  t.push_back(0); t.push_back(1);
  counts(t, {0, 15, 17});
  t.assign(MAX_PIECE + 2, 0);
  counts(t, {0, MAX_PIECE, MAX_PIECE + 2});
  t.resize(MAX_PIECE + 3);
  std::memcpy(t.data() + MAX_PIECE - 13, SEPARATOR, sizeof(SEPARATOR));
  counts(t, {0, MAX_PIECE, MAX_PIECE + 2, MAX_PIECE + 3});
  check(Split(t).separators == 1 && Split(t).length_cuts == 1);
  std::array<float, V> logits{};
  check(std::abs(LogitNll(logits.data(), 204) - std::log(205.0)) < 1e-12);
  logits.fill(999.0f); logits[0] = 1000.0f;
  const double expected = std::log(1.0 + 204.0 * std::exp(-1.0));
  check(std::abs(LogitNll(logits.data(), 0) - expected) < 1e-12);
  const float prob = static_cast<float>(std::exp(-expected));
  Loss loss;
  loss.Add(LogitNll(logits.data(), 0), -std::log(double(prob)));
  check(loss.count == 1 && std::abs(loss.probs - expected) < 1e-7);
#if FX2_NEURAL_CACHE
  {
    NeuralCache cache;
    std::array<float, NeuralCache::D> hidden;
    hidden.fill(1);
    std::array<float, V> base;
    base.fill(1.0f/V);
    cache.Predict(3, hidden.data(), base.data());
    check(cache.probs == base && cache.updates == 0 && cache.sizes[3] == 0);
    cache.Observe(7);
    cache.Predict(3, hidden.data(), base.data());
    check(cache.probs[7] == 1 && cache.probs[8] == 0 && cache.updates == 1);
    cache.Observe(8);
    cache.Predict(4, hidden.data(), base.data());
    check(cache.probs == base && cache.sizes[3] == 2);
    cache.Observe(9);
    for (unsigned i = 0; i < NeuralCache::K; ++i) {
      cache.Predict(3, hidden.data(), base.data());
      cache.Observe(8);
    }
    cache.Predict(3, hidden.data(), base.data());
    check(cache.sizes[3] == NeuralCache::K && cache.probs[7] == 0 && cache.probs[8] == 1);
    cache.Observe(8);
    check(!cache.pending && cache.updates == NeuralCache::K+4);
  }
#endif
#if FX2_DYNAMIC_HEAD
  DynamicHead head;
  std::array<float, DynamicHead::D> hidden;
  hidden.fill(1.0f);
  logits.fill(0);
  head.Predict(hidden.data(), logits.data());
  check(head.updates == 0 && head.logits == logits);
  check(std::abs(head.Nll(7) - LogitNll(logits.data(), 7)) < 1e-12);
  const float alpha = head.RATE / (1e-6f + 192.0f);
  const float target_weight = alpha * (1.0f - head.probs[7]);
  const float other_weight = -alpha * head.probs[7];
  head.Observe(7);
  check(!head.pending && head.updates == 1);
  check(head.weights[7][0] == target_weight && head.weights[0][191] == other_weight);
  head.Predict(hidden.data(), logits.data());
  check(head.Nll(7) < std::log(205.0));
  head.Observe(8);
  const auto weight_before_zero_hidden = head.weights[7][0];
  hidden.fill(0);
  head.Predict(hidden.data(), logits.data());
  head.Observe(7);
  check(head.weights[7][0] == head.KEEP * weight_before_zero_hidden && head.updates == 3);
#endif
  std::puts("ACTIVATION_SELF_TEST passed: boundaries, cross-cut separator, counts, double NLL, FNV-1a");
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--self-test") return SelfTest();
  if (argc == 8 && std::string(argv[1]) == "capture") return Capture(argv);
#ifdef FX2_MLP_UNIT_STATS
  if (argc == 6 && std::string(argv[1]) == "eval") return Eval(argv);
#else
  if (argc == 5+FX2_NEURAL_CACHE && std::string(argv[1]) == "eval") return Eval(argv);
#endif
  std::fprintf(stderr, "usage:\n  %s --self-test\n"
      "  %s capture INPUT TOKENS PRIORS VOCAB OFFSET LENGTH\n"
      "  %s eval TOKENS PRIORS WEIGHTS [CACHE_PAIRS or UNIT_CSV, when enabled]\n", argv[0], argv[0], argv[0]);
  return 2;
}
