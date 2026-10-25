// Fixed byte-bucket continuous cache. Query has no target input; insert only
// after the caller scores the next byte. One independent stream per instance.
#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <vector>

#include "../cpp_infer/src/opt/kda_math.h"

struct NeuralCache {
  static constexpr unsigned V = 205, D = 192, K = 128;
  struct Entry { std::array<float, D> key{}; unsigned char label = 0; };
  // ponytail: exact search within the same last-byte bucket, at most 128 keys.
  std::vector<Entry> entries = std::vector<Entry>(V * K);
  std::array<unsigned, V> sizes{}, cursors{};
  std::array<float, D> query{};
  std::array<float, V> probs{};
  unsigned bucket = 0;
  bool pending = false;
  unsigned long long updates = 0, comparisons = 0, empty_queries = 0;

  static float Dot(const float* a, const float* b) {
    __m256 acc = _mm256_setzero_ps();
    for (unsigned i = 0; i < D; i += 8)
      acc = _mm256_fmadd_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i), acc);
    return fx2::opt::hsum256v(acc);
  }

  void Predict(unsigned token, const float* hidden, const float* base) {
    assert(!pending && token < V);
    const float norm = std::sqrt(Dot(hidden, hidden));
    assert(std::isfinite(norm) && norm > 0);
    for (unsigned d = 0; d < D; ++d) query[d] = hidden[d] / norm;
    bucket = token;
    pending = true;
    if (!sizes[bucket]) {
      std::copy_n(base, V, probs.data());
      ++empty_queries;
      return;
    }
    std::array<float, K> dots;
    const auto* rows = entries.data() + bucket*K;
    float maximum = -2;
    for (unsigned i = 0; i < sizes[bucket]; ++i) {
      dots[i] = Dot(query.data(), rows[i].key.data());
      maximum = std::max(maximum, dots[i]);
    }
    std::array<double, V> sums{};
    double denominator = 0;
    for (unsigned i = 0; i < sizes[bucket]; ++i) {
      const double weight = std::exp(20.0 * (double(dots[i]) - maximum));
      sums[rows[i].label] += weight;
      denominator += weight;
    }
    for (unsigned c = 0; c < V; ++c) probs[c] = float(sums[c] / denominator);
    comparisons += sizes[bucket];
  }

  void Observe(unsigned target) {
    assert(pending && target < V);
    auto& row = entries[bucket*K+cursors[bucket]];
    row.key = query;
    row.label = static_cast<unsigned char>(target);
    cursors[bucket] = (cursors[bucket]+1) % K;
    sizes[bucket] = std::min(K, sizes[bucket]+1);
    pending = false;
    ++updates;
  }
};
