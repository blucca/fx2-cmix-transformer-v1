// Fixed, research-only online residual readout. Predict has no target input;
// Observe consumes its prediction once, after the caller scores the target.
#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>

#include "../cpp_infer/src/opt/glue.h"
#include "../cpp_infer/src/opt/kda_math.h"

struct DynamicHead {
  static constexpr unsigned V = 205, D = 192;
  static constexpr float RATE = 0x1p-5f, KEEP = 1.0f - 0x1p-15f;
  alignas(32) float weights[V][D] = {};
  std::array<float, V> logits{}, probs{};
  const float* pending = nullptr;
  unsigned long long updates = 0;
  double log_normalizer = 0;

  // Same AVX2 FMA chain and reduction as model_opt.cpp's private dot192.
  static float Dot(const float* a, const float* b) {
    __m256 acc = _mm256_setzero_ps();
    for (unsigned i = 0; i < D; i += 8)
      acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
    return fx2::opt::hsum256v(acc);
  }

  void Predict(const float* hidden, const float* capped_logits) {
    assert(!pending);
    pending = hidden;
    for (unsigned c = 0; c < V; ++c)
      logits[c] = capped_logits[c] + Dot(weights[c], hidden);
    const float maximum = *std::max_element(logits.begin(), logits.end());
    std::array<double, V> exponentials;
    double denominator = 0;
    for (unsigned c = 0; c < V; ++c) {
      exponentials[c] = std::exp(double(logits[c]) - double(maximum));
      denominator += exponentials[c];
    }
    log_normalizer = double(maximum) + std::log(denominator);
    for (unsigned c = 0; c < V; ++c)
      probs[c] = static_cast<float>(exponentials[c] / denominator);
  }

  double Nll(unsigned target) const {
    assert(pending && target < V);
    return log_normalizer - double(logits[target]);
  }

  void Observe(unsigned target) {
    assert(pending && target < V);
    const float alpha = RATE / (1e-6f + Dot(pending, pending));
    for (unsigned c = 0; c < V; ++c)
      fx2::opt::axpby_tok192(KEEP, weights[c],
          alpha * ((c == target ? 1.0f : 0.0f) - probs[c]), pending);
    pending = nullptr;
    ++updates;
  }
};
