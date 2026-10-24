// SPDX-License-Identifier: GPL-3.0-only
// Link src/models/{ppmd,byte-model}.cpp with -I src. Run in a scratch directory
// (PPMd creates ppm.temp). Build with and without -DHUTTER_TRACE
// -DHUTTER_PPMD_TRACE; stdout probability streams must compare identical.
#include "models/ppmd.h"
#include "trace.h"
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>

static_assert(offsetof(HutterPpmdTraceRecord, ppmd) == 0);
static_assert(offsetof(HutterPpmdTraceRecord, entropy) == 4);
static_assert(offsetof(HutterPpmdTraceRecord, escape) == 8);
static_assert(offsetof(HutterPpmdTraceRecord, suffix_depth) == 12);
static_assert(offsetof(HutterPpmdTraceRecord, flags) == 14);

int main() {
  std::vector<bool> vocab(256, false);
  for (int i = 0; i < 205; ++i) vocab[i] = true;
  unsigned int byte = 0;
  PPMD::PPMD model(25, 16, byte, vocab);
  std::uint32_t random = 923;
#ifdef HUTTER_PPMD_TRACE
  bool saw_suffix = false;
#endif
  for (unsigned int i = 0; i < 1024; ++i) {
    random = random * 1664525u + 1013904223u;
    const unsigned int next = i < 512 ? i % 7 : random % 205;
#ifdef HUTTER_PPMD_TRACE
    const auto prior = model.Trace();
    assert(prior.flags == (i ? 1 : 0));
    assert(std::isfinite(prior.entropy) && prior.entropy >= 0 && prior.entropy <= 8);
    assert(prior.suffix_depth <= 25);
    if (i == 0) {
      assert(prior.ppmd == 0.5f && prior.entropy == 8);
      assert(prior.escape == -1 && prior.suffix_depth == 0);
    } else {
      assert(prior.escape >= 0 && prior.escape <= 1);
      saw_suffix |= prior.suffix_depth > 0;
      const auto& distribution = model.BytePredict();
      double sum = 0, weighted_log = 0;
      for (int j = 0; j < 256; ++j) {
        const double p = distribution[j];
        assert(vocab[j] ? p > 0 : p == 0);
        sum += p;
        if (p > 0) weighted_log += p * std::log2(p);
      }
      assert(std::abs(prior.entropy - (std::log2(sum) - weighted_log / sum)) < 1e-6);
      if (i == 1) {
        // One observed byte: its root frequency is 5, the other 204 are 1.
        const double expected = std::log2(209.0) - 5.0 / 209 * std::log2(5.0);
        assert(std::abs(prior.entropy - expected) < 1e-6);
        assert(prior.escape == 1.0f / 261 && prior.suffix_depth == 0);
      }
    }
#endif
    for (int shift = 7; shift >= 0; --shift) {
      const float p = model.Predict()[0];
      assert(std::isfinite(p) && p >= 0 && p <= 1);
#ifdef HUTTER_PPMD_TRACE
      const auto& trace = model.Trace();
      assert(trace.ppmd == p);
      assert(trace.entropy == prior.entropy && trace.escape == prior.escape);
      assert(trace.suffix_depth == prior.suffix_depth && trace.flags == prior.flags);
      if (i == 0) assert(p == 0.5f);
#endif
      assert(std::fwrite(&p, sizeof(p), 1, stdout) == 1);
      model.Perceive((next >> shift) & 1);
    }
    byte = next;
    model.ByteUpdate();
    assert(std::fwrite(&model.BytePredict()[0], sizeof(float), 256, stdout) == 256);
  }
#ifdef HUTTER_PPMD_TRACE
  assert(saw_suffix);
#endif
  std::fprintf(stderr, "ppmd-trace-check passed: 1024 bytes, 8192 bits\n");
}
