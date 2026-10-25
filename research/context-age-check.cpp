// SPDX-License-Identifier: GPL-3.0-only
// g++ -O2 -std=c++17 -I src -DHUTTER_CONTEXT_AGE_CHECK \
//   -DHUTTER_CONTEXT_AGE=1 research/context-age-check.cpp \
//   src/mixer/{mixer,sigmoid}.cpp -o /YOUR/temp/context-age-check
// Also run with AGE=0, CACHE_MIXER, and TRACE + MIXER_TRACE definitions.
#include "mixer/mixer.h"
#include <cassert>
#include <cstdio>

#ifndef HUTTER_CONTEXT_AGE_CHECK
#error "Build this check and mixer.cpp with HUTTER_CONTEXT_AGE_CHECK"
#endif

struct ContextAgeCheck {
  static float Decay(std::uint64_t age) {
    return age < 1000000 ? 1.0f : age < 5000000 ? 0.7f :
        age < 25000000 ? 0.3f : 0.2f;
  }

  static void Step(Mixer& mixer, std::uint64_t global,
      std::uint32_t exposures) {
    ContextData* data = mixer.GetContextData();
    data->weights = 0.0f;
    data->extra_weights = 0.0f;
    mixer.steps_ = global;
    auto age = global;
#if HUTTER_CONTEXT_AGE
    data->exposures = exposures;
    if (data != &mixer.context_base_)
      age = std::uint64_t(exposures) * mixer.context_map_.size();
#endif
#ifdef HUTTER_MIXER_TRACE
    data->trace_applied_updates = 0;
#endif
    assert(mixer.Mix() == 0.0f);
    mixer.Perceive(1);
    const float expected = Decay(age) * 0.0625f;
    assert(data->weights[0] == expected);
    assert(data->weights[1] == -0.5f * expected);
    assert(data->extra_weights[0] == 0.25f * expected);
    assert(mixer.steps_ == global + 1);
#if HUTTER_CONTEXT_AGE
    assert(data->exposures == exposures + (exposures < 25000000));
    const ContextData copy = *data;
    assert(copy.exposures == data->exposures);
#endif
#ifdef HUTTER_MIXER_TRACE
    assert(data->trace_applied_updates == 1);
#endif
  }

  static void Run() {
    const std::valarray<float> inputs{1.0f, -0.5f}, extras{0.25f};
    unsigned long long key = 0;
    Mixer single(inputs, extras, key, 0.125f, 1, 1);
#if HUTTER_CONTEXT_AGE
    assert(single.GetContextData()->exposures == 0);
#endif
    // K=1 retains the original thresholds, including saturated mature age.
    for (std::uint32_t n : {0u, 999999u, 1000000u, 4999999u, 5000000u,
                            24999999u, 25000000u})
      Step(single, n, n);
    Step(single, 40000000, 25000000);

    Mixer two(inputs, extras, key, 0.125f, 1, 2);
    two.Mix();
    key = 1;
    two.Mix();
    for (std::uint32_t n : {0u, 499999u, 500000u, 2499999u, 2500000u,
                            12499999u, 12500000u, 25000000u})
      Step(two, 40000000, n);
    key = 2;  // Fallback always uses global age, even on its first visit.
    for (std::uint32_t n : {999999u, 1000000u, 4999999u, 5000000u,
                            24999999u, 25000000u})
      Step(two, n, 0);

    Mixer wide(inputs, extras, key, 0.125f, 1, 512);
    for (key = 0; key < 512; ++key) wide.Mix();
    key = 0;
    Step(wide, 40000000, 8388608);  // 512 * 2^23 = 2^32, not age zero.

    Mixer skipped(inputs, extras, key, 0.0f, 1, 1);
    for (key = 0; key < 2; ++key) {
      skipped.Mix();  // Independent key first, then shared fallback.
      ContextData* data = skipped.GetContextData();
      skipped.steps_ = 40000000;
#if HUTTER_CONTEXT_AGE
      data->exposures = 24999999;
#endif
      for (unsigned i = 0; i < 2; ++i) {
        skipped.Mix();
        skipped.Perceive(1);  // Zero learning rate takes the existing return.
        assert(skipped.steps_ == 40000001 + i);
        assert(data->weights[0] == 0 && data->extra_weights[0] == 0);
#if HUTTER_CONTEXT_AGE
        assert(data->exposures == 25000000);
#endif
#ifdef HUTTER_MIXER_TRACE
        assert(data->trace_applied_updates == 0);
#endif
      }
    }
    using Map = emhash6::HashMap<unsigned int, ContextData>;
    std::printf("context-age-check passed: age=%d, ContextData=%zu, bucket=%zu; "
        "thresholds, K=1, late key, fallback, uint64, early return, saturation\n",
        HUTTER_CONTEXT_AGE, sizeof(ContextData), sizeof(Map::PairT));
  }
};

int main() { ContextAgeCheck::Run(); }
