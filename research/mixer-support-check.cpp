// SPDX-License-Identifier: GPL-3.0-only
// Link src/mixer/{mixer,mixer-input,sigmoid}.cpp with -I src. Build normally
// and with -DHUTTER_TRACE -DHUTTER_MIXER_TRACE; stdout float streams must match.
#include "mixer/mixer.h"
#include "mixer/mixer-input.h"
#include "trace.h"
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>

static_assert(offsetof(HutterMixerTraceRecord, logits) == 0);
static_assert(offsetof(HutterMixerTraceRecord, support) == 46);
static_assert(offsetof(HutterMixerTraceRecord, flags) == 69);
static_assert(offsetof(HutterMixerTraceRecord, reserved) == 70);
static_assert(sizeof(HutterMixerTraceRecord) == 72);

int main() {
#ifdef HUTTER_MIXER_TRACE
  assert(HutterMixerLogitHalf(0.f) == 0);
  assert(HutterMixerLogitHalf(-0.f) == 0x8000);
  assert(HutterMixerLogitHalf(1.f) == 0x3c00);
  assert(HutterMixerLogitHalf(-1.f) == 0xbc00);
  assert(HutterMixerLogitHalf(std::ldexp(1.f, -24)) == 1);
  assert(HutterMixerLogitHalf(std::ldexp(1.f, -25)) == 0);
  assert(HutterMixerLogitHalf(std::ldexp(3.f, -25)) == 2);
  assert(HutterMixerLogitHalf(1.f + std::ldexp(1.f, -11)) == 0x3c00);
  assert(HutterMixerLogitHalf(1.f + std::ldexp(3.f, -11)) == 0x3c02);
  assert(HutterMixerSupport(0, false) == 0);
  assert(HutterMixerSupport(0, true) == 128);
  assert(HutterMixerSupport(1, false) == 1);
  assert(HutterMixerSupport(2, false) == 1);
  assert(HutterMixerSupport(3, false) == 2);
  std::uint32_t saturated = UINT32_MAX - 1;
  assert(HutterMixerSupport(saturated, false) == 31);
  HutterMixerAppliedUpdate(saturated);
  HutterMixerAppliedUpdate(saturated);
  assert(saturated == UINT32_MAX && HutterMixerSupport(saturated, true) == 160);

  // The row stores what SetStretchedInput actually passed to layer two.
  Sigmoid sigmoid(100001);
  MixerInput second_layer(sigmoid, 1.0e-4f);
  second_layer.SetNumModels(23);
  HutterMixerTraceRecord row{};
  for (int i = 0; i < 23; ++i) {
    second_layer.SetStretchedInput(i, (i - 11) * 100.f);
    row.logits[i] = HutterMixerLogitHalf(second_layer.Inputs()[i]);
    row.support[i] = HutterMixerSupport(i, i & 1);
    assert(std::abs(second_layer.Inputs()[i]) < 13);
    assert((row.support[i] & 64) == 0);
  }
  row.flags = 1;
  assert(row.logits[11] == 0 && row.flags == 1);
  assert(row.reserved[0] == 0 && row.reserved[1] == 0);
#endif

  std::valarray<float> inputs{0.25f, -0.75f, 0.5f}, extras{0.125f};
  unsigned long long context = 0;
  Mixer normal(inputs, extras, context, .005f, 0, 2);
  Mixer with_extra(inputs, extras, context, .005f, 1, 2);
  Mixer early_return(inputs, extras, context, 0.f, 1, 2);
  std::array<std::uint32_t, 3> updates{};
  const std::array<unsigned, 8> keys{2, 0, 3, 1, 0, 4, 1, 2};
  std::uint32_t random = 923;
  for (unsigned i = 0; i < 4096; ++i) {
    context = i < 2 ? i : keys[(i - 2) % keys.size()];
    const unsigned selected = context < 2 ? unsigned(context) : 2;
    random = random * 1664525u + 1013904223u;
    inputs[0] = (int(random >> 24) - 128) / 256.f;
    const std::array<float, 3> predictions{
        normal.Mix(), with_extra.Mix(), early_return.Mix()};
    for (float p : predictions) assert(std::isfinite(p));
    assert(predictions[2] == 0);
    assert(std::fwrite(predictions.data(), sizeof(float), predictions.size(), stdout)
        == predictions.size());
#ifdef HUTTER_MIXER_TRACE
    const auto expected = HutterMixerSupport(updates[selected], selected == 2);
    assert(normal.TraceSupport() == expected);
    assert(with_extra.TraceSupport() == expected);
    assert(early_return.TraceSupport() == (selected == 2 ? 128 : 0));
#endif
    const int bit = random >> 31;
    normal.Perceive(bit);
    with_extra.Perceive(bit);
    early_return.Perceive(bit);
#ifdef HUTTER_MIXER_TRACE
    // Trace remains the pre-label snapshot until the next Mix selection.
    assert(normal.TraceSupport() == expected);
    assert(with_extra.TraceSupport() == expected);
    assert(early_return.TraceSupport() == (selected == 2 ? 128 : 0));
#endif
    ++updates[selected];
  }
  std::fprintf(stderr,
      "mixer-support-check passed: 4096 selections, 3 mixers, 72-byte layout\n");
}
