// Link with src/mixer/{mixer,mixer-input,sigmoid}.cpp. Build both with and
// without -DHUTTER_CACHE_MIXER; stdout float streams must compare identical.
#include "mixer/mixer.h"
#include "mixer/mixer-input.h"
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>

int main() {
  Sigmoid sigmoid(100001);
  MixerInput table(sigmoid, 1.0e-4f), original(sigmoid, 1.0e-4f), fast(sigmoid, 1.0e-4f);
  table.SetNumModels(4097);
  // Every FXCM squash output (including the unused zero) and neutral padding.
  for (int i = 0; i <= 4096; ++i) {
    float p = i == 4096 ? 0.5f : i * (1.0f / 4095.0f);
    table.SetInput(i, p);
    original.SetInput(0, p);
    fast.SetStretchedInputUnchecked(0, table.Inputs()[i]);
    float a = original.Inputs()[0], b = fast.Inputs()[0];
    assert(std::memcmp(&a, &b, sizeof(a)) == 0);
  }

  std::valarray<float> inputs(0.0f, 586), extras(0.0f, 23);
  unsigned long long context = 0;
  Mixer mixer(inputs, extras, context, 0.005f, 23);
  std::uint32_t random = 923;
  auto next = [&]() { random = random * 1664525u + 1013904223u; return random; };
  const auto start = std::chrono::steady_clock::now();
  for (unsigned i = 0; i < 50000; ++i) {
    // Cross the 10,000-context cap, reuse earlier contexts, and train fallback.
    context = i < 11000 ? i : next() % 16000;
    for (auto& value : inputs) value = (int(next() >> 20) - 2048) / 256.0f;
    for (auto& value : extras) value = (int(next() >> 20) - 2048) / 256.0f;
    float p = mixer.Mix();
    if (std::fwrite(&p, sizeof(p), 1, stdout) != 1) return 1;
    mixer.Perceive(next() >> 31);
  }
  const double seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - start).count();
  std::fprintf(stderr, "{\"lookup_cases\":4097,\"mixer_steps\":50000,\"seconds\":%.6f}\n", seconds);
}
