// Encoder-only normalization shared by the census and exact recoder.
#pragma once
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <numeric>

using Counts = std::array<uint64_t, 15>;

inline Counts static15_frequencies(const Counts& c) {
  const uint64_t n = std::accumulate(c.begin(), c.end(), uint64_t{0});
  assert(n > 0);
  Counts freq{}, remainder{};
  uint64_t used = 0;
  for (int s = 0; s < 15; ++s) {
    freq[s] = 1 + c[s] * (4096 - 15) / n;
    remainder[s] = c[s] * (4096 - 15) % n;
    used += freq[s];
  }
  while (used < 4096) {
    const auto s = std::max_element(remainder.begin(), remainder.end()) - remainder.begin();
    ++freq[s]; ++used;
    remainder[s] = 0;
  }
  assert(std::accumulate(freq.begin(), freq.end(), uint64_t{0}) == 4096);
  return freq;
}
