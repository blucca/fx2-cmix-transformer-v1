// SPDX-License-Identifier: GPL-3.0-only
// Build like fast-path-check.cpp, linking mixer.cpp and sigmoid.cpp.
#include "mixer/mixer.h"
#include <cassert>
#include <iostream>

int main() {
  const std::valarray<float> input{1.0f}, extra;
  unsigned long long context = 0;
  Mixer original(input, extra, context, 0.005f, 0);
  Mixer full(input, extra, context, 0.005f, 0, 65536);
  // Both paths admit the first 10,000 keys. Later new keys get independent
  // zero-initialized weights only in the expanded model.
  for (context = 0; context < 65536; ++context) {
    const float old = original.Mix(), expanded = full.Mix();
    assert(expanded == 0);
    if (context <= 10000) assert(old == 0);
    else assert(old > 0);
    original.Perceive(1);
    full.Perceive(1);
  }
  context = 65535;
  assert(full.Mix() > 0);  // The final 16-bit key retains its own update.
  std::cout << "context-cap-check passed\n";
}
