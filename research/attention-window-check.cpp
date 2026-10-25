// One scalar boundary check for the fixed KVI8 window experiment.
#include "../cpp_infer/src/opt/attn.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

int main() {
  using namespace fx2::opt;
  auto kv = std::make_unique<AttnKV>();
  std::memset(kv.get(), 0xa5, sizeof(*kv));
  std::vector<std::array<int8_t, 192>> keys(ATTN_WIN), values(ATTN_WIN);
  alignas(32) std::array<int8_t, 192> q, k, v;
  const float coef[3] = {0.0003f, 0.0008f, 0.0012f};
  const float scale[3] = {0.015f, 0.020f, 0.033f};
  int checks = 0;
  double max_error = 0;
  for (int t = 0; t <= ATTN_WIN + 1; ++t) {
    for (int d = 0; d < 192; ++d) {
      q[d] = (t * 7 + d * 11) % 255 - 127;
      k[d] = (t * 13 + d * 3) % 255 - 127;
      v[d] = (t * 17 + d * 19) % 255 - 127;
    }
    const int slot = t % ATTN_WIN, n = std::min(t + 1, ATTN_WIN);
    keys[slot] = k;
    values[slot] = v;
    attn_kv_insert(*kv, slot, k.data(), v.data());
    if (t != 0 && t != 1022 && t != 1023 && t != 2046 &&
        t != 2047 && t != ATTN_WIN && t != ATTN_WIN + 1) continue;
    alignas(32) std::array<float, 192> got;
    int kept[3] = {};
    if (n < ATTN_WIN)
      attn_step_var(*kv, q.data(), coef, scale, n, got.data(), 0, kept);
    else
      attn_step_fixed(*kv, q.data(), coef, scale, got.data(), 0, kept);
    for (int h = 0; h < 3; ++h) {
      std::vector<float> scores(n);
      for (int j = 0; j < n; ++j) {
        int dot = 0;
        for (int d = 0; d < 64; ++d)
          dot += int(q[h * 64 + d]) * int(keys[j][h * 64 + d]);
        scores[j] = coef[h] * float(dot);
      }
      const float top = *std::max_element(scores.begin(), scores.end());
      std::vector<double> exps(n);
      double den = 0;
      for (int j = 0; j < n; ++j) {
        exps[j] = std::exp(double(scores[j] - top));
        den += exps[j];
      }
      if (kept[h] != n) return 1;
      for (int d = 0; d < 64; ++d) {
        double num = 0;
        for (int j = 0; j < n; ++j) num += exps[j] * values[j][h * 64 + d];
        const double expected = num / den * scale[h];
        const double error = std::abs(got[h * 64 + d] - expected);
        max_error = std::max(max_error, error);
        if (!std::isfinite(got[h * 64 + d]) ||
            error > 2e-5 * std::max(1.0, std::abs(expected))) {
          std::fprintf(stderr, "window=%d t=%d n=%d head=%d dim=%d error=%.9g\n",
                       ATTN_WIN, t, n, h, d, error);
          return 1;
        }
      }
    }
    ++checks;
  }
  std::printf("ATTENTION_WINDOW_CHECK passed window=%d checks=%d max_error=%.9g\n",
              ATTN_WIN, checks, max_error);
}
