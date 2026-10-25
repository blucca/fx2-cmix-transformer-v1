// Focused scalar check for the default and FX2_POSITIVE_U8 packed kernels.
// Build with qmat_dense.cpp, qmat_sparse.cpp and glue.cpp; no model needed.
#include "../cpp_infer/src/opt/qmat_dense.h"
#include "../cpp_infer/src/opt/qmat_sparse.h"
#include "../cpp_infer/src/opt/glue.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace fx2::opt;

struct Buf {
  std::vector<uint8_t> bytes;
  uint8_t* p;
  explicit Buf(size_t n) : bytes(n + 64 + QMAT_TAIL_SLACK, 0) {
    p = reinterpret_cast<uint8_t*>((uintptr_t(bytes.data()) + 63) & ~uintptr_t(63));
  }
};

static void sparse_check() {
  // Odd tails, both sides of 9/18-pair flushes, prior and MLP dimensions.
  for (int n : {1, 17, 18, 19, 20, 35, 36, 37, 38, 205, 768}) {
    std::vector<int8_t> w(size_t(192) * n);
    std::vector<uint8_t> q(n);
    std::vector<uint16_t> idx(n + 8);
    alignas(64) float fold[192], packed_fold[192], gotf[192];
    alignas(64) int32_t ref[192], got[192];
    Buf cols(qsparse4_bytes(n));
    for (int r = 0; r < 192; ++r) {
      fold[r] = r % 2 ? -0.03125f : 0.0625f;
      for (int c = 0; c < n; ++c)
        w[size_t(r) * n + c] = r == 0 ? 7 : r == 1 ? -7 : (r * 3 + c * 5) % 15 - 7;
    }
    const QSparse4 m = qsparse4_build(cols.p, packed_fold, w.data(), fold, 192, n);
    for (int pattern = 0; pattern < 4; ++pattern) {
      const int values[] = {0, 127, 128, 254};
      int nnz = 0;
      for (int c = 0; c < n; ++c) {
        int a = pattern == 0 ? 0 : pattern == 1 ? QMAT_RELU2_MAX :
                pattern == 2 ? values[c % 4] : (c % 3 ? 127 : 0);
        q[c] = std::min(a, QMAT_RELU2_MAX);
        if (q[c]) idx[nnz++] = c;
      }
      std::fill(idx.begin() + nnz, idx.end(), nnz ? idx[nnz - 1] : 0);
      for (int r = 0; r < 192; ++r) {
        int32_t sum = 0;
        for (int c = 0; c < n; ++c) sum += int(q[c]) * w[size_t(r) * n + c];
        ref[r] = sum;
      }
      for (bool dense : {false, true}) {
        if (dense) qsparse4_dense_i32(m, q.data(), got);
        else qsparse4_i32(m, q.data(), idx.data(), nnz, got);
        assert(std::memcmp(ref, got, sizeof(ref)) == 0);
        if (dense) qsparse4_dense_f32(m, q.data(), gotf);
        else qsparse4_f32(m, q.data(), idx.data(), nnz, gotf);
        for (int r = 0; r < 192; ++r) assert(gotf[r] == fold[r] * float(ref[r]));
        std::fill(gotf, gotf + 192, 0.25f);
        if (dense) qsparse4_dense_add(m, q.data(), gotf);
        else qsparse4_add(m, q.data(), idx.data(), nnz, gotf);
        for (int r = 0; r < 192; ++r) {
          volatile float product = fold[r] * float(ref[r]);
          assert(gotf[r] == 0.25f + product);
        }
      }
    }
  }
}

static void quant_check() {
  std::vector<int8_t> w(size_t(768) * 192, 0);
  alignas(64) float y[768];
  alignas(64) uint8_t act[192], q[768], reference[768];
  uint16_t idx[768 + 8];
  std::fill(act, act + 192, 128);
  act[0] = 129;  // signed input 1; each dot is exactly 1
  const float values[] = {-4, -0.0f, 0, 1, 3, 15, 16, 64};
  for (int r = 0; r < 768; ++r) {
    w[size_t(r) * 192] = 1;
    y[r] = values[r % 8];
  }
  Buf arena(qdense_bytes(768, 192));
  QDense m = qdense_build(arena.p, w.data(), y, 768, 192, true);
  for (float s : {0.5f, 2.0f, 6.0f}) {
    int nnz = qgemv_relu2q(m, act, s, q, idx), count = 0;
    relu2_quant768(y, s, reference);
    for (int r = 0; r < 768; ++r) {
      float h = std::max(y[r], 0.0f);
      int want = int(std::nearbyint(std::min(h * h / s, float(FX2_POSITIVE_U8 ? 254 : 127))));
      assert(q[r] == want && reference[r] == want);
      if (want) assert(idx[count++] == r);
    }
    assert(nnz == count);
    if (s == 2.0f) assert(q[3] == 0);  // 1/2 ties down to even
    if (s == 6.0f) assert(q[4] == 2);  // 9/6 ties up to even
  }
}

int main() {
  sparse_check();
  quant_check();
  std::printf("positive activation check passed (FX2_POSITIVE_U8=%d)\n", FX2_POSITIVE_U8);
}
