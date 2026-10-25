// FP32 article-reset KDA and full adjoint, with bounded recomputation checkpoints.
// Build: clang++-17 -O3 -mavx2 -ffp-contract=off -shared -fPIC cpu-kda.cpp -o cpu-kda.so
#include <algorithm>
#include <cstdint>
#include <vector>

namespace {
constexpr int BLOCK = 128;

// The same recurrence is used for forward and backward recomputation.
void step(int t, int H, int K, int V, const float* q, const float* k,
          const float* v, const float* d, const float* beta, float* s,
          float* out, float* r, float* u, float* partial) {
  const int KV = K * V;
  for (int h = 0; h < H; ++h) {
    const int th = t * H + h;
    const float *qt = q + th * K, *kt = k + th * K;
    const float *dt = d + th * K, *vt = v + th * V;
    float* st = s + h * KV;
    std::fill(r, r + V, 0.f);
    // Torch's K=64 outer sum cascades groups of 16. Match its FP32 order
    // so tiny differences do not cross later hard-QAT thresholds.
    for (int begin = 0; begin < K; begin += 16) {
      std::fill(partial, partial + V, 0.f);
      for (int i = begin; i < std::min(begin + 16, K); ++i) {
        for (int j = 0; j < V; ++j) {
          st[i * V + j] *= dt[i];
          partial[j] += kt[i] * st[i * V + j];
        }
      }
      for (int j = 0; j < V; ++j) r[j] += partial[j];
    }
    for (int j = 0; j < V; ++j) u[j] = beta[th] * (vt[j] - r[j]);
    float* ot = out ? out + th * V : nullptr;
    if (ot) std::fill(ot, ot + V, 0.f);
    for (int begin = 0; begin < K; begin += 16) {
      std::fill(partial, partial + V, 0.f);
      for (int i = begin; i < std::min(begin + 16, K); ++i) {
        for (int j = 0; j < V; ++j) {
          st[i * V + j] += kt[i] * u[j];
          if (ot) partial[j] += qt[i] * st[i * V + j];
        }
      }
      if (ot) for (int j = 0; j < V; ++j) ot[j] += partial[j];
    }
  }
}
}

extern "C" void kda_forward(
    int T, int H, int K, int V, int N, const int32_t* bounds,
    const float* q, const float* k, const float* v, const float* d,
    const float* beta, float* checkpoints, float* out) {
  const int HKV = H * K * V;
  std::vector<float> s(HKV), r(V), u(V), partial(V);
  for (int a = 0; a < N - 1; ++a) {
    std::fill(s.begin(), s.end(), 0.f);
    for (int t = bounds[a]; t < bounds[a + 1]; ++t) {
      if (t % BLOCK == 0)
        std::copy(s.begin(), s.end(), checkpoints + int64_t(t / BLOCK) * HKV);
      step(t, H, K, V, q, k, v, d, beta, s.data(), out,
           r.data(), u.data(), partial.data());
    }
  }
}

extern "C" void kda_backward(
    int T, int H, int K, int V, int N, const int32_t* bounds,
    const float* q, const float* k, const float* v, const float* d,
    const float* beta, const float* checkpoints, const float* go,
    float* gq, float* gk, float* gv, float* gd, float* gbeta) {
  const int KV = K * V, HKV = H * KV;
  std::vector<float> adj(HKV), zero(HKV, 0.f), s(HKV), states(BLOCK * HKV);
  std::vector<float> A(KV), r(V), u(V), gu(V), partial(V);
  for (int a = 0; a < N - 1; ++a) {
    std::fill(adj.begin(), adj.end(), 0.f);
    for (int end = bounds[a + 1]; end > bounds[a];) {
      const int lo = std::max(bounds[a], (end - 1) / BLOCK * BLOCK);
      const float* initial = lo == bounds[a] ? zero.data()
          : checkpoints + int64_t(lo / BLOCK) * HKV;
      std::copy(initial, initial + HKV, s.data());
      // Recreate at most 128 states; the adjoint crosses every block boundary.
      for (int t = lo; t < end; ++t) {
        step(t, H, K, V, q, k, v, d, beta, s.data(), nullptr,
             r.data(), u.data(), partial.data());
        std::copy(s.begin(), s.end(), states.data() + (t - lo) * HKV);
      }
      for (int t = end - 1; t >= lo; --t) {
        for (int h = 0; h < H; ++h) {
          const int th = t * H + h;
          const float *qt = q + th * K, *kt = k + th * K;
          const float *dt = d + th * K, *vt = v + th * V, *got = go + th * V;
          const float* st = states.data() + (t - lo) * HKV + h * KV;
          const float* prev = t == lo ? initial + h * KV : st - HKV;
          float* G = adj.data() + h * KV;
          std::fill(r.begin(), r.end(), 0.f);
          std::fill(gu.begin(), gu.end(), 0.f);
          for (int i = 0; i < K; ++i) {
            if (i % 16 == 0) std::fill(partial.begin(), partial.end(), 0.f);
            float dq = 0.f;
            for (int j = 0; j < V; ++j) {
              const int ij = i * V + j;
              A[ij] = dt[i] * prev[ij];
              partial[j] += kt[i] * A[ij];
              dq += st[ij] * got[j];
              G[ij] += qt[i] * got[j];
              gu[j] += kt[i] * G[ij];
            }
            gq[th * K + i] = dq;
            if (i % 16 == 15 || i + 1 == K)
              for (int j = 0; j < V; ++j) r[j] += partial[j];
          }
          float db = 0.f;
          for (int j = 0; j < V; ++j) {
            u[j] = beta[th] * (vt[j] - r[j]);
            gv[th * V + j] = beta[th] * gu[j];
            db += gu[j] * (vt[j] - r[j]);
          }
          gbeta[th] = db;
          for (int i = 0; i < K; ++i) {
            float dk = 0.f, dd = 0.f;
            for (int j = 0; j < V; ++j) {
              const int ij = i * V + j;
              const float gr = -beta[th] * gu[j];
              dk += G[ij] * u[j] + A[ij] * gr;
              const float ga = G[ij] + kt[i] * gr;
              dd += ga * prev[ij];
              G[ij] = dt[i] * ga;
            }
            gk[th * K + i] = dk;
            gd[th * K + i] = dd;
          }
        }
      }
      end = lo;
    }
  }
}
