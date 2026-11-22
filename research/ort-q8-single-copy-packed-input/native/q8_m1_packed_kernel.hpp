// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Blucca
#ifndef BLUCCA_Q8_M1_PACKED_KERNEL_HPP
#define BLUCCA_Q8_M1_PACKED_KERNEL_HPP

#include <cstddef>
#include <cstdint>

#include <immintrin.h>

namespace blucca {

// AVX2 + FMA, one FP32 activation row, borrowed symmetric Q8 weights.
// P: int8 [N/16, K/32, 32, 16]; PS: float32 [N/16, K/32, 16].
// The caller checks K/N, shapes and scales. This view belongs to one call;
// it owns no arrays and stores no pointers beyond that call's lifetime.
// Build with -mavx2 -mfma; loads retain the old core's unaligned contract.
class Q8M1PackedView {
 public:
  Q8M1PackedView(std::size_t blocks, std::size_t tiles,
                 const std::int8_t* P, const float* PS) noexcept
      : blocks_(blocks), tiles_(tiles), packed_(P), scales_(PS) {}

  // A: K FP32 values; Y: N FP32 values in distinct caller-owned storage.
  // Weights/scales are immutable, and all accumulators are call-local.
  void Run(const float* A, float* Y) const {
    for (std::size_t tile = 0; tile < tiles_; ++tile) {
      __m256 total_lo = _mm256_setzero_ps();
      __m256 total_hi = _mm256_setzero_ps();
      for (std::size_t block = 0; block < blocks_; ++block) {
        const std::size_t group = tile * blocks_ + block;
        const std::int8_t* weight = packed_ + group * 32 * 16;
        const float* a = A + block * 32;
        __m256 lo0 = _mm256_setzero_ps();
        __m256 lo1 = _mm256_setzero_ps();
        __m256 lo2 = _mm256_setzero_ps();
        __m256 lo3 = _mm256_setzero_ps();
        __m256 hi0 = _mm256_setzero_ps();
        __m256 hi1 = _mm256_setzero_ps();
        __m256 hi2 = _mm256_setzero_ps();
        __m256 hi3 = _mm256_setzero_ps();

        // Four independent chains per eight-lane half, eight terms/chain.
        for (std::size_t inner = 0; inner < 32; inner += 4) {
          FmaPair(weight + (inner + 0) * 16, a[inner + 0], lo0, hi0);
          FmaPair(weight + (inner + 1) * 16, a[inner + 1], lo1, hi1);
          FmaPair(weight + (inner + 2) * 16, a[inner + 2], lo2, hi2);
          FmaPair(weight + (inner + 3) * 16, a[inner + 3], lo3, hi3);
        }

        const __m256 sum_lo = _mm256_add_ps(_mm256_add_ps(lo0, lo1),
                                          _mm256_add_ps(lo2, lo3));
        const __m256 sum_hi = _mm256_add_ps(_mm256_add_ps(hi0, hi1),
                                          _mm256_add_ps(hi2, hi3));
        const float* scale = scales_ + group * 16;
        total_lo = _mm256_fmadd_ps(sum_lo, _mm256_loadu_ps(scale), total_lo);
        total_hi = _mm256_fmadd_ps(sum_hi, _mm256_loadu_ps(scale + 8), total_hi);
      }
      _mm256_storeu_ps(Y + tile * 16, total_lo);
      _mm256_storeu_ps(Y + tile * 16 + 8, total_hi);
    }
  }

 private:
  static inline void FmaPair(const std::int8_t* weight, float activation,
                            __m256& lo, __m256& hi) {
    const __m256 a = _mm256_set1_ps(activation);
    const __m256 w_lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(weight))));
    const __m256 w_hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(weight + 8))));
    lo = _mm256_fmadd_ps(a, w_lo, lo);
    hi = _mm256_fmadd_ps(a, w_hi, hi);
  }

  std::size_t blocks_;
  std::size_t tiles_;
  const std::int8_t* packed_;
  const float* scales_;
};

}  // namespace blucca

#endif  // BLUCCA_Q8_M1_PACKED_KERNEL_HPP
