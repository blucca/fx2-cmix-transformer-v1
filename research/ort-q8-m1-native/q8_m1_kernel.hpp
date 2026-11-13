// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Blucca
#ifndef BLUCCA_Q8_M1_KERNEL_HPP
#define BLUCCA_Q8_M1_KERNEL_HPP

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <immintrin.h>

namespace blucca {

// AVX2 + FMA, one FP32 activation row, constant symmetric Q8 weights.
// B: uint8 [N, K/32, 32], zero point 128; S: float32 [N*(K/32)].
// K is a positive multiple of 32; N is a positive multiple of 16.
// Build with -mavx2 -mfma (GCC/Clang) or /arch:AVX2 (MSVC).
class Q8M1 {
 public:
  const std::int64_t k;
  const std::int64_t n;

  Q8M1(std::int64_t k_value, std::int64_t n_value,
       const std::uint8_t* B, const float* S)
      : k(k_value), n(n_value) {
    if (k <= 0 || k % 32 != 0 || n <= 0 || n % 16 != 0 || !B || !S) {
      throw std::invalid_argument(
          "Q8M1 requires K>0 divisible by 32, N>0 divisible by 16, and B/S");
    }
    blocks_ = static_cast<std::size_t>(k) / 32;
    tiles_ = static_cast<std::size_t>(n) / 16;
    packed_.resize(static_cast<std::size_t>(k) * static_cast<std::size_t>(n));
    scales_.resize(static_cast<std::size_t>(n) * blocks_);

    // Tile -> K block -> K inner -> 16 output lanes. Conversion/transpose
    // happens once here; Run never reads or unpacks the original B or S.
    for (std::size_t tile = 0; tile < tiles_; ++tile) {
      for (std::size_t block = 0; block < blocks_; ++block) {
        const std::size_t group = tile * blocks_ + block;
        std::int8_t* dst = packed_.data() + group * 32 * 16;
        float* scale = scales_.data() + group * 16;
        for (std::size_t lane = 0; lane < 16; ++lane) {
          const std::size_t column = tile * 16 + lane;
          const std::size_t source = (column * blocks_ + block) * 32;
          scale[lane] = S[column * blocks_ + block];
          for (std::size_t inner = 0; inner < 32; ++inner) {
            dst[inner * 16 + lane] = static_cast<std::int8_t>(
                static_cast<int>(B[source + inner]) - 128);
          }
        }
      }
    }
  }

  // A: K FP32 values; Y: N FP32 values in distinct caller-owned storage.
  // Weights/scales are immutable, and all accumulators are call-local.
  void Run(const float* A, float* Y) const {
    for (std::size_t tile = 0; tile < tiles_; ++tile) {
      __m256 total_lo = _mm256_setzero_ps();
      __m256 total_hi = _mm256_setzero_ps();
      for (std::size_t block = 0; block < blocks_; ++block) {
        const std::size_t group = tile * blocks_ + block;
        const std::int8_t* weight = packed_.data() + group * 32 * 16;
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
        const float* scale = scales_.data() + group * 16;
        total_lo = _mm256_fmadd_ps(sum_lo, _mm256_loadu_ps(scale), total_lo);
        total_hi = _mm256_fmadd_ps(sum_hi, _mm256_loadu_ps(scale + 8), total_hi);
      }
      _mm256_storeu_ps(Y + tile * 16, total_lo);
      _mm256_storeu_ps(Y + tile * 16 + 8, total_hi);
    }
  }

  // Native packed array payload only (signed codes + reordered FP32 scales).
  // This excludes vector/object metadata and the source ORT initializers.
  std::size_t PackedBytes() const {
    return packed_.size() * sizeof(std::int8_t) + scales_.size() * sizeof(float);
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

  std::size_t blocks_ = 0;
  std::size_t tiles_ = 0;
  std::vector<std::int8_t> packed_;
  std::vector<float> scales_;
};

}  // namespace blucca

#endif  // BLUCCA_Q8_M1_KERNEL_HPP
