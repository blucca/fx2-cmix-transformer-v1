// SPDX-License-Identifier: GPL-3.0-only
// Native little-endian x86 records, one per coded bit, before model update.
#ifndef HUTTER_RESEARCH_TRACE_H
#define HUTTER_RESEARCH_TRACE_H
#include <cstdint>
#ifdef HUTTER_MIXER_TRACE
#include <cstring>
#endif
#if defined(HUTTER_PPMD_TRACE) && !defined(HUTTER_TRACE)
#error "HUTTER_PPMD_TRACE requires HUTTER_TRACE"
#endif
#if defined(HUTTER_MIXER_TRACE) && !defined(HUTTER_TRACE)
#error "HUTTER_MIXER_TRACE requires HUTTER_TRACE"
#endif
struct HutterTraceRecord {
  std::uint16_t probability;  // Exact arithmetic-coder P(bit=1), scale 65536.
  std::uint16_t flags;        // bit 0: observed bit; bit 1: vocabulary override.
  float neural;              // Current byte-mixer bit probability.
  float fxcm;                // Current FXCM skip probability; may be padding.
  float mixer;               // Final mixer logit, before sigmoid and SSE.
};
static_assert(sizeof(HutterTraceRecord) == 16, "trace layout changed");

// Independent HUTTER_PPMD_TRACE_PATH sidecar, aligned with HutterTraceRecord.
// Byte-level fields describe the prior available before this byte's first bit
// and repeat unchanged across its eight MSB-first bit records.
struct HutterPpmdTraceRecord {
  float ppmd;                 // Raw P(bit=1), before mixer/coder clamping.
  float entropy;              // Normalized byte entropy in bits (vocab when ready).
  float escape;               // First SQ escape freq/total, before vocab masking.
  std::uint16_t suffix_depth; // MaxContext's suffix-link count to the root.
  std::uint16_t flags;        // bit 0: a real byte has prepared this prior.
};
// Initial byte: ppmd=0.5, entropy=8, escape=-1, depth=flags=0. ByteModel's
// initial prior covers all 256 bytes; ByteUpdate applies the vocabulary mask.
static_assert(sizeof(HutterPpmdTraceRecord) == 16, "PPMd trace layout changed");

// Independent HUTTER_MIXER_TRACE_PATH sidecar, aligned with HutterTraceRecord.
// Exactly the 23 clamped logits consumed by the second-layer mixer, encoded
// as IEEE binary16, round-to-nearest ties-to-even (including subnormals).
// support[i]: bits 0..5 = floor(log2(1 + saturated uint32 applied updates)),
// range 0..32; bit 6 is zero; bit 7 marks the shared fallback ContextData.
// Counts and fallback refer to the selection made by this bit's Mix(), before
// its label. flags bit 0 = a complete 23-mixer row, all other bits zero.
// reserved bytes are zero. This sidecar contains no current label.
struct HutterMixerTraceRecord {
  std::uint16_t logits[23];
  std::uint8_t support[23];
  std::uint8_t flags;
  std::uint8_t reserved[2];
};
static_assert(sizeof(HutterMixerTraceRecord) == 72, "mixer trace layout changed");

#ifdef HUTTER_MIXER_TRACE
inline std::uint8_t HutterMixerSupport(std::uint32_t updates, bool fallback) {
  const unsigned bucket = 63 - __builtin_clzll(std::uint64_t(updates) + 1);
  return static_cast<std::uint8_t>(bucket | (fallback ? 128 : 0));
}

inline void HutterMixerAppliedUpdate(std::uint32_t& updates) {
  if (updates != UINT32_MAX) ++updates;
}

// Same IEEE rounding as predictor.cpp's FloatToHalf, expressed with integers
// so observation does not change floating-point state or depend on F16C.
inline std::uint16_t HutterMixerLogitHalf(float value) {
  std::uint32_t x;
  std::memcpy(&x, &value, sizeof(x));
  const std::uint32_t sign = (x >> 16) & 0x8000;
  const int exponent = int((x >> 23) & 255) - 112;
  std::uint32_t mantissa = x & 0x7fffff;
  if (exponent >= 31) return sign | 0x7c00;  // logits are finite and clamped
  if (exponent <= 0) {
    if (exponent < -10) return sign;
    mantissa |= 0x800000;
    const int shift = 14 - exponent;
    std::uint16_t half = mantissa >> shift;
    const std::uint32_t rem = mantissa & ((1u << shift) - 1);
    const std::uint32_t tie = 1u << (shift - 1);
    if (rem > tie || (rem == tie && (half & 1))) ++half;
    return sign | half;
  }
  std::uint16_t half = sign | (exponent << 10) | (mantissa >> 13);
  const std::uint32_t rem = mantissa & 0x1fff;
  if (rem > 0x1000 || (rem == 0x1000 && (half & 1))) ++half;
  return half;
}
#endif
#endif
