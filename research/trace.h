// SPDX-License-Identifier: GPL-3.0-only
// Native little-endian x86 records, one per coded bit, before model update.
#ifndef HUTTER_RESEARCH_TRACE_H
#define HUTTER_RESEARCH_TRACE_H
#include <cstdint>
#if defined(HUTTER_PPMD_TRACE) && !defined(HUTTER_TRACE)
#error "HUTTER_PPMD_TRACE requires HUTTER_TRACE"
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
#endif
