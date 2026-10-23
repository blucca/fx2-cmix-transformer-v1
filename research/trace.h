// SPDX-License-Identifier: GPL-3.0-only
// Native little-endian x86 records, one per coded bit, before model update.
#ifndef HUTTER_RESEARCH_TRACE_H
#define HUTTER_RESEARCH_TRACE_H
#include <cstdint>
struct HutterTraceRecord {
  std::uint16_t probability;  // Exact arithmetic-coder P(bit=1), scale 65536.
  std::uint16_t flags;        // bit 0: observed bit; bit 1: vocabulary override.
  float neural;              // Current byte-mixer bit probability.
  float fxcm;                // Current FXCM skip probability; may be padding.
  float mixer;               // Final mixer logit, before sigmoid and SSE.
};
static_assert(sizeof(HutterTraceRecord) == 16, "trace layout changed");
#endif
