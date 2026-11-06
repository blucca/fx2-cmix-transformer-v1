// Two fixed native families. Compile every shape-dependent object with the
// same FX2_MODEL_LAYERS definition; the original twelve-block family is default.
#pragma once

#include <cstdint>

#ifndef FX2_MODEL_LAYERS
#define FX2_MODEL_LAYERS 12
#endif

namespace fx2 {
namespace shape {

constexpr int V = 205, D = 192, DH = 64, NH = 3, DMLP = 768;
constexpr int WIN = 1024, ROPE_LEN = 131072;

#if FX2_MODEL_LAYERS == 8
constexpr int NL = 8, NK = 6, NV = 2, NSKIP = 4;
constexpr bool KIMI[NL] = {true, true, true, false, true, true, true, false};
#elif FX2_MODEL_LAYERS == 12
constexpr int NL = 12, NK = 9, NV = 3, NSKIP = 6;
constexpr bool KIMI[NL] = {true, true, true, false, true, true,
                           true, false, true, true, true, false};
#else
#error "FX2_MODEL_LAYERS must be 8 or 12"
#endif

constexpr int32_t CONFIG_INTS[11] = {V, D, NL, DH, NH, DMLP,
                                    WIN, DH, NH, 4, 10000};

}  // namespace shape
}  // namespace fx2
