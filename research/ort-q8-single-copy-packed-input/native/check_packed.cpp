// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Blucca
// Small standalone check: directly compare the new borrowed view with the
// unchanged published owner core, using independently indexed packed inputs.
#include "q8_m1_kernel.hpp"
#include "q8_m1_packed_kernel.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

struct Case {
  std::int64_t k;
  std::int64_t n;
  std::int64_t logical_n;
};

void CheckCase(const Case& shape, std::array<bool, 256>& seen_codes) {
  const auto k = static_cast<std::size_t>(shape.k);
  const auto n = static_cast<std::size_t>(shape.n);
  const auto blocks = k / 32;
  const auto tiles = n / 16;
  std::vector<std::uint8_t> B(k * n);
  std::vector<float> S(n * blocks);
  // Deliberate offsets exercise the inherited unaligned-load contract.
  std::vector<std::int8_t> p_storage(k * n + 1);
  std::vector<float> ps_storage(n * blocks + 1);
  auto* P = p_storage.data() + 1;
  auto* PS = ps_storage.data() + 1;

  // Column-first independent equation, not the old owner's packing loops.
  for (std::size_t column = 0; column < n; ++column) {
    const auto tile = column / 16;
    const auto lane = column % 16;
    for (std::size_t block = 0; block < blocks; ++block) {
      const auto s_source = column * blocks + block;
      const auto s_dest = (tile * blocks + block) * 16 + lane;
      // Exactly representable, positive and different for every tile/block/lane.
      S[s_source] = static_cast<float>(s_dest + 1) * (1.0f / 256.0f);
      PS[s_dest] = S[s_source];
      for (std::size_t inner = 0; inner < 32; ++inner) {
        const auto source = (column * blocks + block) * 32 + inner;
        const auto dest = ((tile * blocks + block) * 32 + inner) * 16 + lane;
        const auto code = column < static_cast<std::size_t>(shape.logical_n)
                              ? static_cast<std::uint8_t>((column * 32 + block * 71 + inner) % 256)
                              : static_cast<std::uint8_t>(128);
        B[source] = code;
        P[dest] = static_cast<std::int8_t>(static_cast<int>(code) - 128);
        seen_codes[static_cast<std::size_t>(static_cast<int>(P[dest]) + 128)] = true;
      }
    }
  }

  const blucca::Q8M1 owner(shape.k, shape.n, B.data(), S.data());
  const auto expected_owner_bytes = k * n + n * blocks * sizeof(float);
  if (owner.PackedBytes() != expected_owner_bytes) {
    throw std::runtime_error("old owner payload check failed");
  }
  std::vector<float> a_storage(k + 1), old_storage(n + 1), new_storage(n + 1);
  float* A = a_storage.data() + 1;
  float* old_y = old_storage.data() + 1;
  float* new_y = new_storage.data() + 1;
  for (std::size_t activation_case = 0; activation_case < 3; ++activation_case) {
    for (std::size_t i = 0; i < k; ++i) {
      const auto integer = static_cast<int>((i * 37 + activation_case * 17) % 113) - 56;
      A[i] = static_cast<float>(integer) * (1.0f / 64.0f);
    }
    owner.Run(A, old_y);
    // This invocation alone owns the view; it borrows the current P/PS.
    const blucca::Q8M1PackedView view(blocks, tiles, P, PS);
    view.Run(A, new_y);
    if (std::memcmp(old_y, new_y, n * sizeof(float)) != 0) {
      throw std::runtime_error("new view differs bitwise from published Q8M1");
    }
    for (std::size_t column = static_cast<std::size_t>(shape.logical_n); column < n; ++column) {
      if (new_y[column] != 0.0f) {
        throw std::runtime_error("padded lane produced nonzero output");
      }
    }
  }
}

}  // namespace

int main() {
  try {
    const std::array<Case, 4> cases{{{32, 16, 16}, {96, 48, 48},
                                    {128, 64, 64}, {64, 48, 33}}};
    std::array<bool, 256> seen_codes{};
    for (const auto& shape : cases) {
      CheckCase(shape, seen_codes);
    }
    for (const bool seen : seen_codes) {
      if (!seen) {
        throw std::runtime_error("signed-code coverage omitted a code");
      }
    }
    std::cout << "{\"status\":\"passed\",\"oracle\":\"unchanged published Q8M1 owner, bitwise\","
                 "\"signed_codes_covered\":256,\"shapes\":[[32,16],[96,48],[128,64],[64,48]],"
                 "\"activation_cases_per_shape\":3,\"bitwise_runs\":12,"
                 "\"distinct_tile_block_lane_scales\":true,\"padding_lanes\":15,"
                 "\"offset_unaligned_storage\":true,\"new_native_owned_packed_bytes\":0}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
