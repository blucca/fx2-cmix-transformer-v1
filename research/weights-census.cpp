// Read-only INT4 census. Build with the existing weights_io*.cpp loaders.
// Oracle histograms exclude description cost; causal models start cold.
#include "../cpp_infer/src/weights_io.h"
#include "weights-static15.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

double entropy_bits(const Counts& c) {
  const double n = std::accumulate(c.begin(), c.end(), uint64_t{0});
  double bits = 0;
  for (auto v : c) if (v) bits += v * std::log2(n / v);
  return bits;
}

double kt_probability(const Counts& c, int s) {
  const auto n = std::accumulate(c.begin(), c.end(), uint64_t{0});
  return (c[s] + 0.5) / (n + 7.5);
}

double static_bits(const Counts& c) {
  // A concrete two-part model: 14 uint16 frequencies, final one implied,
  // total 4096. Every symbol gets at least one count. Largest remainders
  // allocate the rest, with symbol order breaking ties.
  const auto freq = static15_frequencies(c);
  double bits = 14 * 16;
  for (int s = 0; s < 15; ++s) if (c[s]) bits += c[s] * std::log2(4096.0 / freq[s]);
  return bits;
}

struct Tree5 {
  std::array<uint16_t, 16> probs;
  Tree5() { probs.fill(1024); }
  double observe(int symbol) {
    int node = 1;
    double bits = 0;
    for (int k = 3; k >= 0; --k) {
      const int bit = (symbol >> k) & 1;
      auto& p = probs[node];
      bits -= std::log2((bit ? 2048 - p : p) / 2048.0);
      p = bit ? p - (p >> 5) : p + ((2048 - p) >> 5);
      node = node * 2 + bit;
    }
    return bits;
  }
};

int main(int argc, char** argv) {
  Counts check{};
  check[0] = check[1] = 2;
  assert(std::abs(entropy_bits(check) - 4.0) < 1e-12);
  assert(std::abs(kt_probability(Counts{}, 7) - 1.0 / 15) < 1e-12);
  assert(static_bits(check) >= entropy_bits(check) + 224);
  assert(std::abs(Tree5{}.observe(0) - 4.0) < 1e-12);
  if (argc != 2) {
    std::cerr << "usage: weights-census INPUT.tfwc2\n";
    return 2;
  }
  const auto wf = fx2::WeightsFile::load_compressed(argv[1]);
  std::vector<std::string> names;
  for (const auto& entry : wf.tensors) names.push_back(entry.first);
  std::sort(names.begin(), names.end());
  Counts global{};
  std::array<uint64_t, 4> elements{}, bytes{}, tensor_counts{};
  std::cout << std::setprecision(12) << "{\"tensors\":[";
  bool first = true;
  for (const auto& name : names) {
    const auto& t = wf.get(name);
    assert(t.dtype < 4);
    elements[t.dtype] += t.numel;
    bytes[t.dtype] += t.data.size();
    ++tensor_counts[t.dtype];
    if (!first) std::cout << ',';
    first = false;
    std::cout << "\n{\"name\":" << std::quoted(name) << ",\"dtype\":" << int(t.dtype)
              << ",\"elements\":" << t.numel << ",\"shape\":[";
    for (size_t j = 0; j < t.shape.size(); ++j) std::cout << (j ? "," : "") << t.shape[j];
    std::cout << ']';
    if (t.dtype == fx2::DT_I8) {
      assert(t.shape.size() == 2 && t.numel > 0);
      const size_t rows = t.shape[0], cols = t.shape[1];
      Counts total{};
      std::vector<Counts> row(rows), col(cols);
      double kt = 0, row64 = 0, col64 = 0, tree5 = 0;
      Tree5 tree;
      for (size_t k = 0; k < t.numel; ++k) {
        const int s = t.i8()[k] + 7;
        assert(s >= 0 && s < 15);
        const size_t r = k / cols, c = k % cols;
        const double p = kt_probability(total, s);
        kt -= std::log2(p);
        // Hierarchical causal estimates; the parent includes previous weights
        // in this row. Known matrix dimensions determine the context for free.
        row64 -= std::log2((row[r][s] + 64 * p) / (c + 64));
        col64 -= std::log2((col[c][s] + 64 * p) / (r + 64));
        tree5 += tree.observe(s);
        ++total[s]; ++row[r][s]; ++col[c][s]; ++global[s];
      }
      double row_h = 0, col_h = 0;
      for (const auto& c : row) row_h += entropy_bits(c);
      for (const auto& c : col) col_h += entropy_bits(c);
      std::cout << ",\"histogram\":[";
      for (int s = 0; s < 15; ++s) std::cout << (s ? "," : "") << total[s];
      std::cout << "],\"oracle_tensor_bits\":" << entropy_bits(total)
                << ",\"oracle_row_bits\":" << row_h
                << ",\"oracle_column_bits\":" << col_h
                << ",\"causal_tensor_kt_bits\":" << kt
                << ",\"causal_row64_bits\":" << row64
                << ",\"causal_column64_bits\":" << col64
                << ",\"reset_tensor_tree5_bits\":" << tree5
                << ",\"static4096_with_28B_table_bits\":" << static_bits(total);
    }
    std::cout << '}';
  }
  std::cout << "\n],\"dtype_totals\":[";
  for (int d = 0; d < 4; ++d)
    std::cout << (d ? "," : "") << "{\"dtype\":" << d
              << ",\"tensors\":" << tensor_counts[d] << ",\"elements\":" << elements[d]
              << ",\"bytes\":" << bytes[d] << '}';
  std::cout << "],\"oracle_global_int4_bits\":" << entropy_bits(global)
            << ",\"note\":\"tree5 is reset per tensor, not original stream attribution; "
               "causal costs include cold start but exclude range rounding/framing; "
               "static4096 includes 28 table bytes per tensor\"}\n";
}
