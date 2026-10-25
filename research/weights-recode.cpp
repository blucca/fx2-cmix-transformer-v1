// Convert the original v2 file to static-per-tensor INT4 v3, preserving order.
// Reuse the existing decoder and model definitions in this translation unit.
// Link weights_io.cpp only; weights_io_compressed.cpp is included below.
#include "../cpp_infer/src/weights_io_compressed.cpp"
#include "weights-static15.h"

#include <fstream>
#include <iostream>
#include <iterator>

struct Encoder {
  uint64_t low = 0;
  uint32_t range = 0xffffffffu;
  uint8_t cache = 0;
  size_t cache_size = 1;
  std::vector<uint8_t> out;
  void shift_low() {
    if (uint32_t(low) < 0xff000000u || (low >> 32)) {
      const uint8_t carry = low >> 32;
      out.push_back(cache + carry);
      while (--cache_size) out.push_back(uint8_t(0xff + carry));
      cache = uint32_t(low) >> 24;
    }
    ++cache_size;
    low = uint32_t(low) << 8;
  }
  void normalize() {
    while (range < (1u << 24)) { range <<= 8; shift_low(); }
  }
  void bit(uint16_t& prob, unsigned b) {
    const uint32_t bound = (range >> 11) * prob;
    if (b) { low += bound; range -= bound; prob -= prob >> 5; }
    else { range = bound; prob += (2048 - prob) >> 5; }
    normalize();
  }
  void tree(uint16_t* probs, unsigned bits, uint32_t symbol) {
    unsigned node = 1;
    for (int k = int(bits) - 1; k >= 0; --k) {
      const auto b = (symbol >> k) & 1;
      bit(probs[node], b);
      node = node * 2 + b;
    }
  }
  void static15(uint32_t cum, uint32_t freq) {
    const uint32_t unit = range >> 12;
    low += unit * cum;
    range = unit * freq;
    normalize();
  }
};

int main(int argc, char** argv) {
  if (argc != 3) { std::cerr << "usage: weights-recode INPUT.tfwc2 OUTPUT.tfwc3\n"; return 2; }
  if (std::string(argv[1]) == argv[2]) { std::cerr << "use a separate output file\n"; return 2; }
  std::ifstream in(argv[1], std::ios::binary);
  if (!in) { std::cerr << "cannot open input\n"; return 1; }
  const std::vector<uint8_t> data((std::istreambuf_iterator<char>(in)), {});
  if (data.size() < 12 || std::memcmp(data.data(), "FX2TFWC2", 8)) {
    std::cerr << "input must be FX2TFWC2\n"; return 1;
  }
  fx2::Reader r{data.data() + 8, data.size() - 8, argv[1]};
  const auto nt = r.u32();
  fx2::BinDecoder dec(r.p, r.left);
  fx2::ModelsV2 old, next;
  Encoder enc;
  auto get_meta = [&]() {
    const auto b = dec.decode_tree(&old.meta[size_t(old.meta_prev) * 256], 8);
    old.meta_prev = b; return b;
  };
  auto put_meta = [&](uint32_t b) {
    enc.tree(&next.meta[size_t(next.meta_prev) * 256], 8, b);
    next.meta_prev = b;
  };
  auto copy_meta = [&]() { const auto b = get_meta(); put_meta(b); return b; };
  auto copy_tree = [&](uint16_t* a, uint16_t* b, unsigned bits) {
    const auto v = dec.decode_tree(a, bits); enc.tree(b, bits, v); return v;
  };
  size_t int4_tensors = 0, int4_values = 0;
  for (uint32_t i = 0; i < nt; ++i) {
    const auto name_len = copy_meta();
    uint32_t c2 = 0, c1 = 0;
    for (uint32_t k = 0; k < name_len; ++k) {
      const auto context = size_t((c2 << 8) | c1) * 256;
      const auto ch = copy_tree(&old.name[context], &next.name[context], 8);
      c2 = c1; c1 = ch;
    }
    const auto dtype = copy_meta(), ndim = copy_meta();
    assert(ndim <= 8);
    size_t numel = 1;
    for (uint32_t d = 0; d < ndim; ++d) {
      uint32_t v = 0;
      for (int k = 0; k < 4; ++k) v |= copy_meta() << (8 * k);
      numel *= v;
    }
    const auto encoding = get_meta();
    put_meta(encoding == fx2::ENC_INT4 ? fx2::ENC_INT4_STATIC15 : encoding);
    const auto bytes = numel * fx2::dtype_size(dtype);
    if (encoding == fx2::ENC_INT4) {
      assert(dtype == fx2::DT_I8 && numel > 0);
      Counts counts{};
      std::vector<uint8_t> symbols(numel);
      for (auto& s : symbols) { s = dec.decode_tree(old.int4.data(), 4); assert(s < 15); ++counts[s]; }
      const auto freq = static15_frequencies(counts);
      Counts cum{};
      for (int s = 1; s < 15; ++s) cum[s] = cum[s - 1] + freq[s - 1];
      for (int s = 0; s < 14; ++s) { put_meta(freq[s] & 255); put_meta(freq[s] >> 8); }
      for (auto s : symbols) enc.static15(cum[s], freq[s]);
      ++int4_tensors; int4_values += numel;
    } else if (encoding == fx2::ENC_BF16) {
      for (size_t k = 0; k < numel; ++k) {
        const auto hi = copy_tree(old.bf16_hi.data(), next.bf16_hi.data(), 8);
        copy_tree(&old.bf16_lo[hi * 256], &next.bf16_lo[hi * 256], 8);
      }
    } else if (encoding == fx2::ENC_PLANE4) {
      for (size_t k = 0; k < bytes; ++k)
        copy_tree(&old.plane[(k & 3) * 256], &next.plane[(k & 3) * 256], 8);
    } else if (encoding == fx2::ENC_RAW) {
      for (size_t k = 0; k < bytes; ++k) copy_tree(old.raw.data(), next.raw.data(), 8);
    } else if (encoding != fx2::ENC_ROPE_SIN && encoding != fx2::ENC_ROPE_COS) {
      std::cerr << "unknown input encoding\n"; return 1;
    }
  }
  for (int k = 0; k < 5; ++k) enc.shift_low();
  std::ofstream out(argv[2], std::ios::binary);
  out.write("FX2TFWC3", 8);
  out.write(reinterpret_cast<const char*>(data.data() + 8), 4);
  out.write(reinterpret_cast<const char*>(enc.out.data()), enc.out.size());
  out.close();
  if (!out) { std::cerr << "output write failed\n"; return 1; }
  const auto before = fx2::WeightsFile::load_compressed(argv[1]);
  const auto after = fx2::WeightsFile::load_compressed(argv[2]);
  assert(before.tensors.size() == after.tensors.size());
  for (const auto& [name, tensor] : before.tensors) {
    const auto& other = after.get(name);
    assert(tensor.dtype == other.dtype && tensor.shape == other.shape && tensor.data == other.data);
  }
  std::cout << "{\"status\":\"passed\",\"tensor_count\":" << nt
            << ",\"int4_tensors\":" << int4_tensors << ",\"int4_values\":" << int4_values
            << ",\"input_bytes\":" << data.size() << ",\"output_bytes\":" << enc.out.size() + 12
            << ",\"saved_bytes\":" << int64_t(data.size()) - int64_t(enc.out.size()) - 12
            << ",\"all_tensor_bytes_equal\":true}\n";
}
