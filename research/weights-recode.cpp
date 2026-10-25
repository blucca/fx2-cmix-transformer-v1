// Convert v2 to static-per-tensor INT4 v3, preserving order and (by default) values.
// Explicit quant7/replacement modes are QAT experiments; defaults stay lossless.
// Reuse the existing decoder and model definitions in this translation unit.
// Link weights_io.cpp only; weights_io_compressed.cpp is included below.
#include "../cpp_infer/src/weights_io_compressed.cpp"
#include "weights-static15.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <unordered_set>

int8_t quant7_value(int q) {
  assert(q >= -7 && q <= 7);
  const int magnitude = (3 * std::abs(q) + 3) / 7;
  return int8_t(q < 0 ? -magnitude : magnitude);
}

uint16_t quant7_scale(uint16_t bits) {
  // ponytail: this experiment accepts the model's finite, nonzero normal BF16 scales.
  // Multiply the exact significand by 7/3 and round ties to even in integers.
  unsigned exponent = (bits >> 7) & 255;
  assert(exponent > 0 && exponent < 255);
  const unsigned numerator = 7 * (128 + (bits & 127));
  unsigned denominator = 6;
  ++exponent;
  if (numerator >= 256 * denominator) { denominator = 12; ++exponent; }
  unsigned q = numerator / denominator, remainder = numerator % denominator;
  q += 2 * remainder > denominator || (2 * remainder == denominator && (q & 1));
  if (q == 256) { q = 128; ++exponent; }
  assert(exponent < 255);
  return uint16_t((bits & 0x8000) | (exponent << 7) | (q - 128));
}

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
  const bool protect39 = argc == 4 && std::string(argv[3]) == "--quant7-protect39";
  const bool quant7 = protect39 || (argc == 4 && std::string(argv[3]) == "--quant7");
  const bool replace_payloads = argc == 5 && std::string(argv[3]) == "--replace-payloads";
  if (argc != 3 && !quant7 && !replace_payloads) {
    std::cerr << "usage: weights-recode INPUT.tfwc2 OUTPUT.tfwc3 "
                 "[--quant7|--quant7-protect39|--replace-payloads RAW.bin]\n"; return 2;
  }
  if (std::string(argv[1]) == argv[2] ||
      (replace_payloads && std::string(argv[4]) == argv[2])) {
    std::cerr << "use a separate output file\n"; return 2;
  }
  std::ifstream in(argv[1], std::ios::binary);
  if (!in) { std::cerr << "cannot open input\n"; return 1; }
  const std::vector<uint8_t> data((std::istreambuf_iterator<char>(in)), {});
  if (data.size() < 12 || std::memcmp(data.data(), "FX2TFWC2", 8)) {
    std::cerr << "input must be FX2TFWC2\n"; return 1;
  }
  const auto before = fx2::WeightsFile::load_compressed(argv[1]);
  fx2::WeightsFile replacement;
  if (replace_payloads) {
    replacement = fx2::WeightsFile::load(argv[4]);
    const std::unordered_set<std::string> derived = {
      "rope.inv_freq", "rope.sin", "rope.cos", "config.ints", "config.kimi"};
    if (before.tensors.size() != 434 || replacement.tensors.size() != 429)
      fx2::die("replacement needs the fixed 434-tensor model and 429 payloads");
    for (const auto& name : derived) {
      if (!before.has(name) || replacement.has(name))
        fx2::die("retain derived tensor %s from the template", name.c_str());
    }
    for (const auto& [name, tensor] : replacement.tensors) {
      const auto& original = before.get(name);
      if (tensor.dtype != original.dtype || tensor.shape != original.shape)
        fx2::die("replacement dtype/shape mismatch: %s", name.c_str());
      for (size_t k = 0; k < tensor.numel; ++k) {
        if (tensor.dtype == fx2::DT_I8) {
          if (tensor.i8()[k] < -7 || tensor.i8()[k] > 7)
            fx2::die("replacement INT4 out of range: %s", name.c_str());
        } else if (tensor.dtype == fx2::DT_BF16) {
          const float value = fx2::bf16_to_f32(tensor.bf16_bits()[k]);
          const bool weight_scale = name.size() >= 13 && name.substr(name.size() - 13) == ".weight.scale";
          if (!std::isfinite(value) || value == 0 || (!weight_scale && value < 0))
            fx2::die("replacement scale must be finite and %s: %s",
                     weight_scale ? "nonzero" : "positive", name.c_str());
        } else if (tensor.dtype != fx2::DT_F32 || !std::isfinite(tensor.f32()[k])) {
          fx2::die("replacement must be finite FP32: %s", name.c_str());
        }
      }
    }
  }
  std::unordered_set<std::string> weight_names, scale_names;
  if (quant7) {
    const int expected[] = {-3,-3,-2,-2,-1,-1,0,0,0,1,1,2,2,3,3};
    for (int q = -7; q <= 7; ++q) assert(quant7_value(q) == expected[q + 7]);
    assert(quant7_scale(0x3f80) == 0x4015);  // 1 * 7/3
    assert(quant7_scale(0x3f81) == 0x4016);  // exact tie, round down to even
    assert(quant7_scale(0x3f87) == 0x401e);  // exact tie, round up to even
    assert(quant7_scale(0x3fdb) == 0x4080);  // rounded exponent boundary
    assert(quant7_scale(0xbf81) == 0xc016);  // retain learned negative scales
    for (const auto& [name, tensor] : before.tensors) {
      if (tensor.dtype != fx2::DT_I8) continue;
      assert(tensor.shape.size() == 2 && name.size() > 2 && name.substr(name.size() - 2) == ".q");
      // One frozen structural group: KDA recurrent gates and small I/O matrices.
      if (protect39 && (name == "embedding.weight.q" ||
          name == "prior_embedding.weight.q" || name == "unembedding.weight.q" ||
          name.find(".attention.forget_gate_projection.") != std::string::npos ||
          name.find(".attention.output_gate_projection.") != std::string::npos)) continue;
      weight_names.insert(name);
      const auto scale = name.substr(0, name.size() - 2) + ".scale";
      const auto& s = before.get(scale);
      assert(s.dtype == fx2::DT_BF16 && s.shape == std::vector<uint32_t>{tensor.shape[0]});
      const bool inserted = scale_names.insert(scale).second;
      assert(inserted);
    }
    assert(weight_names.size() == (protect39 ? 72 : 111));
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
  size_t int4_tensors = 0, int4_values = 0, scale_tensors = 0, scale_values = 0;
  for (uint32_t i = 0; i < nt; ++i) {
    const auto name_len = copy_meta();
    std::string name;
    uint32_t c2 = 0, c1 = 0;
    for (uint32_t k = 0; k < name_len; ++k) {
      const auto context = size_t((c2 << 8) | c1) * 256;
      const auto ch = copy_tree(&old.name[context], &next.name[context], 8);
      name.push_back(char(ch));
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
    const auto* payload = replacement.has(name) ? &replacement.get(name) : nullptr;
    if (encoding == fx2::ENC_INT4) {
      assert(dtype == fx2::DT_I8 && numel > 0);
      Counts counts{};
      std::vector<uint8_t> symbols(numel);
      for (size_t k = 0; k < numel; ++k) {
        auto& s = symbols[k];
        s = dec.decode_tree(old.int4.data(), 4); assert(s < 15);
        if (payload) s = uint8_t(payload->i8()[k] + 7);
        if (weight_names.count(name)) s = uint8_t(quant7_value(int(s) - 7) + 7);
        ++counts[s];
      }
      const auto freq = static15_frequencies(counts);
      Counts cum{};
      for (int s = 1; s < 15; ++s) cum[s] = cum[s - 1] + freq[s - 1];
      for (int s = 0; s < 14; ++s) { put_meta(freq[s] & 255); put_meta(freq[s] >> 8); }
      for (auto s : symbols) enc.static15(cum[s], freq[s]);
      ++int4_tensors; int4_values += numel;
    } else if (encoding == fx2::ENC_BF16) {
      const bool rescale = scale_names.count(name);
      for (size_t k = 0; k < numel; ++k) {
        const auto hi = dec.decode_tree(old.bf16_hi.data(), 8);
        const auto lo = dec.decode_tree(&old.bf16_lo[hi * 256], 8);
        const uint16_t value = payload ? payload->bf16_bits()[k] :
            rescale ? quant7_scale((hi << 8) | lo) : (hi << 8) | lo;
        enc.tree(next.bf16_hi.data(), 8, value >> 8);
        enc.tree(&next.bf16_lo[(value >> 8) * 256], 8, value & 255);
      }
      if (rescale) { ++scale_tensors; scale_values += numel; }
    } else if (encoding == fx2::ENC_PLANE4) {
      for (size_t k = 0; k < bytes; ++k) {
        const auto value = dec.decode_tree(&old.plane[(k & 3) * 256], 8);
        enc.tree(&next.plane[(k & 3) * 256], 8, payload ? payload->data[k] : value);
      }
    } else if (encoding == fx2::ENC_RAW) {
      for (size_t k = 0; k < bytes; ++k) {
        const auto value = dec.decode_tree(old.raw.data(), 8);
        enc.tree(next.raw.data(), 8, payload ? payload->data[k] : value);
      }
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
  const auto after = fx2::WeightsFile::load_compressed(argv[2]);
  assert(before.tensors.size() == after.tensors.size());
  bool all_bytes_equal = true;
  for (const auto& [name, tensor] : before.tensors) {
    const auto& other = after.get(name);
    assert(tensor.dtype == other.dtype && tensor.shape == other.shape);
    all_bytes_equal &= tensor.data == other.data;
    if (replacement.has(name)) {
      assert(replacement.get(name).data == other.data);
    } else if (weight_names.count(name)) {
      for (size_t k = 0; k < tensor.numel; ++k)
        assert(other.i8()[k] == quant7_value(tensor.i8()[k]));
    } else if (scale_names.count(name)) {
      for (size_t k = 0; k < tensor.numel; ++k)
        assert(other.bf16_bits()[k] == quant7_scale(tensor.bf16_bits()[k]));
    } else assert(tensor.data == other.data);
  }
  assert(scale_tensors == scale_names.size());
  std::cout << "{\"status\":\"passed\",\"tensor_count\":" << nt
            << ",\"mode\":\"" << (replace_payloads ? "replace-payloads" :
                                  protect39 ? "quant7-protect39" : quant7 ? "quant7" : "lossless") << "\""
            << ",\"replacement_tensors\":" << replacement.tensors.size()
            << ",\"int4_tensors\":" << int4_tensors << ",\"int4_values\":" << int4_values
            << ",\"quantized_tensors\":" << weight_names.size()
            << ",\"rescaled_tensors\":" << scale_tensors << ",\"rescaled_values\":" << scale_values
            << ",\"input_bytes\":" << data.size() << ",\"output_bytes\":" << enc.out.size() + 12
            << ",\"saved_bytes\":" << int64_t(data.size()) - int64_t(enc.out.size()) - 12
            << ",\"all_tensor_bytes_equal\":" << (all_bytes_equal ? "true" : "false")
            << ",\"expected_tensor_bytes_equal\":true,\"other_tensor_bytes_equal\":true}\n";
}
