#include "models/byte-model.h"
#include "mixer/mixer-input.h"
#include "mixer/sigmoid.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>

void SourceHalvesToFloats(const uint16_t*, float*, size_t);

static uint32_t Bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}

// Exact SetProbs row mapping, followed by the original ByteUpdate method.
// Prediction and Perceive methods are linked from the original source cpp.
class Probe final : public ByteModel {
 public:
  explicit Probe(const std::vector<bool>& vocab) : ByteModel(vocab) {}
  void Load(const float* vocab_probs) {
    unsigned int k = 0;
    for (int i = 0; i < 256; ++i) {
      if (vocab_[i]) {
        probs_[i] = vocab_probs[k];
        ++k;
      } else {
        probs_[i] = 0;
      }
    }
    ByteModel::ByteUpdate();
  }
  int Bottom() const { return bot_; }
  int Top() const { return top_; }
};

struct Observation {
  uint32_t conditional, stretched;
  int lookup, override_kind, bottom, top;
};

static Observation Observe(Probe& model, MixerInput& consumer,
                           const std::unordered_map<uint32_t, int>& inverse) {
  const float p = model.Predict()[0];
  consumer.SetInput(0, p);
  const float stretched = consumer.Inputs()[0];
  const auto found = inverse.find(Bits(stretched));
  if (found == inverse.end()) {
    std::cerr << "real Stretch output absent from real Logit table\n";
    std::exit(4);
  }
  const int override_kind = (p == 0 || p == 1) ? int(p) : -1;
  return {Bits(p), Bits(stretched), found->second, override_kind,
          model.Bottom(), model.Top()};
}

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  const std::string root = argv[1];
  std::ifstream ids(root + "/ids.txt");
  std::vector<bool> vocab(256, false);
  int byte;
  for (int i = 0; i < 205; ++i) {
    if (!(ids >> byte) || byte < 0 || byte > 255 || vocab[byte]) return 3;
    vocab[byte] = true;
  }
  std::array<int, 12> target{};
  for (auto& t : target) if (!(ids >> t) || !vocab[t]) return 3;
  std::ifstream teacher(root + "/teacher.f16", std::ios::binary);
  std::ifstream oracle(root + "/oracle.f16", std::ios::binary);
  std::ofstream decoded(root + "/decoded.f32", std::ios::binary);
  std::ofstream out(root + "/observations.tsv");
  if (!teacher || !oracle || !decoded || !out) return 3;

  const Sigmoid sigmoid(100001);
  MixerInput tc(sigmoid, 1.0e-4f), oc(sigmoid, 1.0e-4f);
  // Invert actual table output bits. This identifies the table entry selected
  // by the original SetInput and Logit methods without replacing their math.
  std::unordered_map<uint32_t, int> inverse;
  inverse.reserve(100001);
  for (int i = 0; i < 100001; ++i) {
    const float centre = (i + 0.5f) / 100001;
    if (!inverse.emplace(Bits(sigmoid.Logit(centre)), i).second) return 4;
  }
  out << "row\tdepth\tbit\tbottom\ttop\tteacher_q_bits\toracle_q_bits"
         "\tteacher_stretch_bits\toracle_stretch_bits\tteacher_lookup"
         "\toracle_lookup\tteacher_override\toracle_override\n";

  Probe tm(vocab), om(vocab);
  const unsigned int vocab_size_ = 205;
  for (int row = 0; row < 12; ++row) {
    for (int which = 0; which < 2; ++which) {
      std::array<uint16_t, 205> half_scratch_{};
      std::array<float, 205> probs_scratch_{};
      auto& input = which ? oracle : teacher;
      input.read(reinterpret_cast<char*>(half_scratch_.data()), 410);
      if (input.gcount() != 410) return 3;
      SourceHalvesToFloats(half_scratch_.data(), probs_scratch_.data(), vocab_size_);
      for (unsigned int i = 0; i < vocab_size_; ++i) {
        if (!(probs_scratch_[i] >= 1e-6f)) probs_scratch_[i] = 1e-6f;
      }
      decoded.write(reinterpret_cast<const char*>(probs_scratch_.data()), 820);
      (which ? om : tm).Load(probs_scratch_.data());
    }
    for (int depth = 0; depth < 8; ++depth) {
      const int bit = (target[row] >> (7 - depth)) & 1;
      const auto t = Observe(tm, tc, inverse), o = Observe(om, oc, inverse);
      if (t.bottom != o.bottom || t.top != o.top) return 5;
      out << row << '\t' << depth << '\t' << bit << '\t' << t.bottom << '\t'
          << t.top << '\t' << t.conditional << '\t' << o.conditional << '\t'
          << t.stretched << '\t' << o.stretched << '\t' << t.lookup << '\t'
          << o.lookup << '\t' << t.override_kind << '\t' << o.override_kind << '\n';
      tm.Perceive(bit);
      om.Perceive(bit);
    }
    if (tm.Bottom() != target[row] || tm.Top() != target[row] ||
        om.Bottom() != target[row] || om.Top() != target[row]) return 5;
  }
  if (teacher.peek() != EOF || oracle.peek() != EOF) return 3;
  std::cout << "original-byte-model-and-mixerinput-observed 12 rows 96 bits; "
               "actual Logit table unique entries " << inverse.size() << '\n';
  return 0;
}
