// Reuse the existing exact frequency contract and inclusive RangeCounter.
#define main hutter_original_replay_main
#include "replay.cpp"
#undef main
#include <iterator>

constexpr const char* names[] = {"format", "page_delta", "old_revision", "new_revision",
    "old_timestamp", "new_timestamp", "username_first", "username_repeat",
    "user_id_first", "user_id_repeat", "ip"};
constexpr const char* association_names[] = {"outside_user_id", "new_username",
    "seen_username_same_id", "seen_username_changed_id"};
struct FieldCost {
  std::uint64_t bytes = 0, overrides = 0;
  double probability_bits = 0, interval_bits = 0;
  void observe(double nll, double interval, bool first_bit, bool override) {
    bytes += first_bit; overrides += override;
    probability_bits += nll; interval_bits += interval;
  }
  void print(const char* name) const {
    std::cout << "{\"name\":\"" << name << "\",\"input_bytes\":" << bytes
              << ",\"probability_bits\":" << probability_bits
              << ",\"interval_bits\":" << interval_bits
              << ",\"override_bits\":" << overrides << '}';
  }
};

std::string read_file(const char* path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot open input/labels");
  std::string out((std::istreambuf_iterator<char>(file)), {});
  if (file.bad()) throw std::runtime_error("read failed");
  return out;
}

int main(int argc, char** argv) {
  try {
    if (argc != 5) throw std::runtime_error("usage: field-cost TRACE INPUT LABELS GOLDEN_BYTES");
    const auto data = read_file(argv[2]), labels = read_file(argv[3]);
    if (data.empty() || data.size() != labels.size())
      throw std::runtime_error("labels must cover every input byte");
    std::ifstream input(argv[1], std::ios::binary);
    if (!input) throw std::runtime_error("cannot open trace");
    RangeCounter range;
    std::array<FieldCost, 11> fields{};
    std::array<FieldCost, 4> associations{};
    FieldCost total;
    std::uint64_t bits = 0;
    std::array<HutterTraceRecord, 8192> block;
    while (input.read(reinterpret_cast<char*>(block.data()), sizeof(block)) || input.gcount()) {
      const auto size = input.gcount();
      if (size % sizeof(HutterTraceRecord)) throw std::runtime_error("truncated trace");
      for (std::size_t i = 0; i < size / sizeof(HutterTraceRecord); ++i) {
        const auto& r = block[i];
        validate(r);
        const auto byte = bits / 8;
        if (byte >= data.size()) throw std::runtime_error("trace longer than input");
        const int bit = r.flags & 1;
        if (bit != ((static_cast<unsigned char>(data[byte]) >> (7 - (bits & 7))) & 1))
          throw std::runtime_error("trace bit/input alignment failed");
        const auto label = static_cast<unsigned char>(labels[byte]);
        const unsigned field = label & 15, association = label >> 4;
        if (field >= fields.size() || association >= associations.size() ||
            ((association != 0) != (field == 8 || field == 9)))
          throw std::runtime_error("invalid field label");
        const auto previous_bytes = range.bytes;
        const double width = std::uint64_t(range.high) - range.low + 1;
        range.encode(r.probability, bit);
        const double after = std::uint64_t(range.high) - range.low + 1;
        const double interval = std::log2(width / after) + 8 * (range.bytes - previous_bytes);
        const double nll = cost(r.probability / 65536.0, bit);
        const bool first_bit = (bits & 7) == 0, override = r.flags & 2;
        fields[field].observe(nll, interval, first_bit, override);
        associations[association].observe(nll, interval, first_bit, override);
        total.observe(nll, interval, first_bit, override);
        ++bits;
      }
    }
    if (!input.eof() || bits != data.size() * 8)
      throw std::runtime_error("incomplete trace/input coverage");
    const auto coded_bytes = range.flushed_bytes();
    if (coded_bytes != std::stoull(argv[4])) throw std::runtime_error("range count differs from golden");
    std::uint64_t covered = 0;
    for (const auto& f : fields) covered += f.bytes;
    if (covered != data.size()) throw std::runtime_error("field byte coverage failed");
    std::cout << std::setprecision(12)
              << "{\"status\":\"complete\",\"trace_bits\":" << bits
              << ",\"base_coded_bytes\":" << coded_bytes
              << ",\"bit_input_alignment\":true,\"field_byte_coverage\":true,\"total\":";
    total.print("all");
    std::cout << ",\"fields\":[";
    for (std::size_t i = 0; i < fields.size(); ++i) {
      if (i) std::cout << ',';
      fields[i].print(names[i]);
    }
    std::cout << "],\"user_id_by_username_history\":[";
    for (std::size_t i = 1; i < associations.size(); ++i) {
      if (i != 1) std::cout << ',';
      associations[i].print(association_names[i]);
    }
    std::cout << "]}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
