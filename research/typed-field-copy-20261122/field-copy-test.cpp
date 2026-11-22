// SPDX-License-Identifier: GPL-3.0-only
#include "field-copy.h"
#include <iostream>

using namespace field_copy;
using namespace std::string_literals;

static std::vector<std::uint8_t> bytes(const std::string& text) {
  return {text.begin(), text.end()};
}
static std::string title(const std::string& spelling) {
  return title_open() + spelling + title_close() + "\n";
}
static Encoded round_trip(const std::string& text, std::size_t tail,
                          Limits limits = {}) {
  const auto source = bytes(text);
  // Fixed fixture: no labels or original bytes are exposed to the decoder.
  const std::vector<std::uint16_t> probabilities(source.size() * 8, 32768);
  auto encoded = encode(source, probabilities, tail, limits);
  const auto decoded = decode(encoded.bytes, probabilities, source.size(), tail, limits);
  require(decoded.source == source, "test: decoded source mismatch");
  require(encoded.stats.same_counts(decoded.stats), "test: encoder/decoder counts differ");
  require(encoded.original_predicts == source.size() * 8 &&
          encoded.original_perceives == source.size() * 8 &&
          decoded.original_predicts == source.size() * 8 &&
          decoded.original_perceives == source.size() * 8,
          "test: each original bit needs exactly one Predict/Perceive pair");
  require(encoded.original_bit_digest == decoded.original_bit_digest,
          "test: original-bit order differs");
  require(encoded.final_model.one == decoded.final_model.one &&
          encoded.final_model.zero == decoded.final_model.zero,
          "test: event model diverged");
  require(encoded.stats.whole.avoided_suffix == decoded.stats.whole.avoided_suffix &&
          encoded.stats.whole.raw_events == decoded.stats.whole.raw_events &&
          encoded.stats.whole.copy_events == decoded.stats.whole.copy_events &&
          encoded.stats.whole.index == decoded.stats.whole.index,
          "test: charged costs differ");
  require(encoded.stats.prefix3_triggers == encoded.stats.empty_group_skips +
          encoded.stats.events,
          "test: missing or duplicate prefix3 event");
  require(encoded.stats.table.heap_peak <= kTableHeapLimit,
          "test: table tracked heap over fixed budget");
  return encoded;
}

int main() {
  try {
    require(title_open().size() == 15 && title_close().size() == 5,
            "test: fixed WRT marker lengths");

    // The copied inclusive range arithmetic matches its exact counter, and
    // the real byte decoder closes a nonuniform fixture without source data.
    const auto raw_source = bytes("inclusive range\0raw fixture"s);
    RangeEncoder raw;
    RangeCounter counter;
    std::vector<std::uint16_t> raw_probabilities;
    for (std::size_t bit = 0; bit < raw_source.size() * 8; ++bit) {
      const auto p = static_cast<std::uint16_t>(1 + (bit * 17011 + 9127) % 65535);
      const int label = (raw_source[bit / 8] >> (7 - bit % 8)) & 1;
      raw_probabilities.push_back(p);
      raw.encode(p, label);
      counter.encode(p, label);
    }
    raw.flush();
    const auto raw_coded = raw.take_output();
    require(raw_coded.size() == counter.flushed_bytes(), "test: original golden counter mismatch");
    RangeDecoder raw_decoder(raw_coded);
    for (std::size_t bit = 0; bit < raw_probabilities.size(); ++bit)
      require(raw_decoder.decode(raw_probabilities[bit]) ==
              ((raw_source[bit / 8] >> (7 - bit % 8)) & 1),
              "test: original inclusive decoder mismatch");

    // One standard truncated-binary group, all ranks, includes the short and
    // long code branches. A singleton sends no index bits.
    RangeEncoder indices;
    for (std::size_t rank = 0; rank < 5; ++rank) encode_index(indices, 5, rank);
    encode_index(indices, 1, 0);
    indices.flush();
    const auto index_bytes = indices.take_output();
    RangeDecoder index_decoder(index_bytes);
    for (std::size_t rank = 0; rank < 5; ++rank)
      require(decode_index(index_decoder, 5) == rank, "test: truncated-binary rank mismatch");
    require(decode_index(index_decoder, 1) == 0 && index_bit_count(1, 0) == 0,
            "test: singleton index");

    // Causal completion and title legality. No partially read current or
    // future title can establish a reference.
    ReferenceTable causal;
    MethodStats causal_stats;
    Parser parser(causal, causal_stats);
    std::uint64_t position = 0;
    auto feed = [&](const std::string& value) {
      for (unsigned char c : value) {
        parser.observe(c, position++);
        (void)parser.take_trigger();
      }
    };
    feed(title_open() + "FUTabcdefghijk");
    require(!causal.contains("FUTabcdefghijk"), "test: current title inserted before close");
    feed(title_close().substr(0, 4));
    require(!causal.contains("FUTabcdefghijk"), "test: partial close inserted title");
    feed(title_close().substr(4));
    require(causal.contains("FUTabcdefghijk"), "test: complete title missing");
    feed(title("BADQcaption") + title("BAD[bracket") + title("BAD]close") + title("BAD\nnewline"));
    require(causal.statistics().rejected_invalid == 4, "test: illegal title entered target table");
    feed("[NEWabcdefghijk");
    require(!causal.contains("NEWabcdefghijk"), "test: current target inserted before delimiter");
    feed("]");
    require(causal.contains("NEWabcdefghijk"), "test: completed target missing");

    // Deterministic global completed-recency eviction and within-prefix MRU.
    ReferenceTable small(Limits{2, kMaxSpellingBytes, kMaxField});
    small.complete("ABCone");
    small.complete("ABCtwo");
    require(small.spelling(small.group("ABC"), 0) == "ABCtwo", "test: initial group MRU");
    small.complete("ABCone");
    require(small.spelling(small.group("ABC"), 0) == "ABCone", "test: repeated completion recency");
    small.complete("DEFthree");
    require(small.contains("ABCone") && small.contains("DEFthree") && !small.contains("ABCtwo") &&
            small.statistics().evictions == 1, "test: global completed-recency eviction");
    ReferenceTable byte_limited(Limits{4, 7, kMaxField});
    byte_limited.complete("ABC");
    byte_limited.complete("DEF");
    byte_limited.complete("GHI");
    require(!byte_limited.contains("ABC") && byte_limited.statistics().spelling_bytes == 6 &&
            byte_limited.statistics().evictions == 1, "test: aggregate spelling-byte eviction");

    const std::string spelling = "ABCabcdefghijklmnop";
    std::string combined = "initial partial spelling]\n[FUTabcdefghijk]\n" +
        title(spelling) + title("ABCotherlongthing") + title("HUGoldspelling") +
        "[" + spelling + "Qcaption-only]\n[[external url]]\n[] [Qempty] [xy]\n" +
        "[ABCzzz] [ABC] [ABC]\n[HUG" + std::string(1100, 'a') + "]\n[ABCunfinished";
    const auto mixed = round_trip(combined, combined.size() / 2);
    require(mixed.stats.copy_events >= 1 && mixed.stats.raw_events >= 1 &&
            mixed.stats.empty_group_skips >= 1 && mixed.stats.captions >= 2 &&
            mixed.stats.external_links == 1 && mixed.stats.overlength_fields == 1 &&
            mixed.stats.partial_fields == 1 && mixed.stats.zero_suffix_matches >= 1,
            "test: fixed parser and charged RAW branches");

    const std::string crossing = title(spelling) + "[" + spelling + "]";
    const auto copy_start = crossing.find("[" + spelling);
    const auto crossed = round_trip(crossing, copy_start + 1 + kPrefix + 5);
    require(crossed.stats.copy_events == 1 && crossed.stats.tail.avoided_suffix == 11 * 8 &&
            crossed.stats.tail.copy_events == 0 && crossed.stats.tail.index == 0 &&
            crossed.tail.recorded, "test: physical/ideal tail checkpoint inside COPY");

    const auto eof3 = round_trip(title("ABC") + "[ABC", 0);
    require(eof3.stats.events == 1 && eof3.stats.raw_events == 1 &&
            eof3.stats.partial_fields == 1 && eof3.stats.complete_targets == 0 &&
            eof3.stats.table.completions == 1, "test: charged RAW at exact prefix3 EOF");

    // Train the one fixed event prior with actual profitable copies. After
    // these observed events, a real zero-suffix COPY is cheaper than RAW.
    const std::string run_name = "RUNabcdefghijklmnopqrstuvw";
    std::string zero = title(run_name) + title("XYZ");
    for (unsigned i = 0; i < 300; ++i) zero += "[" + run_name + "]";
    zero += "[XYZ]";
    const auto zero_result = round_trip(zero, zero.size() - 2);
    require(zero_result.stats.copy_events == 301 && zero_result.stats.zero_suffix_copies == 1,
            "test: actual zero-suffix COPY and one-shot latch");

    const std::string evicted = title("ABCabcdefgh") + title("DEFabcdefgh") +
        "[ABCabcdefgh]" + title("GHIabcdefgh") + "[DEFabcdefgh]";
    const auto eviction_roundtrip = round_trip(evicted, evicted.size() / 2,
                                               Limits{2, kMaxSpellingBytes, kMaxField});
    require(eviction_roundtrip.stats.table.evictions >= 1 &&
            eviction_roundtrip.stats.empty_group_skips >= 1,
            "test: eviction remains synchronized through range decoding");

    EventModel renormalized;
    for (unsigned i = 0; i < 65536; ++i) renormalized.observe(i & 1);
    require(renormalized.one >= 1 && renormalized.zero >= 1 &&
            renormalized.one + renormalized.zero < 65536,
            "test: fixed event counter halving");

    std::cout << "{\"status\":\"passed\",\"checks\":[\"inclusive counter and real decode\","
      "\"truncated-binary index\",\"causal completion and title filter\","
      "\"global and byte-cap eviction\",\"COPY plus charged RAW/parser branches\","
      "\"tail inside copied suffix\",\"charged EOF prefix3\",\"zero-suffix latch\","
      "\"every original bit Predict then Perceive once\",\"event prior halving\"],"
      "\"mixed_copy_events\":" << mixed.stats.copy_events
      << ",\"zero_fixture_copy_events\":" << zero_result.stats.copy_events
      << ",\"table_heap_peak_bytes\":" << mixed.stats.table.heap_peak << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "field-copy-test failed: " << error.what() << '\n';
    return 1;
  }
}
