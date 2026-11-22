// SPDX-License-Identifier: GPL-3.0-only
// Build with -I products/hutter-research/research for the existing trace.h.
// A fixed observation-only u16 fixture screens one charged field mechanism.
#include "field-copy.h"
#include "trace.h"
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/resource.h>

using namespace field_copy;

static std::string json_string(const std::string& value) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << char(c);
    else if (c == '\n') out << "\\n";
    else if (c == '\r') out << "\\r";
    else if (c == '\t') out << "\\t";
    else if (c < 32) {
      out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c)
          << std::dec << std::setfill(' ');
    } else out << char(c);
  }
  out << '"';
  return out.str();
}
static std::uint64_t integer(const char* text) {
  require(text && *text && *text != '-', "invalid unsigned CLI integer");
  std::size_t parsed = 0;
  const auto value = std::stoull(text, &parsed);
  require(parsed == std::string(text).size(), "invalid integer suffix");
  return value;
}
static std::uint64_t file_size(std::ifstream& file) {
  file.seekg(0, std::ios::end);
  const auto size = file.tellg();
  require(size >= 0, "input file size unavailable");
  file.seekg(0, std::ios::beg);
  return static_cast<std::uint64_t>(size);
}
static std::uint64_t fnv64(const std::vector<std::uint8_t>& data) {
  std::uint64_t hash = 14695981039346656037ull;
  for (auto byte : data) { hash ^= byte; hash *= 1099511628211ull; }
  return hash;
}
static std::string hex64(std::uint64_t value) {
  std::ostringstream out;
  out << std::hex << std::setw(16) << std::setfill('0') << value;
  return out.str();
}
static void validate_record(const HutterTraceRecord& record) {
  require(record.probability > 0 && record.flags <= 3 &&
          std::isfinite(record.neural) && std::isfinite(record.fxcm) &&
          std::isfinite(record.mixer) && record.neural >= 0 && record.neural <= 1 &&
          record.fxcm >= 0 && record.fxcm <= 1, "invalid original trace record");
}
static void write_costs(std::ostream& out, const Costs& costs) {
  out << "{\"raw_event_bits\":" << costs.raw_events
      << ",\"copy_event_bits\":" << costs.copy_events
      << ",\"index_bits\":" << costs.index
      << ",\"avoided_native_suffix_bits\":" << costs.avoided_suffix
      << ",\"net_saved_ideal_bits\":" << costs.net_saved() << '}';
}
static void write_table(std::ostream& out, const TableStats& table) {
  out << "{\"accepted_completions\":" << table.completions
      << ",\"distinct_insertions\":" << table.distinct_insertions
      << ",\"repeated_completions\":" << table.repeated_completions
      << ",\"global_completed_recency_evictions\":" << table.evictions
      << ",\"rejected_invalid_spellings\":" << table.rejected_invalid
      << ",\"rejected_overlength_spellings\":" << table.rejected_overlength
      << ",\"current_entries\":" << table.entries
      << ",\"current_spelling_bytes\":" << table.spelling_bytes
      << ",\"peak_entries\":" << table.peak_entries
      << ",\"peak_spelling_bytes\":" << table.peak_spelling_bytes
      << ",\"tracked_heap_live_bytes\":" << table.heap_live
      << ",\"tracked_heap_peak_bytes\":" << table.heap_peak
      << ",\"tracked_allocation_calls\":" << table.heap_allocations << '}';
}
static void write_counts(std::ostream& out, const MethodStats& stats) {
  out << "{\"prefix3_triggers\":" << stats.prefix3_triggers
      << ",\"empty_group_deterministic_skips\":" << stats.empty_group_skips
      << ",\"charged_events\":" << stats.events
      << ",\"raw_events\":" << stats.raw_events
      << ",\"copy_events\":" << stats.copy_events
      << ",\"exact_completed_matches_at_charged_triggers\":" << stats.exact_matches
      << ",\"zero_suffix_matches\":" << stats.zero_suffix_matches
      << ",\"zero_suffix_copies\":" << stats.zero_suffix_copies
      << ",\"copied_suffix_bytes\":" << stats.copied_suffix_bytes
      << ",\"max_prefix_group_size\":" << stats.max_group_size
      << ",\"complete_titles\":" << stats.complete_titles
      << ",\"complete_internal_targets\":" << stats.complete_targets
      << ",\"captions\":" << stats.captions
      << ",\"external_links\":" << stats.external_links
      << ",\"aborts\":" << stats.aborts
      << ",\"overlength_fields\":" << stats.overlength_fields
      << ",\"partial_fields\":" << stats.partial_fields << '}';
}
static bool equal_costs(const Costs& a, const Costs& b) {
  return a.raw_events == b.raw_events && a.copy_events == b.copy_events &&
         a.index == b.index && a.avoided_suffix == b.avoided_suffix;
}

int main(int argc, char** argv) {
  const auto start_wall = std::chrono::steady_clock::now();
  const auto start_cpu = std::clock();
  std::string output_prefix;
  try {
    if (argc != 8) throw std::runtime_error(
      "usage: field-event-replay TRACE PREPARED OFFSET LENGTH TAIL_START GOLDEN OUTPUT_PREFIX");
    const std::string trace_path = argv[1], prepared_path = argv[2];
    const auto offset = integer(argv[3]);
    const auto length64 = integer(argv[4]), tail64 = integer(argv[5]), golden = integer(argv[6]);
    output_prefix = argv[7];
    require(length64 > 0 && length64 <= 8ull * 1024 * 1024 && tail64 <= length64 && golden > 0,
            "replay sample outside fixed first-stage size range");
    const auto length = static_cast<std::size_t>(length64);
    const auto tail_start = static_cast<std::size_t>(tail64);

    std::ifstream prepared(prepared_path, std::ios::binary);
    require(prepared.good(), "prepared stream unavailable");
    const auto prepared_bytes = file_size(prepared);
    require(offset <= prepared_bytes && length64 <= prepared_bytes - offset,
            "declared slice exceeds prepared stream");
    prepared.seekg(static_cast<std::streamoff>(offset));
    std::vector<std::uint8_t> source(length);
    prepared.read(reinterpret_cast<char*>(source.data()), static_cast<std::streamsize>(length));
    require(static_cast<std::size_t>(prepared.gcount()) == length, "prepared slice short read");
    prepared.close();

    std::ifstream trace(trace_path, std::ios::binary);
    require(trace.good(), "declared original trace unavailable");
    const std::uint64_t expected_trace_bytes = length64 * 8 * sizeof(HutterTraceRecord);
    require(file_size(trace) == expected_trace_bytes, "trace file length differs from declared slice");
    std::vector<std::uint16_t> probabilities(length * 8);
    RangeCounter original;
    std::uint64_t original_tail_closed = 0;
    double original_ideal_bits = 0, original_tail_ideal_bits = 0;
    std::uint64_t override_bits = 0;
    std::array<HutterTraceRecord, 65536> block;
    std::size_t bit_position = 0;
    while (bit_position < probabilities.size()) {
      const auto count = std::min(block.size(), probabilities.size() - bit_position);
      const auto bytes_to_read = count * sizeof(HutterTraceRecord);
      trace.read(reinterpret_cast<char*>(block.data()), static_cast<std::streamsize>(bytes_to_read));
      require(static_cast<std::size_t>(trace.gcount()) == bytes_to_read, "original trace short read");
      for (std::size_t i = 0; i < count; ++i, ++bit_position) {
        if (bit_position == tail_start * 8) original_tail_closed = original.flushed_bytes();
        const auto& record = block[i];
        validate_record(record);
        const int label = record.flags & 1;
        require(label == ((source[bit_position / 8] >> (7 - bit_position % 8)) & 1),
                "trace label differs from declared prepared bit");
        probabilities[bit_position] = record.probability;
        original.encode(record.probability, label);
        const auto cost = bit_cost(record.probability, label);
        original_ideal_bits += cost;
        if (bit_position >= tail_start * 8) original_tail_ideal_bits += cost;
        override_bits += (record.flags >> 1) & 1;
      }
    }
    if (tail_start == length) original_tail_closed = original.flushed_bytes();
    trace.close();
    require(original.flushed_bytes() == golden, "original inclusive range count differs from golden");
    const auto trace_done_wall = std::chrono::steady_clock::now();
    const auto trace_done_cpu = std::clock();

    const auto encoded = encode(source, probabilities, tail_start);
    require(encoded.stats.prefix3_triggers == encoded.stats.empty_group_skips +
            encoded.stats.events &&
            encoded.stats.events == encoded.stats.copy_events + encoded.stats.raw_events,
            "encoder event accounting differs from fixed trigger contract");
    std::ofstream coded_out(output_prefix + ".coded.bin", std::ios::binary);
    require(coded_out.good(), "coded output path unavailable");
    coded_out.write(reinterpret_cast<const char*>(encoded.bytes.data()),
                    static_cast<std::streamsize>(encoded.bytes.size()));
    coded_out.close();
    require(coded_out.good(), "coded range stream write failed");
    const auto encode_done_wall = std::chrono::steady_clock::now();
    const auto encode_done_cpu = std::clock();

    // This decoder has no source vector, original trace records or labels.
    // Its probability fixture is an explicit observation-screen dependency.
    const auto decoded = decode(encoded.bytes, probabilities, length, tail_start);
    require(decoded.source == source, "closed range decoder reconstructed different source bytes");
    require(encoded.stats.same_counts(decoded.stats) &&
            equal_costs(encoded.stats.whole, decoded.stats.whole) &&
            equal_costs(encoded.stats.tail, decoded.stats.tail),
            "encoder/decoder table, event or cost accounts differ");
    require(encoded.original_predicts == length64 * 8 && encoded.original_perceives == length64 * 8 &&
            decoded.original_predicts == length64 * 8 && decoded.original_perceives == length64 * 8 &&
            encoded.original_bit_digest == decoded.original_bit_digest,
            "reconstructed original-bit consumption differs");
    require(encoded.final_model.one == decoded.final_model.one &&
            encoded.final_model.zero == decoded.final_model.zero,
            "final event-model counters differ");
    require(encoded.bytes.size() >= encoded.tail.closed_bytes && golden >= original_tail_closed,
            "invalid physical tail checkpoint");
    const auto finish_wall = std::chrono::steady_clock::now();
    const auto finish_cpu = std::clock();
    rusage resources{};
    require(getrusage(RUSAGE_SELF, &resources) == 0, "process RSS accounting unavailable");
    const double cpu_seconds = double(finish_cpu - start_cpu) / CLOCKS_PER_SEC;
    const double wall_seconds = std::chrono::duration<double>(finish_wall - start_wall).count();
    const auto candidate_bytes = static_cast<std::uint64_t>(encoded.bytes.size());
    const auto base_tail_bytes = golden - original_tail_closed;
    const auto candidate_tail_bytes = candidate_bytes - encoded.tail.closed_bytes;

    std::ostringstream out;
    out << std::setprecision(15)
      << "{\"schema\":\"hutter.field-copy.screen.v1\",\"status\":\"complete\","
      << "\"method\":\"TARGET prefix3; exact causal spelling; uniform MRU truncated-binary suffix copy\","
      << "\"fixture_scope\":\"Observation-only fixed u16 probabilities; true range decode and source equality. "
         "Native Predictor integration, full archive S and formal qualification are later gates. No trace ships in entrant.\","
      << "\"inputs\":{\"trace\":" << json_string(trace_path)
      << ",\"prepared\":" << json_string(prepared_path)
      << ",\"prepared_file_bytes\":" << prepared_bytes
      << ",\"offset\":" << offset << ",\"length\":" << length
      << ",\"tail_start\":" << tail_start << ",\"golden_bytes\":" << golden << "},"
      << "\"fixed_contract\":{\"prefix_bytes\":3,\"max_entries\":65536,"
         "\"max_spelling_bytes\":8388608,\"max_field_bytes\":1024,\"table_heap_limit_bytes\":67108864,"
         "\"copy_initial_count\":1,\"raw_initial_count\":256,"
         "\"event_probability\":\"1+floor(65534*one/(one+zero)); halve at total>=65536\","
         "\"target_only\":true,\"delimiter_raw\":true,\"one_event_per_field\":true,"
         "\"empty_group_skip\":true,\"nonempty_prefix3_at_eof_is_raw\":true},"
      << "\"acceptance\":{\"all_trace_labels_match\":true,\"valid_trace_records\":" << bit_position
      << ",\"original_golden_matches\":true,\"source_equal_after_decode\":true,"
         "\"encoder_decoder_counts_equal\":true,\"encoder_decoder_costs_equal\":true,"
         "\"original_predicts_and_perceives_each_once\":true,\"frozen_fixture_bytes\":"
      << probabilities.size() * sizeof(std::uint16_t)
      << ",\"encoder_original_bits\":" << encoded.original_perceives
      << ",\"decoder_original_bits\":" << decoded.original_perceives
      << ",\"original_bit_order_fnv64\":" << json_string(hex64(encoded.original_bit_digest))
      << ",\"source_bytes_fnv64\":" << json_string(hex64(fnv64(source)))
      << ",\"decoded_bytes_fnv64\":" << json_string(hex64(fnv64(decoded.source))) << "},"
      << "\"range\":{\"whole_base_bytes\":" << golden
      << ",\"whole_candidate_bytes\":" << candidate_bytes
      << ",\"whole_saved_bytes\":" << std::int64_t(golden) - std::int64_t(candidate_bytes)
      << ",\"tail_base_bytes\":" << base_tail_bytes
      << ",\"tail_candidate_bytes\":" << candidate_tail_bytes
      << ",\"tail_saved_bytes\":" << std::int64_t(base_tail_bytes) - std::int64_t(candidate_tail_bytes)
      << ",\"base_closed_prefix_at_tail_bytes\":" << original_tail_closed
      << ",\"candidate_closed_prefix_at_tail_bytes\":" << encoded.tail.closed_bytes
      << ",\"candidate_emitted_before_tail_bytes\":" << encoded.tail.emitted_bytes
      << ",\"base_terminal_inclusive_tail_bytes\":" << base_tail_bytes + 1
      << ",\"candidate_terminal_inclusive_tail_bytes\":" << candidate_bytes - encoded.tail.emitted_bytes
      << ",\"tail_definition\":\"Physical closed-prefix byte difference at reconstructed source boundary; "
         "one termination byte cancels. Event/index attributed at prefix3 trigger even across copied suffix.\","
      << "\"decoder_physical_bytes_read\":" << decoded.physical_bytes_read
      << ",\"decoder_original_zero_padding_bytes\":" << decoded.zero_padding_bytes << "},"
      << "\"ideal\":{\"original_base_bits\":" << original_ideal_bits
      << ",\"original_tail_bits\":" << original_tail_ideal_bits
      << ",\"whole\":";
    write_costs(out, encoded.stats.whole);
    out << ",\"tail\":"; write_costs(out, encoded.stats.tail);
    out << "},\"counts\":"; write_counts(out, encoded.stats);
    out << ",\"table\":"; write_table(out, encoded.stats.table);
    out << ",\"event_model_final\":{\"one\":" << encoded.final_model.one
        << ",\"zero\":" << encoded.final_model.zero << "},"
        << "\"resources\":{\"total_cpu_seconds\":" << cpu_seconds
        << ",\"total_wall_seconds\":" << wall_seconds
        << ",\"trace_validation_cpu_seconds\":" << double(trace_done_cpu - start_cpu) / CLOCKS_PER_SEC
        << ",\"event_encoding_cpu_seconds\":" << double(encode_done_cpu - trace_done_cpu) / CLOCKS_PER_SEC
        << ",\"event_decoding_cpu_seconds\":" << double(finish_cpu - encode_done_cpu) / CLOCKS_PER_SEC
        << ",\"trace_validation_wall_seconds\":" << std::chrono::duration<double>(trace_done_wall - start_wall).count()
        << ",\"event_encoding_wall_seconds\":" << std::chrono::duration<double>(encode_done_wall - trace_done_wall).count()
        << ",\"event_decoding_wall_seconds\":" << std::chrono::duration<double>(finish_wall - encode_done_wall).count()
        << ",\"max_process_rss_kib\":" << resources.ru_maxrss
        << ",\"requested_replay_cpu_limit_seconds\":120,\"cpu_within_120_seconds\":"
        << (cpu_seconds <= 120 ? "true" : "false") << "},"
        << "\"override_original_bits\":" << override_bits
        << ",\"artifacts\":{\"coded_stream\":" << json_string(output_prefix + ".coded.bin")
        << ",\"result\":" << json_string(output_prefix + ".result.json") << "}}\n";
    const auto result = out.str();
    std::ofstream report(output_prefix + ".result.json", std::ios::binary);
    require(report.good(), "result output path unavailable");
    report << result;
    report.close();
    require(report.good(), "result JSON write failed");
    std::cout << result;
    return cpu_seconds <= 120 ? 0 : 2;
  } catch (const std::exception& error) {
    const std::string result = "{\"schema\":\"hutter.field-copy.screen.v1\",\"status\":\"failed\",\"error\":" +
        json_string(error.what()) + "}\n";
    if (!output_prefix.empty()) {
      std::ofstream report(output_prefix + ".result.json", std::ios::binary);
      if (report.good()) report << result;
    }
    std::cout << result;
    std::cerr << "field-event-replay failed: " << error.what() << '\n';
    return 1;
  }
}
