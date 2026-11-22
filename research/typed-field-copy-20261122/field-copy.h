// SPDX-License-Identifier: GPL-3.0-only
// Fixed TARGET prefix3 exact-reference transmission screen, round 728.
// Inclusive range arithmetic follows hutter-research/research/replay.cpp
// and src/coder/{encoder,decoder}.cpp. Parser semantics follow the existing
// entity-prefix-replay.cpp; no original/future source is supplied to decode.
#ifndef HUTTER_FIELD_COPY_728_H
#define HUTTER_FIELD_COPY_728_H

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <list>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace field_copy {

constexpr std::size_t kPrefix = 3;
constexpr std::size_t kMaxEntries = 65536;
constexpr std::size_t kMaxSpellingBytes = 8ull * 1024 * 1024;
constexpr std::size_t kMaxField = 1024;
constexpr std::size_t kTableHeapLimit = 64ull * 1024 * 1024;

inline const std::string& title_open() {
  static const std::string marker =
      "\x20\x20\x4c\xdf\x98\x4e\x0a\x20\x20\x20\x20\x4c\xdf\x9b\x4e";
  return marker;
}
inline const std::string& title_close() {
  static const std::string marker = "\x4c\x2f\xdf\x9b\x4e";
  return marker;
}

inline void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
inline double bit_cost(std::uint32_t frequency, int bit) {
  require(frequency > 0 && frequency < 65536, "invalid u16 frequency");
  return -std::log2((bit ? double(frequency) : double(65536 - frequency)) /
                    65536.0);
}

// Exact original inclusive range counter: a checkpoint closes a COPY of the
// interval. It does not flush or change the live encoder state.
struct RangeCounter {
  std::uint32_t low = 0, high = 0xffffffffu;
  std::uint64_t bytes = 0;
  void normalize() {
    while (((low ^ high) & 0xff000000u) == 0) {
      ++bytes;
      low <<= 8;
      high = (high << 8) + 255;
    }
  }
  void encode(std::uint32_t p, int bit) {
    require(p > 0 && p < 65536, "invalid range frequency");
    const std::uint32_t mid = low + ((high - low) >> 16) * p +
        (((high - low) & 0xffffu) * p >> 16);
    if (bit) high = mid;
    else low = mid + 1;
    normalize();
  }
  std::uint64_t flushed_bytes() const {
    auto copy = *this;
    copy.normalize();
    return copy.bytes + 1;
  }
};

class RangeEncoder {
 public:
  explicit RangeEncoder(std::size_t reserve = 0) { output_.reserve(reserve); }
  void encode(std::uint32_t p, int bit) {
    require(!finished_, "encode after range flush");
    require(p > 0 && p < 65536, "invalid range frequency");
    const std::uint32_t mid = low_ + ((high_ - low_) >> 16) * p +
        (((high_ - low_) & 0xffffu) * p >> 16);
    if (bit) high_ = mid;
    else low_ = mid + 1;
    normalize();
  }
  std::uint64_t emitted_bytes() const { return output_.size(); }
  std::uint64_t closed_checkpoint_bytes() const {
    // encode() normalizes the live interval after every symbol.
    return output_.size() + 1;
  }
  void flush() {
    require(!finished_, "duplicate range flush");
    normalize();
    output_.push_back(static_cast<std::uint8_t>(high_ >> 24));
    finished_ = true;
  }
  std::vector<std::uint8_t> take_output() {
    require(finished_, "unflushed range stream");
    return std::move(output_);
  }
 private:
  void normalize() {
    while (((low_ ^ high_) & 0xff000000u) == 0) {
      output_.push_back(static_cast<std::uint8_t>(high_ >> 24));
      low_ <<= 8;
      high_ = (high_ << 8) + 255;
    }
  }
  std::uint32_t low_ = 0, high_ = 0xffffffffu;
  bool finished_ = false;
  std::vector<std::uint8_t> output_;
};

class RangeDecoder {
 public:
  explicit RangeDecoder(const std::vector<std::uint8_t>& input)
      : input_(input) {
    require(!input.empty(), "empty range stream");
    for (unsigned i = 0; i < 4; ++i) x_ = (x_ << 8) + read_byte();
  }
  int decode(std::uint32_t p) {
    require(p > 0 && p < 65536, "invalid range frequency");
    const std::uint32_t mid = low_ + ((high_ - low_) >> 16) * p +
        (((high_ - low_) & 0xffffu) * p >> 16);
    int bit = 0;
    if (x_ <= mid) { bit = 1; high_ = mid; }
    else low_ = mid + 1;
    while (((low_ ^ high_) & 0xff000000u) == 0) {
      low_ <<= 8;
      high_ = (high_ << 8) + 255;
      x_ = (x_ << 8) + read_byte();
    }
    return bit;
  }
  std::uint64_t physical_bytes_read() const { return physical_read_; }
  std::uint64_t zero_padding_bytes() const { return zero_padding_; }
 private:
  std::uint32_t read_byte() {
    if (physical_read_ < input_.size()) return input_[physical_read_++];
    ++zero_padding_;
    return 0;  // Same end-of-file padding as the original native decoder.
  }
  const std::vector<std::uint8_t>& input_;
  std::uint64_t physical_read_ = 0, zero_padding_ = 0;
  std::uint32_t low_ = 0, high_ = 0xffffffffu, x_ = 0;
};

struct EventModel {
  std::uint32_t one = 1, zero = 256;
  std::uint32_t frequency() const {
    return 1u + static_cast<std::uint32_t>(65534ull * one / (one + zero));
  }
  void observe(int copy) {
    if (copy) ++one;
    else ++zero;
    if (one + zero >= 65536) {
      one = std::max(1u, one / 2);
      zero = std::max(1u, zero / 2);
    }
  }
};

inline unsigned index_bit_count(std::size_t n, std::size_t rank) {
  require(n > 0 && n <= kMaxEntries && rank < n, "invalid reference rank");
  unsigned b = 0;
  while ((std::size_t(1) << (b + 1)) <= n) ++b;
  const std::size_t cutoff = (std::size_t(1) << (b + 1)) - n;
  return b + (rank >= cutoff ? 1 : 0);
}
inline void encode_index(RangeEncoder& coder, std::size_t n, std::size_t rank) {
  const unsigned count = index_bit_count(n, rank);
  unsigned b = 0;
  while ((std::size_t(1) << (b + 1)) <= n) ++b;
  const std::size_t cutoff = (std::size_t(1) << (b + 1)) - n;
  const std::size_t value = rank < cutoff ? rank : rank + cutoff;
  for (unsigned j = count; j > 0; --j)
    coder.encode(32768, int((value >> (j - 1)) & 1));
}
inline std::size_t decode_index(RangeDecoder& coder, std::size_t n) {
  require(n > 0 && n <= kMaxEntries, "invalid reference group size");
  unsigned b = 0;
  while ((std::size_t(1) << (b + 1)) <= n) ++b;
  const std::size_t cutoff = (std::size_t(1) << (b + 1)) - n;
  std::size_t value = 0;
  for (unsigned j = 0; j < b; ++j) value = (value << 1) | coder.decode(32768);
  const std::size_t rank = value < cutoff ? value :
      ((value << 1) | coder.decode(32768)) - cutoff;
  require(rank < n, "decoded reference rank outside group");
  return rank;
}

struct HeapStats {
  std::size_t live = 0, peak = 0, limit = kTableHeapLimit;
  std::uint64_t allocations = 0;
};
template <class T> class CountedAllocator {
 public:
  using value_type = T;
  HeapStats* stats = nullptr;
  CountedAllocator() = default;
  explicit CountedAllocator(HeapStats* value) : stats(value) {}
  template <class U> CountedAllocator(const CountedAllocator<U>& other)
      : stats(other.stats) {}
  T* allocate(std::size_t n) {
    require(stats != nullptr, "untracked table allocation");
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
      throw std::bad_alloc();
    const auto bytes = n * sizeof(T);
    if (bytes > stats->limit - stats->live) throw std::bad_alloc();
    T* pointer = std::allocator<T>{}.allocate(n);
    stats->live += bytes;
    stats->peak = std::max(stats->peak, stats->live);
    ++stats->allocations;
    return pointer;
  }
  void deallocate(T* pointer, std::size_t n) noexcept {
    const auto bytes = n * sizeof(T);
    assert(stats && stats->live >= bytes);
    stats->live -= bytes;
    std::allocator<T>{}.deallocate(pointer, n);
  }
  template <class U> bool operator==(const CountedAllocator<U>& other) const {
    return stats == other.stats;
  }
  template <class U> bool operator!=(const CountedAllocator<U>& other) const {
    return !(*this == other);
  }
};

struct Limits {
  std::size_t entries = kMaxEntries;
  std::size_t spelling_bytes = kMaxSpellingBytes;
  std::size_t field_bytes = kMaxField;
  // The executable exposes no parameter overrides. Smaller limits are used
  // only by the one local test to exercise replacement/byte-budget logic.
};
struct TableStats {
  std::uint64_t completions = 0, distinct_insertions = 0, repeated_completions = 0;
  std::uint64_t evictions = 0, rejected_invalid = 0, rejected_overlength = 0;
  std::size_t entries = 0, spelling_bytes = 0;
  std::size_t peak_entries = 0, peak_spelling_bytes = 0;
  std::size_t heap_live = 0, heap_peak = 0;
  std::uint64_t heap_allocations = 0;
};

class ReferenceTable {
 private:
  using Spelling = std::basic_string<char, std::char_traits<char>,
                                    CountedAllocator<char>>;
  struct Entry;
  using MemberList = std::list<Entry*, CountedAllocator<Entry*>>;
  using GlobalList = std::list<Entry, CountedAllocator<Entry>>;
  struct Entry {
    Spelling name;
    std::uint32_t prefix = 0;
    bool grouped = false;
    typename GlobalList::iterator global_position;
    typename MemberList::iterator group_position;
    Entry(std::string_view spelling, HeapStats* heap)
        : name(spelling.data(), spelling.size(), CountedAllocator<char>(heap)) {}
  };
  struct Bucket {
    MemberList members;
    explicit Bucket(HeapStats* heap) : members(CountedAllocator<Entry*>(heap)) {}
  };
  using LookupPair = std::pair<const std::string_view, Entry*>;
  using Lookup = std::unordered_map<std::string_view, Entry*,
      std::hash<std::string_view>, std::equal_to<std::string_view>,
      CountedAllocator<LookupPair>>;
  using BucketPair = std::pair<const std::uint32_t, Bucket>;
  using Buckets = std::unordered_map<std::uint32_t, Bucket,
      std::hash<std::uint32_t>, std::equal_to<std::uint32_t>,
      CountedAllocator<BucketPair>>;
 public:
  struct Group {
    const Bucket* bucket = nullptr;
    std::uint32_t prefix = 0;
    std::size_t size = 0;
  };
  explicit ReferenceTable(Limits limits = {})
      : limits_(limits), global_(CountedAllocator<Entry>(&heap_)),
        lookup_(0, std::hash<std::string_view>{},
                std::equal_to<std::string_view>{}, CountedAllocator<LookupPair>(&heap_)),
        buckets_(0, std::hash<std::uint32_t>{}, std::equal_to<std::uint32_t>{},
                 CountedAllocator<BucketPair>(&heap_)) {
    require(limits.entries > 0 && limits.entries <= kMaxEntries,
            "invalid table entry limit");
    require(limits.spelling_bytes > 0 && limits.spelling_bytes <= kMaxSpellingBytes,
            "invalid table spelling limit");
    require(limits.field_bytes > 0 && limits.field_bytes <= kMaxField,
            "invalid field limit");
  }
  ReferenceTable(const ReferenceTable&) = delete;
  ReferenceTable& operator=(const ReferenceTable&) = delete;

  static bool legal_name(std::string_view name) {
    return !name.empty() && name.find_first_of("[]Q\n") == std::string_view::npos;
  }
  static std::uint32_t prefix_key(std::string_view name) {
    require(name.size() >= kPrefix, "short reference prefix");
    return (std::uint32_t(static_cast<unsigned char>(name[0])) << 16) |
           (std::uint32_t(static_cast<unsigned char>(name[1])) << 8) |
           std::uint32_t(static_cast<unsigned char>(name[2]));
  }
  Group group(std::string_view prefix) const {
    const auto key = prefix_key(prefix);
    const auto found = buckets_.find(key);
    if (found == buckets_.end()) return {nullptr, key, 0};
    return {&found->second, key, found->second.members.size()};
  }
  bool contains(std::string_view name) const { return lookup_.find(name) != lookup_.end(); }
  std::size_t exact_rank(const Group& group, std::string_view name) const {
    if (!group.bucket || name.size() < kPrefix || prefix_key(name) != group.prefix)
      return group.size;
    const auto found = lookup_.find(name);
    if (found == lookup_.end()) return group.size;
    std::size_t rank = 0;
    for (const auto* entry : group.bucket->members) {
      if (entry == found->second) return rank;
      ++rank;
    }
    throw std::runtime_error("lookup/group table inconsistency");
  }
  std::string_view spelling(const Group& group, std::size_t rank) const {
    require(group.bucket && rank < group.size, "invalid frozen group rank");
    auto iterator = group.bucket->members.begin();
    std::advance(iterator, rank);
    const auto& name = (*iterator)->name;
    return {name.data(), name.size()};
  }
  void complete(std::string_view name) {
    if (name.size() > limits_.field_bytes || name.size() > limits_.spelling_bytes) {
      ++stats_.rejected_overlength;
      return;
    }
    if (!legal_name(name)) { ++stats_.rejected_invalid; return; }
    ++stats_.completions;
    const auto found = lookup_.find(name);
    if (found != lookup_.end()) {
      Entry* entry = found->second;
      global_.splice(global_.begin(), global_, entry->global_position);
      if (entry->grouped) {
        auto& members = buckets_.at(entry->prefix).members;
        members.splice(members.begin(), members, entry->group_position);
      }
      ++stats_.repeated_completions;
      return;
    }
    while (global_.size() >= limits_.entries ||
           spelling_bytes_ + name.size() > limits_.spelling_bytes) evict_oldest();
    global_.emplace_front(name, &heap_);
    Entry* entry = &global_.front();
    entry->global_position = global_.begin();
    if (name.size() >= kPrefix) {
      entry->grouped = true;
      entry->prefix = prefix_key(name);
      auto bucket = buckets_.try_emplace(entry->prefix, &heap_).first;
      bucket->second.members.push_front(entry);
      entry->group_position = bucket->second.members.begin();
    }
    lookup_.emplace(std::string_view(entry->name.data(), entry->name.size()), entry);
    spelling_bytes_ += name.size();
    ++stats_.distinct_insertions;
    stats_.peak_entries = std::max(stats_.peak_entries, global_.size());
    stats_.peak_spelling_bytes = std::max(stats_.peak_spelling_bytes, spelling_bytes_);
  }
  TableStats statistics() const {
    auto out = stats_;
    out.entries = global_.size();
    out.spelling_bytes = spelling_bytes_;
    out.heap_live = heap_.live;
    out.heap_peak = heap_.peak;
    out.heap_allocations = heap_.allocations;
    return out;
  }
  const Limits& limits() const { return limits_; }
 private:
  void evict_oldest() {
    require(!global_.empty(), "table budget has no admissible entry");
    auto& entry = global_.back();
    lookup_.erase(std::string_view(entry.name.data(), entry.name.size()));
    if (entry.grouped) {
      auto found = buckets_.find(entry.prefix);
      require(found != buckets_.end(), "missing eviction prefix group");
      found->second.members.erase(entry.group_position);
      if (found->second.members.empty()) buckets_.erase(found);
    }
    spelling_bytes_ -= entry.name.size();
    global_.pop_back();
    ++stats_.evictions;
  }
  // Declared first so tracked containers/strings are destroyed before stats.
  HeapStats heap_;
  Limits limits_;
  TableStats stats_;
  std::size_t spelling_bytes_ = 0;
  GlobalList global_;
  Lookup lookup_;
  Buckets buckets_;
};

struct Costs {
  double raw_events = 0, copy_events = 0, index = 0, avoided_suffix = 0;
  double net_saved() const { return avoided_suffix - raw_events - copy_events - index; }
};
struct MethodStats {
  std::uint64_t prefix3_triggers = 0, empty_group_skips = 0;
  std::uint64_t events = 0, raw_events = 0, copy_events = 0, exact_matches = 0;
  std::uint64_t zero_suffix_matches = 0, zero_suffix_copies = 0;
  std::uint64_t copied_suffix_bytes = 0, max_group_size = 0;
  std::uint64_t complete_titles = 0, complete_targets = 0, captions = 0;
  std::uint64_t external_links = 0, aborts = 0, overlength_fields = 0, partial_fields = 0;
  Costs whole, tail;
  TableStats table;
  bool same_counts(const MethodStats& other) const {
    const std::array<std::uint64_t, 18> a{{prefix3_triggers, empty_group_skips,
      events, raw_events, copy_events, exact_matches,
      zero_suffix_matches, zero_suffix_copies, copied_suffix_bytes, max_group_size,
      complete_titles, complete_targets, captions, external_links, aborts,
      overlength_fields, partial_fields, table.completions}};
    const std::array<std::uint64_t, 18> b{{other.prefix3_triggers, other.empty_group_skips,
      other.events, other.raw_events, other.copy_events,
      other.exact_matches, other.zero_suffix_matches, other.zero_suffix_copies,
      other.copied_suffix_bytes, other.max_group_size, other.complete_titles,
      other.complete_targets, other.captions, other.external_links, other.aborts,
      other.overlength_fields, other.partial_fields, other.table.completions}};
    return a == b && table.distinct_insertions == other.table.distinct_insertions &&
      table.repeated_completions == other.table.repeated_completions &&
      table.evictions == other.table.evictions && table.rejected_invalid == other.table.rejected_invalid &&
      table.rejected_overlength == other.table.rejected_overlength &&
      table.entries == other.table.entries && table.spelling_bytes == other.table.spelling_bytes &&
      table.peak_entries == other.table.peak_entries &&
      table.peak_spelling_bytes == other.table.peak_spelling_bytes &&
      table.heap_live == other.table.heap_live && table.heap_peak == other.table.heap_peak;
  }
};

// Only decoded bytes enter this parser. Table completion occurs at an actual
// delimiter. The trigger latch is set exactly once on the third TARGET byte.
class Parser {
 public:
  enum Mode { OUT, TITLE, TARGET, CAPTION, EXTERNAL, PENDING, TITLE_CLOSE };
  Parser(ReferenceTable& table, MethodStats& stats) : table_(table), stats_(stats) {
    rolling_.reserve(title_open().size());
    name_.reserve(table.limits().field_bytes + 1);
    held_.reserve(title_close().size());
  }
  void observe(std::uint8_t c, std::uint64_t position) {
    rolling_.push_back(static_cast<char>(c));
    if (rolling_.size() > title_open().size()) rolling_.erase(0, 1);
    if (mode_ == OUT && rolling_ == title_open()) { begin(TITLE); return; }
    switch (mode_) {
      case OUT:
        if (c == '[') mode_ = PENDING;
        break;
      case PENDING:
        if (c == '[') { mode_ = EXTERNAL; external_close_run_ = 0; }
        else if (c == ']' || c == '\n') { mode_ = OUT; ++stats_.aborts; }
        else if (c == 'Q') { mode_ = CAPTION; ++stats_.captions; }
        else { begin(TARGET); append(c); }
        break;
      case TARGET:
        if (c == ']' || c == 'Q') {
          complete(false);
          if (c == 'Q') { mode_ = CAPTION; ++stats_.captions; }
          else mode_ = OUT;
        } else if (c == '[' || c == '\n') {
          ++stats_.aborts;
          clear_name();
          mode_ = c == '[' ? PENDING : OUT;
        } else append(c);
        break;
      case CAPTION:
        if (c == ']') mode_ = OUT;
        else if (c == '\n') { mode_ = OUT; ++stats_.aborts; }
        break;
      case EXTERNAL:
        external_close_run_ = c == ']' ? external_close_run_ + 1 : 0;
        if (external_close_run_ == 2) { mode_ = OUT; ++stats_.external_links; }
        else if (c == '\n') { mode_ = OUT; ++stats_.aborts; }
        break;
      case TITLE:
        if (c == static_cast<std::uint8_t>(title_close()[0])) {
          mode_ = TITLE_CLOSE;
          held_.assign(1, static_cast<char>(c));
        } else append(c);
        break;
      case TITLE_CLOSE:
        held_.push_back(static_cast<char>(c));
        if (title_close().compare(0, held_.size(), held_) != 0) {
          for (unsigned char b : held_) append(b);
          held_.clear();
          mode_ = TITLE;
        } else if (held_.size() == title_close().size()) {
          complete(true);
          held_.clear();
          mode_ = OUT;
        }
        break;
    }
    (void)position;  // Same byte positions are supplied to encoder and decoder.
  }
  bool take_trigger() {
    const bool value = pending_trigger_;
    pending_trigger_ = false;
    return value;
  }
  void paid_event() { event_had_group_ = true; }
  Mode mode() const { return mode_; }
  std::string_view name() const { return name_; }
  void finish() { if (mode_ != OUT) ++stats_.partial_fields; }
 private:
  void clear_name() {
    name_.clear();
    field_length_ = 0;
    event_checked_ = false;
    pending_trigger_ = false;
    event_had_group_ = false;
  }
  void begin(Mode mode) { clear_name(); held_.clear(); mode_ = mode; }
  void append(std::uint8_t c) {
    ++field_length_;
    if (field_length_ == table_.limits().field_bytes + 1) ++stats_.overlength_fields;
    if (name_.size() <= table_.limits().field_bytes) name_.push_back(static_cast<char>(c));
    if (mode_ == TARGET && field_length_ == kPrefix && !event_checked_) {
      event_checked_ = true;
      pending_trigger_ = true;
    }
  }
  void complete(bool title) {
    if (title) ++stats_.complete_titles;
    else {
      ++stats_.complete_targets;
      // This diagnostic is evaluated after complete decoded spelling and
      // before inserting the current field. It does not select an event.
      if (event_had_group_ && field_length_ <= table_.limits().field_bytes &&
          table_.contains(name_)) {
        ++stats_.exact_matches;
        if (field_length_ == kPrefix) ++stats_.zero_suffix_matches;
      }
    }
    table_.complete(name_);
    clear_name();
  }
  ReferenceTable& table_;
  MethodStats& stats_;
  Mode mode_ = OUT;
  std::string rolling_, name_, held_;
  std::uint64_t field_length_ = 0;
  unsigned external_close_run_ = 0;
  bool event_checked_ = false, pending_trigger_ = false, event_had_group_ = false;
};

// The frozen fixture contains probabilities only, not labels or source bytes.
// Predict/Perceive enforce exactly one pair per reconstructed original bit.
// A future native implementation replaces this fixture with the original
// Predictor calls; event/index bits never invoke either callback.
class FrozenFixture {
 public:
  explicit FrozenFixture(const std::vector<std::uint16_t>& probabilities)
      : probabilities_(probabilities) {}
  std::uint32_t Predict() {
    require(!waiting_, "duplicate Predict without Perceive");
    require(position_ < probabilities_.size(), "prediction beyond frozen fixture");
    waiting_ = true;
    ++predicts_;
    return probabilities_[position_];
  }
  void Perceive(int bit) {
    require(waiting_ && (bit == 0 || bit == 1), "Perceive without original Predict");
    digest_ ^= static_cast<std::uint64_t>(bit);
    digest_ *= 1099511628211ull;
    ++position_;
    ++perceives_;
    waiting_ = false;
  }
  std::size_t position() const { return position_; }
  std::uint64_t predicts() const { return predicts_; }
  std::uint64_t perceives() const { return perceives_; }
  std::uint64_t digest() const { return digest_; }
  void complete(std::size_t original_bytes) const {
    require(!waiting_ && position_ == original_bytes * 8 &&
            position_ == probabilities_.size() && predicts_ == perceives_,
            "incomplete original-bit consumption");
  }
 private:
  const std::vector<std::uint16_t>& probabilities_;
  std::size_t position_ = 0;
  std::uint64_t predicts_ = 0, perceives_ = 0;
  std::uint64_t digest_ = 14695981039346656037ull;
  bool waiting_ = false;
};

struct Checkpoint {
  bool recorded = false;
  std::uint64_t emitted_bytes = 0, closed_bytes = 0;
};
struct Encoded {
  std::vector<std::uint8_t> bytes;
  MethodStats stats;
  Checkpoint tail;
  std::uint64_t original_predicts = 0, original_perceives = 0, original_bit_digest = 0;
  EventModel final_model;
};
struct Decoded {
  std::vector<std::uint8_t> source;
  MethodStats stats;
  std::uint64_t original_predicts = 0, original_perceives = 0, original_bit_digest = 0;
  std::uint64_t physical_bytes_read = 0, zero_padding_bytes = 0;
  EventModel final_model;
};

inline void add_event_cost(MethodStats& stats, std::size_t position,
                           std::size_t tail_start, std::uint32_t p, bool copy) {
  ++stats.events;
  const double cost = bit_cost(p, copy);
  if (copy) {
    ++stats.copy_events;
    stats.whole.copy_events += cost;
    if (position >= tail_start) stats.tail.copy_events += cost;
  } else {
    ++stats.raw_events;
    stats.whole.raw_events += cost;
    if (position >= tail_start) stats.tail.raw_events += cost;
  }
}
inline void add_index_cost(MethodStats& stats, std::size_t position,
                           std::size_t tail_start, unsigned bits) {
  stats.whole.index += bits;
  if (position >= tail_start) stats.tail.index += bits;
}

inline Encoded encode(const std::vector<std::uint8_t>& source,
                      const std::vector<std::uint16_t>& probabilities,
                      std::size_t tail_start, Limits limits = {}) {
  require(tail_start <= source.size(), "tail outside source");
  require(probabilities.size() == source.size() * 8, "source/fixture size mismatch");
  Encoded result;
  ReferenceTable table(limits);
  Parser parser(table, result.stats);
  FrozenFixture fixture(probabilities);
  EventModel model;
  RangeEncoder coder(source.size() / 4 + 16);
  std::size_t position = 0;
  auto checkpoint = [&]() {
    if (position == tail_start && !result.tail.recorded) {
      result.tail.recorded = true;
      result.tail.emitted_bytes = coder.emitted_bytes();
      result.tail.closed_bytes = coder.closed_checkpoint_bytes();
    }
  };
  checkpoint();
  auto consume = [&](std::uint8_t c, bool raw) {
    for (unsigned j = 8; j > 0; --j) {
      const int bit = (c >> (j - 1)) & 1;
      const auto p = fixture.Predict();
      if (raw) coder.encode(p, bit);
      else {
        const double cost = bit_cost(p, bit);
        result.stats.whole.avoided_suffix += cost;
        if (position >= tail_start) result.stats.tail.avoided_suffix += cost;
      }
      fixture.Perceive(bit);
    }
    parser.observe(c, position);
    ++position;
    checkpoint();  // Also checkpoints a boundary inside a copied suffix.
  };
  while (position < source.size()) {
    consume(source[position], true);
    if (!parser.take_trigger()) continue;
    ++result.stats.prefix3_triggers;
    require(parser.mode() == Parser::TARGET && parser.name().size() == kPrefix,
            "invalid encoder trigger state");
    const auto group = table.group(parser.name());  // Freeze BEFORE lookahead.
    result.stats.max_group_size = std::max<std::uint64_t>(result.stats.max_group_size, group.size);
    if (!group.size) { ++result.stats.empty_group_skips; continue; }
    parser.paid_event();

    std::string full_name(parser.name());
    bool complete_field = false;
    std::size_t end = position;
    for (; end < source.size(); ++end) {
      const auto c = source[end];
      if (c == ']' || c == 'Q') { complete_field = true; break; }
      if (c == '[' || c == '\n' || full_name.size() >= limits.field_bytes) break;
      full_name.push_back(static_cast<char>(c));
    }
    const std::size_t rank = complete_field ? table.exact_rank(group, full_name) : group.size;
    const auto event_p = model.frequency();
    bool copy = false;
    if (rank < group.size) {
      double suffix_cost = 0;
      for (std::size_t at = position; at < end; ++at)
        for (unsigned j = 8; j > 0; --j)
          suffix_cost += bit_cost(probabilities[at * 8 + (8 - j)],
                                  (source[at] >> (j - 1)) & 1);
      copy = bit_cost(event_p, 1) + index_bit_count(group.size, rank) <
             bit_cost(event_p, 0) + suffix_cost;
    }
    const auto event_position = position;
    coder.encode(event_p, copy);
    add_event_cost(result.stats, event_position, tail_start, event_p, copy);
    model.observe(copy);
    if (copy) {
      const auto spelling = table.spelling(group, rank);
      require(spelling.size() == full_name.size() &&
              std::string_view(full_name) == spelling, "reference equality failure");
      encode_index(coder, group.size, rank);
      add_index_cost(result.stats, event_position, tail_start, index_bit_count(group.size, rank));
      const auto suffix = spelling.substr(kPrefix);
      result.stats.copied_suffix_bytes += suffix.size();
      if (suffix.empty()) ++result.stats.zero_suffix_copies;
      for (unsigned char c : suffix) consume(c, false);
      require(!parser.take_trigger() && parser.mode() == Parser::TARGET,
              "copied suffix changed parser/event state");
    }
  }
  parser.finish();
  fixture.complete(source.size());
  require(result.tail.recorded, "missing source-tail checkpoint");
  coder.flush();
  result.bytes = coder.take_output();
  result.stats.table = table.statistics();
  result.original_predicts = fixture.predicts();
  result.original_perceives = fixture.perceives();
  result.original_bit_digest = fixture.digest();
  result.final_model = model;
  return result;
}

inline Decoded decode(const std::vector<std::uint8_t>& coded,
                      const std::vector<std::uint16_t>& probabilities,
                      std::size_t original_length, std::size_t tail_start,
                      Limits limits = {}) {
  require(tail_start <= original_length, "tail outside decoded length");
  require(probabilities.size() == original_length * 8, "length/fixture size mismatch");
  Decoded result;
  result.source.reserve(original_length);
  ReferenceTable table(limits);
  Parser parser(table, result.stats);
  FrozenFixture fixture(probabilities);
  RangeDecoder coder(coded);
  EventModel model;
  auto consume_copy_byte = [&](std::uint8_t c) {
    const auto position = result.source.size();
    for (unsigned j = 8; j > 0; --j) {
      const int bit = (c >> (j - 1)) & 1;
      const auto p = fixture.Predict();
      const auto cost = bit_cost(p, bit);
      result.stats.whole.avoided_suffix += cost;
      if (position >= tail_start) result.stats.tail.avoided_suffix += cost;
      fixture.Perceive(bit);
    }
    parser.observe(c, position);
    result.source.push_back(c);
  };
  while (result.source.size() < original_length) {
    std::uint8_t byte = 0;
    for (unsigned j = 0; j < 8; ++j) {
      const auto p = fixture.Predict();
      const int bit = coder.decode(p);
      byte = static_cast<std::uint8_t>((byte << 1) | bit);
      fixture.Perceive(bit);
    }
    parser.observe(byte, result.source.size());
    result.source.push_back(byte);
    if (!parser.take_trigger()) continue;
    ++result.stats.prefix3_triggers;
    const auto position = result.source.size();
    require(parser.mode() == Parser::TARGET && parser.name().size() == kPrefix,
            "invalid decoder trigger state");
    const auto group = table.group(parser.name());
    result.stats.max_group_size = std::max<std::uint64_t>(result.stats.max_group_size, group.size);
    if (!group.size) { ++result.stats.empty_group_skips; continue; }
    parser.paid_event();
    const auto event_p = model.frequency();
    const bool copy = coder.decode(event_p);
    add_event_cost(result.stats, position, tail_start, event_p, copy);
    model.observe(copy);
    if (copy) {
      const auto rank = decode_index(coder, group.size);
      add_index_cost(result.stats, position, tail_start, index_bit_count(group.size, rank));
      const auto spelling = table.spelling(group, rank);
      require(spelling.size() >= kPrefix && spelling.substr(0, kPrefix) == parser.name(),
              "reference prefix mismatch");
      const auto suffix = spelling.substr(kPrefix);
      require(suffix.size() < original_length - position,
              "copy leaves no original RAW delimiter");
      result.stats.copied_suffix_bytes += suffix.size();
      if (suffix.empty()) ++result.stats.zero_suffix_copies;
      for (unsigned char c : suffix) consume_copy_byte(c);
      require(!parser.take_trigger() && parser.mode() == Parser::TARGET,
              "decoded copy changed parser/event state");
    }
  }
  parser.finish();
  fixture.complete(original_length);
  result.stats.table = table.statistics();
  result.original_predicts = fixture.predicts();
  result.original_perceives = fixture.perceives();
  result.original_bit_digest = fixture.digest();
  result.physical_bytes_read = coder.physical_bytes_read();
  result.zero_padding_bytes = coder.zero_padding_bytes();
  result.final_model = model;
  return result;
}

}  // namespace field_copy
#endif
