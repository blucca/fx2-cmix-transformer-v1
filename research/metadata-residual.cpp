// Cold-state PHDA9-WRT header rewrite; matches metadata-residual.py.
// Build the optional CLI with -DMETADATA_RESIDUAL_MAIN. Without it, link
// metadata_residual::Rewrite(std::string_view, bool) into a caller.
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>

namespace metadata_residual {
namespace {
using I = std::int64_t;
constexpr I kThreshold = I{1} << 24;
constexpr I kMaxId = (I{1} << 31) - 1;
constexpr I kEndSeconds = 36524LL * 86400;  // 2001-01-01 through 2100-12-31.

[[noreturn]] void Fail(const char* message) {
#ifdef __cpp_exceptions
  throw std::runtime_error(message);
#else
  std::fprintf(stderr, "metadata-residual: %s\n", message);
  std::exit(1);
#endif
}

I Add(I a, I b) {
  if ((b > 0 && a > std::numeric_limits<I>::max() - b) ||
      (b < 0 && a < std::numeric_limits<I>::min() - b))
    Fail("integer outside signed 64-bit range");
  return a + b;
}

I Subtract(I a, I b) {
  if ((b > 0 && a < std::numeric_limits<I>::min() + b) ||
      (b < 0 && a > std::numeric_limits<I>::max() + b))
    Fail("integer outside signed 64-bit range");
  return a - b;
}

I Decimal(std::string_view raw) {
  I value = 0;
  const auto result = std::from_chars(raw.data(), raw.data() + raw.size(), value);
  if (result.ec != std::errc{} || result.ptr != raw.data() + raw.size() ||
      std::to_string(value) != raw)
    Fail("noncanonical decimal");
  return value;
}

bool Leap(int year) {
  return year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
}

int MonthDays(int year, int month) {
  constexpr int days[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  return days[month - 1] + (month == 2 && Leap(year));
}

I ParseTime(std::string_view raw) {
  const auto j = raw.find('J');
  if (j == std::string_view::npos || j < 3 || raw[0] < '0' || raw[0] > '9' ||
      raw[1] < '0' || raw[1] > '9')
    Fail("timestamp needs a two-digit year and a day");
  const int year = 2001 + (raw[0] - '0') * 10 + raw[1] - '0';
  const I d = Decimal(raw.substr(2, j - 2));
  const I seconds = Decimal(raw.substr(j + 1));
  if (d < 0 || d >= 372 || seconds < 0 || seconds >= 86400)
    Fail("invalid timestamp day or seconds");
  const int month = int(d / 31) + 1, day = int(d % 31) + 1;
  if (day > MonthDays(year, month)) Fail("invalid calendar date");
  I days = day - 1;
  for (int y = 2001; y < year; ++y) days += 365 + Leap(y);
  for (int m = 1; m < month; ++m) days += MonthDays(year, m);
  return days * 86400 + seconds;
}

std::string FormatTime(I seconds) {
  if (seconds < 0 || seconds >= kEndSeconds)
    Fail("timestamp outside 2001..2100");
  int days = int(seconds / 86400), year = 2001, month = 1;
  while (days >= 365 + Leap(year)) days -= 365 + Leap(year++);
  while (days >= MonthDays(year, month)) days -= MonthDays(year, month++);
  std::string out;
  out += char('0' + (year - 2001) / 10);
  out += char('0' + (year - 2001) % 10);
  return out + std::to_string((month - 1) * 31 + days) + "J" +
         std::to_string(seconds % 86400);
}

I RoundDiv(I n, I d) {
  // New IDs are signed-31-bit positive and times span 100 years, so the
  // interpolation product and its absolute value fit signed 64-bit.
  const I a = n < 0 ? -n : n;
  const I q = a / d + (2 * (a % d) >= d);
  return n < 0 ? -q : q;
}

I OldPredict(const std::map<I, I>& points, I page) {
  if (points.empty()) return 0;
  auto right = points.lower_bound(page), chosen = right;
  if (right == points.end()) {
    chosen = std::prev(right);
  } else if (right != points.begin()) {
    const auto left = std::prev(right);
    if (std::uint64_t(page) - std::uint64_t(left->first) <=
        std::uint64_t(right->first) - std::uint64_t(page))
      chosen = left;
  }
  return Add(chosen->second, Subtract(page, chosen->first));
}

I ClockPredict(const std::map<I, I>& points, I revision) {
  if (points.empty()) return 0;
  if (points.size() == 1) return points.begin()->second;
  auto b = points.lower_bound(revision);
  if (b == points.begin()) ++b;
  if (b == points.end()) --b;
  const auto a = std::prev(b);
  return a->second + RoundDiv((revision - a->first) * (b->second - a->second),
                              b->first - a->first);
}
}  // namespace

std::string Rewrite(std::string_view data, bool decode) {
  if (data.empty()) Fail("empty header");
  std::map<I, I> old_points, new_points;
  std::set<I> seen_pages, old_revisions;
  I page = 0;
  std::size_t cursor = 0;
  std::string out;
  out.reserve(data.size());
  auto expect = [&](std::string_view marker) {
    if (data.substr(cursor, marker.size()) != marker)
      Fail("invalid or incomplete header record");
    cursor += marker.size();
  };
  auto line = [&](std::string_view prefix, bool empty_ok = false) {
    expect(prefix);
    const auto first = cursor;
    cursor = data.find('\n', first);
    if (cursor == std::string_view::npos || (!empty_ok && cursor == first))
      Fail("invalid or incomplete header line");
    return data.substr(first, cursor++ - first);
  };
  while (cursor < data.size()) {
    const auto begin = cursor;
    const I delta = Decimal(line("N"));
    if (delta < -kMaxId || delta > kMaxId)
      Fail("page delta outside signed 31-bit magnitude");
    page = Add(page, delta);
    if (!seen_pages.insert(page).second) Fail("duplicate page key");
    if (data.substr(cursor, 3) == "\xee\xcfN") line("\xee\xcfN", true);
    expect("\xdf\x99N\n");
    const auto r_raw = line("\xdf\x86N"), t_raw = line("\xdf\xcdN");
    expect("\xdf\x8fN\n");
    if (data.substr(cursor, 3) == "\xdf\x87N") {
      line("\xdf\x87N");
      const I contributor = Decimal(line("\xdf\x86N"));
      if (contributor < 0 || contributor > kMaxId) Fail("invalid contributor ID");
    } else {
      line("ipN");
    }
    expect("/\xdf\x8fN\n");

    const bool old = decode ? r_raw.front() == '@' : Decimal(r_raw) < kThreshold;
    I revision;
    std::string replacement;
    std::string_view field;
    if (old) {
      const I pred = OldPredict(old_points, page);
      revision = decode ? Add(pred, Decimal(r_raw.substr(1))) : Decimal(r_raw);
      if (revision <= 0 || revision >= kThreshold)
        Fail("old revision outside branch range");
      ParseTime(t_raw);
      replacement = decode ? std::to_string(revision) :
                             "@" + std::to_string(Subtract(revision, pred));
      field = r_raw;
      if (!old_revisions.insert(revision).second) Fail("duplicate revision key");
      old_points.emplace(page, revision);
    } else {
      revision = Decimal(r_raw);
      if (revision < kThreshold || revision > kMaxId)
        Fail("new revision outside branch range");
      const I pred = ClockPredict(new_points, revision);
      I seconds;
      if (decode) {
        if (t_raw.front() != '@') Fail("new timestamp needs residual marker");
        seconds = Add(pred, Decimal(t_raw.substr(1)));
        replacement = FormatTime(seconds);
      } else {
        seconds = ParseTime(t_raw);
        replacement = "@" + std::to_string(seconds - pred);
      }
      field = t_raw;
      if (!new_points.emplace(revision, seconds).second) Fail("duplicate revision key");
    }
    const auto start = std::size_t(field.data() - data.data());
    out.append(data.data() + begin, start - begin);
    out += replacement;
    const auto end = start + field.size();
    out.append(data.data() + end, cursor - end);
  }
  return out;
}
}  // namespace metadata_residual

#ifdef METADATA_RESIDUAL_MAIN
#include <fstream>
#include <iostream>

namespace {
void Check(bool ok) {
  if (!ok) throw std::runtime_error("self-check failed");
}

void SelfTest() {
  using namespace metadata_residual;
  auto record = [](I delta, I revision, I seconds) {
    return "N" + std::to_string(delta) + "\n\xdf\x99N\n\xdf\x86N" +
           std::to_string(revision) + "\n\xdf\xcdN" + FormatTime(seconds) +
           "\n\xdf\x8fN\n\xdf\x87N@test\n\xdf\x86N1\n/\xdf\x8fN\n";
  };
  const I rows[][3] = {{10, 17000000, 100}, {-20, 1000, 50}, {4, 1005, 90},
                      {25, 17000002, 103}, {-13, 17000001, 102}, {-14, 1002, 40},
                      {15, 17000003, 104}, {-14, 1001, 41}};
  const char* residuals[] = {"@100", "@1000", "@1", "@3", "@0", "@0", "@0", "@-2"};
  std::string source, expected, shifted, shifted_expected;
  for (int i = 0; i < 8; ++i) {
    auto raw = record(rows[i][0], rows[i][1], rows[i][2]);
    auto translated = record(rows[i][0] + (i == 0 ? 100 : 0), rows[i][1], rows[i][2]);
    source += raw;
    shifted += translated;
    const auto field = rows[i][1] < kThreshold ? std::to_string(rows[i][1]) :
                                               FormatTime(rows[i][2]);
    raw.replace(raw.find(field), field.size(), residuals[i]);
    translated.replace(translated.find(field), field.size(), residuals[i]);
    expected += raw;
    shifted_expected += translated;
  }
  Check(Rewrite(source, false) == expected && Rewrite(expected, true) == source);
  Check(Rewrite(shifted, false) == shifted_expected);
  source.insert(source.find("\xdf\x99N"), "\xee\xcfNpreserved\n");
  const std::string user = "\xdf\x87N@test\n\xdf\x86N1\n";
  source.replace(source.find(user), user.size(), "ipN127.0.0.1\n");
  Check(Rewrite(Rewrite(source, false), true) == source);
  Check(OldPredict({{0, 100}, {10, 300}}, 5) == 105);
  Check(ClockPredict({{10, 100}, {12, 103}}, 9) == 98);
  Check(ClockPredict({{10, 100}, {12, 103}}, 13) == 105);
  Check(RoundDiv(1, 2) == 1 && RoundDiv(-1, 2) == -1 && RoundDiv(-3, 2) == -2);
  for (I seconds : {I{0}, I{99999999}, kEndSeconds - 1})
    Check(ParseTime(FormatTime(seconds)) == seconds);
  std::string bad_date = record(1, 17000000, 100);
  bad_date.replace(bad_date.find("000J100"), 7, "0059J0");
  const std::string invalid[] = {"", source.substr(0, source.size() - 1), bad_date,
      record(1, 17000000, 100) + record(0, 17000001, 101),
      record(1, 17000000, 100) + record(1, 17000000, 101)};
  for (const auto& bad : invalid) {
    bool failed = false;
    try { Rewrite(bad, false); } catch (const std::runtime_error&) { failed = true; }
    Check(failed);
  }
  for (const auto& bad : {"-0", "01", "+1"}) {
    bool failed = false;
    try { Decimal(bad); } catch (const std::runtime_error&) { failed = true; }
    Check(failed);
  }
  std::cout << "metadata-residual C++ self-check passed\n";
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string_view(argv[1]) == "--self-test") {
      SelfTest();
      return 0;
    }
    if (argc != 4 || (std::string_view(argv[1]) != "encode" &&
                      std::string_view(argv[1]) != "decode"))
      throw std::runtime_error("usage: metadata-residual encode|decode INPUT OUTPUT | --self-test");
    std::ifstream input(argv[2], std::ios::binary);
    if (!input) throw std::runtime_error("cannot open input");
    const std::string data((std::istreambuf_iterator<char>(input)), {});
    if (input.bad()) throw std::runtime_error("input read failed");
    const auto output = metadata_residual::Rewrite(data, std::string_view(argv[1]) == "decode");
    std::ofstream file(argv[3], std::ios::binary);
    file.write(output.data(), output.size());
    file.close();
    if (!file) throw std::runtime_error("output write failed");
    std::cout << "{\"input_bytes\":" << data.size() << ",\"output_bytes\":"
              << output.size() << "}\n";
  } catch (const std::exception& exc) {
    std::cerr << "metadata-residual: " << exc.what() << '\n';
    return 1;
  }
}
#endif
