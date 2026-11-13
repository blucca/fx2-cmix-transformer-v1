// Narrow, standalone RowwiseMoments probe for pytorch/pytorch#204610.
// Build flags must select AVX2 ATen headers: CPU_CAPABILITY=AVX2,
// CPU_CAPABILITY_AVX2, and the platform's AVX2/FMA compiler flags.
#include <ATen/native/cpu/moments_utils.h>

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

namespace moments = at::native::CPU_CAPABILITY;

struct Reference {
  double mean;
  double variance;
};

template <typename T>
Reference reference(const std::vector<T>& values) {
  // Values are exact multiples of 1/256. The sums of values and squared
  // values remain exactly representable in FP64 for all four requested N.
  double sum = 0;
  double sumsq = 0;
  for (T value : values) {
    const double x = static_cast<double>(value);
    sum += x;
    sumsq += x * x;
  }
  const double count = static_cast<double>(values.size());
  const double mean = sum / count;
  return {mean, sumsq / count - mean * mean};
}

void number(double value) {
  if (std::isfinite(value)) {
    std::cout << value;
  } else {
    std::cout << "null";
  }
}

template <typename T>
void result(const char* name, int64_t specialization,
            const std::pair<T, T>& output, const Reference& ref,
            bool& first) {
  if (!first) std::cout << ',';
  first = false;
  const double mean = static_cast<double>(output.first);
  const double variance = static_cast<double>(output.second);
  std::cout << "{\"name\":\"" << name
            << "\",\"specialization\":" << specialization << ",\"mean\":";
  number(mean);
  std::cout << ",\"variance\":";
  number(variance);
  std::cout << ",\"mean_absdiff\":";
  number(std::abs(mean - ref.mean));
  std::cout << ",\"variance_absdiff\":";
  number(std::abs(variance - ref.variance));
  std::cout << '}';
}

template <typename T>
void probe(const char* dtype, int64_t N) {
  using Vec = at::vec::Vectorized<T>;
  constexpr int64_t vector_size = Vec::size();
  constexpr int64_t chunk_size = moments::kChunkSize;
  const int64_t vector_count = N / vector_size;
  const int64_t chunks = (vector_count + chunk_size - 1) / chunk_size;
  const int64_t depth = at::native::utils::CeilLog2(chunks);
  const int64_t autoselected = depth <= 4 ? 4 : depth <= 8 ? 8 :
                              depth <= 16 ? 16 : depth <= 32 ? 32 : 64;

  std::vector<T> values(static_cast<size_t>(N));
  for (int64_t i = 0; i < N; ++i) {
    const int64_t numerator = (i * 37 % 1009) - 504;
    values[static_cast<size_t>(i)] = static_cast<T>(numerator) / T(256);
  }
  const Reference ref = reference(values);

  std::cout << "{\"dtype\":\"" << dtype << "\",\"n\":" << N
            << ",\"vector_size\":" << vector_size
            << ",\"chunk_size\":" << chunk_size
            << ",\"chunks\":" << chunks << ",\"depth\":" << depth
            << ",\"autoselected_specialization\":" << autoselected
            << ",\"reference\":{\"mean\":";
  number(ref.mean);
  std::cout << ",\"variance\":";
  number(ref.variance);
  std::cout << "},\"variants\":[";

  bool first = true;
  result("auto", autoselected, moments::RowwiseMoments<T>(values.data(), N),
         ref, first);
  if (depth <= 4) {
    result("impl4", 4, moments::RowwiseMomentsImpl<T, 4>(values.data(), N),
           ref, first);
  }
  if (depth <= 8) {
    result("impl8", 8, moments::RowwiseMomentsImpl<T, 8>(values.data(), N),
           ref, first);
  }
  if (depth <= 16) {
    result("impl16", 16, moments::RowwiseMomentsImpl<T, 16>(values.data(), N),
           ref, first);
  }
  std::cout << "]}\n";
}

int main() {
  static_assert(at::vec::Vectorized<float>::size() == 8,
                "This probe requires the AVX2 float specialization.");
  static_assert(at::vec::Vectorized<double>::size() == 4,
                "This probe requires the AVX2 double specialization.");
  std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
  for (int64_t n : {1024LL, 2048LL, 4096LL, 175616LL}) {
    probe<float>("float32", n);
    probe<double>("float64", n);
  }
  return 0;
}
