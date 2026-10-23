// Development codelength / round-trip probe on the exact preprocessed stream.
// Uses the upstream predictor, arithmetic coder, dictionary pretraining and weights.
// GPL-3.0-or-later; original components retain their authors' notices.
#include "predictor.h"
#include "coder/encoder.h"
#include "coder/decoder.h"
#include "preprocess/preprocessor.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <sys/resource.h>

int main(int argc, char** argv) {
  if (argc != 9) {
    std::fprintf(stderr, "usage: probe encode|decode INPUT OUTPUT VOCAB DICTIONARY WEIGHTS OFFSET LENGTH\n");
    return 2;
  }
  const bool encode = std::string(argv[1]) == "encode";
  if (!encode && std::string(argv[1]) != "decode") return 2;
  const auto offset = std::stoull(argv[7]), length = std::stoull(argv[8]);
  if (std::string(argv[2]) == argv[3]) Fail("input and output must differ");
  if (std::ifstream(argv[3]).good()) Fail("probe output already exists; use a fresh path");
  std::ifstream input(argv[2], std::ios::binary);
  std::ifstream vf(argv[4], std::ios::binary);
  if (!input || !vf) Fail("failed to open probe files");
  std::vector<bool> vocab(256, false);
  unsigned count = 0;
  for (unsigned i = 0; i != 32; ++i) {
    int c = vf.get();
    if (c == EOF) Fail("vocab must contain exactly 32 bytes");
    for (unsigned j = 0; j != 8; ++j) {
      vocab[i * 8 + j] = (c >> j) & 1;
      count += vocab[i * 8 + j];
    }
  }
  if (vf.get() != EOF || count != 205) Fail("expected fixed full-stream 205-byte vocabulary");
  if (encode) {
    input.seekg(0, std::ios::end);
    const auto size = static_cast<unsigned long long>(input.tellg());
    if (offset > size || length > size - offset) Fail("slice exceeds input");
    input.seekg(offset);
  }
  std::ofstream output(argv[3], std::ios::binary);
  if (!output) Fail("failed to open probe output");
  srand(923);
  PredictorOptions options;
  options.transformer_weights = argv[6];
  options.num_input_bytes = length;
  const auto started = std::chrono::steady_clock::now();
  Predictor p(vocab, options);
  FILE* dictionary = fopen(argv[5], "rb");
  if (!dictionary) Fail("cannot open dictionary");
  preprocessor::Pretrain(&p, dictionary);
  fclose(dictionary);
  const auto coding_started = std::chrono::steady_clock::now();
  if (encode) {
    Encoder coder(&output, &p);
    for (unsigned long long i = 0; i != length; ++i) {
      const int byte = input.get();
      if (byte == EOF) Fail("short probe input");
      for (int j = 7; j >= 0; --j) coder.Encode((byte >> j) & 1);
      if ((i + 1) % 262144 == 0) {
        std::fprintf(stderr, "probe bytes=%llu coded_bytes=%zu\n", i + 1, coder.OutputSize());
        std::fflush(stderr);
      }
    }
    coder.Flush();
  } else {
    Decoder coder(&input, &p);
    for (unsigned long long i = 0; i != length; ++i) {
      int byte = 1;
      while (byte < 256) byte += byte + coder.Decode();
      output.put(static_cast<char>(byte));
    }
  }
  output.flush();
  if (!output) Fail("probe output write failed");
  const auto finished = std::chrono::steady_clock::now();
  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  std::printf("\nPROBE_RESULT {\"mode\":\"%s\",\"offset\":%llu,\"input_bytes\":%llu,"
              "\"output_bytes\":%llu,\"seconds\":%.6f,\"coding_seconds\":%.6f,"
              "\"cpu_seconds\":%.6f,\"max_rss_kib\":%ld}\n",
              encode ? "encode" : "decode", offset, length,
              static_cast<unsigned long long>(output.tellp()),
              std::chrono::duration<double>(finished - started).count(),
              std::chrono::duration<double>(finished - coding_started).count(),
              usage.ru_utime.tv_sec + usage.ru_utime.tv_usec * 1e-6 +
              usage.ru_stime.tv_sec + usage.ru_stime.tv_usec * 1e-6, usage.ru_maxrss);
}
