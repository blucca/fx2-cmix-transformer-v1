// SPDX-License-Identifier: GPL-3.0-only
// Research adapter for Vladimir Ivanov's fx2-cmix-transformer, based on
// fx2-cmix (Kaido Orav / Byron Knoll) and PHDA9 (Alexander Rhatushnyak).
// Reuse the upstream transforms and boundary checks verbatim. Section GC
// drops the unused original main and compression/model functions.
//
// Minimal build, from the repository root:
// clang++-17 -O3 -std=c++17 -march=x86-64-v3 -mrecip=none \
//   -DSEED=923 -DUPDATE_LIMIT=3000 -ffunction-sections -fdata-sections \
//   research/prepare.cpp src/preprocess/preprocessor.cpp \
//   src/preprocess/dictionary.cpp -Wl,--gc-sections -o /YOUR/temp/prepare
// It may also link the normal objects except runner.o, with section GC.
//
// Usage: prepare ENWIK9 DICTIONARY ARTICLE_ORDER EMPTY_OUTPUT_DIRECTORY
// stream.bin: exact pre-arithmetic-coding bytes, including WRT/block headers.
// vocab.bin: 32 bytes; bit (b % 8) of byte (b / 8) marks source byte b.
// article_boundaries.i32: upstream little-endian int32 separator-END offsets.
// metadata.json is written last. All intermediate files stay in OUTDIR;
// the caller can clean them with Python after checking the completed result.

#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <unistd.h>

[[noreturn]] void PrepareFail(const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  std::fprintf(stderr, "prepare: ");
  std::vfprintf(stderr, fmt, args);
  std::fprintf(stderr, "\n");
  va_end(args);
  std::exit(EXIT_FAILURE);
}

// Upstream PHDA9 and BuildEncodedPattern use tmpfile(). Keep those scratch
// files in the caller's output directory too, rather than the system /tmp.
FILE* PrepareTmpfile() {
  char path[] = ".prepare-tmp-XXXXXX";
  int fd = mkstemp(path);
  if (fd < 0) PrepareFail("creating scratch file: %s", std::strerror(errno));
  FILE* file = fdopen(fd, "w+b");
  if (!file) {
    close(fd);
    PrepareFail("opening scratch file: %s", std::strerror(errno));
  }
  return file;
}

#define main hutter_original_main
#define tmpfile PrepareTmpfile
#define Fail PrepareFail
#include "../src/runner.cpp"
#undef Fail
#undef tmpfile
#undef main

namespace fs = std::filesystem;

fs::path PrepareAbsolute(const char* value) {
  std::error_code ec;
  fs::path path = fs::absolute(value, ec);
  if (ec) PrepareFail("resolving %s: %s", value, ec.message().c_str());
  return path;
}

void PrepareCopy(const fs::path& source, const char* destination) {
  std::error_code ec;
  fs::copy_file(source, destination, ec);
  if (ec) PrepareFail("copying %s: %s", source.c_str(), ec.message().c_str());
}

void PrepareStage(const char* name) {
  std::printf("prepare: %s\n", name);
  std::fflush(stdout);
}

int main(int argc, char** argv) {
  if (argc != 5) {
    PrepareFail("usage: %s ENWIK9 DICTIONARY ARTICLE_ORDER EMPTY_OUTDIR", argv[0]);
  }
  const fs::path input = PrepareAbsolute(argv[1]);
  const fs::path dictionary_path = PrepareAbsolute(argv[2]);
  const fs::path article_order = PrepareAbsolute(argv[3]);
  const fs::path output = PrepareAbsolute(argv[4]);
  RequireReadableFile("enwik9", input.string());
  RequireReadableFile("dictionary", dictionary_path.string());
  RequireReadableFile("article order", article_order.string());
  const auto input_bytes = getFileSize(input.string());
  if (input_bytes != 1000000000ULL) {
    PrepareFail("enwik9 has %zu bytes; expected 1000000000", input_bytes);
  }
  VerifyDictionary(dictionary_path.c_str());

  std::error_code ec;
  const bool exists = fs::exists(output, ec);
  if (ec) PrepareFail("checking OUTDIR: %s", ec.message().c_str());
  if (exists) {
    const bool empty = fs::is_empty(output, ec);
    if (ec || !empty) PrepareFail("OUTDIR must be a new or empty directory");
  } else {
    fs::create_directories(output, ec);
    if (ec) PrepareFail("creating OUTDIR: %s", ec.message().c_str());
  }
  if (chdir(output.c_str()) != 0) {
    PrepareFail("entering OUTDIR: %s", std::strerror(errno));
  }
  PrepareCopy(dictionary_path, ".dict");
  PrepareCopy(article_order, ".new_article_order");
  srand(SEED);

  PrepareStage("split complete enwik9");
  split4Comp(input.c_str());
  const auto main_bytes = getFileSize(".main");
  if (getFileSize(".intro") + main_bytes + getFileSize(".coda") != input_bytes) {
    PrepareFail("split output lengths do not cover enwik9");
  }

  PrepareStage("original article order");
  reorder();
  if (getFileSize(".main_reordered") != main_bytes) {
    PrepareFail("article reorder changed the main-region byte count");
  }

  PrepareStage("PHDA9 and original region concatenation");
  phda9_prepr();
  cat(".main_phda9prepr", ".intro", "un1");
  cat("un1", ".coda", ".ready4cmix");
  const auto ready_bytes = getFileSize(".ready4cmix");
  if (ready_bytes != getFileSize(".main_phda9prepr") +
      getFileSize(".intro") + getFileSize(".coda")) {
    PrepareFail("PHDA9 region concatenation is incomplete");
  }

  PrepareStage("original block detection and WRT dictionary transform");
  FILE* in = fopen(".ready4cmix", "rb");
  FILE* out = fopen("stream.bin", "wb");
  FILE* dictionary = fopen(".dict", "rb");
  if (!in || !out || !dictionary) PrepareFail("opening WRT files failed");
  preprocessor::Encode(in, out, ready_bytes, "wrt-scratch", dictionary);
  const bool io_error = ferror(in) || ferror(out);
  const int input_close = fclose(in);
  const int output_close = fclose(out);
  if (io_error || input_close || output_close) PrepareFail("WRT I/O failed");

  PrepareStage("full-stream vocabulary and original article boundaries");
  const auto stream_bytes = getFileSize("stream.bin");
  std::ifstream stream("stream.bin", std::ios::binary);
  if (!stream || stream_bytes < kMinVocabFileSize) {
    PrepareFail("preprocessed stream is missing or truncated");
  }
  std::vector<bool> vocab(256, false);
  ExtractVocab(stream_bytes, &stream, &vocab);
  if (!stream) PrepareFail("reading full-stream vocabulary failed");
  stream.close();
  const auto vocab_size = std::count(vocab.begin(), vocab.end(), true);
  if (vocab_size != 205) {
    PrepareFail("full-stream vocabulary has %lld symbols; expected 205",
        static_cast<long long>(vocab_size));
  }
  unsigned char mask[32] = {};
  for (unsigned int byte = 0; byte < 256; ++byte) {
    if (vocab[byte]) mask[byte / 8] |= 1u << (byte % 8);
  }
  out = fopen("vocab.bin", "wb");
  if (!out || fwrite(mask, 1, sizeof(mask), out) != sizeof(mask)) {
    PrepareFail("writing vocab.bin failed");
  }
  if (fclose(out)) PrepareFail("closing vocab.bin failed");

  // This upstream function checks the complete WRT marker/index contract,
  // including the first marker at 28, first separator end at 43, and EOF.
  SaveArticleBoundaries("stream.bin", stream_bytes, dictionary, vocab,
      "article_boundaries.i32");
  fclose(dictionary);
  const auto boundary_bytes = getFileSize("article_boundaries.i32");
  if (boundary_bytes % sizeof(int32_t) || boundary_bytes < 3 * sizeof(int32_t)) {
    PrepareFail("article boundary output is incomplete");
  }

  out = fopen("metadata.json", "wb");
  if (!out) PrepareFail("opening metadata.json failed");
  const int written = fprintf(out,
      "{\n"
      "  \"input_bytes\": %llu,\n"
      "  \"stream_bytes\": %llu,\n"
      "  \"vocab_size\": 205,\n"
      "  \"vocab_bytes\": 32,\n"
      "  \"vocab_encoding\": \"lsb-first membership mask over byte values 0..255\",\n"
      "  \"boundary_count\": %llu,\n"
      "  \"boundary_encoding\": \"little-endian int32: 0, separator ends, EOF\",\n"
      "  \"first_separator_end\": 43\n"
      "}\n",
      static_cast<unsigned long long>(input_bytes),
      static_cast<unsigned long long>(stream_bytes),
      static_cast<unsigned long long>(boundary_bytes / sizeof(int32_t)));
  const int metadata_close = fclose(out);
  if (written < 0 || metadata_close) PrepareFail("writing metadata.json failed");
  PrepareStage("complete: stream.bin, vocab.bin, article_boundaries.i32, metadata.json");
  return 0;
}
