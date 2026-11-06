// One causal PPMd pass over the exact prepared stream, retaining whole pieces.
// Link this main with the original PPMd/model objects and the updated predictor.
// PLAN: sorted, disjoint "START END train|validation|audit" lines; end is exclusive.
// Optional audit rows share the same causal pass, with no training-state change.
#include "predictor.h"
#include <array>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <sys/resource.h>
#include <sys/stat.h>
#include <vector>

using U64 = unsigned long long;
struct SampleInterval { U64 begin, end; unsigned role; };

static U64 Size(const std::string& path) {
  struct stat s{};
  if (stat(path.c_str(), &s) || !S_ISREG(s.st_mode)) Fail("input stat failed: %s", path.c_str());
  return s.st_size;
}
static void Fresh(const std::string& path) {
  struct stat s{};
  if (!lstat(path.c_str(), &s) || errno != ENOENT) Fail("output needs a fresh path: %s", path.c_str());
}
static void Progress(U64 processed, const std::array<U64,3>& saved, const char* status) {
  rusage u{}; getrusage(RUSAGE_SELF, &u);
  double cpu = u.ru_utime.tv_sec + u.ru_utime.tv_usec * 1e-6 +
               u.ru_stime.tv_sec + u.ru_stime.tv_usec * 1e-6;
  std::printf("CAUSAL_RESULT {\"status\":\"%s\",\"processed_bytes\":%llu,"
      "\"train_bytes\":%llu,\"validation_bytes\":%llu,\"audit_bytes\":%llu,\"cpu_seconds\":%.6f,"
      "\"max_rss_kib\":%ld}\n", status, processed, saved[0], saved[1], saved[2], cpu, u.ru_maxrss);
  std::fflush(stdout);
}

int main(int argc, char** argv) {
  if (argc != 5) { std::fprintf(stderr,"usage: %s STREAM VOCAB PLAN OUTDIR\n",argv[0]); return 2; }
  const U64 length = Size(argv[1]);
  if (Size(argv[2]) != 32 || length < 2) Fail("invalid input/vocabulary length");
  std::ifstream vf(argv[2], std::ios::binary), plan(argv[3]), input(argv[1], std::ios::binary);
  std::array<unsigned char,32> packed{}; vf.read(reinterpret_cast<char*>(packed.data()),32);
  std::vector<bool> vocab(256,false); std::array<int,256> index{}; unsigned count=0;
  for (unsigned b=0;b<256;++b) { vocab[b]=(packed[b/8]>>(b%8))&1; index[b]=vocab[b]?count++:-1; }
  if (!vf || !plan || !input || count!=205) Fail("open/vocabulary failure");
  std::vector<SampleInterval> intervals; U64 begin,end,previous=0; std::string role;
  std::array<bool,3> enabled{true,true,false};
  while (plan >> begin >> end >> role) {
    if (begin<previous || end<=begin+1 || end>length ||
        (role!="train" && role!="validation" && role!="audit"))
      Fail("invalid sampling interval");
    unsigned r=role=="train"?0:role=="validation"?1:2;
    intervals.push_back({begin,end,r}); enabled[r]=true; previous=end;
  }
  if (!plan.eof() || intervals.empty()) Fail("sampling plan is empty or malformed");
  std::string out=argv[4];
  const std::array<std::string,3> names{"train","validation","audit"};
  std::array<std::ofstream,3> tokens;
  for (unsigned r=0;r<3;++r) if (enabled[r]) {
    for (const auto& ext:{".u8",".f16"}) Fresh(out+"/"+names[r]+ext);
    tokens[r].open(out+"/"+names[r]+".u8",std::ios::binary);
    if (!tokens[r]) Fail("token output open failed");
  }
  U64 processed=0; std::array<U64,3> saved{}; size_t next=0;
  {
    std::array<std::unique_ptr<HalfFileWriter>,3> priors;
    for (unsigned r=0;r<3;++r) if (enabled[r])
      priors[r]=std::make_unique<HalfFileWriter>(out+"/"+names[r]+".f16");
    PredictorOptions options; options.ppmd_only=true; options.num_input_bytes=length;
    std::srand(923); Predictor predictor(vocab,options);
    std::array<unsigned char,1<<20> buffer{};
    while (input.read(reinterpret_cast<char*>(buffer.data()),buffer.size()) || input.gcount()) {
      const auto n=input.gcount();
      for (std::streamsize j=0;j<n;++j,++processed) {
        unsigned byte=buffer[j]; if (index[byte]<0) Fail("invalid source byte at %llu",processed);
        for (int bit=7;bit>=0;--bit) { predictor.Predict(); predictor.Perceive((byte>>bit)&1); }
        if (next<intervals.size() && processed>=intervals[next].begin) {
          const auto& interval=intervals[next]; const char token=static_cast<char>(index[byte]);
          const auto r=interval.role;
          tokens[r].put(token); predictor.WritePpmdRow(*priors[r]); ++saved[r];
          if (processed+1==interval.end) ++next;
        }
      }
      for (unsigned r=0;r<3;++r) if (enabled[r] && !tokens[r]) Fail("token output write failed");
      Progress(processed,saved,"running");
    }
    if (!input.eof() || processed!=length || next!=intervals.size()) Fail("capture input/plan count mismatch");
  } // Flush every original half writer before checking its exact output length.
  for (unsigned r=0;r<3;++r) if (enabled[r]) {
    tokens[r].close();
    if (!tokens[r] || Size(out+"/"+names[r]+".u8")!=saved[r] ||
        Size(out+"/"+names[r]+".f16")!=410*saved[r]) Fail("capture output count mismatch");
  }
  Progress(processed,saved,"passed");
}
