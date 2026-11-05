// One causal PPMd pass over the exact prepared stream, retaining whole pieces.
// Link this main with the original PPMd/model objects and the updated predictor.
// PLAN: sorted, disjoint "START END train|validation" lines; end is exclusive.
#include "predictor.h"
#include <array>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <string>
#include <sys/resource.h>
#include <sys/stat.h>
#include <vector>

using U64 = unsigned long long;
struct SampleInterval { U64 begin, end; bool train; };

static U64 Size(const std::string& path) {
  struct stat s{};
  if (stat(path.c_str(), &s) || !S_ISREG(s.st_mode)) Fail("input stat failed: %s", path.c_str());
  return s.st_size;
}
static void Fresh(const std::string& path) {
  struct stat s{};
  if (!lstat(path.c_str(), &s) || errno != ENOENT) Fail("output needs a fresh path: %s", path.c_str());
}
static void Progress(U64 processed, U64 train, U64 validation, const char* status) {
  rusage u{}; getrusage(RUSAGE_SELF, &u);
  double cpu = u.ru_utime.tv_sec + u.ru_utime.tv_usec * 1e-6 +
               u.ru_stime.tv_sec + u.ru_stime.tv_usec * 1e-6;
  std::printf("CAUSAL_RESULT {\"status\":\"%s\",\"processed_bytes\":%llu,"
      "\"train_bytes\":%llu,\"validation_bytes\":%llu,\"cpu_seconds\":%.6f,"
      "\"max_rss_kib\":%ld}\n", status, processed, train, validation, cpu, u.ru_maxrss);
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
  while (plan >> begin >> end >> role) {
    if (begin<previous || end<=begin+1 || end>length || (role!="train" && role!="validation"))
      Fail("invalid sampling interval");
    intervals.push_back({begin,end,role=="train"}); previous=end;
  }
  if (!plan.eof() || intervals.empty()) Fail("sampling plan is empty or malformed");
  std::string out=argv[4];
  for (auto name:{"train.u8","train.f16","validation.u8","validation.f16"}) Fresh(out+"/"+name);
  std::ofstream train_tokens(out+"/train.u8",std::ios::binary), val_tokens(out+"/validation.u8",std::ios::binary);
  if (!train_tokens || !val_tokens) Fail("token output open failed");
  U64 processed=0,train_bytes=0,val_bytes=0; size_t next=0;
  {
    HalfFileWriter train_prior(out+"/train.f16"),val_prior(out+"/validation.f16");
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
          if (interval.train) { train_tokens.put(token); predictor.WritePpmdRow(train_prior); ++train_bytes; }
          else { val_tokens.put(token); predictor.WritePpmdRow(val_prior); ++val_bytes; }
          if (processed+1==interval.end) ++next;
        }
      }
      if (!train_tokens || !val_tokens) Fail("token output write failed");
      Progress(processed,train_bytes,val_bytes,"running");
    }
    if (!input.eof() || processed!=length || next!=intervals.size()) Fail("capture input/plan count mismatch");
  } // Close and flush both original half writers before checking their sizes.
  train_tokens.close(); val_tokens.close();
  if (Size(out+"/train.u8")!=train_bytes || Size(out+"/train.f16")!=410*train_bytes ||
      Size(out+"/validation.u8")!=val_bytes || Size(out+"/validation.f16")!=410*val_bytes)
    Fail("capture output count mismatch");
  Progress(processed,train_bytes,val_bytes,"passed");
}
