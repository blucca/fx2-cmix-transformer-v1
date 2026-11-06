// Read-only Hutter trace interval accounting. GPL-3.0-only.
// Build with strict FP and -I <original trace source>/research.
#include "trace.h"
#ifdef __FAST_MATH__
#error "mixed interval accounting requires strict floating point"
#endif
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using U64 = std::uint64_t;
struct RangeCounter {
  std::uint32_t low = 0, high = 0xffffffff;
  U64 bytes = 0;
  void encode(std::uint32_t p, int bit) {
    const std::uint32_t mid = low + ((high-low)>>16)*p +
        (((high-low)&0xffff)*p>>16);
    if (bit) high=mid; else low=mid+1;
    while (((low^high)&0xff000000)==0) {
      ++bytes; low<<=8; high=(high<<8)+255;
    }
  }
  U64 closed() const { return bytes+1; }
};
U64 size(const char* path) {
  std::ifstream f(path, std::ios::binary|std::ios::ate);
  if (!f) throw std::runtime_error("file open failed");
  return U64(f.tellg());
}
std::vector<unsigned char> read(const char* path) {
  std::vector<unsigned char> data(size(path));
  std::ifstream f(path,std::ios::binary);
  if (!f.read(reinterpret_cast<char*>(data.data()),data.size()))
    throw std::runtime_error("file read failed");
  return data;
}
struct Totals {
  U64 bytes=0, overrides=0;
  double neural=0, pre_sse=0, final=0;
};
double nll(double p, int bit) {
  const double q=bit?p:1-p;
  if (!(q>0) || !(q<=1)) throw std::runtime_error("observed trace probability failed");
  return -std::log2(q);
}
double logit_nll(double z, int bit) {
  const double x=bit?-z:z;
  return (std::max(x,0.0)+std::log1p(std::exp(-std::abs(x))))/std::log(2.0);
}
int main(int argc,char** argv) try {
  if (argc!=6) {
    std::cerr << "usage: mixed-interval TRACE SCORE_TOKENS_U8 VOCAB SCORE_BEGIN EXPECTED_ENCODED_BYTES\n";
    return 2;
  }
  const U64 trace_bytes=size(argv[1]), begin=std::stoull(argv[4]);
  const U64 expected=std::stoull(argv[5]);
  if (trace_bytes%(8*sizeof(HutterTraceRecord))) throw std::runtime_error("trace layout failed");
  const U64 full_bytes=trace_bytes/(8*sizeof(HutterTraceRecord));
  const auto tokens=read(argv[2]), vocab=read(argv[3]);
  if (vocab.size()!=32 || tokens.empty() || begin>full_bytes || tokens.size()>full_bytes-begin)
    throw std::runtime_error("score geometry failed");
  const U64 end=begin+tokens.size();
  std::array<unsigned char,205> token_to_byte{};
  unsigned nv=0;
  for (unsigned i=0;i<256;++i) if ((vocab[i/8]>>(i%8))&1) {
    if (nv>=205) throw std::runtime_error("vocab count failed");
    token_to_byte[nv++]=i;
  }
  if (nv!=205) throw std::runtime_error("vocab count failed");
  constexpr unsigned char sep[15]={8,8,0x25,0xac,0x65,0x27,5,8,8,8,8,0x25,0xac,0x68,0x27};
  std::array<unsigned char,15> window; window.fill(255);
  std::vector<bool> target(tokens.size(),true);
  U64 in_piece=0,pieces=0;
  for (U64 i=0;i<tokens.size();++i) {
    if (tokens[i]>=205) throw std::runtime_error("token range failed");
    if (in_piece==0) { target[i]=false; ++pieces; }
    std::memmove(window.data(),window.data()+1,14);window.back()=tokens[i];
    ++in_piece;
    if (in_piece==(1ULL<<17) || std::memcmp(window.data(),sep,15)==0) in_piece=0;
  }
  std::ifstream f(argv[1],std::ios::binary);
  std::array<HutterTraceRecord,8*4096> block;
  RangeCounter counter;
  U64 position=0,begin_closed=0,end_closed=0;
  Totals all,nn;
  while (f) {
    f.read(reinterpret_cast<char*>(block.data()),sizeof(block));
    const auto got=f.gcount();
    if (got%(8*sizeof(HutterTraceRecord))) throw std::runtime_error("trace block layout failed");
    for (U64 k=0;k<U64(got)/sizeof(HutterTraceRecord);k+=8,++position) {
      if (position==begin) begin_closed=counter.closed();
      const bool selected=position>=begin && position<end;
      const bool is_target=selected && target[position-begin];
      unsigned source_byte=0;
      for (unsigned b=0;b<8;++b) {
        const auto& row=block[k+b];
        if (!row.probability || row.flags>3 || !std::isfinite(row.neural) ||
            !std::isfinite(row.mixer) || row.neural<0 || row.neural>1)
          throw std::runtime_error("trace record failed");
        const int bit=row.flags&1;
        source_byte=2*source_byte+bit;
        counter.encode(row.probability,bit);
        if (selected) {
          const double l=nll(row.probability/65536.0,bit);
          const double n=nll(double(row.neural),bit);
          const double m=(row.flags&2)?l:logit_nll(double(row.mixer),bit);
          all.final+=l;all.neural+=n;all.pre_sse+=m;all.overrides+=(row.flags>>1)&1;
          if (is_target) { nn.final+=l;nn.neural+=n;nn.pre_sse+=m;nn.overrides+=(row.flags>>1)&1; }
        }
      }
      if (selected) {
        if (source_byte!=token_to_byte[tokens[position-begin]])
          throw std::runtime_error("trace/token byte alignment failed");
        ++all.bytes;nn.bytes+=is_target;
      }
      if (position+1==end) end_closed=counter.closed();
    }
  }
  if (position!=full_bytes || counter.closed()!=expected || all.bytes!=tokens.size() ||
      nn.bytes!=tokens.size()-pieces || !begin_closed || !end_closed)
    throw std::runtime_error("range/count golden failed");
  auto print=[&](const char* name,const Totals& t) {
    std::cout << ",\"" << name << "\":{\"bytes\":" << t.bytes << ",\"neural_bits\":" << t.neural
      << ",\"pre_sse_logit_bits\":" << t.pre_sse << ",\"final_u16_bits\":" << t.final
      << ",\"override_bits\":" << t.overrides << "}";
  };
  std::cout << std::setprecision(17) << "{\"status\":\"passed\",\"full_input_bytes\":" << full_bytes
    << ",\"full_coded_bytes\":" << counter.closed() << ",\"score_begin\":" << begin
    << ",\"score_end\":" << end << ",\"pieces\":" << pieces
    << ",\"interval_coded_bytes\":" << end_closed-begin_closed
    << ",\"begin_closed_bytes\":" << begin_closed << ",\"end_closed_bytes\":" << end_closed
    << ",\"interval_range_residual_bytes\":" << double(end_closed-begin_closed)-all.final/8;
  print("all",all);print("neural_targets",nn);std::cout << "}\n";
} catch (const std::exception& e) { std::cerr << e.what() << "\n";return 1; }
