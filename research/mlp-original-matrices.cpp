#include "weights_io.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <vector>
int main(int argc,char** argv) {
  assert(argc==3);
  auto wf=fx2::WeightsFile::load_compressed(argv[1]);
  assert(wf.tensors.size()==434);
  FILE* out=std::fopen(argv[2],"wb");assert(out);
  std::cout<<"["; bool first=true;
  for(int l=0;l<12;++l) for(auto kind:{"up","down"}) {
    std::string name="blocks."+std::to_string(l)+".mlp."+kind+".weight";
    const auto& q=wf.get(name+".q"); const auto& s=wf.get(name+".scale");
    assert(q.dtype==fx2::DT_I8 && s.dtype==fx2::DT_BF16 && q.shape.size()==2);
    assert(s.numel==q.shape[0]);
    std::vector<float> w(q.numel); std::vector<size_t> counts(15);
    for(size_t k=0;k<q.numel;++k) {
      int v=q.i8()[k];assert(v>=-7 && v<=7);++counts[v+7];
      float scale=fx2::bf16_to_f32(s.bf16_bits()[k/q.shape[1]]);
      assert(std::isfinite(scale) && scale!=0);
      w[k]=v*scale;assert(std::isfinite(w[k]));
    }
    assert(std::fwrite(w.data(),sizeof(float),w.size(),out)==w.size());
    if(!first)std::cout<<",";first=false;
    std::cout<<"{\"name\":"<<std::quoted(name+".q")<<",\"shape\":["<<q.shape[0]<<","<<q.shape[1]<<"],\"histogram\":[";
    for(int k=0;k<15;++k)std::cout<<(k?",":"")<<counts[k];
    std::cout<<"]}";
  }
  assert(std::fclose(out)==0);std::cout<<"]\n";
}
