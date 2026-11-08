// Fixed full12 W0 census only: reuse the original decoder; change no weights.
// Pair each row at columns (0,1), (2,3), ...; retain the old model for tails.
#include "weights_io.h"
#include "weights-static15.h"
#include <array>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

template<size_t N> double entropy(const std::array<uint64_t,N>& c) {
  const double n=std::accumulate(c.begin(),c.end(),uint64_t{0});
  double b=0; for(auto v:c) if(v) b+=v*std::log2(n/v); return b;
}
template<size_t N> std::array<uint64_t,N> frequencies(const std::array<uint64_t,N>& c) {
  // Exactly the existing static15 largest-remainder rule, with N=225.
  const auto n=std::accumulate(c.begin(),c.end(),uint64_t{0}); assert(n>0);
  std::array<uint64_t,N> f{},r{}; uint64_t used=0;
  for(size_t s=0;s<N;++s) { f[s]=1+c[s]*(4096-N)/n; r[s]=c[s]*(4096-N)%n; used+=f[s]; }
  while(used<4096) { const auto s=std::max_element(r.begin(),r.end())-r.begin(); ++f[s]; ++used; r[s]=0; }
  assert(std::accumulate(f.begin(),f.end(),uint64_t{0})==4096);
  for(auto v:f) assert(v>0); return f;
}
template<size_t N> double cost(const std::array<uint64_t,N>& c,const std::array<uint64_t,N>& f,double total) {
  double b=0; for(size_t i=0;i<N;++i) if(c[i]) b+=c[i]*std::log2(total/f[i]); return b;
}
struct Scan {
  Counts all{},left{},right{},tail{};
  std::array<uint64_t,225> joint{};
  uint64_t pairs=0,tails=0;
};
Scan scan(const int8_t* q,size_t rows,size_t cols) {
  assert(rows>0 && cols>0); Scan s;
  for(size_t k=0;k<rows*cols;++k) { const int v=q[k]+7; assert(v>=0 && v<15); ++s.all[v]; }
  for(size_t r=0;r<rows;++r) {
    for(size_t c=0;c+1<cols;c+=2) { ++s.joint[(q[r*cols+c]+7)*15+q[r*cols+c+1]+7]; ++s.pairs; }
    if(cols%2) { ++s.tail[q[r*cols+cols-1]+7]; ++s.tails; }
  }
  for(size_t a=0;a<15;++a) for(size_t b=0;b<15;++b) { s.left[a]+=s.joint[a*15+b]; s.right[b]+=s.joint[a*15+b]; }
  assert(2*s.pairs+s.tails==rows*cols);
  for(size_t i=0;i<15;++i) assert(s.all[i]==s.left[i]+s.right[i]+s.tail[i]);
  return s;
}
template<size_t N> void json_counts(const std::array<uint64_t,N>& c) {
  std::cout<<'['; for(size_t i=0;i<N;++i) std::cout<<(i?",":"")<<c[i]; std::cout<<']';
}
void check() {
  const int8_t independent[]={-7,-7,-7,-6,-6,-7,-6,-6};
  const auto a=scan(independent,4,2);
  assert(std::abs(entropy(a.all)-entropy(a.joint))<1e-12);
  const int8_t correlated[]={-7,-7,-7,-7,-6,-6,-6,-6};
  const auto b=scan(correlated,4,2);
  assert(std::abs(entropy(b.all)-entropy(b.joint)-4)<1e-12);
  const int8_t odd[]={-7,-7,-6,-6,-6,-7};
  const auto c=scan(odd,2,3); assert(c.pairs==2 && c.tails==2);
  const int8_t tail_only[]={-7,-6};
  const auto d=scan(tail_only,2,1); assert(d.pairs==0 && d.tails==2);
  assert(frequencies(a.all)==static15_frequencies(a.all));
  const auto f=frequencies(a.joint); assert(cost(a.joint,f,4096)>=entropy(a.joint));
}
int main(int argc,char** argv) {
  check(); if(argc!=2) { std::cerr<<"usage: pair-census fixed-full12-W0.tfwc3\n"; return 2; }
  const auto wf=fx2::WeightsFile::load_compressed(argv[1]);
  assert(wf.tensors.size()==434);
  std::vector<std::string> names; for(const auto& e:wf.tensors) if(e.second.dtype==fx2::DT_I8) names.push_back(e.first);
  std::sort(names.begin(),names.end()); assert(names.size()==111);
  uint64_t elements=0,np=0,nt=0,table_delta=0;
  double unary=0,pair=0,old_static=0,new_static=0;
  std::cout<<std::setprecision(16)<<"{\"checks\":\"passed\",\"matrices\":[";
  bool first=true;
  for(const auto& name:names) {
    const auto& t=wf.get(name); assert(t.shape.size()==2 && t.numel==size_t(t.shape[0])*t.shape[1]);
    const auto s=scan(t.i8(),t.shape[0],t.shape[1]); const auto f15=static15_frequencies(s.all);
    const double h0=entropy(s.all),n=t.numel;
    const double h1=entropy(s.joint)+cost(s.tail,s.all,n);
    const double b0=cost(s.all,f15,4096);
    const double b1=s.pairs ? cost(s.joint,frequencies(s.joint),4096)+cost(s.tail,f15,4096) : b0;
    // A literal-cost scenario, not a file-size measurement. Keep old tail table.
    const uint64_t extra=s.pairs ? 448+(s.tails?28:0)-28 : 0;
    elements+=t.numel; np+=s.pairs; nt+=s.tails; table_delta+=extra;
    unary+=h0; pair+=h1; old_static+=b0; new_static+=b1;
    if(!first) std::cout<<','; first=false;
    std::cout<<"\n{\"name\":"<<std::quoted(name)<<",\"rows\":"<<t.shape[0]<<",\"cols\":"<<t.shape[1]
      <<",\"elements\":"<<t.numel<<",\"pairs\":"<<s.pairs<<",\"tails\":"<<s.tails
      <<",\"unary_bits\":"<<h0<<",\"pair_oracle_bits_with_old_tail_model\":"<<h1
      <<",\"old_static4096_data_bits\":"<<b0<<",\"pair_static4096_data_bits\":"<<b1
      <<",\"literal_table_delta_bytes\":"<<extra<<",\"histogram\":"; json_counts(s.all);
    std::cout<<",\"joint\":"; json_counts(s.joint); std::cout<<",\"left\":"; json_counts(s.left);
    std::cout<<",\"right\":"; json_counts(s.right); std::cout<<",\"tail\":"; json_counts(s.tail); std::cout<<'}';
  }
  assert(elements==5868864 && np==2934336 && nt==192);
  std::cout<<"\n],\"totals\":{\"elements\":"<<elements<<",\"pairs\":"<<np<<",\"tails\":"<<nt
    <<",\"unary_bits\":"<<unary<<",\"pair_oracle_bits_with_old_tail_model\":"<<pair
    <<",\"old_static4096_data_bits\":"<<old_static<<",\"pair_static4096_data_bits\":"<<new_static
    <<",\"literal_table_delta_bytes\":"<<table_delta<<"}}\n";
}
