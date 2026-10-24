// SPDX-License-Identifier: GPL-3.0-only
// One convex likelihood-capacity diagnostic; no candidate coder / trace changes.
// Compile normally, without fast-math. Input: completed base+PPMd trace directories.
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
struct Base { uint16_t p,flags; float n,fx,m; };
struct Side { float p,h,e; uint16_t depth,flags; };
static_assert(sizeof(Base)==16 && sizeof(Side)==16);
struct Row { double a,b; }; // N/B-1, P/B-1 for a complete byte.
struct Eval { double value=0,g1=0,g2=0,h11=0,h12=0,h22=0; };
Eval calc(const std::vector<Row>& rows,double n,double p,bool second=false) {
  Eval r;
  for(const auto& x:rows) {
    double d=1+n*x.a+p*x.b;
    if(!(d>0)) throw std::runtime_error("invalid mixture");
    r.value+=std::log(d);
    double a=x.a/d,b=x.b/d;
    r.g1+=a; r.g2+=b;
    if(second) {r.h11+=a*a; r.h12+=a*b; r.h22+=b*b;}
  }
  return r;
}
double edge(const std::vector<Row>& rows,int kind) {
  auto grad=[&](double t) {
    double r=0;
    for(const auto& x:rows) {
      double a=kind==0?x.a:(kind==1?x.b:x.b-x.a);
      double d=(kind==2?1+x.a:1)+t*a;
      r+=a/d;
    }
    return r;
  };
  if(grad(0)<=0)return 0;
  if(grad(1)>=0)return 1;
  double lo=0,hi=1;
  for(int i=0;i<42;i++) {double mid=(lo+hi)/2;if(grad(mid)>0)lo=mid;else hi=mid;}
  return (lo+hi)/2;
}
struct Fit {double n=0,p=0,value=0,kkt=0;int iterations=0;};
Fit fit(const std::vector<Row>& rows) {
  Fit best;
  for(int j=0;j<3;j++) {
    double t=edge(rows,j),n=j==0?t:(j==2?1-t:0),p=j==1||j==2?t:0;
    Eval e=calc(rows,n,p);
    if(e.value>best.value)best={n,p,e.value};
  }
  // Every edge optimum is known. Newton tests for a better interior optimum.
  double n=1./3,p=1./3;
  for(int i=0;i<40;i++) {
    auto e=calc(rows,n,p,true);
    if(e.value>best.value)best={n,p,e.value,0,i};
    if(std::max(std::abs(e.g1),std::abs(e.g2))/rows.size()<1e-11)break;
    double det=e.h11*e.h22-e.h12*e.h12;
    if(!(det>1e-20))break;
    double dn=(e.h22*e.g1-e.h12*e.g2)/det,dp=(e.h11*e.g2-e.h12*e.g1)/det;
    double step=1.;
    if(dn<0)step=std::min(step,-.995*n/dn);
    if(dp<0)step=std::min(step,-.995*p/dp);
    if(dn+dp>0)step=std::min(step,.995*(1-n-p)/(dn+dp));
    if(step<1e-12)break;
    bool accepted=false;
    for(int k=0;k<15;k++) {
      double nn=n+step*dn,pp=p+step*dp;
      if(nn>=0 && pp>=0 && nn+pp<=1 && calc(rows,nn,pp).value>=e.value-1e-9) {
        n=nn;p=pp;accepted=true;break;
      }
      step*=.5;
    }
    if(!accepted)break;
  }
  auto e=calc(rows,best.n,best.p);
  double weighted=best.n*e.g1+best.p*e.g2;
  std::array<double,3> relative{-weighted,e.g1-weighted,e.g2-weighted};
  std::array<double,3> weights{1-best.n-best.p,best.n,best.p};
  for(int j=0;j<3;j++) best.kkt=std::max(best.kkt,
    (weights[j]>1e-8?std::abs(relative[j]):std::max(0.,relative[j]))/rows.size());
  return best;
}
void check() {
  // The maximizer is symmetric N/P, with no base: sqrt(.9*.1)>.2.
  std::vector<Row> rows{{3.5,-.5},{-.5,3.5}};
  auto f=fit(rows);
  assert(std::abs(f.n-.5)<1e-8 && std::abs(f.p-.5)<1e-8 && f.kkt<1e-8);
  std::vector<Row> dominated{{-.1,-.2},{-.3,-.4}};
  f=fit(dominated);assert(f.n==0&&f.p==0&&f.value==0&&f.kkt==0);
  // A proper Bayesian prefix update gives the full-byte mixture likelihood.
  std::array<double,3> w{.8,.15,.05},l{1,1,1};double product=1;
  for(auto probabilities: {std::array<double,3>{.8,.6,.4},std::array<double,3>{.2,.7,.6}}) {
    double q=0;for(int j=0;j<3;j++)q+=w[j]*probabilities[j];product*=q;
    for(int j=0;j<3;j++){w[j]*=probabilities[j]/q;l[j]*=probabilities[j];}
  }
  assert(std::abs(product-(.8*l[0]+.15*l[1]+.05*l[2]))<1e-14);
}
int main(int argc,char** argv) try {
  check();
  if(argc==2 && std::string(argv[1])=="--self-test") {std::cout<<"ok\n";return 0;}
  if(argc!=3)throw std::runtime_error("usage: budget PREFIX_DIRECTORY Q2_DIRECTORY");
  auto start=std::chrono::steady_clock::now();
  std::vector<Row> rows; rows.reserve(2097152);std::array<size_t,3> ends{0,0,0};
  std::array<double,2> base_bits{0,0},neural_bits{0,0},ppmd_bits{0,0};
  for(int c=0;c<2;c++) {
    std::filesystem::path dir(argv[c+1]);auto a=dir/"trace.bin",b=dir/"ppmd.bin";
    auto bytes=std::filesystem::file_size(a);
    if(bytes!=134217728 || std::filesystem::file_size(b)!=bytes)
      throw std::runtime_error("requires complete fixed 1MiB traces");
    std::ifstream base(a,std::ios::binary),side(b,std::ios::binary);
    std::array<Base,8192> br;std::array<Side,8192> sr;
    size_t total=0;double lb=1,ln=1,lp=1;
    while(base.read(reinterpret_cast<char*>(br.data()),sizeof(br))) {
      if(!side.read(reinterpret_cast<char*>(sr.data()),sizeof(sr)))throw std::runtime_error("truncated sidecar");
      for(size_t j=0;j<br.size();j++,total++) {
        auto& r=br[j];auto& s=sr[j];
        if(!r.p ||r.flags>3||!std::isfinite(r.n)||r.n<0||r.n>1||!std::isfinite(s.p)||s.p<0||s.p>1||s.flags!=(total<8?0:1))throw std::runtime_error("invalid record");
        double p=r.p/65536.,n=std::clamp(double(r.n),1./65536,65535./65536),pp=std::clamp(double(s.p),1./65536,65535./65536);
        if((r.flags&2)||total<8)n=pp=p;
        if(!(r.flags&1)){p=1-p;n=1-n;pp=1-pp;}
        lb*=p;ln*=n;lp*=pp;
        if((total&7)==7) {
          rows.push_back({ln/lb-1,lp/lb-1});
          base_bits[c]-=std::log2(lb);neural_bits[c]-=std::log2(ln);ppmd_bits[c]-=std::log2(lp);
          lb=ln=lp=1;
        }
      }
    }
    if(total!=8388608)throw std::runtime_error("wrong input rows");ends[c+1]=rows.size();
  }
  auto f=fit(rows);double bn=edge(rows,0),bp=edge(rows,1);
  std::cout<<std::setprecision(15)<<"{\"schema\":\"hutter.byte-simplex-capacity.v1\",\"input_bytes\":"<<rows.size()
    <<",\"weights\":{\"base\":"<<1-f.n-f.p<<",\"neural\":"<<f.n<<",\"ppmd\":"<<f.p<<"},\"nll_saved_bytes\":"<<f.value/std::log(2.)/8
    <<",\"kkt_per_byte\":"<<f.kkt<<",\"interior_iterations\":"<<f.iterations
    <<",\"bn_control\":{\"neural_prior\":"<<bn<<",\"nll_saved_bytes\":"<<calc(rows,bn,0).value/std::log(2.)/8<<"}"
    <<",\"bp_control\":{\"ppmd_prior\":"<<bp<<",\"nll_saved_bytes\":"<<calc(rows,0,bp).value/std::log(2.)/8<<"},\"cases\":[";
  for(int c=0;c<2;c++) {
    if(c)std::cout<<',';double gain=0;std::array<double,8> block{};
    for(size_t i=ends[c];i<ends[c+1];i++){double x=std::log2(1+f.n*rows[i].a+f.p*rows[i].b)/8;gain+=x;block[(i-ends[c])/131072]+=x;}
    std::cout<<"{\"case\":\""<<(c==0?"prefix-1m":"article-q2-1m")<<"\",\"base_loss_bits\":"<<base_bits[c]<<",\"neural_loss_bits\":"<<neural_bits[c]<<",\"ppmd_loss_bits\":"<<ppmd_bits[c]<<",\"nll_saved_bytes\":"<<gain<<",\"blocks_128k_saved_bytes\":[";
    for(int k=0;k<8;k++){if(k)std::cout<<',';std::cout<<block[k];}std::cout<<"]}";
  }
  double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  std::cout<<"],\"elapsed_seconds\":"<<elapsed<<",\"scope\":\"screen-only shared maximum-likelihood prior; ideal log loss; all experts use exact base on cold byte and neural overrides; remaining probabilities clamped to 1/65536..65535/65536; no candidate range counting\"}\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
