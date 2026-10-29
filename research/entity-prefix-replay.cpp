// SPDX-License-Identifier: GPL-3.0-only
// One causal completed-entity trie on exact PHDA9/WRT bytes.
// Existing replay's validation, observed-bit cost, inclusive range counter.
#define main hutter_existing_replay_main
#include "replay.cpp"
#undef main
#include <cstdint>
#include <cstring>
#include <functional>
#include <fstream>
#include <sstream>
#include <limits>

constexpr std::uint32_t NONE=0xffffffffu;
constexpr std::size_t HEAP_LIMIT=256ull*1024*1024;
constexpr std::size_t MAX_ENTITY=1024;
constexpr double LAMBDA=1.0/16;
struct Node { std::uint32_t first=NONE, count=0, terminal=0; };
struct Edge { std::uint32_t child,next; std::uint8_t label; std::uint8_t pad[3]{}; };
static_assert(sizeof(Node)==12 && sizeof(Edge)==12,"pool layout");
struct Trie {
  std::vector<Node> nodes;
  std::vector<Edge> edges;
  std::uint64_t inserts=0,unique=0,name_bytes=0,title_inserts=0,link_inserts=0;
  explicit Trie(std::size_t input_bytes) {
    const std::size_t reserved=(input_bytes+1)*sizeof(Node)+input_bytes*sizeof(Edge)
       +2*input_bytes+128*1024+2*MAX_ENTITY;
    if(reserved>HEAP_LIMIT)throw std::runtime_error("heap reservation exceeds 256MiB");
    nodes.reserve(input_bytes+1);edges.reserve(input_bytes);nodes.emplace_back();
  }
  std::uint32_t child(std::uint32_t node,unsigned char c) const {
    if(node==NONE)return NONE;
    for(auto e=nodes[node].first;e!=NONE;e=edges[e].next)
      if(edges[e].label==c)return edges[e].child;
    return NONE;
  }
  std::uint32_t extend(std::uint32_t node,unsigned char c) {
    auto ch=child(node,c);if(ch!=NONE)return ch;
    if(nodes.size()==nodes.capacity() || edges.size()==edges.capacity())
      throw std::runtime_error("fixed trie pool exhausted");
    ch=nodes.size();nodes.emplace_back();
    const auto e=std::uint32_t(edges.size());
    edges.push_back({ch,nodes[node].first,c,{0,0,0}});nodes[node].first=e;return ch;
  }
  void insert(const std::string&s,int kind) {
    if(s.empty() || s.size()>MAX_ENTITY)return;
    std::uint32_t n=0;++nodes[0].count;
    for(unsigned char c:s) {n=extend(n,c);++nodes[n].count;}
    if(nodes[n].terminal++==0)++unique;
    ++inserts;name_bytes+=s.size();if(kind==1)++title_inserts;else ++link_inserts;
  }
  std::array<std::uint32_t,256> distribution(std::uint32_t node)const {
    std::array<std::uint32_t,256> out{};
    if(node!=NONE)for(auto e=nodes[node].first;e!=NONE;e=edges[e].next)
      out[edges[e].label]=nodes[edges[e].child].count;
    return out;
  }
  std::uint64_t reserved_heap(std::size_t n)const {
    return nodes.capacity()*sizeof(Node)+edges.capacity()*sizeof(Edge)+2*n+128*1024+2*MAX_ENTITY;
  }
};

// Labels describe verified byte spans only. Predictions use this live parser,
// never the offline labels/current byte/future terminator. Only observed bytes
// enter it; complete names enter the trie after their complete delimiter.
struct Parser {
  enum Mode {OUT=0,TITLE=1,TARGET=2,CAPTION=3,EXTERNAL=4,PENDING=5,TITLE_CLOSE=6};
  Mode mode=OUT;
  Trie* trie=nullptr;
  std::vector<unsigned char>*labels=nullptr;
  std::string sep,close,rolling,name,held;
  std::uint32_t node=NONE;
  std::uint64_t start=0,external_start=0,title_count=0,link_count=0,
    external_count=0,caption_count=0,aborts=0,overlength=0,partial_fields=0;
  unsigned external_close_run=0;
  Parser(std::string s,std::string c,Trie*t,std::vector<unsigned char>*l=nullptr)
      :trie(t),labels(l),sep(std::move(s)),close(std::move(c)) {
    rolling.reserve(sep.size());name.reserve(MAX_ENTITY+1);held.reserve(close.size());
  }
  void label(std::uint64_t a,std::uint64_t b,unsigned char kind) {
    if(labels)for(auto i=a;i<b;++i)(*labels)[i]=kind;
  }
  void begin(Mode m,std::uint64_t pos) {mode=m;start=pos;name.clear();held.clear();node=trie?0:NONE;}
  void append(unsigned char c) {
    if(name.size()<=MAX_ENTITY)name.push_back(char(c));
    if(name.size()>MAX_ENTITY)node=NONE;
    else if(trie)node=trie->child(node,c);
  }
  void complete(int kind,std::uint64_t end) {
    label(start,end,kind);
    if(name.size()>MAX_ENTITY)++overlength;
    else if(trie)trie->insert(name,kind);
    if(kind==1)++title_count;else ++link_count;
    name.clear();node=NONE;
  }
  bool active()const {return (mode==TITLE || mode==TARGET) && node!=NONE;}
  void observe(unsigned char c,std::uint64_t pos) {
    // Full article marker is detected from the decoded prefix, not boundaries.
    rolling.push_back(char(c));if(rolling.size()>sep.size())rolling.erase(0,1);
    if(rolling==sep && mode==OUT) {begin(TITLE,pos+1);return;}
    switch(mode) {
    case OUT:
      if(c=='['){mode=PENDING;start=pos+1;}break;
    case PENDING:
      if(c=='['){mode=EXTERNAL;external_start=pos-1;external_close_run=0;}
      else if(c==']' || c=='\n'){mode=OUT;++aborts;}
      else if(c=='Q'){mode=CAPTION;start=pos+1;++caption_count;}
      else {begin(TARGET,start);append(c);}break;
    case TARGET:
      if(c==']' || c=='Q') {complete(2,pos);
        if(c=='Q'){mode=CAPTION;start=pos+1;++caption_count;}
        else mode=OUT;
      } else if(c=='[' || c=='\n') {
        label(start,pos,7);++aborts;mode=(c=='[')?PENDING:OUT;start=pos+1;name.clear();node=NONE;
      } else append(c);break;
    case CAPTION:
      if(c==']'){label(start,pos,3);mode=OUT;}
      else if(c=='\n'){label(start,pos,7);++aborts;mode=OUT;}break;
    case EXTERNAL:
      external_close_run=c==']'?external_close_run+1:0;
      if(external_close_run==2){label(external_start,pos+1,4);++external_count;mode=OUT;}
      else if(c=='\n'){label(external_start,pos,7);++aborts;mode=OUT;}break;
    case TITLE:
      if(c==static_cast<unsigned char>(close[0])){mode=TITLE_CLOSE;held.assign(1,char(c));}
      else append(c);break;
    case TITLE_CLOSE:
      held.push_back(char(c));
      if(close.compare(0,held.size(),held)!=0) {
        for(unsigned char b:held)append(b);held.clear();mode=TITLE;
      } else if(held.size()==close.size()) {
        complete(1,pos+1-close.size());mode=OUT;held.clear();
      }break;
    }
  }
  void finish(std::uint64_t length) {
    if(mode!=OUT){++partial_fields;
      if(mode==TITLE || mode==TITLE_CLOSE || mode==TARGET || mode==CAPTION)label(start,length,7);
      else if(mode==EXTERNAL)label(external_start,length,7);
    }
  }
};

struct Costs {
  std::uint64_t bytes=0,active_bits=0,supported_bits=0,override_bits=0;
  double base_bits=0,capacity_bits=0,active_base_bits=0;
  void print()const {
    std::cout<<"{\"input_bytes\":"<<bytes<<",\"active_bits\":"<<active_bits
      <<",\"supported_bits\":"<<supported_bits<<",\"override_bits\":"<<override_bits
      <<",\"base_probability_bits\":"<<base_bits<<",\"active_base_bits\":"<<active_base_bits
      <<",\"capacity_bits\":"<<capacity_bits<<",\"capacity_bytes\":"<<capacity_bits/8<<'}';
  }
};
static std::string read_all(const char*path) {
  std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error("open markers failed");
  return std::string((std::istreambuf_iterator<char>(f)),{});
}
int main(int argc,char**argv) {
  try {
    if(argc==4 && std::string(argv[1])=="--self-test") {
      auto sep=read_all(argv[2]),close=read_all(argv[3]);Trie t(1024);Parser p(sep,close,&t);
      std::uint64_t pos=0;
      auto feed=[&](const std::string&s){for(unsigned char c:s)p.observe(c,pos++);};
      feed(sep+"ABC");if(t.inserts || p.mode!=Parser::TITLE)throw std::runtime_error("premature title insert");
      feed(close.substr(0,close.size()-1));if(t.inserts)throw std::runtime_error("title inserted before complete close");
      feed(close.substr(close.size()-1));if(t.inserts!=1 || p.mode!=Parser::OUT)throw std::runtime_error("title completion");
      feed("[");if(p.active())throw std::runtime_error("ambiguous bracket predicted");
      feed("A");if(!p.active() || p.mode!=Parser::TARGET)throw std::runtime_error("internal confirmation");
      auto dist=t.distribution(p.node);if(dist['B']!=1)throw std::runtime_error("known title continuation");
      feed("BCQcaption]");if(t.inserts!=2 || p.caption_count!=1)throw std::runtime_error("caption inserted or target missing");
      feed("[[ABC caption]]");if(t.inserts!=2 || p.external_count!=1)throw std::runtime_error("external inserted");
      feed("[Qempty]");if(t.inserts!=2)throw std::runtime_error("empty target inserted");
      feed(sep+"future");if(t.inserts!=2)throw std::runtime_error("future title leaked");
      std::cout<<"{\"status\":\"passed\",\"checks\":[\"insert after complete close\",\"single/double PHDA bracket\",\"caption/external exclusion\",\"same exact title/link trie\",\"future title excluded\"]}\n";return 0;
    }
    if(argc!=12)throw std::runtime_error("usage: entity-gate capacity|replay TRACE STREAM OFFSET LENGTH TAIL_START GOLDEN SEP CLOSE LABEL_OUTPUT MIN_CAPACITY_BYTES");
    const bool replay=std::string(argv[1])=="replay";
    if(!replay && std::string(argv[1])!="capacity")throw std::runtime_error("unknown phase");
    const auto offset=std::stoull(argv[4]),n=std::stoull(argv[5]),tail_start=std::stoull(argv[6]),golden=std::stoull(argv[7]);
    if(!n || n>8388608 || tail_start>=n)throw std::runtime_error("sample size contract");
    const auto sep=read_all(argv[8]),close=read_all(argv[9]);
    if(sep.size()!=15 || close.size()!=5)throw std::runtime_error("prepared marker lengths changed");
    std::string data(n,'\0');std::ifstream stream(argv[3],std::ios::binary);stream.seekg(offset);
    if(!stream.read(data.data(),n))throw std::runtime_error("prepared slice short read");
    std::vector<unsigned char> labels(n,0);
    Parser labeller(sep,close,nullptr,&labels);
    for(std::size_t i=0;i<n;++i)labeller.observe(static_cast<unsigned char>(data[i]),i);
    labeller.finish(n);
    {
      std::ofstream out(argv[10],std::ios::binary);
      if(!out.write(reinterpret_cast<const char*>(labels.data()),labels.size()))throw std::runtime_error("label write failed");
    }
    Trie trie(n);Parser parser(sep,close,&trie);
    RangeCounter base,candidate;std::uint64_t base_at_tail=0,candidate_at_tail=0;
    std::array<Costs,8> fields{};Costs whole,tail;
    std::uint64_t bits=0,queries=0,no_continuation=0;
    std::ifstream trace(argv[2],std::ios::binary);
    if(!trace)throw std::runtime_error("cannot open trace");
    std::array<HutterTraceRecord,8192> block;
    std::array<std::uint32_t,256> dist{};
    std::uint64_t total=0;
    unsigned lo=0,hi=255;double weight=0;
    while(trace.read(reinterpret_cast<char*>(block.data()),sizeof(block)) || trace.gcount()) {
      const auto size=trace.gcount();if(size%sizeof(HutterTraceRecord))throw std::runtime_error("truncated trace");
      for(std::size_t k=0;k<std::size_t(size)/sizeof(HutterTraceRecord);++k) {
        const auto&r=block[k];validate(r);const std::size_t byte=bits/8;const unsigned bpos=bits&7;
        if(byte>=n)throw std::runtime_error("trace longer than prepared slice");
        const int bit=r.flags&1;
        if(bit!=((static_cast<unsigned char>(data[byte])>>(7-bpos))&1))throw std::runtime_error("trace/prepared bit alignment failed");
        if(!bpos) {
          if(byte==tail_start){base_at_tail=base.flushed_bytes();candidate_at_tail=candidate.flushed_bytes();}
          dist.fill(0);total=0;lo=0;hi=255;
          if(parser.active()) {
            dist=trie.distribution(parser.node);for(auto x:dist)total+=x;
            if(total)++queries;else ++no_continuation;
          }
          weight=total?LAMBDA:0;
        }
        const unsigned mid=lo+(hi-lo)/2;
        std::uint64_t denom=0,ones=0;
        if(total)for(unsigned c=lo;c<=hi;++c){denom+=dist[c];if(c>mid)ones+=dist[c];}
        const double bp=r.probability/65536.0;
        const bool active=denom>0,override=r.flags&2;
        const double cp=active?double(ones)/denom:bp;
        const double btruth=bit?bp:1-bp,ctruth=bit?cp:1-cp;
        const double base_cost=cost(bp,bit);
        const double capacity=(active && !override && ctruth>0)?base_cost:0;
        auto observe=[&](Costs&x){x.bytes+=!bpos;x.base_bits+=base_cost;
          x.override_bits+=override;x.active_bits+=active&&!override;
          x.supported_bits+=active&&!override&&ctruth>0;x.capacity_bits+=capacity;
          if(active&&!override)x.active_base_bits+=base_cost;};
        observe(whole);observe(fields[labels[byte]]);if(byte>=tail_start)observe(tail);
        base.encode(r.probability,bit);
        if(replay) {
          std::uint32_t q=r.probability;
          if(weight && active && !override)
            q=std::clamp(std::lround(65536*((1-weight)*bp+weight*cp)),1L,65535L);
          candidate.encode(q,bit);
          if(weight && !override) {
            const double z=(1-weight)*btruth+weight*ctruth;
            weight=z>0?weight*ctruth/z:0;
          }
        }
        if(bit)lo=mid+1;else hi=mid;
        ++bits;
        if(bpos==7)parser.observe(static_cast<unsigned char>(data[byte]),byte);
      }
    }
    parser.finish(n);
    if(!trace.eof() || bits!=n*8)throw std::runtime_error("incomplete trace coverage");
    if(base.flushed_bytes()!=golden)throw std::runtime_error("integer base differs from golden");
    if(parser.title_count!=labeller.title_count || parser.link_count!=labeller.link_count)
      throw std::runtime_error("live/label parser coverage differs");
    const bool enough=tail.capacity_bits/8>=std::stod(argv[11]);
    std::cout<<std::setprecision(15)<<"{\"status\":\"complete\",\"phase\":\""<<(replay?"replay":"capacity")
      <<"\",\"input_bytes\":"<<n<<",\"tail_start\":"<<tail_start<<",\"tail_bytes\":"<<n-tail_start
      <<",\"trace_bits\":"<<bits<<",\"bit_input_alignment\":true,\"base_coded_bytes\":"<<base.flushed_bytes()
      <<",\"base_tail_range_bytes\":"<<base.flushed_bytes()-base_at_tail
      <<",\"capacity_threshold_bytes\":"<<argv[11]<<",\"tail_capacity_threshold_met\":"<<(enough?"true":"false")
      <<",\"whole\":";whole.print();std::cout<<",\"tail\":";tail.print();
    std::cout<<",\"fields\":[";
    const char*names[]={"outside","title","internal_target","internal_caption","external_double_bracket","unused5","unused6","partial_or_aborted"};
    for(unsigned i=0;i<fields.size();++i){if(i)std::cout<<',';std::cout<<"{\"name\":\""<<names[i]<<"\",\"cost\":";fields[i].print();std::cout<<'}';}
    std::cout<<"],\"parser\":{\"complete_titles\":"<<parser.title_count<<",\"complete_targets\":"<<parser.link_count
      <<",\"external_links\":"<<parser.external_count<<",\"captions\":"<<parser.caption_count
      <<",\"aborts\":"<<parser.aborts<<",\"overlength\":"<<parser.overlength<<",\"partial_fields\":"<<parser.partial_fields
      <<"},\"trie\":{\"inserts\":"<<trie.inserts<<",\"unique_spellings\":"<<trie.unique<<",\"title_inserts\":"<<trie.title_inserts
      <<",\"target_inserts\":"<<trie.link_inserts<<",\"nodes\":"<<trie.nodes.size()<<",\"edges\":"<<trie.edges.size()
      <<",\"reserved_heap_bytes\":"<<trie.reserved_heap(n)<<",\"queries\":"<<queries<<",\"no_continuation\":"<<no_continuation<<'}';
    if(replay)std::cout<<",\"lambda\":0.0625,\"candidate_coded_bytes\":"<<candidate.flushed_bytes()
      <<",\"whole_saved_bytes\":"<<std::int64_t(base.flushed_bytes())-std::int64_t(candidate.flushed_bytes())
      <<",\"candidate_tail_range_bytes\":"<<candidate.flushed_bytes()-candidate_at_tail
      <<",\"tail_saved_bytes\":"<<(std::int64_t(base.flushed_bytes())-std::int64_t(base_at_tail))-(std::int64_t(candidate.flushed_bytes())-std::int64_t(candidate_at_tail));
    std::cout<<"}\n";return 0;
  }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
