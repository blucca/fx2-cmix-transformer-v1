// SPDX-License-Identifier: GPL-3.0-only
// Reuse the original WRT Dictionary; no predictor/model linkage.
#include "preprocess/dictionary.h"
#include <cstdio>
#include <string>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <cstdint>
static unsigned char remap(unsigned char byte) {
  int c=byte;
  if(c>='{' && c<127)c+='P'-'{';
  else if(c>='P' && c<'T')c-='P'-'{';
  else if((c>=':' && c<='?') || (c>='J' && c<='O'))c^=0x70;
  if(c=='X' || c=='`')c^='X'^'`';
  return c;
}
int main(int argc,char**argv) {
  try {
    if(argc!=5)throw std::runtime_error("usage: dict-bridge DICTIONARY encode|decode INPUT OUTPUT");
    FILE*d=fopen(argv[1],"rb"),*in=fopen(argv[3],"rb"),*out=fopen(argv[4],"wb");
    if(!d||!in||!out)throw std::runtime_error("open failed");
    fseek(in,0,SEEK_END);long len=ftell(in);rewind(in);
    if(std::string(argv[2])=="encode") {
      preprocessor::Dictionary dict(d,true,false);FILE*tmp=fopen("dictionary-encode.tmp","wb+");
      if(!tmp)throw std::runtime_error("tmpfile failed");
      dict.Encode(in,len,tmp);rewind(tmp);int c;
      while((c=getc(tmp))!=EOF)putc(remap(c),out);
      fclose(tmp);
    } else if(std::string(argv[2])=="decode") {
      // Input must have a non-word sentinel. Decode to that sentinel so buffered
      // dictionary expansions are drained even after the last input byte.
      preprocessor::Dictionary dict(d,false,true);
      if(len==0)throw std::runtime_error("empty decode");
      std::uint64_t emitted=0;
      for(;;) {
        const unsigned char c=dict.Decode(in);putc(c,out);
        if(c=='\n' && ftell(in)==len)break;
        if(++emitted>64*std::uint64_t(len)+1024)throw std::runtime_error("decode sentinel missing");
      }
    } else throw std::runtime_error("unknown mode");
    fclose(in);fclose(out);fclose(d);return 0;
  }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
