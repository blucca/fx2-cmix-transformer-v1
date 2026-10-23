#include "encoder.h"
#ifdef HUTTER_TRACE
#include "../../research/trace.h"
#include <cstdlib>
#endif

Encoder::Encoder(std::ofstream* os, Predictor* p) : os_(os), x1_(0),
    x2_(0xffffffff), p_(p) {
#ifdef HUTTER_TRACE
  const char* path = std::getenv("HUTTER_TRACE_PATH");
  if (path && *path) {
    if (std::ifstream(path).good()) Fail("trace path already exists: %s", path);
    trace_.open(path, std::ios::binary);
    if (!trace_) Fail("cannot create trace: %s", path);
  }
#endif
}

void Encoder::WriteByte(unsigned int byte) {
  out_.push_back(byte);
}

unsigned int Encoder::Discretize(float p) {
  return 1 + 65534 * p;
}

void Encoder::Encode(int bit) {
#ifdef HUTTER_TRACE
  const float probability = p_->Predict();
  const unsigned int p = Discretize(probability);
  if (trace_.is_open()) {
    const float neural = p_->TraceNeural();
    HutterTraceRecord record{
        static_cast<std::uint16_t>(p),
        static_cast<std::uint16_t>(bit | ((neural == 0 || neural == 1) ? 2 : 0)),
        neural, p_->TraceFxcm(), p_->TraceMixer()};
    trace_.write(reinterpret_cast<const char*>(&record), sizeof(record));
    if (!trace_) Fail("trace write failed");
  }
#else
  const unsigned int p = Discretize(p_->Predict());
#endif
  const unsigned int xmid = x1_ + ((x2_ - x1_) >> 16) * p +
      (((x2_ - x1_) & 0xffff) * p >> 16);
  if (bit) {
    x2_ = xmid;
  } else {
    x1_ = xmid + 1;
  }
  p_->Perceive(bit);

  while (((x1_^x2_) & 0xff000000) == 0) {
    WriteByte(x2_ >> 24);
    x1_ <<= 8;
    x2_ = (x2_ << 8) + 255;
  }
}

void Encoder::Flush() {
  while (((x1_^x2_) & 0xff000000) == 0) {
    WriteByte(x2_ >> 24);
    x1_ <<= 8;
    x2_ = (x2_ << 8) + 255;
  }
  WriteByte(x2_ >> 24);

  auto* data = reinterpret_cast<const char*>(out_.data());
  os_->write(data, out_.size());
#ifdef HUTTER_TRACE
  if (trace_.is_open()) {
    trace_.flush();
    if (!trace_) Fail("trace flush failed");
  }
#endif
}

