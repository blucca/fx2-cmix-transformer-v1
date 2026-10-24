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
#ifdef HUTTER_PPMD_TRACE
  const char* ppmd_path = std::getenv("HUTTER_PPMD_TRACE_PATH");
  if (ppmd_path && *ppmd_path) {
    if (!trace_.is_open()) Fail("PPMd sidecar requires HUTTER_TRACE_PATH");
    p_->TracePpmd();  // Fail early when the run loads probabilities without PPMd.
    if (std::ifstream(ppmd_path).good()) Fail("PPMd trace path already exists: %s", ppmd_path);
    ppmd_trace_.open(ppmd_path, std::ios::binary);
    if (!ppmd_trace_) Fail("cannot create PPMd trace: %s", ppmd_path);
  }
#endif
#ifdef HUTTER_MIXER_TRACE
  const char* mixer_path = std::getenv("HUTTER_MIXER_TRACE_PATH");
  if (mixer_path && *mixer_path) {
    if (!trace_.is_open()) Fail("mixer sidecar requires HUTTER_TRACE_PATH");
    p_->TraceMixers();
    if (std::ifstream(mixer_path).good()) Fail("mixer trace path already exists: %s", mixer_path);
    mixer_trace_.open(mixer_path, std::ios::binary);
    if (!mixer_trace_) Fail("cannot create mixer trace: %s", mixer_path);
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
#ifdef HUTTER_PPMD_TRACE
  if (ppmd_trace_.is_open()) {
    const auto& record = p_->TracePpmd();
    ppmd_trace_.write(reinterpret_cast<const char*>(&record), sizeof(record));
    if (!ppmd_trace_) Fail("PPMd trace write failed");
  }
#endif
#ifdef HUTTER_MIXER_TRACE
  if (mixer_trace_.is_open()) {
    const auto& record = p_->TraceMixers();
    mixer_trace_.write(reinterpret_cast<const char*>(&record), sizeof(record));
    if (!mixer_trace_) Fail("mixer trace write failed");
  }
#endif
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
#ifdef HUTTER_PPMD_TRACE
  if (ppmd_trace_.is_open()) {
    ppmd_trace_.flush();
    if (!ppmd_trace_) Fail("PPMd trace flush failed");
  }
#endif
#ifdef HUTTER_MIXER_TRACE
  if (mixer_trace_.is_open()) {
    mixer_trace_.flush();
    if (!mixer_trace_) Fail("mixer trace flush failed");
  }
#endif
}

