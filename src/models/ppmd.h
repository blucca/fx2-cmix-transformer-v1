#ifndef PPMD_H
#define PPMD_H

#include "byte-model.h"
#ifdef HUTTER_PPMD_TRACE
#include "../../research/trace.h"
#endif

#include <memory>

namespace PPMD {

struct ppmd_Model;

class PPMD : public ByteModel {
 public:
  PPMD(int order, int memory, const unsigned int& bit_context,
      const std::vector<bool>& vocab);
  ~PPMD();
  void ByteUpdate();
#ifdef HUTTER_PPMD_TRACE
  std::valarray<float>& Predict() {
    auto& output = ByteModel::Predict();
    trace_.ppmd = output[0];
    return output;
  }
  const HutterPpmdTraceRecord& Trace() const { return trace_; }
#endif
 private:
  const unsigned int& byte_;
  std::unique_ptr<ppmd_Model> ppmd_model_;
  std::valarray<int> byte_map_;
#ifdef HUTTER_PPMD_TRACE
  HutterPpmdTraceRecord trace_{0.5f, 8.0f, -1.0f, 0, 0};
#endif
};

} // namespace PPMD

#endif

