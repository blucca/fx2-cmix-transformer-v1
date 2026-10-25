#ifndef MIXER_H
#define MIXER_H

#include <vector>
#include <valarray>
#include "../ds/emhash_map.hpp"
#include <memory>
#include <cstdint>
#ifndef HUTTER_CONTEXT_AGE
#define HUTTER_CONTEXT_AGE 0
#endif
#ifdef HUTTER_MIXER_TRACE
#include "../../research/trace.h"
#endif

struct ContextData {
  ContextData(unsigned long long input_size,
      unsigned long long extra_input_size) : /*steps(0), */weights(input_size),
      extra_weights(extra_input_size) {};
  ContextData& operator=(const ContextData&) = default;
  ContextData(const ContextData&) = default;
  //unsigned long long steps;
  std::valarray<float> weights, extra_weights;
#if HUTTER_CONTEXT_AGE
  std::uint32_t exposures = 0;  // Visits, including skipped gradient updates.
#endif
#ifdef HUTTER_MIXER_TRACE
  std::uint32_t trace_applied_updates = 0;
#endif
  
};

class Mixer {
 public:
  Mixer(const std::valarray<float>& inputs,
      const std::valarray<float>& extra_inputs, const unsigned long long& context,
      float learning_rate, unsigned int extra_input_size,
      unsigned context_limit = 10000);
  float Mix();
  void Perceive(int bit);
#ifdef HUTTER_MIXER_TRACE
  std::uint8_t TraceSupport() const { return trace_support_; }
#endif

 private:
#ifdef HUTTER_CONTEXT_AGE_CHECK
  friend struct ContextAgeCheck;
#endif
  ContextData* GetContextData();
  const std::valarray<float>& inputs_;
  const std::valarray<float>& extra_inputs_vec_;
  // std::valarray<float> extra_inputs_;
  uint16_t extra_inputs_size_;
  float p_, learning_rate_;
  const unsigned long long& context_;
  unsigned long long /*max_steps_,*/ steps_;
  emhash6::HashMap<unsigned int, ContextData> context_map_;
  ContextData context_base_;
  const unsigned context_limit_;
#ifdef HUTTER_MIXER_TRACE
  std::uint8_t trace_support_ = 0;
#endif
#ifdef HUTTER_CACHE_MIXER
  ContextData* current_data_ = nullptr;
#endif
};

#endif

