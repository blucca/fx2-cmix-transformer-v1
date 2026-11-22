// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Blucca
// ONNX Runtime 1.31.1 / API 31, static constant packed weights and M1.
// Linux build: ./build.sh
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "q8_m1_packed_kernel.hpp"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <vector>

#if defined(_WIN32)
#define BLUCCA_Q8_PACKED_EXPORT __declspec(dllexport)
#else
#define BLUCCA_Q8_PACKED_EXPORT __attribute__((visibility("default")))
#endif

namespace {

// Successful new-kernel constructions over this library's process lifetime.
// It counts scalar objects, not weights, allocations, allocator bytes or RSS.
std::atomic<std::uint64_t> kernel_ctor_count{0};

// Only called from a catch block; exceptions stop at the ORT C callback boundary.
OrtStatus* StatusFromCurrentException(const OrtApi& api) noexcept {
  try {
    throw;
  } catch (const Ort::Exception& error) {
    return api.CreateStatus(error.GetOrtErrorCode(), error.what());
  } catch (const std::invalid_argument& error) {
    return api.CreateStatus(ORT_INVALID_ARGUMENT, error.what());
  } catch (const std::exception& error) {
    return api.CreateStatus(ORT_FAIL, error.what());
  } catch (...) {
    return api.CreateStatus(ORT_FAIL, "Q8MatMulM1Packed: unknown C++ exception");
  }
}

template <typename TensorInfo>
void RequireTensor(const TensorInfo& info, ONNXTensorElementDataType type,
                   const std::vector<std::int64_t>& shape, const char* message) {
  if (info.GetElementType() != type || info.GetShape() != shape) {
    throw std::invalid_argument(message);
  }
}

void RequireM1Shape(const std::vector<std::int64_t>& shape,
                    std::int64_t last_dimension, const char* message) {
  if (shape.empty() || shape.back() != last_dimension) {
    throw std::invalid_argument(message);
  }
  for (std::size_t i = 0; i + 1 < shape.size(); ++i) {
    if (shape[i] != 1) {
      throw std::invalid_argument(message);
    }
  }
}

class Q8M1PackedOrtKernel {
 public:
  Q8M1PackedOrtKernel(const OrtApi& api, const OrtKernelInfo* kernel_info)
      : api_(api) {
    const Ort::ConstKernelInfo info{kernel_info};
    if (info.GetInputCount() != 3 || info.GetOutputCount() != 1) {
      throw std::invalid_argument(
          "Q8MatMulM1Packed requires exactly inputs A, P, PS and output Y");
    }
    k_ = info.GetAttribute<std::int64_t>("k");
    n_ = info.GetAttribute<std::int64_t>("n");
    if (k_ <= 0 || k_ % 32 != 0 || n_ <= 0 || n_ % 16 != 0) {
      throw std::invalid_argument(
          "Q8MatMulM1Packed requires positive k divisible by 32 and n divisible by 16");
    }
    blocks_ = static_cast<std::size_t>(k_) / 32;
    tiles_ = static_cast<std::size_t>(n_) / 16;

    const auto a_type = info.GetInputTypeInfo(0);
    const auto a_info = a_type.GetTensorTypeAndShapeInfo();
    const auto a_shape = a_info.GetShape();
    if (a_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      throw std::invalid_argument("Q8MatMulM1Packed requires A: FLOAT");
    }
    RequireM1Shape(a_shape, k_,
                   "Q8MatMulM1Packed requires static A with last dimension k and M=1");
    auto y_shape = a_shape;
    y_shape.back() = n_;
    const auto y_type = info.GetOutputTypeInfo(0);
    RequireTensor(y_type.GetTensorTypeAndShapeInfo(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                  y_shape, "Q8MatMulM1Packed requires FLOAT Y with A's shape and last dimension n");

    // These borrowed initializer values are used only for constructor checks.
    // Their data, values and views all leave scope before Compute is called.
    int p_constant = 0;
    const auto p = info.GetTensorConstantInput(1, &p_constant);
    if (!p_constant || p == nullptr) {
      throw std::invalid_argument(
          "Q8MatMulM1Packed P must be a constant INT8 initializer");
    }
    int ps_constant = 0;
    const auto ps = info.GetTensorConstantInput(2, &ps_constant);
    if (!ps_constant || ps == nullptr) {
      throw std::invalid_argument(
          "Q8MatMulM1Packed PS must be a constant FLOAT initializer");
    }
    RequireTensor(p.GetTensorTypeAndShapeInfo(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
                  {n_ / 16, k_ / 32, 32, 16},
                  "Q8MatMulM1Packed requires P: INT8 [N/16, K/32, 32, 16]");
    RequireTensor(ps.GetTensorTypeAndShapeInfo(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                  {n_ / 16, k_ / 32, 16},
                  "Q8MatMulM1Packed requires PS: FLOAT [N/16, K/32, 16]");
    const float* scales = ps.GetTensorData<float>();
    const std::size_t scale_count = static_cast<std::size_t>(n_) * blocks_;
    for (std::size_t i = 0; i < scale_count; ++i) {
      if (!(scales[i] > 0.0f) || !std::isfinite(scales[i])) {
        throw std::invalid_argument(
            "Q8MatMulM1Packed PS must contain finite positive scales");
      }
    }
    kernel_ctor_count.fetch_add(1, std::memory_order_relaxed);
  }

  OrtStatus* ComputeV2(OrtKernelContext* context) noexcept {
    try {
      const Ort::KernelContext ctx{context};
      const auto a = ctx.GetInput(0);
      const auto a_info = a.GetTensorTypeAndShapeInfo();
      if (a_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::invalid_argument("Q8MatMulM1Packed requires runtime A: FLOAT");
      }
      auto y_shape = a_info.GetShape();
      RequireM1Shape(y_shape, k_,
                     "Q8MatMulM1Packed requires runtime A with last dimension k and M=1");
      y_shape.back() = n_;
      auto y = ctx.GetOutput(0, y_shape.data(), y_shape.size());

      // Always read P and PS from this invocation's context. The view and
      // borrowed input OrtValues are call-local, and no repack/cache is made.
      const auto p = ctx.GetInput(1);
      const auto ps = ctx.GetInput(2);
      const blucca::Q8M1PackedView view(blocks_, tiles_, p.GetTensorData<std::int8_t>(),
                                       ps.GetTensorData<float>());
      view.Run(a.GetTensorData<float>(), y.GetTensorMutableData<float>());
      return nullptr;
    } catch (...) {
      return StatusFromCurrentException(api_);
    }
  }

 private:
  const OrtApi& api_;
  std::int64_t k_ = 0;
  std::int64_t n_ = 0;
  std::size_t blocks_ = 0;
  std::size_t tiles_ = 0;
};

struct Q8MatMulM1PackedOp
    : Ort::CustomOpBase<Q8MatMulM1PackedOp, Q8M1PackedOrtKernel, true> {
  const char* GetName() const { return "Q8MatMulM1Packed"; }
  const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }
  std::size_t GetInputTypeCount() const { return 3; }
  ONNXTensorElementDataType GetInputType(std::size_t index) const {
    return index == 1 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
  std::size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(std::size_t) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
  OrtStatus* CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info,
                            void** kernel) const noexcept {
    *kernel = nullptr;
    try {
      *kernel = new Q8M1PackedOrtKernel(api, info);
      return nullptr;
    } catch (...) {
      return StatusFromCurrentException(api);
    }
  }
};

struct DomainState {
  Q8MatMulM1PackedOp op;
  Ort::CustomOpDomain domain{"org.blucca"};
  DomainState() { domain.Add(&op); }
};

}  // namespace

extern "C" BLUCCA_Q8_PACKED_EXPORT std::uint64_t
BluccaQ8M1PackedCtorCount() noexcept {
  return kernel_ctor_count.load(std::memory_order_relaxed);
}

extern "C" BLUCCA_Q8_PACKED_EXPORT std::uint64_t
BluccaQ8M1PackedOwnedPackedBytes() noexcept {
  // Native signed-code and reordered-scale owner payload. This adapter and
  // its call-local view own no such arrays. ORT initializer/RSS accounting is
  // separate and must be measured by the full-model worker.
  return 0;
}

extern "C" BLUCCA_Q8_PACKED_EXPORT OrtStatus* ORT_API_CALL
RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api_base) noexcept {
  const OrtApi* api = api_base->GetApi(ORT_API_VERSION);
  if (!api) {
    return api_base->GetApi(1)->CreateStatus(
        ORT_FAIL, "Q8MatMulM1Packed requires ONNX Runtime API 31 (1.31.x)");
  }
  try {
    Ort::InitApi(api);
    // One immutable domain/new-op pair lives for the shared library lifetime,
    // including successive registrations and multiple simultaneous sessions.
    static DomainState state;
    return api->AddCustomOpDomain(options, state.domain);
  } catch (...) {
    return StatusFromCurrentException(*api);
  }
}
