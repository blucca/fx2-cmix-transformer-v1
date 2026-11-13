// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Blucca
// ONNX Runtime 1.31.1 / API 31 adapter for one static, constant-weight M1 op.
// Linux: g++ -O3 -mavx2 -mfma -shared -fPIC -std=c++17 -I sdk q8_m1_ort.cpp -o libq8_m1.so
// Windows x64: compile this source with /O2 /arch:AVX2 /std:c++17 /LD and the same SDK.
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "q8_m1_kernel.hpp"

#include <array>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <vector>

#if defined(_WIN32)
#define BLUCCA_Q8_EXPORT __declspec(dllexport)
#else
#define BLUCCA_Q8_EXPORT __attribute__((visibility("default")))
#endif

namespace {

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
    return api.CreateStatus(ORT_FAIL, "Q8MatMulM1: unknown C++ exception");
  }
}

template <typename TensorInfo>
void RequireTensor(const TensorInfo& info, ONNXTensorElementDataType type,
                   const std::vector<std::int64_t>& shape, const char* message) {
  if (info.GetElementType() != type || info.GetShape() != shape) {
    throw std::invalid_argument(message);
  }
}

class Q8M1OrtKernel {
 public:
  Q8M1OrtKernel(const OrtApi& api, const OrtKernelInfo* kernel_info) : api_(api) {
    const Ort::ConstKernelInfo info{kernel_info};
    if (info.GetInputCount() != 3 || info.GetOutputCount() != 1) {
      throw std::invalid_argument("Q8MatMulM1 requires exactly inputs A, B, S and output Y");
    }
    const std::int64_t k = info.GetAttribute<std::int64_t>("K");
    const std::int64_t n = info.GetAttribute<std::int64_t>("N");
    if (k <= 0 || k % 32 != 0 || n <= 0 || n % 16 != 0) {
      throw std::invalid_argument("Q8MatMulM1 requires positive K divisible by 32 and N divisible by 16");
    }

    const auto a_type = info.GetInputTypeInfo(0);
    RequireTensor(a_type.GetTensorTypeAndShapeInfo(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                  {1, k}, "Q8MatMulM1 requires static A: FLOAT [1, K]");
    const auto y_type = info.GetOutputTypeInfo(0);
    RequireTensor(y_type.GetTensorTypeAndShapeInfo(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                  {1, n}, "Q8MatMulM1 requires static Y: FLOAT [1, N]");

    // These are borrowed initializer OrtValues. Q8M1 copies/prepacks them here;
    // no source B/S pointer or OrtValue is retained in the kernel.
    int b_constant = 0;
    const auto b = info.GetTensorConstantInput(1, &b_constant);
    if (!b_constant || b == nullptr) {
      throw std::invalid_argument("Q8MatMulM1 B must be a constant UINT8 initializer");
    }
    int s_constant = 0;
    const auto s = info.GetTensorConstantInput(2, &s_constant);
    if (!s_constant || s == nullptr) {
      throw std::invalid_argument("Q8MatMulM1 S must be a constant FLOAT initializer");
    }
    RequireTensor(b.GetTensorTypeAndShapeInfo(), ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
                  {n, k / 32, 32}, "Q8MatMulM1 requires B: UINT8 [N, K/32, 32]");
    RequireTensor(s.GetTensorTypeAndShapeInfo(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                  {n * (k / 32)}, "Q8MatMulM1 requires S: FLOAT [N*(K/32)]");
    packed_ = std::make_unique<blucca::Q8M1>(k, n, b.GetTensorData<std::uint8_t>(),
                                           s.GetTensorData<float>());
  }

  OrtStatus* ComputeV2(OrtKernelContext* context) noexcept {
    try {
      const Ort::KernelContext ctx{context};
      const auto a = ctx.GetInput(0);
      const auto a_info = a.GetTensorTypeAndShapeInfo();
      if (a_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
          a_info.GetDimensionsCount() != 2) {
        throw std::invalid_argument("Q8MatMulM1 requires runtime A: FLOAT [1, K]");
      }
      std::array<std::int64_t, 2> a_shape{};
      Ort::ThrowOnError(api_.GetDimensions(a_info, a_shape.data(), a_shape.size()));
      if (a_shape[0] != 1 || a_shape[1] != packed_->k) {
        throw std::invalid_argument("Q8MatMulM1 requires runtime A: FLOAT [1, K]");
      }
      const std::array<std::int64_t, 2> y_shape{1, packed_->n};
      auto y = ctx.GetOutput(0, y_shape.data(), y_shape.size());
      packed_->Run(a.GetTensorData<float>(), y.GetTensorMutableData<float>());
      return nullptr;
    } catch (...) {
      return StatusFromCurrentException(api_);
    }
  }

 private:
  const OrtApi& api_;
  std::unique_ptr<blucca::Q8M1> packed_;
};

struct Q8MatMulM1Op : Ort::CustomOpBase<Q8MatMulM1Op, Q8M1OrtKernel, true> {
  const char* GetName() const { return "Q8MatMulM1"; }
  const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }
  std::size_t GetInputTypeCount() const { return 3; }
  ONNXTensorElementDataType GetInputType(std::size_t index) const {
    return index == 1 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
  std::size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(std::size_t) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
  OrtStatus* CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info,
                            void** kernel) const noexcept {
    *kernel = nullptr;
    try {
      *kernel = new Q8M1OrtKernel(api, info);
      return nullptr;
    } catch (...) {
      return StatusFromCurrentException(api);
    }
  }
};

struct DomainState {
  Q8MatMulM1Op op;
  Ort::CustomOpDomain domain{"org.blucca"};
  DomainState() { domain.Add(&op); }
};

}  // namespace

extern "C" BLUCCA_Q8_EXPORT OrtStatus* ORT_API_CALL
RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api_base) noexcept {
  const OrtApi* api = api_base->GetApi(ORT_API_VERSION);
  if (!api) {
    return api_base->GetApi(1)->CreateStatus(ORT_FAIL, "Q8MatMulM1 requires ONNX Runtime API 31 (1.31.x)");
  }
  try {
    Ort::InitApi(api);
    // One immutable domain/op pair lives for the shared library lifetime,
    // including successive registrations and multiple simultaneous sessions.
    static DomainState state;
    return api->AddCustomOpDomain(options, state.domain);
  } catch (...) {
    return StatusFromCurrentException(*api);
  }
}
