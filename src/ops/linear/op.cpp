#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "cpu/linear_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/linear_cuda.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/linear_maca.cuh"
#endif

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "linear: all tensors must be contiguous.");

    size_t M = in->shape()[0];
    size_t K = in->shape()[1];
    size_t N = weight->shape()[0];

    std::byte *bias_data = nullptr;
    if (bias) {
        bias_data = bias->data();
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), bias_data, out->dtype(), M, N, K);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::linear(out->data(), in->data(), weight->data(), bias_data, out->dtype(), M, N, K,
                                llaisys::core::context().runtime().stream());
#endif
#ifdef ENABLE_METAX_API
    case LLAISYS_DEVICE_METAX:
        return metax::linear(out->data(), in->data(), weight->data(), bias_data, out->dtype(), M, N, K,
                             llaisys::core::context().runtime().stream());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
