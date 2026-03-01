#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "cpu/rms_norm_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/rms_norm_cuda.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/rms_norm_maca.cuh"
#endif

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(in->dtype(), weight->dtype());
    ASSERT(in->isContiguous() && weight->isContiguous(), "rms_norm: all tensors must be contiguous.");

    size_t rows = in->shape()[0];
    size_t cols = in->shape()[1];

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(), out->dtype(), rows, cols, eps);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rms_norm(out->data(), in->data(), weight->data(), out->dtype(), rows, cols, eps,
                                llaisys::core::context().runtime().stream());
#endif
#ifdef ENABLE_METAX_API
    case LLAISYS_DEVICE_METAX:
        return metax::rms_norm(out->data(), in->data(), weight->data(), out->dtype(), rows, cols, eps,
                               llaisys::core::context().runtime().stream());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
