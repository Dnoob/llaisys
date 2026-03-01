#include "op.hpp"
#include "../../core/llaisys_core.hpp"  
#include <cstring>
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rearrange_cuda.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/rearrange_maca.cuh"
#endif

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    CHECK_SAME_DEVICE(out,in);
    ASSERT(in->isContiguous(), "rearrange: all tensors must be contiguous.");

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    size_t size = out->numel() * out->elementSize();

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        std::memcpy(out->data(), in->data(), size);
        return;
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rearrange(out->data(), in->data(), size,
                                 llaisys::core::context().runtime().stream());
#endif
#ifdef ENABLE_METAX_API
    case LLAISYS_DEVICE_METAX:
        return metax::rearrange(out->data(), in->data(), size,
                                llaisys::core::context().runtime().stream());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
