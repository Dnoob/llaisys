#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "cpu/rope_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/rope_cuda.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/rope_maca.cuh"
#endif

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(), "rope: all tensors must be contiguous."); 

    size_t seq_len = in->shape()[0];
    size_t n_heads = in->shape()[1];
    size_t head_dim = in->shape()[2];

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), seq_len, n_heads, head_dim, theta);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), seq_len, n_heads, head_dim, theta,
                            llaisys::core::context().runtime().stream());
#endif
#ifdef ENABLE_METAX_API
    case LLAISYS_DEVICE_METAX:
        return metax::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), seq_len, n_heads, head_dim, theta,
                           llaisys::core::context().runtime().stream());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
