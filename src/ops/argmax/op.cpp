#include "op.hpp"
#include "cpu/argmax_cpu.hpp"
#include "../../utils.hpp"
#include "../../core/llaisys_core.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/argmax_cuda.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/argmax_maca.cuh"
#endif

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    ASSERT(vals->isContiguous(), "argmax: all tensors must be contiguous.");

    // if(vals->deviceType() == LLAISYS_DEVICE_CPU) {
    //     return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
    // }

    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());

    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel(),
                           llaisys::core::context().runtime().stream());
#endif
#ifdef ENABLE_METAX_API
    case LLAISYS_DEVICE_METAX:
        return metax::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel(),
                          llaisys::core::context().runtime().stream());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
