#include "op.hpp"
#include "cpu/embedding_cpu.hpp"
#include "../../utils.hpp"
#include "../../core/llaisys_core.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/embedding_cuda.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/embedding_maca.cuh"
#endif

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    ASSERT(weight->isContiguous(), "embedding: all tensors must be contiguous.");

    llaisys::core::context().setDevice(weight->deviceType(), weight->deviceId());

    switch (weight->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(), index->shape()[0], weight->shape()[1], weight->elementSize());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::embedding(out->data(), index->data(), weight->data(), index->shape()[0], weight->shape()[1], weight->elementSize(),
                                 llaisys::core::context().runtime().stream());
#endif
#ifdef ENABLE_METAX_API
    case LLAISYS_DEVICE_METAX:
        return metax::embedding(out->data(), index->data(), weight->data(), index->shape()[0], weight->shape()[1], weight->elementSize(),
                                llaisys::core::context().runtime().stream());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }

}
} // namespace llaisys::ops
