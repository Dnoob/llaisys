#include "op.hpp"
#include "../../core/llaisys_core.hpp"  
#include <cstring>

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    CHECK_SAME_DEVICE(out,in);
    ASSERT(in->isContiguous(), "rearrange: all tensors must be contiguous.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        std::memcpy(out->data(), in->data(), out->numel() * out->elementSize());
    }
}
} // namespace llaisys::ops
