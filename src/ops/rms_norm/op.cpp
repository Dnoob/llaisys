#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(in->dtype(), weight->dtype());
    ASSERT(in->isContiguous() && weight->isContiguous(), "rms_norm: all tensors must be contiguous.");

    size_t rows = in->shape()[0];
    size_t cols = in->shape()[1];

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(), out->dtype(), rows, cols, eps);
    }
}
} // namespace llaisys::ops
