#include "op.hpp"
#include "cpu/embedding_cpu.hpp"
#include "../../utils.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    ASSERT(weight->isContiguous(), "embedding: all tensors must be contiguous.");

    if(weight->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(), index->shape()[0], weight->shape()[1], weight->elementSize()); 
    }
}
} // namespace llaisys::ops
