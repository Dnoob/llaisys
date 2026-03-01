#include "rearrange_maca.cuh"
#include <mcr/mc_runtime.h>

namespace llaisys::ops::metax {
void rearrange(std::byte *out, const std::byte *in, size_t size, llaisysStream_t stream) {
    mcMemcpyAsync(out, in, size, mcMemcpyDeviceToDevice, (mcStream_t)stream);
}
}
