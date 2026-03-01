#include "../runtime_api.hpp"

#include <mcr/mc_runtime.h>
#include <stdexcept>
#include <string>

#define MACA_CHECK(call)                                                        \
    do {                                                                        \
        mcError_t err = (call);                                                 \
        if (err != mcSuccess) {                                                 \
            throw std::runtime_error(std::string("MACA error: ") + mcGetErrorString(err)); \
        }                                                                       \
    } while(0)


static mcMemcpyKind convertMemcpyKind(llaisysMemcpyKind_t kind) {
    switch (kind) {
        case LLAISYS_MEMCPY_H2H: return mcMemcpyHostToHost;
        case LLAISYS_MEMCPY_H2D: return mcMemcpyHostToDevice;
        case LLAISYS_MEMCPY_D2H: return mcMemcpyDeviceToHost;
        case LLAISYS_MEMCPY_D2D: return mcMemcpyDeviceToDevice;
        default:                 return mcMemcpyDefault;
    }
}


namespace llaisys::device::metax {

namespace runtime_api {
int getDeviceCount() {
    int count = 0;
    MACA_CHECK(mcGetDeviceCount(&count));
    return count;
}

void setDevice(int device_id) {
    MACA_CHECK(mcSetDevice(device_id));
}

void deviceSynchronize() {
    MACA_CHECK(mcDeviceSynchronize());
}

llaisysStream_t createStream() {
    mcStream_t stream;
    MACA_CHECK(mcStreamCreate(&stream));
    return (llaisysStream_t)stream;
}

void destroyStream(llaisysStream_t stream) {
    MACA_CHECK(mcStreamDestroy((mcStream_t)stream));
}
void streamSynchronize(llaisysStream_t stream) {
    MACA_CHECK(mcStreamSynchronize((mcStream_t)stream));
}

void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    MACA_CHECK(mcMalloc(&ptr, size));
    return ptr;
}

void freeDevice(void *ptr) {
    MACA_CHECK(mcFree(ptr));
}

void *mallocHost(size_t size) {
    void *ptr = nullptr;
    MACA_CHECK(mcMallocHost(&ptr, size));
    return ptr;
}

void freeHost(void *ptr) {
    MACA_CHECK(mcFreeHost(ptr));
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    MACA_CHECK(mcMemcpy(dst, src, size, convertMemcpyKind(kind)));
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    MACA_CHECK(mcMemcpyAsync(dst, src, size, convertMemcpyKind(kind), (mcStream_t)stream));
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::metax
