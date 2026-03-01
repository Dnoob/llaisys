# Project 2：GPU 集成 — 项目报告

## 1. 项目概述

### 目标

为 LLAISYS（Let's Learn AI SYStem）推理系统集成 GPU 支持，适配 NVIDIA CUDA 和沐曦 MACA 两个平台，使模型推理能够在 GPU 上执行，获得显著的性能提升。

### 测试模型

DeepSeek-R1-Distill-Qwen-1.5B（BF16 精度）

### 性能总览

| 平台 | GPU 型号 | 推理时间 | 速度 |
|------|---------|---------|------|
| NVIDIA | A100-SXM4-80GB | 0.52s / 81 tokens | **155.8 tokens/s** |
| 沐曦 | C500 32GB | 1.09s / 81 tokens | **74.3 tokens/s** |

---

## 2. 构建系统

### 双平台编译配置

通过 xmake 选项控制是否编译对应平台的 GPU 代码：

```lua
-- NVIDIA CUDA
option("nv-gpu")
    set_default(false)
    set_description("Whether to compile implementations for Nvidia GPU")
option_end()
if has_config("nv-gpu") then add_defines("ENABLE_NVIDIA_API") end

-- 沐曦 MACA
option("mx-gpu")
    set_default(false)
    set_description("Whether to compile implementations for MetaX GPU")
option_end()
if has_config("mx-gpu") then add_defines("ENABLE_METAX_API") end
```

### NVIDIA CUDA 编译

xmake 原生支持 CUDA，配置简洁：

```lua
if has_config("nv-gpu") then
    add_rules("cuda")
    add_cugencodes("native")
    add_files("src/device/nvidia/*.cu", "src/ops/*/nvidia/*.cu")
    add_cuflags("--compiler-options", "-fPIC", {force = true})
    add_links("cudart", "cublas")
    add_linkdirs("/usr/local/cuda/lib64")
end
```

### 沐曦 MACA 编译

xmake 不原生支持 mxcc 编译器。如果用 `add_files("*.cu")`，xmake 会自动检测 CUDA SDK 并报错 `Cuda SDK not found!`。

解决方案：使用 `before_build` 回调手动调用 mxcc 编译：

```lua
if has_config("mx-gpu") then
    add_includedirs("/opt/maca/include")
    add_linkdirs("/opt/maca/lib")
    add_links("mcblas")
    before_build(function (target)
        local mxcc = "/opt/maca/mxgpu_llvm/bin/mxcc"
        local cu_files = {}
        for _, f in ipairs(os.files("src/device/metax/*.cu")) do table.insert(cu_files, f) end
        for _, f in ipairs(os.files("src/ops/*/metax/*.cu")) do table.insert(cu_files, f) end
        for _, sourcefile in ipairs(cu_files) do
            local objectfile = target:objectfile(sourcefile)
            os.mkdir(path.directory(objectfile))
            os.vrunv(mxcc, {"-c", sourcefile, "-o", objectfile, "-fPIC", "-std=c++17",
                "-Iinclude", "-I/opt/maca/include", "-DENABLE_METAX_API"})
            table.insert(target:objectfiles(), objectfile)
        end
    end)
end
```

### 构建命令

```bash
# NVIDIA CUDA
xmake f --nv-gpu=y -cv && xmake && xmake install

# 沐曦 MACA
xmake f --mx-gpu=y --root && xmake --root && xmake install --root
```

---

## 3. 设备抽象层

### 架构设计

```
Context (thread_local) → Runtime (per device) → LlaisysRuntimeAPI (函数指针集)
```

框架通过 `LlaisysRuntimeAPI` 函数指针结构体实现设备抽象，12 个函数覆盖四组操作。上层代码通过统一接口调用，无需关心具体设备。

### 12 个 Runtime API

| 组 | 函数 | CUDA API | MACA API |
|---|---|---|---|
| 设备管理 | `getDeviceCount` | `cudaGetDeviceCount` | `mcGetDeviceCount` |
| | `setDevice` | `cudaSetDevice` | `mcSetDevice` |
| | `deviceSynchronize` | `cudaDeviceSynchronize` | `mcDeviceSynchronize` |
| 流管理 | `createStream` | `cudaStreamCreate` | `mcStreamCreate` |
| | `destroyStream` | `cudaStreamDestroy` | `mcStreamDestroy` |
| | `streamSynchronize` | `cudaStreamSynchronize` | `mcStreamSynchronize` |
| 内存管理 | `mallocDevice` | `cudaMalloc` | `mcMalloc` |
| | `freeDevice` | `cudaFree` | `mcFree` |
| | `mallocHost` | `cudaMallocHost` | `mcMallocHost` |
| | `freeHost` | `cudaFreeHost` | `mcFreeHost` |
| 内存拷贝 | `memcpySync` | `cudaMemcpy` | `mcMemcpy` |
| | `memcpyAsync` | `cudaMemcpyAsync` | `mcMemcpyAsync` |

### 关键设计

- **错误检查宏**：`CUDA_CHECK` / `MACA_CHECK`，每次 API 调用后检查返回值，失败立即抛异常
- **枚举显式转换**：框架的 `llaisysMemcpyKind_t` 通过 switch 转换为平台特定枚举
- **设备分发**：`src/device/runtime_api.cpp` 中通过 switch + `#ifdef` 条件编译分发到对应实现

```cpp
case LLAISYS_DEVICE_NVIDIA:
#ifdef ENABLE_NVIDIA_API
    return llaisys::device::nvidia::getRuntimeAPI();
#else
    return getUnsupportedRuntimeAPI();
#endif
case LLAISYS_DEVICE_METAX:
#ifdef ENABLE_METAX_API
    return llaisys::device::metax::getRuntimeAPI();
#else
    return getUnsupportedRuntimeAPI();
#endif
```

---

## 4. GPU 算子实现

### BF16 支持：template + cast 模式

模型权重使用 BF16 精度，所有 GPU 算子需要同时支持 F32 和 BF16。设计 cast 工具函数统一处理类型转换：

| 平台 | cast 函数 | BF16 类型 | 头文件 |
|------|----------|----------|--------|
| NVIDIA | `cuda_cast<T>()` | `__nv_bfloat16` | `cuda_cast.cuh` |
| 沐曦 | `maca_cast<T>()` | `__maca_bfloat16` | `maca_cast.cuh` |

```cuda
// 以 NVIDIA 为例
template <> __device__ inline float cuda_cast<float, __nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}
template <> __device__ inline __nv_bfloat16 cuda_cast<__nv_bfloat16, float>(float v) {
    return __float2bfloat16(v);
}
```

所有 kernel 使用统一模式：读入时转 float → float 精度计算 → 写出时转回原类型。启动函数中 switch 分发 F32 和 BF16 两个 template 实例化：

```cuda
switch (type) {
case LLAISYS_DTYPE_F32:
    my_kernel<<<grid, block, 0, s>>>(reinterpret_cast<float *>(out), ...);
    return;
case LLAISYS_DTYPE_BF16:
    my_kernel<<<grid, block, 0, s>>>(reinterpret_cast<__nv_bfloat16 *>(out), ...);
    return;
}
```

这样一个 template kernel 同时处理两种类型，避免写两份代码。新增类型（如 FP16）只需加一组 cast 特化，kernel 无需修改。

### 算子分发模式

每个算子的 `op.cpp` 统一使用 switch 分发到对应平台实现：

```cpp
#ifdef ENABLE_NVIDIA_API
#include "nvidia/xxx_cuda.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/xxx_maca.cuh"
#endif

switch (out->deviceType()) {
case LLAISYS_DEVICE_CPU:
    return cpu::xxx(...);
#ifdef ENABLE_NVIDIA_API
case LLAISYS_DEVICE_NVIDIA:
    return nvidia::xxx(..., llaisys::core::context().runtime().stream());
#endif
#ifdef ENABLE_METAX_API
case LLAISYS_DEVICE_METAX:
    return metax::xxx(..., llaisys::core::context().runtime().stream());
#endif
default:
    EXCEPTION_UNSUPPORTED_DEVICE;
}
```

GPU 分支比 CPU 多传一个 `stream` 参数，来源于 `context().runtime().stream()`。

### 9 个算子概览

| 算子 | 实现方式 | 核心技术 |
|------|---------|---------|
| add | template kernel | 逐元素，越界保护 |
| swiglu | template kernel | 逐元素，`expf` |
| argmax | template kernel | shared memory 两阶段归约 |
| embedding | 自写 kernel | 按字节查表复制 |
| rms_norm | template kernel | shared memory 归约 + `rsqrtf` |
| linear | cuBLAS/mcBLAS | `Sgemm`/`GemmEx` + bias kernel |
| rope | template kernel | 前后半配对旋转，三角函数 |
| self_attention | template kernel | 单 kernel 完成 QK·softmax·V |
| rearrange | memcpy API | device-to-device 拷贝 |

### 4.1 逐元素算子

#### add — 逐元素加法

最简单的 CUDA kernel，每个线程处理一个元素：

```cuda
template <typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t numel) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        float va = cuda_cast<float>(a[i]);
        float vb = cuda_cast<float>(b[i]);
        c[i] = cuda_cast<T>(va + vb);
    }
}
```

#### swiglu — SwiGLU 激活函数

与 add 结构相同，只是计算公式不同：

```cuda
float sigmoid_g = 1.0f / (1.0f + expf(-g));
out[i] = cuda_cast<T>(u * g * sigmoid_g);
```

要点：CUDA kernel 中使用 `expf` 而非 `std::exp`。

#### rope — 旋转位置编码

每个线程处理一对位置 `(i, i+half)`：

```cuda
size_t half = head_dim / 2;
float angle = pos / powf(theta, 2.0f * i / (float)head_dim);
out[base + i]        = cuda_cast<T>(a * cosf(angle) - b * sinf(angle));
out[base + i + half] = cuda_cast<T>(b * cosf(angle) + a * sinf(angle));
```

设计思路：总线程数 = `seq_len × n_heads × half`，通过除法和取模将一维索引还原为 `(s, h, i)` 三维坐标。

### 4.2 归约算子

#### argmax — 两阶段归约找最大值

大规模归约无法在单个 block 内完成（block 间不能同步），需要两阶段：

**阶段 1**：多个 block（最多 128 个）并行，各自用 grid-stride loop 扫描一部分数据，block 内树形归约得到局部最大值。

```cuda
// Grid-stride loop 找局部最大
for (size_t i = blockIdx.x * blockDim.x + tid; i < numel; i += blockDim.x * gridDim.x) {
    float v = cuda_cast<float>(vals[i]);
    if (v > local_max) { local_max = v; local_idx = i; }
}
// 存入共享内存，树形归约
```

**阶段 2**：单个 block 归约所有局部结果。

#### rms_norm — 共享内存归约求均方根

每行一个 block，分三步：
1. 每个线程累加局部平方和 → 共享内存树形归约求总和
2. `rsqrtf(sum/cols + eps)` 计算 RMS 倒数（一条 GPU 指令）
3. 逐元素归一化：`out[c] = in[c] * rms * weight[c]`

### 4.3 矩阵运算

#### linear — cuBLAS/mcBLAS 矩阵乘法

Transformer 大部分计算量在矩阵乘法，使用 BLAS 库获取极致性能。

**handle 管理**：function-local static + lambda 懒初始化，只创建一次：

```cuda
// NVIDIA
static cublasHandle_t get_cublas_handle() {
    static cublasHandle_t handle = []() {
        cublasHandle_t h; cublasCreate(&h); return h;
    }();
    return handle;
}

// 沐曦（完全对应）
static mcblasHandle_t get_mcblas_handle() {
    static mcblasHandle_t handle = []() {
        mcblasHandle_t h; mcblasCreate(&h); return h;
    }();
    return handle;
}
```

**行主序 vs 列主序转换**：

cuBLAS/mcBLAS 继承 Fortran BLAS 的列主序，而 C/C++ 是行主序。核心技巧：

> 行主序 M×N 矩阵 = 列主序 N×M 矩阵（内存布局完全一样）

目标：`out[M,N] = in[M,K] × weight^T[K,N]` → 转置得 `out^T[N,M] = weight[N,K] × in^T[K,M]`

```cuda
// NVIDIA F32
cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
            &alpha, weight, K, in, K, &beta, out, N);

// NVIDIA BF16
cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
             &alpha,
             weight, CUDA_R_16BF, K,
             in, CUDA_R_16BF, K,
             &beta,
             out, CUDA_R_16BF, N,
             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

// 沐曦 BF16（API 一一对应）
mcblasGemmEx(handle, MCBLAS_OP_T, MCBLAS_OP_N, N, M, K,
             &alpha,
             weight, MACA_R_16BF, K,
             in, MACA_R_16BF, K,
             &beta,
             out, MACA_R_16BF, N,
             MCBLAS_COMPUTE_32F, MCBLAS_GEMM_DEFAULT);
```

### 4.4 注意力

#### self_attention — 单 kernel 完成 Q·K + Softmax + Score·V

每个 block 处理一个 `(query_pos, head)` 组合，grid_size = query_len × num_heads。

三阶段计算：
1. **Q·K 点积 + causal mask**：每个线程计算一个 key 位置的 score，未来位置设为 `-FLT_MAX`
2. **Softmax**：共享内存树形归约求 max 和 sum，原地归一化
3. **Score·V 加权求和**：每个线程计算输出向量的一个维度

支持 GQA（Grouped Query Attention）：`kv_h = h / group`，多个 Q 头共享一个 KV 头。

### 4.5 数据搬运

#### embedding — 查表复制

按字节粒度并行，不依赖数据类型，F32/BF16/FP16 通用：

```cuda
size_t row = i / row_bytes;   // 第几个 token
size_t col = i % row_bytes;   // 行内字节偏移
int64_t idx = reinterpret_cast<const int64_t *>(index)[row];
out[i] = weight[idx * row_bytes + col];
```

#### rearrange — 设备间数据搬运

直接调用 memcpy API，无需自写 kernel：

```cuda
// NVIDIA
cudaMemcpyAsync(out, in, size, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
// 沐曦
mcMemcpyAsync(out, in, size, mcMemcpyDeviceToDevice, (mcStream_t)stream);
```

---

## 5. 沐曦 MACA 平台适配

### 适配策略

沐曦 MXMACA SDK 是 CUDA 的兼容替代品，API 结构和语义与 CUDA 一一对应，前缀从 `cuda*` 变为 `mc*`。适配核心思路：复制 NVIDIA 实现，全局替换 API 名称。

### CUDA → MACA API 映射

| 类别 | CUDA | MACA |
|------|------|------|
| 头文件 | `cuda_runtime.h` | `mcr/mc_runtime.h` |
| 错误类型 | `cudaError_t` / `cudaSuccess` | `mcError_t` / `mcSuccess` |
| 内存 | `cudaMalloc` / `cudaFree` | `mcMalloc` / `mcFree` |
| 拷贝 | `cudaMemcpy` / `cudaMemcpyAsync` | `mcMemcpy` / `mcMemcpyAsync` |
| 流 | `cudaStream_t` | `mcStream_t` |
| BLAS 头文件 | `cublas_v2.h` | `mcblas/mcblas.h` |
| BLAS Handle | `cublasHandle_t` | `mcblasHandle_t` |
| BLAS 运算 | `cublasSgemm` / `cublasGemmEx` | `mcblasSgemm` / `mcblasGemmEx` |
| BF16 数据类型标识 | `CUDA_R_16BF` | `MACA_R_16BF` |
| 计算类型 | `CUBLAS_COMPUTE_32F` | `MCBLAS_COMPUTE_32F` |
| BF16 类型 | `__nv_bfloat16` | `__maca_bfloat16` |
| BF16 头文件 | `cuda_bf16.h` | `common/maca_bfloat16.h` |
| 编译器 | `nvcc` | `mxcc` |

---

## 6. 模型推理适配

### 修改内容

`qwen2.cc` 中 argmax 的结果在 GPU 显存上，需要 D2H memcpy 读回 CPU：

```cpp
int64_t result;
if (dev == LLAISYS_DEVICE_CPU) {
    std::memcpy(&result, max_idx->data(), sizeof(int64_t));
} else {
    core::context().setDevice(dev, dev_id);
    core::context().runtime().api()->memcpy_sync(
        &result, max_idx->data(), sizeof(int64_t), LLAISYS_MEMCPY_D2H);
}
```

由于使用了 Runtime API 抽象，`else` 分支同时覆盖 NVIDIA 和沐曦，无需针对不同 GPU 平台做额外修改。

---

## 7. 性能数据

### 测试环境

| 平台 | GPU | 显存 | 测试方式 |
|------|-----|------|---------|
| NVIDIA | A100-SXM4-80GB | 80GB | 独占 GPU |
| 沐曦 | C500 | 32GB | 独占 GPU |

### 推理性能（DeepSeek-R1-Distill-Qwen-1.5B，BF16）

| 平台 | 推理时间 | 速度 |
|------|---------|------|
| NVIDIA A100 | 0.52s / 81 tokens | **155.8 tokens/s** |
| 沐曦 C500 | 1.09s / 81 tokens | **74.3 tokens/s** |

---

## 8. 遇到的问题与解决

| 问题 | 平台 | 原因 | 解决 |
|------|------|------|------|
| 链接 `.so` 时 `-fPIC` 报错 | NVIDIA | nvcc 编译的 `.o` 默认不带 `-fPIC` | `add_cuflags("--compiler-options", "-fPIC")` |
| 中间静态库链接失败 | NVIDIA | nvcc 和 g++ 中间产物不完全兼容 | `.cu` 文件直接编入共享库 target |
| `Cuda SDK not found!` | MACA | xmake 对 `.cu` 文件自动触发 CUDA 检测 | 用 `before_build` 手动调 mxcc |
| `std::byte` 未定义 | MACA | mxcc 默认非 C++17 | 加 `-std=c++17` |
| `std::exp` 编译错误 | 通用 | `std::` 是 host 函数 | 使用 device 函数 `expf` 等 |
| `INFINITY` 未声明 | MACA | 缺少头文件 | 加 `#include <cmath>` |
| PyTorch device 报错 | MACA | 定制版 PyTorch 将 CUDA 接口重定向到 MACA，device 用 `"cuda"` | test_utils.py 中 metax 映射为 `torch.device("cuda")` |

---

## 9. 学习总结

### CUDA/GPU 编程核心概念

| 概念 | 理解 |
|------|------|
| Grid → Block → Thread | GPU 线程三级层次，block 内可协作（共享内存、同步），block 间独立 |
| `__global__` / `__device__` | kernel 函数（CPU 启动 GPU 执行）和设备函数（GPU 内调用） |
| Grid-stride loop | 线程数少于数据量时循环处理，保证合并访问 |
| `__shared__` | Block 内共享高速缓存，用于归约和线程通信 |
| `__syncthreads()` | Block 内线程同步屏障 |
| 树形归约 | log2(n) 轮归约 n 个值，GPU 并行归约的基础模式 |
| 两阶段归约 | block 间无法同步，大规模归约拆为多 block 并行 + 单 block 合并 |

### 多平台适配经验

| 经验 | 说明 |
|------|------|
| 设备抽象层是关键 | `LlaisysRuntimeAPI` 函数指针结构体使上层代码完全设备无关 |
| CUDA 兼容 SDK 适配成本低 | 沐曦 MACA 的 API 与 CUDA 1:1 对应，核心工作是 API 名称替换 |
| 构建系统是最大挑战 | 算子代码适配简单，但 xmake 对非 NVIDIA GPU 编译器的支持需要变通 |
| 统一 template 模式 | cast 函数 + template kernel 使多平台代码结构一致，降低维护成本 |
| 测试环境差异 | 沐曦定制版 PyTorch（`2.4.0+metax3.0.0.3`）将 CUDA 接口重定向到 MACA SDK，torch device 仍用 `"cuda"` |

---

## 复现指南

### NVIDIA 环境

```bash
git clone https://github.com/Dnoob/llaisys.git
cd llaisys && git checkout project2/gpu-integration
xmake f --nv-gpu=y -cv && xmake && xmake install
pip install -e ./python/
python test/test_runtime.py --device nvidia
python test/test_infer.py --model <模型路径> --test --device nvidia
```

### 沐曦 MACA 环境

```bash
git clone https://github.com/Dnoob/llaisys.git
cd llaisys && git checkout project2/gpu-integration
export XMAKE_ROOT=y
xmake f --mx-gpu=y --root && xmake --root && xmake install --root
pip install -e ./python/
python test/test_runtime.py --device metax
python test/test_infer.py --model <模型路径> --test --device metax
```
