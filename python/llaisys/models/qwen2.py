from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType, DataType
from ..libllaisys import llaisysTensor_t

from pathlib import Path
import ctypes
from ctypes import (
    Structure, POINTER, c_int, c_size_t, c_float, c_int64, c_void_p
)
import json


# ctypes 结构体
class LlaisysQwen2Meta(Structure):
    _fields_ = [
        ("dtype", c_int),
        ("nlayer", c_size_t),
        ("hs", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("di", c_size_t),
        ("maxseq", c_size_t),
        ("voc", c_size_t),
        ("epsilon", c_float),
        ("theta", c_float),
        ("end_token", c_int64),
    ]


class LlaisysQwen2Weights(Structure):
    _fields_ = [
        ("in_embed", llaisysTensor_t),
        ("out_embed", llaisysTensor_t),
        ("out_norm_w", llaisysTensor_t),
        ("attn_norm_w", POINTER(llaisysTensor_t)),
        ("attn_q_w", POINTER(llaisysTensor_t)),
        ("attn_q_b", POINTER(llaisysTensor_t)),
        ("attn_k_w", POINTER(llaisysTensor_t)),
        ("attn_k_b", POINTER(llaisysTensor_t)),
        ("attn_v_w", POINTER(llaisysTensor_t)),
        ("attn_v_b", POINTER(llaisysTensor_t)),
        ("attn_o_w", POINTER(llaisysTensor_t)),
        ("mlp_norm_w", POINTER(llaisysTensor_t)),
        ("mlp_gate_w", POINTER(llaisysTensor_t)),
        ("mlp_up_w", POINTER(llaisysTensor_t)),
        ("mlp_down_w", POINTER(llaisysTensor_t)),
    ]


# 绑定 C API
LIB_LLAISYS.llaisysQwen2ModelCreate.argtypes = [
    POINTER(LlaisysQwen2Meta), c_int, POINTER(c_int), c_int
]
LIB_LLAISYS.llaisysQwen2ModelCreate.restype = c_void_p

LIB_LLAISYS.llaisysQwen2ModelDestroy.argtypes = [c_void_p]
LIB_LLAISYS.llaisysQwen2ModelDestroy.restype = None

LIB_LLAISYS.llaisysQwen2ModelWeights.argtypes = [c_void_p]
LIB_LLAISYS.llaisysQwen2ModelWeights.restype = POINTER(LlaisysQwen2Weights)

LIB_LLAISYS.llaisysQwen2ModelInfer.argtypes = [c_void_p, POINTER(c_int64), c_size_t]
LIB_LLAISYS.llaisysQwen2ModelInfer.restype = c_int64


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        import safetensors
        import numpy as np
        import ml_dtypes

        self._dtype_map = {
            np.dtype("float32"): int(DataType.F32),
            np.dtype("float16"): int(DataType.F16),
            np.dtype(ml_dtypes.bfloat16): int(DataType.BF16),
        }

        model_path = Path(model_path)

        with open(model_path / "config.json") as f:
            config = json.load(f)

        meta = LlaisysQwen2Meta()
        meta.dtype = int(DataType.BF16)
        meta.nlayer = config["num_hidden_layers"]
        meta.hs = config["hidden_size"]
        meta.nh = config["num_attention_heads"]
        meta.nkvh = config["num_key_value_heads"]
        meta.dh = meta.hs // meta.nh
        meta.di = config["intermediate_size"]
        meta.maxseq = config.get("max_position_embeddings", 131072)
        meta.voc = config["vocab_size"]
        meta.epsilon = config.get("rms_norm_eps", 1e-6)
        meta.theta = config.get("rope_theta", 10000.0)
        eos = config.get("eos_token_id", 151643)
        meta.end_token = eos[0] if isinstance(eos, list) else eos

        self._meta = meta
        self._device = device

        # 创建模型
        device_ids = (c_int * 1)(0)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(meta), int(device), device_ids, 1
        )

        # 获取权重指针
        weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model)
        self._weights = weights_ptr.contents

        # 加载权重
        self._tensors = []
        has_lm_head = False

        for file in sorted(model_path.glob("*.safetensors")):
            data_ = safetensors.safe_open(file, framework="numpy", device="cpu")
            for name_ in data_.keys():
                tensor_np = data_.get_tensor(name_)
                self._load_weight(name_, tensor_np, device)
                if name_ == "lm_head.weight":
                    has_lm_head = True

        if not has_lm_head:
            self._weights.out_embed = self._weights.in_embed

    def _load_weight(self, name, tensor_np, device):
        shape = tensor_np.shape
        shape_arr = (c_size_t * len(shape))(*shape)
        dtype = self._dtype_map.get(tensor_np.dtype, int(DataType.BF16))

        t = LIB_LLAISYS.tensorCreate(
            shape_arr, c_size_t(len(shape)), dtype, int(self._device), c_int(0)
        )
        LIB_LLAISYS.tensorLoad(t, tensor_np.ctypes.data_as(c_void_p))
        self._tensors.append(t)

        w = self._weights

        if name == "model.embed_tokens.weight":
            w.in_embed = t
        elif name == "lm_head.weight":
            w.out_embed = t
        elif name == "model.norm.weight":
            w.out_norm_w = t
        elif ".layers." in name:
            parts = name.split(".")
            layer = int(parts[2])

            if name.endswith("input_layernorm.weight"):
                w.attn_norm_w[layer] = t
            elif name.endswith("q_proj.weight"):
                w.attn_q_w[layer] = t
            elif name.endswith("q_proj.bias"):
                w.attn_q_b[layer] = t
            elif name.endswith("k_proj.weight"):
                w.attn_k_w[layer] = t
            elif name.endswith("k_proj.bias"):
                w.attn_k_b[layer] = t
            elif name.endswith("v_proj.weight"):
                w.attn_v_w[layer] = t
            elif name.endswith("v_proj.bias"):
                w.attn_v_b[layer] = t
            elif name.endswith("o_proj.weight"):
                w.attn_o_w[layer] = t
            elif name.endswith("post_attention_layernorm.weight"):
                w.mlp_norm_w[layer] = t
            elif name.endswith("gate_proj.weight"):
                w.mlp_gate_w[layer] = t
            elif name.endswith("up_proj.weight"):
                w.mlp_up_w[layer] = t
            elif name.endswith("down_proj.weight"):
                w.mlp_down_w[layer] = t

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        tokens = list(inputs)

        # Prefill
        input_arr = (c_int64 * len(tokens))(*tokens)
        next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
            self._model, input_arr, c_size_t(len(tokens))
        )
        tokens.append(next_token)

        # Decode
        steps = 0
        while True:
            if max_new_tokens and steps >= max_new_tokens - 1:
                break
            if next_token == self._meta.end_token:
                break

            single = (c_int64 * 1)(next_token)
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model, single, c_size_t(1)
            )
            tokens.append(next_token)
            steps += 1

        return tokens

    def __del__(self):
        if hasattr(self, "_model") and self._model:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None
