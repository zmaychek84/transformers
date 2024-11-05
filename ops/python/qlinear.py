#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import gc
import logging
import os
import time
from collections import defaultdict
from typing import Tuple
import sys
import numpy as np
import torch
from torch import Tensor

torch.random.manual_seed(123)

import RyzenAI
import ryzenai_torch_cpp


class QLinear(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int = 1,
        out_features: int = 1,
        bias: bool = True,
        device=None,
        x_scale: float = 1,
        y_scale: float = 1,
        quant_mode="w8a8",
        profiler: bool = False,
        dtype="float32",
    ) -> None:
        factory_kwargs = {"device": "cpu", "dtype": None}
        super().__init__()
        self.device = device
        if dtype == "bfloat16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32
        self.in_features = in_features
        self.out_features = out_features
        self.quant_mode = quant_mode
        if self.quant_mode == "w8a8":
            self.weight = None
            self.accum_type = np.int32
            self.act_type = np.int8
            self.x_scaled_abs_max_fp = 128.0
            self.x_scaled_max = 127
            self.x_scaled_min = -128
        elif self.quant_mode == "w8a16":
            self.weight = None
            self.accum_type = np.int64
            self.act_type = np.int16
            self.x_scaled_abs_max_fp = 32768.0
            self.x_scaled_max = 32767
            self.x_scaled_min = -32768
        else:
            self.weight = torch.empty((in_features, out_features), **factory_kwargs)
        self.weight_q = None
        self.dev = os.getenv("DEVICE")
        if self.dev is None:
            print("DEVICE environment variable is not set")
            raise SystemExit
        # Perform checks and error out
        if self.quant_mode == "w8a16" and self.dev != "stx":
            print(f"{self.quant_mode} is not supported on {self.dev} platform")
            raise SystemExit
        if bias is True:
            self.bias = torch.empty(out_features, **factory_kwargs)
        else:
            self.register_parameter("bias", None)
            self.bias = None
        self.x_scale = np.array(x_scale, dtype=np.float32)
        self.y_scale = np.array(y_scale, dtype=np.float32)
        self.profiler = profiler
        if self.profiler:
            self.aie_time_execute_start = 0
            self.aie_time_execute_end = 0
            self.aie_compile_time_start = 0
            self.aie_compile_time_end = 0
            self.quant_time_start = 0
            self.quant_time_end = 0
            self.dequant_time_start = 0
            self.dequant_time_end = 0
            self.pre_time_start = 0
            self.pre_time_end = 0
            self.post_time_start = 0
            self.post_time_end = 0
            self.exec_pybind_time_start = 0
            self.exec_pybind_time_end = 0
            self.exec_c_time_start = 0
            self.exec_c_time_end = 0
            self.bias_add_time_start = 0
            self.bias_add_time_end = 0
        if self.device == "aie":
            if self.profiler:
                self.aie_compile_time_start = time.perf_counter()
            if not os.path.exists("./logs"):
                os.makedirs("./logs")
            self.init_aiegemm()
            if self.profiler:
                self.aie_compile_time_end = time.perf_counter()

        self.forward_dict = defaultdict(
            lambda: self.forward_prefill, {1: self.forward_token}
        )

    def init_aiegemm(self) -> None:
        if self.quant_mode == "w8a16":
            self.aiegemm = RyzenAI.qlinear_2_a16w8acc64("int16", "int8", "int64")
        else:
            self.aiegemm = RyzenAI.qlinear_2_a8w8acc32("int8", "int8", "int32")

    def __repr__(self):
        return f"ryzenAI.QLinear(in_features:{self.in_features}, out_features:{self.out_features}, bias:{True if self.bias is not None else False}, device:{self.device}, quant_mode:{self.quant_mode} )"

    def quantize_weights(self):
        if (self.quant_mode == "w8a8") or (self.quant_mode == "w8a16"):
            self.weight_q = torch.int_repr(self.weight_bias[0]).numpy().astype(np.int8)
            self.weight_q = np.ascontiguousarray(self.weight_q.transpose())
            self.y_scale = np.array(self.weight_bias[0].q_scale(), dtype=np.float32)
            if self.weight_bias[1] is not None:
                if self.weight_bias[1].data.dtype == torch.bfloat16:
                    self.bias = (
                        self.weight_bias[1]
                        .data.to(torch.float32)
                        .numpy()
                        .astype(np.float32)
                    )
                else:
                    self.bias = self.weight_bias[1].data.numpy().astype(np.float32)
            else:
                self.bias = None
            self.aiegemm.initialize_weights(self.weight_q)
            self.wshape = self.weight_q.shape
            del self.weight_q, self.weight_bias

            self.c_fp = np.empty((1, self.out_features), dtype=np.float32)
            self.c_token = np.zeros((1, self.out_features), dtype=self.accum_type)
            self.x_abs = np.empty((1, self.in_features), dtype=np.float32)
            self.x_round = np.empty((1, self.in_features), dtype=np.float32)
            self.x_scaled = np.empty((1, self.in_features), dtype=np.float32)
            self.x_clip = np.empty((1, self.in_features), dtype=np.float32)
            self.x_max = np.array(1.0, dtype=np.float32)

    def forward_prefill(self, x):
        x_scale = np.max(np.fabs(x)) / self.x_scaled_abs_max_fp
        x = np.clip(np.round(x / x_scale), self.x_scaled_min, self.x_scaled_max).astype(
            self.act_type, copy=False
        )

        c = np.zeros((x.shape[0], self.out_features), dtype=self.accum_type)
        self.aiegemm.execute(x, c)
        y = c.astype(np.float32, copy=False) * (x_scale * self.y_scale)
        if self.bias is not None:
            y = y + self.bias
        return y

    def forward_token(self, x):
        if self.profiler:
            self.exec_c_time_start = time.perf_counter()
        self.c_token.fill(0)
        if self.profiler:
            self.exec_c_time_end = time.perf_counter()

        if self.profiler:
            self.quant_time_start = time.perf_counter()
        np.abs(x, out=self.x_abs)
        np.divide(np.max(self.x_abs), self.x_scaled_abs_max_fp, out=self.x_scale)
        np.divide(x, self.x_scale, out=self.x_scaled)
        np.clip(
            np.round(self.x_scaled, out=self.x_round),
            self.x_scaled_min,
            self.x_scaled_max,
            out=self.x_clip,
        )
        if self.profiler:
            self.quant_time_end = time.perf_counter()

        if self.profiler:
            self.exec_pybind_time_start = time.perf_counter()
        self.aiegemm.execute(
            self.x_clip.astype(self.act_type, copy=False), self.c_token
        )
        if self.profiler:
            self.exec_pybind_time_end = time.perf_counter()

        if self.profiler:
            self.dequant_time_start = time.perf_counter()
        np.multiply(
            self.c_token.astype(np.float32, copy=False),
            np.multiply(self.x_scale, self.y_scale),
            out=self.c_fp,
        )
        if self.profiler:
            self.dequant_time_end = time.perf_counter()
        if self.profiler:
            self.bias_add_time_start = time.perf_counter()
        if self.bias is not None:
            np.add(self.c_fp, self.bias, out=self.c_fp)
        if self.profiler:
            self.bias_add_time_end = time.perf_counter()
        return self.c_fp

    def forward(self, x: Tensor) -> Tensor:
        if self.profiler:
            self.aie_time_execute_start = time.perf_counter()
        if len(x.shape) == 3:
            x = x.squeeze(0)
            has_batch = True
        else:
            has_batch = False
        y = self.forward_dict[x.shape[0]](x.numpy())
        if self.profiler:
            self.aie_time_execute_end = time.perf_counter()
            logging.critical(
                f"[PROFILE][AIE] {x.shape[0]} {x.shape[1]} {self.wshape[0]} {self.wshape[1]} {self.aie_compile_time_start} {self.aie_compile_time_end} {self.pre_time_start} {self.pre_time_end} {self.quant_time_start} {self.quant_time_end} {self.aie_time_execute_start} {self.aie_time_execute_end} {self.dequant_time_start} {self.dequant_time_end} {self.post_time_start} {self.post_time_end} {self.exec_pybind_time_start} {self.exec_pybind_time_end} {self.exec_c_time_start} {self.exec_c_time_end} {self.bias_add_time_start} {self.bias_add_time_end}"
            )
        if has_batch is False:
            return torch.from_numpy(y)
        else:
            return torch.from_numpy(np.expand_dims(y, axis=0))


class AIEGEMM:
    llama2_7b_shape_list = [
        [1, 4096, 4096, 128],
        [1, 4096, 11008, 128],
        [1, 11008, 4096, 128],
        [1, 4096, 12288, 128],
        [1, 4096, 32768, 32],
        [1, 4096, 32768, 128],
        [128, 4096, 4096, 128],
        [128, 4096, 11008, 128],
        [128, 11008, 4096, 128],
        [128, 4096, 12288, 128],
        [128, 4096, 32768, 32],
        [128, 4096, 32768, 128],
        [256, 4096, 4096, 128],
        [256, 4096, 11008, 128],
        [256, 11008, 4096, 128],
        [256, 4096, 12288, 128],
        [256, 4096, 32768, 32],
        [256, 4096, 32768, 128],
        [512, 4096, 4096, 128],
        [512, 4096, 11008, 128],
        [512, 11008, 4096, 128],
        [512, 4096, 12288, 128],
        [512, 4096, 32768, 32],
        [512, 4096, 32768, 128],
        [1024, 4096, 4096, 128],
        [1024, 4096, 11008, 128],
        [1024, 11008, 4096, 128],
        [1024, 4096, 12288, 128],
        [1024, 4096, 32768, 32],
        [1024, 4096, 32768, 128],
        [2048, 4096, 4096, 128],
        [2048, 4096, 11008, 128],
        [2048, 11008, 4096, 128],
        [2048, 4096, 12288, 128],
        [2048, 4096, 32768, 32],
        [2048, 4096, 32768, 128],
    ]
    llama3_8b_shape_list = [
        [1, 4096, 4096, 128],
        [1, 4096, 1024, 128],
        # [1, 4096, 6144, 128],
        [1, 4096, 14336, 128],
        [1, 4096, 28672, 128],
        [1, 14336, 4096, 128],
        [1, 4096, 129024, 32],
        [128, 4096, 4096, 128],
        [128, 4096, 1024, 128],
        # [128, 4096, 6144, 128],
        [128, 4096, 14336, 128],
        [128, 4096, 28672, 128],
        [128, 14336, 4096, 128],
        [128, 4096, 129024, 32],
        [256, 4096, 4096, 128],
        [256, 4096, 1024, 128],
        # [256, 4096, 6144, 128],
        [256, 4096, 14336, 128],
        [256, 4096, 28672, 128],
        [256, 14336, 4096, 128],
        [256, 4096, 129024, 32],
        [512, 4096, 4096, 128],
        [512, 4096, 1024, 128],
        # [512, 4096, 6144, 128],
        [512, 4096, 14336, 128],
        [512, 4096, 28672, 128],
        [512, 14336, 4096, 128],
        [512, 4096, 129024, 32],
        [1024, 4096, 4096, 128],
        [1024, 4096, 1024, 128],
        # [1024, 4096, 6144, 128],
        [1024, 4096, 14336, 128],
        [1024, 4096, 28672, 128],
        [1024, 14336, 4096, 128],
        [1024, 4096, 129024, 32],
        [2048, 4096, 4096, 128],
        [2048, 4096, 1024, 128],
        # [2048, 4096, 6144, 128],
        [2048, 4096, 14336, 128],
        [2048, 4096, 28672, 128],
        [2048, 14336, 4096, 128],
        [2048, 4096, 129024, 32],
    ]
    qwen15_7b_shape_list = [
        [1, 4096, 4096, 128],
        [1, 4096, 12288, 128],
        [1, 4096, 11008, 128],
        [1, 4096, 22016, 128],
        [1, 11008, 4096, 128],
        # [1, 4096, 153600, 128],
        [1, 4096, 153600, 32],  # txn size too large
        [128, 4096, 4096, 128],
        [128, 4096, 12288, 128],
        [128, 4096, 11008, 128],
        [128, 4096, 22016, 128],
        [128, 11008, 4096, 128],
        # [128, 4096, 153600, 128],
        [128, 4096, 153600, 32],  # txn size too large
        # [256, 4096, 4096, 128],
        # [256, 4096, 12288, 128],
        # [256, 4096, 11008, 128],
        # [256, 4096, 22016, 128],
        # [256, 11008, 4096, 128],
        # [256, 4096, 153600, 128],
        # [256, 4096, 153600, 32], # txn size too large
        # [512, 4096, 4096, 128],
        # [512, 4096, 12288, 128],
        # [512, 4096, 11008, 128],
        # [512, 4096, 22016, 128],
        # [512, 11008, 4096, 128],
        # [512, 4096, 153600, 128],
        # [512, 4096, 153600, 32], # txn size too large
        [1024, 4096, 4096, 128],
        [1024, 4096, 12288, 128],
        [1024, 4096, 11008, 128],
        [1024, 4096, 22016, 128],
        [1024, 11008, 4096, 128],
        # [1024, 4096, 153600, 128],
        [1024, 4096, 153600, 32],  # txn size too large
        [2048, 4096, 4096, 128],
        [2048, 4096, 12288, 128],
        [2048, 4096, 11008, 128],
        [2048, 4096, 22016, 128],
        [2048, 11008, 4096, 128],
        # [2048, 4096, 153600, 128],
        [2048, 4096, 153600, 32],  # txn size too large
    ]
    llama32_1b_shape_list = [
        [1, 2048, 2048, 128],
        [1, 2048, 512, 128],
        [1, 2048, 8192, 128],
        [1, 8192, 2048, 128],
        [1, 2048, 128256, 128],
        [128, 2048, 2048, 128],
        [128, 2048, 512, 128],
        [128, 2048, 8192, 128],
        [128, 8192, 2048, 128],
        [128, 2048, 128256, 128],
        [256, 2048, 2048, 128],
        [256, 2048, 512, 128],
        [256, 2048, 8192, 128],
        [256, 8192, 2048, 128],
        [256, 2048, 128256, 128],
        [512, 2048, 2048, 128],
        [512, 2048, 512, 128],
        [512, 2048, 8192, 128],
        [512, 8192, 2048, 128],
        [512, 2048, 128256, 128],
        [1024, 2048, 2048, 128],
        [1024, 2048, 512, 128],
        [1024, 2048, 8192, 128],
        [1024, 8192, 2048, 128],
        [1024, 2048, 128256, 128],
        [2048, 2048, 2048, 128],
        [2048, 2048, 512, 128],
        [2048, 2048, 8192, 128],
        [2048, 8192, 2048, 128],
        [2048, 2048, 128256, 128],
    ]
    llama32_3b_shape_list = [
        [1, 3072, 3072, 32],
        [1, 3072, 1024, 32],
        [1, 3072, 8192, 32],
        [1, 8192, 3072, 32],
        [1, 3072, 128256, 32],
        [128, 3072, 3072, 32],
        [128, 3072, 1024, 32],
        [128, 3072, 8192, 32],
        [128, 8192, 3072, 32],
        [128, 3072, 128256, 32],
        [256, 3072, 3072, 32],
        [256, 3072, 1024, 32],
        [256, 3072, 8192, 32],
        [256, 8192, 3072, 32],
        [256, 3072, 128256, 32],
        [512, 3072, 3072, 32],
        [512, 3072, 1024, 32],
        [512, 3072, 8192, 32],
        [512, 8192, 3072, 32],
        [512, 3072, 128256, 32],
        [1024, 3072, 3072, 32],
        [1024, 3072, 1024, 32],
        [1024, 3072, 8192, 32],
        [1024, 8192, 3072, 32],
        [1024, 3072, 128256, 32],
        [2048, 3072, 3072, 32],
        [2048, 3072, 1024, 32],
        [2048, 3072, 8192, 32],
        [2048, 8192, 3072, 32],
        [2048, 3072, 128256, 32],
    ]

    glm3_6b_8b_shape_list = [
        [1, 4096, 4096, 128],
        [1, 4096, 1024, 128],
        # [1, 4096, 6144, 128],
        [1, 4096, 14336, 128],
        [1, 4096, 28672, 128],
        [1, 14336, 4096, 128],
        [1, 4096, 129024, 32],
        [1, 4096, 4608, 128],
        [1, 4096, 27392, 128],
        [1, 13696, 4096, 128],
        [1, 4096, 65024, 128],
        [128, 4096, 4096, 128],
        [128, 4096, 1024, 128],
        # [128, 4096, 6144, 128],
        [128, 4096, 14336, 128],
        [128, 4096, 28672, 128],
        [128, 14336, 4096, 128],
        [128, 4096, 129024, 32],
        [128, 4096, 4608, 128],
        [128, 4096, 27392, 128],
        [128, 13696, 4096, 128],
        [128, 4096, 65024, 128],
        [256, 4096, 4096, 128],
        [256, 4096, 1024, 128],
        # [256, 4096, 6144, 128],
        [256, 4096, 14336, 128],
        [256, 4096, 28672, 128],
        [256, 14336, 4096, 128],
        [256, 4096, 129024, 32],
        [256, 4096, 4608, 128],
        [256, 4096, 27392, 128],
        [256, 13696, 4096, 128],
        [256, 4096, 65024, 128],
        [512, 4096, 4096, 128],
        [512, 4096, 1024, 128],
        # [512, 4096, 6144, 128],
        [512, 4096, 14336, 128],
        [512, 4096, 28672, 128],
        [512, 14336, 4096, 128],
        [512, 4096, 129024, 32],
        [512, 4096, 4608, 128],
        [512, 4096, 27392, 128],
        [512, 13696, 4096, 128],
        [512, 4096, 65024, 128],
        [1024, 4096, 4096, 128],
        [1024, 4096, 1024, 128],
        # [1024, 4096, 6144, 128],
        [1024, 4096, 14336, 128],
        [1024, 4096, 28672, 128],
        [1024, 14336, 4096, 128],
        [1024, 4096, 129024, 32],
        [1024, 4096, 4608, 128],
        [1024, 4096, 27392, 128],
        [1024, 13696, 4096, 128],
        [1024, 4096, 65024, 128],
        [2048, 4096, 4096, 128],
        [2048, 4096, 1024, 128],
        # [2048, 4096, 6144, 128],
        [2048, 4096, 14336, 128],
        [2048, 4096, 28672, 128],
        [2048, 14336, 4096, 128],
        [2048, 4096, 129024, 32],
        [2048, 4096, 4608, 128],
        [2048, 4096, 27392, 128],
        [2048, 13696, 4096, 128],
        [2048, 4096, 65024, 128],
    ]

    single_aiegemm = None
    gemm_torch = 0

    @classmethod
    def select_op_handle(cls, w_bit=4, dev="stx", mladf="2x4x4", model_name=""):
        if AIEGEMM.single_aiegemm is None:
            if (dev == "stx") and (mladf == "2x4x4"):
                AIEGEMM.gemm_torch = 1
            else:
                AIEGEMM.gemm_torch = 0
            if ("Llama-2-7b" in model_name) or ("llama-2-7b" in model_name):
                shape_list = AIEGEMM.llama2_7b_shape_list
            elif "Qwen1.5-7B" in model_name:
                shape_list = AIEGEMM.qwen15_7b_shape_list
            elif "Llama-3-8B" in model_name:
                shape_list = AIEGEMM.llama3_8b_shape_list
            elif "Llama-3.2-1B-Early" in model_name:
                shape_list = (
                    AIEGEMM.llama32_1b_shape_list
                )  # llama31_1b_shape_list  # RuntimeError: Invalid transaction binary string: mladfmatmulbias_mladf_2x4x4_v1_a16fw4acc16f_128_2048_2048_32
            elif "Llama-3.2-3B-Early" in model_name:
                shape_list = (
                    AIEGEMM.llama32_3b_shape_list
                )  # llama32_3b_shape_list # RuntimeError: Invalid transaction binary string: mladfmatmulbias_mladf_2x4x4_v1_a16fw4acc16f_128_3072_3072_32
            elif "Mistral" in model_name:
                shape_list = (
                    AIEGEMM.llama3_8b_shape_list
                )  # llama32_3b_shape_list # RuntimeError: Invalid transaction binary string
            elif "chatglm3" in model_name:
                shape_list = AIEGEMM.glm3_6b_8b_shape_list

            else:  # default list
                shape_list = AIEGEMM.llama2_7b_shape_list
            if (w_bit == 4) and (AIEGEMM.gemm_torch == 1):
                AIEGEMM.single_aiegemm = ryzenai_torch_cpp.aie_gemm_torch(
                    True, "bfloat16", "uint4", "bfloat16", shape_list
                )
            elif (w_bit == 3) and (AIEGEMM.gemm_torch == 1):
                AIEGEMM.single_aiegemm = ryzenai_torch_cpp.aie_gemm_torch(
                    True, "bfloat16", "int4", "bfloat16", shape_list
                )
            else:
                pass
                # print(
                #    f"Encountered unknown GEMM config ... exitting !! {w_bit} {dev} {mladf} {model_name} {AIEGEMM.gemm_torch}"
                # )
                # raise SystemExit


class QLinearPerGrp(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    wts_cnt = 0

    def __init__(
        self,
        in_features: int = 1,
        out_features: int = 1,
        bias: bool = False,
        device=None,
        w_bit: int = 4,
        group_size: int = 128,
        profiler: bool = False,
        model_name="",
    ) -> None:
        factory_kwargs = {"device": "cpu", "dtype": None}
        super().__init__()
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size
        self.weight = None
        self.qweight = None
        self.qzeros = None
        self.scales = None
        self.profiler = profiler
        self.wts_index = QLinearPerGrp.wts_cnt
        self.model_name = model_name
        self.biasexists = None

    def __repr__(self):
        if self.biasexists is None:
            self.biasexists = True if self.bias is not None else False
        return f"ryzenAI.QLinearPerGrp(in_features:{self.in_features}, out_features:{self.out_features}, bias:{self.biasexists}, device:{self.device}, w_bit:{self.w_bit}, model_name:{self.model_name}, gemm_torch:{AIEGEMM.gemm_torch}, group_size:{self.group_size} )"

    @torch.no_grad()
    def pack(self, qw):
        import math

        qcompact = torch.empty(
            qw.shape[0], math.ceil(qw.shape[1] / 2), dtype=torch.uint8
        )
        j = 0
        for i in range(qw.shape[1]):
            if i % 2 == 0:
                qcompact[:, j] = qw[:, i]
                qcompact[:, j] = qcompact[:, j] << 4
            else:
                qcompact[:, j] = torch.bitwise_or(qcompact[:, j], qw[:, i])
                j += 1
        return qcompact

    @torch.no_grad()
    def unpack(self, qcompact, k):
        qw = torch.empty((qcompact.shape[0], k), dtype=torch.int8)
        refmsb = torch.tensor(0xF0, dtype=torch.uint8)
        reflsb = torch.tensor(0x0F, dtype=torch.uint8)
        qw[:, 0::2] = (torch.bitwise_and(qcompact[:, :], refmsb) >> 4).to(torch.int8)
        qw[:, 1::2] = torch.bitwise_and(qcompact[:, :], reflsb).to(torch.int8)
        return qw

    @torch.no_grad()
    def quantize_weights(self):
        if (self.qweight is None) and (self.weight is not None):  # pergrp
            self.w_shape_orig = self.weight.shape
            w = self.weight.reshape(-1, self.group_size)

            # Calculate the maximum (\alpha) and minimum values (\beta) in the tensor.
            max_val = w.amax(dim=1, keepdim=True)
            min_val = w.amin(dim=1, keepdim=True)

            # Calculate the scale factor and zero point.
            max_int = 2**self.w_bit - 1
            self.scales = ((max_val - min_val).clamp(min=1e-5) / max_int).to(
                torch.bfloat16
            )
            assert self.scales.shape == max_val.shape
            self.qzeros = (
                (-torch.round(min_val / self.scales)).clamp_(0, max_int).to(torch.int8)
            )
            assert self.scales.shape == min_val.shape

            assert torch.isnan(self.scales).sum() == 0
            assert torch.isnan(w).sum() == 0

            # Quantize W: Map values in the range [\beta, \alpha] to lie within [0, 2^b - 1] (Formula 3)
            self.qweight = torch.clamp(
                torch.round(w / self.scales) + self.qzeros, 0, max_int
            ).to(torch.int8)

            assert (
                self.qweight.dim() == 2
                and self.qweight.size(0) == self.scales.size(0)
                and self.qweight.size(1) == self.group_size
            )

            self.qweight = self.qweight.reshape(self.w_shape_orig)
            self.qzeros = self.qzeros.reshape(
                (self.w_shape_orig[0], int(self.w_shape_orig[1] / self.group_size))
            ).to(torch.int8)
            self.scales = self.scales.reshape(
                (self.w_shape_orig[0], int(self.w_shape_orig[1] / self.group_size))
            )
            del self.weight, max_val, min_val, w
            self.wshape = self.qweight.shape
            self.qweight = self.pack(self.qweight)
            self.qzeros.requires_grad_(False)
            self.qweight.requires_grad_(False)
            self.scales.requires_grad_(False)
            gc.collect()
        else:
            print(f"Skipping - weights already quantized for this layer.")

    def initialize_parameters(self):
        AIEGEMM.select_op_handle(
            w_bit=self.w_bit,
            dev=os.environ.get("DEVICE"),
            mladf=os.environ.get("MLADF"),
            model_name=self.model_name,
        )
        if self.bias is not None:
            self.bias.data = self.bias.to(torch.bfloat16).to(torch.float32)
            self.biasexists = "True"
        else:
            self.bias = torch.zeros((self.out_features), dtype=torch.float32)
            self.biasexists = "False"

        self.qweight = self.unpack(self.qweight, self.qzeros.shape[1] * self.group_size)

        if self.device == "aie":
            self.qweight = self.qweight.transpose(0, 1)
            self.qzeros = self.qzeros.transpose(0, 1)
            self.scales = self.scales.to(torch.float).transpose(0, 1)
            if AIEGEMM.gemm_torch == 1:
                self.aiegemm = AIEGEMM.single_aiegemm
                self.wts_index = QLinearPerGrp.wts_cnt
                QLinearPerGrp.wts_cnt += 1
                d = dict()
                d["default_shape"] = int(
                    1
                )  # NOTE: here typo in DD, will fix once DD changes
                self.aiegemm.initialize_params(
                    self.qweight,
                    self.qzeros,
                    self.scales,
                    self.bias,
                    self.group_size,
                    d,
                )
            else:
                if self.w_bit == 3:
                    self.aiegemm = RyzenAI.qlinear_2_a16fw4acc32fo16f(
                        "bfloat16", "int4", "float32"
                    )
                else:
                    self.aiegemm = RyzenAI.qlinear_2_a16fw4acc32fo16f(
                        "bfloat16", "uint4", "float32"
                    )
                self.aiegemm.initialize_weights(
                    self.qweight.numpy(),
                    self.qzeros.numpy(),
                    self.scales.to(torch.float).numpy(),
                    self.bias.detach().numpy(),
                    self.group_size,
                )

            if not os.path.exists("./logs"):
                os.makedirs("./logs")
            self.c_token = torch.zeros(1, self.out_features, dtype=torch.bfloat16)
            self.forward_dict_aie_v0 = defaultdict(
                lambda: self.forward_aie_prefill_v0, {1: self.forward_aie_token_v0}
            )
            self.forward_dict_aie_mladf = defaultdict(
                lambda: self.forward_aie_prefill_mladf,
                {1: self.forward_aie_token_mladf},
            )
            if AIEGEMM.gemm_torch == 1:
                self.forward_dict_aie = self.forward_dict_aie_mladf
            else:
                self.forward_dict_aie = self.forward_dict_aie_v0
            self.forward_func = self.forward_aie
            del self.qweight, self.qzeros, self.scales, self.bias

        else:  # cpu
            self.weight = self.qweight - torch.repeat_interleave(
                self.qzeros, self.group_size, dim=1
            )
            self.weight = self.weight * torch.repeat_interleave(
                self.scales, self.group_size, dim=1
            )
            self.weight = self.weight.transpose(0, 1)
            if self.bias is not None:
                self.bias.data = self.bias.data.to(torch.bfloat16)
            self.forward_func = self.forward_cpu
            del self.qweight, self.qzeros, self.scales

        gc.collect()

    def forward_cpu(self, x: Tensor) -> Tensor:
        x = torch.matmul(x.to(torch.bfloat16), self.weight)
        if self.bias is not None:
            x = x + self.bias.to(torch.bfloat16)
        return x

    def forward_aie_token_v0(self, x: Tensor) -> Tensor:
        self.aiegemm.execute(x.view(torch.int16), self.c_token.view(torch.int16))
        return self.c_token

    def forward_aie_prefill_v0(self, x: Tensor) -> Tensor:
        c = torch.empty((x.shape[0], self.out_features), dtype=torch.bfloat16)
        self.aiegemm.execute(x.view(torch.int16), c.view(torch.int16))
        return c

    def forward_aie_token_mladf(self, x: Tensor) -> Tensor:
        self.aiegemm.execute_aie(x, self.c_token, self.wts_index)
        return self.c_token

    def forward_aie_prefill_mladf(self, x: Tensor) -> Tensor:
        c = torch.empty((x.shape[0], self.out_features), dtype=torch.bfloat16)
        self.aiegemm.execute_aie(x, c, self.wts_index)
        return c

    def forward_aie(self, x: Tensor) -> Tensor:
        return self.forward_dict_aie[x.shape[0]](x)

    def forward(self, x: Tensor, zerocpy=False, rettensor=True) -> Tensor:
        if zerocpy or rettensor == False:
            return self.aiegemm.execute_aie_bo(x, self.wts_index, zerocpy, rettensor)
        if len(x.shape) == 3:
            has_batch = True
        else:
            x = x.unsqueeze(0)
            has_batch = False
        y = torch.empty(
            (x.shape[0], x.shape[1], self.out_features), dtype=torch.bfloat16
        )
        for i in range(x.shape[0]):
            y[i] = self.forward_func(x[i])
        if has_batch is False:
            return y.squeeze(0)
        else:
            return y
