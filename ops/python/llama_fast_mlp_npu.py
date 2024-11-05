#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import copy
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from qmodule import WQLinear
import llama_fast_rmsnorm


fast_norm = 0
fast_decode = 0
import os

if os.environ.get("MLADF") and os.environ.get("DEVICE") == "stx":
    import sys

    for name in sys.argv:
        if "--fast_norm" in name:
            fast_norm = 1
        if "--fast_decode" in name:
            fast_decode = 1


class FastMLP_NpuExecutor:
    init_done = False
    fast_norm = False
    fast_decoder = False

    mlp_npu = None

    @classmethod
    def init_npuexecutor(cls, dev="stx", mladf="2x4x4"):
        if not cls.init_done:
            cls.init_done = True

            if mladf == "2x4x4" and dev == "stx":
                import sys
                import ryzenai_torch_cpp

                for name in sys.argv:
                    if "--fast_norm" in name:
                        cls.fast_norm = True
                    if "--fast_decoder" in name:
                        cls.fast_decoder = True
                cls.mlp_npu = ryzenai_torch_cpp.aie_mlp_npu_torch()
            else:
                raise RuntimeError(
                    f"LlamaFastMLP needs os environment DEVICE=stx and MLADF=2x4x4"
                )
        else:
            pass


def unpack(qcompact, k):
    qw = torch.empty((qcompact.shape[0], k), dtype=torch.int8)
    refmsb = torch.tensor(0xF0, dtype=torch.uint8)
    reflsb = torch.tensor(0x0F, dtype=torch.uint8)
    qw[:, 0::2] = (torch.bitwise_and(qcompact[:, :], refmsb) >> 4).to(torch.int8)
    qw[:, 1::2] = torch.bitwise_and(qcompact[:, :], reflsb).to(torch.int8)
    return qw


class LlamaFastMLP(torch.nn.Module):
    def __init__(self, precision):
        super().__init__()
        self.precision = precision
        self.gate_proj = None
        self.up_proj = None
        self.down_proj = None
        self.act_fn = None  # torch.nn.SiLU(inplace=True)

    def init_fastmlp(self, fuse_threshold=128):
        FastMLP_NpuExecutor.init_npuexecutor(
            dev=os.environ.get("DEVICE"),
            mladf=os.environ.get("MLADF"),
        )
        if self.gate_proj.bias is None:
            self.gate_proj.bias = torch.zeros((1, self.gate_proj.qweight.size()[0])).to(
                torch.bfloat16
            )
        if self.down_proj.bias is None:
            self.down_proj.bias = torch.zeros((1, self.gate_proj.qweight.size()[0])).to(
                torch.bfloat16
            )
        if self.up_proj.bias is None:
            self.up_proj.bias = torch.zeros((1, self.gate_proj.qweight.size()[0])).to(
                torch.bfloat16
            )
        self.gate_proj.qweight = unpack(
            self.gate_proj.qweight, self.gate_proj.in_features
        )
        self.up_proj.qweight = unpack(self.up_proj.qweight, self.up_proj.in_features)
        self.down_proj.qweight = unpack(
            self.down_proj.qweight, self.down_proj.in_features
        )
        gate_qw = self.gate_proj.qweight.transpose(0, 1).contiguous()
        gate_qz = self.gate_proj.qzeros.transpose(0, 1).contiguous()
        gate_scale = self.gate_proj.scales.transpose(0, 1).to(torch.float).contiguous()
        gate_bias = self.gate_proj.bias.to(torch.float).contiguous()
        down_qw = self.down_proj.qweight.transpose(0, 1).contiguous()
        down_qz = self.down_proj.qzeros.transpose(0, 1).contiguous()
        down_scale = self.down_proj.scales.transpose(0, 1).to(torch.float).contiguous()
        down_bias = self.down_proj.bias.to(torch.float).contiguous()
        up_qw = self.up_proj.qweight.transpose(0, 1).contiguous()
        up_qz = self.up_proj.qzeros.transpose(0, 1).contiguous()
        up_scale = self.up_proj.scales.transpose(0, 1).to(torch.float).contiguous()
        up_bias = self.up_proj.bias.to(torch.float).contiguous()
        FastMLP_NpuExecutor.mlp_npu.initialize_params(
            gate_qw,
            gate_qz,
            gate_scale,
            gate_bias,
            self.gate_proj.group_size,
            down_qw,
            down_qz,
            down_scale,
            down_bias,
            self.down_proj.group_size,
            up_qw,
            up_qz,
            up_scale,
            up_bias,
            self.up_proj.group_size,
        )
        if FastMLP_NpuExecutor.fast_norm and FastMLP_NpuExecutor.fast_decoder:
            self.fuse_threshold = fuse_threshold
            self.inputs = llama_fast_rmsnorm.op.get_address()

        else:
            self.fuse_threshold = 4096  # rmsnorm on CPU

        del self.gate_proj, self.up_proj, self.down_proj, self.act_fn

    def forward(self, x, q_len=0):
        if FastMLP_NpuExecutor.fast_norm and FastMLP_NpuExecutor.fast_decoder:
            if q_len >= self.fuse_threshold:
                self.inputs[0] = x.data.numpy()[0][0][0]
                self.inputs[2] = q_len
                self.inputs[3] = x.data.numpy()[0][0][3]
            else:
                self.inputs[2] = x.size()[1]
                self.inputs[3] = x.size()[2]
        else:
            self.inputs = [0, 0, x.size()[1], x.size()[2]]
        return FastMLP_NpuExecutor.mlp_npu.execute(
            x, self.inputs, self.fuse_threshold, self.fuse_threshold
        )
