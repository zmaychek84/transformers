#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import gc
import logging
import os
import time

import numpy as np
import psutil
import pytest
import qlinear
import ryzenai_torch_cpp
import torch

# m, k, n, k
# due to the DD runtime error "No space for instruction buffer" for mladfmatmulbias,
# now limit the used shapes for llama in that op, so disable some cases here and
# maybe enable them in the future.
llama2_7b = [
    {"shape": ((1, 4096), (4096, 4096)), "group_size": 128, "target_err_percent": 1.0},
    {"shape": ((1, 4096), (11008, 4096)), "group_size": 128, "target_err_percent": 1.0},
    {"shape": ((1, 4096), (12288, 4096)), "group_size": 128, "target_err_percent": 1.0},
    {"shape": ((1, 4096), (32768, 4096)), "group_size": 32, "target_err_percent": 1.0},
    {
        "shape": ((1, 11008), (4096, 11008)),
        "group_size": 128,
        "target_err_percent": 1.0,
    },
    {
        "shape": ((128, 4096), (4096, 4096)),
        "group_size": 128,
        "target_err_percent": 1.0,
    },
    {
        "shape": ((128, 4096), (11008, 4096)),
        "group_size": 128,
        "target_err_percent": 1.0,
    },
    {
        "shape": ((128, 4096), (12288, 4096)),
        "group_size": 128,
        "target_err_percent": 1.0,
    },
    {
        "shape": ((128, 4096), (32768, 4096)),
        "group_size": 32,
        "target_err_percent": 1.0,
    },
    {
        "shape": ((128, 11008), (4096, 11008)),
        "group_size": 128,
        "target_err_percent": 50.0,
    },
    {
        "shape": ((256, 4096), (4096, 4096)),
        "group_size": 128,
        "target_err_percent": 1.0,
    },
    {
        "shape": ((256, 4096), (11008, 4096)),
        "group_size": 128,
        "target_err_percent": 1.0,
    },
    {
        "shape": ((256, 4096), (12288, 4096)),
        "group_size": 128,
        "target_err_percent": 1.0,
    },
    {
        "shape": ((256, 4096), (32768, 4096)),
        "group_size": 32,
        "target_err_percent": 1.0,
    },
    {
        "shape": ((256, 11008), (4096, 11008)),
        "group_size": 128,
        "target_err_percent": 10.0,
    },
    {
        "shape": ((512, 4096), (4096, 4096)),
        "group_size": 128,
        "target_err_percent": 1.0,
    },
    {
        "shape": ((512, 4096), (11008, 4096)),
        "group_size": 128,
        "target_err_percent": 1.0,
    },
    {
        "shape": ((512, 4096), (12288, 4096)),
        "group_size": 128,
        "target_err_percent": 1.0,
    },
    {
        "shape": ((512, 4096), (32768, 4096)),
        "group_size": 32,
        "target_err_percent": 1.0,
    },
    {
        "shape": ((512, 11008), (4096, 11008)),
        "group_size": 128,
        "target_err_percent": 10.0,
    },
    {
        "shape": ((1024, 4096), (4096, 4096)),
        "group_size": 128,
        "target_err_percent": 1.0,
    },
    {
        "shape": ((1024, 4096), (11008, 4096)),
        "group_size": 128,
        "target_err_percent": 1.0,
    },
    {
        "shape": ((1024, 4096), (12288, 4096)),
        "group_size": 128,
        "target_err_percent": 1.0,
    },
    {
        "shape": ((1024, 4096), (32768, 4096)),
        "group_size": 32,
        "target_err_percent": 1.0,
    },
    {
        "shape": ((1024, 11008), (4096, 11008)),
        "group_size": 128,
        "target_err_percent": 10.0,
    },
    {
        "shape": ((2048, 4096), (4096, 4096)),
        "group_size": 128,
        "target_err_percent": 1.0,
    },
    {
        "shape": ((2048, 4096), (11008, 4096)),
        "group_size": 128,
        "target_err_percent": 1.0,
    },
    {
        "shape": ((2048, 4096), (12288, 4096)),
        "group_size": 128,
        "target_err_percent": 1.0,
    },
    {
        "shape": ((2048, 4096), (32768, 4096)),
        "group_size": 32,
        "target_err_percent": 1.0,
    },
    {
        "shape": ((2048, 11008), (4096, 11008)),
        "group_size": 128,
        "target_err_percent": 50.0,
    },
]

shape_list = [
    [1, 4096, 4096, 128],
    [1, 4096, 12288, 128],
    [1, 4096, 32768, 32],
    [128, 4096, 4096, 128],
    [128, 4096, 12288, 128],
    [128, 4096, 32768, 32],
    [256, 4096, 4096, 128],
    [256, 4096, 12288, 128],
    [256, 4096, 32768, 32],
    [512, 4096, 4096, 128],
    [512, 4096, 12288, 128],
    [512, 4096, 32768, 32],
    [1024, 4096, 4096, 128],
    [1024, 4096, 12288, 128],
    [1024, 4096, 32768, 32],
    [2048, 4096, 4096, 128],
    [2048, 4096, 12288, 128],
    [2048, 4096, 32768, 32],
]


@pytest.mark.parametrize("b_dtype", ["uint4"])
def test_QLinear_pergrp_gemm(b_dtype):
    a_dtype = c_dtype = "bfloat16"
    gemm_aie = ryzenai_torch_cpp.aie_gemm_torch(
        True, a_dtype, b_dtype, c_dtype, shape_list
    )
    cnt = 0
    for xyshape in llama2_7b:
        inp_shape, weight_shape = xyshape["shape"]
        grpsize = xyshape["group_size"]
        target_err_percent = xyshape["target_err_percent"]

        print("")
        print("M: ", inp_shape[0])
        print("K: ", inp_shape[1])
        print("N: ", weight_shape[0])
        print("G: ", grpsize)

        w_bit_dict = {"uint4": 4, "int4": 3}

        torch.random.manual_seed(123)
        np.random.seed(123)

        x_min, x_max = -42.0, 42.0
        x = np.random.uniform(low=x_min, high=x_max, size=inp_shape)
        x = torch.tensor(x).to(torch.bfloat16)

        y_min, y_max = -1.0, 1.0
        y = np.random.uniform(low=y_min, high=y_max, size=weight_shape).astype(
            np.float32
        )

        bias = torch.rand(y.shape[0], dtype=torch.float32)

        gemm_cpu = qlinear.QLinearPerGrp(
            in_features=inp_shape[1],
            out_features=weight_shape[0],
            bias=True,
            device="cpu",
            w_bit=w_bit_dict[b_dtype],
            group_size=grpsize,
        )

        gemm_cpu.weight = torch.from_numpy(y)
        gemm_cpu.bias = bias
        gemm_cpu.quantize_weights()
        d = dict()
        d["default_shape"] = int(1)  # NOTE: here typo in DD, will fix once DD changes
        gemm_aie.initialize_params(
            gemm_cpu.qweight,
            gemm_cpu.qzeros,
            gemm_cpu.scales,
            gemm_cpu.bias,
            gemm_cpu.group_size,
            d,
        )

        x_cpu = gemm_cpu(x)
        x_tor = torch.empty((x.shape[0], weight_shape[0]), dtype=torch.bfloat16)
        gemm_aie.execute_aie(x, x_tor, cnt)
        cnt = cnt + 1

        # print(x_cpu.shape)
        # print(x_tor.shape)
        assert x_cpu.shape == x_tor.shape

        print(x_cpu)
        print(x_tor)

        result = torch.allclose(x_cpu, x_tor, target_err_percent / 100, 45)

        assert result == True


if __name__ == "__main__":
    test_QLinear_pergrp_gemm()
