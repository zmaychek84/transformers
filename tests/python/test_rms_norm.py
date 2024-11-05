#
# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
#

import os

import numpy as np
import pytest
import torch

torch.random.manual_seed(123)
import ryzenai_torch_cpp

from transformers.models.llama.modeling_llama import LlamaRMSNorm
from test_utils import tensor_statistics


@pytest.mark.parametrize(
    "m",
    [
        1,
        128,
        256,
        384,
        512,
        640,
        768,
        1024,
        1152,
        1280,
        1408,
        1536,
        1664,
        1792,
        1920,
        2048,
    ],
)
def test_ryzenai_torch_cpp_rmsnorm(m):
    hidden_size = 4096
    golden = LlamaRMSNorm(hidden_size=hidden_size)

    x = (torch.rand((m, hidden_size)) - 0.5) * 10

    ref = golden(x)

    # offload to NPU
    op = ryzenai_torch_cpp.aie_rmsnorm_torch()
    rms_wts = golden.weight.to(torch.bfloat16)

    x_bf16 = x.to(torch.bfloat16)
    npu = op.execute(x_bf16, rms_wts, False, True)
    result = torch.allclose(ref.to(float), npu.to(float), rtol=0.04, atol=0.04)
    if result:
        print(f"**** PASS: y vs x: {result}")
    else:
        print(f"**** FAIL: y vs x: {result}")
        stats = tensor_statistics(ref.to(float), npu.to(float))
        print(stats)
    assert result == True


def main():
    # For debugging
    test_ryzenai_torch_cpp_rmsnorm(128)


if __name__ == "__main__":
    main()
