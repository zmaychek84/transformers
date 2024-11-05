#
# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
#

import os

import numpy as np
import pytest
import torch

torch.random.manual_seed(123)

prompt_lengths = [
    1,
    128,
    256,
    384,
    512,
    640,
    768,
    800,
    1024,
    1152,
    1280,
    1408,
    1536,
    1664,
    1792,
    1920,
    2048,
]


@pytest.mark.parametrize("m", prompt_lengths)
def test_ryzenai_aie_elem_mult(m):
    inpshape = (m, 14336)
    symmetric_interval_upper_bound = 42
    x = (symmetric_interval_upper_bound * 2) * torch.rand(
        inpshape
    ) - symmetric_interval_upper_bound
    y = (symmetric_interval_upper_bound * 2) * torch.rand(
        inpshape
    ) - symmetric_interval_upper_bound
    xbf = x.to(torch.bfloat16)
    ybf = y.to(torch.bfloat16)

    if os.environ.get("MLADF") and os.environ.get("DEVICE") == "stx":
        import ryzenai_torch_cpp

        op = ryzenai_torch_cpp.aie_elemw_mul_torch()
        out = op.execute(xbf, ybf)
    else:
        o = x * y
        out = o.to(torch.bfloat16)

    ref = x * y
    result = torch.allclose(out, ref.to(torch.bfloat16), rtol=0.02, atol=0.05)

    if result:
        print(f"**** PASS: y vs x: {result}")
    else:
        print(f"**** FAIL: y vs x: {result}")
    assert result == True


def main():
    # For debugging
    test_ryzenai_aie_elem_mult(128)


if __name__ == "__main__":
    main()
