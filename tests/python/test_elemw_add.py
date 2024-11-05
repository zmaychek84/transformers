#
# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
#

import os

import numpy as np
import pytest
import torch

torch.random.manual_seed(123)
import ryzenai_torch_cpp

dims = [2048]


@pytest.mark.parametrize("m", dims)
def test_ryzenai_aie_elemw_add(m):
    inpshape = (m, 4096)
    symmetric_interval_upper_bound = 42
    x = (symmetric_interval_upper_bound * 2) * torch.rand(inpshape).to(
        torch.bfloat16
    ) - symmetric_interval_upper_bound
    y = (symmetric_interval_upper_bound * 2) * torch.rand(inpshape).to(
        torch.bfloat16
    ) - symmetric_interval_upper_bound

    op = ryzenai_torch_cpp.aie_elemw_add_torch()
    out = op.execute(x, y, 0, True)

    ref = x + y
    result = torch.allclose(out, ref)

    err = (out - ref).abs()
    print(f"err: {err.max()}")
    if result:
        print(f"**** PASS: y vs x: {result}")
    else:
        print(f"**** FAIL: y vs x: {result}")

    assert result == True


def main():
    # For debugging
    test_ryzenai_aie_elemw_add(4096)


if __name__ == "__main__":
    main()
