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
def test_ryzenai_torch_cpp_silu(m):
    symmetric_interval_upper_bound = 42
    x = (symmetric_interval_upper_bound * 2) * torch.rand(
        m, 11008
    ) - symmetric_interval_upper_bound

    xbf = x.to(torch.bfloat16)

    ref = torch.nn.functional.silu(x)

    if os.environ.get("MLADF") and os.environ.get("DEVICE") == "stx":
        import ryzenai_torch_cpp

        op = ryzenai_torch_cpp.aie_silu_torch()
        out = op.execute(xbf)
    else:
        print("Not implemented in NPU")
        ot = torch.nn.functional.silu(x)
        out = ot.to(torch.bfloat16)

    result = torch.allclose(out, ref.to(torch.bfloat16), atol=0.8e-1, rtol=0.5e-1)
    if result:
        print(f"**** PASS: y vs x: {result}")
    else:
        print(f"**** FAIL: y vs x: {result}")
        stats = tensor_statistics(ref, out.to(float))
        print(stats)
    assert result == True


def main():
    # For debugging
    test_ryzenai_torch_cpp_silu(128)


if __name__ == "__main__":
    main()
