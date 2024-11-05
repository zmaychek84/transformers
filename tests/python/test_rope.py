#
# Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
#

import os

import numpy as np
import pytest
import torch

torch.random.manual_seed(123)

import llama_fast_attention
import ryzenai_torch_cpp
import time


def golden(query_states, key_states, position_ids, cos, sin):
    query_states, key_states = llama_fast_attention.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )
    return query_states, key_states


@pytest.mark.parametrize("m", [128, 256, 512, 1024, 2048])
def test_ryzenai_torch_cpp_rope(m):

    query_states = torch.rand((1, 32, m, 128)).to(torch.bfloat16)
    key_states = torch.rand((1, 32, m, 128)).to(torch.bfloat16)
    rotary_emb = llama_fast_attention.LlamaRotaryEmbeddingLocal()
    cos, sin = rotary_emb.forward(seq_len=m)

    position_ids = torch.zeros((1, m))
    for i in range(m):
        position_ids[0, i] = i
    position_ids = position_ids.to(torch.int64)

    query_states_ref, key_states_ref = golden(
        query_states, key_states, position_ids, cos, sin
    )

    # need to extract only the relevant parts (dependent on m) of the sin and cos vectors
    trig_1 = cos[position_ids].unsqueeze(1)
    trig_2 = sin[position_ids].unsqueeze(1)
    trig = torch.cat((trig_1, trig_2), dim=1)

    rope_op = ryzenai_torch_cpp.aie_rope_torch()

    st = time.perf_counter()
    query_states_npu = rope_op.execute(
        query_states.transpose(1, 2).reshape(32, m, 128).contiguous(), trig[0]
    ).clone()
    key_states_npu = rope_op.execute(
        key_states.transpose(1, 2).reshape(32, m, 128).contiguous(), trig[0]
    ).clone()
    print(f"Latency: {m} : {time.perf_counter() - st}")
    # NOTE: set with {'Absolute Difference MAX': 0.0078125, 'Relative Difference MAX': 0.9375}
    res_q = torch.allclose(
        query_states_ref, query_states_npu, atol=0.0078125, rtol=0.95
    )
    # NOTE: set with {'Absolute Difference MAX': 0.0078125, 'Relative Difference MAX': 0.96875}
    res_k = torch.allclose(key_states_ref, key_states_npu, atol=0.0078125, rtol=0.98)

    result = (res_q == True) and (res_k == True)
    if result:
        print(f"**** PASS: y vs x: {result}")
    else:
        print(f"**** FAIL: y vs x: {result}")
    assert result == True


def main():
    test_ryzenai_torch_cpp_rope(2048)


if __name__ == "__main__":
    main()
