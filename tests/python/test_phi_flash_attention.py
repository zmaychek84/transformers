#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import copy
import os
import sys

import pytest
import torch
from phi_flash_attention import Phi3FlashAttentionPlus
from phi_flash_attention import PhiFlashAttentionPlus as Phi2FlashAttentionPlus
from test_utils import generate_attention_test_input, pergrp_processing, to_skip

from transformers import PhiConfig as Phi2Config
from transformers.models.phi.modeling_phi import PhiAttention as Phi2Attention

sys.path.append(os.getenv("PYTORCH_AIE_PATH") + "/models/llm/")
from phi3.configuration_phi3 import Phi3Config
from phi3.modeling_phi3 import Phi3Attention

torch.manual_seed(0)


phi_attention_shapes = [
    ("microsoft/phi-2", 1, 1),
    ("microsoft/phi-2", 1, 128),
    ("microsoft/phi-2", 1, 512),
    ("microsoft/phi-2", 1, 717),
    ("microsoft/phi-2", 1, 2000),
    ("microsoft/phi-2", 1, 2545),
    ("microsoft/phi-2", 1, 4000),
    ("microsoft/Phi-3-mini-4k-instruct", 1, 1),
    ("microsoft/Phi-3-mini-4k-instruct", 1, 128),
    ("microsoft/Phi-3-mini-4k-instruct", 1, 512),
    ("microsoft/Phi-3-mini-4k-instruct", 1, 717),
    ("microsoft/Phi-3-mini-4k-instruct", 1, 1024),
    ("microsoft/Phi-3-mini-4k-instruct", 1, 1536),
    ("microsoft/Phi-3-mini-4k-instruct", 1, 1801),
    ("microsoft/Phi-3-mini-4k-instruct", 1, 2000),
    ("microsoft/Phi-3-mini-4k-instruct", 1, 2545),
    ("microsoft/Phi-3-mini-4k-instruct", 1, 3000),
    ("microsoft/Phi-3-mini-4k-instruct", 1, 3333),
    ("microsoft/Phi-3-mini-4k-instruct", 1, 4000),
    # ("microsoft/phi-2", 4, 1   ),
    # ("microsoft/phi-2", 4, 128 ),
    # ("microsoft/phi-2", 4, 512 ),
    # ("microsoft/phi-2", 4, 717 ),
    # ("microsoft/phi-2", 4, 2000),
    # ("microsoft/phi-2", 4, 2545),
    # ("microsoft/phi-2", 4, 4000),
    # ("microsoft/Phi-3-mini-4k-instruct", 4, 1   ),
    # ("microsoft/Phi-3-mini-4k-instruct", 4, 128 ),
    # ("microsoft/Phi-3-mini-4k-instruct", 4, 512 ),
    # ("microsoft/Phi-3-mini-4k-instruct", 4, 717 ),
    # ("microsoft/Phi-3-mini-4k-instruct", 4, 1024),
    # ("microsoft/Phi-3-mini-4k-instruct", 4, 1536),
    # ("microsoft/Phi-3-mini-4k-instruct", 4, 1801),
    # ("microsoft/Phi-3-mini-4k-instruct", 4, 2000),
    # ("microsoft/Phi-3-mini-4k-instruct", 4, 2545),
    # ("microsoft/Phi-3-mini-4k-instruct", 4, 3000),
    # ("microsoft/Phi-3-mini-4k-instruct", 4, 3333),
    # ("microsoft/Phi-3-mini-4k-instruct", 4, 4000),
]


@pytest.mark.parametrize("phi_attention_shape", phi_attention_shapes)
@pytest.mark.parametrize("device", ["aie"])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.quant_combo_skip
def test_phi_flash_attention(phi_attention_shape, device, dtype, quant_mode, w_bit):
    to_skip(device, dtype, quant_mode)
    model_name, batch_size, sequence_length = phi_attention_shape
    is_phi_2 = "phi-2" in model_name
    if is_phi_2:
        PhiConfig, PhiAttention, PhiFlashAttentionPlus = (
            Phi2Config,
            Phi2Attention,
            Phi2FlashAttentionPlus,
        )
    else:  # phi-3
        PhiConfig, PhiAttention, PhiFlashAttentionPlus = (
            Phi3Config,
            Phi3Attention,
            Phi3FlashAttentionPlus,
        )

    config = PhiConfig.from_pretrained(model_name)
    attn = PhiAttention(
        config=config,
        layer_idx=0,
    )
    attn_fa = PhiFlashAttentionPlus(
        config=config,
        layer_idx=0,
        model_name=model_name,
    )
    embedding_dim = config.hidden_size
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads

    # Linear -> QLinearPerGrp
    # Convert before replication, as only separate projections can be copied
    pergrp_processing(attn, w_bit)

    if is_phi_2:
        # Copy separate QKV projections (QLinearPerGrp)
        attn_fa.q_proj = copy.deepcopy(attn.q_proj)
        attn_fa.k_proj = copy.deepcopy(attn.k_proj)
        attn_fa.v_proj = copy.deepcopy(attn.v_proj)
        attn_fa.dense = copy.deepcopy(attn.dense)
        if config.qk_layernorm:
            attn_fa.q_layernorm = copy.deepcopy(attn.q_layernorm)
            attn_fa.k_layernorm = copy.deepcopy(attn.k_layernorm)

        # Post processing for QLinearPerGrp
        attn.q_proj.device = device
        attn.k_proj.device = device
        attn.v_proj.device = device
        attn.dense.device = device
        attn.q_proj.quantize_weights()
        attn.k_proj.quantize_weights()
        attn.v_proj.quantize_weights()
        attn.dense.quantize_weights()

        # Merge QKV projections (QLinearPerGrp)
        attn_fa.init_faplus()

        # Post processing for QLinearPerGrp
        attn_fa.qkv_proj.device = device
        attn_fa.dense.device = device
        attn_fa.qkv_proj.quantize_weights()
        attn_fa.dense.quantize_weights()
    else:
        # Copy separate QKV projections (QLinearPerGrp)
        attn_fa.qkv_proj = copy.deepcopy(attn.qkv_proj)
        attn_fa.o_proj = copy.deepcopy(attn.o_proj)

        # Post processing for QLinearPerGrp
        attn.qkv_proj.device = device
        attn.o_proj.device = device
        attn.qkv_proj.quantize_weights()
        attn.o_proj.quantize_weights()

        # Post processing for QLinearPerGrp
        attn_fa.qkv_proj.device = device
        attn_fa.o_proj.device = device
        attn_fa.qkv_proj.quantize_weights()
        attn_fa.o_proj.quantize_weights()

    attn.eval()
    attn_fa.eval()
    # print(attn)
    # print(attn_fa)
    # print("==========")

    hidden_states, attention_mask, position_ids, past_key_value_vanilla = (
        generate_attention_test_input(
            batch_size,
            num_attention_heads,
            sequence_length,
            embedding_dim,
            kv_cache_type="cache",
            gqa=(num_attention_heads != num_key_value_heads),
            kv_heads=num_key_value_heads,
            attn_type="llama",
            has_mask=True,
            dtype=dtype,
        )
    )
    past_key_value_flash = copy.deepcopy(past_key_value_vanilla)

    output_vanilla, _, _ = attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value_vanilla,
    )

    output_flash, _, _ = attn_fa(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value_flash,
    )

    assert torch.allclose(output_flash, output_vanilla, atol=1e-2)
