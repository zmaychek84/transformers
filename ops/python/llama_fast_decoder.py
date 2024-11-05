#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

from typing import Optional, Tuple
import torch
from transformers import LlamaConfig
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaMLP,
    LlamaRMSNorm,
)

import os

npu_add = 0
fast_attention = 0
fast_mlp = 0
fast_norm = 0
if os.environ.get("MLADF") and os.environ.get("DEVICE") == "stx":
    npu_add = 1
    import sys

    for name in sys.argv:
        if "--fast_attention" in name:
            fast_attention = 1
        if "--fast_mlp" in name:
            fast_mlp = 1
        if "--fast_norm" in name:
            fast_norm = 1


class FastDecoder_NpuExecutor:
    init_done = False
    fast_attention = False
    fast_mlp = False
    fast_norm = False
    npu_add = False

    @classmethod
    def init_npuexecutor(cls, dev="stx", mladf="2x4x4"):
        if not cls.init_done:
            cls.init_done = True
            import ryzenai_torch_cpp

            if mladf == "2x4x4" and dev == "stx":
                import sys

                for name in sys.argv:
                    if "--fast_attention" in name:
                        cls.fast_attention = True
                    if "--fast_mlp" in name:
                        cls.fast_mlp = True
                    if "--fast_norm" in name:
                        cls.fast_norm = True
            else:
                raise RuntimeError(
                    f"LlamaFastDecoder needs os environment DEVICE=stx and MLADF=2x4x4"
                )
        else:
            pass


class LlamaFastDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = None
        self.self_attn = None
        self.mlp = None
        self.input_layernorm = None
        self.post_attention_layernorm = None

        FastDecoder_NpuExecutor.init_npuexecutor(
            dev=os.environ.get("DEVICE"),
            mladf=os.environ.get("MLADF"),
        )

    def __repr__(self):
        return f"LlamaFastDecoder"

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:

        residual = hidden_states
        bsz, q_len, _ = hidden_states.size()
        if FastDecoder_NpuExecutor.fast_norm:
            hidden_states = self.input_layernorm(
                hidden_states, in_zerocpy=False, rettorch=not fast_attention
            )
        else:
            hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            residual=residual,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        if FastDecoder_NpuExecutor.fast_attention and q_len >= 128:
            pass
        else:
            hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        if FastDecoder_NpuExecutor.fast_norm:
            hidden_states = self.post_attention_layernorm(
                hidden_states,
                in_zerocpy=True
                and FastDecoder_NpuExecutor.fast_attention
                and FastDecoder_NpuExecutor.fast_mlp,
                rettorch=False or FastDecoder_NpuExecutor.fast_mlp == 0,
                in_len=q_len,
            )
        else:
            hidden_states = self.post_attention_layernorm(hidden_states)
        if fast_mlp:
            hidden_states = self.mlp(hidden_states, q_len)
        else:
            hidden_states = self.mlp(hidden_states)
        if (
            npu_add == 1
            and hidden_states.shape[1] >= 128
            and FastDecoder_NpuExecutor.fast_mlp == 1
            and FastDecoder_NpuExecutor.fast_norm == 1
        ):
            pass
        else:
            hidden_states = residual + hidden_states
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
