#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import math
import os
import time
from typing import Optional, Tuple

import torch
from ryzenai_attention import *
from transformers import LlamaConfig
from transformers.cache_utils import Cache
from llama_fast_mlp_npu import unpack
import os
import llama_fast_rmsnorm


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaRotaryEmbeddingLocal(torch.nn.Module):
    def __init__(self, base=10000):
        super().__init__()
        self.base = base
        dim = 128
        max_position_embeddings = 4096  # self.max_position_embeddings
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, dim, 2).to(torch.float32) / dim)
        )
        t = torch.arange(max_position_embeddings, device="cpu", dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.emb_cos = emb.cos().to(torch.bfloat16)
        self.emb_sin = emb.sin().to(torch.bfloat16)

    def forward(self, seq_len=None):
        return (
            self.emb_cos[:seq_len],
            self.emb_sin[:seq_len],
        )


class FastAttention_NpuExecutor:
    fast_attention = False
    fast_mlp = False
    fast_norm = False
    fast_decoder = False
    qkv_rope_opt = False

    mha_npu = None
    mha_cpu = None
    rope_npu = None
    elewadd_npu = None

    @classmethod
    def init_npuexecutor(cls, dev="stx", mladf="2x4x4"):
        if cls.mha_cpu is None:
            import ryzenai_torch_cpp

            cls.mha_cpu = ryzenai_torch_cpp.cpu_mha()

            if mladf == "2x4x4" and dev == "stx":
                import sys

                for name in sys.argv:
                    if "--fast_attention" in name:
                        cls.fast_attention = True
                    if "--fast_mlp" in name:
                        cls.fast_mlp = True
                    if "--fast_norm" in name:
                        cls.fast_norm = True
                    if "--fast_decoder" in name:
                        cls.fast_decoder = True
                cls.elewadd_npu = ryzenai_torch_cpp.aie_elemw_add_torch()
                cls.mha_npu = ryzenai_torch_cpp.aie_mha_npu_torch()
                cls.qkv_rope_opt = cls.fast_decoder
                if cls.qkv_rope_opt:
                    cls.rope_npu = ryzenai_torch_cpp.aie_qkv_rope_torch()
                else:
                    cls.rope_npu = ryzenai_torch_cpp.aie_rope_torch()
            else:
                raise RuntimeError(
                    f"LlamaFastAttention needs os environment DEVICE=stx and MLADF=2x4x4"
                )
        else:
            pass


class LlamaFastAttention(RyzenAIAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: Optional[int] = None,
        precision: str = "w4abf16",
        model_name: str = "llama-2-7b-chat",
        flash_config_path: str = "",
        profile: bool = False,
        mhaops: str = "all",
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.precision = precision
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = (
            4096  # config.max_position_embeddings (2048) is not enough
        )
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.scaling = torch.tensor(self.head_dim**-0.5)
        self.profile = profile
        self.mladf_version = os.getenv("MLADF_VERSION", "v1")

        def create_timer():
            start_time = None

            def TIC():
                nonlocal start_time
                start_time = time.perf_counter()

            def TOC(name):
                elapsed_time = time.perf_counter() - start_time
                print(f"{name}: {elapsed_time:.6f}")

            return TIC, TOC

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        def nullfun():
            pass

        def nullfun2(name):
            pass

        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.o_proj = None

        if self.precision == "w4abf16":
            self.matmul_qkv = self.matmul_qkv_w4abf16
            self.dtype = torch.bfloat16
        elif self.precision.startswith("w8"):
            self.matmul_qkv = self.matmul_qkv_w8
            self.dtype = torch.float32

        # For FA
        self.model_name = model_name
        self.scaling_bmm = 1 / math.sqrt(self.head_dim)

        if self.profile:
            self.TIC, self.TOC = create_timer()

        else:
            self.TIC = nullfun
            self.TOC = nullfun2
        self.mhaops = mhaops
        self.rotary_emb = LlamaRotaryEmbeddingLocal(self.rope_theta)

    def init_faplus(self):
        FastAttention_NpuExecutor.init_npuexecutor(
            dev=os.environ.get("DEVICE"),
            mladf=os.environ.get("MLADF"),
        )
        if FastAttention_NpuExecutor.qkv_rope_opt:
            if self.q_proj.bias is None:
                self.q_proj.bias = torch.zeros((1, self.q_proj.qweight.size()[0])).to(
                    torch.bfloat16
                )
            if self.k_proj.bias is None:
                self.k_proj.bias = torch.zeros((1, self.k_proj.qweight.size()[0])).to(
                    torch.bfloat16
                )
            if self.v_proj.bias is None:
                self.v_proj.bias = torch.zeros((1, self.v_proj.qweight.size()[0])).to(
                    torch.bfloat16
                )
            self.q_proj.qweight = unpack(self.q_proj.qweight, self.q_proj.in_features)
            self.k_proj.qweight = unpack(self.k_proj.qweight, self.k_proj.in_features)
            self.v_proj.qweight = unpack(self.v_proj.qweight, self.v_proj.in_features)

            FastAttention_NpuExecutor.rope_npu.initialize_params(
                self.q_proj.qweight.transpose(0, 1),
                self.q_proj.qzeros.transpose(0, 1),
                self.q_proj.scales.transpose(0, 1).to(torch.float),
                self.q_proj.bias.to(torch.float),
                self.q_proj.group_size,
                self.k_proj.qweight.transpose(0, 1),
                self.k_proj.qzeros.transpose(0, 1),
                self.k_proj.scales.transpose(0, 1).to(torch.float),
                self.k_proj.bias.to(torch.float),
                self.k_proj.group_size,
                self.v_proj.qweight.transpose(0, 1),
                self.v_proj.qzeros.transpose(0, 1),
                self.v_proj.scales.transpose(0, 1).to(torch.float),
                self.v_proj.bias.to(torch.float),
                self.v_proj.group_size,
            )
        else:
            if self.precision == "w4abf16":
                pass
                self.merge_qkv_w4abf16(self.q_proj, self.k_proj, self.v_proj)

            elif self.precision.startswith("w8"):
                self.merge_qkv_w8(self.q_proj, self.k_proj, self.v_proj, self.precision)

        del self.q_proj, self.k_proj, self.v_proj

    def init_o_proj(self):
        if self.mladf_version == "v1":
            if self.o_proj.bias is None:
                self.o_proj.bias = torch.zeros((1, self.o_proj.qweight.size()[0])).to(
                    torch.bfloat16
                )
            self.o_proj.qweight = unpack(self.o_proj.qweight, self.o_proj.in_features)

            FastAttention_NpuExecutor.mha_npu.initialize_params(
                self.o_proj.qweight.transpose(0, 1),
                self.o_proj.qzeros.transpose(0, 1),
                self.o_proj.scales.transpose(0, 1).to(torch.float),
                self.o_proj.bias.to(torch.float),
                self.o_proj.group_size,
            )
            del self.o_proj

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,  # Dummy for now.
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if residual is not None:
            bsz, q_len, _ = residual.size()
        else:
            bsz, q_len, _ = hidden_states.size()
        if FastAttention_NpuExecutor.qkv_rope_opt:
            query_states = torch.empty(
                (bsz, self.num_heads, q_len, self.head_dim), dtype=torch.bfloat16
            )
            key_states = torch.empty(
                (bsz, self.num_key_value_heads, q_len, self.head_dim),
                dtype=torch.bfloat16,
            )
            value_states = torch.empty(
                (bsz, q_len, self.num_key_value_heads, self.head_dim),
                dtype=torch.bfloat16,
            )
            self.TIC()
            if q_len > 1:
                kv_seq_len = q_len
            else:
                kv_seq_len = value_states.shape[1]

            if past_key_value is not None:
                kv_seq_len += past_key_value.get_usable_length(
                    kv_seq_len, self.layer_idx
                )

            cos, sin = self.rotary_emb.forward(seq_len=kv_seq_len)
            self.TOC("cos_sin")

            self.TIC()
            trig_1 = cos[position_ids].unsqueeze(1)
            trig_2 = sin[position_ids].unsqueeze(1)
            trig = torch.cat((trig_1, trig_2), dim=1)

            share_bo_threshold = 0
            shared_address = []
            if (
                FastAttention_NpuExecutor.fast_decoder
                and FastAttention_NpuExecutor.fast_norm
            ):
                share_bo_threshold = 128
                shared_address = llama_fast_rmsnorm.op.get_address()

            FastAttention_NpuExecutor.rope_npu.execute(
                shared_address,
                # hidden_states.view(torch.bfloat16).contiguous(),
                hidden_states[0].contiguous(),
                query_states[0].contiguous(),
                key_states[0].contiguous(),
                value_states[0].contiguous(),
                trig[0],
                self.head_dim,
                share_bo_threshold,
            )
            value_states = value_states.transpose(1, 2)
            self.TOC("qkv_rope")
        else:
            self.TIC()
            query_states, key_states, value_states = self.matmul_qkv(hidden_states)
            self.TOC("qkv")
            self.TIC()
            if q_len in [128, 256, 512, 1024, 2048]:
                query_states = query_states.view(
                    bsz, self.num_heads, q_len, self.head_dim
                )  # .transpose(1, 2)
                key_states = key_states.view(
                    bsz, self.num_key_value_heads, q_len, self.head_dim
                )  # .transpose(1, 2)
            else:
                query_states = query_states.view(
                    bsz, q_len, self.num_heads, self.head_dim
                ).transpose(1, 2)
                key_states = key_states.view(
                    bsz, q_len, self.num_key_value_heads, self.head_dim
                ).transpose(1, 2)
            value_states = value_states.view(
                bsz, q_len, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)
            self.TOC("view")
            self.TIC()
            if q_len > 1:
                kv_seq_len = q_len
            else:
                kv_seq_len = value_states.shape[-2]

            if past_key_value is not None:
                kv_seq_len += past_key_value.get_usable_length(
                    kv_seq_len, self.layer_idx
                )

            cos, sin = self.rotary_emb.forward(seq_len=kv_seq_len)
            self.TOC("cos sin")

            self.TIC()
            if q_len in [128, 256, 512, 1024, 2048]:
                trig_1 = cos[position_ids].unsqueeze(1)
                trig_2 = sin[position_ids].unsqueeze(1)
                trig = torch.cat((trig_1, trig_2), dim=1)
                query_states = FastAttention_NpuExecutor.rope_npu.execute(
                    query_states[0].contiguous(), trig[0]
                ).clone()
                key_states = FastAttention_NpuExecutor.rope_npu.execute(
                    key_states[0].contiguous(), trig[0]
                ).clone()
            else:
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, position_ids
                )
            self.TOC("rope")

        self.TIC()
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx
            )
        self.TOC("cache")
        self.TIC()
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        self.TOC("repeat")

        self.TIC()
        if q_len >= 256:
            attn_output = FastAttention_NpuExecutor.mha_npu.execute(
                query_states[0].contiguous(),
                key_states[0].contiguous(),
                value_states[0].contiguous(),
                attention_mask[0].contiguous(),
                # False
                not (FastAttention_NpuExecutor.fast_decoder and q_len >= 128),
            )

        else:
            if False:
                attn_output = FastAttention_NpuExecutor.mha_cpu.mha_top(
                    query_states, key_states, value_states, attention_mask
                )
            else:
                attn_weights = (
                    torch.matmul(query_states, key_states.transpose(2, 3))
                    * self.scaling_bmm
                )
                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask
                attn_weights = torch.nn.functional.softmax(
                    attn_weights.to(torch.float32), -1
                ).to(torch.bfloat16)
                # attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
                attn_output = torch.matmul(attn_weights, value_states)
                attn_output = attn_output.transpose(1, 2).contiguous()

        self.TOC("mha")

        if FastAttention_NpuExecutor.fast_decoder and q_len >= 128:
            if (
                FastAttention_NpuExecutor.fast_norm
                and FastAttention_NpuExecutor.fast_mlp
            ):
                rettensor = False
            else:
                rettensor = True
            self.TIC()
            if q_len >= 256:
                attn_output = self.o_proj(attn_output, True, False)
            else:
                attn_output = self.o_proj(attn_output, False, False)
            self.TOC("o_proj")
            # attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
            attn_output = FastAttention_NpuExecutor.elewadd_npu.execute(
                residual[0], attn_output, 1, rettensor
            ).unsqueeze(0)
        else:
            self.TIC()
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

            attn_output = self.o_proj(attn_output)
            self.TOC("o_proj")

        return attn_output, None, past_key_value
