import time
import math
import torch
from torch import nn
from configuration_chatglm import ChatGLMConfig
from .npu_executor import NpuExecutor


class CoreAttention(torch.nn.Module):
    scaling_bmm = 1 / math.sqrt(128)

    def __init__(self, config: ChatGLMConfig, layer_number):
        super(CoreAttention, self).__init__()

        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)

        projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_partition = projection_size
        self.hidden_size_per_attention_head = (
            projection_size // config.num_attention_heads
        )
        self.num_attention_heads_per_partition = config.num_attention_heads

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff
        self.coeff = coeff

        self.attention_dropout = torch.nn.Dropout(config.attention_dropout)

    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        pytorch_major_version = int(torch.__version__.split(".")[0])
        if pytorch_major_version >= 2:
            query_layer, key_layer, value_layer = [
                k.permute(1, 2, 0, 3) for k in [query_layer, key_layer, value_layer]
            ]
            key_layer = key_layer.contiguous()
            # print("version>2",query_layer.size(), key_layer.size(), value_layer.size())
            #  torch.Size([1, 32, 2048, 128]) torch.Size([1, 32, 2048, 128]) torch.Size([1, 32, 2048, 128])
            #  torch.Size([1, 32, 1, 128])    torch.Size([1, 32, 2049, 128]) torch.Size([1, 32, 2049, 128])
            if (
                attention_mask is None and query_layer.shape[2] == key_layer.shape[2]
            ):  # ttft
                attention_mask_npu = torch.zeros(
                    (1, query_layer.shape[2], query_layer.shape[2])
                ).to(torch.bfloat16)
                # print(attention_mask_npu.size())
                if query_layer.size()[2] >= 128:
                    context_layer = NpuExecutor.mha_npu.execute(
                        query_layer[0].contiguous(),
                        key_layer[0].contiguous(),
                        value_layer[0].contiguous(),
                        attention_mask_npu[0].contiguous(),
                        False,  # rettorch
                    )
                    # context_layer = context_layer.unsqueeze(0).view( query_layer.shape[2] , 1, 32, 128)
                else:
                    context_layer = torch.nn.functional.scaled_dot_product_attention(
                        query_layer, key_layer, value_layer, is_causal=True
                    )
                    context_layer = context_layer.permute(2, 0, 1, 3)
                    # print("cpu context_layer", context_layer.size())
                    new_context_layer_shape = context_layer.size()[:-2] + (
                        self.hidden_size_per_partition,
                    )
                    # print(": core att : " , context_layer.shape, new_context_layer_shape  ) #  | ([2048, 1, 32, 128]) torch.Size([2048, 1, 4096]) | [1, 1, 32, 128]) torch.Size([1, 1, 4096])
                    context_layer = context_layer.reshape(*new_context_layer_shape)
                    # print("CPU context layer reshape", context_layer.size())

            else:  # tps
                if attention_mask is not None:
                    attention_mask = ~attention_mask
                if query_layer.size()[2] >= 4096:
                    context_layer = NpuExecutor.mha_npu.execute(
                        query_layer[0].contiguous(),
                        key_layer[0].contiguous(),
                        value_layer[0].contiguous(),
                        attention_mask[0].contiguous(),
                    )
                    context_layer = context_layer.unsqueeze(0)
                else:
                    # context_layer = torch.nn.functional.scaled_dot_product_attention(
                    #    query_layer, key_layer, value_layer, attention_mask
                    # )
                    context_layer = torch.matmul(query_layer, key_layer.transpose(2, 3))
                    # t0 = time.time()

                    context_layer = context_layer * CoreAttention.scaling_bmm
                    # print(" matmul" , time.time() - t0)
                    # t0 = time.time()
                    if attention_mask is not None:
                        context_layer = context_layer + attention_mask
                    context_layer = torch.nn.functional.softmax(
                        context_layer.to(torch.float32), -1
                    ).to(torch.bfloat16)
                    # print(" sfm " , time.time() - t0)
                    # attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
                    context_layer = torch.matmul(context_layer, value_layer)
                    # context_layer = context_layer.transpose(1, 2).contiguous()

                # t0 = time.time()
                # print("CPU context layer ", context_layer.size())
                context_layer = context_layer.permute(2, 0, 1, 3)
                new_context_layer_shape = context_layer.size()[:-2] + (
                    self.hidden_size_per_partition,
                )
                # print(": core att : " , context_layer.shape, new_context_layer_shape  ) #  | ([2048, 1, 32, 128]) torch.Size([2048, 1, 4096]) | [1, 1, 32, 128]) torch.Size([1, 1, 4096])
                context_layer = context_layer.reshape(*new_context_layer_shape)

        else:
            # Raw attention scores

            # [b, np, sq, sk]
            output_size = (
                query_layer.size(1),
                query_layer.size(2),
                query_layer.size(0),
                key_layer.size(0),
            )

            # [sq, b, np, hn] -> [sq, b * np, hn]
            query_layer = query_layer.view(
                output_size[2], output_size[0] * output_size[1], -1
            )
            # [sk, b, np, hn] -> [sk, b * np, hn]
            key_layer = key_layer.view(
                output_size[3], output_size[0] * output_size[1], -1
            )

            # preallocting input tensor: [b * np, sq, sk]
            matmul_input_buffer = torch.empty(
                output_size[0] * output_size[1],
                output_size[2],
                output_size[3],
                dtype=query_layer.dtype,
                device=query_layer.device,
            )
            # print(query_layer.transpose(1,0).size(), key_layer.transpose(0, 1).transpose(1, 2))
            # Raw attention scores. [b * np, sq, sk]
            matmul_result = torch.baddbmm(
                matmul_input_buffer,
                query_layer.transpose(0, 1),  # [b * np, sq, hn]
                key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
                beta=0.0,
                alpha=(1.0 / self.norm_factor),
            )

            # change view to [b, np, sq, sk]
            attention_scores = matmul_result.view(*output_size)

            # ===========================
            # Attention probs and dropout
            # ===========================

            # attention scores and attention mask [b, np, sq, sk]
            if self.attention_softmax_in_fp32:
                attention_scores = attention_scores.float()
            if self.coeff is not None:
                attention_scores = attention_scores * self.coeff
            if (
                attention_mask is None
                and attention_scores.shape[2] == attention_scores.shape[3]
            ):
                attention_mask = torch.ones(
                    output_size[0],
                    1,
                    output_size[2],
                    output_size[3],
                    device=attention_scores.device,
                    dtype=torch.bool,
                )
                attention_mask.tril_()
                attention_mask = ~attention_mask
            if attention_mask is not None:
                attention_scores = attention_scores.masked_fill(
                    attention_mask, float("-inf")
                )
            attention_probs = F.softmax(attention_scores, dim=-1)
            attention_probs = attention_probs.type_as(value_layer)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = self.attention_dropout(attention_probs)
            # =========================
            # Context layer. [sq, b, hp]
            # =========================

            # value_layer -> context layer.
            # [sk, b, np, hn] --> [b, np, sq, hn]

            # context layer shape: [b, np, sq, hn]
            output_size = (
                value_layer.size(1),
                value_layer.size(2),
                query_layer.size(0),
                value_layer.size(3),
            )
            # change view [sk, b * np, hn]
            value_layer = value_layer.view(
                value_layer.size(0), output_size[0] * output_size[1], -1
            )
            # change view [b * np, sq, sk]
            attention_probs = attention_probs.view(
                output_size[0] * output_size[1], output_size[2], -1
            )
            # matmul: [b * np, sq, hn]
            context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))
            # change view [b, np, sq, hn]
            context_layer = context_layer.view(*output_size)
            # [b, np, sq, hn] --> [sq, b, np, hn]
            context_layer = context_layer.permute(2, 0, 1, 3).contiguous()
            # [sq, b, np, hn] --> [sq, b, hp]
            new_context_layer_shape = context_layer.size()[:-2] + (
                self.hidden_size_per_partition,
            )
            context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


from modeling_chatglm import apply_rotary_pos_emb


class SelfAttention(torch.nn.Module):
    """Parallel self-attention layer abstract class.
    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, config: ChatGLMConfig, layer_number, device=None):
        super(SelfAttention, self).__init__()

        self.layer_number = max(1, layer_number)
        self.projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_attention_head = (
            self.projection_size // config.num_attention_heads
        )
        self.num_attention_heads_per_partition = config.num_attention_heads
        # print(config.multi_query_attention)
        self.multi_query_attention = config.multi_query_attention
        self.qkv_hidden_size = 3 * self.projection_size
        if self.multi_query_attention:
            self.num_multi_query_groups_per_partition = config.multi_query_group_num
            self.qkv_hidden_size = (
                self.projection_size
                + 2 * self.hidden_size_per_attention_head * config.multi_query_group_num
            )
        """ self.query_key_value = nn.Linear(
            config.hidden_size,
            self.qkv_hidden_size,
            bias=config.add_bias_linear or config.add_qkv_bias,
            device=device,
            **_config_to_kwargs(config),
        ) """
        self.query_key_value = None

        self.core_attention = CoreAttention(config, self.layer_number)

        # Output.
        """ self.dense = nn.Linear(
            self.projection_size,
            config.hidden_size,
            bias=config.add_bias_linear,
            device=device,
            **_config_to_kwargs(config),
        ) """
        self.dense = None

    def _allocate_memory(
        self, inference_max_sequence_len, batch_size, device=None, dtype=None
    ):
        if self.multi_query_attention:
            num_attention_heads = self.num_multi_query_groups_per_partition
        else:
            num_attention_heads = self.num_attention_heads_per_partition
        return torch.empty(
            inference_max_sequence_len,
            batch_size,
            num_attention_heads,
            self.hidden_size_per_attention_head,
            dtype=dtype,
            device=device,
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        rotary_pos_emb,
        kv_cache=None,
        use_cache=True,
    ):
        # hidden_states: [sq, b, h]

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        # =====================
        # Query, Key, and Value
        # =====================

        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        # print("-hidden :", hidden_states.shape )#  torch.Size([2048, 1, 4096]) or [1, 4] or [1, 1, 4096]
        if len(hidden_states.shape) == 2:
            hidden_states = torch.index_select(
                hidden_states, 1, torch.tensor([1, 2, 3])
            )
            mixed_x_layer = self.query_key_value(
                hidden_states, zerocpy=True, rettensor=True
            )
        else:
            mixed_x_layer = self.query_key_value(
                hidden_states.transpose(0, 1), zerocpy=False, rettensor=True
            )  #   format: in out M K

        # mixed_x_layer.shape : torch.Size([1, 2048, 4608])   4608=4096+512
        mixed_x_layer = mixed_x_layer.transpose(0, 1)

        # num_attention_heads_per_partition, hidden_size_per_attention_head, num_multi_query_groups_per_partition)#  32 128 2
        if self.multi_query_attention:
            (query_layer, key_layer, value_layer) = mixed_x_layer.split(
                [
                    self.num_attention_heads_per_partition
                    * self.hidden_size_per_attention_head,  # 32*128
                    self.num_multi_query_groups_per_partition
                    * self.hidden_size_per_attention_head,  #  2*128
                    self.num_multi_query_groups_per_partition
                    * self.hidden_size_per_attention_head,  #  2*128
                ],
                dim=-1,
            )
            # query_layer.shape, key_layer.shape # ([2048, 1, 4096]) torch.Size([2048, 1, 256]) | torch.Size([1, 1, 4096]) torch.Size([1, 1, 256])
            query_layer = query_layer.view(
                query_layer.size()[:-1]
                + (
                    self.num_attention_heads_per_partition,
                    self.hidden_size_per_attention_head,
                )
            ).clone()
            key_layer = key_layer.view(
                key_layer.size()[:-1]
                + (
                    self.num_multi_query_groups_per_partition,
                    self.hidden_size_per_attention_head,
                )
            ).clone()
            # query_layer.shape, key_layer.shape) # torch.Size([2048, 1, 32, 128]) torch.Size([2048, 1, 2, 128]) #  torch.Size([1, 1, 32, 128]) torch.Size([1, 1, 2, 128])
            value_layer = value_layer.view(
                value_layer.size()[:-1]
                + (
                    self.num_multi_query_groups_per_partition,
                    self.hidden_size_per_attention_head,
                )
            ).clone()
        else:
            new_tensor_shape = mixed_x_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                3 * self.hidden_size_per_attention_head,
            )
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(
                mixed_x_layer, 3
            )

        # apply relative positional encoding (rotary embedding)
        if rotary_pos_emb is not None:
            query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)

        # adjust key and value for inference
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            key_layer = torch.cat((cache_k, key_layer), dim=0)
            # kv_cache[0].shape # torch.Size([2048, 1, 2, 128])
            # key_layer.shape ) #  torch.Size([2049, 1, 2, 128])
            value_layer = torch.cat((cache_v, value_layer), dim=0)
        if use_cache:
            kv_cache = (key_layer, value_layer)
        else:
            kv_cache = None

        if self.multi_query_attention:
            key_layer_new = key_layer.unsqueeze(-2)
            #   key_layer.shape # ([2048, 1, 2, 1, 128]
            key_layer = key_layer_new.expand(
                -1,
                -1,
                -1,
                self.num_attention_heads_per_partition
                // self.num_multi_query_groups_per_partition,
                -1,
            ).clone()
            # key_layer.shape)# [2048, 1, 2, 16, 128]
            key_layer = (
                key_layer.contiguous()
                .view(
                    key_layer.size()[:2]
                    + (
                        self.num_attention_heads_per_partition,
                        self.hidden_size_per_attention_head,
                    )
                )
                .clone()
            )
            #  key_layer.shape # [2048, 1, 32, 128]
            value_layer = value_layer.unsqueeze(-2)
            value_layer = value_layer.expand(
                -1,
                -1,
                -1,
                self.num_attention_heads_per_partition
                // self.num_multi_query_groups_per_partition,
                -1,
            )
            value_layer = value_layer.contiguous().view(
                value_layer.size()[:2]
                + (
                    self.num_attention_heads_per_partition,
                    self.hidden_size_per_attention_head,
                )
            )

        # ==================================
        # core attention computation
        # ==================================

        context_layer = self.core_attention(
            query_layer, key_layer, value_layer, attention_mask
        )
        # =================
        # Output. [sq, b, h]
        # =================
        # t0 = time.perf_counter()
        #  context_layer  # [1, 2048, 4096]  |   [1, 1, 4096]
        if len(context_layer.shape) == 2:
            output = self.dense(
                context_layer.transpose(0, 1), zerocpy=True, rettensor=False
            )  # ([2048, 1, 4096]  | [1, 1, 4096]
            # ret format is [out_addr, m , k]
            # output = output.transpose(0,1)
        else:
            output = self.dense(
                context_layer.transpose(0, 1)
            )  # ([2048, 1, 4096]  | [1, 1, 4096]
        return output, kv_cache


def _config_to_kwargs(args):
    common_kwargs = {
        "dtype": args.torch_dtype,
    }
    return common_kwargs


def unpack(qcompact, k):
    qw = torch.empty((qcompact.shape[0], k), dtype=torch.int8)
    refmsb = torch.tensor(0xF0, dtype=torch.uint8)
    reflsb = torch.tensor(0x0F, dtype=torch.uint8)
    qw[:, 0::2] = (torch.bitwise_and(qcompact[:, :], refmsb) >> 4).to(torch.int8)
    qw[:, 1::2] = torch.bitwise_and(qcompact[:, :], reflsb).to(torch.int8)
    return qw


class MLP(torch.nn.Module):
    """MLP.
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, config: ChatGLMConfig, device=None):
        super(MLP, self).__init__()

        self.add_bias = config.add_bias_linear

        # Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        self.dense_h_to_4h = None
        # Project back to h.
        self.dense_4h_to_h = None

        # def swiglu(x):
        #    x = torch.chunk(x, 2, dim=-1)
        #    return F.silu(x[0]) * x[1]

        # self.activation_func = swiglu

    def init_fastmlp(self):
        if self.dense_4h_to_h.bias is None:
            self.dense_4h_to_h.bias = torch.zeros(
                (1, self.dense_4h_to_h.qweight.size()[0])
            ).to(torch.bfloat16)
        self.dense_4h_to_h.qweight = unpack(
            self.dense_4h_to_h.qweight, self.dense_4h_to_h.in_features
        )

        gate_qw = self.dense_4h_to_h.qweight.transpose(0, 1).contiguous()
        gate_qz = self.dense_4h_to_h.qzeros.transpose(0, 1).contiguous()
        gate_scale = (
            self.dense_4h_to_h.scales.transpose(0, 1).to(torch.float).contiguous()
        )
        gate_bias = self.dense_4h_to_h.bias.to(torch.float).contiguous()
        NpuExecutor.partial_mlp_npu.initialize_params(
            # chatglm_mlp_npu.initialize_params(
            gate_qw,
            gate_qz,
            gate_scale,
            gate_bias,
            self.dense_4h_to_h.group_size,
        )
        del self.dense_4h_to_h

    def forward(self, hidden_states):
        # [s, b, 4hp]
        # t0 = time.perf_counter()
        # intermediate_parallel = self.dense_h_to_4h(hidden_states.transpose(0,1))

        # hidden_states format # in_addr, out_addr, M, K.   Should be:  in_addr, M, K.
        if len(hidden_states.shape) == 2:
            hidden_states = torch.index_select(
                hidden_states, 1, torch.tensor([1, 2, 3])
            )
            intermediate_parallel = self.dense_h_to_4h(
                hidden_states, zerocpy=True, rettensor=True
            )
        else:
            intermediate_parallel = self.dense_h_to_4h(
                hidden_states, zerocpy=False, rettensor=True
            )
        #   intermediate_parallel  ) # [1, 2048, 27392]

        # print(f"dense_h_to_4h time: {time.perf_counter()-t0}")
        # t0 = time.perf_counter()
        intermediate_parallel = torch.chunk(intermediate_parallel, 2, dim=-1)
        #  intermediate_parallel[0].shape ) # [1, 2048, 13696]  | ([1, 1, 13696]
        # print(f"chunk time: {time.perf_counter()-t0}")
        # t0 = time.perf_counter()
        if len(hidden_states.shape) == 2:
            output = NpuExecutor.partial_mlp_npu.execute(
                intermediate_parallel[0].contiguous(),
                intermediate_parallel[1].contiguous(),
                False,
            )  # False: async
        else:
            output = NpuExecutor.partial_mlp_npu.execute(
                intermediate_parallel[0].contiguous(),
                intermediate_parallel[1].contiguous(),
                True,
            )
        # print(f"silu+mul+dense_4h_to_h time: {time.perf_counter()-t0}")
        # t0 = time.perf_counter()
        # intermediate_parallel = (
        #    F.silu(intermediate_parallel[0]) * intermediate_parallel[1]
        # )
        # [s, b, h]
        # output = self.dense_4h_to_h(intermediate_parallel)
        return output


class GLMFastRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size=4096, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, in_zerocpy=False, rettorch=True, in_len=0):
        if in_zerocpy == False:
            q_len = hidden_states.size()[0]
        else:
            q_len = in_len
        if q_len >= 128:
            if in_zerocpy:
                norm_out = NpuExecutor.rmsnorm_npu.execute(
                    hidden_states, self.weight.data, in_zerocpy, rettorch
                )  # remove this:  .unsqueeze(0)
            else:
                norm_out = NpuExecutor.rmsnorm_npu.execute(
                    hidden_states.view(q_len, hidden_states.shape[2]).contiguous(),
                    self.weight.data,
                    False,
                    rettorch,
                )
                if rettorch:
                    norm_out = norm_out.view(q_len, 1, -1)
            return norm_out
        else:
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(
                variance + self.variance_epsilon
            )
            norm_out = self.weight * hidden_states.to(torch.bfloat16)
            return norm_out


class GLMBlockOpt(torch.nn.Module):
    """A single transformer layer.
    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, config: ChatGLMConfig, layer_number, device=None):
        super(GLMBlockOpt, self).__init__()
        self.layer_number = layer_number

        self.apply_residual_connection_post_layernorm = (
            config.apply_residual_connection_post_layernorm
        )

        self.fp32_residual_connection = config.fp32_residual_connection

        # Layernorm on the input data.
        self.input_layernorm = GLMFastRMSNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon,
        )

        # Self attention.
        self.self_attention = SelfAttention(config, layer_number, device=device)
        self.hidden_dropout = config.hidden_dropout

        # Layernorm on the attention output
        self.post_attention_layernorm = GLMFastRMSNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon,
            # device=device,
            # dtype=config.torch_dtype,
        )
        t0 = time.perf_counter()
        self.mlp = MLP(config, device=device)

    def forward(
        self,
        hidden_states,
        attention_mask,
        rotary_pos_emb,
        kv_cache=None,
        use_cache=True,
    ):
        # hidden_states: [s, b, h]
        # t0 = time.perf_counter()
        # Layer norm at the beginning of the transformer layer.
        #  hidden_states  # [2048, 1, 4096]
        layernorm_output = self.input_layernorm(
            hidden_states, in_zerocpy=False, rettorch=False
        )  #  [2048, 1, 4096]
        # print(f"norm time: {time.perf_counter()-t0}")
        # t0 = time.perf_counter()
        # Self attention.
        # attention_output, kv_cache = self.self_attention(
        layernorm_input, kv_cache = self.self_attention(
            layernorm_output,
            attention_mask,
            rotary_pos_emb,
            kv_cache=kv_cache,
            use_cache=use_cache,
        )
        # print(f"atten time: {time.perf_counter()-t0}")
        # t0 = time.perf_counter()
        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # layernorm_input = torch.nn.functional.dropout(attention_output, p=self.hidden_dropout, training=self.training)
        #  hidden_states.shape  , residual.shape, layernorm_input.shape )
        #  torch.Size([2048, 1, 4096]) torch.Size([2048, 1, 4096]) torch.Size([2048, 1, 4096])
        #  torch.Size([1, 1, 4096]) torch.Size([1, 1, 4096]) torch.Size([1, 1, 4096])

        # layernorm_input = residual + layernorm_input
        # if layernorm_input.shape[0] >= 128:
        if len(layernorm_input.shape) == 2:
            layernorm_input = NpuExecutor.elewadd_npu.execute(
                residual.view(
                    layernorm_input[0][1].item(), layernorm_input[0][2].item()
                ).contiguous(),
                layernorm_input,
                1,
                0,
            )  # in_zero_copy=1  , ret_torch=false

            # layernorm_input's format from add & rmsnorm: [out_addr, M, K]
            # layernorm_input.shape, layernorm_input # ([1, 3] [[1140850688,       2048,       4096]]
            layernorm_output = self.post_attention_layernorm(
                layernorm_input,
                in_zerocpy=True,
                rettorch=False,
                in_len=layernorm_input[0][1].item(),
            )
        else:
            layernorm_input = residual.transpose(0, 1) + layernorm_input
            layernorm_output = self.post_attention_layernorm(
                layernorm_input, in_zerocpy=False, rettorch=True
            )

        # print(f"add time: {time.perf_counter()-t0}")
        # MLPOpt.
        mlp_output = self.mlp(layernorm_output)  # mlp_output format: [ out_addr, M, K]

        # print(f"mlp time: {time.perf_counter()-t0}")
        # t0 = time.perf_counter()
        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:  # false
            residual = layernorm_output
        else:
            residual = layernorm_input

        # output = torch.nn.functional.dropout( mlp_output, p=self.hidden_dropout, training=self.training )

        # output = residual + output
        # if mlp_output.shape[0] >= 128:
        if len(mlp_output.shape) == 2:
            output = (
                NpuExecutor.elewadd_npu.execute(
                    mlp_output, residual, 2, 1
                ).view(  # in_zero_copy=2  , ret_torch=True
                    mlp_output[0][1].item(), 1, mlp_output[0][2].item()
                )
                # .clone()
            )
            #  original output: [2048, 1, 4096]

        else:
            output = residual.transpose(0, 1) + mlp_output

        # print(f"add 2 time: {time.perf_counter()-t0}")
        return output, kv_cache
