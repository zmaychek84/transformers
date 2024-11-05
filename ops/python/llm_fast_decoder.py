import copy
from fast_decoders.npu_executor import NpuExecutor


class LLMFastDecoder:
    @classmethod
    def create_decoder(cls, model, old_node):
        new_node = None
        NpuExecutor.init()

        model_name = model.model_name

        if "Phi-3" in model_name:
            import fast_decoders.phi3_fast_decoder as phi3

            layer_idx = old_node.self_attn.layer_idx
            new_node = phi3.Phi3FastDecoder(model.config, layer_idx)

            new_node.input_layernorm.weight.data = copy.deepcopy(
                old_node.input_layernorm.weight.data
            )
            new_node.input_layernorm.variance_epsilon = (
                old_node.input_layernorm.variance_epsilon
            )

            new_node.self_attn.o_proj = old_node.self_attn.o_proj
            new_node.self_attn.qkv_proj = old_node.self_attn.qkv_proj

            new_node.resid_attn_dropout = old_node.resid_attn_dropout
            new_node.resid_mlp_dropout = old_node.resid_mlp_dropout

            new_node.mlp.gate_up_proj = old_node.mlp.gate_up_proj
            new_node.mlp.down_proj = old_node.mlp.down_proj
            new_node.mlp.activation_fn = old_node.mlp.activation_fn
            new_node.mlp.init_fastmlp()

            new_node.post_attention_layernorm.weight.data = copy.deepcopy(
                old_node.post_attention_layernorm.weight.data
            )
            new_node.post_attention_layernorm.variance_epsilon = (
                old_node.post_attention_layernorm.variance_epsilon
            )

        elif "chatglm3" in model_name:
            from fast_decoders.chatglm3_fast_decoder import GLMBlockOpt

            layer_number = old_node.layer_number
            new_node = GLMBlockOpt(model.config, layer_number, None)

            new_node.input_layernorm.variance_epsilon = old_node.input_layernorm.eps
            new_node.input_layernorm.weight.data = copy.deepcopy(
                old_node.input_layernorm.weight.data
            )

            new_node.self_attention.query_key_value = (
                old_node.self_attention.query_key_value
            )
            new_node.self_attention.dense = old_node.self_attention.dense

            new_node.mlp.dense_h_to_4h = old_node.mlp.dense_h_to_4h
            new_node.mlp.dense_4h_to_h = old_node.mlp.dense_4h_to_h
            new_node.mlp.init_fastmlp()

            new_node.post_attention_layernorm.weight.data = copy.deepcopy(
                old_node.post_attention_layernorm.weight.data
            )
            new_node.post_attention_layernorm.variance_epsilon = (
                old_node.post_attention_layernorm.eps
            )

        elif (
            ("llama" in model_name)
            or ("Llama" in model_name)
            or ("Qwen1" in model_name)
            or ("Mistral" in model_name)
        ):
            from llama_fast_decoder import LlamaFastDecoder

            new_node = LlamaFastDecoder()
            new_node.hidden_size = old_node.hidden_size
            new_node.self_attn = old_node.self_attn
            new_node.mlp = old_node.mlp
            new_node.input_layernorm = old_node.input_layernorm
            new_node.post_attention_layernorm = old_node.post_attention_layernorm

        return new_node
