import copy
import gc
import os
import time
from typing import Dict, List, Optional

import qlinear
import torch

# AWQ
from qmodule import WQLinear

from transformers.models.opt.modeling_opt import OPTAttention
from transformers.models.phi.modeling_phi import PhiAttention

import transformers as tv

tver = int(tv.__version__.split(".")[1])

if tver > 44:
    from transformers.models.phi3.modeling_phi3 import Phi3Attention

from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2MLP,
    Qwen2RMSNorm,
)
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaAttention,
    LlamaMLP,
    LlamaRMSNorm,
)
from transformers.models.mistral.modeling_mistral import (
    MistralAttention,
    MistralMLP,
    MistralRMSNorm,
)


class TransformConfig:
    def __init__(
        self,
        flash_attention_plus,
        fast_norm,
        fast_mlp,
        fast_decoder,
        fast_attention,
        full_optimizer,
        precision,
        model_name,
        target,
        w_bit: int = 4,
        group_size: int = 128,
        profilegemm=False,
        profile_layer=False,
        mhaops=None,
    ):
        self.flash_attention_plus = flash_attention_plus
        self.fast_mlp = fast_mlp
        self.fast_norm = fast_norm
        self.fast_attention = fast_attention
        self.full_optimizer = full_optimizer
        self.fast_decoder = fast_decoder
        self.precision = precision
        self.target = target
        self.model_name = model_name  # args.model_name
        self.w_bit = w_bit
        self.group_size = group_size
        self.profilegemm = profilegemm
        self.profile_layer = profile_layer
        self.mhaops = mhaops


class RyzenAILLMEngine:
    node_count = 0
    supported_models = {
        "flash_attention_plus": [],
        "full_optimizer": [
            "chatglm3-6b",
            "microsoft/Phi-3.5-mini-instruct",
        ],
        "fast_decoder": [
            "llama-2-7b",
            "llama-2-7b-chat",
            "llama-2-13b",
            "llama-2-13b-chat",
            "CodeLlama-7b-hf",
            "code-llama-2-7b",
            "Meta-Llama-3-8B",
            "Qwen1.5-7B",
            "Llama-2-7b-hf",
            "Llama-3.2-1B-Early",  # crashes - no shapes in RoPE
            "Mistral-7B-v0.1",
        ],
        "fast_mlp": [
            "llama-2-7b",
            "llama-2-7b-chat",
            "llama-2-13b",
            "llama-2-13b-chat",
            "CodeLlama-7b-hf",
            "code-llama-2-7b",
            "Mistral-7B-v0.1",
            "Meta-Llama-3-8B",
            "Qwen1.5-7B",
            "Llama-2-7b-hf",
            "Llama-3.2-1B-Early",  # crashes - no shapes in SiLU
            "Mistral-7B-v0.1",
        ],
        "fast_attention": [
            "llama-2-7b",
            "llama-2-7b-chat",
            "Meta-Llama-3-8B",
            "Qwen1.5-7B",
            "Llama-2-7b-hf",
            "Llama-3.2-1B-Early",  # crashes - no shapes in RoPE
            "Llama-3.2-3B-Early",
            "Mistral-7B-v0.1",
        ],
        "fast_norm": [
            "llama-2-7b",
            "llama-2-7b-chat",
            "Meta-Llama-3-8B",
            "Qwen1.5-7B",
            "Llama-2-7b-hf",
            "Llama-3.2-1B-Early",  # Does not crash but not accurate - generates garbage results
            "Mistral-7B-v0.1",
        ],
    }

    @classmethod
    def replace_node(cls, model, old_node, new_node, new_node_args, new_node_kwargs={}):
        cls.node_count = 0

        def _replace(
            module, name, old_node, new_node, new_node_args, new_node_kwargs={}
        ):
            for attr_str in dir(module):
                try:
                    target_attr = getattr(module, attr_str)
                    if type(target_attr) == old_node:
                        _old = target_attr
                        _new = new_node(*new_node_args, **new_node_kwargs)
                        if (
                            _old._get_name() == "DynamicQuantizedLinear"
                        ):  # Replaced by qlinear.QLinear
                            _new.in_features = _old.in_features
                            _new.out_features = _old.out_features
                            _new.weight_bias = _old._packed_params._weight_bias()
                            _new.quantize_weights()
                            del _old
                        elif (
                            _old.__class__.__name__ == "Linear"
                        ):  # Replaced by qlinear.QLinearPerGrp
                            _new.in_features = _old.in_features
                            _new.out_features = _old.out_features
                            _new.bias = _old.bias
                            _new.weight = _old.weight
                            del _old
                        elif (
                            _old.__class__.__name__ == "WQLinear"
                        ):  # Replaced by qlinear.QLinearPerGrp
                            _new.in_features = _old.in_features
                            _new.out_features = _old.out_features
                            _new.bias = _old.bias
                            _new.w_bit = _old.w_bit
                            _new.group_size = _old.group_size
                            _new.qweight = _old.qweight
                            _new.qzeros = _old.qzeros
                            _new.scales = _old.scales
                            del _old
                            gc.collect()
                        elif _old.__class__.__name__ == "Softmax":  # experimental
                            _new.dim = _old.dim
                        elif (
                            _old.__class__.__name__ == "OPTAttention"
                        ):  # Replaced by OPTFlashAttentionPlus
                            _new.head_dim = _old.head_dim
                            _new.is_decoder = _old.is_decoder
                            _new.q_proj = copy.deepcopy(_old.q_proj)
                            _new.k_proj = copy.deepcopy(_old.k_proj)
                            _new.v_proj = copy.deepcopy(_old.v_proj)
                            _new.out_proj = copy.deepcopy(_old.out_proj)
                            del _old
                            _new.init_faplus()
                            gc.collect()

                        elif _old.__class__.__name__ in (
                            "LlamaAttention",
                            "Qwen2Attention",
                            "MistralAttention",
                            "SelfAttention",
                        ):  # Replaced by LlamaFlashAttentionPlus or LlamaFastAttention
                            _new.config = _old.config
                            _new.layer_idx = _old.layer_idx
                            _new.q_proj = copy.deepcopy(_old.q_proj)
                            _new.k_proj = copy.deepcopy(_old.k_proj)
                            _new.v_proj = copy.deepcopy(_old.v_proj)
                            _new.o_proj = copy.deepcopy(_old.o_proj)
                            del _old
                            _new.init_faplus()
                            gc.collect()
                        elif _old.__class__.__name__ in (
                            "LlamaRMSNorm",
                            "Qwen2RMSNorm",
                            "MistralRMSNorm",
                            "RMSNorm",
                            "Phi3RMSNorm",
                        ):
                            _new.weight.data = copy.deepcopy(_old.weight.data)
                            if hasattr(_old, "variance_epsilon"):
                                _new.variance_epsilon = _old.variance_epsilon
                            elif hasattr(_old, "eps"):
                                _new.variance_epsilon = _old.eps
                            del _old
                            gc.collect()
                        elif (
                            _old.__class__.__name__ == "PhiAttention"
                        ):  # Replaced by PhiFlashAttentionPlus
                            _new.config = _old.config
                            _new.layer_idx = _old.layer_idx
                            _new.q_proj = copy.deepcopy(_old.q_proj)
                            _new.k_proj = copy.deepcopy(_old.k_proj)
                            _new.v_proj = copy.deepcopy(_old.v_proj)
                            _new.dense = copy.deepcopy(_old.dense)
                            del _old
                            _new.init_faplus()
                            gc.collect()
                        elif (
                            _old.__class__.__name__ == "Phi3Attention"
                        ):  # Replaced by Phi3FlashAttentionPlus
                            _new.config = _old.config
                            _new.layer_idx = _old.layer_idx
                            _new.qkv_proj = copy.deepcopy(_old.qkv_proj)
                            _new.o_proj = copy.deepcopy(_old.o_proj)
                            del _old
                            gc.collect()

                        elif _old.__class__.__name__ in (
                            "LlamaMLP",
                            "Qwen2MLP",
                            "MistralMLP",
                            "MLP",
                        ):  # Replaced by LlamaFastMLP

                            _new.gate_proj = copy.deepcopy(_old.gate_proj)
                            _new.up_proj = copy.deepcopy(_old.up_proj)
                            _new.down_proj = copy.deepcopy(_old.down_proj)
                            _new.act_fn = _old.act_fn

                            del _old
                            gc.collect()
                        elif (
                            _old.__class__.__name__ == "SelfAttention"
                        ):  # Replaced by SelfAttention in chatglm3-6b
                            _new.layer_number = _old.layer_number
                            _new.qkv_hidden_size = copy.deepcopy(_old.qkv_hidden_size)
                            _new.query_key_value = copy.deepcopy(_old.query_key_value)
                            _new.dense = copy.deepcopy(_old.dense)
                            del _old
                            gc.collect()
                        else:
                            pass
                        setattr(module, attr_str, _new)
                        cls.node_count += 1
                except Exception as e:
                    print(
                        f"[RyzenAILLMEngine] replace_node: Exception encountered with: {attr_str}!!"
                    )
                    print(f"[RyzenAILLMEngine] Exception: {repr(e)}")
                    raise SystemExit

            for name, immediate_child_module in module.named_children():
                _replace(
                    immediate_child_module,
                    name,
                    old_node,
                    new_node,
                    new_node_args,
                    new_node_kwargs,
                )

        print(
            f"[RyzenAILLMEngine] Model transformation: Replacing {old_node} layers with {new_node} ..."
        )
        _replace(model, "model", old_node, new_node, new_node_args, new_node_kwargs)
        print(
            f"[RyzenAILLMEngine] Model transformation done!: Replaced {cls.node_count} {old_node} layers with {new_node}."
        )

    @classmethod
    def qualify(cls, model: torch.nn.Module, user_requested: Dict) -> bool:
        available_opts = {}
        for mode in user_requested.keys():
            available_opts[mode] = False
            if user_requested[mode]:
                for m in cls.supported_models[mode]:
                    if model.model_name in m:
                        available_opts[mode] = True

        ok_to_proceed = True
        for mode in user_requested.keys():
            if ((user_requested[mode] == True) and (available_opts[mode] == True)) or (
                user_requested[mode] == False
            ):
                ok_to_proceed = ok_to_proceed and True
            else:
                ok_to_proceed = False

        # print(f"[RyzenAILLMEngine] user_requested: {user_requested}")
        # print(f"[RyzenAILLMEngine] available_opts: {available_opts}")
        # print(f"[RyzenAILLMEngine] model_name: {model.model_name}")
        return ok_to_proceed

    @classmethod
    def transform(cls, model: torch.nn.Module, transform_confg: TransformConfig):
        user_requested = {
            "flash_attention_plus": transform_confg.flash_attention_plus,
            "fast_mlp": transform_confg.fast_mlp,
            "fast_norm": transform_confg.fast_norm,
            "fast_attention": transform_confg.fast_attention,
            "fast_decoder": transform_confg.fast_decoder,
            "full_optimizer": transform_confg.full_optimizer,
        }
        print(f"[RyzenAILLMEngine] Checking for available optimizations ... ")
        ok_to_proceed = cls.qualify(model, user_requested)
        if ok_to_proceed == False:
            print(
                f"[RyzenAILLMEngine] Optimizations not available for this; run without optimizations ... exiting ... !!!"
            )
            raise SystemExit

        ## Flash Attention and Attention optimizations
        if user_requested["flash_attention_plus"]:
            if "opt" in model.model_name:
                from opt_flash_attention import OPTFlashAttentionPlus

                node_args = ()
                node_kwargs = {
                    "config": model.config,
                    "precision": transform_confg.precision,
                    "model_name": transform_confg.model_name,
                }
                cls.replace_node(
                    model, OPTAttention, OPTFlashAttentionPlus, node_args, node_kwargs
                )
            elif ("llama" in model.model_name) or ("Llama" in model.model_name):
                from llama_flash_attention import LlamaFlashAttentionPlus

                node_args = ()
                node_kwargs = {
                    "config": model.config,
                    "model_name": transform_confg.model_name,
                    "precision": transform_confg.precision,
                }
                cls.replace_node(
                    model,
                    LlamaAttention,
                    LlamaFlashAttentionPlus,
                    node_args,
                    node_kwargs,
                )
            elif "Qwen" in model.model_name:
                from qwen2_flash_attention import Qwen2FlashAttentionPlus

                node_args = ()
                node_kwargs = {
                    "config": model.config,
                    "precision": transform_confg.precision,
                }
                cls.replace_node(
                    model,
                    Qwen2Attention,
                    Qwen2FlashAttentionPlus,
                    node_args,
                    node_kwargs,
                )
            elif "chatglm3" in model.model_name:
                from chatglm3_flash_attention import ChatGLM3FlashAttentionPlus

                node_args = ()
                node_kwargs = {
                    "config": model.config,
                    "layer_number": 0,
                    "model_name": transform_confg.model_name,
                }
                cls.replace_node(
                    model,
                    SelfAttention,
                    ChatGLM3FlashAttentionPlus,
                    node_args,
                    node_kwargs,
                )
            elif "Mistral" in model.model_name:
                from mistral_flash_attention import MistralFlashAttentionPlus

                node_args = ()
                node_kwargs = {
                    "config": model.config,
                    "precision": transform_confg.precision,
                }
                cls.replace_node(
                    model,
                    MistralAttention,
                    MistralFlashAttentionPlus,
                    node_args,
                    node_kwargs,
                )
            elif "phi-2" in model.model_name:
                from phi_flash_attention import PhiFlashAttentionPlus

                node_args = ()
                node_kwargs = {
                    "config": model.config,
                    "precision": transform_confg.precision,
                }
                cls.replace_node(
                    model, PhiAttention, PhiFlashAttentionPlus, node_args, node_kwargs
                )
            elif "Phi-3" in model.model_name:
                from phi_flash_attention import Phi3FlashAttentionPlus

                node_args = ()
                node_kwargs = {
                    "config": model.config,
                    "precision": transform_confg.precision,
                }
                cls.replace_node(
                    model, Phi3Attention, Phi3FlashAttentionPlus, node_args, node_kwargs
                )
        if user_requested["full_optimizer"]:
            from llm_fast_decoder import LLMFastDecoder

            print(
                f"[RyzenAILLMEngine] Model transformation: Replacing DecoderLayer with LLMFastDecoder ..."
            )
            if "chatglm3" in model.model_name:
                model_layers = model.transformer.encoder.layers
            else:
                model_layers = model.model.layers
            new_layers = torch.nn.ModuleList()
            for i, _old in enumerate(model_layers):
                _new = LLMFastDecoder.create_decoder(model, _old)
                new_layers.append(_new)
            if "chatglm3" in model.model_name:
                model.transformer.encoder.layers = new_layers

                # replace the final rmsnorm
                from modeling_chatglm import RMSNorm
                from fast_decoders.chatglm3_fast_decoder import GLMFastRMSNorm

                node_args = ()
                node_kwargs = {}
                cls.replace_node(model, RMSNorm, GLMFastRMSNorm, node_args, node_kwargs)
            else:
                model.model.layers = new_layers

                # replace the final rmsnorm
                from transformers.models.phi3.modeling_phi3 import Phi3RMSNorm
                from fast_decoders.phi3_fast_decoder import Phi3FastRMSNorm

                node_args = ()
                node_kwargs = {}
                cls.replace_node(
                    model, Phi3RMSNorm, Phi3FastRMSNorm, node_args, node_kwargs
                )
            print(f"[RyzenAILLMEngine] Model transformation: DONE")

        if user_requested["fast_decoder"]:
            if (
                ("llama" in model.model_name)
                or ("Llama" in model.model_name)
                or ("Qwen1" in model.model_name)
                or ("Mistral" in model.model_name)
            ):
                from llama_fast_decoder import LlamaFastDecoder

                print(
                    f"[RyzenAILLMEngine] Model transformation: Replacing DecoderLayer with FastDecoder ..."
                )
                new_layers = torch.nn.ModuleList()
                for i, _old in enumerate(model.model.layers):
                    _new = LlamaFastDecoder()
                    _new.hidden_size = _old.hidden_size
                    _new.self_attn = _old.self_attn
                    _new.mlp = _old.mlp
                    _new.input_layernorm = _old.input_layernorm
                    _new.post_attention_layernorm = _old.post_attention_layernorm
                    new_layers.append(_new)
                model.model.layers = new_layers
                print(f"[RyzenAILLMEngine] Model transformation: DONE")

        if user_requested["fast_attention"]:
            if ("llama" in model.model_name) or ("Llama" in model.model_name):
                from llama_fast_attention import LlamaFastAttention

                node_args = ()
                node_kwargs = {
                    "config": model.config,
                    "model_name": transform_confg.model_name,
                    "precision": transform_confg.precision,
                    "profile": transform_confg.profile_layer,
                    "mhaops": transform_confg.mhaops,
                }
                cls.replace_node(
                    model,
                    LlamaAttention,
                    LlamaFastAttention,
                    node_args,
                    node_kwargs,
                )
            elif "Qwen" in model.model_name:
                from llama_fast_attention import LlamaFastAttention

                node_args = ()
                node_kwargs = {
                    "config": model.config,
                    "precision": transform_confg.precision,
                    "profile": transform_confg.profile_layer,
                }
                cls.replace_node(
                    model,
                    Qwen2Attention,
                    LlamaFastAttention,
                    node_args,
                    node_kwargs,
                )
            elif "Mistral" in model.model_name:
                from llama_fast_attention import LlamaFastAttention

                node_args = ()
                node_kwargs = {
                    "config": model.config,
                    "precision": transform_confg.precision,
                    "profile": transform_confg.profile_layer,
                }
                cls.replace_node(
                    model,
                    MistralAttention,
                    LlamaFastAttention,
                    node_args,
                    node_kwargs,
                )
        if user_requested["fast_mlp"]:
            if (transform_confg.precision == "w4abf16") and (
                transform_confg.target == "aie"
            ):
                if ("llama" in model.model_name) or ("Llama" in model.model_name):
                    from llama_fast_mlp_npu import LlamaFastMLP

                    node_args = ()
                    node_kwargs = {"precision": transform_confg.precision}
                    cls.replace_node(
                        model, LlamaMLP, LlamaFastMLP, node_args, node_kwargs
                    )
                elif "Qwen" in model.model_name:
                    from llama_fast_mlp_npu import LlamaFastMLP

                    node_args = ()
                    node_kwargs = {"precision": transform_confg.precision}
                    cls.replace_node(
                        model, Qwen2MLP, LlamaFastMLP, node_args, node_kwargs
                    )
                elif "Mistral" in model.model_name:
                    from llama_fast_mlp_npu import LlamaFastMLP

                    node_args = ()
                    node_kwargs = {"precision": transform_confg.precision}
                    cls.replace_node(
                        model, MistralMLP, LlamaFastMLP, node_args, node_kwargs
                    )
        if user_requested["fast_norm"]:
            if (transform_confg.precision == "w4abf16") and (
                transform_confg.target == "aie"
            ):
                if ("llama" in model.model_name) or ("Llama" in model.model_name):
                    from llama_fast_rmsnorm import LlamaFastRMSNorm

                    node_args = ()
                    node_kwargs = {}
                    cls.replace_node(
                        model, LlamaRMSNorm, LlamaFastRMSNorm, node_args, node_kwargs
                    )
                elif "Qwen" in model.model_name:
                    from llama_fast_rmsnorm import LlamaFastRMSNorm

                    node_args = ()
                    node_kwargs = {}
                    cls.replace_node(
                        model, Qwen2RMSNorm, LlamaFastRMSNorm, node_args, node_kwargs
                    )
                elif "Mistral" in model.model_name:
                    from llama_fast_rmsnorm import LlamaFastRMSNorm

                    node_args = ()
                    node_kwargs = {}
                    cls.replace_node(
                        model, MistralRMSNorm, LlamaFastRMSNorm, node_args, node_kwargs
                    )
        print(model)
        for n, m in model.named_modules():
            if "LlamaFastMLP" in m.__class__.__name__:
                m.init_fastmlp()
            # if "LlamaFastAttention" in m.__class__.__name__:
            #    m.init_o_proj()
        print(model)
        if transform_confg.precision == "w4abf16":
            for n, m in model.named_modules():
                if isinstance(m, qlinear.QLinearPerGrp):
                    m.device = transform_confg.target
                    print(f"[RyzenAILLMEngine] Initializing params of layer : {n} ")
                    m.initialize_parameters()

            model = model.eval().to(torch.bfloat16)
        else:
            if transform_confg.target == "aie":
                node_args = ()
                node_kwargs = {
                    "device": transform_confg.target,
                    "quant_mode": transform_confg.precision,
                    "profiler": transform_confg.profilegemm,
                    "model_name": transform_confg.model_name,
                }
                cls.replace_node(
                    model,
                    torch.ao.nn.quantized.dynamic.modules.linear.Linear,
                    qlinear.QLinear,
                    node_args,
                    node_kwargs,
                )

        gc.collect()
        return model
