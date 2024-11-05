#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import argparse
import builtins
import gc
import logging
import os
import time

import llm_eval
import llm_profile
import psutil
import torch

from transformers import (
    AutoModel,
    AutoTokenizer,
    LlamaTokenizer,
    PreTrainedTokenizerFast,
    set_seed,
)

import transformers as tv

tver = int(tv.__version__.split(".")[1])

if tver > 44:
    from transformers import (
        GemmaForCausalLM,
        GemmaTokenizer,
    )

from llm_eval import (
    AutoModelEval,
    BloomModelEval,
    GPTBigCodeModelEval,
    LlamaModelEval,
    MistralModelEval,
    OPTModelEval,
    Phi2ModelEval,
    Qwen2ModelEval,
    MambaModelEval,
    Phi3ModelEval,
    ChatGLM3ModelEval,
)

from ryzenai_llm_engine import RyzenAILLMEngine, TransformConfig
from ryzenai_llm_quantizer import QuantConfig, RyzenAILLMQuantizer


supported_models = [
    "facebook/opt-125m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "facebook/opt-13b",
    "llama-2-7b",
    "llama-2-7b-chat",
    "llama-2-13b",
    "llama-2-13b-chat",
    "codellama/CodeLlama-7b-hf",
    "codellama/CodeLlama-7b-instruct-hf",
    "code-llama-2-7b",
    "bigcode/starcoder",
    "google/gemma-2b",
    "google/gemma-7b",
    "THUDM/chatglm-6b",
    "THUDM/chatglm3-6b",
    "Qwen/Qwen-7b",
    "Qwen/Qwen1.5-7B",
    "Qwen/Qwen1.5-7B-Chat",
    "microsoft/phi-2",
    "mistralai/Mistral-7B-v0.1",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "state-spaces/mamba-1.4b-hf",
    "state-spaces/mamba-2.8b-hf",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-8B",
    "NousResearch/Meta-Llama-3.1-8B",
    "NousResearch/Meta-Llama-3-8B",
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "meta-llama/Llama-2-7b-hf",
    "nltpt/Llama-3.2-1B",
    "nltpt/Llama-3.2-3B",
    "nltpt/Llama-3.2-11B-Vision",
    "NousResearch/Llama-2-7b-hf",
    "Qwen/Qwen1.5-MoE-A2.7B-Chat",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
]

if __name__ == "__main__":

    if os.environ.get("DEVICE") == "stx":
        p = psutil.Process()
        p.cpu_affinity([0])
    torch.set_num_threads(2)

    set_seed(123)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        help="mode name",
        type=str,
        default="facebook/opt-125m",
        choices=supported_models,
    )
    parser.add_argument(
        "--target",
        help="cpu, aie",
        type=str,
        default="aie",
        choices=["cpu", "aie"],
    )
    parser.add_argument(
        "--profile_layer",
        help="layer profile",
        type=bool,
        default=False,
        choices=[False, True],
    )
    parser.add_argument(
        "--task",
        help="infershapes: shape inference; quantize:quantize the model; decode: Decode set of prompts; benchmark: Benchmark latency w.r.t prompt length; benchmark_long: Benchmark long sequences (compare with flash attn); perplexity: Measure perplexity on wikitext2 dataset; countgops: profile gops of the model for given workload;",
        type=str,
        default="decode",
        choices=[
            "infershapes",
            "quantize",
            "decode",
            "benchmark",
            "benchmark_exact",
            "countgops",
            "perplexity",
            "benchmark_long_output",
            "profilemodel128",
            "profilemodel256",
            "profilemodel512",
            "profilemodel1k",
            "profilemodel2k",
            "profilemodel3k",
            "profilemodel128",
            "mmlu",
            "humaneval",
            "chat",
            "promptcachedemo",
        ],
    )
    parser.add_argument(
        "--precision",
        help="w4abf16 - used for awq, awqplus, pergrp & bf16 runs on cpu",
        type=str,
        default="w4abf16",
        choices=["bf16", "w4abf16"],
    )
    parser.add_argument(
        "--flash_attention_plus",
        help="enable flash attn and other optimizations",
        action="store_true",
    )
    parser.add_argument(
        "--profilegemm",
        help="Log matmul times for prompt and token phases - supported only for AIE target",
        action="store_true",
    )
    parser.add_argument(
        "--dataset",
        help="Dataset - wikitext2-raw-v1, wikitext2-v1",
        type=str,
        default="raw",
        choices=["non-raw", "raw"],
    )
    parser.add_argument("--fast_mlp", help="enable fast mlp", action="store_true")
    parser.add_argument("--fast_norm", help="enable fast norm", action="store_true")
    parser.add_argument(
        "--fast_attention", help="enable fast attention", action="store_true"
    )
    parser.add_argument(
        "--fast_decoder", help="enable fast decoder", action="store_true"
    )
    parser.add_argument(
        "--full_optimizer",
        help="enable full optimize, can't coexist with fast_* option. only for chatglm3-6b",
        action="store_true",
    )
    parser.add_argument("--w_bit", help="3, 4", type=int, default=4, choices=[3, 4])
    parser.add_argument(
        "--group_size", help="128 default", type=int, default=128, choices=[32, 64, 128]
    )
    parser.add_argument(
        "--algorithm",
        help="awq, awqplus, pergrp",
        type=str,
        default="awq",
        choices=["awq", "awqplus", "pergrp"],
    )

    parser.add_argument(
        "--gen_onnx_nodes",
        help="generate onnx nodes for npu-gpu hybrid mode",
        action="store_true",
    )

    parser.add_argument(
        "--mhaops",
        help="enable ops in mha",
        type=str,
        default="all",
        choices=["bmm1", "softmax", "bmm2", "all", "pytorchmha", "libtorchflat"],
    )

    args = parser.parse_args()
    print(f"{args}")

    dev = os.getenv("DEVICE")

    trust_remote_code = False
    if "opt" in args.model_name:
        CausalLMModel = OPTModelEval
    elif ("llama" in args.model_name) or ("Llama" in args.model_name):
        CausalLMModel = LlamaModelEval
    elif "bloom" in args.model_name:
        CausalLMModel = BloomModelEval
    elif "starcoder" in args.model_name:
        CausalLMModel = GPTBigCodeModelEval
    elif "Qwen1.5" in args.model_name:
        CausalLMModel = Qwen2ModelEval
    elif "chatglm" in args.model_name:
        from tokenization_chatglm import ChatGLMTokenizer

        CausalLMModel = ChatGLM3ModelEval
        LMTokenizer = ChatGLMTokenizer
        trust_remote_code = True
    elif "Phi-3" in args.model_name:
        CausalLMModel = Phi3ModelEval
        trust_remote_code = True
    elif "phi" in args.model_name:
        CausalLMModel = Phi2ModelEval
    elif "mistral" in args.model_name:
        CausalLMModel = MistralModelEval
    elif "mamba" in args.model_name:
        CausalLMModel = MambaModelEval
    else:
        CausalLMModel = AutoModelEval

    if "llama-2" in args.model_name:
        LMTokenizer = LlamaTokenizer
    elif "Llama-3" in args.model_name:
        LMTokenizer = PreTrainedTokenizerFast
    else:
        LMTokenizer = AutoTokenizer

    log_dir = "./logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = log_dir + "/log_%s.log" % (args.model_name.replace("/", "_"))
    logging.basicConfig(filename=log_file, filemode="w", level=logging.CRITICAL)

    model_short_name = (
        args.model_name.replace("state-spaces/", "")
        .replace("facebook/", "")
        .replace("meta-llama/", "")
        .replace("bigscience/", "")
        .replace("bigcode/", "")
        .replace("codellama/", "")
        .replace("google/", "")
        .replace("THUDM/", "")
        .replace("Qwen/", "")
        .replace("microsoft/", "")
        .replace("mistralai/", "")
        .replace("TinyLlama/", "")
        .replace("NousResearch/", "")
        .replace("nltpt/", "")
    )
    qmodels_dir = "./quantized_models/"
    if not os.path.exists(qmodels_dir):
        os.makedirs(qmodels_dir)
    ckpt = qmodels_dir + "/quantized_%s_w%d_g%d_%s.pt" % (
        model_short_name,
        args.w_bit,
        args.group_size,
        args.algorithm,
    )

    ############################################################################################
    ### Step 1 - Model Quantization
    ### Step 2 - Model Transformation & Optimization
    ### Step 3 - Inference
    ############################################################################################

    if args.task == "quantize":
        if not os.path.exists(ckpt):
            if ("Qwen1.5" in args.model_name) or ("Qwen2.5" in args.model_name):
                model = CausalLMModel.from_pretrained(
                    args.model_name,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=trust_remote_code,
                    attn_implementation="eager",
                )
                model.tokenizer = LMTokenizer.from_pretrained(
                    args.model_name, trust_remote_code=trust_remote_code
                )
            elif "chatglm" in args.model_name:
                model = (
                    CausalLMModel.from_pretrained(
                        args.model_name,
                        trust_remote_code=trust_remote_code,
                        attn_implementation="eager",
                    )
                    .to(torch.bfloat16)
                    .to("cpu")
                )
                model.tokenizer = LMTokenizer.from_pretrained(
                    args.model_name, trust_remote_code=trust_remote_code
                )
            elif "gemma" in args.model_name:
                model = (
                    GemmaForCausalLM.from_pretrained(
                        args.model_name,
                        trust_remote_code=trust_remote_code,
                        attn_implementation="eager",
                    )
                    .to(torch.bfloat16)
                    .to("cpu")
                )
                model.tokenizer = GemmaTokenizer.from_pretrained(
                    args.model_name, trust_remote_code=trust_remote_code
                )
            else:
                model = CausalLMModel.from_pretrained(
                    args.model_name,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=trust_remote_code,
                    attn_implementation="eager",
                )
                model.tokenizer = LMTokenizer.from_pretrained(
                    args.model_name, trust_remote_code=trust_remote_code
                )
            model.model_name = model_short_name
            print(model)
            # set use_scales = False in quant config to calculate new awq scales
            use_qscales = True
            quant_config = QuantConfig(
                quant_mode=args.algorithm,
                model_name=model_short_name,
                dataset="raw",
                w_bit=args.w_bit,
                group_size=args.group_size,
                use_qscales=use_qscales,
            )

            ##############################################################
            ### Step 1 - Model Quantization
            model = RyzenAILLMQuantizer.quantize(model, quant_config=quant_config)
            print(model)
            ##############################################################

            torch.save(model, ckpt)
            print(f"\n\nSaved Quantized Model ... : {ckpt} !!! \n")
        else:
            print(f"\n\nFound quantized Model on disk : {ckpt} - nothing to do\n")

    else:
        if args.precision == "bf16":
            if ("Qwen1.5" in args.model_name) or ("Qwen2.5" in args.model_name):
                model = CausalLMModel.from_pretrained(
                    args.model_name,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=trust_remote_code,
                    attn_implementation="eager",
                )
                model.tokenizer = LMTokenizer.from_pretrained(
                    args.model_name, trust_remote_code=trust_remote_code
                )
            elif model_short_name == "chatglm-6b" or model_short_name == "chatglm3-6b":
                model = (
                    ChatGLM3ModelEval.from_pretrained(
                        args.model_name,
                        trust_remote_code=trust_remote_code,
                        attn_implementation="eager",
                    )
                    .to(torch.bfloat16)
                    .to("cpu")
                )
                model.tokenizer = LMTokenizer.from_pretrained(
                    args.model_name, trust_remote_code=trust_remote_code
                )
            elif "gemma" in model_short_name:
                model = (
                    GemmaForCausalLM.from_pretrained(
                        args.model_name,
                        trust_remote_code=trust_remote_code,
                        attn_implementation="eager",
                    )
                    .to(torch.bfloat16)
                    .to("cpu")
                )
                model.tokenizer = GemmaTokenizer.from_pretrained(
                    args.model_name, trust_remote_code=trust_remote_code
                )
            else:
                model = CausalLMModel.from_pretrained(
                    args.model_name,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=trust_remote_code,
                    attn_implementation="eager",
                )
                model.tokenizer = LMTokenizer.from_pretrained(
                    args.model_name, trust_remote_code=trust_remote_code
                )
            model.model_name = model_short_name
            print(f"\n\nLoaded bf16 Model {model.model_name} ... !! \n")
            # print(model)
            # print(model.config._attn_implementation)

        else:
            if not os.path.exists(ckpt):
                print(f"\n\nQuantized Model not available ... {ckpt} !!! \n")
                print(
                    f"\n\nRun with --task quantize and generate quantized model first \n"
                )
                raise SystemExit
            if ("Qwen1.5" in args.model_name) or ("Qwen2.5" in args.model_name):
                tokenizer = LMTokenizer.from_pretrained(args.model_name)
                model = CausalLMModel.from_pretrained(args.model_name, device_map="cpu")
                model.tokenizer = tokenizer
                model.model_name = model_short_name
            elif "chatglm" in args.model_name:
                model = CausalLMModel.from_pretrained(
                    args.model_name,
                    trust_remote_code=trust_remote_code,
                    attn_implementation="eager",
                ).to("cpu")
                model.tokenizer = LMTokenizer.from_pretrained(
                    args.model_name, trust_remote_code=trust_remote_code
                )
                model.model_name = model_short_name
            model = torch.load(ckpt)
            print(f"\n\nFound quantized Model on disk : {ckpt} \n")
            print(model)
            print(f"model.model_name: {model.model_name}")
        ##############################################################
        ### Step 2 - Model Transformation & Optimization
        transform_config = TransformConfig(
            flash_attention_plus=args.flash_attention_plus,
            fast_norm=args.fast_norm,
            fast_mlp=args.fast_mlp,
            fast_attention=args.fast_attention,
            fast_decoder=args.fast_decoder,
            full_optimizer=args.full_optimizer,
            precision=args.precision,
            model_name=args.model_name,
            target=args.target,
            w_bit=args.w_bit,
            group_size=args.group_size,
            profilegemm=args.profilegemm,
            profile_layer=args.profile_layer,
            mhaops=args.mhaops,
        )

        model = RyzenAILLMEngine.transform(model, transform_config)
        print(model)
        print(f"After transformation - model.mode_name: {model.model_name}")

        ##############################################################
        ### Step 3 - Inference
        from torch.utils._python_dispatch import TorchDispatchMode
        import ryzenai_torch_cpp

        if (os.environ.get("DEVICE") == "stx") and (os.environ.get("MLADF") == "2x4x4"):
            ryzenai_elemw_add = ryzenai_torch_cpp.aie_elemw_add_torch()

        class DispatchNPU(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args, kwargs=None):
                if func == torch.ops.aten.add.Tensor:
                    # print(f"Dispatch Log: {func} - (*{args}, **{kwargs})")
                    if len(args[0].shape) > 2:
                        if (args[0].shape[1] > 128) and (args[0].shape[2] == 4096):
                            return (
                                ryzenai_elemw_add.execute(args[0], args[1])
                                .unsqueeze(0)
                                .clone()
                            )
                return func(*args, **(kwargs or {}))

        # with DispatchNPU():
        if True:
            if "Llama-3.2-11B-Vision" in args.model_name:
                pass

            apply_chat_tmpl = (
                True
                if (
                    ("Qwen1.5-7B-Chat" in args.model_name)
                    or ("Qwen1.5-7B" in args.model_name)
                    or ("TinyLlama" in args.model_name)
                    or ("Phi-3" in args.model_name)
                )
                else False
            )

            if args.task == "infershapes":
                if args.target != "cpu":
                    print(
                        f"\n\n *** Set --target to CPU to infer shapes *** exiting ... "
                    )
                else:
                    llm_eval.infer_linear_shapes(model, apply_chat_tmpl=apply_chat_tmpl)

            elif args.task == "decode":
                llm_eval.decode_prompts(
                    model, log_file, apply_chat_tmpl=apply_chat_tmpl, max_new_tokens=60
                )

            elif (args.task == "benchmark") or (args.task == "benchmark_exact"):
                llm_eval.benchmark(
                    model,
                    args.dataset,
                    args.task,
                    log_file,
                    apply_chat_tmpl=apply_chat_tmpl,
                )

            # WIP - for SinkCache
            elif args.task == "benchmark_long_output":
                llm_eval.benchmark_ag(
                    model,
                    log_file,
                    max_seqlen=4096,
                    assistant_model=None,
                    do_sample=False,
                )

            elif args.task == "countgops":
                llm_eval.count_gops(model)

            elif args.task == "perplexity":
                start = time.time()
                llm_eval.perplexity(model, dataset=args.dataset)
                print(f"Time taken to measure ppl on RyzenAI: {time.time() - start}s")

            elif args.task == "mmlu":
                start = time.time()
                llm_eval.mmlu(model)
                print(f"Time taken to measure mmlu on RyzenAI: {time.time() - start}s")

            elif args.task == "chat":
                from transformers import DynamicCache, SinkCache, TextStreamer

                past_key_values = DynamicCache()
                max_cache_length = past_key_values.get_max_length()
                print(f"Chat mode: Setting max cache length to : {max_cache_length}")
                streamer = TextStreamer(model.tokenizer)
                messages = []
                while True:
                    print("-" * 20)
                    prompt = input("\nprompt: ")
                    if prompt == "exit":
                        break
                    else:
                        messages.append({"role": "user", "content": prompt})
                        inputs = model.tokenizer.apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            return_tensors="pt",
                            return_dict=True,
                        ).to(model.device)
                        if isinstance(past_key_values, SinkCache):
                            inputs = {
                                k: v[:, -max_cache_length:] for k, v in inputs.items()
                            }

                        input_length = inputs["input_ids"].shape[1]
                        if past_key_values.key_cache == []:
                            print(f"input:{input_length}")
                        else:
                            print(
                                f"input:{input_length}, layers:{len(past_key_values.key_cache)} cache_size:{past_key_values.key_cache[0].shape}"
                            )

                        outputs = model.generate(
                            **inputs,
                            do_sample=False,
                            max_new_tokens=60,
                            past_key_values=past_key_values,
                            streamer=streamer,
                        )
                        completion = model.tokenizer.decode(
                            outputs[0, input_length:], skip_special_tokens=True
                        )
                        messages.append({"role": "assistant", "content": completion})
                        print(
                            f"outputs:{outputs[0, input_length:].shape[0]}, layers:{len(past_key_values.key_cache)} cache_size:{past_key_values.key_cache[0].shape}"
                        )

            elif args.task == "promptcachedemo":
                # Init StaticCache with big enough max-length (1024 tokens for the below example)
                # You can also init a DynamicCache, if that suits you better
                from transformers import DynamicCache, TextStreamer, StaticCache
                import time
                import copy

                streamer = TextStreamer(model.tokenizer)
                prompt_cache = StaticCache(
                    config=model.config,
                    max_batch_size=1,
                    max_cache_len=1024,
                    device="cpu",
                    dtype=torch.bfloat16,
                )
                INITIAL_PROMPT_512 = "Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians, the Achaemenid Persians, and the Macedonians under Alexander the Great. The Greek Ptolemaic Kingdom, formed in the aftermath of Alexander's death, ruled until 30 BC, when, under Cleopatra, it fell to the Roman Empire and became a Roman province.[3] Egypt remained under Roman control until the 640s AD, when it was conquered by the Rashidun Caliphate.The success of ancient Egyptian civilization came partly from its ability to adapt to the conditions of the Nile River valley for agriculture. The predictable flooding and controlled irrigation of the fertile valley produced surplus crops, which supported a more dense population, and social development and culture. With resources to spare, the administration sponsored mineral exploitation of the valley and surrounding desert regions, the early development of an independent writing system, the organization of collective construction and agricultural projects, trade with surrounding regions, and a military intended to assert Egyptian dominance. Motivating and organizing these activities was a bureaucracy of elite scribes, religious leaders, and administrators under the control of a pharaoh, who ensured the "
                inputs_initial_prompt = model.tokenizer(
                    INITIAL_PROMPT_512, return_tensors="pt"
                )
                # This is the common prompt cached, we need to run forward without grad to be abel to copy
                start = time.perf_counter()
                with torch.no_grad():
                    prompt_cache = model(
                        **inputs_initial_prompt, past_key_values=prompt_cache
                    ).past_key_values
                end = time.perf_counter()
                print(f"INITIAL_PROMPT generation time: {end-start}")
                prompts = [
                    "Was Egypt ever invaded?",
                    "How many years was Egypt under Rashidun Caliphate?",
                ]
                for prompt in prompts:
                    print("*" * 20)
                    new_inputs = model.tokenizer(
                        INITIAL_PROMPT_512 + prompt, return_tensors="pt"
                    )
                    logging.critical(f"[PROFILE] tokenizer:")

                    past_key_values = copy.deepcopy(prompt_cache)
                    start = time.perf_counter()
                    outputs = model.generate(
                        **new_inputs,
                        past_key_values=past_key_values,
                        max_new_tokens=60,
                        streamer=streamer,
                    )
                    end = time.perf_counter()

                    generate_time = end - start
                    prompt_tokens = new_inputs.input_ids.shape[1]
                    num_tokens_out = outputs.shape[1]
                    new_tokens_generated = num_tokens_out - prompt_tokens
                    time_per_token = (generate_time / new_tokens_generated) * 1e3
                    logging.critical(
                        f"[PROFILE] generate: {generate_time} for {num_tokens_out} tokens; prompt-tokens: {prompt_tokens}; time per generated token: {time_per_token}"
                    )
                    response = model.tokenizer.batch_decode(outputs)[0]
                    logging.critical(f"response: {response}")

                logging.shutdown()
                out_file = log_file.replace(".log", "_profile.csv")
                out_file = open(out_file, "w")
                from llm_profile import ProfileLLM

                ProfileLLM.analyze_profiling(log_file, out_file)
                out_file.close()

            elif args.task == "humaneval":
                start = time.time()
                import llm_human_eval

                human_eval_path = (
                    os.getenv("PYTORCH_AIE_PATH")
                    + "\\tools\\humaneval-sub\\sixty_acc_dataset.json"
                )
                llm_human_eval.run_human_eval(model, human_eval_path)
                print(
                    f"Time taken to measure humaneval on RyzenAI: {time.time() - start}s"
                )

            elif args.task in [
                "profilemodel1k",
                "profilemodel2k",
                "profilemodel3k",
                "profilemodel256",
                "profilemodel128",
                "profilemodel512",
            ]:
                llm_profile.TorchModuleProfile.register_profile_hooks(model)
                prompt128 = "Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as Intermediate Periods."
                prompt252 = "Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians"
                prompt256 = "Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians"
                prompt257 = "Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians,"
                prompt511 = "Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians, the Achaemenid Persians, and the Macedonians under Alexander the Great. The Greek Ptolemaic Kingdom, formed in the aftermath of Alexander's death, ruled until 30 BC, when, under Cleopatra, it fell to the Roman Empire and became a Roman province.[3] Egypt remained under Roman control until the 640s AD, when it was conquered by the Rashidun Caliphate.The success of ancient Egyptian civilization came partly from its ability to adapt to the conditions of the Nile River valley for agriculture. The predictable flooding and controlled irrigation of the fertile valley produced surplus crops, which supported a more dense population, and social development and culture. With resources to spare, the administration sponsored mineral exploitation of the valley and surrounding desert regions, the early development of an independent writing system, the organization of collective construction and agricultural projects, trade with surrounding regions, and a military intended to assert Egyptian dominance. Motivating and organizing these activities was a bureaucracy of elite scribes, religious leaders, and administrators under the control of a pharaoh, who ensured "
                prompt512 = "Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians, the Achaemenid Persians, and the Macedonians under Alexander the Great. The Greek Ptolemaic Kingdom, formed in the aftermath of Alexander's death, ruled until 30 BC, when, under Cleopatra, it fell to the Roman Empire and became a Roman province.[3] Egypt remained under Roman control until the 640s AD, when it was conquered by the Rashidun Caliphate.The success of ancient Egyptian civilization came partly from its ability to adapt to the conditions of the Nile River valley for agriculture. The predictable flooding and controlled irrigation of the fertile valley produced surplus crops, which supported a more dense population, and social development and culture. With resources to spare, the administration sponsored mineral exploitation of the valley and surrounding desert regions, the early development of an independent writing system, the organization of collective construction and agricultural projects, trade with surrounding regions, and a military intended to assert Egyptian dominance. Motivating and organizing these activities was a bureaucracy of elite scribes, religious leaders, and administrators under the control of a pharaoh, who ensured the"
                prompt513 = "Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians, the Achaemenid Persians, and the Macedonians under Alexander the Great. The Greek Ptolemaic Kingdom, formed in the aftermath of Alexander's death, ruled until 30 BC, when, under Cleopatra, it fell to the Roman Empire and became a Roman province.[3] Egypt remained under Roman control until the 640s AD, when it was conquered by the Rashidun Caliphate.The success of ancient Egyptian civilization came partly from its ability to adapt to the conditions of the Nile River valley for agriculture. The predictable flooding and controlled irrigation of the fertile valley produced surplus crops, which supported a more dense population, and social development and culture. With resources to spare, the administration sponsored mineral exploitation of the valley and surrounding desert regions, the early development of an independent writing system, the organization of collective construction and agricultural projects, trade with surrounding regions, and a military intended to assert Egyptian dominance. Motivating and organizing these activities was a bureaucracy of elite scribes, religious leaders, and administrators under the control of a pharaoh, who ensured the people"
                prompt1k = "Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians, the Achaemenid Persians, and the Macedonians under Alexander the Great. The Greek Ptolemaic Kingdom, formed in the aftermath of Alexander's death, ruled until 30 BC, when, under Cleopatra, it fell to the Roman Empire and became a Roman province.[3] Egypt remained under Roman control until the 640s AD, when it was conquered by the Rashidun Caliphate.The success of ancient Egyptian civilization came partly from its ability to adapt to the conditions of the Nile River valley for agriculture. The predictable flooding and controlled irrigation of the fertile valley produced surplus crops, which supported a more dense population, and social development and culture. With resources to spare, the administration sponsored mineral exploitation of the valley and surrounding desert regions, the early development of an independent writing system, the organization of collective construction and agricultural projects, trade with surrounding regions, and a military intended to assert Egyptian dominance. Motivating and organizing these activities was a bureaucracy of elite scribes, religious leaders, and administrators under the control of a pharaoh, who ensured the cooperation and unity of the Egyptian people in the context of an elaborate system of religious beliefs.[4]The many achievements of the ancient Egyptians include the quarrying, surveying, and construction techniques that supported the building of monumental pyramids, temples, and obelisks; a system of mathematics, a practical and effective system of medicine, irrigation systems, and agricultural production techniques, the first known planked boats,[5] Egyptian faience and glass technology, new forms of literature, and the earliest known peace treaty, made with the Hittites.[6] Ancient Egypt has left a lasting legacy. Its art and architecture were widely copied, and its antiquities were carried off to far corners of the world. Its monumental ruins have inspired the imaginations of travelers and writers for millennia. A newfound respect for antiquities and excavations in the early modern period by Europeans and Egyptians has led to the scientific investigation of Egyptian civilization and a greater appreciation of its cultural legacy.[7]Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians, the Achaemenid Persians, and the Macedonians under Alexander the Great. The Greek Ptolemaic Kingdom"
                prompt2k = "Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians, the Achaemenid Persians, and the Macedonians under Alexander the Great. The Greek Ptolemaic Kingdom, formed in the aftermath of Alexander's death, ruled until 30 BC, when, under Cleopatra, it fell to the Roman Empire and became a Roman province.[3] Egypt remained under Roman control until the 640s AD, when it was conquered by the Rashidun Caliphate.The success of ancient Egyptian civilization came partly from its ability to adapt to the conditions of the Nile River valley for agriculture. The predictable flooding and controlled irrigation of the fertile valley produced surplus crops, which supported a more dense population, and social development and culture. With resources to spare, the administration sponsored mineral exploitation of the valley and surrounding desert regions, the early development of an independent writing system, the organization of collective construction and agricultural projects, trade with surrounding regions, and a military intended to assert Egyptian dominance. Motivating and organizing these activities was a bureaucracy of elite scribes, religious leaders, and administrators under the control of a pharaoh, who ensured the cooperation and unity of the Egyptian people in the context of an elaborate system of religious beliefs.[4]The many achievements of the ancient Egyptians include the quarrying, surveying, and construction techniques that supported the building of monumental pyramids, temples, and obelisks; a system of mathematics, a practical and effective system of medicine, irrigation systems, and agricultural production techniques, the first known planked boats,[5] Egyptian faience and glass technology, new forms of literature, and the earliest known peace treaty, made with the Hittites.[6] Ancient Egypt has left a lasting legacy. Its art and architecture were widely copied, and its antiquities were carried off to far corners of the world. Its monumental ruins have inspired the imaginations of travelers and writers for millennia. A newfound respect for antiquities and excavations in the early modern period by Europeans and Egyptians has led to the scientific investigation of Egyptian civilization and a greater appreciation of its cultural legacy.[7]Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians, the Achaemenid Persians, and the Macedonians under Alexander the Great. The Greek Ptolemaic Kingdom, formed in the aftermath of Alexander's death, ruled until 30 BC, when, under Cleopatra, it fell to the Roman Empire and became a Roman province.[3] Egypt remained under Roman control until the 640s AD, when it was conquered by the Rashidun Caliphate.The success of ancient Egyptian civilization came partly from its ability to adapt to the conditions of the Nile River valley for agriculture. The predictable flooding and controlled irrigation of the fertile valley produced surplus crops, which supported a more dense population, and social development and culture. With resources to spare, the administration sponsored mineral exploitation of the valley and surrounding desert regions, the early development of an independent writing system, the organization of collective construction and agricultural projects, trade with surrounding regions, and a military intended to assert Egyptian dominance. Motivating and organizing these activities was a bureaucracy of elite scribes, religious leaders, and administrators under the control of a pharaoh, who ensured the cooperation and unity of the Egyptian people in the context of an elaborate system of religious beliefs.[4]The many achievements of the ancient Egyptians include the quarrying, surveying, and construction techniques that supported the building of monumental pyramids, temples, and obelisks; a system of mathematics, a practical and effective system of medicine, irrigation systems, and agricultural production techniques, the first known planked boats,[5] Egyptian faience and glass technology, new forms of literature, and the earliest known peace treaty, made with the Hittites.[6] Ancient Egypt has left a lasting legacy. Its art and architecture were widely copied, and its antiquities were carried off to far corners of the world. Its monumental ruins have inspired the imaginations of travelers and writers for millennia. A newfound respect for antiquities and excavations in the early modern period by Europeans and Egyptians has led to the scientific investigation of Egyptian civilization and a greater appreciation of its cultural legacy.[7]Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians, the Achaemenid Persians, and the Macedonians under Alexander the Great. The Greek Ptolemaic Kingdom, formed in the aftermath of Alexander's death, ruled until 30 BC, when, under Cleopatra, it fell to the Roman Empire and became a Roman province.[3] Egypt remained under Roman control until the 640s AD, when it was conquered by the Rashidun Caliphate.The success of ancient Egyptian civilization came partly from its ability to adapt to the conditions of the Nile River valley for agriculture. The predictable flooding and controlled irrigation of the fertile valley produced surplus crops, which supported a more dense population, and social development and culture. With resources to spare, the administration sponsored mineral exploitation of the valley and surrounding desert regions, the early development of an independent writing system, the organization of collective construction and agricultural projects, trade with surrounding regions, and a military intended to assert Egyptian dominance. Motivating and organizing these activities was a bureaucracy of elite scribes, religious leaders, and administrators under the control of a pharaoh, who ensured the cooperation and unity of the Egyptian people in the context of an elaborate system of religious beliefs.[4]The many achievements of the ancient Egyptians include the quarrying, surveying, and construction techniques that supported the building and the structure underneath it a"
                if "Qwen1.5" in args.model_name:
                    prompt128 = "Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as Intermediate Periods. The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age"

                    prompt1k = "Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians, the Achaemenid Persians, and the Macedonians under Alexander the Great. The Greek Ptolemaic Kingdom, formed in the aftermath of Alexander's death, ruled until 30 BC, when, under Cleopatra, it fell to the Roman Empire and became a Roman province.[3] Egypt remained under Roman control until the 640s AD, when it was conquered by the Rashidun Caliphate.The success of ancient Egyptian civilization came partly from its ability to adapt to the conditions of the Nile River valley for agriculture. The predictable flooding and controlled irrigation of the fertile valley produced surplus crops, which supported a more dense population, and social development and culture. With resources to spare, the administration sponsored mineral exploitation of the valley and surrounding desert regions, the early development of an independent writing system, the organization of collective construction and agricultural projects, trade with surrounding regions, and a military intended to assert Egyptian dominance. Motivating and organizing these activities was a bureaucracy of elite scribes, religious leaders, and administrators under the control of a pharaoh, who ensured the cooperation and unity of the Egyptian people in the context of an elaborate system of religious beliefs.[4]The many achievements of the ancient Egyptians include the quarrying, surveying, and construction techniques that supported the building of monumental pyramids, temples, and obelisks; a system of mathematics, a practical and effective system of medicine, irrigation systems, and agricultural production techniques, the first known planked boats,[5] Egyptian faience and glass technology, new forms of literature, and the earliest known peace treaty, made with the Hittites.[6] Ancient Egypt has left a lasting legacy. Its art and architecture were widely copied, and its antiquities were carried off to far corners of the world. Its monumental ruins have inspired the imaginations of travelers and writers for millennia. A newfound respect for antiquities and excavations in the early modern period by Europeans and Egyptians has led to the scientific investigation of Egyptian civilization and a greater appreciation of its cultural legacy.[7]Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians, the Achaemenid Persians, and the Macedonians under Alexander the Great. The Greek Ptolemaic Kingdom, formed in the aftermath of Alexander's death, ruled until 30 BC, when, under Cleopatra, it fell to the Roman Empire and became a Roman province.[3] Egypt remained under Roman control until the 640s AD, when it was conquered by the Rashidun Caliphate.The success of ancient Egyptian civilization came partly from its ability to adapt to the conditions of the Nile River valley for agriculture. The predictable flooding and controlled irrigation of the fertile valley produced surplus crops, which supported a more dense population, and social development and culture. With resources to spare, the administration sponsored mineral exploitation of the valley and surrounding desert regions"
                    prompt2k = "Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians, the Achaemenid Persians, and the Macedonians under Alexander the Great. The Greek Ptolemaic Kingdom, formed in the aftermath of Alexander's death, ruled until 30 BC, when, under Cleopatra, it fell to the Roman Empire and became a Roman province.[3] Egypt remained under Roman control until the 640s AD, when it was conquered by the Rashidun Caliphate.The success of ancient Egyptian civilization came partly from its ability to adapt to the conditions of the Nile River valley for agriculture. The predictable flooding and controlled irrigation of the fertile valley produced surplus crops, which supported a more dense population, and social development and culture. With resources to spare, the administration sponsored mineral exploitation of the valley and surrounding desert regions, the early development of an independent writing system, the organization of collective construction and agricultural projects, trade with surrounding regions, and a military intended to assert Egyptian dominance. Motivating and organizing these activities was a bureaucracy of elite scribes, religious leaders, and administrators under the control of a pharaoh, who ensured the cooperation and unity of the Egyptian people in the context of an elaborate system of religious beliefs.[4]The many achievements of the ancient Egyptians include the quarrying, surveying, and construction techniques that supported the building of monumental pyramids, temples, and obelisks; a system of mathematics, a practical and effective system of medicine, irrigation systems, and agricultural production techniques, the first known planked boats,[5] Egyptian faience and glass technology, new forms of literature, and the earliest known peace treaty, made with the Hittites.[6] Ancient Egypt has left a lasting legacy. Its art and architecture were widely copied, and its antiquities were carried off to far corners of the world. Its monumental ruins have inspired the imaginations of travelers and writers for millennia. A newfound respect for antiquities and excavations in the early modern period by Europeans and Egyptians has led to the scientific investigation of Egyptian civilization and a greater appreciation of its cultural legacy.[7]Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians, the Achaemenid Persians, and the Macedonians under Alexander the Great. The Greek Ptolemaic Kingdom, formed in the aftermath of Alexander's death, ruled until 30 BC, when, under Cleopatra, it fell to the Roman Empire and became a Roman province.[3] Egypt remained under Roman control until the 640s AD, when it was conquered by the Rashidun Caliphate.The success of ancient Egyptian civilization came partly from its ability to adapt to the conditions of the Nile River valley for agriculture. The predictable flooding and controlled irrigation of the fertile valley produced surplus crops, which supported a more dense population, and social development and culture. With resources to spare, the administration sponsored mineral exploitation of the valley and surrounding desert regions, the early development of an independent writing system, the organization of collective construction and agricultural projects, trade with surrounding regions, and a military intended to assert Egyptian dominance. Motivating and organizing these activities was a bureaucracy of elite scribes, religious leaders, and administrators under the control of a pharaoh, who ensured the cooperation and unity of the Egyptian people in the context of an elaborate system of religious beliefs.[4]The many achievements of the ancient Egyptians include the quarrying, surveying, and construction techniques that supported the building of monumental pyramids, temples, and obelisks; a system of mathematics, a practical and effective system of medicine, irrigation systems, and agricultural production techniques, the first known planked boats,[5] Egyptian faience and glass technology, new forms of literature, and the earliest known peace treaty, made with the Hittites.[6] Ancient Egypt has left a lasting legacy. Its art and architecture were widely copied, and its antiquities were carried off to far corners of the world. Its monumental ruins have inspired the imaginations of travelers and writers for millennia. A newfound respect for antiquities and excavations in the early modern period by Europeans and Egyptians has led to the scientific investigation of Egyptian civilization and a greater appreciation of its cultural legacy.[7]Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians, the Achaemenid Persians, and the Macedonians under Alexander the Great. The Greek Ptolemaic Kingdom, formed in the aftermath of Alexander's death, ruled until 30 BC, when, under Cleopatra, it fell to the Roman Empire and became a Roman province.[3] Egypt remained under Roman control until the 640s AD, when it was conquered by the Rashidun Caliphate.The success of ancient Egyptian civilization came partly from its ability to adapt to the conditions of the Nile River valley for agriculture. The predictable flooding and controlled irrigation of the fertile valley produced surplus crops, which supported a more dense population, and social development and culture. With resources to spare, the administration sponsored mineral exploitation of the valley and surrounding desert regions, the early development of an independent writing system, the organization of collective construction and agricultural projects, trade with surrounding regions, and a military intended to assert Egyptian dominance. Motivating and organizing these activities was a bureaucracy of elite scribes, religious leaders, and administrators under the control of a pharaoh, who ensured the cooperation and unity of the Egyptian people in the context of an elaborate system of religious beliefs.[4]The many achievements of the ancient Egyptians include the quarrying, surveying, and construction techniques that supported the building and the structure underneath it a temple of monumental pyramids, temples, and obelisks; a system of mathematics, a practical and effective system of medicine, irrigation systems, and agricultural production techniques, the first known planked boats,[5] Egyptian faience and glass technology, new forms of literature, and the earliest known peace treaty, made with the Hittites.[6] Ancient Egypt has left a lasting legacy. Its art and architecture were widely copied, and its antiquities were carried off to far corners of the world. Its monumental ruins have inspired the imaginations of travelers and writers for millennia. A newfound respect for antiquities and excavations in the early modern period by Europeans and Egyptians has led to the scientific investigation of Egyptian civilization and a greater appreciation of its cultural legacy. Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories:"
                elif "Llama-3" in args.model_name:
                    prompt128 = "Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as Intermediate Periods. Ancient Egypt reached its peak during the New Kingdom (around 1550–1070 BC),"
                    prompt1k = "Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians, the Achaemenid Persians, and the Macedonians under Alexander the Great. The Greek Ptolemaic Kingdom, formed in the aftermath of Alexander's death, ruled until 30 BC, when, under Cleopatra, it fell to the Roman Empire and became a Roman province.[3] Egypt remained under Roman control until the 640s AD, when it was conquered by the Rashidun Caliphate.The success of ancient Egyptian civilization came partly from its ability to adapt to the conditions of the Nile River valley for agriculture. The predictable flooding and controlled irrigation of the fertile valley produced surplus crops, which supported a more dense population, and social development and culture. With resources to spare, the administration sponsored mineral exploitation of the valley and surrounding desert regions, the early development of an independent writing system, the organization of collective construction and agricultural projects, trade with surrounding regions, and a military intended to assert Egyptian dominance. Motivating and organizing these activities was a bureaucracy of elite scribes, religious leaders, and administrators under the control of a pharaoh, who ensured the cooperation and unity of the Egyptian people in the context of an elaborate system of religious beliefs.[4]The many achievements of the ancient Egyptians include the quarrying, surveying, and construction techniques that supported the building of monumental pyramids, temples, and obelisks; a system of mathematics, a practical and effective system of medicine, irrigation systems, and agricultural production techniques, the first known planked boats,[5] Egyptian faience and glass technology, new forms of literature, and the earliest known peace treaty, made with the Hittites.[6] Ancient Egypt has left a lasting legacy. Its art and architecture were widely copied, and its antiquities were carried off to far corners of the world. Its monumental ruins have inspired the imaginations of travelers and writers for millennia. A newfound respect for antiquities and excavations in the early modern period by Europeans and Egyptians has led to the scientific investigation of Egyptian civilization and a greater appreciation of its cultural legacy.[7]Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians, the Achaemenid Persians, and the Macedonians under Alexander the Great. The Greek Ptolemaic Kingdom, formed in the aftermath of Alexander's death, ruled until 30 BC, when, under Cleopatra, it fell to the Roman Empire and became a Roman province.[3] Egypt remained under Roman control until the 640s AD, when it was conquered by the Rashidun Caliphate.The success of ancient Egyptian civilization came partly from its ability to adapt to the conditions of the Nile River valley for agriculture. The predictable flooding and controlled irrigation of the fertile valley produced surplus crops, which supported a more dense population, and social development and culture. With resources to spare, the administration sponsored mineral exploitation of the valley and surrounding desert regions, the early development of an independent writing system"

                    prompt2k = "Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians, the Achaemenid Persians, and the Macedonians under Alexander the Great. The Greek Ptolemaic Kingdom, formed in the aftermath of Alexander's death, ruled until 30 BC, when, under Cleopatra, it fell to the Roman Empire and became a Roman province.[3] Egypt remained under Roman control until the 640s AD, when it was conquered by the Rashidun Caliphate.The success of ancient Egyptian civilization came partly from its ability to adapt to the conditions of the Nile River valley for agriculture. The predictable flooding and controlled irrigation of the fertile valley produced surplus crops, which supported a more dense population, and social development and culture. With resources to spare, the administration sponsored mineral exploitation of the valley and surrounding desert regions, the early development of an independent writing system, the organization of collective construction and agricultural projects, trade with surrounding regions, and a military intended to assert Egyptian dominance. Motivating and organizing these activities was a bureaucracy of elite scribes, religious leaders, and administrators under the control of a pharaoh, who ensured the cooperation and unity of the Egyptian people in the context of an elaborate system of religious beliefs.[4]The many achievements of the ancient Egyptians include the quarrying, surveying, and construction techniques that supported the building of monumental pyramids, temples, and obelisks; a system of mathematics, a practical and effective system of medicine, irrigation systems, and agricultural production techniques, the first known planked boats,[5] Egyptian faience and glass technology, new forms of literature, and the earliest known peace treaty, made with the Hittites.[6] Ancient Egypt has left a lasting legacy. Its art and architecture were widely copied, and its antiquities were carried off to far corners of the world. Its monumental ruins have inspired the imaginations of travelers and writers for millennia. A newfound respect for antiquities and excavations in the early modern period by Europeans and Egyptians has led to the scientific investigation of Egyptian civilization and a greater appreciation of its cultural legacy.[7]Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians, the Achaemenid Persians, and the Macedonians under Alexander the Great. The Greek Ptolemaic Kingdom, formed in the aftermath of Alexander's death, ruled until 30 BC, when, under Cleopatra, it fell to the Roman Empire and became a Roman province.[3] Egypt remained under Roman control until the 640s AD, when it was conquered by the Rashidun Caliphate.The success of ancient Egyptian civilization came partly from its ability to adapt to the conditions of the Nile River valley for agriculture. The predictable flooding and controlled irrigation of the fertile valley produced surplus crops, which supported a more dense population, and social development and culture. With resources to spare, the administration sponsored mineral exploitation of the valley and surrounding desert regions, the early development of an independent writing system, the organization of collective construction and agricultural projects, trade with surrounding regions, and a military intended to assert Egyptian dominance. Motivating and organizing these activities was a bureaucracy of elite scribes, religious leaders, and administrators under the control of a pharaoh, who ensured the cooperation and unity of the Egyptian people in the context of an elaborate system of religious beliefs.[4]The many achievements of the ancient Egyptians include the quarrying, surveying, and construction techniques that supported the building of monumental pyramids, temples, and obelisks; a system of mathematics, a practical and effective system of medicine, irrigation systems, and agricultural production techniques, the first known planked boats,[5] Egyptian faience and glass technology, new forms of literature, and the earliest known peace treaty, made with the Hittites.[6] Ancient Egypt has left a lasting legacy. Its art and architecture were widely copied, and its antiquities were carried off to far corners of the world. Its monumental ruins have inspired the imaginations of travelers and writers for millennia. A newfound respect for antiquities and excavations in the early modern period by Europeans and Egyptians has led to the scientific investigation of Egyptian civilization and a greater appreciation of its cultural legacy.[7]Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians, the Achaemenid Persians, and the Macedonians under Alexander the Great. The Greek Ptolemaic Kingdom, formed in the aftermath of Alexander's death, ruled until 30 BC, when, under Cleopatra, it fell to the Roman Empire and became a Roman province.[3] Egypt remained under Roman control until the 640s AD, when it was conquered by the Rashidun Caliphate.The success of ancient Egyptian civilization came partly from its ability to adapt to the conditions of the Nile River valley for agriculture. The predictable flooding and controlled irrigation of the fertile valley produced surplus crops, which supported a more dense population, and social development and culture. With resources to spare, the administration sponsored mineral exploitation of the valley and surrounding desert regions, the early development of an independent writing system, the organization of collective construction and agricultural projects, trade with surrounding regions, and a military intended to assert Egyptian dominance. Motivating and organizing these activities was a bureaucracy of elite scribes, religious leaders, and administrators under the control of a pharaoh, who ensured the cooperation and unity of the Egyptian people in the context of an elaborate system of religious beliefs.[4]The many achievements of the ancient Egyptians include the quarrying, surveying, and construction techniques that supported the building and the structure underneath it a temple of monumental pyramids, temples, and obelisks; a system of mathematics, a practical and effective system of medicine, irrigation systems, and agricultural production techniques, the first known planked boats,[5] Egyptian faience and glass technology, new forms of literature, and the earliest known peace treaty, made with the Hittites.[6] Ancient Egypt has left a lasting legacy. Its art and architecture were widely copied, and its antiquities were carried off to far corners of the world. Its monumental ruins have inspired the imaginations of travelers and writers for millennia. A newfound respect for antiquities and excavations in the early modern period by Europeans and Egyptians has led to the scientific investigation of Egyptian civilization and a greater appreciation of its cultural legacy. Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze"
                elif "chatglm3" in args.model_name:
                    prompt1k = "Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians, the Achaemenid Persians, and the Macedonians under Alexander the Great. The Greek Ptolemaic Kingdom, formed in the aftermath of Alexander's death, ruled until 30 BC, when, under Cleopatra, it fell to the Roman Empire and became a Roman province.[3] Egypt remained under Roman control until the 640s AD, when it was conquered by the Rashidun Caliphate.The success of ancient Egyptian civilization came partly from its ability to adapt to the conditions of the Nile River valley for agriculture. The predictable flooding and controlled irrigation of the fertile valley produced surplus crops, which supported a more dense population, and social development and culture. With resources to spare, the administration sponsored mineral exploitation of the valley and surrounding desert regions, the early development of an independent writing system, the organization of collective construction and agricultural projects, trade with surrounding regions, and a military intended to assert Egyptian dominance. Motivating and organizing these activities was a bureaucracy of elite scribes, religious leaders, and administrators under the control of a pharaoh, who ensured the cooperation and unity of the Egyptian people in the context of an elaborate system of religious beliefs.[4]The many achievements of the ancient Egyptians include the quarrying, surveying, and construction techniques that supported the building of monumental pyramids, temples, and obelisks; a system of mathematics, a practical and effective system of medicine, irrigation systems, and agricultural production techniques, the first known planked boats,[5] Egyptian faience and glass technology, new forms of literature, and the earliest known peace treaty, made with the Hittites.[6] Ancient Egypt has left a lasting legacy. Its art and architecture were widely copied, and its antiquities were carried off to far corners of the world. Its monumental ruins have inspired the imaginations of travelers and writers for millennia. A newfound respect for antiquities and excavations in the early modern period by Europeans and Egyptians has led to the scientific investigation of Egyptian civilization and a greater appreciation of its cultural legacy.[7]Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians, the Achaemenid Persians, and the Macedonians under Alexander the Great. The Greek Ptolemaic Kingdom, formed in the aftermath of Alexander's death, ruled until 30 BC, when, under Cleopatra, it fell to the Roman Empire and became a Roman province.[3] Egypt remained under Roman control until the 640s AD, when it was conquered by the Rashidun Caliphate.The success of ancient Egyptian civilization came partly from its ability to adapt to the conditions of"
                    prompt2k = "Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians, the Achaemenid Persians, and the Macedonians under Alexander the Great. The Greek Ptolemaic Kingdom, formed in the aftermath of Alexander's death, ruled until 30 BC, when, under Cleopatra, it fell to the Roman Empire and became a Roman province.[3] Egypt remained under Roman control until the 640s AD, when it was conquered by the Rashidun Caliphate.The success of ancient Egyptian civilization came partly from its ability to adapt to the conditions of the Nile River valley for agriculture. The predictable flooding and controlled irrigation of the fertile valley produced surplus crops, which supported a more dense population, and social development and culture. With resources to spare, the administration sponsored mineral exploitation of the valley and surrounding desert regions, the early development of an independent writing system, the organization of collective construction and agricultural projects, trade with surrounding regions, and a military intended to assert Egyptian dominance. Motivating and organizing these activities was a bureaucracy of elite scribes, religious leaders, and administrators under the control of a pharaoh, who ensured the cooperation and unity of the Egyptian people in the context of an elaborate system of religious beliefs.[4]The many achievements of the ancient Egyptians include the quarrying, surveying, and construction techniques that supported the building of monumental pyramids, temples, and obelisks; a system of mathematics, a practical and effective system of medicine, irrigation systems, and agricultural production techniques, the first known planked boats,[5] Egyptian faience and glass technology, new forms of literature, and the earliest known peace treaty, made with the Hittites.[6] Ancient Egypt has left a lasting legacy. Its art and architecture were widely copied, and its antiquities were carried off to far corners of the world. Its monumental ruins have inspired the imaginations of travelers and writers for millennia. A newfound respect for antiquities and excavations in the early modern period by Europeans and Egyptians has led to the scientific investigation of Egyptian civilization and a greater appreciation of its cultural legacy.[7]Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians, the Achaemenid Persians, and the Macedonians under Alexander the Great. The Greek Ptolemaic Kingdom, formed in the aftermath of Alexander's death, ruled until 30 BC, when, under Cleopatra, it fell to the Roman Empire and became a Roman province.[3] Egypt remained under Roman control until the 640s AD, when it was conquered by the Rashidun Caliphate.The success of ancient Egyptian civilization came partly from its ability to adapt to the conditions of the Nile River valley for agriculture. The predictable flooding and controlled irrigation of the fertile valley produced surplus crops, which supported a more dense population, and social development and culture. With resources to spare, the administration sponsored mineral exploitation of the valley and surrounding desert regions, the early development of an independent writing system, the organization of collective construction and agricultural projects, trade with surrounding regions, and a military intended to assert Egyptian dominance. Motivating and organizing these activities was a bureaucracy of elite scribes, religious leaders, and administrators under the control of a pharaoh, who ensured the cooperation and unity of the Egyptian people in the context of an elaborate system of religious beliefs.[4]The many achievements of the ancient Egyptians include the quarrying, surveying, and construction techniques that supported the building of monumental pyramids, temples, and obelisks; a system of mathematics, a practical and effective system of medicine, irrigation systems, and agricultural production techniques, the first known planked boats,[5] Egyptian faience and glass technology, new forms of literature, and the earliest known peace treaty, made with the Hittites.[6] Ancient Egypt has left a lasting legacy. Its art and architecture were widely copied, and its antiquities were carried off to far corners of the world. Its monumental ruins have inspired the imaginations of travelers and writers for millennia. A newfound respect for antiquities and excavations in the early modern period by Europeans and Egyptians has led to the scientific investigation of Egyptian civilization and a greater appreciation of its cultural legacy.[7]Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians, the Achaemenid Persians, and the Macedonians under Alexander the Great. The Greek Ptolemaic Kingdom, formed in the aftermath of Alexander's death, ruled until 30 BC, when, under Cleopatra, it fell to the Roman Empire and became a Roman province.[3] Egypt remained under Roman control until the 640s AD, when it was conquered by the Rashidun Caliphate.The success of ancient Egyptian civilization came partly from its ability to adapt to the conditions of the Nile River valley for agriculture. The predictable flooding and controlled irrigation of the fertile valley produced surplus crops, which supported a more dense population, and social development and culture. With resources to spare, the administration sponsored mineral exploitation of the valley and surrounding desert regions, the early development of an independent writing system, the organization of collective construction and agricultural projects, trade with surrounding regions, and a military intended to assert Egyptian dominance. Motivating and organizing these activities was a bureaucracy of elite scribes, religious leaders, and administrators under the control of a pharaoh, who ensured the cooperation and unity of the Egyptian people in the context of an elaborate system of religious beliefs.[4]The many achievements of the ancient Egyptians include the quarrying, surveying, and construction techniques that supported the building and the structure underneath it a temple of monumental pyramids, temples, and obelisks; a system of mathematics, a practical and effective system of medicine, irrigation systems, and agricultural production techniques, the first known planked boats,[5] Egyptian faience and glass technology, new forms of literature, and the earliest known peace treaty, made with the Hittites.[6] Ancient Egypt has left a lasting legacy. Its art and architecture were widely copied, and its antiquities were carried off to far corners of the world. Its monumental ruins have inspired the imaginations of travelers and writers for millennia. A newfound respect for antiquities and excavations in the early modern period by Europeans and Egyptians has led to the scientific investigation of Egyptian civilization and a greater appreciation of its cultural legacy. Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River"
                    prompt3k = "a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians, the Achaemenid Persians, and the Macedonians under Alexander the Great. The Greek Ptolemaic Kingdom, formed in the aftermath of Alexander's death, ruled until 30 BC, when, under Cleopatra, it fell to the Roman Empire and became a Roman province.[3] Egypt remained under Roman control until the 640s AD, when it was conquered by the Rashidun Caliphate.The success of ancient Egyptian civilization came partly from its ability to adapt to the conditions of the Nile River valley for agriculture. The predictable flooding and controlled irrigation of the fertile valley produced surplus crops, which supported a more dense population, and social development and culture. With resources to spare, the administration sponsored mineral exploitation of the valley and surrounding desert regions, the early development of an independent writing system, the organization of collective construction and agricultural projects, trade with surrounding regions, and a military intended to assert Egyptian dominance. Motivating and organizing these activities was a bureaucracy of elite scribes, religious leaders, and administrators under the control of a pharaoh, who ensured the cooperation and unity of the Egyptian people in the context of an elaborate system of religious beliefs.[4]The many achievements of the ancient Egyptians include the quarrying, surveying, and construction techniques that supported the building of monumental pyramids, temples, and obelisks; a system of mathematics, a practical and effective system of medicine, irrigation systems, and agricultural production techniques, the first known planked boats,[5] Egyptian faience and glass technology, new forms of literature, and the earliest known peace treaty, made with the Hittites.[6] Ancient Egypt has left a lasting legacy. Its art and architecture were widely copied, and its antiquities were carried off to far corners of the world. Its monumental ruins have inspired the imaginations of travelers and writers for millennia. A newfound respect for antiquities and excavations in the early modern period by Europeans and Egyptians has led to the scientific investigation of Egyptian civilization and a greater appreciation of its cultural legacy.[7]Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians, the Achaemenid Persians, and the Macedonians under Alexander the Great. The Greek Ptolemaic Kingdom, formed in the aftermath of Alexander's death, ruled until 30 BC, when, under Cleopatra, it fell to the Roman Empire and became a Roman province.[3] Egypt remained under Roman control until the 640s AD, when it was conquered by the Rashidun Caliphate.The success of ancient Egyptian civilization came partly from its ability to adapt to the conditions of the Nile River valley for agriculture. The predictable flooding and controlled irrigation of the fertile valley produced surplus crops, which supported a more dense population, and social development and culture. With resources to spare, the administration sponsored mineral exploitation of the valley and surrounding desert regions, the early development of an independent writing system, the organization of collective construction and agricultural projects, trade with surrounding regions, and a military intended to assert Egyptian dominance. Motivating and organizing these activities was a bureaucracy of elite scribes, religious leaders, and administrators under the control of a pharaoh, who ensured the cooperation and unity of the Egyptian people in the context of an elaborate system of religious beliefs.[4]The many achievements of the ancient Egyptians include the quarrying, surveying, and construction techniques that supported the building of monumental pyramids, temples, and obelisks; a system of mathematics, a practical and effective system of medicine, irrigation systems, and agricultural production techniques, the first known planked boats,[5] Egyptian faience and glass technology, new forms of literature, and the earliest known peace treaty, made with the Hittites.[6] Ancient Egypt has left a lasting legacy. Its art and architecture were widely copied, and its antiquities were carried off to far corners of the world. Its monumental ruins have inspired the imaginations of travelers and writers for millennia. A newfound respect for antiquities and excavations in the early modern period by Europeans and Egyptians has led to the scientific investigation of Egyptian civilization and a greater appreciation of its cultural legacy.[7]Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians, the Achaemenid Persians, and the Macedonians under Alexander the Great. The Greek Ptolemaic Kingdom, formed in the aftermath of Alexander's death, ruled until 30 BC, when, under Cleopatra, it fell to the Roman Empire and became a Roman province.[3] Egypt remained under Roman control until the 640s AD, when it was conquered by the Rashidun Caliphate.The success of ancient Egyptian civilization came partly from its ability to adapt to the conditions of the Nile River valley for agriculture. The predictable flooding and controlled irrigation of the fertile valley produced surplus crops, which supported a more dense population, and social development and culture. With resources to spare, the administration sponsored mineral exploitation of the valley and surrounding desert regions, the early development of an independent writing system, the organization of collective construction and agricultural projects, trade with surrounding regions, and a military intended to assert Egyptian dominance. Motivating and organizing these activities was a bureaucracy of elite scribes, religious leaders, and administrators under the control of a pharaoh, who ensured the cooperation and unity of the Egyptian people in the context of an elaborate system of religious beliefs.[4]The many achievements of the ancient Egyptians include the quarrying, surveying, and construction techniques that supported the building and the structure underneath it a temple of monumental pyramids, temples, and obelisks; a system of mathematics, a practical and effective system of medicine, irrigation systems, and agricultural production techniques, the first known planked boats,[5] Egyptian faience and glass technology, new forms of literature, and the earliest known peace treaty, made with the Hittites.[6] Ancient Egypt has left a lasting legacy. Its art and architecture were widely copied, and its antiquities were carried off to far corners of the world. Its monumental ruins have inspired the imaginations of travelers and writers for millennia. A newfound respect for antiquities and excavations in the early modern period by Europeans and Egyptians has led to the scientific investigation of Egyptian civilization and a greater appreciation of its cultural legacy. Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians, the Achaemenid Persians, and the Macedonians under Alexander the Great. The Greek Ptolemaic Kingdom, formed in the aftermath of Alexander's death, ruled until 30 BC, when, under Cleopatra, it fell to the Roman Empire and became a Roman province.[3] Egypt remained under Roman control until the 640s AD, when it was conquered by the Rashidun Caliphate.The success of ancient Egyptian civilization came partly from its ability to adapt to the conditions of the Nile River valley for agriculture. The predictable flooding and controlled irrigation of the fertile valley produced surplus crops, which supported a more dense population, and social development and culture. With resources to spare, the administration sponsored mineral exploitation of the valley and surrounding desert regions, the early development of an independent writing system, the organization of collectiveconstruction and agricultural projects, trade with surrounding regions, and a military intended to assert Egyptian dominance. Motivating and organizing these activities was a bureaucracy of elite scribes, religious leaders, and administrators under the control of a pharaoh, who ensured the cooperation and unity of the Egyptian people in the context of an elaborate system of religious beliefs.[4]The many achievements of the ancient Egyptians include the quarrying, surveying, and construction techniques that supported the building of monumental pyramids, temples, and obelisks; a system of mathematics, a practical and effective system of medicine, irrigation systems, and agricultural production techniques, the first known planked boats,[5] Egyptian faience and glass technology, new forms of literature, and the earliest known peace treaty, made with the Hittites.[6] Ancient Egypt has left a lasting legacy. Its art and architecture were widely copied, and its antiquities were carried off to far corners of the world. Its monumental ruins have inspired the imaginations of travelers and writers for millennia. A newfound respect for antiquities and excavations in the early modern period by Europeans and Egyptians has led to the scientific investigation of Egyptian civilization and a greater appreciation of its cultural legacy.[7]Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians, the Achaemenid Persians, and the Macedonians under Alexander the Great. The Greek Ptolemaic Kingdom, formed in the aftermath of Alexander's death, ruled until 30 BC, when, under Cleopatra, it fell to the Roman Empire and became a Roman province.[3] Egypt remained under Roman control until the 640s AD, when it was conquered by the Rashidun Caliphate.The success of ancient Egyptian civilization came partly from its ability to adapt to the conditions of the Nile River valley for agriculture. The predictable flooding and controlled irrigation of the fertile valley produced surplus crops, which supported a"
                elif "Phi-3" in args.model_name:
                    prompt2k = "Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians, the Achaemenid Persians, and the Macedonians under Alexander the Great. The Greek Ptolemaic Kingdom, formed in the aftermath of Alexander's death, ruled until 30 BC, when, under Cleopatra, it fell to the Roman Empire and became a Roman province.[3] Egypt remained under Roman control until the 640s AD, when it was conquered by the Rashidun Caliphate.The success of ancient Egyptian civilization came partly from its ability to adapt to the conditions of the Nile River valley for agriculture. The predictable flooding and controlled irrigation of the fertile valley produced surplus crops, which supported a more dense population, and social development and culture. With resources to spare, the administration sponsored mineral exploitation of the valley and surrounding desert regions, the early development of an independent writing system, the organization of collective construction and agricultural projects, trade with surrounding regions, and a military intended to assert Egyptian dominance. Motivating and organizing these activities was a bureaucracy of elite scribes, religious leaders, and administrators under the control of a pharaoh, who ensured the cooperation and unity of the Egyptian people in the context of an elaborate system of religious beliefs.[4]The many achievements of the ancient Egyptians include the quarrying, surveying, and construction techniques that supported the building of monumental pyramids, temples, and obelisks; a system of mathematics, a practical and effective system of medicine, irrigation systems, and agricultural production techniques, the first known planked boats,[5] Egyptian faience and glass technology, new forms of literature, and the earliest known peace treaty, made with the Hittites.[6] Ancient Egypt has left a lasting legacy. Its art and architecture were widely copied, and its antiquities were carried off to far corners of the world. Its monumental ruins have inspired the imaginations of travelers and writers for millennia. A newfound respect for antiquities and excavations in the early modern period by Europeans and Egyptians has led to the scientific investigation of Egyptian civilization and a greater appreciation of its cultural legacy.[7]Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians, the Achaemenid Persians, and the Macedonians under Alexander the Great. The Greek Ptolemaic Kingdom, formed in the aftermath of Alexander's death, ruled until 30 BC, when, under Cleopatra, it fell to the Roman Empire and became a Roman province.[3] Egypt remained under Roman control until the 640s AD, when it was conquered by the Rashidun Caliphate.The success of ancient Egyptian civilization came partly from its ability to adapt to the conditions of the Nile River valley for agriculture. The predictable flooding and controlled irrigation of the fertile valley produced surplus crops, which supported a more dense population, and social development and culture. With resources to spare, the administration sponsored mineral exploitation of the valley and surrounding desert regions, the early development of an independent writing system, the organization of collective construction and agricultural projects, trade with surrounding regions, and a military intended to assert Egyptian dominance. Motivating and organizing these activities was a bureaucracy of elite scribes, religious leaders, and administrators under the control of a pharaoh, who ensured the cooperation and unity of the Egyptian people in the context of an elaborate system of religious beliefs.[4]The many achievements of the ancient Egyptians include the quarrying, surveying, and construction techniques that supported the building of monumental pyramids, temples, and obelisks; a system of mathematics, a practical and effective system of medicine, irrigation systems, and agricultural production techniques, the first known planked boats,[5] Egyptian faience and glass technology, new forms of literature, and the earliest known peace treaty, made with the Hittites.[6] Ancient Egypt has left a lasting legacy. Its art and architecture were widely copied, and its antiquities were carried off to far corners of the world. Its monumental ruins have inspired the imaginations of travelers and writers for millennia. A newfound respect for antiquities and excavations in the early modern period by Europeans and Egyptians has led to the scientific investigation of Egyptian civilization and a greater appreciation of its cultural legacy.[7]Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians, the Achaemenid Persians, and the Macedonians under Alexander the Great. The Greek Ptolemaic Kingdom, formed in the aftermath of Alexander's death, ruled until 30 BC, when, under Cleopatra, it fell to the Roman Empire and became a Roman province.[3] Egypt remained under Roman control until the 640s AD, when it was conquered by the Rashidun Caliphate.The success of ancient Egyptian civilization came partly from its ability to adapt to the conditions of the Nile River valley for agriculture. The predictable flooding and controlled irrigation of the fertile valley produced surplus crops, which supported a more dense population, and social development and culture. With resources to spare, the administration sponsored mineral exploitation of the valley and surrounding desert regions, the early development of an independent writing system, the organization of collective construction and agricultural projects, trade with surrounding regions, and a military intended to assert Egyptian dominance. Motivating and organizing these activities was a bureaucracy of elite scribes, religious leaders, and administrators under the control of a pharaoh, who ensured the cooperation and unity of the Egyptian people in the context of an elaborate system of religious beliefs.[4]The many achievements of the ancient Egyptians include the quarrying, surveying, and construction techniques that supported the building and the structure underneath it a temple"
                    prompt1k = "Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians, the Achaemenid Persians, and the Macedonians under Alexander the Great. The Greek Ptolemaic Kingdom, formed in the aftermath of Alexander's death, ruled until 30 BC, when, under Cleopatra, it fell to the Roman Empire and became a Roman province.[3] Egypt remained under Roman control until the 640s AD, when it was conquered by the Rashidun Caliphate.The success of ancient Egyptian civilization came partly from its ability to adapt to the conditions of the Nile River valley for agriculture. The predictable flooding and controlled irrigation of the fertile valley produced surplus crops, which supported a more dense population, and social development and culture. With resources to spare, the administration sponsored mineral exploitation of the valley and surrounding desert regions, the early development of an independent writing system, the organization of collective construction and agricultural projects, trade with surrounding regions, and a military intended to assert Egyptian dominance. Motivating and organizing these activities was a bureaucracy of elite scribes, religious leaders, and administrators under the control of a pharaoh, who ensured the cooperation and unity of the Egyptian people in the context of an elaborate system of religious beliefs.[4]The many achievements of the ancient Egyptians include the quarrying, surveying, and construction techniques that supported the building of monumental pyramids, temples, and obelisks; a system of mathematics, a practical and effective system of medicine, irrigation systems, and agricultural production techniques, the first known planked boats,[5] Egyptian faience and glass technology, new forms of literature, and the earliest known peace treaty, made with the Hittites.[6] Ancient Egypt has left a lasting legacy. Its art and architecture were widely copied, and its antiquities were carried off to far corners of the world. Its monumental ruins have inspired the imaginations of travelers and writers for millennia. A newfound respect for antiquities and excavations in the early modern period by Europeans and Egyptians has led to the scientific investigation of Egyptian civilization and a greater appreciation of its cultural legacy.[7]Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians, the Achaemenid Persians, and the Macedonians under Alexander the Great. The Greek Ptolemaic Kingdom,"
                if "code" in model.model_name.lower():
                    prompt = "def fibonacci_recursive(n):"
                    prompts = [prompt]
                elif args.task == "profilemodel1k":
                    prompts = [prompt1k]
                elif args.task == "profilemodel256":
                    prompts = [
                        # prompt252,
                        prompt256,
                        # prompt257,
                        # prompt511,
                        # prompt512,
                        # prompt513,
                    ]
                elif args.task == "profilemodel128":
                    prompts = [
                        # prompt252,
                        prompt128,
                        # prompt257,
                        # prompt511,
                        # prompt512,
                        # prompt513,
                    ]
                elif args.task == "profilemodel512":
                    prompts = [prompt512]
                elif args.task == "profilemodel3k":
                    if "chatglm3" not in args.model_name:
                        print(f"profilemodel3k only supports model chatglm3!!!")
                        raise SystemExit
                    prompts = [prompt3k]
                else:
                    prompts = [prompt2k]

                print("here ...")
                for prompt in prompts:
                    for cnt in range(1):
                        llm_eval.decode_prompt(
                            model, model.tokenizer, prompt, max_new_tokens=60
                        )
                    print("Enter a key")
                llm_profile.TorchModuleProfile.generate_report(model)
                logging.shutdown()
                out_file = log_file.replace(".log", "_profile.csv")
                out_file = open(out_file, "w")
                llm_profile.ProfileLLM.analyze_profiling(log_file, out_file)
                out_file.close()
