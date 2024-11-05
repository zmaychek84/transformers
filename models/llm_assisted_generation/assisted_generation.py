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
import psutil
import torch
import llm_profile

from llm_eval import AutoModelEval, LlamaForCausalLMPadded, LlamaModelEval, OPTModelEval
from ryzenai_llm_engine import RyzenAILLMEngine, TransformConfig

import transformers
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, set_seed

set_seed(12)

qmodels_dir = "..\llm\quantized_models"


def load_awq_model(model_short_name="opt-6.7b", dev="aie"):
    w_bit, group_size = 4, 128
    ckpt = qmodels_dir + "/quantized_%s_w%d_g%d_awqplus.pth" % (
        model_short_name,
        w_bit,
        group_size,
    )
    if not os.path.exists(ckpt):
        print(f"[load_awq_model] File not found : {ckpt}")
        print(
            f"[load_awq_model] quantize using ..\\llm\\run_awq.py and then run assisted generation ... exiting ..."
        )
        raise SystemExit

    model = torch.load(ckpt)
    print(f"[load_awq_model] model.mode_name: {model.model_name}")

    transform_config = TransformConfig(
        flash_attention_plus=False,
        fast_attention=True,
        fast_mlp=True,
        fast_norm=True,
        precision="w4abf16",
        model_name=model_short_name,
        target=dev,
        w_bit=w_bit,
        group_size=group_size,
        profilegemm=False,
        profile_layer=False,
        mhaops="ag",
    )
    model = RyzenAILLMEngine.transform(model, transform_config)
    print(f"[load_awq_model] model loaded ...")
    print(model)
    return model


def load_smoothquant_model(model_short_name="opt-125m", dev="cpu"):
    ckpt = qmodels_dir + "/quantized_%s_smoothquant.pth" % (model_short_name)
    if not os.path.exists(ckpt):
        print(f"[load_smoothquant_model] File not found : {ckpt}")
        print(
            f"[load_smoothquant_model] quantize using ..\\llm\\run_smoothquant.py and then run assisted generation ... exiting ..."
        )
        raise SystemExit
    model = torch.load(ckpt)
    print(f"[load_smoothquant_model] model.mode_name: {model.model_name}")
    if "opt" in model_short_name:
        model_name = "facebook/" + model_short_name
    else:
        model_name = model_short_name
    transform_config = TransformConfig(
        flash_attention_plus=False,
        fast_mlp=False,
        precision="w8a8",
        model_name=model_name,
        target=dev,
        profilegemm=False,
    )
    model = RyzenAILLMEngine.transform(model, transform_config)
    print(f"[load_smoothquant_model] model loaded ...")
    print(model)
    return model


def load_models(args):
    if True:
        target_model = load_awq_model(model_short_name=args.model_name, dev="aie")
    else:
        # target on CPU BF16 - do not enable this - only for testing
        target_model = LlamaForCausalLM.from_pretrained(
            "codellama/CodeLlama-7b-hf",
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        target_model.tokenizer = LlamaTokenizer.from_pretrained(
            "codellama/CodeLlama-7b-hf"
        )
        target_model.model_name = "CodeLlama-7b-hf"
        print(target_model)

    if args.assisted_generation:
        if args.draft_precision == "bf16":
            from transformers import AutoModelForCausalLM

            if "opt" in args.model_name:
                assistant_model = AutoModelForCausalLM.from_pretrained(
                    "facebook/opt-125m", torch_dtype=torch.bfloat16
                )
            elif "llama-2-7b" in args.model_name:
                # assistant_model = AutoModelForCausalLM.from_pretrained(
                #    qmodels_dir + "\\..\\amd-pretrained-135m-2k-no-book3",
                #    torch_dtype=torch.bfloat16,
                # )
                # assistant_model = AutoModelForCausalLM.from_pretrained(
                #    qmodels_dir + "\\..\\amd-pretrained-135m-hf_4.37.2",
                #    torch_dtype=torch.bfloat16,
                # )
                assistant_model = AutoModelForCausalLM.from_pretrained(
                    "JackFram/llama-160m",
                    torch_dtype=torch.bfloat16,
                )
            elif "Qwen1.5-7B-Chat" in args.model_name:
                assistant_model = AutoModelForCausalLM.from_pretrained(
                    "Qwen/Qwen1.5-0.5B-Chat", torch_dtype=torch.bfloat16
                )
            else:  # CodeLlama:
                assistant_model = LlamaForCausalLMPadded.from_pretrained(
                    qmodels_dir + "\\..\\amd-code-135m_104k_python_only_hf_4.37.2",
                    torch_dtype=torch.bfloat16,
                )
                assistant_model.generation_config.num_assistant_tokens = 6  # K number
                assistant_model.generation_config.num_assistant_tokens_schedule = (
                    "constant"
                )

            assistant_model = assistant_model.to(torch.bfloat16)
            print(f"[load_models] assistant model loaded ...")
            print(assistant_model)
        else:
            if "opt" in args.model_name:
                if args.assistant_aie is True:
                    dev = "aie"
                else:
                    dev = "cpu"
                if args.draft_precision == "w4abf16":
                    assistant_model = load_awq_model(
                        assistant_ckpt, opt_model="opt-125m", dev=dev
                    )
                else:
                    assistant_model = load_smoothquant_model(
                        model_short_name="opt-125m", dev=dev
                    )
                print(
                    f"[load_models] assistant_model.model_name: {assistant_model.model_name}"
                )
            elif "code" in args.model_name.lower():
                assistant_model = torch.load(
                    qmodels_dir + "\\amd-code-135m_smoothquant.pth"
                )  # Smoothquant version of amd-code-135m_104k_python_only_hf_4.37.2
                torch.ao.quantization.quantize_dynamic(
                    assistant_model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
                )
                assistant_model.generation_config.num_assistant_tokens = 6  # K number
                assistant_model.generation_config.num_assistant_tokens_schedule = (
                    "constant"
                )

            elif "llama" in args.model_name.lower():
                from transformers import AutoModelForCausalLM

                # assistant_model = AutoModelForCausalLM.from_pretrained(
                #    qmodels_dir + "\\..\\amd-pretrained-135m-2k-no-book3"
                # )
                assistant_model = AutoModelForCausalLM.from_pretrained(
                    "JackFram/llama-160m"
                )
                torch.ao.quantization.quantize_dynamic(
                    assistant_model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
                )
            else:
                print(
                    f"[load_models] No quantized model available, use bf16 for assistant model ... exiting "
                )
                raise SystemExit
            print(f"[load_models] assistant model loaded ...")
            print(assistant_model)
    else:
        assistant_model = None
    return target_model, assistant_model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # no speedup on "Qwen1.5-7B-Chat"
    # needs update on "opt-6.7b""
    parser.add_argument(
        "--model_name",
        help="LLM model name",
        type=str,
        default="llama-2-7b",
        choices=["llama-2-7b", "CodeLlama-7b-hf"],
    )
    parser.add_argument(
        "--task",
        help="benchmark: Benchmark latency w.r.t prompt length; decode: Decode set of prompts;",
        type=str,
        default="decode",
        choices=["decode", "benchmark", "benchmark_code"],
    )
    parser.add_argument(
        "--assisted_generation", help="Enable assisted generation", action="store_true"
    )
    parser.add_argument(
        "--draft_precision",
        help="Precision of draft model",
        type=str,
        default="bf16",
        choices=["w8af32", "bf16", "w4abf16"],
    )
    # parser.add_argument("--assistant_aie", help="run assistant on AIE - currently supported only for opt-125m", action='store_true')

    args = parser.parse_args()
    print(f"{args}")
    dev = os.getenv("DEVICE")

    log_dir = "./logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = log_dir + "/log_%s.log" % args.model_name

    logging.basicConfig(filename=log_file, filemode="w", level=logging.CRITICAL)

    target_model, assistant_model = load_models(args)

    if args.task == "decode":
        llm_eval.decode_prompts(target_model, log_file, assistant_model=assistant_model)

    elif args.task == "benchmark":
        llm_profile.TorchModuleProfile.register_profile_hooks(target_model)
        prompt256 = "Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians"
        prompt512 = "Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians, the Achaemenid Persians, and the Macedonians under Alexander the Great. The Greek Ptolemaic Kingdom, formed in the aftermath of Alexander's death, ruled until 30 BC, when, under Cleopatra, it fell to the Roman Empire and became a Roman province.[3] Egypt remained under Roman control until the 640s AD, when it was conquered by the Rashidun Caliphate.The success of ancient Egyptian civilization came partly from its ability to adapt to the conditions of the Nile River valley for agriculture. The predictable flooding and controlled irrigation of the fertile valley produced surplus crops, which supported a more dense population, and social development and culture. With resources to spare, the administration sponsored mineral exploitation of the valley and surrounding desert regions, the early development of an independent writing system, the organization of collective construction and agricultural projects, trade with surrounding regions, and a military intended to assert Egyptian dominance. Motivating and organizing these activities was a bureaucracy of elite scribes, religious leaders, and administrators under the control of a pharaoh, who ensured the"
        prompt1k = "Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians, the Achaemenid Persians, and the Macedonians under Alexander the Great. The Greek Ptolemaic Kingdom, formed in the aftermath of Alexander's death, ruled until 30 BC, when, under Cleopatra, it fell to the Roman Empire and became a Roman province.[3] Egypt remained under Roman control until the 640s AD, when it was conquered by the Rashidun Caliphate.The success of ancient Egyptian civilization came partly from its ability to adapt to the conditions of the Nile River valley for agriculture. The predictable flooding and controlled irrigation of the fertile valley produced surplus crops, which supported a more dense population, and social development and culture. With resources to spare, the administration sponsored mineral exploitation of the valley and surrounding desert regions, the early development of an independent writing system, the organization of collective construction and agricultural projects, trade with surrounding regions, and a military intended to assert Egyptian dominance. Motivating and organizing these activities was a bureaucracy of elite scribes, religious leaders, and administrators under the control of a pharaoh, who ensured the cooperation and unity of the Egyptian people in the context of an elaborate system of religious beliefs.[4]The many achievements of the ancient Egyptians include the quarrying, surveying, and construction techniques that supported the building of monumental pyramids, temples, and obelisks; a system of mathematics, a practical and effective system of medicine, irrigation systems, and agricultural production techniques, the first known planked boats,[5] Egyptian faience and glass technology, new forms of literature, and the earliest known peace treaty, made with the Hittites.[6] Ancient Egypt has left a lasting legacy. Its art and architecture were widely copied, and its antiquities were carried off to far corners of the world. Its monumental ruins have inspired the imaginations of travelers and writers for millennia. A newfound respect for antiquities and excavations in the early modern period by Europeans and Egyptians has led to the scientific investigation of Egyptian civilization and a greater appreciation of its cultural legacy.[7]Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians, the Achaemenid Persians, and the Macedonians under Alexander the Great. The Greek Ptolemaic Kingdom"
        prompt2k = "Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians, the Achaemenid Persians, and the Macedonians under Alexander the Great. The Greek Ptolemaic Kingdom, formed in the aftermath of Alexander's death, ruled until 30 BC, when, under Cleopatra, it fell to the Roman Empire and became a Roman province.[3] Egypt remained under Roman control until the 640s AD, when it was conquered by the Rashidun Caliphate.The success of ancient Egyptian civilization came partly from its ability to adapt to the conditions of the Nile River valley for agriculture. The predictable flooding and controlled irrigation of the fertile valley produced surplus crops, which supported a more dense population, and social development and culture. With resources to spare, the administration sponsored mineral exploitation of the valley and surrounding desert regions, the early development of an independent writing system, the organization of collective construction and agricultural projects, trade with surrounding regions, and a military intended to assert Egyptian dominance. Motivating and organizing these activities was a bureaucracy of elite scribes, religious leaders, and administrators under the control of a pharaoh, who ensured the cooperation and unity of the Egyptian people in the context of an elaborate system of religious beliefs.[4]The many achievements of the ancient Egyptians include the quarrying, surveying, and construction techniques that supported the building of monumental pyramids, temples, and obelisks; a system of mathematics, a practical and effective system of medicine, irrigation systems, and agricultural production techniques, the first known planked boats,[5] Egyptian faience and glass technology, new forms of literature, and the earliest known peace treaty, made with the Hittites.[6] Ancient Egypt has left a lasting legacy. Its art and architecture were widely copied, and its antiquities were carried off to far corners of the world. Its monumental ruins have inspired the imaginations of travelers and writers for millennia. A newfound respect for antiquities and excavations in the early modern period by Europeans and Egyptians has led to the scientific investigation of Egyptian civilization and a greater appreciation of its cultural legacy.[7]Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians, the Achaemenid Persians, and the Macedonians under Alexander the Great. The Greek Ptolemaic Kingdom, formed in the aftermath of Alexander's death, ruled until 30 BC, when, under Cleopatra, it fell to the Roman Empire and became a Roman province.[3] Egypt remained under Roman control until the 640s AD, when it was conquered by the Rashidun Caliphate.The success of ancient Egyptian civilization came partly from its ability to adapt to the conditions of the Nile River valley for agriculture. The predictable flooding and controlled irrigation of the fertile valley produced surplus crops, which supported a more dense population, and social development and culture. With resources to spare, the administration sponsored mineral exploitation of the valley and surrounding desert regions, the early development of an independent writing system, the organization of collective construction and agricultural projects, trade with surrounding regions, and a military intended to assert Egyptian dominance. Motivating and organizing these activities was a bureaucracy of elite scribes, religious leaders, and administrators under the control of a pharaoh, who ensured the cooperation and unity of the Egyptian people in the context of an elaborate system of religious beliefs.[4]The many achievements of the ancient Egyptians include the quarrying, surveying, and construction techniques that supported the building of monumental pyramids, temples, and obelisks; a system of mathematics, a practical and effective system of medicine, irrigation systems, and agricultural production techniques, the first known planked boats,[5] Egyptian faience and glass technology, new forms of literature, and the earliest known peace treaty, made with the Hittites.[6] Ancient Egypt has left a lasting legacy. Its art and architecture were widely copied, and its antiquities were carried off to far corners of the world. Its monumental ruins have inspired the imaginations of travelers and writers for millennia. A newfound respect for antiquities and excavations in the early modern period by Europeans and Egyptians has led to the scientific investigation of Egyptian civilization and a greater appreciation of its cultural legacy.[7]Ancient Egypt was a civilization of ancient Northeast Africa. It was concentrated along the lower reaches of the Nile River, situated in the place that is now the country Egypt. Ancient Egyptian civilization followed prehistoric Egypt and coalesced around 3100 BC (according to conventional Egyptian chronology)[1] with the political unification of Upper and Lower Egypt under Menes (often identified with Narmer).[2] The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by periods of relative instability known as “Intermediate Periods.” The various kingdoms fall into one of three categories: the Old Kingdom of the Early Bronze Age, the Middle Kingdom of the Middle Bronze Age, or the New Kingdom of the Late Bronze Age.Ancient Egypt reached the pinnacle of its power during the New Kingdom, ruling much of Nubia and a sizable portion of the Levant. After this period, it entered an era of slow decline. During the course of its history, Ancient Egypt was invaded or conquered by a number of foreign powers, including the Hyksos, the Nubians, the Assyrians, the Achaemenid Persians, and the Macedonians under Alexander the Great. The Greek Ptolemaic Kingdom, formed in the aftermath of Alexander's death, ruled until 30 BC, when, under Cleopatra, it fell to the Roman Empire and became a Roman province.[3] Egypt remained under Roman control until the 640s AD, when it was conquered by the Rashidun Caliphate.The success of ancient Egyptian civilization came partly from its ability to adapt to the conditions of the Nile River valley for agriculture. The predictable flooding and controlled irrigation of the fertile valley produced surplus crops, which supported a more dense population, and social development and culture. With resources to spare, the administration sponsored mineral exploitation of the valley and surrounding desert regions, the early development of an independent writing system, the organization of collective construction and agricultural projects, trade with surrounding regions, and a military intended to assert Egyptian dominance. Motivating and organizing these activities was a bureaucracy "
        prompts = [prompt256, prompt512, prompt1k, prompt2k]
        for prompt in prompts:
            llm_eval.decode_prompt(
                target_model,
                target_model.tokenizer,
                prompt,
                max_new_tokens=60,
                assistant_model=assistant_model,
            )
        llm_profile.TorchModuleProfile.generate_report(target_model)
        logging.shutdown()
        out_file = log_file.replace(".log", "_profile.csv")
        out_file = open(out_file, "w")
        llm_profile.ProfileLLM.analyze_profiling(log_file, out_file)
        out_file.close()

        # llm_eval.benchmark(
        #    target_model,
        #    "raw",
        #    "benchmark_ag",
        #    log_file,
        #    assistant_model=assistant_model,
        #    apply_chat_tmpl=False,
        # )
    elif args.task == "benchmark_code":
        if "llama" in args.model_name or "Qwen1.5" in args.model_name:
            max_seqlen = 4096
        else:
            max_seqlen = 2048
        llm_eval.benchmark_code(
            target_model,
            log_file,
            max_seqlen=max_seqlen,
            assistant_model=assistant_model,
        )
