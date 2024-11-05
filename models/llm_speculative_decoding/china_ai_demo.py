import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd

# Create parameter parser
parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("json_file", type=str, help="Path to the JSON file")

args = parser.parse_args()


dic = defaultdict(list)
with open(
    args.json_file, "r"
) as f:  ##NOTE: input the generated json file by speculative decoding.
    data = json.load(f)

    target_only_prefill_time = []
    target_only_inference_time = []
    target_mean_inf_time = []
    target_only_generated_tokens = []
    spec_dec_prefill_time = []
    spec_dec_inference_time = []
    spec_dec_mean_inf_time = []
    spec_dec_generated_tokens = []
    all_to_dict = {}
    for inx, item in enumerate(data):
        print("*********************", inx, "***********************")
        target_only_generated_tokens.append(int(item["generated_tokens_base"]))
        target_only_inference_time.append(
            float(item["total time base"]) - float(item["prefill_time_base"])
        )
        target_mean_inf_time.append(float(item["total time base"]))
        target_only_prefill_time.append(float(item["prefill_time_base"]))
        for k, v in item.items():
            # if "prefill_time_draft" in k:
            #     prefill_time_draft  = float(v)
            if "prefill_time_target" in k:
                prefill_time_target = float(v)
            if "total time spec_dec" in k:
                spec_dec_inf_t = float(v)
            if "generated_tokens_spec_dec" in k:
                spec_dec_gen_tokens = int(v)
            if "t_infer_target" in k:
                print(k)
                print(v)
            if "t_infer_draft" in k:
                prefill_time_draft = np.sum(v[:6])
                print(k)
                print(v)
        spec_dec_prefill_t = prefill_time_draft + prefill_time_target
        spec_dec_prefill_time.append(spec_dec_prefill_t)
        spec_dec_mean_inf_time.append(spec_dec_inf_t)
        spec_dec_inference_time.append(spec_dec_inf_t - spec_dec_prefill_t)
        spec_dec_generated_tokens.append(spec_dec_gen_tokens)

    all_to_dict["target_only_prefill_time"] = target_only_prefill_time
    all_to_dict["target_mean_inf_time"] = target_mean_inf_time
    all_to_dict["target_only_inference_time"] = target_only_inference_time
    all_to_dict["target_only_generated_tokens"] = target_only_generated_tokens

    all_to_dict["spec_dec_prefill_time"] = spec_dec_prefill_time
    all_to_dict["spec_dec_mean_inf_time"] = spec_dec_mean_inf_time
    all_to_dict["spec_dec_inference_time"] = spec_dec_inference_time
    all_to_dict["spec_dec_generated_tokens"] = spec_dec_generated_tokens

    print("all_to_dict:\n", all_to_dict)

    print(
        "###########################Below, we have basic statistics############################"
    )
    for k, v in all_to_dict.items():
        print(
            "**************************************THIS IS A SPLIT LINE******************************************"
        )
        print(k)
        print(v)
        print(np.sum(v))
        if k == "spec_dec_generated_tokens":
            print("sum:\t", np.sum(v))
            print("min:\t", np.min(v))
            print("max:\t", np.max(v))

    print(
        "###########################Below, we have some wanted metrics############################"
    )
    target_latency = np.mean(all_to_dict["target_only_inference_time"]) / (
        np.mean(all_to_dict["target_only_generated_tokens"]) - 1
    )
    target_throughput = np.mean(all_to_dict["target_only_generated_tokens"]) / np.mean(
        all_to_dict["target_mean_inf_time"]
    )
    print("Target Decoding Latency:\t", target_latency)
    print("Target Decoding Throughput:\t", target_throughput)
    spec_dec_latency = np.mean(all_to_dict["spec_dec_inference_time"]) / (
        np.mean(all_to_dict["spec_dec_generated_tokens"]) - 2
    )
    spec_dec_throughput = np.mean(all_to_dict["spec_dec_generated_tokens"]) / np.mean(
        all_to_dict["spec_dec_mean_inf_time"]
    )
    print("Spec Decoding Latency:\t", spec_dec_latency)
    print("Spec Decoding Throughput:\t", spec_dec_throughput)
    print("Latency's speedup:\t", target_latency / spec_dec_latency)
    print("Throughput's speedup:\t", spec_dec_throughput / target_throughput)
