import argparse
import copy
import json
import os
import re
import subprocess
from collections import defaultdict

import numpy as np
import torch
from decoding_with_time_record import clip_input, infer_input_ids
from human_eval.data import write_jsonl

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

task_name = "humaneval"


def main(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ########## 1. Set models && Load models
    model_path = args.target_model
    draft_model_path = args.draft_model
    draft_model_name = re.findall(r"iter-\d+-ckpt", draft_model_path)[0]

    max_seql = 2048

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    model = model.to("cuda:0").eval()
    target_model_device = torch.device("cuda:0")

    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_path, torch_dtype=torch.float32
    )
    draft_model = draft_model.to("cuda:1").eval()
    draft_model_device = torch.device("cuda:1")

    tokenizer = AutoTokenizer.from_pretrained(model_path)  ##model_path

    ########### 2. Load simulated dataset
    ## Note: Load simulated dataset
    humanevalpath = args.humanevalpath
    with open(humanevalpath, "r") as json_file:
        humaneval_data = json.load(json_file)

    ## Note: Load full dataset
    """
    from datasets import load_from_disk
    humanevalpath = args.humanevalpath
    humaneval = load_from_disk(humanevalpath)
    humaneval_data  = humaneval['test']
    """

    prompt_shots = ""

    ########### 3. Add wanted settings, manually
    """following this format:
    wanted_setting == 'spec_dec' + 't' + wanted temperature + k + wanted max_new_tokens
    """
    added_name = "spec_dec_" + str(args.temperature) + "_k_" + str(args.max_step_draft)
    name_to_variable = "result_" + copy.deepcopy(added_name)
    main_metrics = {}
    wanted_settings = ["base", "base_draft"]
    wanted_settings.append(added_name)
    for name in wanted_settings:
        main_metrics["completion_{}".format(name)] = []
        main_metrics["time_{}".format(name)] = []
        main_metrics["token_time_{}".format(name)] = []
        main_metrics["generated_tokens_{}".format(name)] = []
        main_metrics["prefill_time_{}".format(name)] = []
        if "spec_dec" in name:
            main_metrics["matchness_{}".format(name)] = []
            main_metrics["num_drafted_tokens_{}".format(name)] = []
            main_metrics["matchness_list_{}".format(name)] = []

    ########### 4. Launch Speculative Decoding within a file context management
    result_data_path = (
        "./results/code_llama_spec_speedup_{}_with_{}_on_mi250_temp_{}_{}.json".format(
            task_name, draft_model_name, args.temperature, args.max_step_draft
        )
    )
    with open(result_data_path, "w") as json_file:
        list_of_metrics = []
        for i, prompt in enumerate(humaneval_data):
            print(i)
            input_ids = clip_input(
                tokenizer,
                prompt,
                task_name,
                max_new_tokens=args.max_new_tokens,
                prompt_shots=prompt_shots,
                max_seql=max_seql,
            )

            # NOTE: Run base/draft model for further calculation of speedup, and so on.
            print("#" * 20, "Started TARGET Model Base", "#" * 20)
            result_base = infer_input_ids(
                target_model=model,
                draft_model=draft_model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                generate_fn="base",
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                early_stop=args.early_stop,
                device=target_model_device,
                draft_device=draft_model_device,
            )
            print("#" * 20, "Finished TARGET Model Base", "#" * 20)

            print("#" * 20, "Started DRAFT Model Base", "#" * 20)
            result_base_draft = infer_input_ids(
                target_model=draft_model,
                draft_model=draft_model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                generate_fn="base",
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                early_stop=True,
                device=target_model_device,
                draft_device=draft_model_device,
            )
            print("#" * 20, "Finished DRAFT Model Base", "#" * 20)

            print("#" * 20, "Started Speculative Decoding", "#" * 20)
            # NOTE: Start speculative decoding under hyper-settings
            # create and initiate a variable
            create_variable(
                name_to_variable, dict()
            )  ##NOTE: used for flexible name whenever Temp or K are chaged
            print(name_to_variable)
            name_to_variable = infer_input_ids(
                target_model=model,
                draft_model=draft_model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                generate_fn="speculative_sample",
                early_stop=True,
                max_new_tokens=args.max_new_tokens,
                max_step_draft=args.max_step_draft,
                do_sample=True,
                do_sample_draft=False,
                temperature=args.temperature,
                device=target_model_device,
                draft_device=draft_model_device,
            )
            print("#" * 20, "Finished Speculative Decoding", "#" * 20)

            ##################################### NOTE: post-process all metrics based on the outputed results ########################################
            results = [  ## to add more rested results' settings
                ("base", result_base),
                ("base_draft", result_base_draft),
                (added_name, name_to_variable),
            ]
            # name_to_variable = 'result_' + copy.deepcopy(added_name)

            for key, result in results:
                main_metrics["time_" + key].append(result["time"])
                main_metrics["token_time_" + key].append(
                    result["time"] / result["generate_ids"].shape[1]
                )
                if "base" not in key:
                    main_metrics["matchness_" + key].append(result["matchness"])
                    main_metrics["num_drafted_tokens_" + key].append(
                        result["num_drafted_tokens"]
                    )
                    main_metrics["matchness_list_{}".format(key)].append(
                        result["matchness_list"]
                    )

            metric = {}
            for key, result in results:
                metric["Completion_" + key] = result.get("completion", None)
                metric["generated_tokens_" + key] = result["generate_ids"].shape[1]
                if "base" in key:
                    metric["prefill_time_" + key] = (
                        result["time_records"]["base_gen_0_after"]
                        - result["time_records"]["base_gen_0_before"]
                    )
                if "spec_dec" in key:
                    metric[f"mean matchness {key}"] = np.mean(
                        main_metrics["matchness_{}".format(key)]
                    )
                    metric[f"mean num_drafted_tokens {key}"] = np.mean(
                        main_metrics["num_drafted_tokens_{}".format(key)]
                    )
                    ##NOTE: calculate prefilling time
                    t_infer_draft = list()
                    t_infer_target = list()
                    for step in range(512):
                        for draft_step in range(10):
                            t_infer_draft.append(
                                result["time_records"][f"time02-{step}-{draft_step}"]
                                - result["time_records"][f"time01-{step}-{draft_step}"]
                                if result["time_records"].get(
                                    f"time02-{step}-{draft_step}", None
                                )
                                else 0
                            )
                    metric[f"t_infer_draft {key}"] = [
                        time for time in t_infer_draft if time != 0
                    ]
                    metric[f"prefill_time_draft {key}"] = metric[
                        f"t_infer_draft {key}"
                    ][0]
                    for step in range(512):
                        t_infer_target.append(
                            result["time_records"][f"time04-{step}"]
                            - result["time_records"][f"time03-{step}"]
                            if result["time_records"].get(f"time04-{step}", None)
                            else 0
                        )
                    metric[f"t_infer_target {key}"] = [
                        time for time in t_infer_target if time != 0
                    ]
                    metric[f"prefill_time_target {key}"] = metric[
                        f"t_infer_target {key}"
                    ][0]

                metric[f"total time {key}"] = result["time"]
                metric[f"mean token time {key}"] = np.mean(
                    main_metrics["token_time_{}".format(key)]
                )

                if "base" not in key:
                    metric[f"matchness_list {key}"] = main_metrics[
                        "matchness_list_{}".format(key)
                    ]

            ##NOTE: calculate speedups
            for key, result in results:
                if "spec_dec" in key:
                    metric[f"E2E mean speed up {key}"] = np.mean(
                        main_metrics["time_base"]
                    ) / np.mean(main_metrics["time_{}".format(key)])
                    metric[f"E2E mean token speed up {key}"] = np.mean(
                        main_metrics["token_time_base"]
                    ) / np.mean(main_metrics["token_time_{}".format(key)])

            for key, value in metric.items():
                if isinstance(value, float):
                    metric[key] = f"{value:.4f}"

            metric["task_id"] = prompt["task_id"]
            metric["prompts tokens num"] = len(input_ids)
            time_records = metric["time_records"] = name_to_variable.get(
                "time_records", None
            )
            metric["init_infer_time"] = name_to_variable.get("init_infer_time", None)
            name_to_variable = "result_" + copy.deepcopy(added_name)
            print("prompt['task_id']:\t", prompt["task_id"])

            print(f"data {i},{metric}")
            list_of_metrics.append(metric)
        json.dump(list_of_metrics, json_file)

    to_obtain_speedup_infos(result_data_path)
    to_accquire_pass_one_scores(result_data_path)
    run_command_with_json_file(
        result_data_path
    )  ## To have all statistics of "china ai demo"


def create_variable(var_name, value):
    locals()[var_name] = value


def to_obtain_speedup_infos(result_data: str):

    final_res = {}
    data = []
    wanted = [
        "mean token time",
        "mean time",
        "mean matchness",
        "matchness_list",
        "speed up",
        "token speed up",
    ]

    with open(result_data, "r") as f:
        eval_data = json.load(f)
        for item in eval_data:
            tmp = dict()
            for key, val in item.items():
                is_wanted = False
                for string in wanted:
                    if string in key:
                        is_wanted = True
                if is_wanted:
                    tmp[key] = val
            data.append(tmp)

    print(
        " iiiiiiiiiiiiiiiiii based on {} samples. iiiiiiiiiiiiiiiiiii".format(len(data))
    )

    for d in data:
        for k in d:
            if isinstance(d[k], list):
                continue
            if k not in final_res:
                final_res[k] = 0.0
            final_res[k] += float(d[k])

    for k in final_res:
        if isinstance(final_res[k], list):
            continue
        final_res[k] /= len(data)

    print("====================================================")
    for k in final_res:
        if "mean token time" in k:  ##mean speed.
            print(k, 1 / final_res[k], "tokens/sec")

    print("====================================================")
    for k in final_res:
        if "mean time" in k:
            print(k, final_res[k], "sec")

    print("====================================================")
    for k in final_res:
        if "mean matchness" in k:
            hyperK = int(k[-1]) if k[-1] != "0" else int(k[-2:])
            print("hyperK:\t", hyperK)
            print(
                k + "tokens",
                final_res[k],
                "\t",
                final_res[k] * hyperK,
                "\t",
                (1 - final_res[k]) * hyperK,
            )

    print("====================================================")
    for k in final_res:
        if "matchness_list" in k:
            print(k, final_res[k])

    print("====================================================")
    for k in final_res:
        if "speed up" in k and "token" not in k:
            print(k, final_res[k])
    print("====================================================")

    for k in final_res:
        if "token speed up" in k:
            print(k, final_res[k])
    print("based on {} samples.".format(len(data)))


def to_accquire_pass_one_scores(result_data: str):

    def count_indent(text: str) -> int:
        count = 0
        for char in text:
            if char == " ":
                count += 1
            else:
                break
        return count

    def fix_indents(text: str, multiple: int = 2):
        outputs = []
        for line in text.split("\n"):
            while count_indent(line) % multiple != 0:
                line = " " + line
            outputs.append(line)
        return "\n".join(outputs)

    def filter_code(completion: str, model=None) -> str:
        completion = completion.lstrip("\n")
        return completion.split("\n\n")[0]

    cnter = 0

    sixty_acc_samples = [
        6,
        7,
        22,
        28,
        35,
        57,
        59,
        129,
        151,
        154,
    ]  ##NOTE: this is decided by the sampled specific code prompts

    to_mk_path = "./results/samples_{}".format(result_data[:-5][10:])
    if not os.path.exists(to_mk_path):
        os.mkdir(to_mk_path)
    samples_dir = to_mk_path

    with open(result_data, "r") as f:
        data = json.load(f)
        big_reorg_dict = defaultdict(list)
        for inx, item in enumerate(data):
            metrics = item  ##json.loads(item)
            completion_keys = list()
            for key, val in metrics.items():
                if "Completion" in key:
                    completion_keys.append(key)
                    # val_completion = filter_code(val)
                    val_completion = "    " + filter_code(fix_indents(val))
                    # comp_item = dict(task_id="HumanEval/{}".format(inx), completion=val_completion)
                    comp_item = dict(
                        task_id="HumanEval/{}".format(sixty_acc_samples[inx]),
                        completion=val_completion,
                    )
                    big_reorg_dict[key].append(comp_item)
            cnter += 1
            print("*" * 20, cnter, "*" * 20)

    for key, completion in big_reorg_dict.items():
        write_jsonl("{0}/{1}.jsonl".format(samples_dir, key), completion)

    # Bash script file path
    bash_script = "./batched_exec_humaneval.sh"

    def run_bash_script(script_path):
        # Running the Bash script using subprocess.run
        process = subprocess.run(
            [script_path, to_mk_path], capture_output=True, text=True
        )

        # Checking the execution result
        if process.returncode == 0:
            print("Bash script executed successfully.")
            # Outputting the standard output and standard error of the Bash script
            print("STDOUT:", process.stdout)
            print("STDERR:", process.stderr)
        else:
            print("Bash script failed to execute.")
            print("STDERR:", process.stderr)

    print("start pass@1 score evaluating!!!")
    run_bash_script(bash_script)


def run_command_with_json_file(json_file):
    """
    Run a command with the specified JSON file.

    Args:
        json_file (str): Path to the JSON file.
    """
    # Define the command to execute, where json_file is a variable
    command = f"python china_ai_demo.py {json_file}"

    # Use subprocess to run the command
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    # Wait for the command to finish executing
    stdout, stderr = process.communicate()

    # Print the output of the command
    print("STDOUT:", stdout.decode())
    print("STDERR:", stderr.decode())

    # Check if the command executed successfully
    if process.returncode == 0:
        print("Command executed successfully.")
    else:
        print("Command failed with return code:", process.returncode)


class ParameterManager:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Model Training Parameters")
        self.parser.add_argument(
            "--target_model", type=str, default="target_model", help="Target model"
        )
        self.parser.add_argument(
            "--draft_model", type=str, default="draft_model", help="Draft model"
        )
        self.parser.add_argument(
            "--early_stop",
            type=bool,
            default=True,
            help="Whether to use early stopping",
        )
        self.parser.add_argument(
            "--max_new_tokens", type=int, default=512, help="Maximum new tokens"
        )
        self.parser.add_argument(
            "--max_step_draft", type=int, default=4, help="Maximum steps for draft"
        )
        self.parser.add_argument(
            "--do_sample", type=bool, default=True, help="Whether to use sampling"
        )
        self.parser.add_argument(
            "--do_sample_draft",
            type=bool,
            default=False,
            help="Whether to sample for draft",
        )
        self.parser.add_argument(
            "--temperature", type=float, default=0.0, help="Temperature for sampling"
        )
        self.parser.add_argument("--seed", type=int, default=42, help="Seed value")
        self.parser.add_argument(
            "--device",
            type=str,
            default="target_model_device",
            help="Device for main model",
        )
        self.parser.add_argument(
            "--draft_device",
            type=str,
            default="draft_model_device",
            help="Device for draft model",
        )
        self.parser.add_argument(
            "--humanevalpath",
            type=str,
            default="./humaneval-sub/sixty_acc_dataset.json",
            help="Path to humaneval file",
        )

    def parse_args(self):
        return self.parser.parse_args()


if __name__ == "__main__":

    param_manager = ParameterManager()
    args = param_manager.parse_args()

    # NOTE: Created to store results
    subdirectory = "results"
    current_directory = os.getcwd()
    subdirectory_path = os.path.join(current_directory, subdirectory)
    if not os.path.exists(subdirectory_path):
        os.makedirs(subdirectory_path)
        print(f"Subdirectory '{subdirectory}' created successfully.")
    else:
        print(f"Subdirectory '{subdirectory}' already exists.")

    main(args)
