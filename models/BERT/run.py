import argparse
import os
import time

import model_utils
import numpy as np
import pandas as pd
import qlinear
import qlinear_experimental
import torch
from utils import Utils

from transformers import BertForQuestionAnswering, BertTokenizer


def decode(n):
    for i in range(n):
        print("*" * 20)
        question, text = model_utils.get_question_text(data)
        answer, prof = model_utils.get_answer(model, tokenizer, question, text)

        print("\nQuestion:\n{}".format(question.capitalize()))
        if answer is not None:
            print("\nAnswer:\n{}.".format(answer.capitalize()))
        else:
            print("\nAnswer: \nNone")
        print(f"{i} Inference time: {prof*1e-9}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--target",
        help="cpu, custumcpu, aie",
        type=str,
        default="cpu",
        choices=["cpu", "customcpu", "aie"],
    )
    parser.add_argument(
        "-p",
        "--precision",
        help="fp or int",
        type=str,
        default="fp",
        choices=["fp", "int"],
    )
    args = parser.parse_args()
    print(f"{args}")

    if not os.path.exists("CoQA_data.csv"):
        model_utils.data_prep()
        print("Downloaded dataset !")
    else:
        print("Dataset already available on local drive!")
    data = pd.read_csv("CoQA_data.csv")
    data.head()
    print("Number of question and answers in dataset: ", len(data))

    model = BertForQuestionAnswering.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad"
    )
    tokenizer = BertTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad"
    )

    if args.precision == "fp":
        if args.target == "aie":
            print("Not supported")
            raise SystemExit
        elif args.target == "cpu":
            pass
        else:  # (args.target == "customcpu")
            node_kwargs = {"quant_mode": None}
            Utils.replace_node(
                model,
                torch.nn.Linear,
                qlinear_experimental.QLinearExperimentalCPU,
                (),
                node_kwargs,
            )
    else:
        model = torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        if args.target == "aie":
            node_kwargs = {"device": "aie"}
            Utils.replace_node(
                model,
                torch.ao.nn.quantized.dynamic.modules.linear.Linear,
                qlinear.QLinear,
                (),
                node_kwargs,
            )
        elif args.target == "cpu":
            pass
        else:  # (args.target == "customcpu")
            node_kwargs = {"quant_mode": 1}
            Utils.replace_node(
                model,
                torch.ao.nn.quantized.dynamic.modules.linear.Linear,
                qlinear_experimental.QLinearExperimentalCPU,
                (),
                node_kwargs,
            )

    print(model)
    decode(5)
