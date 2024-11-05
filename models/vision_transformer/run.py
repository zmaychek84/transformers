import argparse
import time

import qlinear
import qlinear_experimental
import torch
from datasets import load_dataset
from utils import LinearShapes, Utils

from transformers import AutoImageProcessor, ViTForImageClassification


def classify(n):
    # first time everything is initialized
    # so dont use it for profiling
    for i in range(n):
        start = time.time_ns()
        with torch.no_grad():
            logits = model(**inputs).logits
        end = time.time_ns()
        predicted_label = logits.argmax(-1).item()

        print(
            f"{i} Quantized model prediction: {model.config.id2label[predicted_label]} in time: {(end-start)*1e-9}s"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--modelsize",
        help="base, large",
        type=str,
        default="base",
        choices=["base", "large"],
    )
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

    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]

    image_processor = AutoImageProcessor.from_pretrained(
        "google/vit-%s-patch16-224" % args.modelsize
    )
    model = ViTForImageClassification.from_pretrained(
        "google/vit-%s-patch16-224" % args.modelsize
    )
    inputs = image_processor(image, return_tensors="pt")

    if args.precision == "fp":
        if args.target == "aie":
            print("Not supported")
            raise SystemExit
        elif args.target == "cpu":
            pass
        else:  # (args.target == "customcpu")
            node_args = ()
            node_kwargs = {"quant_mode": None}
            Utils.replace_node(
                model,
                torch.nn.Linear,
                qlinear_experimental.QLinearExperimentalCPU,
                node_args,
                node_kwargs,
            )

    else:
        model = torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        if args.target == "aie":
            node_args = ()
            node_kwargs = {"device": "aie", "quant_mode": 1}
            Utils.replace_node(
                model,
                torch.ao.nn.quantized.dynamic.modules.linear.Linear,
                qlinear.QLinear,
                node_args,
                node_kwargs,
            )
        elif args.target == "cpu":
            pass
        else:  # (args.target == "customcpu")
            node_args = ()
            node_kwargs = {"quant_mode": 1}
            Utils.replace_node(
                model,
                torch.ao.nn.quantized.dynamic.modules.linear.Linear,
                qlinear_experimental.QLinearExperimentalCPU,
                node_args,
                node_kwargs,
            )

    print(model)
    classify(5)

    if (args.precision == "fp") and (args.target == "customcpu"):
        count = Utils.count_layers(model)
        print(f"Layers in model: {count}")
        LinearShapes.get_shapes()
