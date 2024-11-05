import argparse

import onnx_tool

parser = argparse.ArgumentParser(description="onnx model analysis")
parser.add_argument("--model", "-m", type=str)

args = parser.parse_args()
# print(args.model)

onnx_tool.model_profile(args.model, savenode="profile.csv")
