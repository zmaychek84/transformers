import onnx
from onnx import IR_VERSION, TensorProto, __version__, shape_inference
from onnx.checker import check_model
from onnx.defs import onnx_opset_version
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info

print(
    f"onnx.__version__={__version__!r}, opset={onnx_opset_version()}, IR_VERSION={IR_VERSION}"
)

import os
import sys
import time

import numpy as np
import torch

sys.path.append(os.getenv("PYTORCH_AIE_PATH") + "/ext/stable_diffusion/")

from stable_diffusion_pytorch import model_loader

pytorch_diffusion = model_loader.load_diffusion(torch.device("cpu"))

# FP32 Model
model_path = "diffusion_onnx\diffusion.onnx"
onnx_model = onnx.load(model_path)

input_latents = torch.rand((2, 4, 64, 64))
context = torch.rand((2, 77, 768))
time_embedding = torch.rand((1, 320))

EP_list = ["CPUExecutionProvider"]
ort_sess = ort.InferenceSession(model_path, providers=EP_list)
# ort_sess.set_providers(['CPUExecutionProvider'])

start = time.time_ns()
pt_outputs = pytorch_diffusion(input_latents, context, time_embedding)
end = time.time_ns()
print(f"Pytorch FP32 Diffuser : {(end-start)*1e-9}s")

pytorch_diffusion_quant = torch.ao.quantization.quantize_dynamic(
    pytorch_diffusion, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
)

start = time.time_ns()
pt_outputs = pytorch_diffusion_quant(input_latents, context, time_embedding)
end = time.time_ns()
print(f"Pytorch Quant Diffuser : {(end-start)*1e-9}s")


start = time.time_ns()
outputs = ort_sess.run(
    None,
    {
        "input_latents": input_latents.numpy(),
        "context": context.numpy(),
        "time_embedding": time_embedding.numpy(),
    },
)
end = time.time_ns()
print(f"ONNX FP32 Diffuser : {(end-start)*1e-9}s")

model_quant = "diffusion_onnx_quant\diffusion_quant.onnx"

if True:
    quantized_model = quantize_dynamic(
        model_path, model_quant, weight_type=QuantType.QUInt8
    )
    print(f"ONNX Quantized")

onnx_model_quant = onnx.load(model_quant)
ort_sess_quant = ort.InferenceSession(model_quant, providers=EP_list)

start = time.time_ns()
outputs = ort_sess_quant.run(
    None,
    {
        "input_latents": input_latents.numpy(),
        "context": context.numpy(),
        "time_embedding": time_embedding.numpy(),
    },
)
end = time.time_ns()
print(f"ONNX Quant Diffuser : {(end-start)*1e-9}s")
