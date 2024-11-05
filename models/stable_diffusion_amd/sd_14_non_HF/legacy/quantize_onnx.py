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
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)


class UnetP1DataReader(CalibrationDataReader):
    def __init__(self, model_path: str):
        self.enum_data = []
        self.data_size = 2

        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        for i in range(self.data_size):
            self.enum_data.append(
                {
                    "input_latents": torch.rand((2, 4, 64, 64)).numpy(),
                    "context": torch.rand((2, 77, 768)).numpy(),
                    "time_embedding": torch.rand((1, 320)).numpy(),
                }
            )
        self.enum_data = iter(self.enum_data)

    def get_next(self):
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None


diffusion_p1_path = "diffusion_onnx\diffusion_p1.onnx"
diffusion_p2_path = "diffusion_onnx\diffusion_p2.onnx"

EP_list = ["CPUExecutionProvider"]
p1_ort_sess = ort.InferenceSession(diffusion_p1_path, providers=EP_list)
p2_ort_sess = ort.InferenceSession(diffusion_p2_path, providers=EP_list)


input_latents = torch.rand((2, 4, 64, 64))
context = torch.rand((2, 77, 768))
time_embedding = torch.rand((1, 320))

start = time.time_ns()
out = p1_ort_sess.run(
    None,
    {
        "input_latents": input_latents.numpy(),
        "context": context.numpy(),
        "time_embedding": time_embedding.numpy(),
    },
)
d1_output = out[0]
skip_connections = out[1:-1]
intermediate_time = out[-1]
print(type(skip_connections[0]))

end = time.time_ns()
print(f"ONNX FP32 Diffuser P1: {(end-start)*1e-9}s")
start = time.time_ns()
out = p2_ort_sess.run(
    None,
    {
        "input_latents": d1_output,
        "context": context.numpy(),
        "time_embedding": intermediate_time,
        "skip_connections_1": skip_connections[0],
        "skip_connections_2": skip_connections[1],
        "skip_connections_3": skip_connections[2],
        "skip_connections_4": skip_connections[3],
        "skip_connections_5": skip_connections[4],
        "skip_connections_6": skip_connections[5],
        "skip_connections_7": skip_connections[6],
        "skip_connections_8": skip_connections[7],
        "skip_connections_9": skip_connections[8],
        "skip_connections_10": skip_connections[9],
        "skip_connections_11": skip_connections[10],
        "skip_connections_12": skip_connections[11],
    },
)
end = time.time_ns()
print(f"ONNX FP32 Diffuser P2: {(end-start)*1e-9}s")


quant_model_p1_quant = "diffusion_onnx_quant\diffusion_p1_quant.onnx"
quant_model_p2_quant = "diffusion_onnx_quant\diffusion_p2_quant.onnx"

quant_model_p1_quant = "diffusion_onnx_quant\diffusion_p1_quant.onnx"
quant_model_p2_quant = "diffusion_onnx_quant\diffusion_p2_quant.onnx"


quantized_model_p1 = quantize_static(
    diffusion_p1_path,
    quant_model_p1_quant,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    quant_format=QuantFormat.QDQ,
    calibration_data_reader=UnetP1DataReader(diffusion_p1_path),
    op_types_to_quantize=["Conv", "Gemm", "MatMul"],
    use_external_data_format=True,
    extra_options={
        "ActivationSymmetric": True,
        "WeightSymmetric": True,
        "QuantizeBias": True,
    },
)
quantized_model_p2 = quantize_static(
    diffusion_p2_path,
    quant_model_p2_quant,
    weight_type=QuantType.QInt8,
    quant_format=QuantFormat.QDQ,
    use_external_data_format=True,
)
print(f"ONNX Quantized")

onnx_model_quant_p1 = onnx.load(quant_model_p1_quant)
onnx_model_quant_p2 = onnx.load(quant_model_p2_quant)
ort_sess_quant_p1 = ort.InferenceSession(onnx_model_quant_p1, providers=EP_list)
ort_sess_quant_p2 = ort.InferenceSession(onnx_model_quant_p2, providers=EP_list)

start = time.time_ns()
d1_output, skip_connections, intermediate_time = ort_sess_quant_p1.run(
    None,
    {
        "input_latents": input_latents.numpy(),
        "context": context.numpy(),
        "time_embedding": time_embedding.numpy(),
    },
)
end = time.time_ns()
print(f"ONNX Quant Diffuser P1 : {(end-start)*1e-9}s")

start = time.time_ns()
out = ort_sess_quant_p2.run(
    None,
    {
        "input_latents": d1_output.numpy(),
        "context": context.numpy(),
        "time_embedding": intermediate_time.numpy(),
        "skip_connections": skip_connections.numpy(),
    },
)
end = time.time_ns()
print(f"ONNX Quant Diffuser P2 : {(end-start)*1e-9}s")
