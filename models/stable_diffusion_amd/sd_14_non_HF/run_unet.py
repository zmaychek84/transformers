import time

import onnxruntime
import torch

onnx_model_path = "diffusion_quant_p1.onnx"

config_file_path = "vaip_config_merged.json"

input_latents = torch.rand((1, 4, 64, 64))
context = torch.rand((1, 77, 768))
time_embedding = torch.rand((1, 320))

session = onnxruntime.InferenceSession(
    onnx_model_path,
    providers=["VitisAIExecutionProvider"],
    provider_options=[{"config_file": config_file_path}],
)

start = time.time_ns()
session.run(
    None,
    {
        "input_latents": input_latents.numpy(),
        "context": context.numpy(),
        "time_embedding": time_embedding.numpy(),
    },
)
end = time.time_ns()
print(f"ONNX Quant Diffuser (Multi-Dpu) : {(end-start)*1e-9}s")
