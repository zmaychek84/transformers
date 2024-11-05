import time

import numpy as np
import onnxruntime
import torch


def run():
    diffusion_quant_p1_path = "diffusion_quant_p1.onnx"
    diffusion_quant_p2_path = "diffusion_quant_p2.onnx"

    config_file_path = "vaip_config_merged.json"

    diffusion_quant_p1_vitisai_session = onnxruntime.InferenceSession(
        diffusion_quant_p1_path,
        providers=["VitisAIExecutionProvider"],
        provider_options=[{"config_file": config_file_path}],
    )
    diffusion_quant_p2_vitisai_session = onnxruntime.InferenceSession(
        diffusion_quant_p2_path,
        providers=["VitisAIExecutionProvider"],
        provider_options=[{"config_file": config_file_path}],
    )
    diffusion_quant_p1_cpu_session = onnxruntime.InferenceSession(
        diffusion_quant_p1_path,
        providers=["CPUExecutionProvider"],
    )
    diffusion_quant_p2_cpu_session = onnxruntime.InferenceSession(
        diffusion_quant_p2_path,
        providers=["CPUExecutionProvider"],
    )
    diffusion_p1_cpu_session = onnxruntime.InferenceSession(
        "diffusion_p1.onnx",
        providers=["CPUExecutionProvider"],
    )
    diffusion_p2_cpu_session = onnxruntime.InferenceSession(
        "diffusion_p2.onnx",
        providers=["CPUExecutionProvider"],
    )

    input_latents = torch.rand((2, 4, 64, 64))
    context = torch.rand((2, 77, 768))
    time_embedding = torch.rand((1, 320))

    start = time.time_ns()
    (
        d1_output_quant_vai,
        skip_connections_1,
        skip_connections_2,
        skip_connections_3,
        skip_connections_4,
        skip_connections_5,
        skip_connections_6,
        skip_connections_7,
        skip_connections_8,
        skip_connections_9,
        skip_connections_10,
        skip_connections_11,
        skip_connections_12,
        intermediate_time,
    ) = diffusion_quant_p1_vitisai_session.run(
        None,
        {
            "input_latents": input_latents.detach().numpy(),
            "context": context.detach().numpy(),
            "time_embedding": time_embedding.detach().numpy(),
        },
    )
    end = time.time_ns()
    print(f"[PROFILE] ONNX Quant Diffuser P1 (VitisAI EP) time: {(end-start)*1e-9}s")
    start = time.time_ns()
    output_quant_vai = diffusion_quant_p2_vitisai_session.run(
        None,
        {
            "input_latents": d1_output_quant_vai,
            "context": context.detach().numpy(),
            "time_embedding": intermediate_time,
            "skip_connections_1": skip_connections_1,
            "skip_connections_2": skip_connections_2,
            "skip_connections_3": skip_connections_3,
            "skip_connections_4": skip_connections_4,
            "skip_connections_5": skip_connections_5,
            "skip_connections_6": skip_connections_6,
            "skip_connections_7": skip_connections_7,
            "skip_connections_8": skip_connections_8,
            "skip_connections_9": skip_connections_9,
            "skip_connections_10": skip_connections_10,
            "skip_connections_11": skip_connections_11,
            "skip_connections_12": skip_connections_12,
        },
    )
    end = time.time_ns()
    print(f"[PROFILE] ONNX Quant Diffuser P2 (VitisAI EP) time: {(end-start)*1e-9}s")

    start = time.time_ns()
    (
        d1_output_quant_cpu,
        skip_connections_1,
        skip_connections_2,
        skip_connections_3,
        skip_connections_4,
        skip_connections_5,
        skip_connections_6,
        skip_connections_7,
        skip_connections_8,
        skip_connections_9,
        skip_connections_10,
        skip_connections_11,
        skip_connections_12,
        intermediate_time,
    ) = diffusion_quant_p1_cpu_session.run(
        None,
        {
            "input_latents": input_latents.detach().numpy(),
            "context": context.detach().numpy(),
            "time_embedding": time_embedding.detach().numpy(),
        },
    )
    output_quant_cpu = diffusion_quant_p2_cpu_session.run(
        None,
        {
            "input_latents": d1_output_quant_cpu,
            "context": context.detach().numpy(),
            "time_embedding": intermediate_time,
            "skip_connections_1": skip_connections_1,
            "skip_connections_2": skip_connections_2,
            "skip_connections_3": skip_connections_3,
            "skip_connections_4": skip_connections_4,
            "skip_connections_5": skip_connections_5,
            "skip_connections_6": skip_connections_6,
            "skip_connections_7": skip_connections_7,
            "skip_connections_8": skip_connections_8,
            "skip_connections_9": skip_connections_9,
            "skip_connections_10": skip_connections_10,
            "skip_connections_11": skip_connections_11,
            "skip_connections_12": skip_connections_12,
        },
    )

    end = time.time_ns()
    print(f"[PROFILE] ONNX Quant Diffuser (CPU EP) time: {(end-start)*1e-9}s")

    start = time.time_ns()
    (
        d1_output_default_cpu,
        skip_connections_1,
        skip_connections_2,
        skip_connections_3,
        skip_connections_4,
        skip_connections_5,
        skip_connections_6,
        skip_connections_7,
        skip_connections_8,
        skip_connections_9,
        skip_connections_10,
        skip_connections_11,
        skip_connections_12,
        intermediate_time,
    ) = diffusion_p1_cpu_session.run(
        None,
        {
            "input_latents": input_latents.detach().numpy(),
            "context": context.detach().numpy(),
            "time_embedding": time_embedding.detach().numpy(),
        },
    )
    output_default_cpu = diffusion_p2_cpu_session.run(
        None,
        {
            "input_latents": d1_output_default_cpu,
            "context": context.detach().numpy(),
            "time_embedding": intermediate_time,
            "skip_connections_1": skip_connections_1,
            "skip_connections_2": skip_connections_2,
            "skip_connections_3": skip_connections_3,
            "skip_connections_4": skip_connections_4,
            "skip_connections_5": skip_connections_5,
            "skip_connections_6": skip_connections_6,
            "skip_connections_7": skip_connections_7,
            "skip_connections_8": skip_connections_8,
            "skip_connections_9": skip_connections_9,
            "skip_connections_10": skip_connections_10,
            "skip_connections_11": skip_connections_11,
            "skip_connections_12": skip_connections_12,
        },
    )

    end = time.time_ns()
    print(f"[PROFILE] ONNX Diffuser (CPU EP) time: {(end-start)*1e-9}s")
    print(
        f"L2 Norm difference between output_quant_vai and output_quant_cpu (VAI functional-wise correct?){np.linalg.norm(output_quant_cpu[0]-output_quant_vai[0])}"
    )
    # print (output_quant_vai[0])
    # print(output_quant_cpu[0])
    # print(output_quant_vai[0]==output_quant_cpu[0])
    # print(d1_output_quant_vai==d1_output_quant_cpu)
    print(
        f"L2 Norm difference between output_quant_cpu and output_default_cpu (Quantization correct?) {np.linalg.norm(output_quant_cpu[0]-output_default_cpu[0])}"
    )
    # print(output_quant_cpu[0]==output_default_cpu[0])
    # print(d1_output_quant_cpu==d1_output_default_cpu)


if __name__ == "__main__":

    run()
