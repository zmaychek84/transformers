import sys

import torch

# setting path
sys.path.append("../diffusers_lib_amd")
sys.path.append("../utils")
import model_exec
from pipeline_stable_diffusion_amd import StableDiffusionPipeline

if __name__ == "__main__":
    args = model_exec.parser.parse_args()
    model_exec.check_config(args)
    prompt = "Photo of a ultra realistic sailing ship, dramatic light, pale sunrise, cinematic lighting, battered, low angle, trending on artstation, 4k, hyper realistic, focused, extreme details, unreal engine 5, cinematic, masterpiece, art by studio ghibli, intricate artwork by john william turner"
    height = 512
    width = 512
    sd_version = "sd_14"
    quant_type = args.quant_type
    dtype = args.dtype
    if dtype == "fp32":
        torch_dtype = torch.float32
    elif dtype == "bf16":
        torch_dtype = torch.bfloat16

    model_id = "CompVis/stable-diffusion-v1-4"

    # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    pipe = pipe.to("cpu")

    model_exec.run(
        pipe,
        dtype=dtype,
        use_unet_amd_onnx=args.use_unet_amd_onnx,
        quant=args.quant,
        vitisai=args.vitisai,
        dump_tensors=args.dump_tensors,
        num_inference_steps=args.num_inference_steps,
        export_to_onnx=args.export_to_onnx,
        quant_type=quant_type,
        sd_version=sd_version,
        prompt=prompt,
        height=height,
        width=width,
        ipu_platform=args.ipu_platform,
    )
