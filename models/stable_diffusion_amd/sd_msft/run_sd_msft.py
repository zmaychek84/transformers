import sys

import torch

# setting path
sys.path.append("../diffusers_lib_amd")
sys.path.append("../utils")
import model_exec
from diffusers import DPMSolverMultistepScheduler
from pipeline_stable_diffusion_amd import StableDiffusionPipeline

if __name__ == "__main__":
    args = model_exec.parser.parse_args()
    model_exec.check_config(args)
    prompt = "Photo of a ultra realistic sailing ship, dramatic light, pale sunrise, cinematic lighting, battered, low angle, trending on artstation, 4k, hyper realistic, focused, extreme details, unreal engine 5, cinematic, masterpiece, art by studio ghibli, intricate artwork by john william turner"
    height = 512
    width = 512
    sd_version = "sd_msft"

    model_id = "stabilityai/stable-diffusion-2-1"

    # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cpu")

    model_exec.run(
        pipe,
        use_msft_onnx=args.use_msft_onnx,
        vitisai=args.vitisai,
        num_inference_steps=args.num_inference_steps,
        sd_version=sd_version,
        prompt=prompt,
        height=height,
        width=width,
        ipu_platform=args.ipu_platform,
    )
