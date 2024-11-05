## pipeline.py modified to make it ready to run on Phoenix

import os
import sys

sys.path.append(os.getenv("PYTORCH_AIE_PATH") + "/ext/stable_diffusion/")

import time

import numpy as np
import qlinear
import qlinear_experimental
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from PIL import Image
from stable_diffusion_pytorch import model_loader, util
from stable_diffusion_pytorch.clip import CLIP
from stable_diffusion_pytorch.decoder import Decoder
from stable_diffusion_pytorch.encoder import Encoder
from stable_diffusion_pytorch.samplers import (
    KEulerAncestralSampler,
    KEulerSampler,
    KLMSSampler,
)
from stable_diffusion_pytorch.tokenizer import Tokenizer
from tqdm import tqdm
from utils import Utils


def run(
    prompts,
    strength=0.9,
    cfg_scale=7.5,
    height=512,
    width=512,
    sampler="k_euler",
    n_inference_steps=50,
    save_onnx=False,
):
    onnx_clip_saved = False
    onnx_diff_saved = False
    onnx_deco_saved = False
    with torch.no_grad():
        if height % 8 or width % 8:
            raise ValueError("height and width must be a multiple of 8")
        uncond_prompts = [""] * len(prompts)
        device = torch.device("cpu")  # Dynamic Quantization only works on CPU

        generator = torch.Generator(device=device)
        generator.manual_seed(123)

        tokenizer = Tokenizer()
        clip = model_loader.load_clip(device)
        layer_counts = Utils.count_layers(clip)
        print(f"Clip layer counts: {layer_counts}")

        # print(clip)
        # input("Enter any key")
        clip.to(device)

        cond_tokens = tokenizer.encode_batch(prompts)
        cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)

        start = time.time_ns()
        cond_context = clip(cond_tokens)
        end = time.time_ns()
        print(f"[PROFILE] Clip time: {(end-start)*1e-9}s")

        uncond_tokens = tokenizer.encode_batch(uncond_prompts)
        uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
        uncond_context = clip(uncond_tokens)
        context = torch.cat([cond_context, uncond_context])

        if (save_onnx == True) and (onnx_clip_saved == False):
            save_dir = "clip_onnx"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.onnx.export(
                clip,  # model being run
                (cond_tokens),  # model input (or a tuple for multiple inputs)
                ".\\"
                + save_dir
                + "\clip.onnx",  # where to save the model (can be a file or file-like object)
                export_params=True,  # store the trained parameter weights inside the model file
                opset_version=17,  # the ONNX version to export the model to
                do_constant_folding=True,  # whether to execute constant folding for optimization
                input_names=["cond_tokens"],  # the model's input names
                output_names=["cond_context"],  # the model's output names
            )
            onnx_clip_saved = True
            print("Clip ONNX model saved")

        del tokenizer, clip

        if sampler == "k_lms":
            sampler = KLMSSampler(n_inference_steps=n_inference_steps)
        elif sampler == "k_euler":
            sampler = KEulerSampler(n_inference_steps=n_inference_steps)
        elif sampler == "k_euler_ancestral":
            sampler = KEulerAncestralSampler(
                n_inference_steps=n_inference_steps, generator=generator
            )
        else:
            raise ValueError(
                "Unknown sampler value %s. "
                "Accepted values are {k_lms, k_euler, k_euler_ancestral}" % sampler
            )

        noise_shape = (len(prompts), 4, height // 8, width // 8)

        latents = torch.randn(noise_shape, generator=generator, device=device)
        latents *= sampler.initial_scale

        diffusion = model_loader.load_diffusion(device)
        # torch.save(diffusion, "diffusion.pt")
        # diffusion = torch.load("diffusion.pt")
        # diffusion.eval()

        # layer_counts = Utils.count_layers(diffusion)
        # print(f"Diffusion layer counts: {layer_counts}")

        timesteps = tqdm(sampler.timesteps)
        diff_time_arr = []
        print(f"[PROFILE] number of diffusion timesteps: {len(timesteps)}")

        for i, timestep in enumerate(timesteps):
            time_embedding = util.get_time_embedding(timestep).to(device)

            input_latents = latents * sampler.get_input_scale()
            input_latents = input_latents.repeat(2, 1, 1, 1)

            """
            print(f"input_latents: {input_latents.shape}")
            print(f"context: {context.shape}")
            print(f"time_embedding: {time_embedding.shape}")
            input_latents: torch.Size([2, 4, 64, 64])
            context: torch.Size([2, 77, 768])
            time_embedding: torch.Size([1, 320])
            """

            start = time.time_ns()
            output = diffusion(input_latents, context, time_embedding)
            end = time.time_ns()
            diff_time_arr.append(end - start)

            output_cond, output_uncond = output.chunk(2)
            output = cfg_scale * (output_cond - output_uncond) + output_uncond

            latents = sampler.step(latents, output)

        print(
            f"[PROFILE] Diffusion for timesteps={timesteps} time: {sum(diff_time_arr)*1e-9}s"
        )

        del diffusion

        decoder = model_loader.load_decoder(device)
        # print(decoder)
        # input("Enter any key")
        layer_counts = Utils.count_layers(decoder)
        print(f"Decoder layer counts: {layer_counts}")

        start = time.time_ns()
        images = decoder(latents)
        end = time.time_ns()
        print(f"[PROFILE] Decode time (image generation): {(end-start)*1e-9}s")
        if (save_onnx == True) and (onnx_deco_saved == False):
            save_dir = "decoder_onnx"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.onnx.export(
                decoder,  # model being run
                (latents),  # model input (or a tuple for multiple inputs)
                ".\\"
                + save_dir
                + "\decoder.onnx",  # where to save the model (can be a file or file-like object)
                export_params=True,  # store the trained parameter weights inside the model file
                opset_version=17,  # the ONNX version to export the model to
                do_constant_folding=True,  # whether to execute constant folding for optimization
                input_names=["latents"],  # the model's input names
                output_names=["output"],  # the model's output names
            )
            onnx_deco_saved = True
            print("Decoder ONNX model saved")
        del decoder

        images = util.rescale(images, (-1, 1), (0, 255), clamp=True)
        images = util.move_channel(images, to="last")
        images = images.to("cpu", torch.uint8).numpy()

        return [Image.fromarray(image) for image in images]


if __name__ == "__main__":
    timesteps = 20
    prompts = [
        "Photo of a ultra realistic sailing ship, dramatic light, pale sunrise, cinematic lighting, battered, low angle, trending on artstation, 4k, hyper realistic, focused, extreme details, unreal engine 5, cinematic, masterpiece, art by studio ghibli, intricate artwork by john william turner"
    ]
    images = run(prompts, sampler="k_euler", n_inference_steps=timesteps)
    images[0].save("out.png")
