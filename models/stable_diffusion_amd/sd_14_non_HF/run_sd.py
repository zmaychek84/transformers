import argparse
import builtins
import json
import os
import sys
import time
from json import JSONEncoder

import numpy
import onnxruntime
import torch
from PIL import Image
from tqdm import tqdm

sys.path.append(os.getenv("PYTORCH_AIE_PATH") + "/ext/stable_diffusion/")
from stable_diffusion_pytorch import model_loader, util
from stable_diffusion_pytorch.samplers import (
    KEulerAncestralSampler,
    KEulerSampler,
    KLMSSampler,
)
from stable_diffusion_pytorch.tokenizer import Tokenizer


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


p1TensorsData = {}
p2TensorsData = {}
tensorCounter = 0

clip_path = "clip.onnx"
diffusion_quant_p1_path = "diffusion_quant_p1.onnx"
diffusion_quant_p2_path = "diffusion_quant_p2.onnx"
diffusion_p1_path = "diffusion_p1.onnx"
diffusion_p2_path = "diffusion_p2.onnx"
decoder_path = "decoder.onnx"
config_file_path = "../vaip_config_stx.json"


def model_init(target="vai", quant=True, n_inference_steps=5):
    device = torch.device("cpu")
    ## Clip
    # clip_session = onnxruntime.InferenceSession(
    #                 clip_path,
    #                 providers=["CPUExecutionProvider"],
    #             )
    clip = model_loader.load_clip(device)
    clip.to(device)
    tokenizer = Tokenizer()
    ## Diffuser
    sess_options = onnxruntime.SessionOptions()
    sess_options.enable_profiling = False
    if target == "vai":
        if quant:
            diffusion_p1_session = onnxruntime.InferenceSession(
                "diffusion_quant_p1.onnx",
                sess_options,
                providers=["VitisAIExecutionProvider"],
                provider_options=[{"config_file": config_file_path}],
            )
            diffusion_p2_session = onnxruntime.InferenceSession(
                "diffusion_quant_p2.onnx",
                sess_options,
                providers=["VitisAIExecutionProvider"],
                provider_options=[{"config_file": config_file_path}],
            )
        else:
            print(
                f" *** (VitisAI + Non-Quantized) MODE NOT SUPPORTED *** : check help and readme"
            )
            raise SystemExit
    else:
        if quant:
            diffusion_p1_session = onnxruntime.InferenceSession(
                "diffusion_quant_p1.onnx",
                sess_options,
                providers=["CPUExecutionProvider"],
            )
            diffusion_p2_session = onnxruntime.InferenceSession(
                "diffusion_quant_p2.onnx",
                sess_options,
                providers=["CPUExecutionProvider"],
            )
        else:
            diffusion_p1_session = onnxruntime.InferenceSession(
                diffusion_p1_path,
                sess_options,
                providers=["CPUExecutionProvider"],
            )
            diffusion_p2_session = onnxruntime.InferenceSession(
                diffusion_p2_path,
                sess_options,
                providers=["CPUExecutionProvider"],
            )
    ## Decoder
    decoder_session = onnxruntime.InferenceSession(
        decoder_path,
        providers=["CPUExecutionProvider"],
    )

    return tokenizer, clip, diffusion_p1_session, diffusion_p2_session, decoder_session


def run(
    prompts,
    n_inference_steps,
    tokenizer,
    clip,
    sampler,
    diffusion_p1_session,
    diffusion_p2_session,
    decoder_session,
    target="vai",
    quant=True,
    dump_tensors=False,
    cfg_scale=7.5,
    height=512,
    width=512,
):
    global tensorCounter
    device = torch.device("cpu")
    generator = torch.Generator(device=device)
    generator.manual_seed(123)

    ## === CLIP ===
    cond_tokens = tokenizer.encode_batch(prompts)
    cond_tokens = torch.tensor(cond_tokens, dtype=torch.long)
    uncond_prompts = ["ugly"] * len(prompts)
    uncond_tokens = tokenizer.encode_batch(uncond_prompts)
    uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long)

    start = time.time_ns()
    # cond_context = clip_session.run(None, {'cond_tokens':cond_tokens.numpy()})
    cond_context = clip(cond_tokens)
    end = time.time_ns()
    print(f"[PROFILE] Clip time: {(end-start)*1e-9}s")
    # uncond_context = clip_session.run(None, {'cond_tokens':uncond_tokens.numpy()})
    uncond_context = clip(uncond_tokens)
    context = torch.cat([cond_context, uncond_context])

    ## === DIFFUSION ===
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

    diff_time_arr = []
    timesteps = tqdm(sampler.timesteps)
    cond_context = cond_context.detach().numpy()
    uncond_context = uncond_context.detach().numpy()
    for i, timestep in enumerate(timesteps):
        time_embedding = util.get_time_embedding(timestep).to(device)

        input_latents = latents * sampler.get_input_scale()
        # input_latents = input_latents.repeat(2, 1, 1, 1)

        start = time.time_ns()

        input_latents = input_latents.detach().numpy()
        time_embedding = time_embedding.detach().numpy()
        if dump_tensors and target == "cpu" and not quant:
            p1TensorsData[f"{tensorCounter}_input_latents"] = input_latents
            p1TensorsData[f"{tensorCounter}_context"] = cond_context
            p1TensorsData[f"{tensorCounter}_time_embedding"] = time_embedding
        # Pass in Cond Context
        (
            d1_output,
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
        ) = diffusion_p1_session.run(
            None,
            {
                "input_latents": input_latents,
                "context": cond_context,
                "time_embedding": time_embedding,
            },
        )
        if dump_tensors and target == "cpu" and not quant:
            p2TensorsData[f"{tensorCounter}_input_latents"] = d1_output
            p2TensorsData[f"{tensorCounter}_context"] = cond_context
            p2TensorsData[f"{tensorCounter}_time_embedding"] = intermediate_time
            p2TensorsData[f"{tensorCounter}_skip_connections_1"] = skip_connections_1
            p2TensorsData[f"{tensorCounter}_skip_connections_2"] = skip_connections_2
            p2TensorsData[f"{tensorCounter}_skip_connections_3"] = skip_connections_3
            p2TensorsData[f"{tensorCounter}_skip_connections_4"] = skip_connections_4
            p2TensorsData[f"{tensorCounter}_skip_connections_5"] = skip_connections_5
            p2TensorsData[f"{tensorCounter}_skip_connections_6"] = skip_connections_6
            p2TensorsData[f"{tensorCounter}_skip_connections_7"] = skip_connections_7
            p2TensorsData[f"{tensorCounter}_skip_connections_8"] = skip_connections_8
            p2TensorsData[f"{tensorCounter}_skip_connections_9"] = skip_connections_9
            p2TensorsData[f"{tensorCounter}_skip_connections_10"] = skip_connections_10
            p2TensorsData[f"{tensorCounter}_skip_connections_11"] = skip_connections_11
            p2TensorsData[f"{tensorCounter}_skip_connections_12"] = skip_connections_12
            tensorCounter += 1
        output1 = diffusion_p2_session.run(
            None,
            {
                "input_latents": d1_output,
                "context": cond_context,
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
        # Pass in UnCond Context
        (
            d1_output,
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
        ) = diffusion_p1_session.run(
            None,
            {
                "input_latents": input_latents,
                "context": uncond_context,
                "time_embedding": time_embedding,
            },
        )
        output2 = diffusion_p2_session.run(
            None,
            {
                "input_latents": d1_output,
                "context": uncond_context,
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
        diff_time_arr.append(end - start)
        # output = torch.tensor(output[0])
        # output_cond, output_uncond = output.chunk(2)
        output_cond, output_uncond = torch.tensor(output1[0]), torch.tensor(output2[0])
        output = cfg_scale * (output_cond - output_uncond) + output_uncond

        latents = sampler.step(latents, output)

    print(
        f"[PROFILE] ONNX Diffuser total time: {sum(diff_time_arr)*1e-9}s; per iteration time: {sum(diff_time_arr)/n_inference_steps*1e-9}s"
    )
    # print(f"[PROFILE] ONNX Quant Diffuser (Multi-Dpu) : {sum(diff_time_arr)*1e-9}s")

    ## === DECODER ===
    start = time.time_ns()
    images = decoder_session.run(None, {"latents": latents.detach().numpy()})
    end = time.time_ns()
    print(f"[PROFILE] ONNX Decode time (image generation): {(end-start)*1e-9}s")

    images = torch.tensor(images[0])
    images = util.rescale(images, (-1, 1), (0, 255), clamp=True)
    images = util.move_channel(images, to="last")
    images = images.to("cpu", torch.uint8).numpy()

    return [Image.fromarray(image) for image in images]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target", help="cpu, vai", type=str, default="vai", choices=["cpu", "vai"]
    )
    parser.add_argument(
        "--quant", help="Running on quantized model or not", action="store_true"
    )
    parser.add_argument(
        "--dump_tensors",
        help="Dump intermediate tensors for quantization purpose",
        action="store_true",
    )
    parser.add_argument(
        "--timesteps", help="Num of diffusion steps", type=int, default=10
    )
    parser.add_argument(
        "--output_name", help="Output pic name", type=str, default="out.png"
    )
    parser.add_argument(
        "--quant_mode",
        help="Quantization mode - w8a8",
        type=str,
        default="w8a8",
        choices=["w8a8", "none"],
    )
    parser.add_argument(
        "--impl",
        help="Choose between different implementations for aie target",
        type=str,
        default="v0",
        choices=["v0", "v1"],
    )

    args = parser.parse_args()
    builtins.impl = args.impl
    builtins.quant_mode = args.quant_mode
    print(f"{args}")

    tokenizer, clip, diffusion_p1_session, diffusion_p2_session, decoder_session = (
        model_init(target=args.target, quant=args.quant, n_inference_steps=5)
    )
    prompts = [
        # 'a photo of an astronaut riding a horse on mars',
        "Photo of a ultra realistic sailing ship, dramatic light, pale sunrise, cinematic lighting, battered, low angle, trending on artstation, 4k, hyper realistic, focused, extreme details, unreal engine 5, cinematic, masterpiece, art by studio ghibli, intricate artwork by john william turner",
        #    'Mountain pine forest with old stone and build a lighthouse in the middle of the mountains, small river flowing through town, daytime, full moon in the sky, hyper realistic, ambient lighting, concept art, intricate, hyper detailed, smooth, dynamic volumetric lighting, octane, raytrace, cinematic, high quality, high resolution',
        #    'Renaissance-style portrait of an astronaut in space, detailed starry background, reflective helmet. digital painting, artstation, concept art, art by artgerm and donato giancola and Joseph Christian Leyendecker, Ross Tran, WLOP',
        #    'the street of amedieval fantasy town, at dawn, dark, 4k, highly detailed',
        #    'a highly detailed matte painting of a man on a hill watching a rocket launch in the distance by studio ghibli, makoto shinkai, by artgerm, by wlop, by greg rutkowski, volumetric lighting, octane render, 4 k resolution, trending on artstation, masterpiece | hyperrealism| highly detailed| insanely detailed| intricate| cinematic lighting| depth of field.',
        #    'overwhelmingly beautiful eagle framed with vector flowers, long shiny wavy flowing hair, polished, ultra detailed vector floral illustration mixed with hyper realism, muted pastel colors, vector floral details in background, muted colors, hyper detailed ultra intricate overwhelming realism in detailed complex scene with magical fantasy atmosphere, no signature, no watermark',
        #    'exquisitely intricately detailed illustration, of a small world with a lake and a rainbow, inside a closed glass jar.',
        #    'Cute small Fox sitting in a movie theater eating popcorn watching a movie ,unreal engine, cozy indoor lighting, artstation, detailed, digital painting,cinematic,character design by mark ryden and pixar and hayao miyazaki, unreal 5, daz, hyperrealistic, octane render',
        #    'cute toy owl made of suede, geometric accurate, relief on skin, plastic relief surface of body, intricate details, cinematic',
        #    'futuristic lighthouse, flash light, hyper realistic, epic composition, cinematic, landscape vista photography, landscape veduta photo & tdraw, detailed landscape painting rendered in enscape, miyazaki, 4k detailed post processing, unreal engineered'
    ]
    for i in range(len(prompts)):
        images = run(
            [prompts[i]],
            n_inference_steps=args.timesteps,
            tokenizer=tokenizer,
            clip=clip,
            sampler="k_euler",
            diffusion_p1_session=diffusion_p1_session,
            diffusion_p2_session=diffusion_p2_session,
            decoder_session=decoder_session,
            target=args.target,
            quant=args.quant,
            dump_tensors=args.dump_tensors,
        )
        quant_str = "quant" if args.quant else ""
        images[0].save(f"out_{args.target}_{quant_str}_{i}.png")

    ## Write tensors into json files
    if args.dump_tensors and args.target == "cpu" and not args.quant:
        print("serialize p1TensorsData into JSON and write into a file")
        with open("p1TensorsData.json", "w") as write_file:
            json.dump(p1TensorsData, write_file, cls=NumpyArrayEncoder)
        print("Done writing serialized p1TensorsData into file")
        print("serialize p2TensorsData into JSON and write into a file")
        with open("p2TensorsData.json", "w") as write_file:
            json.dump(p2TensorsData, write_file, cls=NumpyArrayEncoder)
        print("Done writing serialized p2TensorsData into file")
