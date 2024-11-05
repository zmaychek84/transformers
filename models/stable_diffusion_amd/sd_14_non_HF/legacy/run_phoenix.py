## pipeline.py modified to make it ready to run on Phoenix

## pipeline.py modified to make it ready to run on Phoenix

import os
import sys

sys.path.append(os.getenv("PYTORCH_AIE_PATH") + "/ext/stable_diffusion/")

import time

import numpy as np
import qlinear
import qlinear_experimental
from torchmetrics import StructuralSimilarityIndexMeasure
from tqdm import tqdm
from utils import Utils


def calc_ssim(im1, im2):
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    res = ssim(im1, im2)
    return res


def run(
    prompts,
    strength=0.9,
    cfg_scale=7.5,
    height=512,
    width=512,
    sampler="k_euler",
    n_inference_steps=50,
    target="cpu",
    quant_mode=1,
    ptdq={"Clip": False, "Diffuser": False, "Decoder": False},
    save_onnx=False,
):
    onnx_clip_saved = False
    onnx_diff_saved = False
    onnx_deco_saved = False
    with torch.no_grad():
        if height % 8 or width % 8:
            raise ValueError("height and width must be a multiple of 8")
        uncond_prompts = [""] * len(prompts)
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device("cpu")  # Dynamic Quantization only works on CPU

        generator = torch.Generator(device=device)
        generator.manual_seed(123)

        tokenizer = Tokenizer()
        clip = model_loader.load_clip(device)
        layer_counts = Utils.count_layers(clip)
        print(f"Clip layer counts: {layer_counts}")

        # Quantizing clip causes weird multiple images within single image
        if ptdq["Clip"]:
            start = time.time_ns()
            clip = torch.ao.quantization.quantize_dynamic(
                clip, {torch.nn.Linear}, dtype=torch.qint8
            )
            end = time.time_ns()
            print(f"[PROFILE] Clip quantization time: {(end-start)*1e-9}s")

            if target == "cpu":
                # FP32*int8 inp range: [-2538.1, 7968.2] [-128, 127]
                # FP32*int8 out range: [-210121, 220558]

                # int32=int8*int8 out range: [-116256 122036]
                #     req_out_scale = 122036/32768 = 3.72 => 4 to fit in int16
                if False:
                    node_args = ()
                    node_kwargs = {
                        "quant_mode": quant_mode,
                        "requantize_in_scale": 1,
                        "requantize_out_scale": 4,  # for rsby2 use 2
                    }

                    Utils.replace_node(
                        clip,
                        torch.ao.nn.quantized.dynamic.modules.linear.Linear,
                        qlinear_experimental.QLinearExperimentalCPU,
                        node_args,
                        node_kwargs,
                    )
                    print(
                        f"[PROFILE] Number of Linear nodes replaced: {Utils.node_count}"
                    )
            elif target == "aie":
                node_args = ()
                node_kwargs = {
                    "requantize_in_scale": 1,
                    "requantize_out_scale": 4,
                    "quant_mode": 1,
                }
                Utils.replace_node(
                    clip,
                    torch.ao.nn.quantized.dynamic.modules.linear.Linear,
                    qlinear.QLinear,
                    node_args,
                    node_kwargs,
                )
                print(f"[PROFILE] Number of Linear nodes replaced: {Utils.node_count}")
            else:
                pass
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
            if ptdq["Clip"] == False:
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
            else:
                print("ONNX save is only available for FP32 Models")

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
        layer_counts = Utils.count_layers(diffusion)
        print(f"Diffusion layer counts: {layer_counts}")

        if ptdq["Diffuser"]:
            start = time.time_ns()
            diffusion = torch.ao.quantization.quantize_dynamic(
                diffusion, {torch.nn.Linear}, dtype=torch.qint8
            )
            end = time.time_ns()
            print(f"[PROFILE] Diffusion quantization time: {(end-start)*1e-9}s")
            if target == "cpu":
                # FP32*int8 inp range: [-2028993, 1338020] [-128, 127]
                # FP32*int8 out range: [-7712, 4994]

                # int32=int8*int8 out range: [-568704, 503773]
                #     req_out_scale = 568704/32768 = 17.3 => 32 to fit in int16
                if False:
                    node_args = ()
                    node_kwargs = {
                        "quant_mode": quant_mode,
                        "requantize_in_scale": 1,
                        "requantize_out_scale": 32,  # for rsby2 use 16
                    }

                    Utils.replace_node(
                        clip,
                        torch.ao.nn.quantized.dynamic.modules.linear.Linear,
                        qlinear_experimental.QLinearExperimentalCPU,
                        node_args,
                        node_kwargs,
                    )
                    print(
                        f"[PROFILE] Number of Linear nodes replaced: {Utils.node_count}"
                    )
            elif target == "aie":
                node_args = ()
                node_kwargs = {
                    "requantize_in_scale": 1,
                    "requantize_out_scale": 32,
                    "quant_mode": 1,
                }
                Utils.replace_node(
                    clip,
                    torch.ao.nn.quantized.dynamic.modules.linear.Linear,
                    qlinear.QLinear,
                    node_args,
                    node_kwargs,
                )

                print(f"[PROFILE] Number of Linear nodes replaced: {Utils.node_count}")
            else:
                pass

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

            """### Torch profile to get ops
            with torch.profiler.profile( activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True ) as prof:
                with torch.profiler.record_function("model_inference"):
                    output = diffusion(input_latents, context, time_embedding)
            with open("diffuser_ops_profile_torch.log", 'w') as f:
                print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=-1), file=f)
            """
            if (save_onnx == True) and (onnx_diff_saved == False):
                if ptdq["Diffuser"] == False:
                    save_dir = "diffusion_onnx"
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    torch.onnx.export(
                        diffusion,  # model being run
                        (
                            input_latents,
                            context,
                            time_embedding,
                        ),  # model input (or a tuple for multiple inputs)
                        ".\\"
                        + save_dir
                        + "\diffusion.onnx",  # where to save the model (can be a file or file-like object)
                        export_params=True,  # store the trained parameter weights inside the model file
                        opset_version=17,  # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names=[
                            "input_latents",
                            "context",
                            "time_embedding",
                        ],  # the model's input names
                        output_names=["output"],  # the model's output names
                    )
                    onnx_diff_saved = True
                    print("Diffuser ONNX model saved")
                else:
                    print("ONNX save is only available for FP32 Models")

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

        if ptdq["Decoder"]:
            start = time.time_ns()
            decoder = torch.ao.quantization.quantize_dynamic(
                decoder, {torch.nn.Linear}, dtype=torch.qint8
            )
            end = time.time_ns()
            print(f"[PROFILE] Decoder quantization time: {(end-start)*1e-9}s")

            if target == "cpu":
                # FP32 inp range: [-4158.1, 4156.4] [-128, 127]
                # FP32 out range: [-6179, 6105]

                # int32 out range: [-102741, 86373]
                #     req_out_scale = 102741/32768 = 3.14 => 4 to fit in int16
                if False:
                    node_args = ()
                    node_kwargs = {
                        "quant_mode": quant_mode,
                        "requantize_in_scale": 1,
                        "requantize_out_scale": 4,  # for rsby2 use 2
                    }

                    Utils.replace_node(
                        clip,
                        torch.ao.nn.quantized.dynamic.modules.linear.Linear,
                        qlinear_experimental.QLinearExperimentalCPU,
                        node_args,
                        node_kwargs,
                    )
                    print(
                        f"[PROFILE] Number of Linear nodes replaced: {Utils.node_count}"
                    )
            elif target == "aie":
                node_args = ()
                node_kwargs = {
                    "requantize_in_scale": 1,
                    "requantize_out_scale": 4,
                    "quant_mode": 1,
                }
                Utils.replace_node(
                    clip,
                    torch.ao.nn.quantized.dynamic.modules.linear.Linear,
                    qlinear.QLinear,
                    node_args,
                    node_kwargs,
                )
                print(f"[PROFILE] Number of Linear nodes replaced: {Utils.node_count}")
            else:
                pass

        start = time.time_ns()
        images = decoder(latents)
        end = time.time_ns()
        print(f"[PROFILE] Decode time (image generation): {(end-start)*1e-9}s")
        if (save_onnx == True) and (onnx_deco_saved == False):
            if ptdq["Decoder"] == False:
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
            else:
                print("ONNX save is only available for FP32 Models")
        del decoder

        images = util.rescale(images, (-1, 1), (0, 255), clamp=True)
        images = util.move_channel(images, to="last")
        images = images.to("cpu", torch.uint8).numpy()

        return [Image.fromarray(image) for image in images]


if __name__ == "__main__":
    prompts = ["albert bierstadt style painting of los angeles"]
    # prompts = ["a man floating in space, large scale, realistic proportions, highly detailed, smooth sharp focus, ray tracing, digital painting, art illustration"]

    ptdq_list = [  # {'Clip': True, 'Diffuser':False, 'Decoder':False},
        # {'Clip': False, 'Diffuser':True, 'Decoder':False},
        # {'Clip': False, 'Diffuser':False, 'Decoder':True},
        # {'Clip': True, 'Diffuser':True, 'Decoder':True}
        {"Clip": False, "Diffuser": False, "Decoder": False}
    ]

    # load fp32 image for ssim comparison
    fp32_image = Image.open("..\..\ext\stable_diffusion\output_euler_50_fp32_A.jpg")
    fp32_im_torch = torch.tensor(np.array(fp32_image)).unsqueeze(0) / 255.0
    fp32_im_torch = fp32_im_torch.reshape(
        (1, fp32_im_torch.shape[3], fp32_im_torch.shape[1], fp32_im_torch.shape[2])
    )
    ssim = calc_ssim(fp32_im_torch, fp32_im_torch)
    print(f"SSIM: FP32: {ssim}")

    # cpu forward call options
    quant_mode = 1
    target = "cpu"
    timesteps = 1

    if quant_mode == 1:
        qstr = "int16_int8xint8"
    elif quant_mode == 2:  # done
        qstr = "fp32_fp32xint8"
    elif quant_mode == 3:  # done
        qstr = "fp32_fp16xint8"
    elif quant_mode == 4:  # done
        qstr = "fp32_bfloat16xint8"
    elif quant_mode == 5:  # done
        qstr = "bfloat16_bfloat16xint8"
    elif quant_mode == 6:  #
        qstr = "int32_int8xint8"
    else:
        raise SystemError

    for ptdq in ptdq_list:
        images = run(
            prompts,
            sampler="k_euler",
            n_inference_steps=timesteps,
            target=target,
            ptdq=ptdq,
            quant_mode=quant_mode,
            save_onnx=True,
        )
        import time

        start = time.time_ns()
        im_torch = torch.tensor(np.array(images[0])).unsqueeze(0) / 255.0
        im_torch = im_torch.reshape(
            (1, im_torch.shape[3], im_torch.shape[1], im_torch.shape[2])
        )
        ssim = calc_ssim(fp32_im_torch, im_torch)
        print(f"SSIM: {ssim}")
        images[0].save(
            "sd_image_%s_euler_%d_%s_clip%s_diffuser%s_decoder%s_ssim_%0.4f.jpg"
            % (
                target,
                timesteps,
                qstr,
                str(ptdq["Clip"]),
                str(ptdq["Diffuser"]),
                str(ptdq["Decoder"]),
                ssim,
            )
        )
        end = time.time_ns()
        print(f"Time to generate image with target={target} is {(end-start)*1e-9}s")

# 3601s for diffuser cpu int32 -> int16
