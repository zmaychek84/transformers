import argparse
import os
import sys

# setting path
sys.path.append("../diffusers_lib_amd")
from clip_score import calc_clip_score

parser = argparse.ArgumentParser()
parser.add_argument(
    "--vitisai",
    help="Running on quantized model with vitisai ep or not",
    action="store_true",
)
parser.add_argument(
    "--quant", help="Running on quantized model or not", action="store_true"
)
parser.add_argument(
    "--use_unet_amd_onnx", help="Running on amd onnx models", action="store_true"
)
parser.add_argument(
    "--use_msft_onnx", help="Running on MSFT onnx models", action="store_true"
)
parser.add_argument(
    "--use_fp16_clip_onnx_dml",
    help="Running on FP16 CLIP onnx models with DmlEP",
    action="store_true",
)
parser.add_argument(
    "--use_fp16_decoder_onnx_dml",
    help="Running on FP16 VAE Decoder onnx models with DmlEP",
    action="store_true",
)
parser.add_argument(
    "--use_fp32_decoder_onnx",
    help="Running on FP32 VAE Decoder onnx models",
    action="store_true",
)
parser.add_argument(
    "--export_to_onnx", help="Export pt models into onnx format", action="store_true"
)
parser.add_argument(
    "--dump_tensors",
    help="Dump intermediate tensors for quantization purpose",
    action="store_true",
)
parser.add_argument(
    "--num_inference_steps", help="Num of diffusion steps", type=int, default=5
)
parser.add_argument(
    "--quant_type",
    help="Quantization mode - w8a8",
    type=str,
    default="a8w8",
    choices=["a8w8", "a16w8"],
)
parser.add_argument(
    "--dtype",
    help="Float model data type: float32",
    type=str,
    default="fp32",
    choices=["fp32", "bf16"],
)
parser.add_argument(
    "--ipu_platform",
    help="IPU platform the model is running",
    type=str,
    default="stx",
    choices=["phx", "stx"],
)


def run(
    pipe,
    dtype="fp32",
    use_unet_amd_onnx=False,
    use_msft_onnx=False,
    quant=False,
    vitisai=False,
    dump_tensors=False,
    num_inference_steps=5,
    export_to_onnx=False,
    quant_type="a8w8",
    sd_version="sd_14",
    prompt="",
    height=512,
    width=512,
    ipu_platform="stx",
    guidance_scale=7.5,
    use_fp16_clip_onnx_dml=False,
    use_fp16_decoder_onnx_dml=False,
    use_fp32_decoder_onnx=False,
):

    if ipu_platform == "stx":
        config_file_path = "../vaip_config_stx.json"
    else:
        config_file_path = "../vaip_config_phx.json"

    images = pipe(
        prompt=[prompt],
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        use_unet_amd_onnx=use_unet_amd_onnx,
        use_msft_onnx=use_msft_onnx,
        quant=quant,
        quant_type=quant_type,
        vitisai=vitisai,
        guidance_scale=guidance_scale,
        dump_tensors=dump_tensors,
        export_to_onnx=export_to_onnx,
        config_file_path=config_file_path,
        sd_version=sd_version,
        use_fp16_clip_onnx_dml=use_fp16_clip_onnx_dml,
        use_fp16_decoder_onnx_dml=use_fp16_decoder_onnx_dml,
        use_fp32_decoder_onnx=use_fp32_decoder_onnx,
    ).images

    for i in range(len(images)):
        image = images[i]
        name_prefix = sd_version + "_" + dtype
        if use_unet_amd_onnx:
            name_prefix += "_unet_amd_onnx"
        elif use_msft_onnx:
            name_prefix += "_msft_onnx"
        else:
            name_prefix += "_pytorch"
        if use_fp16_clip_onnx_dml:
            name_prefix += "_fp16Clip_onnx_dml"
        if use_fp16_decoder_onnx_dml:
            name_prefix += "_fp16Decoder_onnx_dml"
        if quant:
            name_prefix += "_quant_" + quant_type
        if vitisai:
            name_prefix += "_vai"
        else:
            name_prefix += "_cpu"
        f_name = f"{name_prefix}_p{i}_{height}_{num_inference_steps}.png"
        if os.path.isfile(f_name):
            os.remove(f_name)
        image.save(f_name)
        calc_clip_score(image, [prompt])


def check_config(args):
    if (
        not args.use_unet_amd_onnx
        and args.vitisai
        or not args.use_unet_amd_onnx
        and args.quant
        or not args.use_unet_amd_onnx
        and args.dump_tensors
        or args.use_unet_amd_onnx
        and args.export_to_onnx
        or args.vitisai
        and not args.quant
        or args.use_unet_amd_onnx
        and args.use_msft_onnx
        or args.export_to_onnx
        and args.use_msft_onnx
    ):
        print(f" *** MODE NOT SUPPORTED *** : check help and readme")
        raise SystemExit
