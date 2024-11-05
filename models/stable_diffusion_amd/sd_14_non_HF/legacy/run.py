import pdb
import time

import torch

# pdb.set_trace()
from diffusers import DiffusionPipeline
from utils import Utils


def run_14():
    pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

    if True:
        print(pipeline.unet)
        print("Saving unet ...")
        # pipeline.unet.save_pretrained("unet_pytorch")
        # pipeline.save_pretrained("model_pytorch")

        input("Enter a key")

    """
    optimum-cli export onnx -m unet_pytorch unet_onnx --framework pt --no-post-process --opset 20

    """
    start = time.time()
    image = pipeline("Albert Bierstadt style image of Bison in Yosemite").images[0]
    end = time.time()
    print("Time for image generate: ", end - start)
    # Time for image generate:  598.8939747810364
    print(image)
    image.save("sd_image.png")


def run_15():
    from diffusers import StableDiffusionPipeline

    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipe = pipe.to("cpu")

    prompt = "a photo of an astronaut riding a horse on mars"

    print(pipe.components.keys())

    for key in pipe.components.keys():
        print(f"{key} ********************")
        try:
            layer_counts = Utils.count_layers(pipe.components[key])
            print(f"{layer_counts}")
            input(pipe.components[key])
        except:
            pass

    image = pipe(prompt, num_inference_steps=10).images[0]
    image.save("astronaut_rides_horse.png")
    # SD1.5 FP32 - 6.7s/iter


if __name__ == "__main__":
    run_15()
