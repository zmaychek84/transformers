import torch
from diffusers import StableDiffusionPipeline


def count_gops(model, kwargs, model_id=""):
    report_name = model_id + "_" + model.__class__.__name__ + ".csv"
    f = open(report_name, "w")
    f.write("event_id,op_name,op_count,flops,inp_shapes\n")
    if False:
        """Detailed aten::op list"""
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True
        ) as prof:
            with torch.profiler.record_function("model_inference"):
                image = model(**kwargs)  # .images[0]
        print(
            prof.key_averages(group_by_input_shape=True).table(
                sort_by="cpu_time_total", row_limit=-1
            )
        )
    else:
        # https://github.com/pytorch/pytorch/blob/main/torch/profiler/profiler.py#L508
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            with_flops=True,
        ) as prof:
            with torch.profiler.record_function("model_inference"):
                image = model(**kwargs)  # .images[0]
        profile_data = prof.key_averages(
            group_by_input_shape=True
        )  # .table(sort_by="cpu_time_total", row_limit=-1)

        # reference: # https://github.com/pytorch/pytorch/blob/main/torch/autograd/profiler_util.py#L1019
        op_counts = {}
        op_name_exp_list = [
            "aten::softmax",
            "aten::silu",
            "aten::group_norm",
            "aten::layer_norm",
            "aten::gelu",
        ]
        op_counts_exp = {}
        for event_id in range(len(profile_data)):
            # print(event_id, profile_data[event_id])
            event = profile_data[event_id]
            # print(type(event)) #class 'torch.autograd.profiler_util.FunctionEventAvg'
            # reference: https://github.com/pytorch/pytorch/blob/main/torch/autograd/profiler_util.py#L656
            flops = event.flops
            op_name = event.key
            op_count = event.count
            inp_shapes = event.input_shapes
            if flops != 0:
                # print(f"FLOPS : event_id:{event_id}  op_name:{op_name}  op_count:{op_count}  flops:{flops}  inp_shapes:{inp_shapes}")
                f.write(f"{event_id},{op_name},{op_count},{flops},{inp_shapes}\n")
                if op_counts.get(op_name) == None:
                    op_counts[op_name] = flops
                else:
                    op_counts[op_name] += flops

            if (flops == 0) and (op_name in op_name_exp_list):
                # print(f"EXP FLOPS : event_id:{event_id}  op_name:{op_name}  op_count:{op_count}  flops:{flops}  inp_shapes: {inp_shapes}")
                f.write(f"{event_id},{op_name},{op_count},{flops},{inp_shapes}\n")
                if op_counts_exp.get(op_name) == None:
                    op_counts_exp[op_name] = op_count
                else:
                    op_counts_exp[op_name] += op_count

            else:
                pass
                # Duplicate ops

        total_ops = 0
        f.write("\n\n")
        for op in op_counts.keys():
            print(f"{op}: {op_counts[op]}")
            f.write(f"{op}: {op_counts[op]}\n")
            total_ops += op_counts[op]

        print(f"Total GOPs: {total_ops*1e-9}")
        f.write(f"Total GOPs: {total_ops*1e-9}\n\n")

        for op in op_counts_exp.keys():
            print(f"{op}: {op_counts_exp[op]}")
            f.write(f"{op}: {op_counts_exp[op]}\n")

        f.close()


def profile_pipeline(pipe):
    prompt = "a photo of an astronaut riding a horse on mars"
    if False:
        """Detailed aten::op list"""
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True
        ) as prof:
            with torch.profiler.record_function("model_inference"):
                image = pipe(prompt, num_inference_steps=1).images[0]
        print(
            prof.key_averages(group_by_input_shape=True).table(
                sort_by="cpu_time_total", row_limit=-1
            )
        )
    else:
        # https://github.com/pytorch/pytorch/blob/main/torch/profiler/profiler.py#L508
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            with_flops=True,
        ) as prof:
            with torch.profiler.record_function("model_inference"):
                image = pipe(prompt, num_inference_steps=1).images[0]
        profile_data = prof.key_averages(
            group_by_input_shape=True
        )  # .table(sort_by="cpu_time_total", row_limit=-1)

        # reference: # https://github.com/pytorch/pytorch/blob/main/torch/autograd/profiler_util.py#L1019
        op_counts = {}
        op_name_exp_list = [
            "aten::softmax",
            "aten::silu",
            "aten::group_norm",
            "aten::layer_norm",
        ]
        op_counts_exp = {}
        for event_id in range(len(profile_data)):
            # print(event_id, profile_data[event_id])
            event = profile_data[event_id]
            # print(type(event)) #class 'torch.autograd.profiler_util.FunctionEventAvg'
            # reference: https://github.com/pytorch/pytorch/blob/main/torch/autograd/profiler_util.py#L656
            flops = event.flops
            op_name = event.key
            op_count = event.count
            inp_shapes = event.input_shapes
            if flops != 0:
                print(
                    f"FLOPS : event_id:{event_id}  op_name:{op_name}  op_count:{op_count}  flops:{flops}  inp_shapes:{inp_shapes}"
                )
                if op_counts.get(op_name) == None:
                    op_counts[op_name] = flops
                else:
                    op_counts[op_name] += flops

            if (flops == 0) and (op_name in op_name_exp_list):
                print(
                    f"EXP FLOPS : event_id:{event_id}  op_name:{op_name}  op_count:{op_count}  flops:{flops}  inp_shapes:{inp_shapes}"
                )
                if op_counts_exp.get(op_name) == None:
                    op_counts_exp[op_name] = op_count
                else:
                    op_counts_exp[op_name] += op_count

            else:
                pass
                # print(f"NOFLOPS : event_id:{event_id}  op_name:{op_name}  op_count:{op_count}  flops:{flops}  inp_shapes:{inp_shapes}")

        total_ops = 0
        for op in op_counts.keys():
            print(f"{op}: {op_counts[op]}")
            total_ops += op_counts[op]

        print(f"Total GOPs: {total_ops*1e-9}")

        for op in op_counts_exp.keys():
            print(f"{op}: {op_counts_exp[op]}")

        """
        aten::add: 347680955
        aten::addmm: 541835132928
        aten::mul: 162056679
        aten::bmm: 299836743680
        aten::conv2d: 3359769755648
        aten::mm: 108246759424
        Total GOPs: 4310.198129314001

        aten::layer_norm: 148
        aten::softmax: 81
        aten::silu: 97
        aten::group_norm: 91
        """

    image.save("astronaut_rides_horse.png")


def run_clip(pipe, model_id=""):
    """
    ***** pipe.text_encoder inputs *****
    text_input_ids.shape=torch.Size([1, 77]) text_input_ids.dtype=torch.int64
    attention_mask=None
    ***** pipe.text_encoder outputs *****
    prompt_embeds.shape=torch.Size([1, 77, 768]) prompt_embeds.dtype=torch.float32
    """
    text_input_ids = torch.rand((1, 77)).to(torch.int64)
    attention_mask = None

    kwargs = {"input_ids": text_input_ids, "attention_mask": attention_mask}
    count_gops(pipe.text_encoder, kwargs, model_id)

    prompt_embeds = pipe.text_encoder(text_input_ids, attention_mask=attention_mask)
    prompt_embeds = prompt_embeds[0]
    # print(f"{prompt_embeds.shape=}")
    # prompt_embeds.shape=torch.Size([1, 77, 768])


def run_unet(pipe, model_id=""):
    """
    ***** pipe.unet inputs *****
    latent_model_input.shape=torch.Size([2, 4, 64, 64]) latent_model_input.dtype=torch.float32
    t=tensor(1) t.dtype=torch.int64
    prompt_embeds.shape=torch.Size([2, 77, 768]) prompt_embeds.dtype=torch.float32
    timestep_cond=None
    cross_attention_kwargs=None
    added_cond_kwargs=None
    ***** pipe.unet outputs *****
    noise_pred.shape=torch.Size([2, 4, 64, 64]) noise_pred.dtype=torch.float32
    """
    latent_model_input = torch.rand((2, 4, 64, 64)).to(torch.float32)
    t = torch.tensor(1).to(torch.int64)
    prompt_embeds = torch.rand((2, 77, 768)).to(torch.float32)
    timestep_cond = None
    cross_attention_kwargs = None
    added_cond_kwargs = None

    kwargs = {
        "sample": latent_model_input,
        "timestep": t,
        "encoder_hidden_states": prompt_embeds,
        "timestep_cond": timestep_cond,
        "cross_attention_kwargs": cross_attention_kwargs,
        "added_cond_kwargs": added_cond_kwargs,
    }

    count_gops(pipe.unet, kwargs, model_id)

    kwargs = {
        "sample": latent_model_input,
        "timestep": t,
        "encoder_hidden_states": prompt_embeds,
        "timestep_cond": timestep_cond,
        "cross_attention_kwargs": cross_attention_kwargs,
        "added_cond_kwargs": added_cond_kwargs,
    }

    count_gops(pipe.unet, kwargs, model_id)

    noise_pred = pipe.unet(
        latent_model_input,
        t,
        encoder_hidden_states=prompt_embeds,
        timestep_cond=timestep_cond,
        cross_attention_kwargs=cross_attention_kwargs,
        added_cond_kwargs=added_cond_kwargs,
        return_dict=False,
    )[0]

    # print(f"{noise_pred.shape=} {noise_pred.dtype=}")
    # noise_pred.shape=torch.Size([2, 4, 64, 64]) noise_pred.dtype=torch.float32


def run_decoder(pipe, model_id=""):
    """
    ***** pipe.vae.decode inputs *****
    del1.shape=torch.Size([1, 4, 64, 64]) del1.dtype=torch.float32
    generator=None
    ***** pipe.vae.decode outputs *****
    image.shape=torch.Size([1, 3, 512, 512]) image.dtype=torch.float32
    """
    latents = torch.rand((1, 4, 64, 64)).to(torch.float32)
    generator = None

    kwargs = {"z": latents, "return_dict": False, "generator": generator}

    count_gops(pipe.vae.decode, kwargs, model_id)

    image = pipe.vae.decode(latents, return_dict=False, generator=generator)[0]
    # print(f"{image.shape=} {image.dtype=}")
    # image.shape=torch.Size([1, 3, 512, 512]) image.dtype=torch.float32


def run_safety_checker(pipe, model_id=""):
    """
    ***** pipe.safety_checker inputs *****
    image.shape=torch.Size([1, 3, 512, 512]) image.dtype=torch.float32
    clip_input.shape=torch.Size([1, 3, 224, 224]) clip_input.dtype=torch.float32
    ***** pipe.safety_checker outputs *****
    image.shape=torch.Size([1, 3, 512, 512]) image.dtype=torch.float32
    has_nsfw_concept=[True]
    """
    image = torch.rand((1, 3, 512, 512)).to(torch.float32)
    clip_input = torch.rand((1, 3, 224, 224)).to(torch.float32)

    kwargs = {"clip_input": clip_input, "images": image}
    count_gops(pipe.safety_checker, kwargs, model_id)

    image, has_nsfw_concept = pipe.safety_checker(images=image, clip_input=clip_input)
    # print(image.shape)
    # torch.Size([1, 3, 512, 512])


if __name__ == "__main__":
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float)
    pipe = pipe.to("cpu")
    prompt = "a photo of an astronaut riding a horse on mars"

    # profile_pipeline(pipe)

    print(f"*** Pipe.config *** ")
    print(pipe.config)
    print(f"*** Submodels of pipe *** ")
    for key in pipe.components.keys():
        print(key)

    if False:
        image = pipe(prompt, num_inference_steps=1).images[0]
        count_gops(pipe, {"prompt": prompt, "num_inference_steps": 1})

    # Analyze SD1.5
    for func in [run_clip, run_unet, run_decoder, run_safety_checker]:
        func(pipe, model_id="sd1_5")

    # Analyze SD1.4
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float)
    print(f"*** Pipe.config *** ")
    print(pipe.config)
    print(f"*** Submodels of pipe *** ")
    for key in pipe.components.keys():
        print(key)
    for func in [run_clip, run_unet, run_decoder]:
        func(pipe, model_id="sd1_4")
