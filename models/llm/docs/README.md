# LLMs on RyzenAI with Pytorch

## Performance on PHX, HPT and STX

The following table provides the supported model list. To quantize and use the model for text generation /other downstream inference tasks, please refer to individual README files in the table below. The README files consist of commands to quantize, execute the model.

| Model Name                                               | Quantization |
|----------------------------------------------------------|--------------|
| [TinyLlama/TinyLlama-1.1B-Chat-v1.0](./tinyLlama-1.1b.md)| PerGrp       |
| [facebook/opt-1.3b](./opt-1.3b_w4abf16.md)               | AWQ          |
| [google/gemma-2b](./gemma-2b.md)                         | AWQ          |
| [microsoft/phi-2](./phi-2.md)                            | PerGrp       |
| [microsoft/phi-3](./phi-3.md)                            | PerGrp       |
| [microsoft/Phi-3.5-mini-instruct](./phi-3.5.md)          | PerGrp/AWQ   |
| [THUDM/chatglm-6b](./chatglm-6b.md)                      | PerGrp       |
| [THUDM/chatglm3-6b](./chatglm3-6b.md)                    | PerGrp       |
| [facebook/opt-6.7b](./opt-6.7b_w4abf16.md)               | AWQ          |
| [llama-2-7b-chat](./llama2_w4abf16.md)                   | AWQ          |
| [Metal-Llama-3-8B-Instruct](./llama3-8b.md)              | PerGrp/AWQ   |
| [Metal-Llama-3.1-8B](./llama31-8b.md)                    | PerGrp       |
| [codellama/CodeLlama-7b-hf](./codeLlama-7b.md)           | AWQ          |
| [Qwen/Qwen-7b](./qwen-7b.md)                             | AWQ          |
| [Qwen/Qwen1.5-7B](./qwen1.5-7b.md)                       | AWQ          |
| [Qwen/Qwen1.5-7B-Chat](./qwen1.5-7b.md)                  | AWQ          |
| [google/gemma-7b](./gemma-7b.md)                         | AWQ          |
| [mistralai/Mistral-7B-v0.1](./mistral-7b.md)             | PerGrp       |
| [state-spaces/mamba-1.4b-hf](./mamba-1.4b.md)            | PerGrp       |
| [state-spaces/mamba-2.8b-hf](./mamba-2.8b.md)            | PerGrp       |
| [Metal-Llama-3.2-1B-Early](./llama32-1b.md)              | PerGrp/AWQ   |
| [Metal-Llama-3.2-3B-Early](./llama32-3b.md)              | PerGrp       |

[Smoothquant + w8a8 models](./smoothquant_latency.md)

There are 4 quantization recipes, as described in [main readme](./../../../README.md).

## Steps to run the models

### Recipe 1: Smoothquant with w8a8/w8a16

:pushpin: `w8a16` only supported on STX

```python run_smoothquant.py --help```

```powershell
# CPU - bf16
python run_smoothquant.py --model_name llama-2-7b --task benchmark --target cpu --precision bf16

# AIE (w8a16 only supported on STX)
python run_smoothquant.py --model_name llama-2-7b --task quantize
python run_smoothquant.py --model_name llama-2-7b --task benchmark --target aie --precision w8a8
python run_smoothquant.py --model_name llama-2-7b --task benchmark --target aie --precision w8a16
```

### Recipe 2: AWQ with w4abf16

```python run_awq.py --help```

```powershell
# CPU
python run_awq.py --model_name llama-2-7b-chat --task benchmark --target cpu --precision bf16

# AIE
python run_awq.py --model_name llama-2-7b-chat --task quantize
```

### Recipe 3: AWQ + lm_head with w4abf16

```powershell
python run_awq.py --model_name llama-2-7b-chat --task quantize --algorithm awqplus
python run_awq.py --model_name llama-2-7b-chat --task decode --algorithm awqplus
```

### Recipe 4: All Linear layers with w4abf16

```powershell
python run_awq.py --model_name llama-2-7b-chat --task quantize --algorithm pergrp
python run_awq.py --model_name llama-2-7b-chat --task decode --algorithm pergrp
```

### Recipe 5: Layers profiling with w4abf16

```powershell
model_prof.bat llama-2-7b proflemodel2k
```

**Note:** Each run generates a log file in `./logs` directory with name `log_<model_name>.log`.


## Profiling

The time-to-first-token and token-time calculation is described in the below figure.

![Token time](../figures/ttft_and_token_time_calc.png)
