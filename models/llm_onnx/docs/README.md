# LLMs on RyzenAI with ONNXRuntime

The following models are supported on RyzenAI with the 4 bit quantization.

:pushpin: The numbers are for input sequence length = 8, Max new Tokens = 11

| Model Name                                               | Quantization |  HPT  |  STX  | CPU bf16 HPT |
|----------------------------------------------------------|--------------|-------|-------|--------------|
| [facebook/opt-125m](./opt-125m_w4abf16.md)               | Blockwise    | 20.30 | 31.15 |   50.76      |
| [facebook/opt-1.3b](./opt-1.3b_w4abf16.md)               | Blockwise    |  6.79 |  9.16 |   -          |
| [facebook/opt-2.7b](./opt-2.7b_w4abf16.md)               | Blockwise    |  4.20 |  5.44 |   -          |
| [meta-llama/Llama-2-7b-hf](./llama2_w4abf16.md)          | Blockwise    |  4.05 |  7.30 |   -          |
| [Qwen/Qwen1.5-7B-Chat](./qwen1.5-7b.md)                  | Blockwise    |  6.00 |  6.90 |   -          |
| [THUDM/chatglm3-6b](./chatglm3-6b.md)                    | Blockwise    |   -   |   -   |   -          |
| [codellama/CodeLlama-7b-hf](./codellama-7b.md)           | Blockwise    |  6.28 |  7.35 |   -          |

:pushpin: [Blockwise WOQ](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/matmul_4bits_quantizer.py)

:pushpin: We recommend to use minimum 64 GB RAM
## Prerequisites

:pushpin: To request access for Llama-2,
visit [Meta's website](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)
and accept [Huggingface license](https://huggingface.co/meta-llama/Llama-2-7b-hf).

:pushpin: Conda environment with python 3.10

Create conda environment:
```powershell
conda update -n base -c defaults conda -y
conda env create --file=env.yaml
conda activate llm_onnx
```

## Steps to run the models

### Prepare the model

Use "prepare_model.py" script to export, optimize and quantize the LLMs. You can also optimize or quantize an existing ONNX model by providing the path to the model directory.

Check script usage
```powershell
python prepare_model.py --help

usage: prepare_model.py [-h]
                        [--model_name {facebook/opt-125m,facebook/opt-1.3b,facebook/opt-2.7b,facebook/opt-6.7b,llama-2-7b,meta-llama/Llama-2-7b-hf,Qwen/Qwen1.5-7B-Chat,THUDM/chatglm3-6b,codellama/CodeLlama-7b-hf}]
                        [--groupsize {32,64,128}] --output_model_dir OUTPUT_MODEL_DIR [--input_model INPUT_MODEL] [--only_onnxruntime]
                        [--opt_level {0,1,2,99}] [--export] [--optimize] [--quantize]

optional arguments:
  -h, --help            show this help message and exit
  --model_name {facebook/opt-125m,facebook/opt-1.3b,facebook/opt-2.7b,facebook/opt-6.7b,llama-2-7b,meta-llama/Llama-2-7b-hf,Qwen/Qwen1.5-7B-Chat,THUDM/chatglm3-6b,codellama/CodeLlama-7b-hf}
                        model name
  --groupsize {32,64,128}
                        group size for blockwise quantization
  --output_model_dir OUTPUT_MODEL_DIR
                        output directory path
  --input_model INPUT_MODEL
                        input model path to optimize/quantize
  --only_onnxruntime    optimized by onnxruntime only, and no graph fusion in Python
  --opt_level {0,1,2,99}
                        onnxruntime optimization level. 0 will disable onnxruntime graph optimization. Level 2 and 99 are intended for --only_onnxruntime.
  --export              export float model
  --optimize            optimize exported model
  --quantize            quantize float model
```

#### Export, Optimize and quantize the model

```powershell
<copy llama-2-7b weights to llm_onnx folder>

python .\prepare_model.py --model_name <model_name> --output_model_dir <output directory> --export --optimize --quantize
```
#### Optimize and quantize existing model

```powershell
python .\prepare_model.py --model_name <model_name> --input_model <input model path> --output_model_dir <output directory> --optimize --quantize
```

#### Quantize existing model

```powershell
python .\prepare_model.py --input_model <input model path> --output_model_dir <output directory> --quantize
```

### Running Inference

Check script usage
```powershell
python infer.py --help

usage: infer.py [-h] --model_dir MODEL_DIR [--draft_model_dir DRAFT_MODEL_DIR] --model_name
                {facebook/opt-125m,facebook/opt-1.3b,facebook/opt-2.7b,facebook/opt-6.7b,meta-llama/Llama-2-7b-hf,Qwen/Qwen1.5-7B-Chat,THUDM/chatglm3-6b,codellama/CodeLlama-7b-hf}
                [--tokenizer TOKENIZER] [--dll DLL] [--target {cpu,aie}] [--task {decode,benchmark,perplexity}] [--seqlen SEQLEN [SEQLEN ...]]
                [--max_new_tokens MAX_NEW_TOKENS] [--ort_trace] [--view_trace] [--prompt PROMPT] [--max_length MAX_LENGTH] [--profile] [--power_profile]
                [-v]

LLM Inference on Ryzen-AI

optional arguments:
  -h, --help            show this help message and exit
  --model_dir MODEL_DIR
                        Model directory path
  --draft_model_dir DRAFT_MODEL_DIR
                        Draft Model directory path for speculative decoding
  --model_name {facebook/opt-125m,facebook/opt-1.3b,facebook/opt-2.7b,facebook/opt-6.7b,meta-llama/Llama-2-7b-hf,Qwen/Qwen1.5-7B-Chat,THUDM/chatglm3-6b,codellama/CodeLlama-7b-hf}
                        model name
  --tokenizer TOKENIZER
                        Path to the tokenizer (Optional).
  --dll DLL             Path to the Ryzen-AI Custom OP Library
  --target {cpu,aie}    Target device (CPU or Ryzen-AI)
  --task {decode,benchmark,perplexity}
                        Run model with a specified task
  --seqlen SEQLEN [SEQLEN ...]
                        Input Sequence length for benchmarks
  --max_new_tokens MAX_NEW_TOKENS
                        Number of new tokens to be generated
  --ort_trace           Enable ORT Trace dump
  --view_trace          Display trace summary on console
  --prompt PROMPT       User prompt
  --max_length MAX_LENGTH
                        Number of tokens to be generated
  --profile             Enable profiling summary
  --power_profile       Enable power profiling via AGM
  -v, --verbose         Enable argument display
```

> As we are using an `int4` quantized model, responses might not be as accurate
as `float32` model. The quantizer used is `MatMul4BitsQuantizer` from onnxruntime

 > As for the optimizer , ORT optimizer is used.
### Using ONNX Runtime Interface

**Note:** Each run generates a log file in `./logs` directory with name `log_<model_name>.log`.
```powershell
python .\infer.py --model_name <model_name> --target <cpu/aie> --model_dir <input model directory> --profile --task decode
```
