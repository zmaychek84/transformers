# Transformers

This repository consists of methods to run Transformers in PyTorch and ONNX with operators dispatch to AIE.

The target models are all Transformer models including Generate AI, LLMs, Stable Diffusion and like.

## Features

- Extension of Pytorch with custom operators in C++
- Eager mode execution with quantization and in-place op replacement strategy
- Flash Attention v2 for OPT to reduce memory utilization and increase prefill phase performance
- State of the art [AWQ](https://arxiv.org/pdf/2306.00978.pdf) for 3-bit and 4-bit quantization
- State-of-the-art [SmoothQuant](https://arxiv.org/pdf/2211.10438.pdf) to condition weights for 8-bit quantization
- [Dynamic quantization of Transformer models (Generative LLMs, Stable Diffusion, etc.)](./models/llm/run_awq.py)
- Model analysis with observer insertion
- Layer parameter caching, checkpointing
- Perplexity, MMLU and HumanEval accuracy measurement of LLMs
- [Benchmarking LLMs with state-of-the-art methods](./models/llm/docs/README.md)
- Pytorch -> ONNX using Optimum ORT Quantizer framework and eager execution on ONNX-EP
- Automatic selection of custom compute kernels for optimal prompt/prefill phase latency
- **Speculative Decoding** with HF pipline supported for Llama2, OPT and CodeLlama2-7b
- **GGUF** model support with **[llama.cpp](./models/llm_gguf/docs/README.md)** framework
- Common C++ backend for Pytorch, ONNX and GGUF frameworks

## Models supported with Pytorch flow

The following models are supported on RyzenAI with the 4 quantization recipes described in here.

| Model Name                   | SmoothQuant |   AWQ   | AWQPlus | PerGroup | Quant Model Size
|------------------------------|-------------|---------|---------|----------|-----------------
| facebook/opt-125m            | &check;     | &check; | &check; | &check;  | 0.07
| facebook/opt-1.3b            | &check;     | &check; | &check; | &check;  | 0.8
| facebook/opt-2.7b            | &check;     | &check; | &check; | &check;  | 1.4
| facebook/opt-6.7b            | &check;     | &check; | &check; | &check;  | 3.8
| facebook/opt-13b             |             | &check; | &check; | &check;  | 7.5
| llama-2-7b*                  | &check;     |         |         | &check;  | 3.9
| llama-2-7b-chat*             | &check;     | &check; | &check; | &check;  | 3.9
| llama-2-13b*                 |             |         |         | &check;  | 7.2
| llama-2-13b-chat*            |             | &check; | &check; | &check;  | 7.2
| Meta-Llama-3-8B-Instruct  *  |             | &check; | &check; | &check;  | 4.8
| Meta-Llama-3-8B  *           |             | &check; | &check; | &check;  | 4.8
| Meta-Llama-3.1-8B  *         |             |         |         | &check;  | 4.8
| Meta-Llama-3.2-1B-Early      |             | &check; | &check; | &check;  | 0.3
| Meta-Llama-3.2-3B-Early      |             |         |         | &check;  | 4.8
| bigscience/bloom-560m        | &check;     |         |         | &check;  | 1.6
| bigscience/bloom-1b1         | &check;     |         |         | &check;  | 0.65
| bigscience/bloom-3b          | &check;     |         |         | &check;  | 1.7
| bigcode/starcoder            |             | &check; | &check; | &check;  | 8.0
| code-llama-2-7b*             |             | &check; | &check; | &check;  | 3.9
| codellama/CodeLlama-7b-hf    |             | &check; | &check; | &check;  | 3.9
| codellama/CodeLlama-7b-instruct-hf     |             | &check; | &check; | &check;  | 3.9
| google/gemma-2b  **          |             | &check; | &check; | &check;  | 1.2
| google/gemma-7b  **          |             | &check; | &check; | &check;  | 4.0
| THUDM/chatglm-6b             |             |         |         | &check;  | 3.3
| THUDM/chatglm3-6b            |             | &check; | &check; | &check;  | 4.1
| Qwen/Qwen-7b                 |             | &check; | &check; | &check;  | 4.1
| Qwen/Qwen1.5-7B              |             | &check; | &check; | &check;  | 4.1
| Qwen/Qwen1.5-7B-Chat         |             | &check; | &check; | &check;  | tbd
| microsoft/phi-2              |             |         |         | &check;  | tbd
| microsoft/phi-3              |             |         |         | &check;  | tbd
| microsoft/Phi-3.5-mini-instruct |             | &check; | &check; | &check;  | tbd
| mistralai/Mistral-7B-v0.1    |             |         |         | &check;  | tbd
| TinyLlama-1.1B-Chat-v1.0     |             |         |         | &check;  | tbd
| mamba-1.4b-hf  **            |             |         |         | &check;  | tbd
| mamba-2.8b-hf  **            |             |         |         | &check;  | tbd

:pushpin: **Important**
> \* Need local weights for these models.

> \** Needs transformers==4.39.1 ; ```pip install transformers==4.39.1``` and follow same `run_awq.py` commands.

* [Jump to running LLMs in Pytorch - following installations instructions below first](./models/llm/docs/README.md)

## Prequisites

The main branch is intended for continuous development. All developers must strictly adhere to [Contribution Guidelines](./README.md#code-contribution-guidelines) enumerated in subsequent sections.

### Request PHX or STX Ryzen-AI PC

- Request the board using ChangeGear. Setup the board using the instructions provided in [this link](https://confluence.xilinx.com/pages/viewpage.action?pageId=672012798#TVMDPUruntime(AIEsim,SimNowLite,IPUboard)-Windows/board).
- Run the unit testcases to confirm driver installation works correctly.

### Install dependencies

On the PC, install the following dependencies

- [Install Anaconda](https://www.anaconda.com/download)
- [Install Visual Studio 2022 Community Edition](https://visualstudio.microsoft.com/downloads)
- [Install Git](https://git-scm.com/downloads)
- Install AIE driver as described in [this link](https://confluence.xilinx.com/pages/viewpage.action?pageId=672012798#TVMDPUruntime(AIEsim,SimNowLite,IPUboard)-Windows/board)


## Setup Transformers Env

### Step 1: Download repository and setup conda environment

Open Anaconda Command Prompt **or** Anconda Powershell on Windows PC and clone Transformers repo:
```powershell
git config --global core.longpaths true
git clone --recurse-submodules https://gitenterprise.xilinx.com/VitisAI/transformers.git
cd transformers
```

Create conda environment:
```powershell
conda update -n base -c defaults conda -y
conda env create --file=env.yaml
conda activate ryzenai-transformers
build_dependencies.bat
# build_dependencies.ps1
```

AWQ Model zoo has precomputed scales, clips and zeros for various LLMs including OPT, Llama. Get the precomputed results:
```powershell
git lfs install
cd <transformers>\ext
git clone https://huggingface.co/datasets/mit-han-lab/awq-model-zoo awq_cache
copy <transformers>\models\llm\Qwen1.5-7B-Chat-w4-g128.pt <transformers>\ext\awq_cache\
copy <transformers>\models\llm\Qwen1.5-7B-w4-g128.pt <transformers>\ext\awq_cache\
copy <transformers>\models\llm\Qwen-7b-w4-g128.pt <transformers>\ext\awq_cache\
```

⚠️ **Warning:** Windows has a path length limit that you may hit when building the project or installing the wheels, resulting in cryptic errors.
To work around it, use a virtual drive to shorten the path the repository is cloned to:

*On Command Prompt*
```batch
@REM use any unused drive letter, Z: for example
subst Z: %cd%
@REM switch to the Z: drive
Z:
```

You can remove the virtual drive with:

*On Command Prompt*
```batch
subst /d Z:
```

### Step 2: Setup target environment

*On Anaconda Command Prompt*
```
## For PHX
.\setup_phx.bat

## For STX
.\setup_stx.bat
```

*On Anaconda PowerShell*
```powershell
## For PHX
.\setup_phx.ps1

## For STX
.\setup_stx.ps1
```

Remember to setup the target environment again if you switch to or from a virtual drive!

### Step 3: Build dependencies

```powershell
pip install ops\cpp --force-reinstall
pip install ops\torch_cpp --force-reinstall
```

### Step 4: Install ONNX EP for running ONNX based flows

For running onnxruntime apps, please refer to [Vitis-AI EP Installation](./docs/onnxrt_ep_setup.md) instructions.

### Step 5: Verify Installation

* [Python unit testcases](./tests/python/README.md)
* [C++ unit testcases](./tests/cpp/README.md)

### Step 6: MMLU

To measure MMLU on LLMs, download the [data](https://people.eecs.berkeley.edu/~hendrycks/data.tar), extract it, rename the folder to "mmlu_data" and place it in ```<transformers>/models/llm``` directory

## Run LLMs

* [All LLMs in Pytorch](./models/llm/docs/README.md)
* [Speculative Decoding of LLMs in Pytorch](./models/llm_assisted_generation/README.md)
* [GGUF Models with llama.cpp](./models/llm_gguf/docs/README.md)

## Eager Mode Execution Flow

The following figure shows default execution of Pytorch models on CPU.

<p align="center">
  <img src="./figures/llm-eager-flow.png" />
</p>

This flow has a Pytorch C++ extension and the hardware acceleration is within C++ extension. The C++ app has stationary weights, padding, tiling, calling AIE hardware and intermediate accumulation on CPU.  Dynamic quantization from PyTorch is leveraged for this. It will be extended to a higher accuracy quantizer.

## Code Contribution Guidelines

- Developers are required to use a fork of this repository to develop features and use it to create pull requests.
- Developers are required to add meaningful commit messages/PR titles.
- Code-checkin must happen at every low level submodule first before the check-ins to upper module is submitted.
- The PR should have the CI details from submodule to ensure traceability.

The figure below describes different components of this project for eager mode. At each level, developer is expected to write unit-tests, ensure they work on board. After all unit-tests work, model level performance analysis needs to be done.

### Hard Requirements

Refer to LLM [README](./models/llm/docs/README.md)

* ***All C++ and Python unit tests must pass***
* ***OPT and Llama2 benchmark should not regress***
* ***All models should generate good results with no degradation in performance***

### Pre-Commit

You can use [pre-commit](https://pre-commit.com/) to run formatting and linting steps.

- After cloning the repository, run `pre-commit install` to let it run the linting steps prior to every commit.
- You can also run it manually with `pre-commit run --from-ref origin/main --to-ref HEAD`.

:pushpin: **Note:** The repository does not currently meet all the checks so if you run `pre-commit run --all-files`, it will change many files.

### Python Formatting

We use isort, and black to format the Python files.

- Ensure that function annotations are used throughout the implementation.

:pushpin: **Note:** Ensure **RyzenAI** is imported at the end of all imports in `ops\python\*.py`.

### C++ Formatting

We use clang-format to format the C/C++ files.

```powershell
pre-commit run clang-format -a

# to format all files
.\ci\format_all.bat
```
