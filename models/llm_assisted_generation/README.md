# Assisted generation

Assisted generation is similar to speculative decoding, released by [HuggingFace](https://huggingface.co/blog/assisted-generation).

# Setup
## Step 1: Generate AWQ checkpoints
Generate the quantized checkpoints by running ```models/llm/run_awq.py``` for target models and ```models/llm/run_smoothquant.py``` for opt-125m from ```models\llm``` directory. AMD draft models are used with bf16 precision on CPU.

## Step 2: Get draft models
* For opt-125m, generate checkpoint using ```models/llm/run_smoothquant.py```
* For CodeLlama2-7b and Llama2-7b, obtain **amd-code-135m_104k_python_only_hf_4.37.2** and **amd-pretrained-135m-hf_4.37.2** from engineering team and copy into ```models\llm``` directory, obtain **amd-code-135m_smoothquant.pth** from engineering team and copy into ```models\llm\quantized_models``` directory.

* Speed up observed is 2-3x in token time. Please look at individual readmes.

|   Target Model                | Assistant Model            | README                                  |
|-------------------------------|----------------------------|-----------------------------------------|
| CodeLlama-7b-hf (AWQ-w4abf16) | amd-code-135m (bf16/w8a8)       | [CodeLlama2-7b](./codeLlama2-7b.md)     |
| llama-2-7b-chat (AWQ-w4abf16) | amd-pretrained-135m (bf16/w8a8) | [Llama2-7b](./llama2-7b.md)                |
| Qwen1.5-7B-Chat (AWQ-w4abf16) | Qwen1.5-0.5B-Chat (bf16)   | No speed-up |
| OPT-6.7b (AWQ-w4abf16)        | OPT-125M (bf16/SQ-w8a8)         |TBD                 |
