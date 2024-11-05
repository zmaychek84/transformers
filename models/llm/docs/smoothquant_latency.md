# LLMs with SmoothQuant

### All CPU baselines are at bfloat16
### Units: ms

# PHX - w8a8
## Prompt length: 16 tokens

 Model                 | CPU Prefill | CPU Token | AIE Prefill | AIE Token |
-----------------------|-------------|-----------|-------------|-----------|
facebook/opt-125m      |  32         |  20       |   89        |  64       |
facebook/opt-1.3b      | 171         | 126       |  190        | 128       |
facebook/opt-1.3b + FA |  no support |no support |  152        | 110       |
facebook/opt-2.7b      | 275         | 208       |  764        | 600       |
facebook/opt-6.7b      | 495         | 390       | 1104        | 812       |
llama-2-7b             | 579         | 428       | 1110        | 889       |
llama-2-7b + FA        |  no support |no support | 1084        | 856       |
bigscience/bloom-560m  | 118         | 64        |  176        | 135       |
bigscience/bloom-1b1   | 170         | 104       |  187        | 133       |
bigscience/bloom-3b    | 337         | 222       |  768        | 603       |

```
# Examples to run the script:
# CPU
python run_smoothquant.py --model_name bigscience/bloom-560m --task benchmark --target cpu --precision bf16

# AIE
python run_smoothquant.py --model_name bigscience/bloom-560m --task quantize
python run_smoothquant.py --model_name bigscience/bloom-560m --task benchmark --target aie

python run_smoothquant.py --model_name llama-2-7b --task benchmark --target aie --flash_attention
```

# HPT - w8a8
## Prompt length: 16 tokens

 Model                 | CPU Prefill | CPU Token | AIE Prefill | AIE Token |
-----------------------|-------------|-----------|-------------|-----------|
facebook/opt-1.3b      |             |           |             |           |
facebook/opt-1.3b + FA |  no support |no support |             |           |
facebook/opt-2.7b      |             |           |             |           |
facebook/opt-6.7b      |             |           |             |           |
llama-2-7b             |             |           |             |           |
llama-2-7b + FA        |  no support |no support |             |           |
bigscience/bloom-3b    |             |           |             |           |


# STX - w8a16
## Prompt length: 16 tokens

 Model                 | CPU Prefill | CPU Token | AIE Prefill | AIE Token |
-----------------------|-------------|-----------|-------------|-----------|
facebook/opt-1.3b      |             |           |             |           |
facebook/opt-1.3b + FA |  no support |no support |             |           |
facebook/opt-2.7b      |             |           |             |           |
facebook/opt-6.7b      |             |           |             |           |
llama-2-7b             |             |           |             |           |
llama-2-7b + FA        |  no support |no support |             |           |
bigscience/bloom-3b    |             |           |             |           |
