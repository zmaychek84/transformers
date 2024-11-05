# Meta-Llama-3-8B-Instruct / Meta-Llama-3-8B

```python run_awq.py --task quantize --model_name meta-llama/Meta-Llama-3-8B --algorithm pergrp```
```python run_awq.py --task quantize --model_name meta-llama/Meta-Llama-3-8B --algorithm awqplus```

```
LlamaModelEval(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 4096)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): ryzenAI.QLinearPerGrp(in_features:4096, out_features:4096, bias:None, device:aie, w_bit:4 group_size:128  )
          (k_proj): ryzenAI.QLinearPerGrp(in_features:4096, out_features:1024, bias:None, device:aie, w_bit:4 group_size:128  )
          (v_proj): ryzenAI.QLinearPerGrp(in_features:4096, out_features:1024, bias:None, device:aie, w_bit:4 group_size:128  )
          (o_proj): ryzenAI.QLinearPerGrp(in_features:4096, out_features:4096, bias:None, device:aie, w_bit:4 group_size:128  )
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): ryzenAI.QLinearPerGrp(in_features:4096, out_features:14336, bias:None, device:aie, w_bit:4 group_size:128  )
          (up_proj): ryzenAI.QLinearPerGrp(in_features:4096, out_features:14336, bias:None, device:aie, w_bit:4 group_size:128  )
          (down_proj): ryzenAI.QLinearPerGrp(in_features:14336, out_features:4096, bias:None, device:aie, w_bit:4 group_size:128  )
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): ryzenAI.QLinearPerGrp(in_features:4096, out_features:128256, bias:None, device:aie, w_bit:4 group_size:128  )
)
model.mode_name: Meta-Llama-3-8B-Instruct
****************************************
prompt: What is the meaning of life?
Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
What is the meaning of life? It is a question that has puzzled philosophers, scientists, and theologians for centuries. There is no one answer to this question, as the meaning of life is subjective and can vary greatly from person to person. However, here are some possible answers:

1. To find happiness and fulfillment: Many people
response: What is the meaning of life? It is a question that has puzzled philosophers, scientists, and theologians for centuries. There is no one answer to this question, as the meaning of life is subjective and can vary greatly from person to person. However, here are some possible answers:

1. To find happiness and fulfillment: Many people
****************************************
prompt: Tell me something you don't know.
Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
Tell me something you don't know. Make it interesting."
"I don't know... hmm... did you know that there's a species of jellyfish that's immortal? The Turritopsis dohrnii, also known as the 'immortal jellyfish,' is a type of jellyfish that can transform its body into a younger state
response: Tell me something you don't know. Make it interesting."
"I don't know... hmm... did you know that there's a species of jellyfish that's immortal? The Turritopsis dohrnii, also known as the 'immortal jellyfish,' is a type of jellyfish that can transform its body into a younger state
****************************************
prompt: What does Xilinx do?
Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
What does Xilinx do? Xilinx is a leading provider of programmable logic solutions, including FPGAs (Field-Programmable Gate Arrays), SoCs (Systems-on-Chip), and 3D ICs (Three-Dimensional Integrated Circuits). Their products are used in a wide range of industries, including:


response: What does Xilinx do? Xilinx is a leading provider of programmable logic solutions, including FPGAs (Field-Programmable Gate Arrays), SoCs (Systems-on-Chip), and 3D ICs (Three-Dimensional Integrated Circuits). Their products are used in a wide range of industries, including:


```
## Performance on STX B0 with AWQ Plus

```python run_awq.py --model_name meta-llama/Meta-Llama-3-8B --algorithm awqplus --fast_attention --fast_mlp --fast_norm --fast_decoder --task decode ```
|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        8 |                     60 |          9.97512 |              410.475 |           151.217 |      6.61301 |
|          2 |                        9 |                     60 |          9.59566 |              363.652 |           150.554 |      6.64214 |
|          3 |                        7 |                     60 |          9.5631  |              349.807 |           150.638 |      6.63844 |
|          4 |                        8 |                     60 |          9.65426 |              360.593 |           151.887 |      6.58386 |
|          5 |                        6 |                     60 |          9.62973 |              344.288 |           151.851 |      6.58539 |
|          6 |                        5 |                     60 |          9.67328 |              345.259 |           152.348 |      6.5639  |
|          7 |                        8 |                     60 |          9.63352 |              352.944 |           151.909 |      6.58289 |
|          8 |                        7 |                     60 |          9.61636 |              350.189 |           151.628 |      6.59508 |
|          9 |                        7 |                     60 |          9.64061 |              347.036 |           151.946 |      6.58128 |
|         10 |                        7 |                     60 |          9.59865 |              351.553 |           151.123 |      6.61713 |

```python run_awq.py --model_name meta-llama/Meta-Llama-3-8B --algorithm awqplus --fast_attention --fast_mlp --fast_norm --fast_decoder --task profilemodel256 ```
|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                      223 |                     60 |          12.3212 |              2051.07 |           164.239 |      6.08869 |

```python run_awq.py --model_name meta-llama/Meta-Llama-3-8B --algorithm awqplus --fast_attention --fast_mlp --fast_norm --fast_decoder --task profilemodel512```
|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                      442 |                     60 |          13.1588 |              2362.97 |           173.345 |      5.76884 |

```python run_awq.py --model_name meta-llama/Meta-Llama-3-8B --algorithm awqplus --fast_attention --fast_mlp --fast_norm --fast_decoder --task profilemodel1k```
|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                     1024 |                     60 |          17.2737 |              2742.24 |            233.89 |      4.27552 |

```python run_awq.py --model_name meta-llama/Meta-Llama-3-8B --algorithm awqplus --fast_attention --fast_mlp --fast_norm --fast_decoder --task profilemodel2k```
|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                     2048 |                     60 |          25.9379 |              6064.12 |           320.299 |      3.12208 |

```python run_awq.py --model_name meta-llama/Meta-Llama-3-8B --algorithm awqplus --fast_attention --fast_mlp --fast_norm --fast_decoder --task benchmark_exact```
|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                      128 |                     60 |          12.3904 |              1405.83 |           169.008 |      5.91689 |
|          2 |                      256 |                     60 |          12.3257 |              1582.44 |           175.434 |      5.70014 |
|          3 |                      512 |                     60 |          13.7471 |              2535.05 |           183.958 |      5.43602 |
|          4 |                     1024 |                     60 |          18.0433 |              3691.1  |           234.008 |      4.27336 |
|          5 |                     2048 |                     60 |          26.3571 |              6576.08 |           328.635 |      3.04289 |

```python run_awq.py --model_name meta-llama/Meta-Llama-3-8B --algorithm awqplus --fast_attention --fast_mlp --fast_norm --fast_decoder --task benchmark```
|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        5 |                     60 |          11.3605 |              565.963 |           167.093 |      5.98468 |
|          2 |                       78 |                     60 |          11.8671 |             1579.69  |           168.816 |      5.92362 |
|          3 |                      128 |                     60 |          12.1376 |             1610.8   |           171.089 |      5.84492 |
|          4 |                      140 |                     60 |          12.5177 |             2541.78  |           163.299 |      6.12374 |
|          5 |                      256 |                     60 |          12.3281 |             1544.04  |           176.817 |      5.65557 |
|          6 |                      293 |                     60 |          13.4424 |             3401.7   |           164.17  |      6.09126 |
|          7 |                      372 |                     60 |          13.8797 |             3490.42  |           170.737 |      5.85696 |
|          8 |                      512 |                     60 |          13.4978 |             2500.14  |           179.412 |      5.57377 |
|          9 |                      580 |                     60 |          16.6158 |             5134.24  |           188.032 |      5.31825 |
|         10 |                      717 |                     60 |          18.2095 |             5430.53  |           208.999 |      4.78472 |
|         11 |                      790 |                     60 |          18.5101 |             5597.09  |           212.376 |      4.70864 |
|         12 |                      800 |                     60 |          18.4869 |             5587.45  |           211.704 |      4.72358 |
|         13 |                      900 |                     60 |          18.8048 |             5672.73  |           213.089 |      4.69287 |
|         14 |                     1024 |                     60 |          17.3202 |             3681.44  |           224.806 |      4.44827 |
|         15 |                     1050 |                     60 |          22.8778 |             9104.92  |           226.915 |      4.40693 |
|         16 |                     1320 |                     60 |          25.2745 |             9600.31  |           255.839 |      3.90871 |
|         17 |                     1537 |                     60 |          26.9278 |            10045.3   |           275.839 |      3.6253  |
|         18 |                     1580 |                     60 |          26.8172 |            10108.9   |           272.779 |      3.66597 |
|         19 |                     1670 |                     60 |          27.7887 |            10424.3   |           283.324 |      3.52952 |
|         20 |                     1900 |                     60 |          29.8146 |            10746.4   |           316.203 |      3.16253 |
|         21 |                     2048 |                     60 |          26.3403 |             6538.26  |           328.925 |      3.04021 |


## Accuracy

Measured with transformers==4.40.0

**Perplexity ios measured on wikitext2-raw dataset**

Perplexity measured on n=4 samples

**Seqlen** | **CPU bf16** | **NPU w4abf16 (g:32)** | **NPU w4abf16 (g:128)** | **NPU w4abf16 (AWQ)** | **NPU w4abf16 (AWQ+)**
-----------|--------------|------------------------|-------------------------|-----------------------|-----------------------
16         | 40.463       |  59.571                | 67.417                  |  57.100               | 57.449
96         | 11.524       |  13.463                | 13.810                  |  12.615               | 12.596
192        |  9.980       |  11.440                | 12.009                  |  11.210               | 11.232
256        |  9.309       |  10.580                | 11.004                  |  10.311               | 10.371
512        |  6.906       |  8.102                 |  8.725                  |   8.392               |  8.439
768        |  6.884       |  7.780                 |  8.203                  |   7.959               |  8.019
1024       |  7.032       |  7.876                 |  8.314                  |   8.067               |  8.126
1536       |  7.341       |  8.046                 |  8.403                  |   8.220               |  8.281
2048       |  6.813       |  7.378                 |  7.708                  |   7.556               |  7.607

**MMLU is measured on 30 samples of abstract_algebra**

##  Meta-Llama-3-8B

 **Device**                         | **MMLU**
------------------------------------|----------
CPU BF16                            | 40.00
NPUw4abf16 (Per Grp, 4-bit, g:128)  | 40.00

##  Meta-Llama-3-8B-Instruct

| **Precision + Config**               | **Device** | **Perplexity (2 samples)** | **MMLU**
|--------------------------------------|------------|----------------------------|-----------
BF16                                   | CPU        |  9.239                     | 36.67
w4abf16 (Per Grp, 4-bit, g:128)        | NPU        |  9.881                     | 40.00
