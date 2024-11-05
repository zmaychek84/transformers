# Llama-3.2-1B

## Quantize
```python run_awq.py --model_name nltpt/Llama-3.2-1B --task quantize --algorithm pergrp```

```
LlamaModelEval(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 2048)
    (layers): ModuleList(
      (0-15): 16 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): ryzenAI.QLinearPerGrp(in_features:2048, out_features:2048, bias:False, device:aie, w_bit:4, model_name:Llama-3.2-1B, gemm_torch:1, group_size:128 )
          (k_proj): ryzenAI.QLinearPerGrp(in_features:2048, out_features:512, bias:False, device:aie, w_bit:4, model_name:Llama-3.2-1B, gemm_torch:1, group_size:128 )
          (v_proj): ryzenAI.QLinearPerGrp(in_features:2048, out_features:512, bias:False, device:aie, w_bit:4, model_name:Llama-3.2-1B, gemm_torch:1, group_size:128 )
          (o_proj): ryzenAI.QLinearPerGrp(in_features:2048, out_features:2048, bias:False, device:aie, w_bit:4, model_name:Llama-3.2-1B, gemm_torch:1, group_size:128 )
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): ryzenAI.QLinearPerGrp(in_features:2048, out_features:8192, bias:False, device:aie, w_bit:4, model_name:Llama-3.2-1B, gemm_torch:1, group_size:128 )
          (up_proj): ryzenAI.QLinearPerGrp(in_features:2048, out_features:8192, bias:False, device:aie, w_bit:4, model_name:Llama-3.2-1B, gemm_torch:1, group_size:128 )
          (down_proj): ryzenAI.QLinearPerGrp(in_features:8192, out_features:2048, bias:False, device:aie, w_bit:4, model_name:Llama-3.2-1B, gemm_torch:1, group_size:128 )
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((2048,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): ryzenAI.QLinearPerGrp(in_features:2048, out_features:128256, bias:False, device:aie, w_bit:4, model_name:Llama-3.2-1B, gemm_torch:1, group_size:128 )
)
model.mode_name: Llama-3.2-1B
****************************************
<|begin_of_text|>What is the meaning of life? It is a question that has puzzled humans for centuries. Our ancestors lived simple lives, with limited material needs and no concept of death. As they grew old, they often died, and their lives were considered complete. But with the advent of civilization and the rise of complex societies, the meaning of life
****************************************
<|begin_of_text|>Tell me something you don't know. A fascinating story behind the movie "Ghostbusters" (1984)
I'm here to share my story and get to know a new topic today.
If you are ready to share your story and hear about something new, you're in the right place!
Today, I'm looking to share a
****************************************
<|begin_of_text|>What does Xilinx do? - Understanding the company and its products
What does Xilinx do?
Xilinx, Inc. is a leading innovator of high-performance analog and digital semiconductor products and artificial intelligence platforms. The company specializes in developing field-programmable gate arrays (FPGAs), integrated circuits (ICs), and
****************************************
```

## Performance on STX
### Group size 128
```python run_awq.py --model_name nltpt/Llama-3.2-1B --task decode --algorithm pergrp```
|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        8 |                     60 |          6.28733 |              236.923 |           91.3845 |      10.9428 |
|          2 |                        9 |                     60 |          5.68973 |              212.477 |           86.8282 |      11.517  |
|          3 |                        7 |                     60 |          5.83818 |              196.315 |           89.301  |      11.1981 |
|          4 |                        8 |                     60 |          5.67541 |              223.363 |           86.5349 |      11.556  |
|          5 |                        6 |                     60 |          5.54553 |              199.03  |           84.8884 |      11.7802 |
|          6 |                        5 |                     60 |          5.77527 |              190.675 |           88.4467 |      11.3062 |
|          7 |                        8 |                     60 |          5.54881 |              197.94  |           85.0158 |      11.7625 |
|          8 |                        7 |                     60 |          5.50413 |              194.471 |           84.2976 |      11.8627 |
|          9 |                        7 |                     60 |          5.7875  |              193.207 |           88.5808 |      11.2891 |
|         10 |                        7 |                     60 |          5.52774 |              192.723 |           84.6426 |      11.8144 |

### Group size 32
```python run_awq.py --model_name nltpt/Llama-3.2-1B --task decode --algorithm pergrp --group_size 32```
|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        8 |                     60 |          8.65289 |              350.615 |           129.656 |      7.71273 |
|          2 |                        9 |                     60 |          8.38578 |              285.662 |           131.526 |      7.60303 |
|          3 |                        7 |                     60 |          8.23486 |              305.634 |           128.33  |      7.79242 |
|          4 |                        8 |                     60 |          8.52686 |              276.752 |           133.824 |      7.47251 |
|          5 |                        6 |                     60 |          8.18197 |              272.469 |           128.267 |      7.79626 |
|          6 |                        5 |                     60 |          8.42949 |              268.085 |           132.369 |      7.55462 |
|          7 |                        8 |                     60 |          8.19083 |              273.11  |           128.366 |      7.7902  |
|          8 |                        7 |                     60 |          8.45197 |              286.687 |           132.427 |      7.55135 |
|          9 |                        7 |                     60 |          8.22123 |              272.388 |           128.399 |      7.78823 |
|         10 |                        7 |                     60 |          8.55156 |              285.031 |           133.601 |      7.48497 |

### AWQ
```python run_awq.py --model_name nltpt/Llama-3.2-1B --task decode --algorithm awq```
|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        8 |                     60 |          7.58163 |              236.658 |           111.092 |      9.00151 |
|          2 |                        9 |                     60 |          6.84954 |              226.839 |           105.739 |      9.45725 |
|          3 |                        7 |                     60 |          7.11322 |              218.561 |           110.504 |      9.04948 |
|          4 |                        8 |                     60 |          6.72304 |              221.995 |           104.119 |      9.60443 |
|          5 |                        6 |                     60 |          7.06033 |              219.944 |           109.66  |      9.11909 |
|          6 |                        5 |                     60 |          6.77321 |              208.716 |           104.363 |      9.58196 |
|          7 |                        8 |                     60 |          6.76366 |              233.796 |           104.194 |      9.59749 |
|          8 |                        7 |                     60 |          7.13844 |              217.793 |           110.39  |      9.05878 |
|          9 |                        7 |                     60 |          6.76039 |              219.404 |           104.216 |      9.59547 |
|         10 |                        7 |                     60 |          7.07769 |              221.302 |           109.205 |      9.1571  |

### AWQ+
```python run_awq.py --model_name nltpt/Llama-3.2-1B --task decode --algorithm awqplus```
|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        8 |                     60 |          6.63996 |              281.658 |           96.5234 |      10.3602 |
|          2 |                        9 |                     60 |          6.38436 |              213.203 |           98.1019 |      10.1935 |
|          3 |                        7 |                     60 |          6.14341 |              207.17  |           94.5276 |      10.5789 |
|          4 |                        8 |                     60 |          6.20544 |              207.032 |           95.3851 |      10.4838 |
|          5 |                        6 |                     60 |          6.04361 |              240.043 |           92.0181 |      10.8674 |
|          6 |                        5 |                     60 |          5.9812  |              202.01  |           91.6199 |      10.9147 |
|          7 |                        8 |                     60 |          6.25742 |              207.728 |           96.371  |      10.3766 |
|          8 |                        7 |                     60 |          5.99384 |              230.57  |           91.5922 |      10.918  |
|          9 |                        7 |                     60 |          5.97085 |              207.333 |           91.3893 |      10.9422 |
|         10 |                        7 |                     60 |          6.24713 |              205.4   |           95.8238 |      10.4358 |


## Perplexity
Prompt Length | CPU BF16 | NPU pergrp g:32 | NPU pergrp g:128 | NPU AWQ | NPU AWQ+
--------------|----------|-----------------|------------------|---------|---------
16            | 221.6687 | 142.1365        | 148.4800         |         |
96            |  27.7163 |  29.3082        |  31.6714         |         |
192           |  20.6202 |  22.6469        |  25.2215         |         |
256           |  19.7360 |  21.5702        |  23.6359         |         |
512           |  15.5950 |  17.1475        |  18.8711         |         |
768           |  13.7800 |  14.9571        |  16.4670         |         |
1024          |  14.3109 |  15.5633        |  17.1228         |         |
1536          |  14.3362 |  15.5969        |  17.2408         |         |
2048          |  13.6760 |  15.0836        |  16.9680         |         |
