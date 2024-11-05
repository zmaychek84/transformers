# TinyLlama/TinyLlama-1.1B-Chat-v1.0
```
python run_awq.py --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --task quantize --algorithm pergrp
```

## PHX

``` python run_awq.py --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --task decode --algorithm pergrp ```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                       39 |                     30 |          6.54532 |              1118.99 |           184.715 |      5.41375 |
|          2 |                       42 |                     30 |          7.08049 |              1623.37 |           186.387 |      5.36519 |
|          3 |                       39 |                     30 |          6.52739 |              1083.31 |           185.989 |      5.37665 |
|          4 |                       39 |                     17 |          4.07095 |              1086.43 |           184.741 |      5.41299 |
|          5 |                       37 |                     30 |          6.44235 |              1087.68 |           182.909 |      5.4672  |
|          6 |                       36 |                     30 |          6.38725 |              1076.72 |           181.351 |      5.51416 |
|          7 |                       41 |                     30 |          6.91571 |              1608.92 |           181.232 |      5.51778 |
|          8 |                       39 |                     30 |          6.39547 |              1084.25 |           181.438 |      5.51152 |
|          9 |                       41 |                     30 |          7.01343 |              1621.03 |           184.201 |      5.42886 |
|         10 |                       38 |                     30 |          6.55553 |              1079.77 |           187.109 |      5.34448 |

## HPT (With MCDM)

``` python run_awq.py --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --task decode --algorithm pergrp ```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                       39 |                     30 |          4.57083 |              769.363 |           127.994 |      7.81287 |
|          2 |                       42 |                     30 |          4.79396 |             1106.85  |           125.335 |      7.97862 |
|          3 |                       39 |                     30 |          4.41538 |              738.725 |           125.043 |      7.99724 |
|          4 |                       39 |                     17 |          2.74997 |              728.94  |           124.551 |      8.02884 |
|          5 |                       37 |                     30 |          4.39974 |              732.404 |           124.72  |      8.01794 |
|          6 |                       36 |                     30 |          4.41039 |              717.951 |           125.574 |      7.96341 |
|          7 |                       41 |                     30 |          4.92054 |             1098.69  |           129.837 |      7.70199 |
|          8 |                       39 |                     30 |          4.48191 |              741.27  |           127.252 |      7.85845 |
|          9 |                       41 |                     30 |          4.76005 |             1093.75  |           124.644 |      8.02284 |
|         10 |                       38 |                     30 |          4.43295 |              738.875 |           125.646 |      7.95884 |

## STX B0 (with MCDM)

``` python run_awq.py --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --task decode --algorithm pergrp ```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                       39 |                     30 |          3.33118 |              587.995 |           91.5281 |      10.9256 |
|          2 |                       42 |                     30 |          3.48725 |              771.641 |           91.6872 |      10.9066 |
|          3 |                       39 |                     30 |          3.24659 |              535.056 |           91.276  |      10.9558 |
|          4 |                       39 |                     30 |          3.23013 |              530.884 |           91.2307 |      10.9612 |
|          5 |                       37 |                     30 |          3.23223 |              526.56  |           91.3933 |      10.9417 |
|          6 |                       36 |                     30 |          3.23859 |              522.437 |           91.741  |      10.9003 |
|          7 |                       41 |                     30 |          3.52952 |              807.975 |           91.9372 |      10.877  |
|          8 |                       39 |                     30 |          3.30511 |              586.798 |           91.8147 |      10.8915 |
|          9 |                       41 |                     30 |          3.53162 |              811.771 |           91.8608 |      10.886  |
|         10 |                       38 |                     30 |          3.29549 |              571.688 |           91.9657 |      10.8736 |

```
...
...
LlamaModelEval(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 2048)
    (layers): ModuleList(
      (0-21): 22 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): ryzenAI.QLinearPerGrp(in_features:2048, out_features:2048, bias:None, device:aie, w_bit:4 group_size:128  )
          (k_proj): ryzenAI.QLinearPerGrp(in_features:2048, out_features:256, bias:None, device:aie, w_bit:4 group_size:128  )
          (v_proj): ryzenAI.QLinearPerGrp(in_features:2048, out_features:256, bias:None, device:aie, w_bit:4 group_size:128  )
          (o_proj): ryzenAI.QLinearPerGrp(in_features:2048, out_features:2048, bias:None, device:aie, w_bit:4 group_size:128  )
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): ryzenAI.QLinearPerGrp(in_features:2048, out_features:5632, bias:None, device:aie, w_bit:4 group_size:128  )
          (up_proj): ryzenAI.QLinearPerGrp(in_features:2048, out_features:5632, bias:None, device:aie, w_bit:4 group_size:128  )
          (down_proj): ryzenAI.QLinearPerGrp(in_features:5632, out_features:2048, bias:None, device:aie, w_bit:4 group_size:128  )
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): ryzenAI.QLinearPerGrp(in_features:2048, out_features:32000, bias:None, device:aie, w_bit:4 group_size:128  )
)
model.mode_name: TinyLlama-1.1B-Chat-v1.0


...
...

****************************************
prompt: What is the meaning of life?
C:\Users\rajeevp\AppData\Local\anaconda3\envs\ryzenai-transformers\lib\site-packages\transformers\generation\configuration_utils.py:392: UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `None` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.
  warnings.warn(
response: <|system|>
You are a helpful assistant.
<|user|>
What is the meaning of life?
<|assistant|>
The meaning of life is a question that has been debated and discussed for centuries, and there are many different interpretations and perspectives on this
****************************************
prompt: Tell me something you don't know.
response: <|system|>
You are a helpful assistant.
<|user|>
Tell me something you don't know.
<|assistant|>
I don't know anything. However, I can provide you with some interesting facts and trivia:

1. The first recorded use
****************************************
prompt: What does Xilinx do?
response: <|system|>
You are a helpful assistant.
<|user|>
What does Xilinx do?
<|assistant|>
Xilinx is a leading provider of programmable logic devices (PLDs) and system-on-chip (SoC) solutions
****************************************

```
