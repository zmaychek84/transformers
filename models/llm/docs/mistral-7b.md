# mistralai/Mistral-7B-v0.1 (PHX)
```
python run_awq.py --model_name mistralai/Mistral-7B-v0.1 --task quantize --algorithm pergrp
python run_awq.py --model_name mistralai/Mistral-7B-v0.1 --task decode --target aie --algorithm pergrp

MistralModelEval(
  (model): MistralModel(
    (embed_tokens): Embedding(32000, 4096)
    (layers): ModuleList(
      (0-31): 32 x MistralDecoderLayer(
        (self_attn): MistralAttention(
          (q_proj): ryzenAI.QLinearPerGrp(in_features:4096, out_features:4096, bias:None, device:aie, w_bit:4 group_size:128  )
          (k_proj): ryzenAI.QLinearPerGrp(in_features:4096, out_features:1024, bias:None, device:aie, w_bit:4 group_size:128  )
          (v_proj): ryzenAI.QLinearPerGrp(in_features:4096, out_features:1024, bias:None, device:aie, w_bit:4 group_size:128  )
          (o_proj): ryzenAI.QLinearPerGrp(in_features:4096, out_features:4096, bias:None, device:aie, w_bit:4 group_size:128  )
          (rotary_emb): MistralRotaryEmbedding()
        )
        (mlp): MistralMLP(
          (gate_proj): ryzenAI.QLinearPerGrp(in_features:4096, out_features:14336, bias:None, device:aie, w_bit:4 group_size:128  )
          (up_proj): ryzenAI.QLinearPerGrp(in_features:4096, out_features:14336, bias:None, device:aie, w_bit:4 group_size:128  )
          (down_proj): ryzenAI.QLinearPerGrp(in_features:14336, out_features:4096, bias:None, device:aie, w_bit:4 group_size:128  )
          (act_fn): SiLU()
        )
        (input_layernorm): MistralRMSNorm()
        (post_attention_layernorm): MistralRMSNorm()
      )
    )
    (norm): MistralRMSNorm()
  )
  (lm_head): ryzenAI.QLinearPerGrp(in_features:4096, out_features:32000, bias:None, device:aie, w_bit:4 group_size:128  )
)
model.mode_name: Mistral-7B-v0.1
****************************************
prompt: What is the meaning of life?
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
response: What is the meaning of life?

This is a question that has been asked by philosophers and theologians for centuries.

There are many different answers to this question
****************************************
prompt: Tell me something you don't know.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
response: Tell me something you don't know.

I'm not sure what I'm looking for.

I'm not sure what I'm looking for.


****************************************
prompt: What does Xilinx do?
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
response: What does Xilinx do?

Xilinx is a leading provider of All Programmable FPGAs, SoCs, MPSoCs and 3D
****************************************
...
```


|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        8 |                     30 |          14.4961 |              845.578 |           468.521 |      2.13438 |
|          2 |                       10 |                     30 |          15.9834 |             2323.5   |           469.243 |      2.13109 |
|          3 |                        8 |                     30 |          14.4047 |              759.55  |           468.689 |      2.13361 |
|          4 |                        8 |                     30 |          14.4126 |              762.548 |           468.865 |      2.13281 |
|          5 |                        6 |                     30 |          14.3483 |              740.249 |           467.385 |      2.13956 |
|          6 |                        6 |                     30 |          14.4328 |              741.626 |           470.186 |      2.12682 |
|          7 |                        8 |                     30 |          14.5223 |              760.774 |           472.712 |      2.11545 |
|          8 |                        8 |                     30 |          14.6033 |              757.445 |           475.628 |      2.10248 |
|          9 |                        9 |                     30 |          16.1047 |             2311.97  |           473.814 |      2.11053 |
|         10 |                        7 |                     30 |          14.5631 |              747.776 |           474.502 |      2.10747 |

## PHX

```python run_awq.py --model_name mistralai/Mistral-7B-v0.1  --task decode --target aie --algorithm pergrp```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        8 |                     30 |          14.4961 |              845.578 |           468.521 |      2.13438 |
|          2 |                       10 |                     30 |          15.9834 |             2323.5   |           469.243 |      2.13109 |
|          3 |                        8 |                     30 |          14.4047 |              759.55  |           468.689 |      2.13361 |
|          4 |                        8 |                     30 |          14.4126 |              762.548 |           468.865 |      2.13281 |
|          5 |                        6 |                     30 |          14.3483 |              740.249 |           467.385 |      2.13956 |
|          6 |                        6 |                     30 |          14.4328 |              741.626 |           470.186 |      2.12682 |
|          7 |                        8 |                     30 |          14.5223 |              760.774 |           472.712 |      2.11545 |
|          8 |                        8 |                     30 |          14.6033 |              757.445 |           475.628 |      2.10248 |
|          9 |                        9 |                     30 |          16.1047 |             2311.97  |           473.814 |      2.11053 |
|         10 |                        7 |                     30 |          14.5631 |              747.776 |           474.502 |      2.10747 |

```python run_awq.py --model_name mistralai/Mistral-7B-v0.1  --task decode --target aie --algorithm pergrp --flash_attention_plus```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        8 |                     30 |          36.1892 |              792.004 |           436.204 |      2.29251 |
|          2 |                       10 |                     30 |          15.0059 |             2285.1   |           436.85  |      2.28912 |
|          3 |                        8 |                     30 |          13.4857 |              725.553 |           438.142 |      2.28237 |
|          4 |                        8 |                     30 |          13.4521 |              721.859 |           437.156 |      2.28751 |
|          5 |                        6 |                     30 |          13.4135 |              707.197 |           436.136 |      2.29286 |
|          6 |                        6 |                     30 |          13.3682 |              707.995 |           434.662 |      2.30064 |
|          7 |                        8 |                     30 |          25.3397 |              728.759 |           434.846 |      2.29966 |
|          8 |                        8 |                     30 |          13.5121 |              727.641 |           439.017 |      2.27781 |
|          9 |                        9 |                     30 |          15.007  |             2273.11  |           437.198 |      2.28729 |
|         10 |                        7 |                     30 |          13.4476 |              713.8   |           437.087 |      2.28788 |

## HPT (With MCDM)

```python run_awq.py --model_name mistralai/Mistral-7B-v0.1  --task decode --target aie --algorithm pergrp```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        8 |                     30 |          9.37074 |              625.685 |           298.243 |      3.35297 |
|          2 |                       10 |                     30 |         10.1082  |             1504.55  |           294.834 |      3.39174 |
|          3 |                        8 |                     30 |          9.18985 |              514.244 |           297.254 |      3.36412 |
|          4 |                        8 |                     30 |          9.27881 |              513.596 |           300.46  |      3.32823 |
|          5 |                        6 |                     30 |          9.18676 |              497.862 |           297.694 |      3.35916 |
|          6 |                        6 |                     30 |          9.16776 |              496.857 |           297.11  |      3.36576 |
|          7 |                        8 |                     30 |          9.20139 |              513.373 |           297.752 |      3.3585  |
|          8 |                        8 |                     30 |          9.22406 |              513.94  |           298.516 |      3.3499  |
|          9 |                        9 |                     30 |         10.1803  |             1495.99  |           297.594 |      3.36028 |
|         10 |                        7 |                     30 |          9.09828 |              497.377 |           294.644 |      3.39393 |

```python run_awq.py --model_name mistralai/Mistral-7B-v0.1  --task decode --target aie --algorithm pergrp --flash_attention_plus```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        8 |                     30 |          8.79006 |              576.691 |           279.995 |      3.57149 |
|          2 |                       10 |                     30 |          9.65684 |             1480.07  |           279.887 |      3.57288 |
|          3 |                        8 |                     30 |          8.63721 |              493.251 |           278.936 |      3.58506 |
|          4 |                        8 |                     30 |          8.59021 |              494.126 |           277.403 |      3.60487 |
|          5 |                        6 |                     30 |          8.62853 |              467.356 |           279.502 |      3.57779 |
|          6 |                        6 |                     30 |          8.60827 |              466.413 |           278.55  |      3.59002 |
|          7 |                        8 |                     30 |          8.64113 |              496.568 |           278.991 |      3.58435 |
|          8 |                        8 |                     30 |          8.62054 |              492.547 |           278.43  |      3.59157 |
|          9 |                        9 |                     30 |          9.66335 |             1480.6   |           280.304 |      3.56755 |
|         10 |                        7 |                     30 |          8.61555 |              474.229 |           278.803 |      3.58676 |


## STX B0 (With MCDM)

```python run_awq.py --model_name mistralai/Mistral-7B-v0.1  --task decode --target aie --algorithm pergrp```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        8 |                     30 |          6.86447 |              531.37  |           214.8   |      4.6555  |
|          2 |                       10 |                     30 |          7.28559 |              996.569 |           214.784 |      4.65583 |
|          3 |                        8 |                     30 |          6.66296 |              363.84  |           215.136 |      4.64823 |
|          4 |                        8 |                     30 |          6.64468 |              352.917 |           214.99  |      4.65139 |
|          5 |                        6 |                     30 |          6.6568  |              345.528 |           215.259 |      4.64556 |
|          6 |                        6 |                     30 |          6.62843 |              331.411 |           215.031 |      4.65049 |
|          7 |                        8 |                     30 |          6.65199 |              353.589 |           215.159 |      4.64772 |
|          8 |                        8 |                     30 |          6.66609 |              356.773 |           215.546 |      4.63939 |
|          9 |                        9 |                     30 |          7.28692 |              979.725 |           215.429 |      4.6419  |
|         10 |                        7 |                     30 |          6.6482  |              339.988 |           215.365 |      4.64328 |

```python run_awq.py --model_name mistralai/Mistral-7B-v0.1  --task decode --target aie --algorithm pergrp --flash_attention_plus```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        8 |                     30 |          6.4327  |              432.8   |           203.351 |      4.9176  |
|          2 |                       10 |                     30 |          6.94362 |              987.741 |           203.337 |      4.91794 |
|          3 |                        8 |                     30 |          6.28802 |              338.563 |           203.049 |      4.92493 |
|          4 |                        8 |                     30 |          6.28042 |              339.276 |           202.644 |      4.93477 |
|          5 |                        6 |                     30 |          6.27104 |              319.729 |           202.614 |      4.93549 |
|          6 |                        6 |                     30 |          6.25401 |              317.496 |           202.588 |      4.93613 |
|          7 |                        8 |                     30 |          6.27495 |              338.476 |           202.505 |      4.93815 |
|          8 |                        8 |                     30 |          6.29466 |              337.447 |           203.328 |      4.91815 |
|          9 |                        9 |                     30 |          6.9149  |              964.872 |           203.081 |      4.92415 |
|         10 |                        7 |                     30 |          6.26743 |              324.907 |           202.735 |      4.93254 |
