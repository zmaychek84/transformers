# microsoft/Phi-3-mini-4k-instruct

Quantize and save model

```python run_awq.py --model_name microsoft/Phi-3-mini-4k-instruct --task decode --algorithm pergrp```

# HPT (With MCDM)

```python run_awq.py --model_name microsoft/Phi-3-mini-4k-instruct --task decode --algorithm pergrp```

# PHX

```python run_awq.py --model_name microsoft/Phi-3-mini-4k-instruct --task decode --algorithm pergrp```

# STX B0 (With MCDM) - 5/16/2024

```python run_awq.py --model_name microsoft/Phi-3-mini-4k-instruct --task decode --algorithm pergrp```
|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                       12 |                     60 |         10.7132  |              674.762 |           166.652 |      6.00055 |
|          2 |                       14 |                     60 |         10.5737  |              611.239 |           165.394 |      6.04618 |
|          3 |                       12 |                     60 |         10.5439  |              594.581 |           165.375 |      6.04688 |
|          4 |                       12 |                     60 |         10.5447  |              593.764 |           165.262 |      6.051   |
|          5 |                       10 |                     60 |         10.5413  |              611.718 |           165.056 |      6.05856 |
|          6 |                        9 |                     60 |         10.5376  |              592.922 |           165.19  |      6.05365 |
|          7 |                       13 |                     22 |          4.10856 |              589.114 |           164.207 |      6.08988 |
|          8 |                       12 |                     60 |         10.5358  |              589.256 |           165.168 |      6.05445 |
|          9 |                       13 |                     60 |         10.5382  |              593.916 |           165.021 |      6.05984 |
|         10 |                       11 |                     60 |         10.6058  |              583.469 |           166.467 |      6.00721 |

```python run_awq.py --model_name microsoft/Phi-3-mini-4k-instruct --task benchmark --algorithm pergrp```
|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                       55 |                     30 |          6.331   |              1290.5  |           170.216 |      5.87488 |
|          2 |                       59 |                     30 |          6.29605 |              1306.84 |           168.873 |      5.9216  |
|          3 |                       67 |                     30 |          6.44815 |              1426.65 |           169.796 |      5.88941 |
|          4 |                       83 |                     30 |          6.91129 |              1869.83 |           170.538 |      5.86381 |
|          5 |                       88 |                     30 |          6.69015 |              1785.67 |           166.087 |      6.02095 |
|          6 |                      115 |                     30 |          7.28126 |              2382.42 |           166.391 |      6.00994 |
|          7 |                      128 |                     30 |          7.48967 |              2519.17 |           167.435 |      5.97245 |
|          8 |                      256 |                     30 |         10.1932  |              5046.86 |           174.859 |      5.71889 |


```
Phi3ModelEval(
  (model): Phi3Model(
    (embed_tokens): Embedding(32064, 3072, padding_idx=32000)
    (embed_dropout): Dropout(p=0.0, inplace=False)
    (layers): ModuleList(
      (0-31): 32 x Phi3DecoderLayer(
        (self_attn): Phi3Attention(
          (o_proj): ryzenAI.QLinearPerGrp(in_features:3072, out_features:3072, bias:None, device:aie, w_bit:4 group_size:128  )
          (qkv_proj): ryzenAI.QLinearPerGrp(in_features:3072, out_features:9216, bias:None, device:aie, w_bit:4 group_size:128  )
          (rotary_emb): Phi3RotaryEmbedding()
        )
        (mlp): Phi3MLP(
          (gate_up_proj): ryzenAI.QLinearPerGrp(in_features:3072, out_features:16384, bias:None, device:aie, w_bit:4 group_size:128  )
          (down_proj): ryzenAI.QLinearPerGrp(in_features:8192, out_features:3072, bias:None, device:aie, w_bit:4 group_size:128  )
          (activation_fn): SiLU()
        )
        (input_layernorm): Phi3RMSNorm()
        (resid_attn_dropout): Dropout(p=0.0, inplace=False)
        (resid_mlp_dropout): Dropout(p=0.0, inplace=False)
        (post_attention_layernorm): Phi3RMSNorm()
      )
    )
    (norm): Phi3RMSNorm()
  )
  (lm_head): ryzenAI.QLinearPerGrp(in_features:3072, out_features:32064, bias:None, device:aie, w_bit:4 group_size:128  )
)
model.mode_name: Phi-3-mini-4k-instruct
****************************************
<s><s><|user|> What is the meaning of life?<|end|><|assistant|> While the question of the meaning of life can be deeply philosophical and subjective, it's crucial to remember that its interpretation can vary greatly among individuals. Some philosophical perspectives suggest that life's meaning is a personal journey of self-discovery and growth. Here'
****************************************
<s><s><|user|> Tell me something you don't know.<|end|><|assistant|> As an AI, I don't have personal knowledge and don't possess beliefs or knowledge like a human. However, I can provide information on a wide range of topics! One example of something I don't know personally is new information or developments occurring beyond my last training
****************************************
<s><s><|user|> What does Xilinx do?<|end|><|assistant|> Xilinx is a prominent and globally recognized American company in the field of semiconductor design and integrated circuits, particularly known for its expertise in Programmable Logic Devices (PLDs) and Field-Programmable Gate Arrays (FPGAs). F
****************************************```
