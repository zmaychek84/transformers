# THUDM/chatglm-6b

## Quantize and save ChatGLM model

```python run_awq.py --model_name THUDM/chatglm-6b --task quantize --algorithm pergrp```

ChatGLM seems to perform well with just pergrp quantization (without AWQ)
```
python run_awq.py --model_name THUDM/chatglm-6b --task quantize --algorithm pergrp

...
...
ChatGLMForConditionalGeneration(
  (transformer): ChatGLMModel(
    (word_embeddings): Embedding(130528, 4096)
    (layers): ModuleList(
      (0-27): 28 x GLMBlock(
        (input_layernorm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        (attention): SelfAttention(
          (rotary_emb): RotaryEmbedding()
          (query_key_value): ryzenAI.QLinearPerGrp(in_features:4096, out_features:12288, bias:torch.Size([12288]), device:aie, w_bit:4 group_size:128  )
          (dense): ryzenAI.QLinearPerGrp(in_features:4096, out_features:4096, bias:torch.Size([4096]), device:aie, w_bit:4 group_size:128  )
        )
        (post_attention_layernorm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        (mlp): GLU(
          (dense_h_to_4h): ryzenAI.QLinearPerGrp(in_features:4096, out_features:16384, bias:torch.Size([16384]), device:aie, w_bit:4 group_size:128  )
          (dense_4h_to_h): ryzenAI.QLinearPerGrp(in_features:16384, out_features:4096, bias:torch.Size([4096]), device:aie, w_bit:4 group_size:128  )
        )
      )
    )
    (final_layernorm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): ryzenAI.QLinearPerGrp(in_features:4096, out_features:130528, bias:None, device:aie, w_bit:4 group_size:128  )
)

```
## HPT (With MCDM)

```python run_awq.py --model_name THUDM/chatglm-6b --task decode --target aie --algorithm pergrp```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        9 |                     30 |          8.35488 |              1693.63 |           222.957 |      4.48516 |
|          2 |                       11 |                     30 |          8.33576 |              1929.29 |           215.191 |      4.64703 |
|          3 |                        8 |                     30 |          7.82845 |              1417.8  |           215.406 |      4.6424  |
|          4 |                        9 |                     22 |          6.17137 |              1560.69 |           213.403 |      4.68597 |
|          5 |                        7 |                     30 |          7.61771 |              1236.45 |           214.334 |      4.66561 |
|          6 |                        6 |                     30 |          7.49355 |              1074.63 |           215.552 |      4.63924 |
|          7 |                        9 |                     17 |          5.25898 |              1616.68 |           220.688 |      4.53128 |
|          8 |                        8 |                     30 |          8.01863 |              1427.85 |           220.745 |      4.53011 |
|          9 |                        8 |                     30 |          8.06573 |              1454.99 |           222.062 |      4.50325 |
|         10 |                        8 |                     30 |          7.9704  |              1447.19 |           218.662 |      4.57328 |

## PHX

```python run_awq.py --model_name THUDM/chatglm-6b --task decode --target aie --algorithm pergrp```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        9 |                     30 |         11.8949  |              2499.94 |           318.131 |      3.14336 |
|          2 |                       11 |                     30 |         12.2708  |              2943.75 |           316.156 |      3.163   |
|          3 |                        8 |                     30 |         11.5183  |              2196.5  |           316.11  |      3.16345 |
|          4 |                        9 |                     22 |          9.14882 |              2432.99 |           314.442 |      3.18024 |
|          5 |                        7 |                     30 |         11.234   |              1906.68 |           315.92  |      3.16536 |
|          6 |                        6 |                     30 |         10.9389  |              1668.72 |           314.296 |      3.18172 |
|          7 |                        9 |                     17 |          7.56684 |              2444.2  |           314.622 |      3.17842 |
|          8 |                        8 |                     30 |         11.4993  |              2184.53 |           315.955 |      3.16501 |
|          9 |                        8 |                     30 |         11.5306  |              2192.63 |           316.697 |      3.15759 |
|         10 |                        8 |                     30 |         11.5638  |              2197.1  |           317.728 |      3.14734 |

## STX B0 (With MCDM)

```python run_awq.py --model_name THUDM/chatglm-6b --task decode --target aie --algorithm pergrp```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        9 |                     30 |          6.6737  |             1315.26  |           176.105 |      5.67844 |
|          2 |                       11 |                     30 |          6.78339 |             1542.5   |           173.489 |      5.76404 |
|          3 |                        8 |                     30 |          6.37055 |             1152.27  |           172.651 |      5.79202 |
|          4 |                        9 |                     22 |          5.03785 |             1279.34  |           171.627 |      5.82659 |
|          5 |                        7 |                     30 |          6.20285 |             1006.46  |           172.006 |      5.81374 |
|          6 |                        6 |                     30 |          6.0507  |              856.113 |           171.676 |      5.82494 |
|          7 |                        9 |                     17 |          4.14807 |             1288.97  |           171.339 |      5.83638 |
|          8 |                        8 |                     30 |          6.36958 |             1147.17  |           172.63  |      5.79273 |
|          9 |                        8 |                     30 |          6.36194 |             1148.95  |           172.587 |      5.79418 |
|         10 |                        8 |                     30 |          6.35931 |             1149.67  |           172.514 |      5.79664 |

## Benchmark
Benchmarking with wikitext2 will not work for this model with current dataset. It will error out with following message.

```Token indices sequence length is longer than the specified maximum sequence length for this model (2766242 > 2048). Running this sequence through the model will result in indexing errors```
