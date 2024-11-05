# OPT 6.7b - w4abf16 with AWQ

# Save AWQ checkpoints
4-bit AWQ has higher perplexity than 3-bit AWQ with same performance. But both are available in RyzenAI Transformers stack.

```python run_awq.py --model_name facebook/opt-6.7b --task quantize ```

* User can also use --w_bit 3 other quantization algorithms - check ```python run_awq.py --help``` for details

```
OPTModelEval(
  (model): OPTModel(
    (decoder): OPTDecoder(
      (embed_tokens): Embedding(50272, 4096, padding_idx=1)
      (embed_positions): OPTLearnedPositionalEmbedding(2050, 4096)
      (final_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
      (layers): ModuleList(
        (0-31): 32 x OPTDecoderLayer(
          (self_attn): OPTFlashAttentionPlus(
            (out_proj): ryzenAI.QLinearPerGrp(in_features:4096, out_features:4096, bias:torch.Size([4096]), device:aie, w_bit:4 group_size:128  )
            (qkv_proj): ryzenAI.QLinearPerGrp(in_features:4096, out_features:12288, bias:torch.Size([12288]), device:aie, w_bit:4 group_size:128  )
          )
          (activation_fn): ReLU()
          (self_attn_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
          (fc1): ryzenAI.QLinearPerGrp(in_features:4096, out_features:16384, bias:torch.Size([16384]), device:aie, w_bit:4 group_size:128  )
          (fc2): ryzenAI.QLinearPerGrp(in_features:16384, out_features:4096, bias:torch.Size([4096]), device:aie, w_bit:4 group_size:128  )
          (final_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        )
      )
    )
  )
  (lm_head): Linear(in_features=4096, out_features=50272, bias=False)
)
model.mode_name: opt-6.7b
****************************************

```

# HPT (With MCDM)

```python run_awq.py --model_name facebook/opt-6.7b  --target aie --task benchmark```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        4 |                     30 |          7.01876 |              372.646 |           226.061 |      4.42358 |
|          2 |                        8 |                     30 |          6.94211 |              373.342 |           223.341 |      4.47746 |
|          3 |                       16 |                     30 |          7.6057  |             1072.17  |           221.798 |      4.50861 |
|          4 |                       32 |                     30 |          7.70551 |             1122.01  |           223.356 |      4.47716 |
|          5 |                       64 |                     30 |          8.84058 |             2202.24  |           225.278 |      4.43896 |
|          6 |                      128 |                     30 |         11.235   |             4382.96  |           232.33  |      4.30422 |
|          7 |                      256 |                     30 |         15.809   |             8778.15  |           238.465 |      4.19349 |

```python run_awq.py --model_name facebook/opt-6.7b  --target aie --task benchmark_long```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                      512 |                     30 |          25.7872 |              18195.3 |           256.067 |      3.90522 |
|          2 |                     1024 |                     30 |          49.9635 |              41261.4 |           291.166 |      3.43446 |
|          3 |                     1536 |                     30 |          73.0037 |              63476.5 |           317.413 |      3.15047 |
|          4 |                     2000 |                     30 |          98.036  |              87752   |           339.564 |      2.94495 |

```python run_awq.py --model_name facebook/opt-6.7b  --target aie --task benchmark --flash_attention_plus```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        4 |                     30 |          6.45793 |              359.569 |           207.249 |      4.82512 |
|          2 |                        8 |                     30 |          6.46264 |              353.471 |           207.544 |      4.81825 |
|          3 |                       16 |                     30 |          7.10808 |             1046.79  |           205.681 |      4.86191 |
|          4 |                       32 |                     30 |          7.19863 |             1093.81  |           207.047 |      4.82982 |
|          5 |                       64 |                     30 |          8.27773 |             2152.16  |           207.58  |      4.81741 |
|          6 |                      128 |                     30 |         10.6014  |             4276.42  |           214.008 |      4.67273 |
|          7 |                      256 |                     30 |         15.2886  |             8747.19  |           221.429 |      4.51611 |

```python run_awq.py --model_name facebook/opt-6.7b  --target aie --task benchmark_long --flash_attention_plus```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                      512 |                     30 |          24.9555 |              17871.3 |           238.224 |      4.19773 |
|          2 |                     1024 |                     30 |          45.5863 |              37519.3 |           269.087 |      3.71628 |
|          3 |                     1536 |                     30 |          68.9136 |              59776   |           301.842 |      3.31299 |
|          4 |                     2000 |                     30 |          91.6292 |              81818.5 |           321.658 |      3.10889 |

# PHX

```python run_awq.py --model_name facebook/opt-6.7b  --target aie --task benchmark```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        4 |                     30 |          10.7493 |              555.839 |           348.768 |      2.86723 |
|          2 |                        8 |                     30 |          10.7355 |              546.272 |           348.664 |      2.86809 |
|          3 |                       16 |                     30 |          11.8015 |             1635.02  |           347.652 |      2.87644 |
|          4 |                       32 |                     30 |          11.8615 |             1684.81  |           347.799 |      2.87523 |
|          5 |                       64 |                     30 |          13.5206 |             3281.96  |           349.931 |      2.85771 |
|          6 |                      128 |                     30 |          16.8685 |             6524.02  |           353.19  |      2.83133 |
|          7 |                      256 |                     30 |          23.8328 |            13256.4   |           361.039 |      2.76978 |

```python run_awq.py --model_name facebook/opt-6.7b  --target aie --task benchmark_long```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                      512 |                     30 |          38.3094 |              27157.7 |           379.493 |      2.63509 |
|          2 |                     1024 |                     30 |          70.4689 |              58600.1 |           402.269 |      2.4859  |
|          3 |                     1536 |                     30 |         105.744  |              92964.3 |           432.117 |      2.31419 |
|          4 |                     2000 |                     30 |         141.162  |             127465   |           460.266 |      2.17266 |

```python run_awq.py --model_name facebook/opt-6.7b --target aie --task benchmark --flash_attention_plus```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        4 |                     30 |          9.55983 |              514.964 |           309.341 |      3.23268 |
|          2 |                        8 |                     30 |          9.51499 |              504.6   |           308.084 |      3.24587 |
|          3 |                       16 |                     30 |         10.6312  |             1577.47  |           309.406 |      3.232   |
|          4 |                       32 |                     30 |         10.7735  |             1621.67  |           312.557 |      3.19942 |
|          5 |                       64 |                     30 |         12.3953  |             3199.99  |           313.878 |      3.18595 |
|          6 |                      128 |                     30 |         15.558   |             6327.65  |           314.597 |      3.17867 |
|          7 |                      256 |                     30 |         22.4152  |            12900.7   |           324.571 |      3.08099 |

```python run_awq.py --model_name facebook/opt-6.7b --target aie --task benchmark_long --flash_attention_plus```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                      512 |                     30 |          36.8034 |              26545.3 |           348.284 |      2.87122 |
|          2 |                     1024 |                     30 |          67.0829 |              55723.2 |           383.632 |      2.60666 |
|          3 |                     1536 |                     30 |          97.7637 |              85953.5 |           396.724 |      2.52064 |
|          4 |                     2000 |                     30 |         129.409  |             116754   |           422.458 |      2.3671  |

# STX B0 (With MCDM)

```python run_awq.py --model_name facebook/opt-6.7b  --target aie --task benchmark```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        4 |                     30 |          7.75751 |              929.491 |           229.099 |      4.36493 |
|          2 |                        8 |                     30 |          7.7276  |              903.673 |           228.282 |      4.38055 |
|          3 |                       16 |                     30 |          8.22397 |             1830.96  |           211.783 |      4.7218  |
|          4 |                       32 |                     30 |          8.42609 |             2052.66  |           214.543 |      4.66107 |
|          5 |                       64 |                     30 |          9.03414 |             3013.25  |           202.8   |      4.93096 |
|          6 |                      128 |                     30 |         10.8329  |             4858.48  |           200.778 |      4.98064 |
|          7 |                      256 |                     30 |         17.7703  |             9850.46  |           267.575 |      3.73726 |

```python run_awq.py --model_name facebook/opt-6.7b  --target aie --task benchmark_long```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                      512 |                     30 |          26.5756 |              19152.6 |           249.131 |      4.01396 |
|          2 |                     1024 |                     30 |          56.1293 |              46800.4 |           311.354 |      3.21178 |
|          3 |                     1536 |                     30 |          97.6922 |              86601.3 |           369.838 |      2.70389 |
|          4 |                     2000 |                     30 |         143.947  |             130843   |           434.553 |      2.30121 |

```python run_awq.py --model_name facebook/opt-6.7b  --target aie --task benchmark --flash_attention_plus```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        4 |                     30 |          7.35777 |              883.905 |           212.261 |      4.71117 |
|          2 |                        8 |                     30 |          7.48341 |              898.459 |           218.921 |      4.56785 |
|          3 |                       16 |                     30 |          7.84366 |             1449.23  |           215.184 |      4.64719 |
|          4 |                       32 |                     30 |          7.94948 |             1690.11  |           207.153 |      4.82735 |
|          5 |                       64 |                     30 |          8.68188 |             2862.69  |           196.118 |      5.09897 |
|          6 |                      128 |                     30 |         10.2976  |             4706.9   |           187.946 |      5.32067 |
|          7 |                      256 |                     30 |         14.4656  |             8380.98  |           204.907 |      4.88025 |

```python run_awq.py --model_name facebook/opt-6.7b  --target aie --task benchmark_long --flash_attention_plus```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                      512 |                     30 |          30.93   |              18692.4 |           239.239 |      4.17992 |
|          2 |                     1024 |                     30 |          51.4779 |              42405.4 |           301.747 |      3.31403 |
|          3 |                     1536 |                     30 |          84.1433 |              73303   |           357.913 |      2.79397 |
|          4 |                     2000 |                     30 |         120.246  |             107221   |           425.608 |      2.34958 |
