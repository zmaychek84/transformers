# OPT 1.3b - w4abf16 with AWQ

# Save AWQ checkpoints
4-bit AWQ has higher perplexity than 3-bit AWQ with same performance. But both are available in RyzenAI Transformers stack.

```python run_awq.py --model_name facebook/opt-1.3b --task quantize ```

* User can also use --w_bit 3 other quantization algorithms - check ```python run_awq.py --help``` for details

# Decode prompts (HPT)
```
OPTModelEval(
  (model): OPTModel(
    (decoder): OPTDecoder(
      (embed_tokens): Embedding(50272, 2048, padding_idx=1)
      (embed_positions): OPTLearnedPositionalEmbedding(2050, 2048)
      (final_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
      (layers): ModuleList(
        (0-23): 24 x OPTDecoderLayer(
          (self_attn): OPTFlashAttentionPlus(
            (out_proj): ryzenAI.QLinearPerGrp(in_features:2048, out_features:2048, bias:torch.Size([2048]), device:aie, w_bit:4 group_size:128  )
            (qkv_proj): ryzenAI.QLinearPerGrp(in_features:2048, out_features:6144, bias:torch.Size([6144]), device:aie, w_bit:4 group_size:128  )
          )
          (activation_fn): ReLU()
          (self_attn_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
          (fc1): ryzenAI.QLinearPerGrp(in_features:2048, out_features:8192, bias:torch.Size([8192]), device:aie, w_bit:4 group_size:128  )
          (fc2): ryzenAI.QLinearPerGrp(in_features:8192, out_features:2048, bias:torch.Size([2048]), device:aie, w_bit:4 group_size:128  )
          (final_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
        )
      )
    )
  )
  (lm_head): Linear(in_features=2048, out_features=50272, bias=False)
)
model.mode_name: opt-1.3b
python run_awq.py --model_name opt-1.3b --task decode
...
...
****************************************
prompt: What does Xilinx do?
response: What does Xilinx do?
Xilinx is a company that makes chips for the internet of things.
****************************************
prompt: What is the mass of earth?
response: What is the mass of earth?

The mass of earth is the mass of the earth. The mass of earth is the mass of the earth. The mass of earth is the
****************************************
...
```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        8 |                     30 |          3.37866 |              194.013 |           106.755 |      9.3672  |
|          2 |                        9 |                     30 |          3.63444 |              459.202 |           106.403 |      9.39821 |
|          3 |                        8 |                     30 |          3.33381 |              167.136 |           106.15  |      9.42062 |
|          4 |                        8 |                     30 |          3.33496 |              163.619 |           106.564 |      9.38406 |
|          5 |                        6 |                     30 |          3.3219  |              157.763 |           106.255 |      9.41132 |
|          6 |                        6 |                     30 |          3.33737 |              156.909 |           106.851 |      9.35882 |
|          7 |                        8 |                     30 |          3.32545 |              160.93  |           106.381 |      9.4002  |
|          8 |                        7 |                     30 |          3.33959 |              166.967 |           106.302 |      9.40713 |
|          9 |                        7 |                     27 |          2.99892 |              162.844 |           106.222 |      9.41421 |
|         10 |                        7 |                     30 |          3.32339 |              159.641 |           105.882 |      9.44447 |


# HPT (With MCDM)

```python run_awq.py --model_name facebook/opt-1.3b  --target aie --task benchmark```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        4 |                     30 |          3.45821 |              174.943 |           110.204 |      9.07409 |
|          2 |                        8 |                     30 |          3.4598  |              172.317 |           110.466 |      9.05258 |
|          3 |                       16 |                     30 |          3.76375 |              464.376 |           110.763 |      9.02828 |
|          4 |                       32 |                     30 |          3.81014 |              496.928 |           111.123 |      8.99908 |
|          5 |                       64 |                     30 |          4.29665 |              947.207 |           112.041 |      8.92531 |
|          6 |                      128 |                     30 |          5.32397 |             1921.36  |           113.589 |      8.80365 |
|          7 |                      256 |                     30 |          7.29049 |             3803.23  |           116.232 |      8.60347 |

```python run_awq.py --model_name facebook/opt-1.3b  --target aie --task benchmark_long```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                      512 |                     30 |          11.668  |              7901.41 |           124.004 |      8.06424 |
|          2 |                     1024 |                     30 |          21.6392 |             17401.3  |           137.388 |      7.27868 |
|          3 |                     1536 |                     30 |          34.2581 |             29480.7  |           153.333 |      6.52176 |
|          4 |                     2000 |                     30 |          46.7615 |             41739.5  |           159.219 |      6.28065 |

```python run_awq.py --model_name facebook/opt-1.3b  --target aie --task benchmark --flash_attention_plus```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        4 |                     30 |          3.12606 |              158.239 |           99.3147 |     10.069   |
|          2 |                        8 |                     30 |          3.12317 |              163.101 |           99.1388 |     10.0869  |
|          3 |                       16 |                     30 |          3.4014  |              446.638 |           98.8752 |     10.1138  |
|          4 |                       32 |                     30 |          3.42562 |              464.161 |           98.9977 |     10.1012  |
|          5 |                       64 |                     30 |          3.89048 |              894.989 |           99.7702 |     10.023   |
|          6 |                      128 |                     30 |          4.82078 |             1799.56  |          100.477  |      9.95253 |
|          7 |                      256 |                     30 |          6.69605 |             3610.7   |          102.325  |      9.77277 |

```python run_awq.py --model_name facebook/opt-1.3b  --target aie --task benchmark_long --flash_attention_plus```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                      512 |                     30 |          10.8566 |               7536.6 |           108.912 |      9.18172 |
|          2 |                     1024 |                     30 |          20.0321 |              16228.2 |           122.841 |      8.14063 |
|          3 |                     1536 |                     30 |          30.8768 |              26627.1 |           135.033 |      7.40559 |
|          4 |                     2000 |                     30 |          41.77   |              37254.8 |           142.1   |      7.03729 |

# PHX

```python run_awq.py --model_name facebook/opt-1.3b --target aie --task benchmark```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        4 |                     30 |          5.09741 |              262.9   |           163.816 |      6.10443 |
|          2 |                        8 |                     30 |          5.08023 |              252     |           163.886 |      6.1018  |
|          3 |                       16 |                     30 |          5.52331 |              692.763 |           163.805 |      6.10481 |
|          4 |                       32 |                     30 |          5.53949 |              711.31  |           163.64  |      6.11098 |
|          5 |                       64 |                     30 |          6.25441 |             1389.81  |           164.647 |      6.07359 |
|          6 |                      128 |                     30 |          7.60549 |             2759.77  |           163.831 |      6.10384 |
|          7 |                      256 |                     30 |         10.5628  |             5581.48  |           168.106 |      5.94863 |

```python run_awq.py --model_name facebook/opt-1.3b --target aie --task benchmark_long```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                      512 |                     30 |          17.073  |              11821   |           176.454 |      5.66719 |
|          2 |                     1024 |                     30 |          31.2634 |              25570.3 |           189.668 |      5.27236 |
|          3 |                     1536 |                     30 |          48.2039 |              42035.9 |           203.866 |      4.90519 |
|          4 |                     2000 |                     30 |          66.0078 |              59397.7 |           216.907 |      4.61028 |

```python run_awq.py --model_name facebook/opt-1.3b --target aie --task benchmark --flash_attention_plus```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        4 |                     30 |          4.51304 |              232.633 |           144.875 |      6.90252 |
|          2 |                        8 |                     30 |          4.50458 |              225.505 |           144.75  |      6.90844 |
|          3 |                       16 |                     30 |          4.92931 |              658.901 |           144.495 |      6.92066 |
|          4 |                       32 |                     30 |          4.93515 |              686.131 |           143.725 |      6.95771 |
|          5 |                       64 |                     30 |          5.59544 |             1315.73  |           144.497 |      6.92058 |
|          6 |                      128 |                     30 |          6.95512 |             2642.98  |           145.369 |      6.87907 |
|          7 |                      256 |                     30 |          9.6794  |             5304.45  |           147.327 |      6.78761 |

```python run_awq.py --model_name facebook/opt-1.3b --target aie --task benchmark_long --flash_attention_plus```
|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                      512 |                     30 |          15.5103 |              10907.7 |           154.214 |      6.48451 |
|          2 |                     1024 |                     30 |          27.5258 |              22549.4 |           165.088 |      6.05736 |
|          3 |                     1536 |                     30 |          41.4275 |              35990.5 |           178.569 |      5.60008 |
|          4 |                     2000 |                     30 |          55.4219 |              49659   |           187.449 |      5.33478 |

# STX B0 (With MCDM)

```python run_awq.py --model_name facebook/opt-1.3b  --target aie --task benchmark```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        4 |                     30 |          5.16462 |              692.255 |           145.458 |      6.87485 |
|          2 |                        8 |                     30 |          5.16665 |              701.142 |           147.39  |      6.78471 |
|          3 |                       16 |                     30 |          5.3793  |              998.057 |           142.08  |      7.03827 |
|          4 |                       32 |                     30 |          5.48268 |             1164.15  |           139.446 |      7.17122 |
|          5 |                       64 |                     30 |          5.87042 |             1928.99  |           128.204 |      7.80006 |
|          6 |                      128 |                     30 |          6.41016 |             3088.26  |           109.281 |      9.15075 |
|          7 |                      256 |                     30 |          8.50768 |             5276.5   |           106.112 |      9.424   |

```python run_awq.py --model_name facebook/opt-1.3b  --target aie --task benchmark_long```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                      512 |                     30 |          14.3554 |              10681.4 |           119.724 |      8.35252 |
|          2 |                     1024 |                     30 |          31.8507 |              27309.7 |           146.935 |      6.80574 |
|          3 |                     1536 |                     30 |          55.5624 |              50323.7 |           168.79  |      5.92452 |
|          4 |                     2000 |                     30 |          83.2899 |              77175.8 |           195.937 |      5.10368 |

```python run_awq.py --model_name facebook/opt-1.3b  --target aie --task benchmark --flash_attention_plus```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        4 |                     30 |          4.9442  |              674.616 |          136.621  |      7.31952 |
|          2 |                        8 |                     30 |          4.9525  |              692.041 |          139.569  |      7.16492 |
|          3 |                       16 |                     30 |          5.07573 |              911.659 |          133.978  |      7.46393 |
|          4 |                       32 |                     30 |          5.26818 |             1117.41  |          136.51   |      7.32546 |
|          5 |                       64 |                     30 |          5.58084 |             1851.91  |          123.183  |      8.11801 |
|          6 |                      128 |                     30 |          6.07394 |             2876.93  |          105.43   |      9.48493 |
|          7 |                      256 |                     30 |          7.84216 |             4864.43  |           97.8981 |     10.2147  |

```python run_awq.py --model_name facebook/opt-1.3b  --target aie --task benchmark_long --flash_attention_plus```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                      512 |                     30 |          13.2382 |              9919.15 |           108.248 |      9.23809 |
|          2 |                     1024 |                     30 |          27.9775 |             23864.6  |           132.352 |      7.55561 |
|          3 |                     1536 |                     30 |          48.5763 |             43618.3  |           158.409 |      6.31277 |
|          4 |                     2000 |                     30 |          71.7531 |             66181.4  |           177.491 |      5.6341  |
