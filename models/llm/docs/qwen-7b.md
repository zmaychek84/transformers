# Qwen/Qwen-7b (PHX)
```
python run_awq.py --model_name Qwen/Qwen-7b --task quantize
```

# HPT (With MCDM)

``` python run_awq.py --model_name Qwen/Qwen-7b --task decode --target aie```
|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        7 |                     30 |          6.94296 |              428.556 |           199.423 |      5.01445 |
|          2 |                        8 |                     30 |          6.7998  |              337.251 |           198.854 |      5.02882 |
|          3 |                        6 |                     30 |          6.81763 |              321.358 |           199.773 |      5.00569 |
|          4 |                        7 |                     30 |          6.85191 |              327.008 |           199.735 |      5.00664 |
|          5 |                        5 |                     30 |          6.78756 |              309.607 |           198.716 |      5.0323  |
|          6 |                        4 |                     30 |          6.72518 |              296.822 |           198.027 |      5.04982 |
|          7 |                        7 |                     30 |          6.80183 |              322.918 |           199.395 |      5.01518 |
|          8 |                        6 |                     11 |          2.54284 |              314.967 |           196.749 |      5.08263 |
|          9 |                        6 |                      2 |          0.55249 |              318.556 |           192.108 |      5.2054  |
|         10 |                        6 |                     30 |          6.76535 |              318.981 |           198.419 |      5.03984 |

``` python run_awq.py --model_name Qwen/Qwen-7b --task decode --target aie --algorithm awqplus```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        7 |                     30 |         6.71209  |              539.551 |           187.019 |      5.34705 |
|          2 |                        8 |                     30 |         6.43113  |              333.751 |           185.842 |      5.38093 |
|          3 |                        6 |                     30 |         6.39586  |              302.001 |           186.495 |      5.36206 |
|          4 |                        7 |                     30 |         6.45003  |              317.883 |           185.943 |      5.37799 |
|          5 |                        5 |                     30 |         6.39201  |              300.795 |           185.842 |      5.38093 |
|          6 |                        4 |                     30 |         6.36375  |              284.578 |           185.404 |      5.39364 |
|          7 |                        7 |                     30 |         6.42219  |              316.634 |           186.463 |      5.36299 |
|          8 |                        6 |                     11 |         2.38502  |              303.351 |           182.056 |      5.49282 |
|          9 |                        6 |                      2 |         0.537163 |              304.098 |           178.127 |      5.61396 |
|         10 |                        6 |                     30 |         6.39505  |              307.22  |           186.033 |      5.37538 |

# PHX
``` python run_awq.py --model_name Qwen/Qwen-7b --task decode --target aie```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        7 |                     30 |         9.31064  |              508.875 |           280.85  |      3.56062 |
|          2 |                        8 |                     30 |         9.28386  |              462.927 |           282.145 |      3.54427 |
|          3 |                        6 |                     30 |         9.21257  |              439.417 |           280.189 |      3.56902 |
|          4 |                        7 |                     30 |         9.26553  |              447.704 |           280.962 |      3.5592  |
|          5 |                        5 |                     30 |         9.22965  |              429.432 |           280.432 |      3.56592 |
|          6 |                        4 |                     30 |         9.20774  |              419.504 |           281.432 |      3.55326 |
|          7 |                        7 |                     30 |         9.21056  |              446.907 |           280.51  |      3.56493 |
|          8 |                        6 |                     11 |         3.43808  |              430.544 |           276.685 |      3.61422 |
|          9 |                        6 |                      2 |         0.750846 |              439.599 |           271.116 |      3.68846 |
|         10 |                        6 |                     30 |         9.26614  |              430.667 |           282.531 |      3.53944 |

``` python run_awq.py --model_name Qwen/Qwen-7b --task decode --target aie --algorithm awqplus```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        7 |                     30 |         9.48458  |              628.649 |           280.728 |      3.56217 |
|          2 |                        8 |                     30 |         9.28417  |              463.664 |           280.941 |      3.55947 |
|          3 |                        6 |                     30 |         9.23674  |              436.275 |           280.221 |      3.56862 |
|          4 |                        7 |                     30 |         9.2631   |              445.459 |           279.274 |      3.58071 |
|          5 |                        5 |                     30 |         9.25379  |              422.49  |           280.567 |      3.56422 |
|          6 |                        4 |                     30 |         9.26079  |              422.329 |           280.85  |      3.56062 |
|          7 |                        7 |                     30 |         9.20571  |              444.23  |           278.881 |      3.58576 |
|          8 |                        6 |                     11 |         3.43328  |              433.537 |           273.652 |      3.65427 |
|          9 |                        6 |                      2 |         0.753254 |              440.536 |           267.342 |      3.74053 |
|         10 |                        6 |                     30 |        20.0338   |              436.406 |           279.11  |      3.58281 |

# STX B0 (with MCDM)
``` python run_awq.py --model_name Qwen/Qwen-7b --task decode --target aie```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        7 |                     30 |         8.55218  |              459.83  |           252.535 |      3.95985 |
|          2 |                        8 |                     30 |         8.45062  |              338.185 |           254.147 |      3.93473 |
|          3 |                        6 |                     30 |         8.29516  |              300.139 |           250.413 |      3.99341 |
|          4 |                        7 |                     30 |         8.4466   |              316.387 |           254.007 |      3.9369  |
|          5 |                        5 |                     30 |         8.26613  |              296.515 |           249.338 |      4.01062 |
|          6 |                        4 |                     30 |         8.1474   |              270.444 |           246.814 |      4.05163 |
|          7 |                        7 |                     30 |         8.37915  |              320.353 |           252.644 |      3.95814 |
|          8 |                        6 |                     11 |         2.91615  |              301.692 |           234.196 |      4.26993 |
|          9 |                        6 |                      2 |         0.583068 |              299.407 |           224.934 |      4.44575 |
|         10 |                        6 |                     30 |         8.19469  |              300.865 |           246.766 |      4.05242 |


``` python run_awq.py --model_name Qwen/Qwen-7b --task decode --target aie --algorithm awqplus```

Loss of accuracy must be evaluated

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        7 |                     30 |         6.2705   |              537.778 |           171.42  |      5.83363 |
|          2 |                        8 |                     30 |         6.04829  |              263.23  |           174.283 |      5.7378  |
|          3 |                        6 |                     30 |         5.86362  |              223.608 |           169.43  |      5.90214 |
|          4 |                        7 |                     30 |         5.96826  |              239.004 |           171.533 |      5.82977 |
|          5 |                        5 |                     30 |         5.84547  |              215.017 |           169.167 |      5.91131 |
|          6 |                        4 |                     30 |         5.80654  |              206.681 |           168.363 |      5.93954 |
|          7 |                        7 |                     30 |         5.97061  |              243.704 |           172.613 |      5.79331 |
|          8 |                        6 |                     11 |         2.01906  |              226.305 |           152.518 |      6.55662 |
|          9 |                        6 |                      2 |         0.418949 |              230.255 |           140.583 |      7.11323 |
|         10 |                        6 |                     30 |         5.86589  |              226.438 |           169.758 |      5.89073 |

```
QWenLMHeadModel(
  (transformer): QWenModel(
    (wte): Embedding(151936, 4096)
    (drop): Dropout(p=0.0, inplace=False)
    (rotary_emb): RotaryEmbedding()
    (h): ModuleList(
      (0-31): 32 x QWenBlock(
        (ln_1): RMSNorm()
        (attn): QWenAttention(
          (c_attn): ryzenAI.QLinearPerGrp(in_features:4096, out_features:12288, bias:torch.Size([12288]), device:aie, w_bit:4 group_size:128  )
          (c_proj): ryzenAI.QLinearPerGrp(in_features:4096, out_features:4096, bias:None, device:aie, w_bit:4 group_size:128  )
          (attn_dropout): Dropout(p=0.0, inplace=False)
        )
        (ln_2): RMSNorm()
        (mlp): QWenMLP(
          (w1): ryzenAI.QLinearPerGrp(in_features:4096, out_features:11008, bias:None, device:aie, w_bit:4 group_size:128  )
          (w2): ryzenAI.QLinearPerGrp(in_features:4096, out_features:11008, bias:None, device:aie, w_bit:4 group_size:128  )
          (c_proj): ryzenAI.QLinearPerGrp(in_features:11008, out_features:4096, bias:None, device:aie, w_bit:4 group_size:128  )
        )
      )
    )
    (ln_f): RMSNorm()
  )
  (lm_head): Linear(in_features=4096, out_features=151936, bias=False)
)
model.mode_name: Qwen-7b
****************************************
prompt: What is the meaning of life?
C:\Users\rajeevp\AppData\Local\anaconda3\envs\ryzenai-transformers\lib\site-packages\transformers\generation\configuration_utils.py:415: UserWarning: `do_sample` is set to `False`. However, `top_p` is set to `0.8` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_p`.
  warnings.warn(
C:\Users\rajeevp\AppData\Local\anaconda3\envs\ryzenai-transformers\lib\site-packages\transformers\generation\configuration_utils.py:427: UserWarning: `do_sample` is set to `False`. However, `top_k` is set to `0` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_k`.
  warnings.warn(
response: What is the meaning of life? What is the meaning of life? What is the meaning of life? What is the meaning of life? What is the meaning of life? What is
****************************************
prompt: Tell me something you don't know.
response: Tell me something you don't know. I want to know something that I don't know. I want to know something that you don't know. I want to know something that you don
****************************************
prompt: What does Xilinx do?
response: What does Xilinx do? What is the difference between FPGA and ASIC? What is the difference between FPGA and CPLD? What is the difference between FPGA and FPGAs?
****************************************
...
...
```
