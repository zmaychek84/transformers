# THUDM/chatglm-6b

## Prerequisites
```
# ChatGLM is supported with transformers version 4.37.2 currently and not with 4.44.2
pip install transformers==4.37.2

# ChatGLM works with non optimized NPU kernels,
set MLADF=
```

## Quantize and save ChatGLM3 model

```python run_awq.py --model_name THUDM/chatglm3-6b --task quantize --algorithm pergrp --group_size 32```

```python run_awq.py --model_name THUDM/chatglm3-6b --task quantize --algorithm pergrp --group_size 128```

```
...
...
ChatGLMForConditionalGeneration(
  (transformer): ChatGLMModel(
    (embedding): Embedding(
      (word_embeddings): Embedding(65024, 4096)
    )
    (rotary_pos_emb): RotaryEmbedding()
    (encoder): GLMTransformer(
      (layers): ModuleList(
        (0-27): 28 x GLMBlock(
          (input_layernorm): RMSNorm()
          (self_attention): SelfAttention(
            (query_key_value): ryzenAI.QLinearPerGrp(in_features:4096, out_features:4608, bias:torch.Size([1]), device:cpu, w_bit:4 group_size:128  )
            (core_attention): CoreAttention(
              (attention_dropout): Dropout(p=0.0, inplace=False)
            )
            (dense): ryzenAI.QLinearPerGrp(in_features:4096, out_features:4096, bias:torch.Size([1]), device:cpu, w_bit:4 group_size:128  )
          )
          (post_attention_layernorm): RMSNorm()
          (mlp): MLP(
            (dense_h_to_4h): ryzenAI.QLinearPerGrp(in_features:4096, out_features:27392, bias:torch.Size([1]), device:cpu, w_bit:4 group_size:128  )
            (dense_4h_to_h): ryzenAI.QLinearPerGrp(in_features:13696, out_features:4096, bias:torch.Size([1]), device:cpu, w_bit:4 group_size:128  )
          )
        )
      )
      (final_layernorm): RMSNorm()
    )
    (output_layer): ryzenAI.QLinearPerGrp(in_features:4096, out_features:65024, bias:torch.Size([1]), device:cpu, w_bit:4 group_size:128  )
  )
)

```
## STX

```python run_awq.py --model_name THUDM/chatglm3-6b --task decode --algorithm pergrp```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        9 |                     60 |         13.844   |             1140.37  |           211.995 |      4.71709 |
|          2 |                       11 |                     60 |         14.1787  |             1325.31  |           214.834 |      4.65475 |
|          3 |                       10 |                     60 |         14.0233  |             1203.63  |           214.189 |      4.66877 |
|          4 |                        9 |                     31 |          6.73998 |             1098.91  |           185.028 |      5.40459 |
|          5 |                        7 |                     60 |         13.2927  |              854.534 |           207.788 |      4.8126  |
|          6 |                        7 |                     60 |         13.2372  |              855.318 |           206.695 |      4.83804 |
|          7 |                        9 |                     15 |          3.51358 |             1117.08  |           168.165 |      5.94653 |
|          8 |                        8 |                     60 |         13.5041  |              984.9   |           209.203 |      4.78005 |
|          9 |                        8 |                     60 |         13.5673  |              966.539 |           209.956 |      4.76291 |
|         10 |                        8 |                     60 |         13.5998  |              979.362 |           210.932 |      4.74087 |


## Perplexity on STX

```python run_awq.py --model_name THUDM/chatglm3-6b --task perplexity --algorithm pergrp --group_size 32 --target cpu --precision w4abf16```

```python run_awq.py --model_name THUDM/chatglm3-6b --task perplexity --algorithm pergrp --group_size 128 --target cpu --precision w4abf16```

Measured on 4 samples on wikitext2-raw test set

Sequence Length | CPU (bf16) | NPU (pergrp:32) | NPU (pergrp:128)
----------------|------------|-----------------|-----------------
16              | 1322.792   | 1633.430        | 1140.318
96              |   68.481   |   73.183        |   74.046
192             |   49.907   |   54.173        |   53.962
256             |   62.842   |   66.634        |   64.332
512             |   46.518   |   48.371        |   49.907
768             |   43.699   |   45.797        |   46.156
1024            |   44.388   |   45.618        |   45.087
1536            |   41.536   |   44.215        |   43.700
2048            |   40.574   |   43.022        |   44.388



# Outdated data below
## HPT (With MCDM)
### No optimization

```python run_awq.py --model_name THUDM/chatglm3-6b --task decode --target aie --algorithm pergrp```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        9 |                     30 |          7.01493 |              1537.31 |           185.064 |      5.40353 |
|          2 |                       11 |                     30 |          7.36869 |              1790.89 |           189.297 |      5.2827  |
|          3 |                       10 |                     30 |          7.04368 |              1569.88 |           185.652 |      5.38643 |
|          4 |                        9 |                     30 |          6.90231 |              1419.25 |           185.989 |      5.37666 |
|          5 |                        7 |                     30 |          6.56079 |              1108.97 |           184.947 |      5.40697 |
|          6 |                        7 |                     30 |          6.58996 |              1119.39 |           185.579 |      5.38853 |
|          7 |                        9 |                     15 |          4.00133 |              1413.03 |           181.735 |      5.5025  |
|          8 |                        8 |                     30 |          6.86491 |              1269.48 |           189.744 |      5.27027 |
|          9 |                        8 |                     30 |          6.82924 |              1319.9  |           186.983 |      5.34807 |
|         10 |                        8 |                     30 |          6.80947 |              1303.73 |           186.524 |      5.36123 |

### With optimizations
```python run_awq.py --model_name THUDM/chatglm3-6b --task decode --target aie --algorithm pergrp --flash_attention_plus```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        9 |                     30 |          6.86146 |             1410.77  |           183.921 |      5.4371  |
|          2 |                       11 |                     30 |          6.91825 |             1507.13  |           183.518 |      5.44906 |
|          3 |                       10 |                     30 |          6.78454 |             1388.97  |           183     |      5.46449 |
|          4 |                        9 |                     30 |          6.64776 |             1270.06  |           182.309 |      5.48518 |
|          5 |                        7 |                     30 |          6.28827 |              900.413 |           182.628 |      5.47561 |
|          6 |                        7 |                     30 |          6.30289 |              900.017 |           183.196 |      5.45864 |
|          7 |                        9 |                     15 |          3.85962 |             1255.8   |           182.737 |      5.47233 |
|          8 |                        8 |                     30 |          6.42827 |             1023.31  |           183.251 |      5.45698 |
|          9 |                        8 |                     30 |          6.40987 |             1020.52  |           182.623 |      5.47577 |
|         10 |                        8 |                     30 |          6.39635 |             1017.71  |           182.454 |      5.48085 |

## PHX
### No optimization

```python run_awq.py --model_name THUDM/chatglm3-6b --task decode --target aie --algorithm pergrp```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        9 |                     30 |         10.5785  |              2286.18 |           282.89  |      3.53494 |
|          2 |                       11 |                     30 |         10.9823  |              2692.59 |           282.861 |      3.5353  |
|          3 |                       10 |                     30 |         10.7295  |              2470.82 |           281.737 |      3.54941 |
|          4 |                        9 |                     30 |         10.43    |              2213.24 |           280.291 |      3.56772 |
|          5 |                        7 |                     30 |         10.0097  |              1757.36 |           281.586 |      3.55132 |
|          6 |                        7 |                     30 |         28.1237  |              1761.99 |           283.362 |      3.52906 |
|          7 |                        9 |                     15 |          6.23319 |              2239.23 |           282.113 |      3.54468 |
|          8 |                        8 |                     30 |         10.4127  |              2029.15 |           286.032 |      3.49611 |
|          9 |                        8 |                     30 |         10.3778  |              2003.19 |           285.781 |      3.49919 |
|         10 |                        8 |                     30 |         10.3884  |              2011.35 |           285.851 |      3.49833 |

### With optimizations
```python run_awq.py --model_name THUDM/chatglm3-6b --task decode --target aie --algorithm pergrp --flash_attention_plus```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        9 |                     30 |         10.361   |              2071.16 |           282.516 |      3.53962 |
|          2 |                       11 |                     30 |         10.6661  |              2345.67 |           283.869 |      3.52275 |
|          3 |                       10 |                     30 |         10.5313  |              2168.55 |           285.277 |      3.50537 |
|          4 |                        9 |                     30 |         10.2415  |              1971.79 |           282.135 |      3.5444  |
|          5 |                        7 |                     30 |          9.73157 |              1411.38 |           283.775 |      3.52392 |
|          6 |                        7 |                     30 |          9.72294 |              1417.82 |           283.367 |      3.529   |
|          7 |                        9 |                     15 |          5.98185 |              1970.19 |           283.456 |      3.52788 |
|          8 |                        8 |                     30 |          9.90598 |              1601.2  |           283.307 |      3.52974 |
|          9 |                        8 |                     30 |          9.92152 |              1608.21 |           283.664 |      3.52529 |
|         10 |                        8 |                     30 |          9.91769 |              1613.46 |           283.346 |      3.52926 |

## STX B0 (with MCDM)
### No optimization

```python run_awq.py --model_name THUDM/chatglm3-6b --task decode --target aie --algorithm pergrp```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        9 |                     30 |          6.21695 |             1147.89  |           170.269 |      5.87308 |
|          2 |                       11 |                     30 |          6.39373 |             1275.1   |           172.984 |      5.78089 |
|          3 |                       10 |                     30 |          6.25368 |             1170.4   |           171.663 |      5.82536 |
|          4 |                        9 |                     30 |          6.0933  |             1044.02  |           170.355 |      5.8701  |
|          5 |                        7 |                     30 |          5.77664 |              824.284 |           167.213 |      5.98041 |
|          6 |                        7 |                     30 |          5.75982 |              809.912 |           167.128 |      5.98344 |
|          7 |                        9 |                     15 |          3.32685 |             1043.26  |           159.443 |      6.27185 |
|          8 |                        8 |                     30 |          5.93369 |              933.099 |           168.877 |      5.92148 |
|          9 |                        8 |                     30 |          5.91814 |              929.446 |           168.559 |      5.93264 |
|         10 |                        8 |                     30 |          5.92378 |              931.024 |           168.709 |      5.92736 |

### With optimizations
```python run_awq.py --model_name THUDM/chatglm3-6b --task decode --target aie --algorithm pergrp --flash_attention_plus```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        9 |                     30 |          5.12234 |             1043.78  |           136.276 |      7.33803 |
|          2 |                       11 |                     30 |          5.1122  |             1081.04  |           135.462 |      7.38215 |
|          3 |                       10 |                     30 |          5.03373 |              987.041 |           136.022 |      7.35175 |
|          4 |                        9 |                     30 |          4.94196 |              904.651 |           135.632 |      7.37291 |
|          5 |                        7 |                     30 |          4.68401 |              656.801 |           135.327 |      7.38952 |
|          6 |                        7 |                     30 |          4.67916 |              651.799 |           135.329 |      7.38943 |
|          7 |                        9 |                     15 |          2.84167 |              899.818 |           135.023 |      7.40615 |
|          8 |                        8 |                     30 |          4.7895  |              744.899 |           135.622 |      7.37341 |
|          9 |                        8 |                     30 |          4.77855 |              743.287 |           135.692 |      7.36964 |
|         10 |                        8 |                     30 |          4.77751 |              745.032 |           135.519 |      7.37905 |
