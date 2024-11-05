# Qwen/Qwen1.5-7B-Chat

```
Qwen2ModelEval(
  (model): Qwen2Model(
    (embed_tokens): Embedding(151936, 4096)
    (layers): ModuleList(
      (0-31): 32 x Qwen2DecoderLayer(
        (self_attn): Qwen2FlashAttentionPlus(
          (rotary_emb): Qwen2RotaryEmbedding()
          (o_proj): ryzenAI.QLinearPerGrp(in_features:4096, out_features:4096, bias:None, device:aie, w_bit:4 group_size:128  )
          (qkv_proj): ryzenAI.QLinearPerGrp(in_features:4096, out_features:12288, bias:torch.Size([12288]), device:aie, w_bit:4 group_size:128  )
        )
        (mlp): Qwen2MLP(
          (gate_proj): ryzenAI.QLinearPerGrp(in_features:4096, out_features:11008, bias:None, device:aie, w_bit:4 group_size:128  )
          (up_proj): ryzenAI.QLinearPerGrp(in_features:4096, out_features:11008, bias:None, device:aie, w_bit:4 group_size:128  )
          (down_proj): ryzenAI.QLinearPerGrp(in_features:11008, out_features:4096, bias:None, device:aie, w_bit:4 group_size:128  )
          (act_fn): SiLU()
        )
        (input_layernorm): Qwen2RMSNorm()
        (post_attention_layernorm): Qwen2RMSNorm()
      )
    )
    (norm): Qwen2RMSNorm()
  )
  (lm_head): Linear(in_features=4096, out_features=151936, bias=False)
)
```
## Latency : STX B0

### AWQ Plus

```python run_awq.py --model_name Qwen/Qwen1.5-7B --algorithm awqplus --task decode --fast_attention --fast_mlp --fast_norm --fast_decoder```
|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                       26 |                     60 |          9.50457 |              510.835 |           141.416 |      7.07134 |
|          2 |                       27 |                     60 |          8.98255 |              448.795 |           139.953 |      7.14527 |
|          3 |                       25 |                     60 |          8.98326 |              440.235 |           139.979 |      7.14393 |
|          4 |                       26 |                     60 |          8.9608  |              440.454 |           140.014 |      7.14216 |
|          5 |                       24 |                     60 |          8.97102 |              428.335 |           139.896 |      7.14817 |
|          6 |                       23 |                     60 |          9.00449 |              425.687 |           140.23  |      7.13115 |
|          7 |                       26 |                     60 |          8.97505 |              438.906 |           140.244 |      7.13044 |
|          8 |                       25 |                     60 |          9.01231 |              434.741 |           140.58  |      7.11336 |
|          9 |                       25 |                     60 |          9.06598 |              445.545 |           141.285 |      7.07789 |
|         10 |                       25 |                     60 |          9.04631 |              447.766 |           141.022 |      7.0911  |

```python run_awq.py --model_name Qwen/Qwen1.5-7B --algorithm awqplus --task profilemodel256 --fast_attention --fast_mlp --fast_norm --fast_decoder```
|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                      224 |                     60 |          12.4991 |              2825.22 |           152.838 |      6.54286 |

```python run_awq.py --model_name Qwen/Qwen1.5-7B --algorithm awqplus --task profilemodel512 --fast_attention --fast_mlp --fast_norm --fast_decoder```
|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                      446 |                     35 |          9.43946 |              2978.43 |           172.424 |      5.79965 |

```python run_awq.py --model_name Qwen/Qwen1.5-7B --algorithm awqplus --task profilemodel1k --fast_attention --fast_mlp --fast_norm --fast_decoder```
|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                     1024 |                     60 |          17.1354 |              2695.12 |           230.413 |      4.34003 |

```python run_awq.py --model_name Qwen/Qwen1.5-7B --algorithm awqplus --task profilemodel2k --fast_attention --fast_mlp --fast_norm --fast_decoder```
|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                     2048 |                     60 |           24.192 |              5738.34 |           295.264 |       3.3868 |

```python run_awq.py --model_name Qwen/Qwen1.5-7B --algorithm awqplus --task benchmark_exact --fast_attention --fast_mlp --fast_norm --fast_decoder```
|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                      128 |                     60 |          12.1429 |              1878.06 |           160.447 |      6.23259 |
|          2 |                      256 |                     60 |          12.0324 |              2221.12 |           161.372 |      6.19684 |
|          3 |                      512 |                     60 |          13.5529 |              2746.85 |           178.018 |      5.61741 |
|          4 |                     1024 |                     60 |          17.1916 |              3561.86 |           225.936 |      4.42604 |
|          5 |                     2048 |                     60 |          24.2573 |              6317.18 |           298.567 |      3.34934 |

```python run_awq.py --model_name Qwen/Qwen1.5-7B --algorithm awqplus --task benchmark --fast_attention --fast_mlp --fast_norm --fast_decoder```
|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                       24 |                     60 |          19.1311 |              1129.22 |           285.65  |      3.50079 |
|          2 |                       78 |                     60 |          18.9761 |              1439.42 |           292.073 |      3.4238  |
|          3 |                      128 |                     60 |          19.2616 |              1409.58 |           297.991 |      3.35581 |
|          4 |                      140 |                     60 |          21.1275 |              2159.55 |           316.659 |      3.15797 |
|          5 |                      256 |                     60 |          21.0738 |              1646.35 |           324.175 |      3.08476 |
|          6 |                      293 |                     60 |          21.3259 |              3416.77 |           297.864 |      3.35724 |
|          7 |                      372 |                     60 |          21.5804 |              3160.91 |           307.09  |      3.25637 |
|          8 |                      512 |                     60 |          21.8758 |              2630.53 |           321.538 |      3.11005 |
|          9 |                      580 |                     60 |          24.4638 |              4583.17 |           331.506 |      3.01654 |
|         10 |                      717 |                     60 |          25.0828 |              4764.43 |           339.446 |      2.94598 |
|         11 |                      900 |                     60 |          26.6648 |              4775.07 |           366.154 |      2.73109 |
|         12 |                     1024 |                     60 |          26.1391 |              3740.82 |           374.065 |      2.67333 |
|         13 |                     1050 |                     60 |          31.2098 |              8509.01 |           379.568 |      2.63457 |
|         14 |                     1320 |                     60 |          32.0923 |              8419.66 |           395.864 |      2.52612 |
|         15 |                     1537 |                     60 |          34.1343 |              8463.77 |           429.59  |      2.3278  |
|         16 |                     1580 |                     60 |          33.3824 |              8531.92 |           414.879 |      2.41034 |
|         17 |                     1670 |                     60 |          34.1494 |              8521.82 |           429.761 |      2.32688 |
|         18 |                     1900 |                     60 |          35.8816 |              8723.71 |           455.39  |      2.19592 |
|         19 |                     2048 |                     60 |          34.3126 |              6413.11 |           467.95  |      2.13698 |


# Accuracy

Perplexity is measured on **wikitext2-raw** dataset.

MMLU is measured across 30 examples of abstract_algebrate dataset.

## Qwen1.5-7B-Chat
AWQ with autoscale off for performance evaluations.

| **Precision + Config**                         | **Dev** | **Perplexity (8 samples)** | **Perplexity (All)** | **MMLU**
|------------------------------------------------|---------|----------------------------|----------------------|-------------------------------
BF16                                             | CPU     |   9.519                    |  9.954               | 23.33
w4abf16 (AWQ, 4-bit, g:128) + FA                 | NPU     |   9.912                    | 10.325               | 26.67
w4abf16 (AWQ, 4-bit, g:128) + FA + lm_head(g:32) | NPU     |  10.007                    | 10.424               | 23.33

## Qwen/Qwen1.5-7B
AWQ with autoscale off for performance evaluations.

| **Precision + Config**                         | **Dev** | **Perplexity (8 samples)** | **MMLU**
|------------------------------------------------|---------|----------------------------|-----------
BF16                                             | CPU     |   7.118                    | 46.67
w4abf16 (AWQ, 4-bit, g:128) + FA                 | NPU     |   7.527                    | 46.67
w4abf16 (AWQ, 4-bit, g:128) + FA + lm_head(g:32) | NPU     |   7.654                    | 23.33
