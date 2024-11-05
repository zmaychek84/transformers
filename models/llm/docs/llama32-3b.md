# Llama-3.2-3B

## Quantize
```python run_awq.py --model_name nltpt/Llama-3.2-3B --task quantize --algorithm pergrp```

```
LlamaModelEval(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 3072)
    (layers): ModuleList(
      (0-27): 28 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): ryzenAI.QLinearPerGrp(in_features:3072, out_features:3072, bias:False, device:aie, w_bit:4, model_name:Llama-3.2-3B, gemm_torch:0, group_size:128 )
          (k_proj): ryzenAI.QLinearPerGrp(in_features:3072, out_features:1024, bias:False, device:aie, w_bit:4, model_name:Llama-3.2-3B, gemm_torch:0, group_size:128 )
          (v_proj): ryzenAI.QLinearPerGrp(in_features:3072, out_features:1024, bias:False, device:aie, w_bit:4, model_name:Llama-3.2-3B, gemm_torch:0, group_size:128 )
          (o_proj): ryzenAI.QLinearPerGrp(in_features:3072, out_features:3072, bias:False, device:aie, w_bit:4, model_name:Llama-3.2-3B, gemm_torch:0, group_size:128 )
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): ryzenAI.QLinearPerGrp(in_features:3072, out_features:8192, bias:False, device:aie, w_bit:4, model_name:Llama-3.2-3B, gemm_torch:0, group_size:128 )
          (up_proj): ryzenAI.QLinearPerGrp(in_features:3072, out_features:8192, bias:False, device:aie, w_bit:4, model_name:Llama-3.2-3B, gemm_torch:0, group_size:128 )
          (down_proj): ryzenAI.QLinearPerGrp(in_features:8192, out_features:3072, bias:False, device:aie, w_bit:4, model_name:Llama-3.2-3B, gemm_torch:0, group_size:128 )
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((3072,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((3072,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((3072,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): ryzenAI.QLinearPerGrp(in_features:3072, out_features:128256, bias:False, device:aie, w_bit:4, model_name:Llama-3.2-3B, gemm_torch:0, group_size:128 )
)
After transformation - model.mode_name: Llama-3.2-3B
****************************************

```

## Decode
|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        7 |                     33 |          5.57972 |              447.423 |           144.468 |      6.92194 |
|          2 |                        8 |                     60 |          9.38975 |              458.679 |           146.889 |      6.80786 |
|          3 |                        6 |                     60 |          9.25601 |              400.902 |           145.632 |      6.86662 |
|          4 |                        7 |                     60 |         10.1069  |              427.413 |           159.353 |      6.27537 |
|          5 |                        5 |                     60 |         10.2129  |              403.155 |           161.688 |      6.18475 |
|          6 |                        4 |                     60 |         10.1381  |              420.719 |           160.47  |      6.2317  |
|          7 |                        7 |                     60 |          9.89914 |              423.864 |           156.248 |      6.40008 |
|          8 |                        6 |                     60 |         10.5136  |              425.928 |           166.294 |      6.01346 |
|          9 |                        6 |                     60 |          9.75971 |              413.449 |           153.875 |      6.49878 |
|         10 |                        6 |                     60 |         10.5837  |              457.172 |           167.041 |      5.98656 |
