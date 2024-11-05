# google/gemma-7b (PHX)
```
python run_awq.py --task quantize --model_name google/gemma-7b
```

## HPT (With MCDM)

```python run_awq.py --task decode --model_name google/gemma-7b --target aie```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        8 |                     30 |          14.8476 |              735.749 |           449.881 |      2.22281 |
|          2 |                       10 |                     30 |          15.8485 |             1630.99  |           452.828 |      2.20834 |
|          3 |                        8 |                     30 |          14.812  |              652.683 |           450.999 |      2.2173  |
|          4 |                        8 |                     30 |          14.8245 |              647.645 |           451.554 |      2.21457 |
|          5 |                        6 |                     30 |          14.7348 |              639.843 |           451.956 |      2.2126  |
|          6 |                        5 |                     30 |          14.7731 |              627.987 |           451.529 |      2.2147  |
|          7 |                        8 |                     30 |          14.8222 |              655.656 |           451.526 |      2.21471 |
|          8 |                        6 |                     30 |          14.7668 |              641.125 |           452.48  |      2.21004 |
|          9 |                        7 |                     30 |          14.7968 |              646.879 |           451.058 |      2.21701 |
|         10 |                        7 |                     30 |          14.9828 |              654.161 |           456.597 |      2.19012 |

## PHX

```python run_awq.py --task decode --model_name google/gemma-7b --target aie```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        8 |                     30 |          20.4442 |             1017.42  |           640.133 |      1.56217 |
|          2 |                       10 |                     30 |          22.0263 |             2456.25  |           642.593 |      1.5562  |
|          3 |                        8 |                     30 |          20.3066 |              957.949 |           638.057 |      1.56726 |
|          4 |                        8 |                     30 |          20.0964 |              951.85  |           631.785 |      1.58282 |
|          5 |                        6 |                     30 |          20.2271 |              904.972 |           638.408 |      1.5664  |
|          6 |                        5 |                     30 |          20.1515 |              905.94  |           635.153 |      1.57442 |
|          7 |                        8 |                     30 |          20.3039 |              958.864 |           638.28  |      1.56671 |
|          8 |                        6 |                     30 |          20.2461 |              876.98  |           638.887 |      1.56522 |
|          9 |                        7 |                     30 |          20.3719 |              946.677 |           639.732 |      1.56315 |
|         10 |                        7 |                     30 |          20.3115 |              941.833 |           638.377 |      1.56647 |

## STX B0 (With MCDM)

```python run_awq.py --task decode --model_name google/gemma-7b --target aie```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        8 |                     30 |          26.9486 |             1066.99  |           880     |      1.13636 |
|          2 |                       10 |                     30 |          27.2911 |             1674.07  |           871.563 |      1.14736 |
|          3 |                        8 |                     30 |          26.738  |              978.751 |           876.471 |      1.14094 |
|          4 |                        8 |                     30 |          27.1517 |             1010.96  |           889.518 |      1.1242  |
|          5 |                        6 |                     30 |          26.2074 |              912.98  |           860.304 |      1.16238 |
|          6 |                        5 |                     30 |          26.9836 |              971.748 |           885.248 |      1.12963 |
|          7 |                        8 |                     30 |          25.4709 |             1015.73  |           831.985 |      1.20194 |
|          8 |                        6 |                     30 |          26.9143 |             1022.35  |           880.926 |      1.13517 |
|          9 |                        7 |                     30 |          26.3599 |             1069.27  |           860.305 |      1.16238 |
|         10 |                        7 |                     30 |          27.3947 |             1028.84  |           877.948 |      1.13902 |
```
GemmaForCausalLM(
  (model): GemmaModel(
    (embed_tokens): Embedding(256000, 3072, padding_idx=0)
    (layers): ModuleList(
      (0-27): 28 x GemmaDecoderLayer(
        (self_attn): GemmaAttention(
          (q_proj): ryzenAI.QLinearPerGrp(in_features:3072, out_features:4096, bias:None, device:aie, w_bit:4 group_size:128  )
          (k_proj): ryzenAI.QLinearPerGrp(in_features:3072, out_features:4096, bias:None, device:aie, w_bit:4 group_size:128  )
          (v_proj): ryzenAI.QLinearPerGrp(in_features:3072, out_features:4096, bias:None, device:aie, w_bit:4 group_size:128  )
          (o_proj): ryzenAI.QLinearPerGrp(in_features:4096, out_features:3072, bias:None, device:aie, w_bit:4 group_size:128  )
          (rotary_emb): GemmaRotaryEmbedding()
        )
        (mlp): GemmaMLP(
          (gate_proj): ryzenAI.QLinearPerGrp(in_features:3072, out_features:24576, bias:None, device:aie, w_bit:4 group_size:128  )
          (up_proj): ryzenAI.QLinearPerGrp(in_features:3072, out_features:24576, bias:None, device:aie, w_bit:4 group_size:128  )
          (down_proj): ryzenAI.QLinearPerGrp(in_features:24576, out_features:3072, bias:None, device:aie, w_bit:4 group_size:128  )
          (act_fn): GELUActivation()
        )
        (input_layernorm): GemmaRMSNorm()
        (post_attention_layernorm): GemmaRMSNorm()
      )
    )
    (norm): GemmaRMSNorm()
  )
  (lm_head): Linear(in_features=3072, out_features=256000, bias=False)
)
model.mode_name: gemma-7b
****************************************
prompt: What is the meaning of life?
response: What is the meaning of life?

What is the meaning of life?

What is the meaning of life?

What is the meaning of life?

What is the meaning of
****************************************
prompt: Tell me something you don't know.
response: Tell me something you don't know.


****************************************
prompt: What does Xilinx do?
response: What does Xilinx do?

Xilinx is a company that designs and sells programmable logic devices and integrated circuits for the electronics industry.

What is Xilinx used for
****************************************
```
