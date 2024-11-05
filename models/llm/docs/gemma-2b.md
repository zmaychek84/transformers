# google/gemma-2b

```
python run_awq.py --task quantize --model_name google/gemma-2b
```

## PHX
```python run_awq.py --task decode --model_name google/gemma-2b --target aie```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        8 |                     30 |          13.7673 |              652.144 |           433.762 |      2.30541 |
|          2 |                       10 |                     30 |          14.6383 |             1566.6   |           430.59  |      2.3224  |
|          3 |                        8 |                     30 |          13.5245 |              580.49  |           430.863 |      2.32092 |
|          4 |                        8 |                     30 |          13.5221 |              583.211 |           430.878 |      2.32084 |
|          5 |                        6 |                     27 |          12.1691 |              596.111 |           428.89  |      2.3316  |
|          6 |                        5 |                     30 |          13.4268 |              584.428 |           427.45  |      2.33946 |
|          7 |                        8 |                     30 |          13.371  |              575.62  |           425.718 |      2.34897 |
|          8 |                        6 |                     30 |          13.3543 |              595.517 |           427.161 |      2.34104 |
|          9 |                        7 |                     30 |          13.4554 |              568.241 |           430.811 |      2.3212  |
|         10 |                        7 |                     30 |          13.3988 |              567.978 |           428.848 |      2.33183 |

## HPT (With MCDM)

```python run_awq.py --task decode --model_name google/gemma-2b --target aie```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        8 |                     30 |         10.8228  |              513.363 |           333.76  |      2.99616 |
|          2 |                       10 |                     30 |         11.3266  |             1064.43  |           330.905 |      3.02201 |
|          3 |                        8 |                     30 |         10.6792  |              429.056 |           334.339 |      2.99098 |
|          4 |                        8 |                     30 |         10.6331  |              445.158 |           334.142 |      2.99274 |
|          5 |                        6 |                     27 |          9.53723 |              445.855 |           331.915 |      3.01282 |
|          6 |                        5 |                     30 |         10.5855  |              414.303 |           334.337 |      2.99099 |
|          7 |                        8 |                     30 |         10.6027  |              431.575 |           333.151 |      3.00164 |
|          8 |                        6 |                     30 |         10.583   |              449.86  |           335.186 |      2.98342 |
|          9 |                        7 |                     30 |         10.6571  |              420.254 |           336.516 |      2.97162 |
|         10 |                        7 |                     30 |         10.666   |              420.876 |           336.073 |      2.97554 |

## STX B0 (With MCDM)

```python run_awq.py --task decode --model_name google/gemma-2b --target aie```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        8 |                     30 |          23.6767 |              956.604 |           770.983 |      1.29705 |
|          2 |                       10 |                     30 |          23.6103 |             1203.53  |           760.843 |      1.31433 |
|          3 |                        8 |                     30 |          23.2994 |              844.18  |           762.41  |      1.31163 |
|          4 |                        8 |                     30 |          23.6048 |              834.831 |           773.277 |      1.2932  |
|          5 |                        6 |                     30 |          23.5681 |              766.823 |           774.254 |      1.29157 |
|          6 |                        5 |                     30 |          23.3859 |              824.545 |           766.108 |      1.3053  |
|          7 |                        8 |                     30 |          23.4239 |              833.072 |           766.835 |      1.30406 |
|          8 |                        6 |                     30 |          22.8071 |              752.873 |           748.669 |      1.3357  |
|          9 |                        7 |                     30 |          23.5236 |              836.33  |           770.353 |      1.29811 |
|         10 |                        7 |                     30 |          23.2727 |              759.372 |           764.327 |      1.30834 |

```
GemmaForCausalLM(
  (model): GemmaModel(
    (embed_tokens): Embedding(256000, 2048, padding_idx=0)
    (layers): ModuleList(
      (0-17): 18 x GemmaDecoderLayer(
        (self_attn): GemmaAttention(
          (q_proj): ryzenAI.QLinearPerGrp(in_features:2048, out_features:2048, bias:None, device:aie, w_bit:4 group_size:128  )
          (k_proj): ryzenAI.QLinearPerGrp(in_features:2048, out_features:256, bias:None, device:aie, w_bit:4 group_size:128  )
          (v_proj): ryzenAI.QLinearPerGrp(in_features:2048, out_features:256, bias:None, device:aie, w_bit:4 group_size:128  )
          (o_proj): ryzenAI.QLinearPerGrp(in_features:2048, out_features:2048, bias:None, device:aie, w_bit:4 group_size:128  )
          (rotary_emb): GemmaRotaryEmbedding()
        )
        (mlp): GemmaMLP(
          (gate_proj): ryzenAI.QLinearPerGrp(in_features:2048, out_features:16384, bias:None, device:aie, w_bit:4 group_size:128  )
          (up_proj): ryzenAI.QLinearPerGrp(in_features:2048, out_features:16384, bias:None, device:aie, w_bit:4 group_size:128  )
          (down_proj): ryzenAI.QLinearPerGrp(in_features:16384, out_features:2048, bias:None, device:aie, w_bit:4 group_size:128  )
          (act_fn): GELUActivation()
        )
        (input_layernorm): GemmaRMSNorm()
        (post_attention_layernorm): GemmaRMSNorm()
      )
    )
    (norm): GemmaRMSNorm()
  )
  (lm_head): Linear(in_features=2048, out_features=256000, bias=False)
)
model.mode_name: gemma-2b
****************************************
prompt: What is the meaning of life?
response: What is the meaning of life?

It’s a question that has been asked for centuries, and one that has no definitive answer. But for many people, the answer is simple
****************************************
prompt: What does Xilinx do?
response: What does Xilinx do?

Xilinx is a fabless semiconductor company that designs and manufactures programmable logic devices (PLDs) and field-programmable gate arrays (F
****************************************
...
...
```
