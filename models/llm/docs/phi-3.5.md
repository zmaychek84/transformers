# Phi 3.5 mini

`
Phi3ModelEval(
  (model): Phi3Model(
    (embed_tokens): Embedding(32064, 3072, padding_idx=32000)
    (embed_dropout): Dropout(p=0.0, inplace=False)
    (layers): ModuleList(
      (0-31): 32 x Phi3DecoderLayer(
        (self_attn): Phi3Attention(
          (o_proj): Linear(in_features=3072, out_features=3072, bias=False)
          (qkv_proj): Linear(in_features=3072, out_features=9216, bias=False)
          (rotary_emb): Phi3LongScaledRotaryEmbedding()
        )
        (mlp): Phi3MLP(
          (gate_up_proj): Linear(in_features=3072, out_features=16384, bias=False)
          (down_proj): Linear(in_features=8192, out_features=3072, bias=False)
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
  (lm_head): Linear(in_features=3072, out_features=32064, bias=False)
)
model.mode_name: Phi-3.5-mini-instruct
`

## STX Performance
```
****************************************
<|system|> You are a helpful assistant.<|end|><|user|> What is the meaning of life?<|end|><|assistant|> The meaning of life is a profound philosophical question that has been contemplated
 by humans for millennia. Different individuals, cultures, and religions offer various interpretations. Some common perspectives include:

1. **Existential Meaning:** Many exist
****************************************
<|system|> You are a helpful assistant.<|end|><|user|> Tell me something you don't know.<|end|><|assistant|> As an artificial intelligence, I don't have personal knowledge and don't know in the human sense. However, I can tell you that I, as of my last update, don't know things that are created or updated in real-time. For instance, I can't provide
****************************************
<|system|> You are a helpful assistant.<|end|><|user|> What does Xilinx do?<|end|><|assistant|> Xilinx, Inc. is a multinomanial technology company that specializes in the design and manufacturing of integrated circuit (IC) products, particularly those used for embedded systems. The company plays a significant role in the semiconductor industry, known for several key contributions:

****************************************
<|system|> You are a helpful assistant.<|end|><|user|> What is the mass of earth?<|end|><|assistant|> The mass of Earth is approximately 5.972 × 10^24 kilograms. This value is based on scientific measurements and calculations that take into account the distribution and density of Earth's materials. It is the product of Earth's average density and its volume.
****************************************
```

## MLADF
```python run_awq.py --model_name microsoft/Phi-3.5-mini-instruct --task decode --algorithm awqplus```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                       18 |                     60 |          9.02457 |              433.349 |           135.761 |      7.36587 |
|          2 |                       20 |                     60 |          8.71783 |              451.587 |           138.283 |      7.23155 |
|          3 |                       18 |                     60 |          8.74657 |              410.941 |           139.545 |      7.16614 |
|          4 |                       18 |                     60 |          8.79734 |              417.413 |           140.167 |      7.13434 |
|          5 |                       16 |                     60 |          8.46414 |              385.061 |           135.139 |      7.39977 |
|          6 |                       15 |                     60 |          8.48388 |              384.677 |           135.613 |      7.37394 |
|          7 |                       19 |                     47 |          6.67963 |              405.796 |           134.76  |      7.4206  |
|          8 |                       18 |                     60 |          8.4162  |              405.427 |           134.067 |      7.45898 |
|          9 |                       19 |                     60 |          8.52757 |              410.304 |           135.974 |      7.35432 |
|         10 |                       17 |                     60 |          8.44231 |              397.146 |           134.83  |      7.41674 |

## Non-MLADF
```set MLADF=```
```python run_awq.py --model_name microsoft/Phi-3.5-mini-instruct --task decode --algorithm awqplus```
|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                       18 |                     60 |          9.08169 |              829.327 |           133.131 |      7.51139 |
|          2 |                       20 |                     60 |          8.63742 |              801.369 |           131.117 |      7.62678 |
|          3 |                       18 |                     60 |          8.54535 |              765.576 |           130.138 |      7.68418 |
|          4 |                       18 |                     60 |          8.52657 |              765.85  |           129.804 |      7.70392 |
|          5 |                       16 |                     60 |          8.51931 |              753.527 |           129.992 |      7.69275 |
|          6 |                       15 |                     60 |          8.54923 |              758.394 |           130.266 |      7.67662 |
|          7 |                       19 |                     28 |          4.31842 |              769.518 |           129.761 |      7.70647 |
|          8 |                       18 |                     60 |          8.55053 |              771.917 |           130.19  |      7.68109 |
|          9 |                       19 |                     60 |          8.51666 |              769.972 |           129.713 |      7.70934 |
|         10 |                       17 |                     60 |          8.53657 |              768.997 |           129.975 |      7.69378 |

# Perplexity

Prompt Length | CPU bf16 | CPU int4 pergrp | NPU int4 pergrp | NPU int4 AWQ | NPU int4 AWQ+
--------------|----------|-----------------|-----------------|--------------|---------------
16            | 62.1246  | 78.0238         | 79.9690         | 84.1953      | 86.5667
96            | 11.9047  | 13.7668         | 14.0373         | 15.0704      | 15.4621
192           |  9.5358  | 10.8599         | 11.0039         | 11.3098      | 11.7084
256           |  8.5410  |  9.6927         |  9.7682         | 10.4760      | 10.8843
512           |  7.7075  |  8.5642         |  8.6372         |  8.9593      |  9.2292
758           |  6.7772  |  7.4971         |  7.5444         |  7.6539      |  7.8973
1024          |  6.6997  |  7.4106         |  7.4356         |  7.4708      |  7.7055
1536          |  6.8596  |  7.6134         |  7.6553         |  7.5296      |  7.7797
2048          |  6.6874  |  7.4509         |  7.4864         |  7.4010      |  7.6284

## BF16 CPU
```python run_awq.py --model_name microsoft/Phi-3.5-mini-instruct --target cpu --precision bf16 --task perplexity```

## int4 per grp CPU
```python run_awq.py --model_name microsoft/Phi-3.5-mini-instruct --task quantize --algorithm pergrp --group_size 32```
```python run_awq.py --model_name microsoft/Phi-3.5-mini-instruct --target cpu --task perplexity --algorithm pergrp --group_size 32 --precision w4abf16```

## int4 AWQ NPU
```python run_awq.py --model_name microsoft/Phi-3.5-mini-instruct --task quantize --algorithm awq```
```python run_awq.py --model_name microsoft/Phi-3.5-mini-instruct --task perplexity --algorithm awq```

## int4 AWQ+ NPU
```python run_awq.py --model_name microsoft/Phi-3.5-mini-instruct --task quantize --algorithm awqplus```
```python run_awq.py --model_name microsoft/Phi-3.5-mini-instruct --task perplexity --algorithm awqplus```
