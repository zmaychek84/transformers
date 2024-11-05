# Meta-Llama-3.1-8B

```
python run_awq.py --task quantize --model_name NousResearch/Meta-Llama-3.1-8B --algorithm pergrp --group_size 32
```

```
LlamaModelEval(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 4096)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): ryzenAI.QLinearPerGrp(in_features:4096, out_features:4096, bias:None, device:aie, w_bit:4 group_size:32 )
          (k_proj): ryzenAI.QLinearPerGrp(in_features:4096, out_features:1024, bias:None, device:aie, w_bit:4 group_size:32 )
          (v_proj): ryzenAI.QLinearPerGrp(in_features:4096, out_features:1024, bias:None, device:aie, w_bit:4 group_size:32 )
          (o_proj): ryzenAI.QLinearPerGrp(in_features:4096, out_features:4096, bias:None, device:aie, w_bit:4 group_size:32 )
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): ryzenAI.QLinearPerGrp(in_features:4096, out_features:14336, bias:None, device:aie, w_bit:4 group_size:32 )
          (up_proj): ryzenAI.QLinearPerGrp(in_features:4096, out_features:14336, bias:None, device:aie, w_bit:4 group_size:32 )
          (down_proj): ryzenAI.QLinearPerGrp(in_features:14336, out_features:4096, bias:None, device:aie, w_bit:4 group_size:32 )
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): ryzenAI.QLinearPerGrp(in_features:4096, out_features:128256, bias:None, device:aie, w_bit:4 group_size:32 )
)
model.mode_name: Meta-Llama-3.1-8B
****************************************
****************************************
<|begin_of_text|>What is the meaning of life? 5 things you need to know if you want to live a meaningful life
We have all struggled with the big question, “What is the meaning of life?” At least one person has asked this question in his or her mind. What is the purpose of life? Why are we born? Is
****************************************
<|begin_of_text|>Tell me something you don't know. I'm serious. I just read a recent study, "Aging and the Knowledge Economy: What We Do, Know, and Know We Need to Know."
 It's worth a quick look. I'll wait. Okay, ready? So, the point of the study? Well, it's actually
****************************************
<|begin_of_text|>What does Xilinx do? Xilinx Inc. designs and manufactures programmable gate arrays, digital signal processing chips, field programmable gate arrays, and embedded processors that allow electronic designers to alter the hardware of the devices after they are assembled.
It also makes related software programming tools.
What is Xilinx known for?
Xilinx designs
****************************************
<|begin_of_text|>What is the mass of earth? - Quora. What is the mass of earth? - Quora.
What is the mass of earth?
The mass of the Earth is 5.972 × 1024 kg or 5,972,000,000,000,000,000,000 kg or 6 trillion trillion
```

## Accuracy

Measured with transformers==4.43.3

**Perplexity was measured on wikitext2-raw dataset**

Perplexity measured on n=4 samples

NPU: w4abf16 (Per Grp, 4-bit, g:32)
CPU: bf16

**Seqlen** | **CPU bf16** | **NPU w4abf16 (g:32)**
-----------|--------------|-----------------
16         | 44.256       | 67.543
96         | 11.910       | 13.522
192        | 10.282       | 11.601
256        |  9.520       | 10.755
512        |  7.332       | 8.318
768        |  7.207       | 7.967
1024       |  7.345       | 8.012
1536       |  7.562       | 8.201
2048       |  7.049       | 7.567


## STX (with MCDM)

```python run_awq.py --model_nameNousResearch/Meta-Llama-3.1-8B --task decode --algorithm pergrp --group_size 32```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        8 |                     60 |          18.6955 |              905.922 |           281.296 |    3.55498   |
|          2 |                        9 |                     60 |          18.8872 |             1848.44  |           280.5   |    3.56506   |
|          3 |                        7 |                     60 |          17.8287 |              797.039 |           280.363 |    3.56681   |
|          4 |                        8 |                     60 |          17.8472 |              810.902 |           280.524 |    3.56475   |
|          5 |                        6 |                     60 |          17.8193 |              790.24  |           280.161 |    3.56937   |
|          6 |                        5 |                     60 |          17.7909 |              775.505 |           279.951 |    3.57205   |
|          7 |                        8 |                     60 |          17.8477 |              812.615 |           280.561 |    3.56429   |
|          8 |                        7 |                     60 |          17.8524 |              798.193 |           280.775 |    3.56157   |
|          9 |                        7 |                     60 |          18.1938 |              793.893 |           286.8   |    3.48675   |
