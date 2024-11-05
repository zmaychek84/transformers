# microsoft/phi-2 (PHX)

Quantize and save model

```python run_awq.py --model_name microsoft/phi-2 --task quantize --algorithm pergrp```

# HPT (With MCDM)

```python run_awq.py --model_name microsoft/phi-2 --task decode --algorithm pergrp```
|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        7 |                     30 |          5.10927 |              316.377 |           162.294 |      6.16167 |
|          2 |                        8 |                     30 |          5.02889 |              271.496 |           161.624 |      6.18721 |
|          3 |                        7 |                     30 |          5.01534 |              261.851 |           161.405 |      6.19559 |
|          4 |                        7 |                     30 |          5.03631 |              259.096 |           162.369 |      6.15882 |
|          5 |                        5 |                     30 |          5.01927 |              255.751 |           161.921 |      6.17585 |
|          6 |                        5 |                     30 |          4.99301 |              249.814 |           161.317 |      6.19899 |
|          7 |                        7 |                     30 |          5.01958 |              259.05  |           161.713 |      6.18379 |
|          8 |                        6 |                     30 |          5.00398 |              258.237 |           161.226 |      6.20249 |
|          9 |                        6 |                     30 |          5.03063 |              262.373 |           161.988 |      6.17331 |
|         10 |                        6 |                     30 |          5.02665 |              260.649 |           161.865 |      6.17799 |

```python run_awq.py --model_name microsoft/phi-2 --task decode --algorithm pergrp --flash_attention_plus```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        7 |                     30 |          4.61604 |              304.699 |           146.14  |      6.84277 |
|          2 |                        8 |                     30 |          4.57547 |              253.199 |           146.593 |      6.8216  |
|          3 |                        7 |                     30 |          4.55244 |              241.504 |           146.16  |      6.84179 |
|          4 |                        7 |                     30 |          4.49685 |              240.559 |           144.406 |      6.92494 |
|          5 |                        5 |                     30 |          4.5107  |              236.007 |           144.959 |      6.89852 |
|          6 |                        5 |                     30 |          4.49506 |              233.637 |           144.602 |      6.91555 |
|          7 |                        7 |                     30 |          4.54506 |              242.938 |           145.966 |      6.8509  |
|          8 |                        6 |                     30 |          4.52085 |              246.083 |           145.094 |      6.89209 |
|          9 |                        6 |                     30 |          4.50877 |              248.334 |           144.317 |      6.9292  |
|         10 |                        6 |                     30 |          4.52617 |              249.269 |           145.1   |      6.89182 |

# PHX

```python run_awq.py --model_name microsoft/phi-2 --task decode --algorithm pergrp```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        7 |                     30 |          7.36599 |              443.699 |           236.042 |      4.23653 |
|          2 |                        8 |                     30 |          7.12104 |              379.011 |           230.266 |      4.3428  |
|          3 |                        7 |                     30 |          7.12575 |              356.105 |           231.131 |      4.32655 |
|          4 |                        7 |                     30 |          7.22359 |              362.684 |           234.277 |      4.26845 |
|          5 |                        5 |                     30 |          7.25634 |              361.051 |           235.448 |      4.24723 |
|          6 |                        5 |                     30 |          7.26938 |              365.754 |           235.859 |      4.23982 |
|          7 |                        7 |                     30 |          7.27765 |              373.563 |           235.875 |      4.23953 |
|          8 |                        6 |                     30 |          7.28173 |              369.771 |           236.026 |      4.23682 |
|          9 |                        6 |                     30 |          7.28237 |              375.114 |           235.879 |      4.23946 |
|         10 |                        6 |                     30 |          7.35879 |              370.835 |           238.659 |      4.19008 |

```python run_awq.py --model_name microsoft/phi-2 --task decode --algorithm pergrp --flash_attention_plus```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        7 |                     30 |          6.64827 |              373.147 |           212.695 |      4.70156 |
|          2 |                        8 |                     30 |          6.62264 |              356.546 |           213.799 |      4.67729 |
|          3 |                        7 |                     30 |          6.56164 |              334.879 |           212.155 |      4.71354 |
|          4 |                        7 |                     30 |          6.56202 |              342.741 |           212.231 |      4.71184 |
|          5 |                        5 |                     30 |          6.52558 |              326.596 |           211.395 |      4.73048 |
|          6 |                        5 |                     30 |         16.3206  |              329.897 |           213.174 |      4.69101 |
|          7 |                        7 |                     30 |          6.54643 |              340.306 |           211.748 |      4.72259 |
|          8 |                        6 |                     30 |          6.54067 |              343.679 |           211.508 |      4.72795 |
|          9 |                        6 |                     30 |          6.53734 |              343.781 |           211.33  |      4.73193 |
|         10 |                        6 |                     30 |          6.55967 |              342.433 |           212.136 |      4.71396 |

# STX B0 (With MCDM)

```python run_awq.py --model_name microsoft/phi-2 --task decode --algorithm pergrp```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        7 |                     30 |          4.07097 |              317.469 |           125.843 |      7.94641 |
|          2 |                        8 |                     30 |          4.04274 |              290.182 |           126.548 |      7.90217 |
|          3 |                        7 |                     30 |          4.01496 |              255.65  |           126.672 |      7.89441 |
|          4 |                        7 |                     30 |          4.01723 |              259.229 |           126.611 |      7.89818 |
|          5 |                        5 |                     30 |          3.97594 |              218.501 |           126.424 |      7.90988 |
|          6 |                        5 |                     30 |          3.95241 |              218.175 |           126.039 |      7.93403 |
|          7 |                        7 |                     22 |          2.96909 |              262.145 |           126.046 |      7.93363 |
|          8 |                        6 |                     30 |          3.95739 |              244.096 |           125.245 |      7.98434 |
|          9 |                        6 |                     30 |          3.96588 |              236.1   |           125.821 |      7.94779 |
|         10 |                        6 |                     30 |          3.98332 |              237.078 |           126.358 |      7.91405 |

```python run_awq.py --model_name microsoft/phi-2 --task decode --algorithm pergrp --flash_attention_plus```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        7 |                     30 |          3.74102 |              334.636 |           114.052 |      8.76791 |
|          2 |                        8 |                     30 |          3.61246 |              267.764 |           112.501 |      8.8888  |
|          3 |                        7 |                     30 |          3.64506 |              245.089 |           114.268 |      8.75138 |
|          4 |                        7 |                     30 |          3.64333 |              251     |           114.128 |      8.76205 |
|          5 |                        5 |                     30 |          3.61132 |              212.935 |           114.347 |      8.74534 |
|          6 |                        5 |                     30 |          3.59228 |              207.857 |           113.917 |      8.77833 |
|          7 |                        7 |                     30 |          3.63823 |              244.489 |           114.143 |      8.76092 |
|          8 |                        6 |                     30 |          3.62263 |              227.719 |           114.256 |      8.75231 |
|          9 |                        6 |                     30 |          3.61162 |              222.457 |           113.991 |      8.77266 |
|         10 |                        6 |                     30 |          3.60855 |              222.151 |           113.954 |      8.7755  |

```
Phi2ModelEval(
  (model): PhiModel(
    (embed_tokens): Embedding(51200, 2560)
    (embed_dropout): Dropout(p=0.0, inplace=False)
    (layers): ModuleList(
      (0-31): 32 x PhiDecoderLayer(
        (self_attn): PhiAttention(
          (q_proj): ryzenAI.QLinearPerGrp(in_features:2560, out_features:2560, bias:torch.Size([2560]), device:aie, w_bit:4 group_size:128  )
          (k_proj): ryzenAI.QLinearPerGrp(in_features:2560, out_features:2560, bias:torch.Size([2560]), device:aie, w_bit:4 group_size:128  )
          (v_proj): ryzenAI.QLinearPerGrp(in_features:2560, out_features:2560, bias:torch.Size([2560]), device:aie, w_bit:4 group_size:128  )
          (dense): ryzenAI.QLinearPerGrp(in_features:2560, out_features:2560, bias:torch.Size([2560]), device:aie, w_bit:4 group_size:128  )
          (rotary_emb): PhiRotaryEmbedding()
        )
        (mlp): PhiMLP(
          (activation_fn): NewGELUActivation()
          (fc1): ryzenAI.QLinearPerGrp(in_features:2560, out_features:10240, bias:torch.Size([10240]), device:aie, w_bit:4 group_size:128  )
          (fc2): ryzenAI.QLinearPerGrp(in_features:10240, out_features:2560, bias:torch.Size([2560]), device:aie, w_bit:4 group_size:128  )
        )
        (input_layernorm): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
        (resid_dropout): Dropout(p=0.1, inplace=False)
      )
    )
    (final_layernorm): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): ryzenAI.QLinearPerGrp(in_features:2560, out_features:51200, bias:torch.Size([51200]), device:aie, w_bit:4 group_size:128  )
)
model.mode_name: phi-2
****************************************
prompt: What is the meaning of life?
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
response: What is the meaning of life?

Answer: The meaning of life is a philosophical question that has been debated for centuries.

Exercise 3:
What is the difference
****************************************
prompt: Tell me something you don't know.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
response: Tell me something you don't know.

The man's eyes widened, and he looked around nervously. He seemed to be searching for an escape route.

"I don't
****************************************
prompt: What does Xilinx do?
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
response: What does Xilinx do?
Xilinx is a company that makes special computer chips that can do many different things. They are very smart and can help us do many different
****************************************
```
