# state-spaces/mamba-1.4b-hf
```
python run_awq.py --model_name state-spaces/mamba-1.4b-hf --task quantize --algorithm pergrp --group_size 32
python run_awq.py --model_name state-spaces/mamba-1.4b-hf --task quantize --algorithm pergrp --group_size 64


python run_awq.py --model_name state-spaces/mamba-1.4b-hf --task decode --algorithm pergrp --group_size 32

MambaModelEval(
  (backbone): MambaModel(
    (embeddings): Embedding(50280, 2048)
    (layers): ModuleList(
      (0-47): 48 x MambaBlock(
        (norm): MambaRMSNorm()
        (mixer): MambaMixer(
          (conv1d): Conv1d(4096, 4096, kernel_size=(4,), stride=(1,), padding=(3,), groups=4096)
          (act): SiLU()
          (in_proj): ryzenAI.QLinearPerGrp(in_features:2048, out_features:8192, bias:None, device:aie, w_bit:4 group_size:32  )
          (x_proj): ryzenAI.QLinearPerGrp(in_features:4096, out_features:160, bias:None, device:aie, w_bit:4 group_size:32  )
          (dt_proj): ryzenAI.QLinearPerGrp(in_features:128, out_features:4096, bias:torch.Size([4096]), device:aie, w_bit:4 group_size:32  )
          (out_proj): ryzenAI.QLinearPerGrp(in_features:4096, out_features:2048, bias:None, device:aie, w_bit:4 group_size:32  )
        )
      )
    )
    (norm_f): MambaRMSNorm()
  )
  (lm_head): ryzenAI.QLinearPerGrp(in_features:2048, out_features:50280, bias:None, device:aie, w_bit:4 group_size:32  )
)
model.mode_name: mamba-1.4b-hf
****************************************
prompt: What is the meaning of life?
What is the meaning of life? What is the meaning of your life? What's the endgame? Life is an unending stream of bad things that keep you from what you could
response: What is the meaning of life? What is the meaning of your life? What's the endgame? Life is an unending stream of bad things that keep you from what you could
****************************************
prompt: Tell me something you don't know.
Tell me something you don't know. If you don't know, you haven't a right to talk to me."

"I don't know everything I should."

His
response: Tell me something you don't know. If you don't know, you haven't a right to talk to me."

"I don't know everything I should."

His
****************************************
prompt: What does Xilinx do?
What does Xilinx do? Well, when you program an MCU, the program moves from memory to the processor. However, you must often send something to an MCU from
response: What does Xilinx do? Well, when you program an MCU, the program moves from memory to the processor. However, you must often send something to an MCU from
****************************************
...
```
# PHX

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        7 |                     30 |         10.0826  |              573.545 |           324.775 |      3.07906 |
|          2 |                        8 |                     30 |          9.96925 |              537.655 |           323.187 |      3.09418 |
|          3 |                        7 |                     30 |          9.92239 |              506.394 |           322.633 |      3.0995  |
|          4 |                        7 |                     30 |          9.9822  |              505.797 |           324.841 |      3.07843 |
|          5 |                        5 |                     30 |          9.92251 |              465.228 |           324.056 |      3.08588 |
|          6 |                        4 |                     30 |          9.85084 |              440.414 |           322.522 |      3.10057 |
|          7 |                        7 |                     30 |          9.96144 |              525.401 |           323.308 |      3.09303 |
|          8 |                        6 |                     30 |          9.94201 |              491.939 |           323.8   |      3.08833 |
|          9 |                        6 |                     30 |          9.99056 |              494.124 |           325.455 |      3.07262 |
|         10 |                        6 |                     30 |          9.94774 |              485.627 |           324.259 |      3.08396 |

```python run_awq.py --model_name state-spaces/mamba-1.4b-hf --task decode --algorithm pergrp --group_size 64```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        7 |                     30 |         10.1756  |              559.877 |           329.41  |      3.03573 |
|          2 |                        8 |                     30 |         10.1862  |              541.852 |           330.538 |      3.02537 |
|          3 |                        7 |                     30 |         10.0758  |              505.367 |           327.882 |      3.04988 |
|          4 |                        7 |                     30 |         10.1062  |              529.07  |           328.243 |      3.04652 |
|          5 |                        5 |                     30 |          9.99386 |              470.93  |           326.377 |      3.06394 |
|          6 |                        4 |                     30 |         10.0207  |              453.154 |           327.841 |      3.05026 |
|          7 |                        7 |                     30 |         10.0956  |              519.216 |           328.129 |      3.04758 |
|          8 |                        6 |                     30 |         10.049   |              497.381 |           327.306 |      3.05525 |
|          9 |                        6 |                     30 |         10.0366  |              492.955 |           327.098 |      3.05718 |
|         10 |                        6 |                     30 |         10.129   |              507.572 |           329.782 |      3.03231 |

## HPT (With MCDM)

```python run_awq.py --model_name state-spaces/mamba-1.4b-hf --task decode --algorithm pergrp --group_size 32```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        7 |                     30 |          7.02369 |              454.48  |           223.538 |      4.47351 |
|          2 |                        8 |                     30 |          6.9634  |              438.769 |           222.898 |      4.48636 |
|          3 |                        7 |                     30 |          6.92327 |              406.904 |           222.546 |      4.49346 |
|          4 |                        7 |                     30 |          6.98634 |              402.188 |           224.992 |      4.44461 |
|          5 |                        5 |                     30 |          6.87313 |              353.359 |           222.708 |      4.49019 |
|          6 |                        4 |                     30 |          6.87017 |              340.106 |           222.925 |      4.48581 |
|          7 |                        7 |                     30 |          6.96665 |              400.099 |           224.332 |      4.45769 |
|          8 |                        6 |                     30 |          6.94831 |              384.155 |           223.804 |      4.46821 |
|          9 |                        6 |                     30 |          6.94448 |              388.64  |           223.989 |      4.46451 |
|         10 |                        6 |                     30 |          6.9131  |              379.56  |           223.168 |      4.48092 |

```python run_awq.py --model_name state-spaces/mamba-1.4b-hf --task decode --algorithm pergrp --group_size 64```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        7 |                     30 |          6.9747  |              426.208 |           223.408 |      4.47611 |
|          2 |                        8 |                     30 |          6.96205 |              431.654 |           223.045 |      4.48339 |
|          3 |                        7 |                     30 |          6.93937 |              402.801 |           223.188 |      4.48052 |
|          4 |                        7 |                     30 |          6.93437 |              404.896 |           223.068 |      4.48294 |
|          5 |                        5 |                     30 |          6.91546 |              354.059 |           224.152 |      4.46126 |
|          6 |                        4 |                     30 |          6.853   |              327.811 |           222.636 |      4.49164 |
|          7 |                        7 |                     30 |          6.94959 |              410.699 |           223.296 |      4.47836 |
|          8 |                        6 |                     30 |          6.93723 |              379.378 |           223.953 |      4.46522 |
|          9 |                        6 |                     30 |          6.95623 |              378.82  |           224.45  |      4.45533 |
|         10 |                        6 |                     30 |          6.95987 |              384.281 |           224.415 |      4.45603 |

## STX B0 (With MCDM)

```python run_awq.py --model_name state-spaces/mamba-1.4b-hf --task decode --algorithm pergrp --group_size 32```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        7 |                     30 |          5.91357 |              550.957 |           181.383 |      5.51319 |
|          2 |                        8 |                     30 |          5.90331 |              553.963 |           181.678 |      5.50423 |
|          3 |                        7 |                     30 |          5.84474 |              502.429 |           181.373 |      5.5135  |
|          4 |                        7 |                     30 |          5.82286 |              494.982 |           181.203 |      5.51868 |
|          5 |                        5 |                     30 |          5.74065 |              410.181 |           181.257 |      5.51702 |
|          6 |                        4 |                     30 |          5.71104 |              372.794 |           181.579 |      5.50723 |
|          7 |                        7 |                     30 |          5.83344 |              494.469 |           181.55  |      5.50812 |
|          8 |                        6 |                     30 |          5.78825 |              455.113 |           181.377 |      5.51337 |
|          9 |                        6 |                     30 |          5.79076 |              445.838 |           181.809 |      5.50027 |
|         10 |                        6 |                     30 |          5.77979 |              452.625 |           181.131 |      5.52085 |

```python run_awq.py --model_name state-spaces/mamba-1.4b-hf --task decode --algorithm pergrp --group_size 64```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        7 |                     30 |          5.70574 |              455.099 |           178.245 |      5.61024 |
|          2 |                        8 |                     30 |          5.84032 |              532.138 |           180.458 |      5.54147 |
|          3 |                        7 |                     30 |          5.77303 |              482.971 |           179.812 |      5.56136 |
|          4 |                        7 |                     30 |          5.77487 |              483.517 |           179.975 |      5.55634 |
|          5 |                        5 |                     30 |          5.68917 |              410.483 |           179.522 |      5.57036 |
|          6 |                        4 |                     30 |          5.62915 |              366.671 |           178.947 |      5.58826 |
|          7 |                        7 |                     30 |          5.77162 |              481.534 |           179.579 |      5.56859 |
|          8 |                        6 |                     30 |          5.71962 |              446.57  |           179.292 |      5.5775  |
|          9 |                        6 |                     30 |          5.74125 |              448.999 |           180.012 |      5.5552  |
|         10 |                        6 |                     30 |          5.75278 |              461.69  |           179.99  |      5.55588 |
