# state-spaces/mamba-2.8b-hf

```
python run_awq.py --model_name state-spaces/mamba-2.8b-hf --task quantize --algorithm pergrp --group_size 32
python run_awq.py --model_name state-spaces/mamba-2.8b-hf --task decode --algorithm pergrp --group_size 32

MambaModelEval(
  (backbone): MambaModel(
    (embeddings): Embedding(50280, 2560)
    (layers): ModuleList(
      (0-63): 64 x MambaBlock(
        (norm): MambaRMSNorm()
        (mixer): MambaMixer(
          (conv1d): Conv1d(5120, 5120, kernel_size=(4,), stride=(1,), padding=(3,), groups=5120)
          (act): SiLU()
          (in_proj): ryzenAI.QLinearPerGrp(in_features:2560, out_features:10240, bias:None, device:aie, w_bit:4 group_size:32  )
          (x_proj): ryzenAI.QLinearPerGrp(in_features:5120, out_features:192, bias:None, device:aie, w_bit:4 group_size:32  )
          (dt_proj): ryzenAI.QLinearPerGrp(in_features:160, out_features:5120, bias:torch.Size([5120]), device:aie, w_bit:4 group_size:32  )
          (out_proj): ryzenAI.QLinearPerGrp(in_features:5120, out_features:2560, bias:None, device:aie, w_bit:4 group_size:32  )
        )
      )
    )
    (norm_f): MambaRMSNorm()
  )
  (lm_head): ryzenAI.QLinearPerGrp(in_features:2560, out_features:50280, bias:None, device:aie, w_bit:4 group_size:32  )
)
model.mode_name: mamba-2.8b-hf
****************************************
prompt: What is the meaning of life?
What is the meaning of life? What makes the gods sing? What is life's meaning? Does life have meaning? Do we have meaning as we live and die? What makes life
response: What is the meaning of life? What makes the gods sing? What is life's meaning? Does life have meaning? Do we have meaning as we live and die? What makes life
****************************************
prompt: Tell me something you don't know.
Tell me something you don't know. If it's the same woman, I'll know it. Something you think was real but might have been a hallucination or a delusion—something
response: Tell me something you don't know. If it's the same woman, I'll know it. Something you think was real but might have been a hallucination or a delusion—something
****************************************
prompt: What does Xilinx do?
What does Xilinx do? Well, when you connect an internal memory, XDC automatically connects each pins to each storage element. If we were to connect a 512 bits DSP
response: What does Xilinx do? Well, when you connect an internal memory, XDC automatically connects each pins to each storage element. If we were to connect a 512 bits DSP
****************************************

```

# PHX

```python run_awq.py --model_name state-spaces/mamba-2.8b-hf --task decode --algorithm pergrp --group_size 32```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        7 |                     30 |          19.5679 |              987.77  |           638.135 |      1.56707 |
|          2 |                        8 |                     30 |          19.3997 |              915.605 |           635.259 |      1.57416 |
|          3 |                        7 |                     30 |          19.3821 |              879.306 |           635.909 |      1.57255 |
|          4 |                        7 |                     30 |          19.3763 |              888.587 |           635.49  |      1.57359 |
|          5 |                        5 |                     30 |          19.396  |              880.832 |           636.251 |      1.57171 |
|          6 |                        4 |                     30 |          19.3501 |              838.145 |           636.275 |      1.57165 |
|          7 |                        7 |                     30 |          19.4228 |              882.493 |           637.138 |      1.56952 |
|          8 |                        6 |                     30 |          19.4488 |              918.921 |           636.741 |      1.5705  |
|          9 |                        6 |                     30 |          19.4366 |              908.849 |           636.845 |      1.57024 |
|         10 |                        6 |                     30 |          19.4626 |              910.138 |           637.661 |      1.56823 |

## HPT (With MCDM)

```python run_awq.py --model_name state-spaces/mamba-2.8b-hf --task decode --algorithm pergrp --group_size 32```


|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        7 |                     30 |          13.1058 |              739.041 |           423.598 |      2.36073 |
|          2 |                        8 |                     30 |          13.0992 |              684.958 |           425.855 |      2.34821 |
|          3 |                        7 |                     30 |          13.0386 |              652.969 |           424.878 |      2.35362 |
|          4 |                        7 |                     30 |          13.0776 |              656.104 |           426.228 |      2.34616 |
|          5 |                        5 |                     30 |          13.0463 |              641.919 |           425.238 |      2.35163 |
|          6 |                        4 |                     30 |          13.0463 |              605.877 |           426.888 |      2.34254 |
|          7 |                        7 |                     30 |          13.0935 |              658.305 |           426.592 |      2.34416 |
|          8 |                        6 |                     30 |          13.1114 |              683.411 |           426.091 |      2.34692 |
|          9 |                        6 |                     30 |          13.1075 |              688.921 |           425.968 |      2.34759 |
|         10 |                        6 |                     30 |          13.1337 |              684.596 |           427.134 |      2.34118 |

## STX B0 (With MCDM)

```python run_awq.py --model_name state-spaces/mamba-2.8b-hf --task decode --algorithm pergrp --group_size 32```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        7 |                     30 |          10.64   |              916.183 |           332.245 |      3.00983 |
|          2 |                        8 |                     30 |          10.6542 |              958.511 |           331.68  |      3.01495 |
|          3 |                        7 |                     30 |          10.6198 |              886.715 |           332.807 |      3.00475 |
|          4 |                        7 |                     30 |          10.6139 |              895.171 |           332.555 |      3.00702 |
|          5 |                        5 |                     30 |          10.4623 |              754.332 |           332.069 |      3.01143 |
|          6 |                        4 |                     30 |          10.4642 |              705.51  |           333.85  |      2.99536 |
|          7 |                        7 |                     30 |          10.5945 |              887.946 |           332.039 |      3.0117  |
|          8 |                        6 |                     30 |          10.5459 |              794.72  |           333.566 |      2.9979  |
|          9 |                        6 |                     30 |          10.5375 |              822.452 |           331.795 |      3.01391 |
|         10 |                        6 |                     30 |          10.5341 |              829.853 |           331.992 |      3.01212 |

*  [Different version than HF - for reference - this is not used in this repo](https://github.com/alxndrTL/mamba.py)
