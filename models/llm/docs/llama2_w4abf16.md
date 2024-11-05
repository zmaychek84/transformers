# w4abf16 with AWQ

AWQ enables 3-bit and 4-bit weights for LLMs. This reduces model size of Llama2 7B from 52-58% of int8 model depending on group size and whether the last layer is quantized.

AWQ scales are obtained from MIT-Han-Lab. THe script also contains hooks to calculate scales instead of using precomputed scales. All layers other than "lm_head" are quantized using AWQ. This software stack of RyzenAI can also quantize lm_head layer using per group quantization scheme, with group sizes varying from 32-256.

Linear layers are replaced with QLinearPerGrp custom int4 compute layer afetr AWQ.
![AWQ pic](../figures/llama2_awqplus.png)

Matmul grouping in done when flash attention is enabled, which reduces number of dispatches to AIE by grouping matmuls of QKV matrices into a single grouped matmul.
This in addition to static memory allocation in token phases provides 8-10% better performance than vanilla attention.

![AWQ FA](../figures/llama2_awq_fa.png)

## Save quantized checkpoints
```
python run_awq.py --model_name llama-2-7b --task quantize --algorithm awqplus
python run_awq.py --model_name llama-2-7b --task quantize --algorithm awq
```
3-bit model has same latency as 4-bit model due to same bit packing. But 4-bit model has better accuracy.

## Sample outputs on STX
```
****************************************
<s>What is the meaning of life? This is a question that has been asked by philosophers and theologians for centuries. It is a question that has no easy answer, and one that is still debated today.
There are many different theories about the meaning of life. Some people believe that the meaning of life is to find
****************************************
<s>Tell me something you don't know.
I don't know how to play the piano.
I don't know how to play the guitar.
I don't know how to play the violin.
I don't know how to play the drums.
I don't know how to play the sa
****************************************
<s>What does Xilinx do?
Xilinx is a semiconductor company that designs and sells programmable logic devices (PLDs), including field-programmable gate arrays (FPGAs), complex programmable logic devices (CPLDs), and 3D ICs.
How do
****************************************
<s>What is the mass of earth?
How much does the earth weigh?
How much does the earth weigh in tons?
How much does the earth weigh in pounds?
How much does the earth weigh in kilograms?
How much does the earth weigh in grams?
How much does
****************************************
<s>Who is Gilgamesh?
Gilgamesh is the king of Uruk, the most powerful city in Mesopotamia. He is two-thirds god and one-third human. He is the son of the goddess Ninsun and Lugalbanda, the king of U
****************************************
```

## Latency : STX B0

### AWQ Plus

```python run_awq.py --model_name llama-2-7b --algorithm awqplus --task decode --fast_attention --fast_mlp --fast_norm --fast_decoder```
|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        8 |                     60 |          9.23344 |              347.243 |           143.679 |      6.95996 |
|          2 |                       10 |                     60 |          9.38206 |              342.701 |           149.476 |      6.69004 |
|          3 |                        8 |                     60 |         15.7408  |              397.907 |           255.59  |      3.91252 |
|          4 |                        8 |                     60 |          9.20595 |              462.665 |           146.189 |      6.84045 |
|          5 |                        6 |                     60 |          8.89612 |              302.699 |           142.741 |      7.00572 |
|          6 |                        5 |                     60 |          8.10499 |              296.412 |           129.723 |      7.70875 |
|          7 |                        9 |                     60 |          8.10694 |              319.256 |           129.29  |      7.73457 |
|          8 |                        8 |                     60 |          8.05763 |              311.018 |           128.731 |      7.76812 |
|          9 |                        9 |                     60 |          8.07345 |              319.488 |           128.799 |      7.76402 |
|         10 |                        7 |                     60 |          8.09992 |              311.107 |           129.339 |      7.7316  |

```python run_awq.py --model_name llama-2-7b --algorithm awqplus --task benchmark --fast_attention --fast_mlp --fast_norm --fast_decoder```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        5 |                     60 |          11.5468 |              552.707 |           170.868 |      5.85248 |
|          2 |                       78 |                     60 |          12.9581 |             1926.6   |           183.476 |      5.45031 |
|          3 |                      128 |                     60 |          12.125  |             1900.71  |           169.959 |      5.88376 |
|          4 |                      140 |                     60 |          12.2139 |             2598.4   |           158.339 |      6.31556 |
|          5 |                      256 |                     60 |          13.8542 |             1670.3   |           201.774 |      4.95604 |
|          6 |                      293 |                     60 |          13.2688 |             3726.6   |           158.763 |      6.2987  |
|          7 |                      372 |                     60 |          14.1908 |             4070.79  |           168.284 |      5.94234 |
|          8 |                      512 |                     60 |          14.5883 |             2355.35  |           201.498 |      4.96283 |
|          9 |                      580 |                     60 |          18.7181 |             6489.83  |           202.984 |      4.9265  |
|         10 |                      717 |                     60 |          20.9692 |             7904.24  |           213.62  |      4.68122 |
|         11 |                      790 |                     60 |          20.4631 |             7074.36  |           221.75  |      4.50959 |
|         12 |                      800 |                     60 |          21.3361 |             7306.37  |           232.482 |      4.30141 |
|         13 |                      900 |                     60 |          22.8457 |             8429.35  |           234.319 |      4.26768 |
|         14 |                     1024 |                     60 |          19.5022 |             4247.45  |           253.95  |      3.93778 |
|         15 |                     1050 |                     60 |          27.0448 |            11213.4   |           259.148 |      3.85879 |
|         16 |                     1320 |                     60 |          29.5758 |            13535.6   |           266.703 |      3.74948 |
|         17 |                     1537 |                     60 |          32.1058 |            13858.5   |           293.644 |      3.40548 |
|         18 |                     1580 |                     60 |          31.4236 |            13201.2   |           292.94  |      3.41367 |
|         19 |                     1670 |                     60 |          33.9553 |            14195.4   |           316.369 |      3.16087 |
|         20 |                     1900 |                     60 |          35.2891 |            16231     |           308.438 |      3.24214 |
|         21 |                     2048 |                     60 |          29.9226 |             7954.15  |           351.844 |      2.84217 |


```python run_awq.py --model_name llama-2-7b --algorithm awqplus --task benchmark_exact --fast_attention --fast_mlp --fast_norm --fast_decoder```
|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                      128 |                     60 |          10.7363 |              1426.3  |           148.884 |      6.71663 |
|          2 |                      256 |                     60 |          10.9643 |              1459.87 |           157.324 |      6.35631 |
|          3 |                      512 |                     60 |          13.0511 |              2336.62 |           175.156 |      5.70918 |
|          4 |                     1024 |                     60 |          16.4676 |              3383    |           216.139 |      4.62665 |
|          5 |                     2048 |                     60 |          23.4814 |              5608.1  |           286.572 |      3.48952 |

```python run_awq.py --model_name llama-2-7b --fast_attention --fast_mlp --fast_norm --fast_decoder --task profilemodel256 --algorithm awqplus```
|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                      256 |                     60 |          9.90756 |              868.801 |           145.264 |      6.88403 |

```python run_awq.py --model_name llama-2-7b --fast_attention --fast_mlp --fast_norm --fast_decoder --task profilemodel512 --algorithm awqplus```
|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                      512 |                     60 |          11.6315 |              1209.59 |            166.82 |      5.99449 |

```python run_awq.py --model_name llama-2-7b --fast_attention --fast_mlp --fast_norm  --fast_decoder --task profilemodel1k --algorithm awqplus```
|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                     1024 |                     60 |           15.462 |              2296.38 |           209.989 |      4.76216 |

```python run_awq.py --model_name llama-2-7b --fast_attention --fast_mlp --fast_norm --fast_decoder --task profilemodel2k --algorithm awqplus```
|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                     2048 |                     60 |           22.701 |              4896.71 |           282.071 |      3.54521 |
