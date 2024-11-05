# Llama2 - w8a8, w8a16 with SmoothQuant

# Step 1: Save model
```
python run_smoothquant.py --model_name llama-2-7b --task quantize
```

# Step 2: Use run.py to decode, profile or analyze performance

```python run_smoothquant.py --help```

## Decode prompts
```
python run_smoothquant.py --model_name llama-2-7b --task decode --target aie

...
****************************************
prompt: What is the meaning of life?
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
response: What is the meaning of life? This is a question that has puzzled philosophers, theologians, scientists, and many other thinkers throughout history. Here are some possible answers:
1. Evolutionary perspective: From an evolutionary perspective, the meaning of life
****************************************
prompt: Tell me something you don't know.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
response: Tell me something you don't know.
 I'm not sure I understand what you're asking. Can you explain?

You're right, I apologize for the confusion. I was asking you to tell me something that you don't know. It's
****************************************
prompt: What does Xilinx do?
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
response: What does Xilinx do?
Xilinx is a technology company that designs and manufactures programmable logic devices (PLDs) and field-programmable gate arrays (FPGAs). PLDs are integrated circuits that can be programmed to perform
****************************************
prompt: What is the mass of earth?
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
response: What is the mass of earth?

The mass of Earth is approximately 5.972 x 10^24 kilograms. This is the total mass of all the matter that makes up our planet, including the solid Earth, the oceans,
****************************************
...

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        8 |                     30 |          24.9415 |             1003.36  |           824.246 |      1.21323 |
|          2 |                       10 |                     30 |          24.9095 |              973.884 |           824.323 |      1.21312 |
|          3 |                        8 |                     30 |          25.0221 |              958.436 |           828.812 |      1.20655 |
|          4 |                        8 |                     30 |          25.0218 |              955.663 |           828.82  |      1.20654 |
|          5 |                        6 |                     30 |          25.0222 |              932.749 |           829.651 |      1.20533 |
|          6 |                        5 |                     30 |          24.8371 |              922.185 |           823.694 |      1.21404 |
|          7 |                        9 |                     30 |          25.0538 |              980.29  |           829.064 |      1.20618 |
|          8 |                        8 |                     30 |          24.9777 |              955.552 |           827.277 |      1.20879 |
|          9 |                        9 |                     30 |          25.1422 |              975.003 |           832.371 |      1.20139 |
|         10 |                        7 |                     30 |          25.022  |              949.967 |           829.036 |      1.20622 |
```

## PHX w8a8
### No optimizations

```python run_smoothquant.py --model_name llama-2-7b --task benchmark --target aie```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        4 |                     60 |          52.788  |              1032.15 |           873.93  |      1.14426 |
|          2 |                        8 |                     60 |          52.7647 |              1031.81 |           873.776 |      1.14446 |
|          3 |                       16 |                     60 |          62.0788 |              1112.14 |           876.745 |      1.14058 |
|          4 |                       32 |                     60 |          53.2743 |              1310.41 |           877.584 |      1.13949 |
|          5 |                       64 |                     60 |          53.8229 |              1928.1  |           876.287 |      1.14118 |
|          6 |                      128 |                     60 |          56.2297 |              3726.48 |           886.685 |      1.1278  |
|          7 |                      256 |                     60 |          60.836  |              7539.44 |           899.623 |      1.11158 |

```python run_smoothquant.py --model_name llama-2-7b --task benchmark_long --target aie```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                      512 |                     60 |          69.0161 |              15386.8 |           900.356 |     1.11067  |
|          2 |                     1024 |                     60 |          91.2709 |              34394.3 |           954.752 |     1.04739  |
|          3 |                     1536 |                     60 |         116.238  |              55958.3 |          1015.38  |     0.98485  |
|          4 |                     2000 |                     60 |         143.019  |              79606.1 |          1058.77  |     0.944495 |
|          5 |                     3000 |                     60 |         206.81   |             137722   |          1163.67  |     0.859352 |
|          6 |                     4000 |                     60 |         300.506  |             211550   |          1466.44  |     0.681922 |

### With Optimizations

```python run_smoothquant.py --model_name llama-2-7b --task benchmark --target aie --flash_attention_plus```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        4 |                     60 |          51.7585 |              1015.91 |           856.715 |      1.16725 |
|          2 |                        8 |                     60 |          51.8869 |              1018.98 |           859.091 |      1.16402 |
|          3 |                       16 |                     60 |          51.9262 |              1077.99 |           858.668 |      1.16459 |
|          4 |                       32 |                     60 |          52.0376 |              1304.67 |           856.569 |      1.16745 |
|          5 |                       64 |                     60 |          52.7856 |              1772.54 |           860.83  |      1.16167 |
|          6 |                      128 |                     60 |          54.9102 |              3515.05 |           866.822 |      1.15364 |
|          7 |                      256 |                     60 |          59.612  |              7075.72 |           884.586 |      1.13047 |

```python run_smoothquant.py --model_name llama-2-7b --task benchmark_long --target aie --flash_attention_plus```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                      512 |                     60 |          67.3747 |              13830.3 |           899.098 |     1.11223  |
|          2 |                     1024 |                     60 |          85.7291 |              28790.1 |           950.83  |     1.05171  |
|          3 |                     1536 |                     60 |         106.167  |              45694.7 |          1005.59  |     0.994438 |
|          4 |                     2000 |                     60 |         127.232  |              63889.4 |          1049.29  |     0.953027 |
|          5 |                     3000 |                     60 |         175.721  |             106148   |          1145.91  |     0.872667 |
|          6 |                     4000 |                     60 |         225.579  |             150265   |          1234.08  |     0.810319 |


## STX B0 w8a16 (With MCDM)
### No optimizations

```python run_smoothquant.py --model_name llama-2-7b --task benchmark --target aie --precision w8a16```
|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        4 |                     60 |          13.9992 |              998.907 |           216.218 |      4.62496 |
|          2 |                        8 |                     60 |          14.0128 |             1016.83  |           216.035 |      4.62888 |
|          3 |                       16 |                     60 |          14.3395 |             1348.55  |           216.081 |      4.6279  |
|          4 |                       32 |                     60 |          14.7582 |             2381.99  |           206.468 |      4.84336 |
|          5 |                       64 |                     60 |          16.0211 |             3828.07  |           203.286 |      4.91918 |
|          6 |                      128 |                     60 |          18.1269 |             5027.62  |           218.197 |      4.58301 |
|          7 |                      256 |                     60 |          22.6202 |             8274.94  |           238.581 |      4.19145 |

```python run_smoothquant.py --model_name llama-2-7b --task benchmark_long --target aie --precision w8a16```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                      512 |                     60 |          35.4601 |              18617.3 |           278.586 |     3.58956  |
|          2 |                     1024 |                     60 |          66.4419 |              45340.5 |           351.916 |     2.84159  |
|          3 |                     1536 |                     60 |         360.96   |              82373.1 |           427.525 |     2.33904  |
|          4 |                     2000 |                     60 |         152.826  |             122377   |           494.966 |     2.02034  |
|          5 |                     3000 |                     60 |         279.065  |             239088   |           653.784 |     1.52956  |
|          6 |                     4000 |                     60 |         470.923  |             386847   |          1405.56  |     0.711463 |

### With Optimizations

```python run_smoothquant.py --model_name llama-2-7b --task  benchmark --target aie --flash_attention_plus --precision w8a16```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        4 |                     60 |          14.7823 |              952.653 |           230.6   |      4.33651 |
|          2 |                        8 |                     60 |          14.9603 |             1046.15  |           232.091 |      4.30865 |
|          3 |                       16 |                     60 |          15.2968 |             1443     |           230.901 |      4.33085 |
|          4 |                       32 |                     60 |          15.7714 |             2289.32  |           224.102 |      4.46224 |
|          5 |                       64 |                     60 |          16.9476 |             3556.11  |           222.419 |      4.49601 |
|          6 |                      128 |                     60 |          19.1114 |             4837.58  |           236.474 |      4.22879 |
|          7 |                      256 |                     60 |          23.4283 |             7927.36  |           256.091 |      3.90486 |

```python run_smoothquant.py --model_name llama-2-7b --task  benchmark_long --target aie --flash_attention_plus --precision w8a16```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                      512 |                     60 |          34.9882 |              16758.8 |           298.385 |      3.35137 |
|          2 |                     1024 |                     60 |          61.633  |              38659.9 |           370.651 |      2.69796 |
|          3 |                     1536 |                     60 |          96.2063 |              68218.3 |           448.359 |      2.23035 |
|          4 |                     2000 |                     60 |         237.324  |              96291.8 |           528.858 |      1.89087 |
|          5 |                     3000 |                     60 |         298.896  |             188250   |           656.132 |      1.52408 |
|          6 |                     4000 |                     60 |         362.253  |             311838   |           793.177 |      1.26075 |

## STX B0 w8a8
### No optimizations

```python run_smoothquant.py --model_name llama-2-7b --task benchmark --target aie```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        4 |                     60 |          25.6936 |              1242.14 |           410.922 |      2.43355 |
|          2 |                        8 |                     60 |          25.7667 |              1336.72 |           410.622 |      2.43533 |
|          3 |                       16 |                     60 |          26.0959 |              1972.03 |           405.154 |      2.4682  |
|          4 |                       32 |                     60 |          26.7442 |              2722.28 |           403.766 |      2.47668 |
|          5 |                       64 |                     60 |          28.0511 |              3889.66 |           405.79  |      2.46433 |
|          6 |                      128 |                     60 |          30.4157 |              5433.01 |           420.004 |      2.38093 |
|          7 |                      256 |                     60 |          35.5509 |              9201.66 |           442.802 |      2.25834 |

```python run_smoothquant.py --model_name llama-2-7b --task benchmark_long --target aie```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                      512 |                     60 |          48.5804 |              19484.8 |           483.495 |     2.06827  |
|          2 |                     1024 |                     60 |          82.2389 |              48861.6 |           555.252 |     1.80098  |
|          3 |                     1536 |                     60 |         123.745  |              83124.3 |           622.072 |     1.60753  |
|          4 |                     2000 |                     60 |         172.105  |             130272   |           687.318 |     1.45493  |
|          5 |                     3000 |                     60 |         299.094  |             248314   |           851.543 |     1.17434  |
|          6 |                     4000 |                     60 |         673.06   |             394404   |          1538.13  |     0.650138 |

### With Optimizations

```python run_smoothquant.py --model_name llama-2-7b --task  benchmark --target aie --flash_attention_plus ```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        4 |                     60 |          25.6353 |              1224.27 |           409.712 |      2.44074 |
|          2 |                        8 |                     60 |          25.8295 |              1355.54 |           410.83  |      2.4341  |
|          3 |                       16 |                     60 |          26.2348 |              2127.57 |           404.722 |      2.47083 |
|          4 |                       32 |                     60 |          26.8588 |              2667.95 |           405.645 |      2.46521 |
|          5 |                       64 |                     60 |          28.0519 |              3768.92 |           407.265 |      2.45541 |
|          6 |                      128 |                     60 |          30.2238 |              5100.96 |           420.7   |      2.37699 |
|          7 |                      256 |                     60 |          35.0951 |              8635.22 |           441.505 |      2.26498 |

```python run_smoothquant.py --model_name llama-2-7b --task  benchmark_long --target aie --flash_attention_plus ```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                      512 |                     60 |          46.7633 |              17285   |           489.153 |      2.04435 |
|          2 |                     1024 |                     60 |          74.7372 |              40334.7 |           564.987 |      1.76995 |
|          3 |                     1536 |                     60 |         109.594  |              70474.6 |           637.128 |      1.56954 |
|          4 |                     2000 |                     60 |         146.562  |             103076   |           703.209 |      1.42205 |
|          5 |                     3000 |                     60 |         250.699  |             197910   |           849.056 |      1.17778 |
|          6 |                     4000 |                     60 |         375.973  |             314523   |           981.638 |      1.01871 |

## HPT w8a8 (With MCDM)
### No optimizations

```python run_smoothquant.py --model_name llama-2-7b --task benchmark --target aie```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        4 |                     60 |          30.7765 |              620.101 |           507.759 |      1.96944 |
|          2 |                        8 |                     60 |          30.8081 |              656.439 |           507.865 |      1.96903 |
|          3 |                       16 |                     60 |          30.9625 |              712.704 |           509.579 |      1.96241 |
|          4 |                       32 |                     60 |          31.2896 |              984.035 |           510.431 |      1.95913 |
|          5 |                       64 |                     60 |          32.1142 |             1691.25  |           511.964 |      1.95326 |
|          6 |                      128 |                     60 |          34.1293 |             3289.47  |           519.434 |      1.92517 |
|          7 |                      256 |                     60 |          39.1359 |             7012.83  |           540.454 |      1.8503  |

```python run_smoothquant.py --model_name llama-2-7b --task benchmark_long --target aie```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                      512 |                     60 |          47.9243 |              13752.6 |           568.966 |      1.75757 |
|          2 |                     1024 |                     60 |          67.1222 |              30350.3 |           612.533 |      1.63257 |
|          3 |                     1536 |                     60 |          89.4715 |              50022.7 |           661.734 |      1.51118 |
|          4 |                     2000 |                     60 |         113.023  |              70202.7 |           705.129 |      1.41818 |
|          5 |                     3000 |                     60 |         171.057  |             122162   |           820.2   |      1.21922 |
|          6 |                     4000 |                     60 |         254.588  |             186051   |          1105.75  |      0.90436 |

### With Optimizations

```python run_smoothquant.py --model_name llama-2-7b --task  benchmark --target aie --flash_attention_plus ```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        4 |                     60 |          30.3247 |              600.22  |           500.372 |      1.99851 |
|          2 |                        8 |                     60 |          30.2637 |              641.315 |           498.672 |      2.00533 |
|          3 |                       16 |                     60 |          30.4586 |              706.964 |           500.735 |      1.99706 |
|          4 |                       32 |                     60 |          30.727  |              950.403 |           500.92  |      1.99632 |
|          5 |                       64 |                     60 |          31.7066 |             1539.42  |           507.169 |      1.97173 |
|          6 |                      128 |                     60 |          33.4336 |             2932.38  |           512.026 |      1.95303 |
|          7 |                      256 |                     60 |          37.9189 |             5871.68  |           536.295 |      1.86464 |

```python run_smoothquant.py --model_name llama-2-7b --task  benchmark_long --target aie --flash_attention_plus ```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                      512 |                     60 |          47.1375 |              13490.5 |           560.016 |      1.78566 |
|          2 |                     1024 |                     60 |          63.5764 |              26663.1 |           607.806 |      1.64526 |
|          3 |                     1536 |                     60 |          82.3353 |              42291.4 |           653.613 |      1.52996 |
|          4 |                     2000 |                     60 |         102.327  |              59349.7 |           696.242 |      1.43628 |
|          5 |                     3000 |                     60 |         151.644  |             102036   |           795.519 |      1.25704 |
|          6 |                     4000 |                     60 |         207.111  |             150090   |           904.082 |      1.10609 |
