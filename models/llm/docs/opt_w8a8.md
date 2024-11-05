# OPT
```
python run_smoothquant.py --model_name facebook/opt-1.3b --task quantize
python run_smoothquant.py --model_name facebook/opt-1.3b  --target aie --task decode
```

# Decode (SmoothQuant with w8a8 - PHX)
```
python run_smoothquant.py --model_name facebook/opt-1.3b --task benchmark --target aie --precision w8a8
...
****************************************
prompt: What is the meaning of life?
response: What is the meaning of life?

The meaning of life is the question that is asked by many people. The meaning of life is the
****************************************
prompt: What does Xilinx do?
response: What does Xilinx do?

Xilinx is a global leader in the design and manufacture of integrated circuits, software, and services
****************************************
prompt: What is recursion?
response: What is recursion?

Recursion is a mathematical concept that describes the relationship between two or more mathematical operations. It is a mathematical concept
****************************************
```
|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        8 |                     30 |          3.51495 |              160.895 |           114.625 |      8.72413 |
|          2 |                        9 |                     30 |          3.53688 |              152.433 |           115.698 |      8.64319 |
|          3 |                        8 |                     30 |          3.52958 |              144.461 |           115.664 |      8.64571 |
|          4 |                        8 |                     30 |          3.5336  |              148.211 |           115.699 |      8.64311 |
|          5 |                        6 |                     30 |          3.52286 |              142.687 |           115.521 |      8.65646 |
|          6 |                        6 |                     30 |          3.49047 |              141.76  |           114.44  |      8.73821 |
|          7 |                        8 |                     30 |          3.51855 |              147.136 |           115.218 |      8.67919 |
|          8 |                        7 |                     30 |          3.50885 |              145.742 |           114.928 |      8.70106 |
|          9 |                        7 |                     30 |          3.51532 |              145.837 |           115.129 |      8.68591 |
|         10 |                        7 |                     30 |          3.47598 |              141.124 |           113.923 |      8.77783 |


# Profiling latency (PHX)

## No Optimizations

```python run_smoothquant.py --model_name facebook/opt-1.3b  --target aie --task benchmark```
|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        4 |                     30 |          3.71794 |              162.487 |           120.341 |      8.30975 |
|          2 |                        8 |                     30 |          3.71355 |              150.72  |           120.607 |      8.2914  |
|          3 |                       16 |                     30 |          3.74337 |              173.508 |           120.708 |      8.28447 |
|          4 |                       32 |                     30 |          3.84878 |              246.191 |           121.693 |      8.21737 |
|          5 |                       64 |                     30 |          4.05098 |              369.286 |           124.419 |      8.03733 |
|          6 |                      128 |                     30 |          4.61639 |              872.101 |           126.365 |      7.91361 |
|          7 |                      256 |                     30 |          5.6491  |             1772.18  |           130.653 |      7.65388 |

```python run_smoothquant.py --model_name facebook/opt-1.3b  --target aie --task benchmark_long```
|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                      512 |                     30 |          8.18918 |              3992.59 |           140.498 |      7.11752 |
|          2 |                     1024 |                     30 |         14.6453  |              9768.46 |           161.599 |      6.18816 |
|          3 |                     1536 |                     30 |         23.7856  |             18311.3  |           180.597 |      5.5372  |
|          4 |                     2000 |                     30 |         33.3854  |             27336.9  |           198.079 |      5.0485  |

## Attention Optimizations

```python run_smoothquant.py --model_name facebook/opt-1.3b  --target aie --task benchmark --flash_attention_plus```
|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        4 |                     30 |          3.3343  |              130.086 |           108.15  |      9.24642 |
|          2 |                        8 |                     30 |          3.34703 |              136.712 |           108.434 |      9.22223 |
|          3 |                       16 |                     30 |          3.38615 |              163.928 |           108.803 |      9.19089 |
|          4 |                       32 |                     30 |          3.46436 |              217.003 |           109.463 |      9.1355  |
|          5 |                       64 |                     30 |          3.55989 |              311.427 |           109.511 |      9.1315  |
|          6 |                      128 |                     30 |          4.02131 |              682.973 |           112.599 |      8.88107 |
|          7 |                      256 |                     30 |          4.92365 |             1438.48  |           117.72  |      8.49471 |

```python run_smoothquant.py --model_name facebook/opt-1.3b  --target aie --task benchmark_long --flash_attention_plus```
|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                      512 |                     30 |          6.64346 |              2790.86 |           128.449 |      7.78521 |
|          2 |                     1024 |                     30 |         10.7061  |              6175.17 |           149.419 |      6.69259 |
|          3 |                     1536 |                     30 |         15.751   |             10603.7  |           169.103 |      5.91355 |
|          4 |                     2000 |                     30 |         21.04    |             15360.7  |           184.755 |      5.41256 |


# Profiling w8a8 (STX A0)

## No Optimizations

```python run_smoothquant.py --model_name facebook/opt-1.3b  --target aie --task benchmark```

```python run_smoothquant.py --model_name facebook/opt-1.3b  --target aie --task benchmark_long```

## Attention Optimizations

```python run_smoothquant.py --model_name facebook/opt-1.3b  --target aie --task benchmark --flash_attention_plus```

```python run_smoothquant.py --model_name facebook/opt-1.3b  --target aie --task benchmark_long --flash_attention_plus```


# Profiling w8a8 (STX B0)

## No Optimizations

```python run_smoothquant.py --model_name facebook/opt-1.3b  --target aie --task benchmark```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        4 |                     30 |          4.26011 |              648.688 |          115.837  |      8.6328  |
|          2 |                        8 |                     30 |          4.35275 |              671.728 |          121.702  |      8.21681 |
|          3 |                       16 |                     30 |          4.4642  |              714.91  |          119.933  |      8.33801 |
|          4 |                       32 |                     30 |          4.65149 |              999.141 |          118.369  |      8.44816 |
|          5 |                       64 |                     30 |          4.85948 |             1273.36  |          118.822  |      8.41592 |
|          6 |                      128 |                     30 |          5.47237 |             2442.71  |           96.3907 |     10.3744  |
|          7 |                      256 |                     30 |          6.55512 |             4037.1   |           82.8856 |     12.0648  |

```python run_smoothquant.py --model_name facebook/opt-1.3b  --target aie --task benchmark_long```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                      512 |                     30 |          10.9347 |              7977.01 |           96.6095 |     10.3509  |
|          2 |                     1024 |                     30 |          24.8898 |             21207.1  |          119.224  |      8.38754 |
|          3 |                     1536 |                     30 |          45.8796 |             41375.9  |          144.97   |      6.89797 |
|          4 |                     2000 |                     30 |        7396.53   |             64407.6  |          160.379  |      6.23521 |

## Attention Optimizations

```python run_smoothquant.py --model_name facebook/opt-1.3b  --target aie --task benchmark --flash_attention_plus```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        4 |                     30 |          4.01788 |              623.303 |          107.851  |      9.27205 |
|          2 |                        8 |                     30 |          4.12086 |              647.742 |          112.498  |      8.88907 |
|          3 |                       16 |                     30 |          4.25785 |              702.269 |          117.898  |      8.48188 |
|          4 |                       32 |                     30 |          4.39568 |              923.897 |          113.167  |      8.83653 |
|          5 |                       64 |                     30 |          4.61949 |             1105.01  |          116.067  |      8.61569 |
|          6 |                      128 |                     30 |          5.20328 |             1702.66  |          111.267  |      8.98736 |
|          7 |                      256 |                     30 |          6.16875 |             3381.85  |           92.4118 |     10.8211  |

```python run_smoothquant.py --model_name facebook/opt-1.3b  --target aie --task benchmark_long --flash_attention_plus```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                      512 |                     30 |          9.73175 |              6863.28 |           93.1316 |     10.7375  |
|          2 |                     1024 |                     30 |         18.4525  |             14815.2  |          116.841  |      8.55864 |
|          3 |                     1536 |                     30 |         31.0763  |             26570.8  |          144.452  |      6.92271 |
|          4 |                     2000 |                     30 |         46.0864  |             40730.7  |          170.405  |      5.86837 |

# Profiling w8a8 (HPT) (With MCDM)

## No Optimizations

```python run_smoothquant.py --model_name facebook/opt-1.3b  --target aie --task benchmark```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        4 |                     30 |          2.58661 |              113.399 |           82.7515 |      12.0844 |
|          2 |                        8 |                     30 |          2.58457 |              112.68  |           82.8109 |      12.0757 |
|          3 |                       16 |                     30 |          2.64174 |              132.372 |           83.9475 |      11.9122 |
|          4 |                       32 |                     30 |          2.75332 |              215.289 |           84.8646 |      11.7835 |
|          5 |                       64 |                     30 |          2.90074 |              344.977 |           85.4139 |      11.7077 |
|          6 |                      128 |                     30 |          3.56449 |              921.826 |           88.1277 |      11.3472 |
|          7 |                      256 |                     30 |          4.56314 |             1770.66  |           92.8293 |      10.7725 |

```python run_smoothquant.py --model_name facebook/opt-1.3b  --target aie --task benchmark_long```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                      512 |                     30 |           7.3009 |              4113.87 |           104.7   |      9.55111 |
|          2 |                     1024 |                     30 |          13.8188 |             10022    |           123.096 |      8.12372 |
|          3 |                     1536 |                     30 |          21.3028 |             16860.2  |           142.669 |      7.00921 |
|          4 |                     2000 |                     30 |          32.5829 |             27621.9  |           157.483 |      6.34991 |

## Attention Optimizations

```python run_smoothquant.py --model_name facebook/opt-1.3b  --target aie --task benchmark --flash_attention_plus```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        4 |                     30 |          2.28569 |             100.819  |           72.8847 |      13.7203 |
|          2 |                        8 |                     30 |          2.30787 |              96.5871 |           73.7336 |      13.5623 |
|          3 |                       16 |                     30 |          2.33135 |             122.09   |           73.6854 |      13.5712 |
|          4 |                       32 |                     30 |          2.43625 |             195.786  |           74.5007 |      13.4227 |
|          5 |                       64 |                     30 |          2.56297 |             315.466  |           74.8409 |      13.3617 |
|          6 |                      128 |                     30 |          3.10736 |             752.826  |           78.4096 |      12.7535 |
|          7 |                      256 |                     30 |          4.04601 |            1514.96   |           84.5252 |      11.8308 |

```python run_smoothquant.py --model_name facebook/opt-1.3b  --target aie --task benchmark_long --flash_attention_plus```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                      512 |                     30 |           6.1883 |              3190.61 |           97.7057 |     10.2348  |
|          2 |                     1024 |                     30 |          11.1274 |              7536.84 |          115.311  |      8.67221 |
|          3 |                     1536 |                     30 |          16.3835 |             12097.9  |          137.239  |      7.28656 |
|          4 |                     2000 |                     30 |          22.2444 |             17456.7  |          150.613  |      6.63955 |
