# Assisted generation - Llama2

# STX B0 with MCDM - 7/13/2024

## Target: AIE/w4abf16 Assistant: CPU/bf16

* top_p=0.95, tempertature=0.1, do_sample=True (nucleus sampling), based on pass@1 score configuration noted in original [Llamav2 publication](https://arxiv.org/pdf/2307.09288.pdf).

* ```assistant_model.generation_config.num_assistant_tokens = 6```  # K number

* ```assistant_model.generation_config.num_assistant_tokens_schedule = ("constant")```

## With nucleas sampling and no assisted generation

```python assisted_generation.py --model_name llama-2-7b --task decode```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        8 |                     30 |          4.33197 |              389.549 |           132.503 |      7.54701 |
|          2 |                       10 |                     30 |          4.2387  |              359.133 |           130.903 |      7.63927 |
|          3 |                        8 |                     30 |          4.89735 |              371.092 |           153.015 |      6.53529 |
|          4 |                        8 |                     30 |          5.2458  |              440.501 |           162.564 |      6.15141 |
|          5 |                        6 |                     30 |          5.11762 |              423.351 |           158.864 |      6.29468 |
|          6 |                        5 |                     30 |          5.06801 |              405.516 |           157.748 |      6.33921 |
|          7 |                        9 |                     30 |          5.05139 |              417.083 |           156.7   |      6.3816  |
|          8 |                        8 |                     30 |          4.98728 |              420.598 |           154.375 |      6.47774 |
|          9 |                        9 |                     30 |          4.96415 |              401.658 |           154.157 |      6.4869  |
|         10 |                        7 |                     30 |          4.98413 |              411.113 |           154.646 |      6.46639 |

## Assisted generation with draft: JackFram/llama-160m

Default set to use this draft model

```python assisted_generation.py --model_name llama-2-7b --task decode --assisted_generation```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        8 |                     30 |          4.92681 |              409.691 |          130.377  |      7.67007 |
|          2 |                       10 |                     30 |          3.63294 |              383.532 |           90.4192 |     11.0596  |
|          3 |                        8 |                     30 |          4.07666 |              374.224 |          103.049  |      9.70411 |
|          4 |                        8 |                     30 |          6.82334 |              396.037 |          189.359  |      5.28096 |
|          5 |                        6 |                     30 |          3.93237 |              411.499 |           97.8539 |     10.2193  |
|          6 |                        5 |                     30 |          5.08169 |              411.543 |          136.481  |      7.327   |
|          7 |                        9 |                     30 |          5.72248 |              428.994 |          160.499  |      6.23056 |
|          8 |                        8 |                     30 |          6.50049 |              422.007 |          177.674  |      5.6283  |
|          9 |                        9 |                     30 |          6.81678 |              431.38  |          192.2    |      5.20293 |
|         10 |                        7 |                     30 |          4.22256 |              412.779 |          109.23   |      9.15501 |


```python assisted_generation.py --model_name llama-2-7b --task benchmark --assisted_generation```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                      256 |                     60 |          16.217  |              1660.34 |          201.209  |      4.96996 |
|          2 |                      512 |                     60 |          23.0221 |              4894.9  |          238.038  |      4.20102 |
|          3 |                     1024 |                     60 |          20.4996 |              9443.6  |           83.6048 |     11.961   |
|          4 |                     1968 |                     60 |          46.409  |             11961.5  |          424.398  |      2.35628 |


## Assisted generation with draft: amd-pretrained-135m-2k-no-book3

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                        8 |                     30 |         11.0142  |              478.239 |           322.112 |      3.10451 |
|          2 |                       10 |                     30 |          5.94047 |              446.387 |           153.619 |      6.5096  |
|          3 |                        8 |                     30 |          7.77908 |              435.511 |           210.712 |      4.74582 |
|          4 |                        8 |                     30 |          7.71374 |              431.012 |           217.775 |      4.5919  |
|          5 |                        6 |                     30 |          7.69269 |              418.339 |           210.149 |      4.75852 |
|          6 |                        5 |                     30 |          7.76392 |              402.636 |           218.691 |      4.57266 |
|          7 |                        9 |                     30 |          7.59735 |              436.827 |           208.158 |      4.80403 |
|          8 |                        8 |                     30 |          7.97549 |              435.514 |           224.441 |      4.45551 |
|          9 |                        9 |                     30 |          5.8401  |              442.431 |           145.555 |      6.87023 |
|         10 |                        7 |                     30 |          8.21691 |              424.887 |           225.472 |      4.43514 |
