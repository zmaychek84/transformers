# codellama/CodeLlama-7b-hf

Quantize the weights and save checkpoint

```python run_awq.py --model_name codellama/CodeLlama-7b-hf --task quantize```

```
LlamaModelEval(
  (model): LlamaModel(
    (embed_tokens): Embedding(32016, 4096)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaFlashAttentionPlus(
          (rotary_emb): LlamaRotaryEmbedding()
          (o_proj): ryzenAI.QLinearPerGrp(in_features:4096, out_features:4096, bias:None, device:aie, w_bit:4 group_size:128  )
          (qkv_proj): ryzenAI.QLinearPerGrp(in_features:4096, out_features:12288, bias:None, device:aie, w_bit:4 group_size:128  )
        )
        (mlp): LlamaFastMLP(
          (gate_proj): ryzenAI.QLinearPerGrp(in_features:4096, out_features:11008, bias:None, device:aie, w_bit:4 group_size:128  )
          (up_proj): ryzenAI.QLinearPerGrp(in_features:4096, out_features:11008, bias:None, device:aie, w_bit:4 group_size:128  )
          (down_proj): ryzenAI.QLinearPerGrp(in_features:11008, out_features:4096, bias:None, device:aie, w_bit:4 group_size:128  )
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(in_features=4096, out_features=32016, bias=False)
)
model.mode_name: CodeLlama-7b-hf
****************************************
```

## Latency on HPT (With MCDM)
```python run_awq.py --model_name codellama/CodeLlama-7b-hf --task decode```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                       69 |                    200 |          42.6868 |              2002.88 |           202.075 |      4.94866 |
|          2 |                       95 |                    200 |          43.7045 |              2653.28 |           204.101 |      4.89954 |
|          3 |                       89 |                    200 |          43.2758 |              2454.39 |           202.865 |      4.92939 |
|          4 |                      102 |                    200 |          43.5563 |              2732.94 |           203.063 |      4.92457 |
|          5 |                       67 |                    200 |          42.6462 |              1888.64 |           202.785 |      4.93134 |
|          6 |                       89 |                    200 |          43.6269 |              2441.9  |           204.819 |      4.88235 |
|          7 |                      101 |                    200 |          43.7724 |              2703.93 |           204.302 |      4.89471 |
|          8 |                      160 |                    200 |          46.4783 |              4307.49 |           209.749 |      4.76761 |
|          9 |                       74 |                    200 |          43.0447 |              2381.06 |           202.289 |      4.94343 |
|         10 |                      102 |                    200 |          43.512  |              2696.96 |           202.857 |      4.92958 |
|         11 |                       96 |                    200 |          43.315  |              2480.45 |           202.947 |      4.9274  |
|         12 |                       86 |                    200 |          43.6589 |              2447.1  |           204.953 |      4.87918 |
|         13 |                       92 |                    200 |          43.377  |              2449.19 |           203.549 |      4.91282 |

```python run_awq.py --model_name codellama/CodeLlama-7b-hf --task decode --flash_attention_plus ```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                       69 |                    200 |          38.901  |              1936.07 |           183.506 |      5.44941 |
|          2 |                       95 |                    200 |          39.544  |              2458.53 |           184.14  |      5.43066 |
|          3 |                       89 |                    200 |          39.5396 |              2372.09 |           184.364 |      5.42406 |
|          4 |                      102 |                    200 |          39.833  |              2622.42 |           184.867 |      5.40929 |
|          5 |                       67 |                    200 |          38.3788 |              1827.48 |           181.591 |      5.50688 |
|          6 |                       89 |                    200 |          39.2044 |              2357.73 |           183.007 |      5.46428 |
|          7 |                      101 |                    200 |          39.5294 |              2612.16 |           183.375 |      5.4533  |
|          8 |                      160 |                    200 |          42.1135 |              4147.18 |           188.575 |      5.30292 |
|          9 |                       74 |                    200 |          39.0003 |              2303.35 |           182.266 |      5.4865  |
|         10 |                      102 |                    200 |          39.6799 |              2624.51 |           184     |      5.43478 |
|         11 |                       96 |                    200 |          39.4186 |              2390.37 |           183.883 |      5.43824 |
|         12 |                       86 |                    200 |          39.2942 |              2336.86 |           183.584 |      5.4471  |
|         13 |                       92 |                    200 |          39.3038 |              2383.07 |           183.392 |      5.45281 |

```python run_awq.py --model_name codellama/CodeLlama-7b-hf --task decode --flash_attention_plus --algorithm awqplus```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                       69 |                    200 |          38.0325 |              1964.29 |           179.145 |      5.58206 |
|          2 |                       95 |                    200 |          38.7794 |              2488.36 |           180.311 |      5.54596 |
|          3 |                       89 |                    200 |          38.781  |              2382.99 |           180.72  |      5.53341 |
|          4 |                      102 |                    200 |          38.986  |              2641.44 |           180.613 |      5.53671 |
|          5 |                       67 |                    200 |          37.8446 |              1830.04 |           179.046 |      5.58517 |
|          6 |                       89 |                    200 |          38.627  |              2376.26 |           180.154 |      5.55081 |
|          7 |                      101 |                    200 |          39.0439 |              2638.5  |           180.945 |      5.52654 |
|          8 |                      160 |                    200 |          41.5318 |              4169.01 |           185.693 |      5.38522 |
|          9 |                       74 |                    200 |          38.3595 |              2323.15 |           179.103 |      5.58337 |
|         10 |                      102 |                    200 |          39.1527 |              2643.63 |           181.409 |      5.51239 |
|         11 |                       96 |                    200 |          38.8935 |              2396.66 |           181.371 |      5.51356 |
|         12 |                       86 |                    200 |          38.7774 |              2370.77 |           180.963 |      5.526   |
|         13 |                       92 |                    200 |          38.7033 |              2385.24 |           180.496 |      5.54028 |

## Latency on PHX

```python run_awq.py --model_name codellama/CodeLlama-7b-hf --task decode```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                       69 |                    200 |          61.562  |              2930.43 |           292.574 |      3.41794 |
|          2 |                       95 |                    200 |          63.0102 |              3735.56 |           296.009 |      3.37828 |
|          3 |                       89 |                    200 |          62.8093 |              3645.14 |           295.405 |      3.38519 |
|          4 |                      102 |                    200 |          63.3617 |              4057.55 |           296.181 |      3.37631 |
|          5 |                       67 |                    200 |          61.5767 |              2810.67 |           293.547 |      3.4066  |
|          6 |                       89 |                    200 |          62.4175 |              3627.46 |           293.636 |      3.40557 |
|          7 |                      101 |                    200 |          62.8861 |              4006.01 |           294.123 |      3.39993 |
|          8 |                      160 |                    200 |          65.977  |              6215.86 |           298.502 |      3.35007 |
|          9 |                       74 |                    200 |          62.1225 |              3554.17 |           292.556 |      3.41815 |
|         10 |                      102 |                    200 |          63.0264 |              4022.69 |           294.645 |      3.39391 |
|         11 |                       96 |                    200 |          62.8263 |              3670.34 |           295.416 |      3.38506 |
|         12 |                       86 |                    200 |          62.4779 |              3617.23 |           294.007 |      3.40128 |
|         13 |                       92 |                    200 |          62.7563 |              3646.78 |           295.203 |      3.3875  |

```python run_awq.py --model_name codellama/CodeLlama-7b-hf --task decode --flash_attention_plus ```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                       69 |                    200 |          56.2255 |              2804.91 |           266.477 |      3.75267 |
|          2 |                       95 |                    200 |          57.0499 |              3573.61 |           266.851 |      3.74741 |
|          3 |                       89 |                    200 |          56.9601 |              3487.5  |           266.632 |      3.75049 |
|          4 |                      102 |                    200 |          57.5509 |              3861.21 |           267.998 |      3.73137 |
|          5 |                       67 |                    200 |          55.6962 |              2680.23 |           264.669 |      3.7783  |
|          6 |                       89 |                    200 |          56.8062 |              3483.37 |           266.12  |      3.75771 |
|          7 |                      101 |                    200 |          57.2929 |              3850.32 |           266.766 |      3.74861 |
|          8 |                      160 |                    200 |          60.2872 |              5993.88 |           270.958 |      3.6906  |
|          9 |                       74 |                    200 |          56.6469 |              3437.28 |           265.617 |      3.76482 |
|         10 |                      102 |                    200 |          57.2775 |              3867.85 |           266.539 |      3.75179 |
|         11 |                       96 |                    200 |          56.8088 |              3515.84 |           265.94  |      3.76025 |
|         12 |                       86 |                    200 |          56.53   |              3471.98 |           264.858 |      3.77561 |
|         13 |                       92 |                    200 |          56.6432 |              3480.61 |           265.347 |      3.76865 |

```python run_awq.py --model_name codellama/CodeLlama-7b-hf --task decode --flash_attention_plus --algorithm awqplus```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                       69 |                    200 |          55.5225 |              2849.32 |           262.675 |      3.80699 |
|          2 |                       95 |                    200 |          56.3779 |              3613.08 |           263.221 |      3.79908 |
|          3 |                       89 |                    200 |          56.3854 |              3516.21 |           263.71  |      3.79205 |
|          4 |                      102 |                    200 |          57.007  |              3892.13 |           265.018 |      3.77332 |
|          5 |                       67 |                    200 |          55.4252 |              2704.84 |           263.069 |      3.80128 |
|          6 |                       89 |                    200 |          56.2823 |              3521.38 |           263.24  |      3.79881 |
|          7 |                      101 |                    200 |          56.6838 |              3896.77 |           263.391 |      3.79664 |
|          8 |                      160 |                    200 |          59.6893 |              6037.81 |           267.686 |      3.73572 |
|          9 |                       74 |                    200 |          56.1871 |              3469.48 |           263.036 |      3.80177 |
|         10 |                      102 |                    200 |          57.0485 |              3908.57 |           265.065 |      3.77267 |
|         11 |                       96 |                    200 |          56.6645 |              3559.27 |           264.955 |      3.77422 |
|         12 |                       86 |                    200 |          56.3727 |              3509.83 |           263.776 |      3.79109 |
|         13 |                       92 |                    200 |          56.7659 |              3549.04 |           265.543 |      3.76587 |

## Latency on STX B0 (with MCDM)

```python run_awq.py --model_name codellama/CodeLlama-7b-hf --task decode```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                       69 |                    200 |          37.7643 |              2159.01 |           176.314 |      5.67169 |
|          2 |                       95 |                    200 |          38.975  |              2726.1  |           179.772 |      5.56261 |
|          3 |                       89 |                    200 |          38.6802 |              2566.67 |           178.686 |      5.59642 |
|          4 |                      102 |                    200 |          39.3649 |              2909.91 |           180.881 |      5.52851 |
|          5 |                       67 |                    200 |          37.5009 |              1975.92 |           176.247 |      5.67387 |
|          6 |                       89 |                    200 |          38.7146 |              2609.42 |           179.102 |      5.58342 |
|          7 |                      101 |                    200 |          39.295  |              2930.18 |           180.388 |      5.54361 |
|          8 |                      160 |                    200 |          42.623  |              4805.88 |           187.635 |      5.32949 |
|          9 |                       74 |                    200 |          37.8765 |              2240.9  |           176.788 |      5.65648 |
|         10 |                      102 |                    200 |          39.3452 |              2901.45 |           180.758 |      5.53225 |
|         11 |                       96 |                    200 |          73.8592 |              2705.8  |           180.01  |      5.55523 |
|         12 |                       86 |                    200 |          38.5879 |              2563.38 |           178.695 |      5.59611 |
|         13 |                       92 |                    200 |          38.7725 |              2650.83 |           179.036 |      5.58548 |

```python run_awq.py --model_name codellama/CodeLlama-7b-hf --task decode --flash_attention_plus ```


|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                       69 |                    200 |          34.5511 |              1996.28 |           161.153 |      6.20527 |
|          2 |                       95 |                    200 |          35.6911 |              2659.83 |           163.605 |      6.11229 |
|          3 |                       89 |                    200 |          35.5913 |              2522.2  |           163.491 |      6.11655 |
|          4 |                      102 |                    200 |          36.0658 |              2786.43 |           164.88  |      6.06503 |
|          5 |                       67 |                    200 |          34.2122 |              1879.65 |           160.196 |      6.24237 |
|          6 |                       89 |                    200 |          35.317  |              2496.34 |           162.602 |      6.15    |
|          7 |                      101 |                    200 |          35.8277 |              2667.06 |           164.339 |      6.08497 |
|          8 |                      160 |                    200 |          39.2003 |              4563.13 |           171.679 |      5.82482 |
|          9 |                       74 |                    200 |          34.8073 |              2293.34 |           161.061 |      6.20881 |
|         10 |                      102 |                    200 |          36.049  |              2840.41 |           164.49  |      6.07941 |
|         11 |                       96 |                    200 |          35.6551 |              2606.51 |           163.724 |      6.10784 |
|         12 |                       86 |                    200 |          35.265  |              2473.85 |           162.462 |      6.15527 |
|         13 |                       92 |                    200 |          35.4414 |              2555.37 |           162.902 |      6.13865 |

```python run_awq.py --model_name codellama/CodeLlama-7b-hf --task decode --flash_attention_plus --algorithm awqplus```

|   Example# |   Prompt Length (tokens) |   New Tokens Generated |   Total Time (s) |   Prefill Phase (ms) |   Time/Token (ms) |   Tokens/Sec |
|------------|--------------------------|------------------------|------------------|----------------------|-------------------|--------------|
|          1 |                       69 |                    200 |          31.1764 |              1982.42 |           144.428 |      6.92384 |
|          2 |                       95 |                    200 |          32.5233 |              2586.55 |           148.264 |      6.74471 |
|          3 |                       89 |                    200 |          32.2572 |              2433.33 |           147.664 |      6.77211 |
|          4 |                      102 |                    200 |          32.8656 |              2732.36 |           149.3   |      6.69792 |
|          5 |                       67 |                    200 |          31.0258 |              1855.02 |           144.455 |      6.92258 |
|          6 |                       89 |                    200 |          32.1958 |              2443.13 |           147.368 |      6.78575 |
|          7 |                      101 |                    200 |          32.841  |              2751.47 |           149.064 |      6.70851 |
|          8 |                      160 |                    200 |          36.0963 |              4434.46 |           156.869 |      6.37475 |
|          9 |                       74 |                    200 |          31.6182 |              2245.41 |           145.434 |      6.87599 |
|         10 |                      102 |                    200 |          32.9011 |              2767.13 |           149.125 |      6.70577 |
|         11 |                       96 |                    200 |          32.487  |              2514.49 |           148.433 |      6.73705 |
|         12 |                       86 |                    200 |          32.0884 |              2392.15 |           147.094 |      6.7984  |
|         13 |                       92 |                    200 |          32.3016 |              2469.78 |           147.793 |      6.76621 |

## Below is generated code with AWQ on AIE

```
model.mode_name: CodeLlama-7b-hf
****************************************
prompt: def print_hello_world():
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
response: def print_hello_world():
    print("Hello World!")


def print_hello_world_with_name(name):
    print("Hello " + name + "!")


def print_hello_world_with_name_and_age(name, age):
    print("Hello " + name + ", you are " + str(age) + " years old!")


def print_hello_world_with_name_and_age_and_city(name, age, city):
    print("Hello " + name + ", you are " + str(age) + " years old and you live in " + city + "!")


def print_hello_world_with_name_and_age_and_city_and_country(name, age, city, country):
    print("Hello " + name + ", you are " + str(age) + " years old, you live in " + city + " and you are from " + country + "!")


def print_hello_world_with_name_and_age_and_city_and_country_and_job(name, age, city, country, job):
    print("Hello " + name + ", you are " + str(age) + " years old, you live in " + city + ", you are from " + country + " and you are a " + job + "!")


def print_hello_world_with_name_and_age_and_city_and_country_and_job_and_pet(name, age, city, country, job, pet):
    print("Hello " + name + ", you are " + str(age) + " years old, you live in " + city + ", you are from " + country + ", you are a " + job
****************************************
prompt: def fibonacci_recursive(n):
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
response: def fibonacci_recursive(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)


def fibonacci_iterative(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        a = 0
        b = 1
        for i in range(n-1):
            a, b = b, a+b
        return a


def fibonacci_memoized(n, memo={}):
    if n in memo:
        return memo[n]
    elif n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        memo[n] = fibonacci_memoized(n-1, memo) + fibonacci_memoized(n-2, memo)
        return memo[n]


def fibonacci_tabulated(n):
    fib = [0, 1]
    for i in range(2, n+1):
        fib.append(fib[i-1] + fib[i-2])
    return fib[n]


def fibonacci_tabulated_memoized(n, memo={}):
    if n in memo:
        return memo[n]
    elif n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        memo[n] = fibonacci_tabulated_memoized(n
****************************************
prompt: import socket

def ping_exponential_backoff(host: str):
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
response: import socket

def ping_exponential_backoff(host: str):
    """
    Ping a host with exponential backoff.
    :param host: The host to ping.
    :return: True if the host is reachable, False otherwise.
    """
    # Setup the socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1)

    # Try to connect to the host
    try:
        s.connect((host, 80))
        s.shutdown(socket.SHUT_RDWR)
        return True
    except socket.error:
        return False


def ping_exponential_backoff_with_timeout(host: str, timeout: float):
    """
    Ping a host with exponential backoff and a timeout.
    :param host: The host to ping.
    :param timeout: The timeout in seconds.
    :return: True if the host is reachable, False otherwise.
    """
    # Setup the socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)

    # Try to connect to the host
    try:
        s.connect((host, 80))
        s.shutdown(socket.SHUT_RDWR)
        return True
    except socket.error:
        return False


def ping_exponential_backoff_with_timeout_and_retries(host: str, timeout: float, retries: int):
    """
    Ping a host with exponential backoff and a timeout and retries.
    :param host: The host to ping.
    :param timeout: The timeout in seconds.
 ```
