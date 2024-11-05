## Performance on HPT and STX

### Save quantized checkpoints
```
python .\prepare_model.py --model_name <model_name> --output_model_dir <output directory> --export --optimize --quantize --opt_level {0,1,2}
```
⚠️ **Disclaimer:** The output model dir has fp32, fp32_optm and quant folders,copy the models.data.onnx file from the quant folder to the current folder(models/llm_onnx). If not done , we get wrong results for AIE flow:

### Run Qwen1.5-7b decode examples
```powershell
# Go to test directory
cd models\llm_onnx

#Run qwen1.5-7b inference with decode option
python .\infer.py  --target <cpu/aie> --model-dir <model output directory> --profile --task decode
```

### Run qwen1.5b-7b inference with benchmark option
```
# This runs benchmark for different sequence lengths
python .\infer.py --model_name <model_name> --target <cpu/aie> --model_dir <input model directory> --profile --task benchmark

# Run for specific sequence length (e.g. 2048)
python .\infer.py --model_name <model_name> --target <cpu/aie> --model_dir <input model directory> --profile --task benchmark --seqlen 2048
```
:pushpin: We recommend to use opt_level = 0 for this model for better performance, the numbers are here are for optimization level 0  

## Latency: STRIX (VAI-EP) (AMD Eng Sample: 100-000000994-37_Y, 12 Core(s))
### Group Size = 128
| Example# |    Prompt Length (tokens) |  New Tokens Generated |    Total Time (s) |    Prefill Phase (ms) |    Time/Token (ms) |   Tokens/Sec |
| -------- | ------------------------- | --------------------- | ----------------- | --------------------- | ------------------ | ------------ |
| 1        | 8                         | 11                    | 1.69532           | 213.546               | 144.72             | 6.90991      |
| 2        | 16                        | 7                     | 1.38939           | 513.483               | 142.508            | 7.01713      |
| 3        | 32                        | 11                    | 2.00233           | 543.385               | 142.55             | 7.01507      |
| 4        | 64                        | 11                    | 2.52569           | 1023.04               | 147.255            | 6.79094      |
| 5        | 128                       | 11                    | 3.95757           | 2232.54               | 167.925            | 5.95503      |
| 6        | 256                       | 11                    | 6.31914           | 4206.4                | 206.364            | 4.8458       |
| 7        | 512                       | 11                    | 11.8928           | 9062.29               | 273.546            | 3.65569      |
| 8        | 1024                      | 11                    | 24.0343           | 19940.2               | 390.244            | 2.5625       |
| 9        | 1536                      | 11                    | 36.8097           | 31498.6               | 498.136            | 2.00748      |
| 10       | 2048                      | 11                    | 49.5173           | 43772.8               | 550.269            | 1.81729      |
| 11       | 4096                      | 11                    | 115.567           | 104409                | 1077.94            | 0.927697     |

## Latency: Hawkpoint (VAI-EP) (AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 8 cores)
### Group Size = 128

| Example# |    Prompt Length (tokens) |  New Tokens Generated |    Total Time (s) |    Prefill Phase (ms) |    Time/Token (ms) |   Tokens/Sec |
| -------- | ------------------------- | --------------------- | ----------------- | --------------------- | ------------------ | ------------ |
| 1        | 8                         | 11                    | 1.96293           | 275.016               | 166.419            | 6.00891      |
| 2        | 16                        | 11                    | 2.4875            | 793.194               | 167.041            | 5.98655      |
| 3        | 32                        | 11                    | 2.51126           | 815.698               | 167.293            | 5.97754      |
| 4        | 64                        | 7                     | 2.65776           | 1606.33               | 172.614            | 5.79328      |
| 5        | 128                       | 11                    | 5.68463           | 3822.6                | 183.737            | 5.44256      |
| 6        | 256                       | 5                     | 7.9414            | 7097.98               | 207.183            | 4.82665      |
| 7        | 512                       | 11                    | 17.4393           | 14820.8               | 257.637            | 3.88143      |
| 8        | 1024                      | 11                    | 38.2995           | 34828.7               | 339.057            | 2.94935      |
| 9        | 1536                      | 11                    | 57.6541           | 52970.3               | 453.435            | 2.20539      |
| 10       | 2048                      | 11                    | 67.3137           | 61376.1               | 583.61             | 1.71347      |
| 11       | 4096                      | 11                    | 153.586           | 143536                | 985.403            | 1.01481      |