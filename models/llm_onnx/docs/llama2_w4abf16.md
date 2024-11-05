## Performance on HPT and STX

### Save quantized checkpoints
```
python .\prepare_model.py --model_name <model_name> --output_model_dir <output directory> --export --optimize --quantize --opt_level {0,1,2}
```
⚠️ **Disclaimer:** The output model dir has fp32, fp32_optm and quant folders,copy the models.data.onnx file from the quant folder to the current folder(models/llm_onnx). If not done , we get wrong results for AIE flow:

### Run LLAMA2 decode examples
```powershell
# Go to test directory
cd models\llm_onnx

#Run llama2 inference with decode option
python .\infer.py  --target <cpu/aie> --model-dir <model output directory> --profile --task decode
```

### Run llama2 inference with benchmark option
```
# This runs benchmark for different sequence lengths
python .\infer.py --model_name <model_name> --target <cpu/aie> --model_dir <input model directory> --profile --task benchmark

# Run for specific sequence length (e.g. 2048)
python .\infer.py --model_name <model_name> --target <cpu/aie> --model_dir <input model directory> --profile --task benchmark --seqlen 2048
```
:pushpin: We recommend to use opt_level = 1 for this model for better accuracy, the numbers are here are for optimization level 1  

## Latency: STRIX (VAI-EP) (AMD Eng Sample: 100-000000994-37_Y, 12 Core(s))
### Group Size = 128

| Example# | Prompt Length (tokens) | New Tokens Generated | Total Time (s) | Prefill Phase (ms) | Time/Token (ms) | Tokens/Sec |
| -------- | ---------------------- | -------------------- | -------------- | ------------------ | --------------- | ---------- |
| 1        | 8                      | 11                   | 1.59237        | 202.525            | 136.919         | 7.30361    |
| 2        | 16                     | 11                   | 1.81741        | 469.247            | 133.081         | 7.51424    |
| 3        | 32                     | 11                   | 1.84786        | 493.72             | 133.761         | 7.47603    |
| 4        | 64                     | 11                   | 2.31402        | 922.425            | 137.664         | 7.26405    |
| 5        | 128                    | 11                   | 3.55367        | 2072.13            | 146.709         | 6.81621    |
| 6        | 256                    | 11                   | 5.55136        | 3819.52            | 171.651         | 5.82577    |
| 7        | 512                    | 11                   | 10.7295        | 8306.92            | 239.556         | 4.1744     |
| 8        | 1024                   | 11                   | 21.5019        | 17845.7            | 358.629         | 2.7884     |
| 9        | 1536                   | 11                   | 34.4956        | 29870.4            | 451.779         | 2.21347    |
| 10       | 2048                   | 11                   | 46.3595        | 40662.5            | 556.327         | 1.7975     |

## Latency: Hawkpoint (VAI-EP) (AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 8 cores)
### Group Size = 128


| Example# | Prompt Length (tokens) | New Tokens Generated | Total Time (s) | Prefill Phase (ms) | Time/Token (ms) | Tokens/Sec |
| -------- | ---------------------- | -------------------- | -------------- | ------------------ | --------------- | ---------- |
| 1        | 8                      | 11                   | 2.87135        | 390.372            | 246.765         | 4.05243    |
| 2        | 16                     | 11                   | 3.56951        | 1119.37            | 243.713         | 4.10319    |
| 3        | 32                     | 11                   | 3.61356        | 1138.21            | 246.242         | 4.06104    |
| 4        | 64                     | 11                   | 4.76517        | 2247.44            | 250.25          | 3.996      |
| 5        | 128                    | 11                   | 7.54611        | 4867.21            | 266.152         | 3.75724    |
| 6        | 256                    | 11                   | 13.1554        | 10216.3            | 292.726         | 3.41616    |
| 7        | 512                    | 11                   | 25.2159        | 21671.4            | 352.752         | 2.83485    |
| 8        | 1024                   | 11                   | 47.0729        | 42502.8            | 453.674         | 2.20423    |
| 9        | 1536                   | 11                   | 69.0546        | 63428.8            | 558.257         | 1.79129    |
| 10       | 2048                   | 11                   | 92.1606        | 85022.6            | 708.801         | 1.41083    |