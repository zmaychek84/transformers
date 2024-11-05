## Performance on HPT and STX

### Save quantized checkpoints
```
python .\prepare_model.py --model_name <model_name> --output_model_dir <output directory> --export --optimize --quantize --opt_level {0,1,2}
```
⚠️ **Disclaimer:** The output model dir has fp32, fp32_optm and quant folders,copy the models.data.onnx file from the quant folder to the current folder(models/llm_onnx). If not done , we get wrong results for AIE flow:

### Run OPT-2.7b decode examples
```powershell
# Go to test directory
cd models\llm_onnx

#Run opt-2.7b inference with decode option
python .\infer.py  --target <cpu/aie> --model-dir <model output directory> --profile --task decode
```

### Run opt-2.7b inference with benchmark option
```
# This runs benchmark for different sequence lengths
python .\infer.py --model_name <model_name> --target <cpu/aie> --model_dir <input model directory> --profile --task benchmark

# Run for specific sequence length (e.g. 2048)
python .\infer.py --model_name <model_name> --target <cpu/aie> --model_dir <input model directory> --profile --task benchmark --seqlen 2048
```
:pushpin: We recommend to use opt_level = 0 for this model for better performance, the numbers are here are for optimization level 0  

## Latency: STRIX (VAI-EP) (AMD Eng Sample: 100-000000994-37_Y, 12 Core(s))
### Group Size = 128
| Example# | Prompt Length (tokens) |  New Tokens Generated | Total Time (s) |  Prefill Phase (ms) | Time/Token (ms) | Tokens/Sec |
| -------- | ---------------------- | --------------------- | -------------- | ------------------- | --------------- | ---------- |
| 1        | 8                      | 11                    | 2.06964        | 213.314             | 183.616         | 5.44616    |
| 2        | 16                     | 11                    | 2.16333        | 328.652             | 181.445         | 5.51131    |
| 3        | 32                     | 11                    | 2.24142        | 405.634             | 181.418         | 5.51212    |
| 4        | 64                     | 11                    | 2.70453        | 855.893             | 182.764         | 5.47155    |
| 5        | 128                    | 11                    | 3.49055        | 1625.48             | 184.769         | 5.41216    |
| 6        | 256                    | 11                    | 5.27614        | 3308.1              | 194.672         | 5.13685    |
| 7        | 512                    | 11                    | 7.463          | 5201.53             | 221.828         | 4.508      |
| 8        | 1024                   | 11                    | 14.7707        | 12112.6             | 260.075         | 3.84505    |
| 9        | 1536                   | 11                    | 24.1722        | 21211.3             | 284.708         | 3.51237    |

## Latency: Hawkpoint (VAI-EP) (AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 8 cores)
### Group Size = 128

| Example# | Prompt Length (tokens) | New Tokens Generated | Total Time (s) | Prefill Phase (ms) | Time/Token (ms) | Tokens/Sec |
| -------- | ---------------------- | -------------------- | -------------- | ------------------ | --------------- | ---------- |
| 1        | 8                      | 11                   | 2.69056        | 295.518            | 237.928         | 4.20296    |
| 2        | 16                     | 11                   | 3.00054        | 586.441            | 240.086         | 4.16517    |
| 3        | 32                     | 11                   | 3.10681        | 669.206            | 242.156         | 4.12957    |
| 4        | 64                     | 11                   | 4.16765        | 1743.81            | 240.847         | 4.15201    |
| 5        | 128                    | 11                   | 6.20316        | 3755.24            | 243.436         | 4.10786    |
| 6        | 256                    | 11                   | 9.35606        | 6833.47            | 250.951         | 3.98485    |
| 7        | 512                    | 11                   | 17.9067        | 15215.4            | 267.119         | 3.74366    |
| 8        | 1024                   | 11                   | 26.8816        | 23943.1            | 290.666         | 3.44038    |
| 9        | 1536                   | 11                   | 37.6176        | 34371.5            | 320.406         | 3.12104    |