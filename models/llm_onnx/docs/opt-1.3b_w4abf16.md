## Performance on HPT and STX

### Save quantized checkpoints
```
python .\prepare_model.py --model_name <model_name> --output_model_dir <output directory> --export --optimize --quantize --opt_level {0,1,2}
```
⚠️ **Disclaimer:** The output model dir has fp32, fp32_optm and quant folders,copy the models.data.onnx file from the quant folder to the current folder(models/llm_onnx). If not done , we get wrong results for AIE flow:

### Run OPT-1.3b decode examples
```powershell
# Go to test directory
cd models\llm_onnx

#Run opt-1.3b inference with decode option
python .\infer.py  --target <cpu/aie> --model-dir <model output directory> --profile --task decode
```

### Run opt-1.3b inference with benchmark option
```
# This runs benchmark for different sequence lengths
python .\infer.py --model_name <model_name> --target <cpu/aie> --model_dir <input model directory> --profile --task benchmark

# Run for specific sequence length (e.g. 2048)
python .\infer.py --model_name <model_name> --target <cpu/aie> --model_dir <input model directory> --profile --task benchmark --seqlen 2048
```
:pushpin: We recommend to use opt_level = 0 for this model for better performance, the numbers are here are for optimization level 0  

## Latency: STRIX (VAI-EP) (AMD Eng Sample: 100-000000994-37_Y, 12 Core(s))
### Group Size = 128
| Example# |  Prompt Length (tokens) | New Tokens Generated | Total Time (s) | Prefill Phase (ms) | Time/Token (ms) |  Tokens/Sec |
| -------- | ----------------------- | -------------------- | -------------- | ------------------ | --------------- | ----------- |
| 1        | 8                       | 11                   | 1.24104        | 131.9              | 109.081         | 9.16754     |
| 2        | 16                      | 11                   | 1.30438        | 219.587            | 106.644         | 9.37696     |
| 3        | 32                      | 11                   | 1.33925        | 261.396            | 105.966         | 9.43702     |
| 4        | 64                      | 11                   | 1.64483        | 551.624            | 107.455         | 9.30621     |
| 5        | 128                     | 11                   | 2.15906        | 1033.23            | 109.843         | 9.10388     |
| 6        | 256                     | 11                   | 3.34404        | 2135.59            | 118.532         | 8.43653     |
| 7        | 512                     | 11                   | 5.04962        | 3610.69            | 141.72          | 7.05619     |
| 8        | 1024                    | 11                   | 9.64283        | 7887.17            | 168.162         | 5.94664     |
| 9        | 1536                    | 11                   | 15.8456        | 13908.1            | 186.123         | 5.3728      |

## Latency: Hawkpoint (VAI-EP) (AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 8 cores)
### Group Size = 128

| Example# | Prompt Length (tokens) | New Tokens Generated | Total Time (s) | Prefill Phase (ms) | Time/Token (ms) | Tokens/Sec |
| -------- | ---------------------- | -------------------- | -------------- | ------------------ | --------------- | ---------- |
| 1        | 8                      | 11                   | 1.68583        | 194.968            | 147.203         | 6.79336    |
| 2        | 16                     | 11                   | 1.89587        | 413.379            | 146.879         | 6.80831    |
| 3        | 32                     | 11                   | 2.10951        | 621.083            | 147.482         | 6.7805     |
| 4        | 64                     | 11                   | 3.38691        | 1886.07            | 148.847         | 6.71833    |
| 5        | 128                    | 11                   | 6.34364        | 4792.94            | 153.159         | 6.52916    |
| 6        | 256                    | 11                   | 11.2723        | 9629.32            | 163.092         | 6.13153    |
| 7        | 512                    | 11                   | 16.0959        | 14324.1            | 175.773         | 5.68915    |
| 8        | 1024                   | 11                   | 23.9508        | 21981.4            | 194.825         | 5.1328     |
| 9        | 1536                   | 11                   | 30.4419        | 28245.9            | 217.224         | 4.60355    |
