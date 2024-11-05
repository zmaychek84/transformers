## Performance on HPT and STX

### Save quantized checkpoints
```
python .\prepare_model.py --model_name <model_name> --output_model_dir <output directory> --export --optimize --quantize --opt_level {0,1,2}
```
⚠️ **Disclaimer:** The output model dir has fp32, fp32_optm and quant folders,copy the models.data.onnx file from the quant folder to the current folder(models/llm_onnx). If not done , we get wrong results for AIE flow:

### Run Codellama decode examples
```powershell
# Go to test directory
cd models\llm_onnx

#Run codellama inference with decode option
python .\infer.py  --target <cpu/aie> --model-dir <model output directory> --profile --task decode
```

### Run codellama inference with benchmark option
```
# This runs benchmark for different sequence lengths
python .\infer.py --model_name <model_name> --target <cpu/aie> --model_dir <input model directory> --profile --task benchmark

# Run for specific sequence length (e.g. 2048)
python .\infer.py --model_name <model_name> --target <cpu/aie> --model_dir <input model directory> --profile --task benchmark --seqlen 2048
```
:pushpin: We recommend to use opt_level = 0 for this model for better accuracy, the numbers are here are for optimization level 0  

## Latency: STRIX (VAI-EP) (AMD Eng Sample: 100-000000994-37_Y, 12 Core(s))
### Group Size = 128

| Example# | Prompt Length (tokens) | (New Tokens Generated) |  Total Time (s) |  Prefill Phase (ms) | Time/Token (ms) | Tokens/Sec |
| -------- | ---------------------- | ---------------------- | --------------- | ------------------- | --------------- | ---------- |
| 1        | 8                      | 11                     | 1.57928         | 200.055             | 135.951         | 7.35559    |
| 2        | 16                     | 11                     | 1.81095         | 467.281             | 132.632         | 7.53966    |
| 3        | 32                     | 11                     | 1.83709         | 491.998             | 132.881         | 7.5255     |
| 4        | 64                     | 11                     | 2.30014         | 920.064             | 136.561         | 7.32274    |
| 5        | 128                    | 11                     | 3.52392         | 2043.57             | 146.679         | 6.8176     |
| 6        | 256                    | 11                     | 5.53951         | 3818.26             | 170.551         | 5.86335    |
| 7        | 512                    | 11                     | 10.6962         | 8292.56             | 237.842         | 4.20448    |
| 8        | 1024                   | 11                     | 21.5842         | 17930.1             | 357.471         | 2.79743    |
| 9        | 1536                   | 11                     | 34.7247         | 30055.1             | 456.683         | 2.1897     |
| 10       | 2048                   | 11                     | 47.7427         | 42128.8             | 551.413         | 1.81352    |
| 11       | 4096                   | 11                     | 111.316         | 101083              | 1006.32         | 0.99372    |

## Latency: Hawkpoint (VAI-EP) (AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 8 cores)
### Group Size = 128

| Example# | Prompt Length (tokens) | New Tokens Generated | Total Time (s) | Prefill Phase (ms) | Time/Token (ms) | Tokens/Sec |
| -------- | ---------------------- | -------------------- | -------------- | ------------------ | --------------- | ---------- |
| 1        | 8                      | 11                   | 1.86041        | 254.676            | 159.093         | 6.28564    |
| 2        | 16                     | 11                   | 2.36193        | 739.054            | 160.107         | 6.24584    |
| 3        | 32                     | 11                   | 2.3875         | 754.88             | 162.079         | 6.16983    |
| 4        | 64                     | 11                   | 3.17036        | 1493.96            | 166.617         | 6.0018     |
| 5        | 128                    | 11                   | 5.07626        | 3283.13            | 178.267         | 5.60956    |
| 6        | 256                    | 11                   | 8.24275        | 6227.61            | 200.164         | 4.9959     |
| 7        | 512                    | 11                   | 16.4936        | 13919.1            | 255.442         | 3.91478    |
| 8        | 1024                   | 11                   | 32.7317        | 29191.7            | 349.284         | 2.863      |
| 9        | 1536                   | 11                   | 47.674         | 42913.1            | 469.705         | 2.129      |
| 10       | 2048                   | 11                   | 64.4792        | 58319.3            | 608.596         | 1.64313    |
| 11       | 4096                   | 11                   | 140.705        | 130598             | 998.007         | 1.002      |