## Performance on HPT and STX

### Save quantized checkpoints
```
python .\prepare_model.py --model_name <model_name> --output_model_dir <output directory> --export --optimize --quantize --opt_level {0,1,2}
```
⚠️ **Disclaimer:** The output model dir has fp32, fp32_optm and quant folders,copy the models.data.onnx file from the quant folder to the current folder(models/llm_onnx). If not done , we get wrong results for AIE flow:

### Run OPT-125m decode examples
```powershell
# Go to test directory
cd models\llm_onnx

#Run opt-125m inference with decode option
python .\infer.py  --target <cpu/aie> --model-dir <model output directory> --profile --task decode
```

### Run opt-125m inference with benchmark option
```
# This runs benchmark for different sequence lengths
python .\infer.py --model_name <model_name> --target <cpu/aie> --model_dir <input model directory> --profile --task benchmark

# Run for specific sequence length (e.g. 2048)
python .\infer.py --model_name <model_name> --target <cpu/aie> --model_dir <input model directory> --profile --task benchmark --seqlen 2048
```
:pushpin: We recommend to use opt_level = 0 for this model for better performance, the numbers are here are for optimization level 0  

## Latency: STRIX (VAI-EP) (AMD Eng Sample: 100-000000994-37_Y, 12 Core(s))
### Group Size = 128

| Example# | Prompt Length (tokens) | New Tokens Generated | Total Time (s) | Prefill Phase (ms) | Time/Token (ms) | Tokens/Sec |
| -------- | ---------------------- | -------------------- | -------------- | ------------------ | --------------- | ---------- |
| 1        | 8                      | 11                   | 0.381779       | 45.5335            | 32.0956         | 31.157     |
| 2        | 16                     | 11                   | 0.413316       | 85.1037            | 31.2718         | 31.9777    |
| 3        | 32                     | 11                   | 0.424787       | 90.0428            | 31.7825         | 31.4638    |
| 4        | 64                     | 11                   | 0.504126       | 170.459            | 31.6204         | 31.6251    |
| 5        | 128                    | 11                   | 0.668498       | 327.938            | 32.2505         | 31.0073    |
| 6        | 256                    | 11                   | 1.05892        | 651.431            | 38.1514         | 26.2114    |
| 7        | 512                    | 11                   | 1.85569        | 1349.87            | 46.5773         | 21.4697    |
| 8        | 1024                   | 11                   | 3.52879        | 2976.49            | 49.8903         | 20.044     |
| 9        | 1536                   | 11                   | 5.42484        | 4792.74            | 57.127          | 17.5048    |

## Latency: Hawkpoint (VAI-EP) (AMD Ryzen 7 8845HS w/ Radeon 780M Graphics, 8 cores)
### Group Size = 128

| Example# | Prompt Length (tokens) | New Tokens Generated | Total Time (s) | Prefill Phase (ms) | Time/Token (ms) | Tokens/Sec |
| -------- | ---------------------- | -------------------- | -------------- | ------------------ | --------------- | ---------- |
| 1        | 8                      | 11                   | 0.58856        | 83.5313            | 49.2435         | 20.3072    |
| 2        | 16                     | 11                   | 0.698625       | 198.185            | 48.9002         | 20.4498    |
| 3        | 32                     | 11                   | 0.7205         | 202.513            | 50.5756         | 19.7724    |
| 4        | 64                     | 11                   | 0.968445       | 458.817            | 49.7356         | 20.1063    |
| 5        | 128                    | 11                   | 1.49847        | 983.111            | 50.1653         | 19.9341    |
| 6        | 256                    | 11                   | 2.12025        | 1597.49            | 50.857          | 19.663     |
| 7        | 512                    | 11                   | 4.2594         | 3718.34            | 52.3821         | 19.0905    |
| 8        | 1024                   | 11                   | 8.0525         | 7476.88            | 55.7328         | 17.9428    |
| 9        | 1536                   | 11                   | 12.8734        | 12262.1            | 58.8781         | 16.9843    |

