# Running Stable Diffusion Model Family with RyzenAI

## Introduction

Run Stable Diffusion Model Family (SD1.4, SD1.5, SD2.1, SDXL Turbo) on RyzenAI NPU. Support configuration from two AIE Overlay (CONV DPU, V1 Gemm).

## Steps

### Edit environment file (env.yaml)
Change the python version to 3.10 for voe and onnxruntime compatibility.
```
- python=3.11
+ python=3.10
```

### Create Conda Env
```
conda update -n base -c defaults conda
conda env create --file=env.yaml
```
### Install Dependencies

#### Install IPU driver
  - For PHX, download driver from this [link](http://mkmartifactory.amd.com/artifactory/atg-cvml-generic-local/builds/ipu/Release/IPU_PHX_R24.01.31_RZ1.1_RC3/jenkins-CVML-IPU_Driver-ipu-windows-release-128/Release/).
  - For STX, download driver from this [link](http://mkmartifactory.amd.com/artifactory/atg-cvml-generic-local/builds/ipu/Release/NPU_MCDM_STX1.0_MSFT_197_R24.08.05_RC1_201/jenkins-CVML-IPU_Driver-ipu-windows-release-201/Release/).
#### Install onnxruntime package
  - For PHX download `voe-4.0-win_amd64.zip` package from [pre-built package](http://xcoartifactory/ui/native/vai-rt-ipu-prod-local/com/amd/onnx-rt/phx/dev/974/windows/) and extract it.
  - For STX download `voe-4.0-win_amd64.zip` package from [pre-built package](http://xcoartifactory.xilinx.com/ui/native/vai-rt-ipu-prod-local/com/amd/onnx-rt/stx/win24_0805_rc/1614/windows/) and extract it.
  - If run into errors, try using the latest onnxruntime-vitisai [pre-built package](
    http://xcoartifactory.xilinx.com/ui/native/vai-rt-ipu-prod-local/com/amd/onnx-rt/stx/win24_0805_rc/latest/windows/)
  - Activate your `ryzenai-transformers` conda environment
  ```
  conda activate ryzenai-transformers
  ```
  - Then execute following command to copy files and install packages:
  ```
  cd voe-4.0-win_amd64
  pip install --upgrade --force-reinstall onnxruntime_vitisai-*.whl
  pip install --upgrade --force-reinstall voe-*.whl
  python installer.py

  cd stable_diffusion_amd
  ```
  - if running into numpy version error, run the following
  ```
  pip install numpy==1.26.4 --force-reinstall
  ```
#### Install vai-q quantizer
  - Download vai-q wheel from [here](http://xcoartifactory/artifactory/vitis-ai-pip-master-local/Vitis-AI-IPU-Release-GHE/vai_q_onnx/1.17.0/131/windows/vai_q_onnx-1.17.0+4968cd3-py2.py3-none-win_amd64.whl)
  ```shell
  pip install vai_q_onnx-*.whl
  ```

#### Install diffusers
  ```shell
  pip install diffusers==0.23.1
  ```

### Setup Env Variables
  ```shell
  # For PHX
  .\setup_phx.bat
  # For STX
  .\setup_stx.bat
  ```

### Running Stable Diffusion Model Family
#### Models Categories:
  - SD models executed through pipeline call (Adapted from HuggingFace diffusers library with AMD RyzenAI related changes)
    - sd_14, sd_15, sd_21, sd_xl_turbo
  - SD models from MSFT
    - sd_msft
  - SD models with non-huggingface pipeline
    - sd_14_non_HF

#### Running SD models from Microsoft
- Download models (contact SP for permission to access the models)
- Copy the onnx model files (PSQ1.onnx, PSQ2.onnx, PSR.onnx, PSS.onnx) into models/stable_diffusion_amd/sd_msft
```shell
  # we use sd_14 as the example here
  cd models/stable_diffusion_amd/sd_msft
  python run_sd_msft.py --num_inference_steps 10 --use_msft_onnx
  # Image named 'sd_msft_fp32_msft_onnx_fp32_cpu_p0_512_10.png' will be generated in current directory
  ```
#### (SDXL Turbo only) Run clip and decoder onnx models for DmlEP (iGPU) with FP16 data format
- \[Prerequisite\]: Install AMD GPU Driver; Install onnxruntime-directml
- Download clip and decoder onnx files from [here](https://amdcloud-my.sharepoint.com/:f:/g/personal/chiz_amd_com/Ej7uYGO4sFJCu6kURkcJ5hwBHADBtTsFSyvisw7vMTrggA?e=Z9abkj)
- Copy clip1/clip1.onnx into 'models/stable_diffusion_amd/unet_onnx/sdxl_turbo/clip1'
- Copy clip2/clip2.onnx and clip2/clip2.onnx.data into 'models/stable_diffusion_amd/unet_onnx/sdxl_turbo/clip2'
- Copy decoder/fp16_decoder.onnx into 'models/stable_diffusion_amd/unet_onnx/sdxl_turbo/decoder'
- Run SDXL-T with clip/decoder in fp16 onnx mode, and unet in fp32 pytorch mode:
  ```shell
  python run_sdxl_turbo.py --num_inference_steps 1 --use_fp16_decoder_onnx_dml --use_fp16_clip_onnx_dml
  ```

####  Running SD models through HF pipeline call
- All these models (sd_14, sd_15, sd_21, sd_xl_turbo) share the same flow
  ```shell
  # we use sd_xl_turbo as the example here
  cd sd_xl_turbo
  ```
  -  Run original PyTorch model in bf16/fp32
  ```shell
  python run_sdxl_turbo.py --num_inference_steps 2 --dtype fp32
  # Image named 'sdxl_turbo_fp32_pytorch_cpu_p0_512_2' will be generated in current directory
  python run_sdxl_turbo.py --num_inference_steps 2 --dtype bf16
  # Image named 'sdxl_turbo_bf16_pytorch_cpu_p0_512_2' will be generated in current directory
  ```
  -  Export to onnx models
  ```shell
  python run_sdxl_turbo.py --num_inference_steps 1 --export_to_onnx
  # Fp32 ONNX models exported to 'models/stable_diffusion_amd/unet_onnx/sdxl_turbo'
  ```
  -  Run exported fp32 ONNX model
  ```shell
  python run_sdxl_turbo.py --num_inference_steps 2 --use_unet_amd_onnx
  # Image named 'sdxl_turbo_fp32_amd_onnx_cpu_p0_512_2.png' will be generated in current directory
  ```
  -  QDQ Model Quantization

  ```shell
  # Dump tensors data for quantization calibration
  python run_sdxl_turbo.py --num_inference_steps 5 --use_unet_amd_onnx --dump_tensors
  # unetTensorsData.json will be generated in current directory
  python quantize_unet_with_vaiq.py --quant_type [a8w8|a16w8]
  # Quantized ONNX models saved to  'models/stable_diffusion_amd/unet_quant_onnx/sd_14'
  ```
  -  Run quantized ONNX model
  ```shell
  python run_sdxl_turbo.py --num_inference_steps 2 --use_unet_amd_onnx --quant --quant_type a8w8
  # Image named 'sdxl_turbo_fp32_amd_onnx_quant_a8w8_cpu_p0_512_2' will be generated in current directory
  python run_sdxl_turbo.py --num_inference_steps 2 --use_unet_amd_onnx --quant --quant_type a16w8
  # Image named 'sdxl_turbo_fp32_amd_onnx_quant_a16w8_cpu_p0_512_2' will be generated in current directory
  ```
  -  Run quantized ONNX model with OnnxRuntime VitisAI EP (current with CONV DPU Only) on RyzenAI NPU
  ```shell
  python run_sdxl_turbo.py --num_inference_steps 2 --use_unet_amd_onnx --quant --vitisai
  # Image named 'sdxl_turbo_fp32_amd_onnx_quant_a8w8_vai_p0_512_2' will be generated in current directory
  ```

####  Running custom SD 1.4 model (non-HF pipeline)
- Download the stable [diffusion onnx models](https://amdcloud-my.sharepoint.com/:f:/g/personal/chiz_amd_com/EiIJiYigrpJNmZibYNkcNj0BZibRTy1EE0D63Np5xdf1Nw?e=p0czTD). Put them under models/stable_diffusion_amd/sd_14_non_HF/
- Setup transformers/ext/stable_diffusion by downloading the data files into that directory.  Download `data.v20221029.tar` from [here](https://huggingface.co/jinseokim/stable-diffusion-pytorch-data/resolve/main/data.v20221029.tar) and unpack in `transformers/ext/stable_diffusion`

- End-to-end Stable Diffusion Run (CPU EP)
  ```shell
  cd sd_14_non_HF
  ## CPU EP on original onnx model
  python run_sd.py --target cpu --timesteps 5
  ## CPU EP on quantized onnx model
  python run_sd.py --target cpu --quant --timesteps 5
  ## The generated pic will be named out_cpu_quant_0.png in the current directory.
  ```
- End-to-end Stable Diffusion Run (Vitisai EP)
  ```shell
  ## VitisAI EP on quantized onnx model
  ## Current vaip_config only enables Conv DPU overlay.
  python run_sd.py --target vai --quant --timesteps 5
  ## The generated pic will be out_{args.target}_{args.quant}_{i}.png in the current directory.
  ```


### Licence

  The model weights are converted from Stable Diffusion [checkpoint files](https://huggingface.co/CompVis/stable-diffusion-v1-4) which are licensed with [CreativeML Open RAIL-M](https://huggingface.co/spaces/CompVis/stable-diffusion-license) License.

  :pushpin: The weights values were not changed, only the layer names have been changed, which does not alter the behavior of the model.
