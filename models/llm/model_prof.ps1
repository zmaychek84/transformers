#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

$env:MLADF = "2x4x4"
$env:MLADF_VERSION = "v1"
echo f | xcopy /f /y modeling_llama_amd.py modeling_llama_amd_bak.py
xcopy /f /y modeling_llama_add_prof.py modeling_llama_amd.py
echo f | xcopy /f /y modeling_qwen2_amd.py modeling_qwen2_amd_bak.py
xcopy /f /y modeling_qwen2_prof.py modeling_qwen2_amd.py
python run_awq.py --model_name %1 --fast_attention --fast_mlp --fast_norm --algorithm awqplus --task quantize
python run_awq.py --model_name %1 --fast_attention --fast_mlp --fast_norm --algorithm awqplus --task %2 --profile_layer True > log_prof.txt 2>&1
python summary_prof.py log_prof.txt
xcopy /f /y modeling_llama_amd_bak.py modeling_llama_amd.py
xcopy /f /y modeling_qwen2_amd_bak.py modeling_qwen2_amd.py
del modeling_llama_amd_bak.py
del modeling_qwen2_amd_bak.py
