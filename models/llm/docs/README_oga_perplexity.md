# readme_oga_perplexity

The llm_eval.py file contains a `perplexity()` function that can now be used to measure perplexity of an OGA model.

## Requirements

In order to run an OGA model on the CPU, the onnxruntime-genai package should be installed (if not already done):

```
pip install onnxruntime-genai
```

In order to run any OGA model on the NPU, the following commands should be executed to install dependencies for the Vitis EP:

**Download and extract the wheel files folder from this [link](https://amdcloud-my.sharepoint.com/:f:/r/personal/walaamer_amd_com/Documents/oga-perplexity-wheels?csf=1&web=1&e=or8TZG).**
```
cd oga-perplexity-wheels
pip install --force-reinstall onnxruntime_vitisai-1.19.0-cp310-cp310-win_amd64.whl
pip install --force-reinstall onnxruntime_genai-0.4.0.dev0-cp310-cp310-win_amd64.whl
pip install --force-reinstall voe-1.2.0-cp310-cp310-win_amd64.whl
```

The user should also modify the genai_config.json to pick the device on which the model is running:

If running on CPU:
```
"model": {
        "decoder": {
            "session_options": {
                "provider_options": []
            }
         }
}
```

If running on NPU:
```
"model": {
        "decoder": {
            "session_options": {
                "provider_options": [
                    {
                        "VitisAI": {
                            "config_file": "vaip_llm.json"
                        }
                    }
                ]
            }
         }
}
```

## Usage
```
llm_eval.perplexity(<model>, framework="pytorch" {"oga", "pytorch"}, dataset="raw" {"raw", "non-raw"}, device="cpu" {"cpu", "aie", "npugpu"}, fsamples=1 {fraction of the entire datset to use (<=1)})
```
⚠️ To run the function with OGA, `<model>` should be the model's path, but running it with pytorch requires passing the loaded model itself.


## Example
```
import llm_eval
llm_eval.perplexity(<model_path>, framework="oga", device="aie")
```
