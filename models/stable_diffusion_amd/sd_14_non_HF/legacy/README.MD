# Stable Diffusion


Runs a single example for 10 steps with CPU backend. Also saves ONNX models of Clip, Diffuser and Decoder (all FP32), traced with actual intermediate tensors.
```
python run_phoenix.py
```

The following command runs SD from HF. User can potentially use Optimum for model exports but it does not with yet.
```
python run.py
```
