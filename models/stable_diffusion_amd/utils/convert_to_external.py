import onnx
from onnx.external_data_helper import convert_model_to_external_data

onnx_model = onnx.load("unet/unet.onnx")

onnx.save_model(
    onnx_model,
    "unet.onnx",
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location="unet.onnx.data",
    size_threshold=1024,
    convert_attribute=False,
)
