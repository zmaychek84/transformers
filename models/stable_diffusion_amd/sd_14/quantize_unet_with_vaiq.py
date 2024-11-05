import argparse
import json
import sys

import onnx
import vai_q_onnx
from vai_q_onnx.tools.convert_customqdq_to_qdq import (
    convert_customqdq_to_qdq,
    custom_ops_infer_shapes,
)

# setting path
sys.path.append("../utils")
from quantization_util import (
    UnetDownblocksDataReader,
    UnetMidblockDataReader,
    UnetUpblocksDataReader,
)

downblocks_float_model_path = "../unet_onnx/sd_14/down_blocks/down.onnx"
downblocks_quant_model_path = "../unet_quant_onnx/sd_14/down_blocks/"
midblock_float_model_path = "../unet_onnx/sd_14/mid_block/mid.onnx"
midblock_quant_model_path = "../unet_quant_onnx/sd_14/mid_block/"
upblocks_float_model_path = "../unet_onnx/sd_14/up_blocks/up.onnx"
upblocks_quant_model_path = "../unet_quant_onnx/sd_14/up_blocks/"

print("Started Reading Tensor Data JSON file")
read_file = open("unetTensorsData.json", "r")
print("Converting JSON encoded data into Numpy array")
tensorsData = json.load(read_file)


def quantize(quant_type):
    if quant_type == "a8w8":
        vai_q_onnx.quantize_static(
            downblocks_float_model_path,
            downblocks_quant_model_path + "down_a8w8.onnx",
            UnetDownblocksDataReader(tensorsData),
            quant_format=vai_q_onnx.QuantFormat.QDQ,
            activation_type=vai_q_onnx.QuantType.QInt8,
            calibrate_method=vai_q_onnx.PowerOfTwoMethod.NonOverflow,
            weight_type=vai_q_onnx.QuantType.QInt8,
            op_types_to_quantize=["Conv", "Gemm", "MatMul", "Softmax"],
            use_external_data_format=True,
            enable_ipu_cnn=True,
            extra_options={
                "ActivationSymmetric": True,
                "WeightSymmetric": True,
                "QuantizeBias": True,
            },
        )

        vai_q_onnx.quantize_static(
            midblock_float_model_path,
            midblock_quant_model_path + "mid_a8w8.onnx",
            UnetMidblockDataReader(tensorsData),
            quant_format=vai_q_onnx.QuantFormat.QDQ,
            activation_type=vai_q_onnx.QuantType.QInt8,
            calibrate_method=vai_q_onnx.PowerOfTwoMethod.NonOverflow,
            weight_type=vai_q_onnx.QuantType.QInt8,
            op_types_to_quantize=["Conv", "Gemm", "MatMul", "Softmax"],
            use_external_data_format=True,
            enable_ipu_cnn=True,
            extra_options={
                "ActivationSymmetric": True,
                "WeightSymmetric": True,
                "QuantizeBias": True,
            },
        )

        vai_q_onnx.quantize_static(
            upblocks_float_model_path,
            upblocks_quant_model_path + "up_a8w8.onnx",
            UnetUpblocksDataReader(tensorsData),
            quant_format=vai_q_onnx.QuantFormat.QDQ,
            activation_type=vai_q_onnx.QuantType.QInt8,
            calibrate_method=vai_q_onnx.PowerOfTwoMethod.NonOverflow,
            weight_type=vai_q_onnx.QuantType.QInt8,
            op_types_to_quantize=["Conv", "Gemm", "MatMul", "Softmax"],
            use_external_data_format=True,
            enable_ipu_cnn=True,
            extra_options={
                "ActivationSymmetric": True,
                "WeightSymmetric": True,
                "QuantizeBias": True,
            },
        )

    if quant_type == "a16w8":
        vai_q_onnx.quantize_static(
            downblocks_float_model_path,
            downblocks_quant_model_path + "down_a16w8_vitis.onnx",
            UnetDownblocksDataReader(tensorsData),
            quant_format=vai_q_onnx.VitisQuantFormat.QDQ,
            activation_type=vai_q_onnx.VitisQuantType.QInt16,
            calibrate_method=vai_q_onnx.PowerOfTwoMethod.NonOverflow,
            weight_type=vai_q_onnx.QuantType.QInt8,
            op_types_to_quantize=["Conv", "MatMul"],
            use_external_data_format=True,
            enable_ipu_cnn=True,
            extra_options={
                "ActivationSymmetric": False,
                "WeightSymmetric": True,
                "QuantizeBias": True,
                "UseQDQVitisCustomOps": True,
                "ConvertInstanceNormToDPUVersion": True,
                "FuseInstanceNorm": True,
                "AutoMixprecision": {
                    "TargetQuantType": "QInt8",
                    "OutputIndex": 0,
                    "NumTarget": 4,
                    "NoInputQDQShared": True,
                },
            },
        )

        model = onnx.load_model(downblocks_quant_model_path + "down_a16w8_vitis.onnx")
        converted_model = convert_customqdq_to_qdq(model)
        converted_model = custom_ops_infer_shapes(converted_model)
        onnx.save(converted_model, downblocks_quant_model_path + "down_a16w8.onnx")

        vai_q_onnx.quantize_static(
            midblock_float_model_path,
            midblock_quant_model_path + "mid_a16w8_vitis.onnx",
            UnetMidblockDataReader(tensorsData),
            quant_format=vai_q_onnx.VitisQuantFormat.QDQ,
            activation_type=vai_q_onnx.VitisQuantType.QInt16,
            calibrate_method=vai_q_onnx.PowerOfTwoMethod.NonOverflow,
            weight_type=vai_q_onnx.QuantType.QInt8,
            op_types_to_quantize=["Conv", "MatMul"],
            use_external_data_format=True,
            enable_ipu_cnn=True,
            extra_options={
                "ActivationSymmetric": False,
                "WeightSymmetric": True,
                "QuantizeBias": True,
                "UseQDQVitisCustomOps": True,
                "ConvertInstanceNormToDPUVersion": True,
                "FuseInstanceNorm": True,
                "AutoMixprecision": {
                    "TargetQuantType": "QInt8",
                    "OutputIndex": 0,
                    "NumTarget": 4,
                    "NoInputQDQShared": True,
                },
            },
        )

        model = onnx.load_model(midblock_quant_model_path + "mid_a16w8_vitis.onnx")
        converted_model = convert_customqdq_to_qdq(model)
        converted_model = custom_ops_infer_shapes(converted_model)
        onnx.save(converted_model, midblock_quant_model_path + "mid_a16w8.onnx")

        vai_q_onnx.quantize_static(
            upblocks_float_model_path,
            upblocks_quant_model_path + "up_a16w8_vitis.onnx",
            UnetUpblocksDataReader(tensorsData),
            quant_format=vai_q_onnx.VitisQuantFormat.QDQ,
            activation_type=vai_q_onnx.VitisQuantType.QInt16,
            calibrate_method=vai_q_onnx.PowerOfTwoMethod.NonOverflow,
            weight_type=vai_q_onnx.QuantType.QInt8,
            op_types_to_quantize=["Conv", "MatMul"],
            use_external_data_format=True,
            enable_ipu_cnn=True,
            extra_options={
                "ActivationSymmetric": False,
                "WeightSymmetric": True,
                "QuantizeBias": True,
                "UseQDQVitisCustomOps": True,
                "ConvertInstanceNormToDPUVersion": True,
                "FuseInstanceNorm": True,
                "AutoMixprecision": {
                    "TargetQuantType": "QInt8",
                    "OutputIndex": 0,
                    "NumTarget": 4,
                    "NoInputQDQShared": True,
                },
            },
        )

        model = onnx.load_model(upblocks_quant_model_path + "up_a16w8_vitis.onnx")
        converted_model = convert_customqdq_to_qdq(model)
        converted_model = custom_ops_infer_shapes(converted_model)
        onnx.save(converted_model, upblocks_quant_model_path + "up_a16w8.onnx")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quant_type",
        help="Quantization mode - w8a8",
        type=str,
        default="a8w8",
        choices=["a8w8", "a16w8"],
    )
    args = parser.parse_args()
    quantize(args.quant_type)
