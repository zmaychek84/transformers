import argparse
import json

import numpy
import onnx
import vai_q_onnx
from onnxruntime.quantization import CalibrationDataReader
from vai_q_onnx.tools.convert_customqdq_to_qdq import (
    convert_customqdq_to_qdq,
    custom_ops_infer_shapes,
)

downblock_1_float_model_path = "../unet_onnx/sdxl_turbo/downblock_1/down.onnx"
downblock_1_quant_model_path = "../unet_quant_onnx/sdxl_turbo/downblock_1/"
downblock_2_float_model_path = "../unet_onnx/sdxl_turbo/downblock_2/down.onnx"
downblock_2_quant_model_path = "../unet_quant_onnx/sdxl_turbo/downblock_2/"
downblock_3_float_model_path = "../unet_onnx/sdxl_turbo/downblock_3/down.onnx"
downblock_3_quant_model_path = "../unet_quant_onnx/sdxl_turbo/downblock_3/"
midblock_float_model_path = "../unet_onnx/sdxl_turbo/midblock/mid.onnx"
midblock_quant_model_path = "../unet_quant_onnx/sdxl_turbo/midblock/"
upblock_1_float_model_path = "../unet_onnx/sdxl_turbo/upblock_1/up.onnx"
upblock_1_quant_model_path = "../unet_quant_onnx/sdxl_turbo/upblock_1/"
upblock_2_float_model_path = "../unet_onnx/sdxl_turbo/upblock_2/up.onnx"
upblock_2_quant_model_path = "../unet_quant_onnx/sdxl_turbo/upblock_2/"

print("Started Reading Tensor Data JSON file")
read_file = open("unetTensorsData.json", "r")
print("Converting JSON encoded data into Numpy array")
tensorsData = json.load(read_file)


class UnetDownblock1DataReader(CalibrationDataReader):
    def __init__(self):
        self.enum_data = []
        self.data_size = 2

        for i in range(1, self.data_size):
            self.enum_data.append(
                {
                    "sample_in": numpy.asarray(
                        tensorsData[f"{i}_downblock_1_sample_in"], dtype=numpy.float32
                    ),
                    "emb": numpy.asarray(
                        tensorsData[f"{i}_downblock_1_emb"], dtype=numpy.float32
                    ),
                    "down_block_res_samples": numpy.asarray(
                        tensorsData[f"{i}_downblock_1_down_block_res_samples"],
                        dtype=numpy.float32,
                    ),
                }
            )
        self.enum_data = iter(self.enum_data)

    def get_next(self):
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None


class UnetDownblock2DataReader(CalibrationDataReader):
    def __init__(self):
        self.enum_data = []
        self.data_size = 2

        for i in range(1, self.data_size):
            self.enum_data.append(
                {
                    "sample_in": numpy.asarray(
                        tensorsData[f"{i}_downblock_2_sample_in"], dtype=numpy.float32
                    ),
                    "emb": numpy.asarray(
                        tensorsData[f"{i}_downblock_2_emb"], dtype=numpy.float32
                    ),
                    "encoder_hidden_states": numpy.asarray(
                        tensorsData[f"{i}_downblock_2_encoder_hidden_states"],
                        dtype=numpy.float32,
                    ),
                    "down_block_res_samples_input_1": numpy.asarray(
                        tensorsData[f"{i}_downblock_2_down_block_res_samples_input_1"],
                        dtype=numpy.float32,
                    ),
                    "down_block_res_samples_input_2": numpy.asarray(
                        tensorsData[f"{i}_downblock_2_down_block_res_samples_input_2"],
                        dtype=numpy.float32,
                    ),
                    "down_block_res_samples_input_3": numpy.asarray(
                        tensorsData[f"{i}_downblock_2_down_block_res_samples_input_3"],
                        dtype=numpy.float32,
                    ),
                    "down_block_res_samples_input_4": numpy.asarray(
                        tensorsData[f"{i}_downblock_2_down_block_res_samples_input_4"],
                        dtype=numpy.float32,
                    ),
                }
            )
        self.enum_data = iter(self.enum_data)

    def get_next(self):
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None


class UnetDownblock3DataReader(CalibrationDataReader):
    def __init__(self):
        self.enum_data = []
        self.data_size = 2

        for i in range(1, self.data_size):
            self.enum_data.append(
                {
                    "sample_in": numpy.asarray(
                        tensorsData[f"{i}_downblock_3_sample_in"], dtype=numpy.float32
                    ),
                    "emb": numpy.asarray(
                        tensorsData[f"{i}_downblock_3_emb"], dtype=numpy.float32
                    ),
                    "encoder_hidden_states": numpy.asarray(
                        tensorsData[f"{i}_downblock_3_encoder_hidden_states"],
                        dtype=numpy.float32,
                    ),
                    "down_block_res_samples_input_1": numpy.asarray(
                        tensorsData[f"{i}_downblock_3_down_block_res_samples_input_1"],
                        dtype=numpy.float32,
                    ),
                    "down_block_res_samples_input_2": numpy.asarray(
                        tensorsData[f"{i}_downblock_3_down_block_res_samples_input_2"],
                        dtype=numpy.float32,
                    ),
                    "down_block_res_samples_input_3": numpy.asarray(
                        tensorsData[f"{i}_downblock_3_down_block_res_samples_input_3"],
                        dtype=numpy.float32,
                    ),
                    "down_block_res_samples_input_4": numpy.asarray(
                        tensorsData[f"{i}_downblock_3_down_block_res_samples_input_4"],
                        dtype=numpy.float32,
                    ),
                    "down_block_res_samples_input_5": numpy.asarray(
                        tensorsData[f"{i}_downblock_3_down_block_res_samples_input_5"],
                        dtype=numpy.float32,
                    ),
                    "down_block_res_samples_input_6": numpy.asarray(
                        tensorsData[f"{i}_downblock_3_down_block_res_samples_input_6"],
                        dtype=numpy.float32,
                    ),
                    "down_block_res_samples_input_7": numpy.asarray(
                        tensorsData[f"{i}_downblock_3_down_block_res_samples_input_7"],
                        dtype=numpy.float32,
                    ),
                }
            )
        self.enum_data = iter(self.enum_data)

    def get_next(self):
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None


class UnetMidblockDataReader(CalibrationDataReader):
    def __init__(self):
        self.enum_data = []
        self.data_size = 2

        for i in range(1, self.data_size):
            self.enum_data.append(
                {
                    "sample_in": numpy.asarray(
                        tensorsData[f"{i}_midblock_sample_in"], dtype=numpy.float32
                    ),
                    "emb": numpy.asarray(
                        tensorsData[f"{i}_midblock_emb"], dtype=numpy.float32
                    ),
                    "encoder_hidden_states": numpy.asarray(
                        tensorsData[f"{i}_midblock_encoder_hidden_states"],
                        dtype=numpy.float32,
                    ),
                }
            )
        self.enum_data = iter(self.enum_data)

    def get_next(self):
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None


class UnetUpblock1DataReader(CalibrationDataReader):
    def __init__(self):
        self.enum_data = []
        self.data_size = 2

        for i in range(1, self.data_size):
            self.enum_data.append(
                {
                    "sample_in": numpy.asarray(
                        tensorsData[f"{i}_upblock_1_sample_in"], dtype=numpy.float32
                    ),
                    "emb": numpy.asarray(
                        tensorsData[f"{i}_upblocks_emb"], dtype=numpy.float32
                    ),
                    "encoder_hidden_states": numpy.asarray(
                        tensorsData[f"{i}_upblocks_encoder_hidden_states"],
                        dtype=numpy.float32,
                    ),
                    "down_block_res_samples_input_7": numpy.asarray(
                        tensorsData[f"{i}_upblock_1_down_block_res_samples_input_7"],
                        dtype=numpy.float32,
                    ),
                    "down_block_res_samples_input_8": numpy.asarray(
                        tensorsData[f"{i}_upblock_1_down_block_res_samples_input_8"],
                        dtype=numpy.float32,
                    ),
                    "down_block_res_samples_input_9": numpy.asarray(
                        tensorsData[f"{i}_upblock_1_down_block_res_samples_input_9"],
                        dtype=numpy.float32,
                    ),
                }
            )
        self.enum_data = iter(self.enum_data)

    def get_next(self):
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None


class UnetUpblock2DataReader(CalibrationDataReader):
    def __init__(self):
        self.enum_data = []
        self.data_size = 2

        for i in range(1, self.data_size):
            self.enum_data.append(
                {
                    "sample_in": numpy.asarray(
                        tensorsData[f"{i}_upblock_2_sample_in"], dtype=numpy.float32
                    ),
                    "emb": numpy.asarray(
                        tensorsData[f"{i}_upblocks_emb"], dtype=numpy.float32
                    ),
                    "encoder_hidden_states": numpy.asarray(
                        tensorsData[f"{i}_upblocks_encoder_hidden_states"],
                        dtype=numpy.float32,
                    ),
                    "down_block_res_samples_input_1": numpy.asarray(
                        tensorsData[f"{i}_upblock_2_down_block_res_samples_input_1"],
                        dtype=numpy.float32,
                    ),
                    "down_block_res_samples_input_2": numpy.asarray(
                        tensorsData[f"{i}_upblock_2_down_block_res_samples_input_2"],
                        dtype=numpy.float32,
                    ),
                    "down_block_res_samples_input_3": numpy.asarray(
                        tensorsData[f"{i}_upblock_2_down_block_res_samples_input_3"],
                        dtype=numpy.float32,
                    ),
                    "down_block_res_samples_input_4": numpy.asarray(
                        tensorsData[f"{i}_upblock_2_down_block_res_samples_input_4"],
                        dtype=numpy.float32,
                    ),
                    "down_block_res_samples_input_5": numpy.asarray(
                        tensorsData[f"{i}_upblock_2_down_block_res_samples_input_5"],
                        dtype=numpy.float32,
                    ),
                    "down_block_res_samples_input_6": numpy.asarray(
                        tensorsData[f"{i}_upblock_2_down_block_res_samples_input_6"],
                        dtype=numpy.float32,
                    ),
                }
            )
        self.enum_data = iter(self.enum_data)

    def get_next(self):
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None


def quantize(quant_type):
    if quant_type == "a8w8":
        vai_q_onnx.quantize_static(
            downblock_1_float_model_path,
            downblock_1_quant_model_path + "down_a8w8.onnx",
            UnetDownblock1DataReader(),
            quant_format=vai_q_onnx.QuantFormat.QDQ,
            activation_type=vai_q_onnx.QuantType.QInt8,
            calibrate_method=vai_q_onnx.PowerOfTwoMethod.NonOverflow,
            weight_type=vai_q_onnx.QuantType.QInt8,
            op_types_to_quantize=["Conv", "Gemm", "MatMul"],
            use_external_data_format=True,
            enable_ipu_cnn=True,
            extra_options={
                "ActivationSymmetric": True,
                "WeightSymmetric": True,
                "QuantizeBias": True,
            },
        )

        vai_q_onnx.quantize_static(
            downblock_2_float_model_path,
            downblock_2_quant_model_path + "down_a8w8.onnx",
            UnetDownblock2DataReader(),
            quant_format=vai_q_onnx.QuantFormat.QDQ,
            activation_type=vai_q_onnx.QuantType.QInt8,
            calibrate_method=vai_q_onnx.PowerOfTwoMethod.NonOverflow,
            weight_type=vai_q_onnx.QuantType.QInt8,
            op_types_to_quantize=["Conv", "Gemm", "MatMul"],
            use_external_data_format=True,
            enable_ipu_cnn=True,
            extra_options={
                "ActivationSymmetric": True,
                "WeightSymmetric": True,
                "QuantizeBias": True,
            },
        )

        vai_q_onnx.quantize_static(
            downblock_3_float_model_path,
            downblock_3_quant_model_path + "down_a8w8.onnx",
            UnetDownblock3DataReader(),
            quant_format=vai_q_onnx.QuantFormat.QDQ,
            activation_type=vai_q_onnx.QuantType.QInt8,
            calibrate_method=vai_q_onnx.PowerOfTwoMethod.NonOverflow,
            weight_type=vai_q_onnx.QuantType.QInt8,
            op_types_to_quantize=["Conv", "Gemm", "MatMul"],
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
            UnetMidblockDataReader(),
            quant_format=vai_q_onnx.QuantFormat.QDQ,
            activation_type=vai_q_onnx.QuantType.QInt8,
            calibrate_method=vai_q_onnx.PowerOfTwoMethod.NonOverflow,
            weight_type=vai_q_onnx.QuantType.QInt8,
            op_types_to_quantize=["Conv", "Gemm", "MatMul"],
            use_external_data_format=True,
            enable_ipu_cnn=True,
            extra_options={
                "ActivationSymmetric": True,
                "WeightSymmetric": True,
                "QuantizeBias": True,
            },
        )

        vai_q_onnx.quantize_static(
            upblock_1_float_model_path,
            upblock_1_quant_model_path + "up_a8w8.onnx",
            UnetUpblock1DataReader(),
            quant_format=vai_q_onnx.QuantFormat.QDQ,
            activation_type=vai_q_onnx.QuantType.QInt8,
            calibrate_method=vai_q_onnx.PowerOfTwoMethod.NonOverflow,
            weight_type=vai_q_onnx.QuantType.QInt8,
            op_types_to_quantize=["Conv", "Gemm", "MatMul"],
            use_external_data_format=True,
            enable_ipu_cnn=True,
            extra_options={
                "ActivationSymmetric": True,
                "WeightSymmetric": True,
                "QuantizeBias": True,
            },
        )

        vai_q_onnx.quantize_static(
            upblock_2_float_model_path,
            upblock_2_quant_model_path + "up_a8w8.onnx",
            UnetUpblock2DataReader(),
            quant_format=vai_q_onnx.QuantFormat.QDQ,
            activation_type=vai_q_onnx.QuantType.QInt8,
            calibrate_method=vai_q_onnx.PowerOfTwoMethod.NonOverflow,
            weight_type=vai_q_onnx.QuantType.QInt8,
            op_types_to_quantize=["Conv", "Gemm", "MatMul"],
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
            downblock_1_float_model_path,
            downblock_1_quant_model_path + "down_a16w8_vitis.onnx",
            UnetDownblock1DataReader(),
            quant_format=vai_q_onnx.VitisQuantFormat.QDQ,
            activation_type=vai_q_onnx.VitisQuantType.QInt16,
            calibrate_method=vai_q_onnx.PowerOfTwoMethod.NonOverflow,
            weight_type=vai_q_onnx.QuantType.QInt8,
            op_types_to_quantize=["Conv", "Gemm", "MatMul"],
            use_external_data_format=True,
            enable_ipu_cnn=True,
            extra_options={
                "ActivationSymmetric": True,
                "WeightSymmetric": True,
                "QuantizeBias": True,
            },
        )
        model = onnx.load_model(downblock_1_quant_model_path + "down_a16w8_vitis.onnx")
        converted_model = convert_customqdq_to_qdq(model)
        converted_model = custom_ops_infer_shapes(converted_model)
        onnx.save(converted_model, downblock_1_quant_model_path + "down_a16w8.onnx")

        vai_q_onnx.quantize_static(
            downblock_2_float_model_path,
            downblock_2_quant_model_path + "down_a16w8_vitis.onnx",
            UnetDownblock2DataReader(),
            quant_format=vai_q_onnx.VitisQuantFormat.QDQ,
            activation_type=vai_q_onnx.VitisQuantType.QInt16,
            calibrate_method=vai_q_onnx.PowerOfTwoMethod.NonOverflow,
            weight_type=vai_q_onnx.QuantType.QInt8,
            op_types_to_quantize=["Conv", "Gemm", "MatMul"],
            use_external_data_format=True,
            enable_ipu_cnn=True,
            extra_options={
                "ActivationSymmetric": True,
                "WeightSymmetric": True,
                "QuantizeBias": True,
            },
        )
        model = onnx.load_model(downblock_2_quant_model_path + "down_a16w8_vitis.onnx")
        converted_model = convert_customqdq_to_qdq(model)
        converted_model = custom_ops_infer_shapes(converted_model)
        onnx.save(converted_model, downblock_2_quant_model_path + "down_a16w8.onnx")

        vai_q_onnx.quantize_static(
            downblock_3_float_model_path,
            downblock_3_quant_model_path + "down_a16w8_vitis.onnx",
            UnetDownblock3DataReader(),
            quant_format=vai_q_onnx.VitisQuantFormat.QDQ,
            activation_type=vai_q_onnx.VitisQuantType.QInt16,
            calibrate_method=vai_q_onnx.PowerOfTwoMethod.NonOverflow,
            weight_type=vai_q_onnx.QuantType.QInt8,
            op_types_to_quantize=["Conv", "Gemm", "MatMul"],
            use_external_data_format=True,
            enable_ipu_cnn=True,
            extra_options={
                "ActivationSymmetric": True,
                "WeightSymmetric": True,
                "QuantizeBias": True,
            },
        )
        model = onnx.load_model(downblock_3_quant_model_path + "down_a16w8_vitis.onnx")
        converted_model = convert_customqdq_to_qdq(model)
        converted_model = custom_ops_infer_shapes(converted_model)
        onnx.save(converted_model, downblock_3_quant_model_path + "down_a16w8.onnx")

        vai_q_onnx.quantize_static(
            midblock_float_model_path,
            midblock_quant_model_path + "mid_a16w8_vitis.onnx",
            UnetMidblockDataReader(),
            quant_format=vai_q_onnx.VitisQuantFormat.QDQ,
            activation_type=vai_q_onnx.VitisQuantType.QInt16,
            calibrate_method=vai_q_onnx.PowerOfTwoMethod.NonOverflow,
            weight_type=vai_q_onnx.QuantType.QInt8,
            op_types_to_quantize=["Conv", "Gemm", "MatMul"],
            use_external_data_format=True,
            enable_ipu_cnn=True,
            extra_options={
                "ActivationSymmetric": True,
                "WeightSymmetric": True,
                "QuantizeBias": True,
            },
        )
        model = onnx.load_model(midblock_quant_model_path + "mid_a16w8_vitis.onnx")
        converted_model = convert_customqdq_to_qdq(model)
        converted_model = custom_ops_infer_shapes(converted_model)
        onnx.save(converted_model, midblock_quant_model_path + "mid_a16w8.onnx")

        vai_q_onnx.quantize_static(
            upblock_1_float_model_path,
            upblock_1_quant_model_path + "up_a16w8_vitis.onnx",
            UnetUpblock1DataReader(),
            quant_format=vai_q_onnx.VitisQuantFormat.QDQ,
            activation_type=vai_q_onnx.VitisQuantType.QInt16,
            calibrate_method=vai_q_onnx.PowerOfTwoMethod.NonOverflow,
            weight_type=vai_q_onnx.QuantType.QInt8,
            op_types_to_quantize=["Conv", "Gemm", "MatMul"],
            use_external_data_format=True,
            enable_ipu_cnn=True,
            extra_options={
                "ActivationSymmetric": True,
                "WeightSymmetric": True,
                "QuantizeBias": True,
            },
        )
        model = onnx.load_model(upblock_1_quant_model_path + "up_a16w8_vitis.onnx")
        converted_model = convert_customqdq_to_qdq(model)
        converted_model = custom_ops_infer_shapes(converted_model)
        onnx.save(converted_model, upblock_1_quant_model_path + "up_a16w8.onnx")

        vai_q_onnx.quantize_static(
            upblock_2_float_model_path,
            upblock_2_quant_model_path + "up_a16w8_vitis.onnx",
            UnetUpblock2DataReader(),
            quant_format=vai_q_onnx.VitisQuantFormat.QDQ,
            activation_type=vai_q_onnx.VitisQuantType.QInt16,
            calibrate_method=vai_q_onnx.PowerOfTwoMethod.NonOverflow,
            weight_type=vai_q_onnx.QuantType.QInt8,
            op_types_to_quantize=["Conv", "Gemm", "MatMul"],
            use_external_data_format=True,
            enable_ipu_cnn=True,
            extra_options={
                "ActivationSymmetric": True,
                "WeightSymmetric": True,
                "QuantizeBias": True,
            },
        )
        model = onnx.load_model(upblock_2_quant_model_path + "up_a16w8_vitis.onnx")
        converted_model = convert_customqdq_to_qdq(model)
        converted_model = custom_ops_infer_shapes(converted_model)
        onnx.save(converted_model, upblock_2_quant_model_path + "up_a16w8.onnx")


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
