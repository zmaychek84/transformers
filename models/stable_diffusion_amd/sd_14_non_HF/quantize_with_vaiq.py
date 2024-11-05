import json
from json import JSONEncoder

import numpy
import onnxruntime as ort
import vai_q_onnx
from onnxruntime.quantization import CalibrationDataReader

p1_float_model_path = "diffusion_p1.onnx"
p1_quant_model_path = "diffusion_quant_p1.onnx"
p2_float_model_path = "diffusion_p2.onnx"
p2_quant_model_path = "diffusion_quant_p2.onnx"


print("Started Reading P1 Tensor Data JSON file")
read_file = open("p1TensorsData.json", "r")
print("Converting JSON encoded data into Numpy array")
p1TensorsData = json.load(read_file)
print("Started Reading P2 Tensor Data JSON file")
read_file = open("p2TensorsData.json", "r")
print("Converting JSON encoded data into Numpy array")
p2TensorsData = json.load(read_file)


class DiffusionP1DataReader(CalibrationDataReader):
    def __init__(self):
        self.enum_data = []
        self.data_size = 2

        for i in range(self.data_size):
            self.enum_data.append(
                {
                    "input_latents": numpy.asarray(
                        p1TensorsData[f"{i}_input_latents"], dtype=numpy.float32
                    ),
                    "context": numpy.asarray(
                        p1TensorsData[f"{i}_context"], dtype=numpy.float32
                    ),
                    "time_embedding": numpy.asarray(
                        p1TensorsData[f"{i}_time_embedding"], dtype=numpy.float32
                    ),
                }
            )
        self.enum_data = iter(self.enum_data)

    def get_next(self):
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None


class DiffusionP2DataReader(CalibrationDataReader):
    def __init__(self):
        self.enum_data = []
        self.data_size = 2

        for i in range(self.data_size):
            self.enum_data.append(
                {
                    "input_latents": numpy.asarray(
                        p2TensorsData[f"{i}_input_latents"], dtype=numpy.float32
                    ),
                    "context": numpy.asarray(
                        p2TensorsData[f"{i}_context"], dtype=numpy.float32
                    ),
                    "time_embedding": numpy.asarray(
                        p2TensorsData[f"{i}_time_embedding"], dtype=numpy.float32
                    ),
                    "skip_connections_1": numpy.asarray(
                        p2TensorsData[f"{i}_skip_connections_1"], dtype=numpy.float32
                    ),
                    "skip_connections_2": numpy.asarray(
                        p2TensorsData[f"{i}_skip_connections_2"], dtype=numpy.float32
                    ),
                    "skip_connections_3": numpy.asarray(
                        p2TensorsData[f"{i}_skip_connections_3"], dtype=numpy.float32
                    ),
                    "skip_connections_4": numpy.asarray(
                        p2TensorsData[f"{i}_skip_connections_4"], dtype=numpy.float32
                    ),
                    "skip_connections_5": numpy.asarray(
                        p2TensorsData[f"{i}_skip_connections_5"], dtype=numpy.float32
                    ),
                    "skip_connections_6": numpy.asarray(
                        p2TensorsData[f"{i}_skip_connections_6"], dtype=numpy.float32
                    ),
                    "skip_connections_7": numpy.asarray(
                        p2TensorsData[f"{i}_skip_connections_7"], dtype=numpy.float32
                    ),
                    "skip_connections_8": numpy.asarray(
                        p2TensorsData[f"{i}_skip_connections_8"], dtype=numpy.float32
                    ),
                    "skip_connections_9": numpy.asarray(
                        p2TensorsData[f"{i}_skip_connections_9"], dtype=numpy.float32
                    ),
                    "skip_connections_10": numpy.asarray(
                        p2TensorsData[f"{i}_skip_connections_10"], dtype=numpy.float32
                    ),
                    "skip_connections_11": numpy.asarray(
                        p2TensorsData[f"{i}_skip_connections_11"], dtype=numpy.float32
                    ),
                    "skip_connections_12": numpy.asarray(
                        p2TensorsData[f"{i}_skip_connections_12"], dtype=numpy.float32
                    ),
                }
            )
        self.enum_data = iter(self.enum_data)

    def get_next(self):
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None


vai_q_onnx.quantize_static(
    p1_float_model_path,
    p1_quant_model_path,
    DiffusionP1DataReader(),
    quant_format=vai_q_onnx.QuantFormat.QDQ,
    activation_type=vai_q_onnx.QuantType.QUInt8,
    calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
    weight_type=vai_q_onnx.QuantType.QInt8,
    op_types_to_quantize=["Conv", "Gemm", "MatMul", "Concat"],
    use_external_data_format=True,
    enable_dpu=True,
    extra_options={
        "ActivationSymmetric": True,
        "WeightSymmetric": True,
        "QuantizeBias": True,
    },
)

vai_q_onnx.quantize_static(
    p1_float_model_path,
    p1_quant_model_path,
    DiffusionP1DataReader(),
    quant_format=vai_q_onnx.QuantFormat.QDQ,
    activation_type=vai_q_onnx.QuantType.QUInt8,
    calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
    weight_type=vai_q_onnx.QuantType.QInt8,
    op_types_to_quantize=["Conv", "Gemm", "MatMul", "Concat"],
    use_external_data_format=True,
    enable_dpu=True,
    extra_options={
        "ActivationSymmetric": True,
        "WeightSymmetric": True,
        "QuantizeBias": True,
    },
)

vai_q_onnx.quantize_static(
    p2_float_model_path,
    p2_quant_model_path,
    DiffusionP2DataReader(),
    quant_format=vai_q_onnx.QuantFormat.QDQ,
    activation_type=vai_q_onnx.QuantType.QUInt8,
    calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
    weight_type=vai_q_onnx.QuantType.QInt8,
    op_types_to_quantize=["Conv", "Gemm", "MatMul", "Concat"],
    use_external_data_format=True,
    enable_dpu=True,
    extra_options={
        "ActivationSymmetric": True,
        "WeightSymmetric": True,
        "QuantizeBias": True,
    },
)
