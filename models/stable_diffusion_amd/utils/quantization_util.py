import numpy
from onnxruntime.quantization import CalibrationDataReader


class UnetDownblocksDataReader(CalibrationDataReader):
    def __init__(self, tensorsData):
        self.enum_data = []
        self.data_size = 5

        for i in range(1, self.data_size):
            self.enum_data.append(
                {
                    "sample_in": numpy.asarray(
                        tensorsData[f"{i}_downblocks_sample_in"], dtype=numpy.float32
                    ),
                    "emb": numpy.asarray(
                        tensorsData[f"{i}_downblocks_emb"], dtype=numpy.float32
                    ),
                    "encoder_hidden_states": numpy.asarray(
                        tensorsData[f"{i}_downblocks_down_encoder_hidden_states"],
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
    def __init__(self, tensorsData):
        self.enum_data = []
        self.data_size = 5

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


class UnetUpblocksDataReader(CalibrationDataReader):
    def __init__(self, tensorsData):
        self.enum_data = []
        self.data_size = 5

        for i in range(1, self.data_size):
            self.enum_data.append(
                {
                    "sample_in": numpy.asarray(
                        tensorsData[f"{i}_upblocks_sample_in"], dtype=numpy.float32
                    ),
                    "emb": numpy.asarray(
                        tensorsData[f"{i}_upblocks_emb"], dtype=numpy.float32
                    ),
                    "encoder_hidden_states": numpy.asarray(
                        tensorsData[f"{i}_upblocks_encoder_hidden_states"],
                        dtype=numpy.float32,
                    ),
                    "down_block_res_samples_input_1": numpy.asarray(
                        tensorsData[f"{i}_upblocks_down_block_res_samples_input_1"],
                        dtype=numpy.float32,
                    ),
                    "down_block_res_samples_input_2": numpy.asarray(
                        tensorsData[f"{i}_upblocks_down_block_res_samples_input_2"],
                        dtype=numpy.float32,
                    ),
                    "down_block_res_samples_input_3": numpy.asarray(
                        tensorsData[f"{i}_upblocks_down_block_res_samples_input_3"],
                        dtype=numpy.float32,
                    ),
                    "down_block_res_samples_input_4": numpy.asarray(
                        tensorsData[f"{i}_upblocks_down_block_res_samples_input_4"],
                        dtype=numpy.float32,
                    ),
                    "down_block_res_samples_input_5": numpy.asarray(
                        tensorsData[f"{i}_upblocks_down_block_res_samples_input_5"],
                        dtype=numpy.float32,
                    ),
                    "down_block_res_samples_input_6": numpy.asarray(
                        tensorsData[f"{i}_upblocks_down_block_res_samples_input_6"],
                        dtype=numpy.float32,
                    ),
                    "down_block_res_samples_input_7": numpy.asarray(
                        tensorsData[f"{i}_upblocks_down_block_res_samples_input_7"],
                        dtype=numpy.float32,
                    ),
                    "down_block_res_samples_input_8": numpy.asarray(
                        tensorsData[f"{i}_upblocks_down_block_res_samples_input_8"],
                        dtype=numpy.float32,
                    ),
                    "down_block_res_samples_input_9": numpy.asarray(
                        tensorsData[f"{i}_upblocks_down_block_res_samples_input_9"],
                        dtype=numpy.float32,
                    ),
                    "down_block_res_samples_input_10": numpy.asarray(
                        tensorsData[f"{i}_upblocks_down_block_res_samples_input_10"],
                        dtype=numpy.float32,
                    ),
                    "down_block_res_samples_input_11": numpy.asarray(
                        tensorsData[f"{i}_upblocks_down_block_res_samples_input_11"],
                        dtype=numpy.float32,
                    ),
                    "down_block_res_samples_input_12": numpy.asarray(
                        tensorsData[f"{i}_upblocks_down_block_res_samples_input_12"],
                        dtype=numpy.float32,
                    ),
                }
            )
        self.enum_data = iter(self.enum_data)

    def get_next(self):
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None
