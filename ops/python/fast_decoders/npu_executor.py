#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import os
import ryzenai_torch_cpp


class NpuExecutor:
    mha_npu = None
    rope_npu = None
    elewadd_npu = None
    partial_mlp_npu = None
    rmsnorm_npu = None

    @classmethod
    def init(cls):
        if cls.mha_npu is None:
            device = os.environ.get("DEVICE", "")
            mladf = os.environ.get("MLADF", "")

            if mladf == "2x4x4" and device == "stx":
                cls.mha_npu = ryzenai_torch_cpp.aie_mha_npu_torch()
                cls.rope_npu = ryzenai_torch_cpp.aie_rope_torch()
                cls.elewadd_npu = ryzenai_torch_cpp.aie_elemw_add_torch()
                cls.partial_mlp_npu = ryzenai_torch_cpp.aie_partial_mlp_npu_torch()
                cls.rmsnorm_npu = ryzenai_torch_cpp.aie_rmsnorm_torch()
            else:
                raise RuntimeError(
                    f"LlamaFastAttention needs os environment DEVICE=stx and MLADF=2x4x4"
                )
            # print("######--- NpuExecutor init done!")
