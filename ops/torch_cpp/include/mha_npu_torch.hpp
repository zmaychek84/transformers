#ifndef __MHA_NPU_TORCH__
#define __MHA_NPU_TORCH__
#include <ATen/Functions.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "../include/bmm_torch.hpp"
#include "../include/softmax_torch.hpp"

#define HEAD_DIM 128

#if __has_include(<ryzenai/dynamic_dispatch/ops/bmm/bmm.hpp>)
#include <ryzenai/dynamic_dispatch/ops/bmm/bmm.hpp>
#include <ryzenai/dynamic_dispatch/ops/maskedsoftmax/maskedsoftmax.hpp>
#else
#include <ops/bmm/bmm.hpp>
#include <ops/maskedsoftmax/maskedsoftmax.hpp>
#endif

namespace aie {
class mha_npu_torch {
  std::unique_ptr<ryzenai::bmm<uint16_t, uint16_t, uint16_t>> bmm1;
  std::unique_ptr<ryzenai::masked_softmax<uint16_t, uint16_t, uint16_t>>
      softmax;
  std::unique_ptr<ryzenai::bmm<uint16_t, uint16_t, uint16_t>> bmm2;

  std::vector<xrt::bo> bmm1_inputs;
  std::vector<xrt::bo> bmm1_outputs;
  std::vector<xrt::bo> bmm2_inputs;
  std::vector<xrt::bo> bmm2_outputs;

  xrt::bo softmax_mask;
  std::string mladf_v_str_ = "v0";
  std::vector<uint64_t> out_ddr;

public:
  torch::Tensor bmm_scale;
  torch::Tensor neg_inf;

  mha_npu_torch();
  ~mha_npu_torch();

  torch::Tensor execute(torch::Tensor query_states, torch::Tensor key_states,
                        torch::Tensor value_states,
                        torch::Tensor attention_mask, bool rettorch = false);
};
} // namespace aie
#endif
