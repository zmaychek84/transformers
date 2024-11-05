#ifndef __ELEMWADD_TORCH_HEADER__
#define __ELEMWADD_TORCH_HEADER__

#include <ATen/Functions.h>
#include <torch/extension.h>
#include <torch/torch.h>

#if __has_include(<ryzenai/dynamic_dispatch/ops/mladfadd/mladfadd.hpp>)
#include <ryzenai/dynamic_dispatch/ops/mladfadd/mladfadd.hpp>
#else
#include <ops/mladfadd/mladfadd.hpp>
#endif

namespace aie {
class elemw_add_torch {
private:
  std::unique_ptr<ryzenai::mladf_add<uint16_t, uint16_t, uint16_t>> kernel;

  std::vector<xrt::bo> input_bos;
  std::vector<xrt::bo> output_bos;
  std::vector<uint64_t> out_ddr;

public:
  elemw_add_torch();
  ~elemw_add_torch();
  torch::Tensor execute(const torch::Tensor &x, const torch::Tensor &y,
                        int zerocpy, bool rettorch);
};
} // namespace aie

#endif
