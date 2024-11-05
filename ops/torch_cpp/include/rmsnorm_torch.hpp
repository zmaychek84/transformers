#ifndef __RMSNORM_TORCH_HEADER__
#define __RMSNORM_TORCH_HEADER__

#include <ATen/Functions.h>
#include <torch/extension.h>
#include <torch/torch.h>

#if __has_include(<ryzenai/dynamic_dispatch/ops/mladfrmsnorm/mladfrmsnorm.hpp>)
#include <ryzenai/dynamic_dispatch/ops/mladfrmsnorm/mladfrmsnorm.hpp>
#else
#include <ops/mladfrmsnorm/mladfrmsnorm.hpp>
#endif

namespace aie {
class rmsnorm_torch {
private:
  std::unique_ptr<ryzenai::rms_norm<uint16_t, uint16_t, uint16_t>> kernel;

  std::vector<xrt::bo> input_bos;
  std::vector<xrt::bo> output_bos;
  uint64_t input_bo_addr;
  size_t M;
  size_t K;

  std::vector<uint64_t> out_ddr;

public:
  rmsnorm_torch();
  ~rmsnorm_torch();
  std::vector<uint64_t> get_address();
  torch::Tensor execute(const torch::Tensor &x, const torch::Tensor &y,
                        bool zerocpy, bool rettorch = true);
};
} // namespace aie

#endif
