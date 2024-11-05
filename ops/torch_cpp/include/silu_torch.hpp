#ifndef __SILU_TORCH_HEADER__
#define __SILU_TORCH_HEADER__

#include <ATen/Functions.h>
#include <torch/extension.h>
#include <torch/torch.h>

// XRT headers
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include <ryzenai/dynamic_dispatch/ops/silu/silu.hpp>

namespace aie {
class silu_torch {
  ryzenai::silu<uint16_t, uint16_t> kernel{
      ryzenai::silu<uint16_t, uint16_t>("bfloat16", true)};
  void run_silu(uint16_t *aInput, uint16_t *aieOut, int M, int N,
                bool debug = false);

public:
  std::string xclbinFileName;
  silu_torch();
  ~silu_torch();
  torch::Tensor execute(const torch::Tensor &x);
};
} // namespace aie

#endif
