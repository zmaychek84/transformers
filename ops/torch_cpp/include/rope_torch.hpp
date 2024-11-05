#ifndef __ROPE_TORCH__
#define __ROPE_TORCH__
#include <ATen/Functions.h>
#include <torch/extension.h>
#include <torch/torch.h>

#if __has_include(<ryzenai/dynamic_dispatch/ops/mladfmharope/mladfmharope.hpp>)
#include <ryzenai/dynamic_dispatch/ops/mladfmharope/mladfmharope.hpp>
#else
#include <ops/mladfmharope/mladfmharope.hpp>
#endif

namespace aie {
class rope_torch {
  std::unique_ptr<ryzenai::mha_rope<uint16_t, uint16_t, uint16_t>>
      mladfropeKernel;

  std::vector<xrt::bo> rope_inbos;
  std::vector<xrt::bo> rope_outbos;

public:
  rope_torch();
  ~rope_torch();
  torch::Tensor execute(const torch::Tensor &x, const torch::Tensor &y);
};
} // namespace aie
#endif
