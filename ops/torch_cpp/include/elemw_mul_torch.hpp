#ifndef __ELEMMUL_TORCH_HEADER__
#define __ELEMMUL_TORCH_HEADER__
#include <ATen/Functions.h>
#include <torch/extension.h>
#include <torch/torch.h>

#if __has_include(<ryzenai/dynamic_dispatch/ops/elwmul/elwmul.hpp>)
#include <ryzenai/dynamic_dispatch/ops/elwmul/elwmul.hpp>
#else
#include <ops/elwmul/elwmul.hpp>
#endif
namespace aie {
class elemw_mul_torch {
  ryzenai::elw_mul<uint16_t, uint16_t, uint16_t> kernel{
      ryzenai::elw_mul<uint16_t, uint16_t, uint16_t>("bfloat16", true)};
  void run_elem_mul(uint16_t *aInput, uint16_t *bInput, uint16_t *aieOut, int M,
                    int N, bool debug = false);

public:
  std::string xclbinFileName;
  elemw_mul_torch();
  ~elemw_mul_torch();
  torch::Tensor execute(const torch::Tensor &x, const torch::Tensor &y);
};
} // namespace aie
#endif
