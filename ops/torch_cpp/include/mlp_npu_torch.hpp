#ifndef __MLP_NPU_TORCH__
#define __MLP_NPU_TORCH__
#include <ATen/Functions.h>
#include <torch/extension.h>
#include <torch/torch.h>

#if __has_include(<ryzenai/dynamic_dispatch/ops/elwmul/elwmul.hpp>)
#include <ryzenai/dynamic_dispatch/ops/elwmul/elwmul.hpp>
#else
#include <ops/elwmul/elwmul.hpp>
#endif
#if __has_include(<ryzenai/dynamic_dispatch/ops/silu/silu.hpp>)
#include <ryzenai/dynamic_dispatch/ops/silu/silu.hpp>
#else
#include <ops/silu/silu.hpp>
#endif
#include "../include/elemw_add_torch.hpp"
#include "../include/gemm_torch.hpp"
#include "../include/rmsnorm_torch.hpp"

namespace aie {
class mlp_npu_torch {
  std::map<std::string, int> attr = {{"skip_create_input", 1}};

  std::unique_ptr<ryzenai::elw_mul<uint16_t, uint16_t, uint16_t>> mulKernel;
  std::unique_ptr<ryzenai::silu<uint16_t, uint16_t>> siluKernel;
  // std::unique_ptr<ryzenai::rms_norm<uint16_t, uint16_t, uint16_t>> rmsnorm;
  std::unique_ptr<ryzenai::mladf_add<uint16_t, uint16_t, uint16_t>> kernel;

  // gemm_torch gate_proj{false, "bfloat16", "uint4", "bfloat16"};
  // // TODO: also support uint4 here?
  // gemm_torch up_proj{false, "bfloat16", "uint4", "bfloat16"};
  // gemm_torch down_proj{false, "bfloat16", "uint4", "bfloat16"};
  std::unique_ptr<gemm_torch> gate_proj;
  std::unique_ptr<gemm_torch> up_proj;
  std::unique_ptr<gemm_torch> down_proj;

  int cnt;
  std::vector<size_t> gate_wts_shape_;
  std::vector<size_t> up_wts_shape_;
  std::vector<size_t> down_wts_shape_;
  int gs_down_;
  int gs_up_;
  int gs_gate_;
  std::string mladf_v_str_ = "v0";
  size_t N;
  size_t K;

public:
  torch::Tensor bmm_scale;
  mlp_npu_torch();
  ~mlp_npu_torch();
  void initialize_params(torch::Tensor qw_gate, torch::Tensor qz_gate,
                         torch::Tensor sc_gate, torch::Tensor b_gate,
                         int gs_gate, torch::Tensor qw_down,
                         torch::Tensor qz_down, torch::Tensor sc_down,
                         torch::Tensor b_down, int gs_down, torch::Tensor qw_up,
                         torch::Tensor qz_up, torch::Tensor sc_up,
                         torch::Tensor b_up, int gs_up);
  torch::Tensor execute(torch::Tensor x, std::vector<uint64_t> shape_addr,
                        size_t NPU_threshold_rm, size_t NPU_threshold_add);

private:
  std::vector<std::vector<int>> shape_list{
      {1, 4096, 11008, 128},    {1, 11008, 4096, 128},
      {128, 4096, 11008, 128},  {128, 11008, 4096, 128},
      {256, 4096, 11008, 128},  {256, 11008, 4096, 128},
      {512, 4096, 11008, 128},  {512, 11008, 4096, 128},
      {1024, 4096, 11008, 128}, {1024, 11008, 4096, 128},
      {2048, 4096, 11008, 128}, {2048, 11008, 4096, 128},
      {1, 4096, 14336, 128},    {1, 14336, 4096, 128},
      {128, 4096, 14336, 128},  {128, 14336, 4096, 128},
      {256, 4096, 14336, 128},  {256, 14336, 4096, 128},
      {512, 4096, 14336, 128},  {512, 14336, 4096, 128},
      {1024, 4096, 14336, 128}, {1024, 14336, 4096, 128},
      {2048, 4096, 14336, 128}, {2048, 14336, 4096, 128}};
};
} // namespace aie
#endif
