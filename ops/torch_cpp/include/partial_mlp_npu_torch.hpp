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
class partial_mlp_npu_torch {
  std::map<std::string, int> attr = {{"skip_create_input", 1}};

  std::unique_ptr<ryzenai::elw_mul<uint16_t, uint16_t, uint16_t>> mulKernel;
  std::unique_ptr<ryzenai::silu<uint16_t, uint16_t>> siluKernel;
  std::unique_ptr<ryzenai::mladf_add<uint16_t, uint16_t, uint16_t>> kernel;
  std::unique_ptr<gemm_torch> down_proj;

  int cnt;

  std::vector<size_t> down_wts_shape_;

  int gs_down_;
  std::string mladf_v_str_ = "v0";
  size_t N_;
  size_t K_;
  std::array<uint64_t, 3> out_ddr;

public:
  torch::Tensor bmm_scale;
  partial_mlp_npu_torch();
  ~partial_mlp_npu_torch();
  void initialize_params(torch::Tensor qw_down, torch::Tensor qz_down,
                         torch::Tensor sc_down, torch::Tensor b_down,
                         int gs_down);
  torch::Tensor execute(torch::Tensor x, torch::Tensor y, bool rettorch);

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
      {2048, 4096, 14336, 128}, {2048, 14336, 4096, 128},
      {1, 4096, 13696, 128},    {1, 13696, 4096, 128},
      {128, 4096, 13696, 128},  {128, 13696, 4096, 128},
      {256, 4096, 13696, 128},  {256, 13696, 4096, 128},
      {512, 4096, 13696, 128},  {512, 13696, 4096, 128},
      {1024, 4096, 13696, 128}, {1024, 13696, 4096, 128},
      {2048, 4096, 13696, 128}, {2048, 13696, 4096, 128},
      {1, 3072, 8192, 128},     {1, 8192, 3072, 128},
      {128, 3072, 8192, 128},   {128, 8192, 3072, 128},
      {256, 3072, 8192, 128},   {256, 8192, 3072, 128},
      {512, 3072, 8192, 128},   {512, 8192, 3072, 128},
      {1024, 3072, 8192, 128},  {1024, 8192, 3072, 128},
      {2048, 3072, 8192, 128},  {2048, 8192, 3072, 128}};
};
} // namespace aie
