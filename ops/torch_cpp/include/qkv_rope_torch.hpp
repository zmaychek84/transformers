#ifndef __QKV_ROPE_TORCH__
#define __QKV_ROPE_TORCH__
#include <ATen/Functions.h>
#include <torch/extension.h>
#include <torch/torch.h>

#if __has_include(<ryzenai/dynamic_dispatch/ops/mladfmharope/mladfmharope.hpp>)
#include <ryzenai/dynamic_dispatch/ops/mladfmharope/mladfmharope.hpp>
#else
#include <ops/mladfmharope/mladfmharope.hpp>
#endif
#include "../include/gemm_torch.hpp"

namespace aie {
class qkv_rope_torch {
public:
  qkv_rope_torch();
  ~qkv_rope_torch();

  // torch::Tensor execute(const torch::Tensor &x, const torch::Tensor &y);
  void aie::qkv_rope_torch::execute(const std::vector<uint64_t> &shared_address,
                                    const torch::Tensor &hidden,
                                    torch::Tensor &q, torch::Tensor &k,
                                    torch::Tensor &v, const torch::Tensor &y,
                                    int head_dim, int share_bo_threshold);
  void aie::qkv_rope_torch::initialize_params(
      torch::Tensor q_w, torch::Tensor q_z, torch::Tensor q_s,
      torch::Tensor q_b, int q_gs, torch::Tensor k_w, torch::Tensor k_z,
      torch::Tensor k_s, torch::Tensor k_b, int k_gs, torch::Tensor v_w,
      torch::Tensor v_z, torch::Tensor v_s, torch::Tensor v_b, int v_gs);

private:
  std::unique_ptr<ryzenai::mha_rope<uint16_t, uint16_t, uint16_t>>
      mladfropeKernel;
  std::unique_ptr<gemm_torch> q_proj_;
  std::unique_ptr<gemm_torch> k_proj_;
  std::unique_ptr<gemm_torch> v_proj_;

  std::vector<xrt::bo> rope_inbos;
  std::vector<xrt::bo> rope_outbos;

  int q_gs_;
  int k_gs_;
  int v_gs_;
  size_t q_N_;
  size_t k_N_;
  int cnt;
  std::map<std::string, int> attr = {{"skip_create_input", 1}};
};
} // namespace aie
#endif
