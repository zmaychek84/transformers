#ifndef __GEMM_TORCH__
#define __GEMM_TORCH__
#include <ATen/Functions.h>
#include <torch/extension.h>
#include <torch/torch.h>

#if __has_include(<ryzenai/dynamic_dispatch/ops/mladfmatmulbias/mladfmatmulbias.hpp>)
#include <ryzenai/dynamic_dispatch/ops/mladfmatmulbias/mladfmatmulbias.hpp>
#else
#include <ops/mladfmatmulbias/mladfmatmulbias.hpp>
#endif
namespace aie {
class gemm_torch {
  const std::string a_dtype_;
  const std::string b_dtype_;
  const std::string c_dtype_;

public:
  int m, k, n;
  bool bval;
  torch::Tensor qweight, qzero; // int8 container holding (u)int4 values
  torch::Tensor scales;         // bfloat16
  torch::Tensor scales_float;   // float32
  torch::Tensor bias;           // bfloat16
  torch::Tensor bias_float;     // float32
  int group_size;
  std::vector<size_t> wts_shape;
  gemm_torch(bool bb, std::string a_dtype, std::string b_dtype,
             std::string c_dtype,
             const std::vector<std::vector<int>> &shape_list);
  ~gemm_torch();
  void initialize_params(torch::Tensor qw, torch::Tensor qz, torch::Tensor sc,
                         torch::Tensor b, int gs,
                         const std::map<std::string, int> &attr = {});
  torch::Tensor execute(torch::Tensor x, int wts_index = 0);
  void execute_aie(torch::Tensor x, torch::Tensor y, int wts_index = 0);
  torch::Tensor execute_aie_bo(torch::Tensor x, int wts_index = 0,
                               bool zerocpy = false, bool rettensor = false);
  std::unique_ptr<ryzenai::mladfmatmulbias<int16_t, int8_t, int16_t>> gemm_;
  std::vector<std::pair<std::vector<size_t>, int>> wts_shapes_;
  std::vector<std::vector<int>> shape_list_;
  std::vector<uint64_t> out_ddr;

private:
};
} // namespace aie

#endif
