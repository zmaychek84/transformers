#ifndef __GEMM_DML_TORCH__
#define __GEMM_DML_TORCH__
#include <ATen/Functions.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "../gpu/opInterface.h" //surgery

using namespace ryzenai::onnx_utils;

namespace aie {
class gemm_dml_torch {
  hstring a_dtype_;
  hstring b_dtype_;
  hstring c_dtype_;

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
  std::string op_name_;
  gemm_dml_torch(bool bb, int mm, int kk, int nn, std::string a_dtype,
                 std::string b_dtype, std::string c_dtype);
  ~gemm_dml_torch();
  void initialize_params(torch::Tensor qw, torch::Tensor qz, torch::Tensor sc,
                         torch::Tensor b, int gs);
  torch::Tensor execute(torch::Tensor x);

  torch::Tensor execute_cpu(torch::Tensor x, int wts_index);

  torch::Tensor pack(torch::Tensor x);
  torch::Tensor unpack(torch::Tensor compact, int K);

private:
  DML_Ops::DMLOps *dml_instance_;              // surgery add DMLOps to project
  std::vector<OnnxTensorInfo> tensor_inputs_;  // surgery change the ONNX
  std::vector<OnnxTensorInfo> tensor_outputs_; // surgery change the ONNX
};
} // namespace aie

#endif
