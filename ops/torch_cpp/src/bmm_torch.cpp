#include "../include/bmm_torch.hpp"
#include <fstream>
#include <iostream>
#include <tuple>
#include <vector>

aie::bmm_torch::bmm_torch(bool transpose) : transpose(transpose) {
  std::map<std::string, std::any> attr;
  mladf_v_str = "v0";
  std::string version = Utils::get_env_var("MLADF_VERSION", "v1");
  if (version == "v0" || version == "v1")
    mladf_v_str = version;
  attr["op_version"] = mladf_v_str;
  bmm_ = std::make_unique<ryzenai::bmm<uint16_t, uint16_t, uint16_t>>(
      "bfloat16", "bfloat16", "bfloat16", false, transpose, attr);
  bmm_->debug(false);
  int B;
  int M;
  int K;
  if (transpose == true) {
    B = 32;
    M = 2048;
    K = 128;
  } else {
    B = 32;
    M = 2048;
    K = 2048;
  }
  size_t Bs = static_cast<size_t>(B);
  size_t Ms = static_cast<size_t>(M);
  size_t Ks = static_cast<size_t>(K);
  std::vector<size_t> a_shape = {Bs, Ms, Ks};
  bmm_->set_params("BMM", a_shape);
}

aie::bmm_torch::~bmm_torch() {}

void aie::bmm_torch::run_bmm(uint16_t *aInput, uint16_t *bInput,
                             uint16_t *aie_out, int M, int K, int N, int B) {
  size_t Bs = static_cast<size_t>(B);
  size_t Ms = static_cast<size_t>(M);
  size_t Ks = static_cast<size_t>(K);
  size_t Ns = static_cast<size_t>(N);
  std::vector<size_t> a_shape = {Bs, Ms, Ks};
  std::vector<size_t> b_shape = {Bs, Ks, Ns};
  std::vector<size_t> aie_out_shape = {Bs, Ms, Ns};
  bmm_->set_execute_kernel_shape(a_shape);
  std::vector<Tensor> const_Tensor;
  const_Tensor = {{bInput, b_shape, "bfloat16"}};
  bmm_->initialize_const_params(const_Tensor);
  std::vector<Tensor> input_Tensor;
  input_Tensor = {{aInput, a_shape, "bfloat16"}};
  std::vector<Tensor> output_Tensor;
  output_Tensor = {{aie_out, aie_out_shape, "bfloat16"}};
  bmm_->execute(input_Tensor, output_Tensor);
}

torch::Tensor aie::bmm_torch::execute(torch::Tensor x, torch::Tensor y) {
  int B = x.sizes()[0];
  int M = x.sizes()[1];
  int K = x.sizes()[2];
  int N = y.sizes()[2];
  if (transpose == true) {
    N = y.sizes()[1];
  }
  auto z = torch::empty({B, M, N}).to(torch::kBFloat16);
  if (mladf_v_str == "v1") {
    if (transpose == false) // BMM2 now contains output transpose
      z = torch::empty({M, B, N}).to(torch::kBFloat16);
  }
  auto xCasted = static_cast<uint16_t *>(x.data_ptr());
  auto yCasted = static_cast<uint16_t *>(y.data_ptr());
  auto zCasted = static_cast<uint16_t *>(z.data_ptr());
  run_bmm(xCasted, yCasted, zCasted, M, K, N, B);
  return z;
}
