#include "../include/partial_mlp_npu_torch.hpp"

#if __has_include(<ryzenai/dyanmic_dispatch/utils/logging.hpp>)
#include <ryzenai/dyanmic_dispatch/utils/logging.hpp>
#else
#include <utils/logging.hpp>
#endif
#include <fstream>
#include <iostream>
#include <tuple>
#include <vector>

#ifdef RYZENAI_PERF
using namespace ryzenai;
#endif

aie::partial_mlp_npu_torch::partial_mlp_npu_torch() {
  std::map<std::string, std::any> attr_mul;
  // attr_mul["skip_create_input"] = 1;

  std::string version = Utils::get_env_var("MLADF_VERSION", "v1");
  if (version == "v0" || version == "v1")
    mladf_v_str_ = version;
  attr_mul["op_version"] = mladf_v_str_;

  std::vector<std::vector<int>> shapes = {{1, 11008},
                                          {128, 11008},
                                          {256, 11008},
                                          {512, 11008},
                                          {1024, 11008},
                                          {2048, 11008},
                                          {3072, 11008},
                                          // llama3-8b
                                          {1, 14336},
                                          {128, 14336},
                                          {256, 14336},
                                          {512, 14336},
                                          {1024, 14336},
                                          {2048, 14336},
                                          {3072, 14336},
                                          // chatglm3
                                          {1, 13696},
                                          {128, 13696},
                                          {256, 13696},
                                          {512, 13696},
                                          {1024, 13696},
                                          {2048, 13696},
                                          {3072, 13696},
                                          // phi-3.5
                                          {1, 8192},
                                          {128, 8192},
                                          {256, 8192},
                                          {512, 8192},
                                          {1024, 8192},
                                          {2048, 8192},
                                          {3072, 8192}};
  attr_mul["shapes"] = shapes;

  mulKernel = std::make_unique<ryzenai::elw_mul<uint16_t, uint16_t, uint16_t>>(
      "bfloat16", true, attr_mul);
  siluKernel = std::make_unique<ryzenai::silu<uint16_t, uint16_t>>(
      "bfloat16", true, attr_mul);
  siluKernel->debug(false);
  mulKernel->debug(false);
  cnt = 0;

  down_proj = std::make_unique<gemm_torch>(false, "bfloat16", "uint4",
                                           "bfloat16", shape_list);

  std::map<std::string, std::any> attr;
  attr["op_version"] = mladf_v_str_;
}

aie::partial_mlp_npu_torch::~partial_mlp_npu_torch() {}

void aie::partial_mlp_npu_torch::initialize_params(torch::Tensor qw_down,
                                                   torch::Tensor qz_down,
                                                   torch::Tensor sc_down,
                                                   torch::Tensor b_down,
                                                   int gs_down) {
  down_proj->initialize_params(qw_down, qz_down, sc_down, b_down, gs_down,
                               attr);
  gs_down_ = gs_down;
  K_ = qw_down.sizes()[0];
  N_ = qw_down.sizes()[1];
}

torch::Tensor aie::partial_mlp_npu_torch::execute(torch::Tensor x,
                                                  torch::Tensor y,
                                                  bool rettorch) {
  if (!x.is_contiguous()) {
    std::cout << "Warning: gemm_torch was provided a noncontiguous input "
                 "tensor, this will impact performance!"
              << std::endl;
    x = x.contiguous();
  }
  size_t B = static_cast<size_t>(x.sizes()[0]);
  size_t M = static_cast<size_t>(x.sizes()[1]);
  size_t K = static_cast<size_t>(x.sizes()[2]);

  auto down_outputs = down_proj->gemm_->get_outputs(M);
  std::vector<size_t> down_shape = {M, K};
  down_wts_shape_ = {K_, N_};
  down_proj->gemm_->set_shape(down_shape, down_wts_shape_, gs_down_);
  auto down_wts = down_proj->gemm_->get_const();

  auto silu_inputs = siluKernel->get_inputs();
  auto mul_inputs = mulKernel->get_inputs();

  uint16_t *in_map_silu = silu_inputs[0].map<uint16_t *>();
  auto in_ptr_silu = static_cast<uint16_t *>(x.data_ptr());

  memcpy((void *)in_map_silu, (void *)in_ptr_silu, M * K * sizeof(uint16_t));
  silu_inputs[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);

  uint16_t *in_map_mul = mul_inputs[1].map<uint16_t *>();
  auto in_ptr_mul = static_cast<uint16_t *>(y.data_ptr());
  memcpy((void *)in_map_mul, (void *)in_ptr_mul,
         y.sizes()[1] * y.sizes()[2] * sizeof(uint16_t));
  mul_inputs[1].sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto outputs = siluKernel->get_outputs();
  std::vector<size_t> a_shape_silu = {M, K};
  siluKernel->set_kernel_shape(a_shape_silu);
  siluKernel->execute(silu_inputs, outputs);

  mulKernel->set_kernel_shape(a_shape_silu);
  std::vector<xrt::bo> inputs_2 = {outputs[0], mul_inputs[1]};
  auto outputs_mul = mulKernel->get_outputs();
  mulKernel->execute(inputs_2, outputs_mul);

  std::vector<xrt::bo> down_in = {outputs_mul[0], down_wts[cnt]};
  down_proj->gemm_->execute(down_in, down_outputs);

  torch::Tensor out;
  if (rettorch) {
    down_outputs[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    uint16_t *c_bo_map;
    c_bo_map = down_outputs[0].map<uint16_t *>();
    out = torch::from_blob((void *)c_bo_map, {(int)B, (int)M, (int)N_},
                           torch::kBFloat16);
  } else {
    out_ddr[1] = M;
    out_ddr[2] = N_;
    out_ddr[0] = down_outputs[0].address();
    out = torch::from_blob((void *)out_ddr.data(), {(int)1, (int)3},
                           torch::kLong);
  }
  if (down_wts.size() == (cnt + 1))
    cnt = 0;
  else
    cnt++;
  return out;
}
