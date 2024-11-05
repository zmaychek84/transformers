#include "../include/mlp_npu_torch.hpp"

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

aie::mlp_npu_torch::mlp_npu_torch() {
  std::map<std::string, std::any> attr_mul;
  attr_mul["skip_create_input"] = 1;

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
                                          // llama3-8b
                                          {1, 14336},
                                          {128, 14336},
                                          {256, 14336},
                                          {512, 14336},
                                          {1024, 14336},
                                          {2048, 14336}};
  attr_mul["shapes"] = shapes;

  mulKernel = std::make_unique<ryzenai::elw_mul<uint16_t, uint16_t, uint16_t>>(
      "bfloat16", true, attr_mul);
  siluKernel = std::make_unique<ryzenai::silu<uint16_t, uint16_t>>(
      "bfloat16", true, attr_mul);
  siluKernel->debug(false);
  mulKernel->debug(false);
  cnt = 0;
  gate_proj = std::make_unique<gemm_torch>(false, "bfloat16", "uint4",
                                           "bfloat16", shape_list);
  up_proj = std::make_unique<gemm_torch>(false, "bfloat16", "uint4", "bfloat16",
                                         shape_list);
  down_proj = std::make_unique<gemm_torch>(false, "bfloat16", "uint4",
                                           "bfloat16", shape_list);

  std::vector<std::vector<int>> shape_list = {{1, 4096},    {128, 4096},
                                              {256, 4096},  {512, 4096},
                                              {1024, 4096}, {2048, 4096}};
  std::map<std::string, std::any> attr;
  attr["shapes"] = shape_list;
  attr["op_version"] = mladf_v_str_;
  attr["skip_create_intput"] = 1;

  kernel = std::make_unique<ryzenai::mladf_add<uint16_t, uint16_t, uint16_t>>(
      "bfloat16", true, attr);
  std::vector<std::vector<int>> shape_list_rms = {
      {2048, 4096}, {1920, 4096}, {1792, 4096}, {1664, 4096}, {1536, 4096},
      {1408, 4096}, {1280, 4096}, {1152, 4096}, {1024, 4096}, {800, 4096},
      {768, 4096},  {640, 4096},  {512, 4096},  {384, 4096},  {256, 4096},
      {128, 4096},  {1, 4096}};
  /* std::map<std::string, std::any> attr_rms;
   attr_rms["shapes"] = shape_list_rms;
   attr_rms["op_version"] = mladf_v_str_;
   attr_rms["skip_create_output"] = 1;
   rmsnorm = std::make_unique<ryzenai::rms_norm<uint16_t, uint16_t, uint16_t>>(
       "bfloat16", true, attr_rms);*/
}

aie::mlp_npu_torch::~mlp_npu_torch() {}

void aie::mlp_npu_torch::initialize_params(
    torch::Tensor qw_gate, torch::Tensor qz_gate, torch::Tensor sc_gate,
    torch::Tensor b_gate, int gs_gate, torch::Tensor qw_down,
    torch::Tensor qz_down, torch::Tensor sc_down, torch::Tensor b_down,
    int gs_down, torch::Tensor qw_up, torch::Tensor qz_up, torch::Tensor sc_up,
    torch::Tensor b_up, int gs_up) {
  gate_proj->initialize_params(qw_gate, qz_gate, sc_gate, b_gate, gs_gate);
  down_proj->initialize_params(qw_down, qz_down, sc_down, b_down, gs_down,
                               attr);
  up_proj->initialize_params(qw_up, qz_up, sc_up, b_up, gs_up, attr);
  gs_gate_ = gs_gate;
  gs_up_ = gs_up;
  gs_down_ = gs_down;
  N = qw_gate.sizes()[1];
  K = qw_gate.sizes()[0];
}

// input format:  add_output_addr, rmsnorm_output_addr, tensor_shape[1],
// tensor_shape[2],
torch::Tensor aie::mlp_npu_torch::execute(torch::Tensor x,
                                          std::vector<uint64_t> shape_addr,
                                          size_t NPU_threshold_rm,
                                          size_t NPU_threshold_add) {

  auto gate_outputs = gate_proj->gemm_->get_outputs(shape_addr[2]);
  auto up_outputs = up_proj->gemm_->get_outputs(shape_addr[2]);
  auto down_outputs = down_proj->gemm_->get_outputs(shape_addr[2]);

  std::vector<size_t> shape = {static_cast<size_t>(shape_addr[2]),
                               static_cast<size_t>(shape_addr[3])};
  std::vector<size_t> down_shape = {static_cast<size_t>(shape_addr[2]), N};
  gate_wts_shape_ = {static_cast<size_t>(static_cast<size_t>(shape_addr[3])),
                     N};
  up_wts_shape_ = {static_cast<size_t>(static_cast<size_t>(shape_addr[3])), N};
  down_wts_shape_ = {
      N,
      static_cast<size_t>(static_cast<size_t>(shape_addr[3])),
  };

  gate_proj->gemm_->set_shape(shape, gate_wts_shape_, gs_gate_);
  up_proj->gemm_->set_shape(shape, up_wts_shape_, gs_up_);
  down_proj->gemm_->set_shape(down_shape, down_wts_shape_, gs_down_);

  auto gate_wts = gate_proj->gemm_->get_const();
  auto up_wts = up_proj->gemm_->get_const();
  auto down_wts = down_proj->gemm_->get_const();

  if (shape_addr[2] >= NPU_threshold_rm) { // share address from rmsnorm
    // rmsnorm->execute(inputs, rms_out);
    gate_outputs[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    std::vector<uint64_t> gate_in = {shape_addr[1], gate_wts[cnt].address()};
    gate_proj->gemm_->execute(gate_in, gate_outputs, false);
    std::vector<uint64_t> up_in = {shape_addr[1], up_wts[cnt].address()};
    up_proj->gemm_->execute(up_in, up_outputs, false);
  } else {
    auto gate_inputs = gate_proj->gemm_->get_inputs(x.sizes()[1]);

    uint16_t *in_map = gate_inputs[0].map<uint16_t *>();
    auto in_ptr = static_cast<uint16_t *>(x.data_ptr());
    memcpy((void *)in_map, (void *)in_ptr,
           x.sizes()[1] * x.sizes()[2] * sizeof(uint16_t));
    gate_inputs[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    std::vector<xrt::bo> gate_in = {gate_inputs[0], gate_wts[cnt]};
    gate_proj->gemm_->execute(gate_in, gate_outputs, false);
    std::vector<xrt::bo> up_in = {gate_inputs[0], up_wts[cnt]};
    up_proj->gemm_->execute(up_in, up_outputs, false);
  }

  auto outputs = siluKernel->get_outputs();

  std::vector<size_t> a_shape_silu = {static_cast<size_t>(shape_addr[2]), N};
  siluKernel->set_kernel_shape(a_shape_silu);
  siluKernel->execute(gate_outputs, outputs, false);
  mulKernel->set_kernel_shape(a_shape_silu);
  std::vector<xrt::bo> inputs_2 = {outputs[0], up_outputs[0]};
  auto outputs_mul = mulKernel->get_outputs();
  mulKernel->execute(inputs_2, outputs_mul, false);
  std::vector<xrt::bo> down_in = {outputs_mul[0], down_wts[cnt]};

  down_proj->gemm_->execute(down_in, down_outputs,
                            shape_addr[2] < NPU_threshold_add);
  uint16_t *c_bo_map;

  if (shape_addr[2] >= NPU_threshold_add) { // add on NPU
    kernel->set_kernel_shape(shape);
    auto output_bos = kernel->get_outputs();
    std::vector<uint64_t> input_bos = {shape_addr[0],
                                       down_outputs[0].address()};
    std::vector<uint64_t> output_bos_addr = {output_bos[0].address()};
    kernel->execute(input_bos, output_bos_addr);
    output_bos[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    c_bo_map = output_bos[0].map<uint16_t *>();
  } else {
    down_outputs[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    c_bo_map = down_outputs[0].map<uint16_t *>();
  }
  auto out = torch::from_blob((void *)c_bo_map, {1, (int)shape_addr[2], int(K)},
                              torch::kBFloat16);
  if (gate_wts.size() == (cnt + 1))
    cnt = 0;
  else
    cnt++;
  return out;
}
