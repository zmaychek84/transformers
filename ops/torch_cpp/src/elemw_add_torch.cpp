#include "../include/elemw_add_torch.hpp"

#if __has_include(<ryzenai/dynamic_dispatch/ops/mladfadd/mladfadd.hpp>)
#include <ryzenai/dynamic_dispatch/ops/mladfadd/mladfadd.hpp>
#else
#include <ops/mladfadd/mladfadd.hpp>
#endif

#if __has_include(<ryzenai/dyanmic_dispatch/utils/utils.hpp>)
#include <ryzenai/dyanmic_dispatch/utils/utils.hpp>
#else
#include <utils/utils.hpp>
#endif

#include <fstream>
#include <iostream>
#include <tuple>
#include <vector>

aie::elemw_add_torch::elemw_add_torch() {
  std::string mladf_v_str = "v0";
  std::map<std::string, std::any> attr;
  std::string version = Utils::get_env_var("MLADF_VERSION", "v1");
  if (version == "v0" || version == "v1")
    mladf_v_str = version;
  attr["op_version"] = mladf_v_str;
  std::vector<std::vector<int>> shape_list = {{1, 4096},    {128, 4096},
                                              {256, 4096},  {512, 4096},
                                              {1024, 4096}, {2048, 4096}};
  attr["shapes"] = shape_list;
  kernel = std::make_unique<ryzenai::mladf_add<uint16_t, uint16_t, uint16_t>>(
      "bfloat16", true, attr);

  input_bos = kernel->get_inputs();
  output_bos = kernel->get_outputs();
  out_ddr = {output_bos[0].address(), 0, 0};
}
aie::elemw_add_torch::~elemw_add_torch() {}

torch::Tensor aie::elemw_add_torch::execute(const torch::Tensor &x,
                                            const torch::Tensor &y, int zerocpy,
                                            bool rettorch) {
  uint16_t *out = output_bos[0].map<uint16_t *>();
  size_t M = 0;
  size_t K = 0;
  if (zerocpy == 1) {

    M = x.sizes()[0];
    K = x.sizes()[1];

    std::vector<size_t> shape{M, K};
    kernel->set_kernel_shape(shape);

    auto xCasted = static_cast<uint16_t *>(x.data_ptr());

    uint16_t *a_bo_map = input_bos[0].map<uint16_t *>();
    memcpy((void *)a_bo_map, (void *)xCasted, M * K * sizeof(uint16_t));
    auto input_addr_y = static_cast<uint64_t *>(y.data_ptr());
    input_bos[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    std::vector<uint64_t> inputs = {input_bos[0].address(), input_addr_y[0]};
    std::vector<uint64_t> outputs = {output_bos[0].address()};
    kernel->execute(inputs, outputs, rettorch);
  } else if (zerocpy == 2) {
    auto input_addr = static_cast<uint64_t *>(x.data_ptr());
    M = input_addr[1];
    K = input_addr[2];
    auto input_addr_y = static_cast<uint64_t *>(y.data_ptr());

    std::vector<size_t> shape{M, K};
    kernel->set_kernel_shape(shape);

    // auto xCasted = static_cast<uint16_t *>(x.data_ptr());
    // auto yCasted = static_cast<uint16_t *>(y.data_ptr());

    // uint16_t *a_bo_map = input_bos[0].map<uint16_t *>();
    // memcpy((void *)a_bo_map, (void *)xCasted, M * K * sizeof(uint16_t));
    // uint16_t *b_bo_map = input_bos[1].map<uint16_t *>();
    // memcpy((void *)b_bo_map, (void *)yCasted, M * K * sizeof(uint16_t));

    // input_bos[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    // input_bos[1].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    std::vector<uint64_t> inputs = {input_addr[0], input_addr_y[0]};
    std::vector<uint64_t> outputs = {output_bos[0].address()};
    kernel->execute(inputs, outputs, rettorch);

  } else {
    if (x.dtype() != torch::kBFloat16 || x.dtype() != y.dtype()) {
      throw std::runtime_error(
          "MLADFADD expects ONLY bfloat16 operands and results");
    }
    if ((x.dim() != 2) || (y.dim() != 2)) {
      throw std::runtime_error(
          "MLADFADD expects ONLY rank 2 tensors [M,K] for operands");
    }
    if (x.sizes() != y.sizes()) {
      throw std::runtime_error("MLADFADD expects shame shaped operands");
    }

    M = x.sizes()[0];
    K = x.sizes()[1];

    std::vector<size_t> shape{M, K};
    kernel->set_kernel_shape(shape);

    auto xCasted = static_cast<uint16_t *>(x.data_ptr());
    auto yCasted = static_cast<uint16_t *>(y.data_ptr());

    uint16_t *a_bo_map = input_bos[0].map<uint16_t *>();
    memcpy((void *)a_bo_map, (void *)xCasted, M * K * sizeof(uint16_t));
    uint16_t *b_bo_map = input_bos[1].map<uint16_t *>();
    memcpy((void *)b_bo_map, (void *)yCasted, M * K * sizeof(uint16_t));

    input_bos[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    input_bos[1].sync(XCL_BO_SYNC_BO_TO_DEVICE);

    kernel->execute(input_bos, output_bos, rettorch);
  }
  torch::Tensor out_tensor;
  if (rettorch) {
    output_bos[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    out_tensor =
        torch::from_blob((void *)out, {(int)M, (int)K}, torch::kBFloat16);
  } else {
    out_ddr[0] = output_bos[0].address();
    out_ddr[1] = M;
    out_ddr[2] = K;
    out_tensor = torch::from_blob((void *)out_ddr.data(), {(int)1, (int)3},
                                  torch::kLong);
  }
  return out_tensor;
}
