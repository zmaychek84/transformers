#include "../include/rmsnorm_torch.hpp"
#if __has_include(<ryzenai/dyanmic_dispatch/utils/utils.hpp>)
#include <ryzenai/dyanmic_dispatch/utils/utils.hpp>
#else
#include <utils/utils.hpp>
#endif

aie::rmsnorm_torch::rmsnorm_torch() {
  std::string mladf_v_str = "v0";
  std::map<std::string, std::any> attr;
  std::string version = Utils::get_env_var("MLADF_VERSION", "v1");
  if (version == "v0" || version == "v1")
    mladf_v_str = version;
  attr["op_version"] = mladf_v_str;
  std::vector<std::vector<int>> shape_list = {
      {2048, 4096}, {1920, 4096}, {1792, 4096}, {1664, 4096}, {1536, 4096},
      {1408, 4096}, {1280, 4096}, {1152, 4096}, {1024, 4096}, {512, 4096},
      {384, 4096},  {256, 4096},  {128, 4096},  {1, 4096},    {3072, 4096}};
  attr["shapes"] = shape_list;

  kernel = std::make_unique<ryzenai::rms_norm<uint16_t, uint16_t, uint16_t>>(
      "bfloat16", true, attr);
  input_bos = kernel->get_inputs();
  output_bos = kernel->get_outputs();
  out_ddr = {input_bos[0].address(), output_bos[0].address(), 0, 0};
}
aie::rmsnorm_torch::~rmsnorm_torch() {}
std::vector<uint64_t> aie::rmsnorm_torch::get_address() {
  return {(uint64_t)input_bos[0].address(), (uint64_t)output_bos[0].address(),
          M, K};
}

// return : the address of  input output , and tensor size
torch::Tensor aie::rmsnorm_torch::execute(const torch::Tensor &x,
                                          const torch::Tensor &y, bool zerocpy,
                                          bool rettorch) {

  input_bo_addr = input_bos[0].address();

  auto yCasted = static_cast<uint16_t *>(y.data_ptr());
  uint16_t *b_bo_map = input_bos[1].map<uint16_t *>();
  uint16_t *out = output_bos[0].map<uint16_t *>();

  if (zerocpy) {
    auto input_addr = static_cast<uint64_t *>(x.data_ptr());
    M = input_addr[1];
    K = input_addr[2];

    std::vector<size_t> shape{M, K};
    kernel->set_kernel_shape(shape);
    memcpy((void *)b_bo_map, (void *)yCasted, K * sizeof(uint16_t));

    input_bos[1].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    std::vector<uint64_t> inputs = {input_addr[0], input_bos[1].address()};
    std::vector<uint64_t> outputs = {output_bos[0].address()};
    kernel->execute(inputs, outputs, rettorch);
    input_bo_addr = input_addr[0];
  } else {
    if (x.dtype() != torch::kBFloat16 || x.dtype() != y.dtype()) {
      throw std::runtime_error(
          "madlfRMSNorm expects ONLY bfloat16 operands and results");
    }
    if ((x.dim() != 2)) {
      throw std::runtime_error(
          "madlfRMSNorm expects ONLY rank 2 tensors [M,K] for activation");
    }

    if (y.dim() != 1) {
      throw std::runtime_error("madlfRMSNorm expects ONLY rank 1 tensors [K] "
                               "s for the weights of the RMSNorm");
    }
    M = x.sizes()[0];
    K = x.sizes()[1];
    std::vector<size_t> shape{M, K};
    kernel->set_kernel_shape(shape);
    auto xCasted = static_cast<uint16_t *>(x.data_ptr());
    uint16_t *a_bo_map = input_bos[0].map<uint16_t *>();
    memcpy((void *)a_bo_map, (void *)xCasted, M * K * sizeof(uint16_t));
    input_bos[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    memcpy((void *)b_bo_map, (void *)yCasted, K * sizeof(uint16_t));

    input_bos[1].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    kernel->execute(input_bos, output_bos, rettorch);
  }
  torch::Tensor out_tensor;
  if (rettorch) {
    output_bos[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    out_tensor =
        torch::from_blob((void *)out, {(int)M, (int)K}, torch::kBFloat16);
  } else {

    out_ddr[2] = M;
    out_ddr[3] = K;
    out_ddr[0] = input_bo_addr;
    out_ddr[1] = output_bos[0].address();
    out_tensor = torch::from_blob((void *)out_ddr.data(), {(int)1, (int)4},
                                  torch::kLong);
  }
  return out_tensor;
}
