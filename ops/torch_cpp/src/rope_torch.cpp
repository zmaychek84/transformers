#include "../include/rope_torch.hpp"

aie::rope_torch::rope_torch() {
  std::string mladf_v_str = "v0";
  std::map<std::string, std::any> attr;
  std::string version = Utils::get_env_var("MLADF_VERSION", "v1");
  if (version == "v0" || version == "v1")
    mladf_v_str = version;
  attr["op_version"] = mladf_v_str;
  std::vector<std::vector<int>> shape_list = {
      {32, 4096, 128}, {32, 2048, 128}, {32, 1024, 128}, {32, 512, 128},
      {32, 256, 128},  {32, 128, 128},  {32, 1, 128}};
  attr["shapes"] = shape_list;
  std::string type = "input";
  attr["transpose"] = type;
  mladfropeKernel =
      std::make_unique<ryzenai::mha_rope<uint16_t, uint16_t, uint16_t>>(
          "bfloat16", true, attr);
  rope_inbos = mladfropeKernel->get_inputs();
  rope_outbos = mladfropeKernel->get_outputs();
}
aie::rope_torch::~rope_torch() {}

torch::Tensor aie::rope_torch::execute(const torch::Tensor &x,
                                       const torch::Tensor &y) {
  if (x.dtype() != torch::kBFloat16 || x.dtype() != y.dtype()) {
    throw std::runtime_error(
        "mladfrope expects ONLY bfloat16 operands and results");
  }
  if ((x.dim() != 3) || (y.dim() != 3)) {
    throw std::runtime_error(
        "mladfrope expects ONLY rank 3 tensors [B,M,K] for operands");
  }
  if (y.sizes()[0] != 2) {
    throw std::runtime_error("mladfrope expects ONLY rank 3 tensors [B=2,M,K] "
                             "sfor the sin and cos matrix");
  }

  size_t B = x.sizes()[0];
  size_t M = x.sizes()[1];
  size_t K = x.sizes()[2];

  size_t new_B = (B < 32) ? 32 : B;
  std::vector<size_t> shape{new_B, M, K};
  mladfropeKernel->set_params("rope", shape);

  auto xCasted = static_cast<uint16_t *>(x.data_ptr());
  auto yCasted = static_cast<uint16_t *>(y.data_ptr());

  uint16_t *a_bo_map = rope_inbos[0].map<uint16_t *>();
  // rope supports input transpose, so need padding here if B<32
  if (B == new_B) {
    memcpy((void *)a_bo_map, (void *)xCasted, B * M * K * sizeof(uint16_t));
  } else {
    for (auto m = 0; m < M; m++) {
      memcpy((void *)(a_bo_map + m * new_B * K), (void *)(xCasted + m * B * K),
             B * K * sizeof(uint16_t));
    }
  }

  uint16_t *b_bo_map = rope_inbos[1].map<uint16_t *>();
  memcpy((void *)b_bo_map, (void *)yCasted, 2 * M * K * sizeof(uint16_t));

  uint16_t *rope_out = rope_outbos[0].map<uint16_t *>();

  rope_inbos[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  rope_inbos[1].sync(XCL_BO_SYNC_BO_TO_DEVICE);

  mladfropeKernel->execute(rope_inbos, rope_outbos);
  rope_outbos[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  auto rope_t = torch::from_blob((void *)rope_out, {1, (int)B, (int)M, (int)K},
                                 torch::kBFloat16);

  return rope_t;
}
