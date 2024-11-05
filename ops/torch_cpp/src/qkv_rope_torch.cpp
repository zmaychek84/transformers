#include "../include/qkv_rope_torch.hpp"

aie::qkv_rope_torch::qkv_rope_torch() {
  std::map<std::string, std::any> attr;
  std::vector<std::vector<int>> rope_shape_list{
      {32, 4096, 128}, {32, 2048, 128}, {32, 1024, 128}, {32, 512, 128},
      {32, 256, 128},  {32, 128, 128},  {32, 1, 128}};
  attr["shapes"] = rope_shape_list;
  attr["transpose"] = std::string("input");

  mladfropeKernel =
      std::make_unique<ryzenai::mha_rope<uint16_t, uint16_t, uint16_t>>(
          "bfloat16", true, attr);
  rope_inbos = mladfropeKernel->get_inputs();
  rope_outbos = mladfropeKernel->get_outputs();

  std::vector<std::vector<int>> gemm_shape_list{
      {1, 4096, 4096, 128},   {128, 4096, 4096, 128},  {256, 4096, 4096, 128},
      {512, 4096, 4096, 128}, {1024, 4096, 4096, 128}, {2048, 4096, 4096, 128}};
  q_proj_ = std::make_unique<gemm_torch>(false, "bfloat16", "uint4", "bfloat16",
                                         gemm_shape_list);
  k_proj_ = std::make_unique<gemm_torch>(false, "bfloat16", "uint4", "bfloat16",
                                         gemm_shape_list);
  v_proj_ = std::make_unique<gemm_torch>(false, "bfloat16", "uint4", "bfloat16",
                                         gemm_shape_list);

  cnt = 0;
}
aie::qkv_rope_torch::~qkv_rope_torch() {}

void aie::qkv_rope_torch::initialize_params(
    torch::Tensor q_w, torch::Tensor q_z, torch::Tensor q_s, torch::Tensor q_b,
    int q_gs, torch::Tensor k_w, torch::Tensor k_z, torch::Tensor k_s,
    torch::Tensor k_b, int k_gs, torch::Tensor v_w, torch::Tensor v_z,
    torch::Tensor v_s, torch::Tensor v_b, int v_gs) {
  q_proj_->initialize_params(q_w, q_z, q_s, q_b, q_gs);
  k_proj_->initialize_params(k_w, k_z, k_s, k_b, k_gs, attr);
  v_proj_->initialize_params(v_w, v_z, v_s, v_b, v_gs, attr);

  q_gs_ = q_gs;
  k_gs_ = k_gs;
  v_gs_ = v_gs;
  q_N_ = q_w.sizes()[1];
  k_N_ = k_w.sizes()[1];
}

void aie::qkv_rope_torch::execute(const std::vector<uint64_t> &shared_address,
                                  const torch::Tensor &hidden, torch::Tensor &q,
                                  torch::Tensor &k, torch::Tensor &v,
                                  const torch::Tensor &y, int head_dim,
                                  int share_bo_threshold) {
  /* if (hidden.dtype() != torch::kBFloat16 || hidden.dtype() != y.dtype()) {
    throw std::runtime_error(
        "qkv_rope expects ONLY bfloat16 operands and results");
  } */
  if ((hidden.dim() != 2) || (y.dim() != 3)) {
    throw std::runtime_error(
        "qkv_rope expects ONLY rank 3 tensors [B,M,K] for operands");
  }
  if (y.sizes()[0] != 2) {
    throw std::runtime_error("qkv_rope expects ONLY rank 3 tensors [B=2,M,K] "
                             "for the sin and cos matrix");
  }

  size_t M = hidden.sizes()[0];
  size_t K = hidden.sizes()[1];
  size_t q_num_heads = q_N_ / head_dim;
  size_t k_num_heads = k_N_ / head_dim;

  if (share_bo_threshold > 0 && hidden.dtype() == torch::kLong) {
    auto input_addr = static_cast<uint64_t *>(hidden.data_ptr());
    M = input_addr[2];
    K = input_addr[3];
  }

  auto hCasted = static_cast<uint16_t *>(hidden.data_ptr());
  auto qCasted = static_cast<uint16_t *>(q.data_ptr());
  auto kCasted = static_cast<uint16_t *>(k.data_ptr());
  auto vCasted = static_cast<uint16_t *>(v.data_ptr());
  auto yCasted = static_cast<uint16_t *>(y.data_ptr());

  std::vector<size_t> qkv_in_shape = {M, K};
  std::vector<size_t> q_wts_shape = {K, q_N_};
  std::vector<size_t> kv_wts_shape = {K, k_N_};
  q_proj_->gemm_->set_shape(qkv_in_shape, q_wts_shape, q_gs_);
  k_proj_->gemm_->set_shape(qkv_in_shape, kv_wts_shape, k_gs_);
  v_proj_->gemm_->set_shape(qkv_in_shape, kv_wts_shape, v_gs_);

  auto inbos = q_proj_->gemm_->get_inputs(M);
  auto q_outbos = q_proj_->gemm_->get_outputs(M);
  auto k_outbos = k_proj_->gemm_->get_outputs(M);
  auto v_outbos = v_proj_->gemm_->get_outputs(M);
  auto q_wts_bos = q_proj_->gemm_->get_const();
  auto k_wts_bos = k_proj_->gemm_->get_const();
  auto v_wts_bos = v_proj_->gemm_->get_const();

  if (share_bo_threshold > 0 && M >= share_bo_threshold) {
    std::vector<uint64_t> q_in = {shared_address[1], q_wts_bos[cnt].address()};
    q_proj_->gemm_->execute(q_in, q_outbos, false);
    std::vector<uint64_t> k_in = {shared_address[1], k_wts_bos[cnt].address()};
    k_proj_->gemm_->execute(k_in, k_outbos, false);
    std::vector<uint64_t> v_in = {shared_address[1], v_wts_bos[cnt].address()};
    v_proj_->gemm_->execute(v_in, v_outbos);
    v_outbos[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  } else {
    uint16_t *in_map = inbos[0].map<uint16_t *>();
    memcpy((void *)in_map, (void *)hCasted, M * K * sizeof(uint16_t));
    inbos[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::vector<xrt::bo> q_in = {inbos[0], q_wts_bos[cnt]};
    q_proj_->gemm_->execute(q_in, q_outbos, false);
    std::vector<xrt::bo> k_in = {inbos[0], k_wts_bos[cnt]};
    k_proj_->gemm_->execute(k_in, k_outbos, false);
    std::vector<xrt::bo> v_in = {inbos[0], v_wts_bos[cnt]};
    v_proj_->gemm_->execute(v_in, v_outbos);
    v_outbos[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  }

  auto supported_shape =
      (M == 1 || M == 128 || M == 256 || M == 512 || M == 1024 || M == 2048);
  size_t new_M = M;
  if (!supported_shape) {
    if (M < 128) {
      new_M = 128;
    } else if (M < 256) {
      new_M = 256;
    } else if (M < 512) {
      new_M = 512;
    } else if (M < 1024) {
      new_M = 1024;
    } else if (M < 2048) {
      new_M = 2048;
    }
  }

  static size_t last_M = 1;
  static size_t last_B = 1;
  size_t q_B =
      (q_num_heads <= 8) ? 8 : ((q_num_heads <= 32) ? 32 : q_num_heads);
  size_t q_K = head_dim;
  size_t k_B =
      (k_num_heads <= 8) ? 8 : ((k_num_heads <= 32) ? 32 : k_num_heads);
  size_t k_K = head_dim;
  // std::cout<<"###++ rope last_M=="<<last_M<<", M=="<<M<<",
  // new_M=="<<new_M<<std::endl; std::cout<<"###++ rope
  // q_num_heads=="<<q_num_heads<<", q_B=="<<q_B<<std::endl; std::cout<<"###++
  // rope k_num_heads=="<<k_num_heads<<", k_B=="<<k_B<<std::endl;
  if (last_M != new_M || last_B != q_B) {
    std::vector<size_t> q_shape{q_B, new_M, q_K};
    mladfropeKernel->set_params("q_rope", q_shape);
    last_M = new_M;
    last_B = q_B;
  }

  uint16_t *b_bo_map = rope_inbos[1].map<uint16_t *>();
  uint16_t *rope_out = rope_outbos[0].map<uint16_t *>();

  if (supported_shape) {
    memcpy((void *)b_bo_map, (void *)yCasted, 2 * M * q_K * sizeof(uint16_t));
    rope_inbos[1].sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::vector<xrt::bo> q_rope_in = {q_outbos[0], rope_inbos[1]};
    mladfropeKernel->execute(q_rope_in, rope_outbos);
    rope_outbos[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    memcpy((void *)qCasted, (void *)rope_out,
           q_num_heads * M * q_K * sizeof(uint16_t));

    if (q_num_heads != k_num_heads) {
      std::vector<size_t> k_shape{k_B, M, k_K};
      mladfropeKernel->set_params("k_rope", k_shape);
      last_B = k_B;
    }

    std::vector<xrt::bo> k_rope_in = {k_outbos[0], rope_inbos[1]};
    mladfropeKernel->execute(k_rope_in, rope_outbos);
    rope_outbos[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    memcpy((void *)kCasted, (void *)rope_out,
           k_num_heads * M * k_K * sizeof(uint16_t));
  } else {
    memcpy((void *)(b_bo_map), (void *)yCasted, M * q_K * sizeof(uint16_t));
    memcpy((void *)(b_bo_map + new_M * q_K), (void *)(yCasted + M * q_K),
           M * q_K * sizeof(uint16_t));
    rope_inbos[1].sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::vector<xrt::bo> q_rope_in = {q_outbos[0], rope_inbos[1]};
    mladfropeKernel->execute(q_rope_in, rope_outbos);
    rope_outbos[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    for (auto b = 0; b < q_num_heads; b++)
      memcpy((void *)(qCasted + b * M * q_K),
             (void *)(rope_out + b * new_M * q_K), M * q_K * sizeof(uint16_t));

    if (q_num_heads != k_num_heads) {
      std::vector<size_t> k_shape{k_B, new_M, k_K};
      mladfropeKernel->set_params("k_rope", k_shape);
      last_B = k_B;
    }

    std::vector<xrt::bo> k_rope_in = {k_outbos[0], rope_inbos[1]};
    mladfropeKernel->execute(k_rope_in, rope_outbos);
    rope_outbos[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    for (auto b = 0; b < k_num_heads; b++)
      memcpy((void *)(kCasted + b * M * k_K),
             (void *)(rope_out + b * new_M * k_K), M * k_K * sizeof(uint16_t));
  }

  uint16_t *v_out = v_outbos[0].map<uint16_t *>();
  memcpy((void *)vCasted, (void *)v_out,
         k_num_heads * M * q_K * sizeof(uint16_t));

  if (q_wts_bos.size() == (cnt + 1))
    cnt = 0;
  else
    cnt++;

  return;
}
