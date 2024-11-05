#include "../include/mha_npu_torch.hpp"

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

aie::mha_npu_torch::mha_npu_torch() {
  const float sc[] = {(float)HEAD_DIM};
  bmm_scale = torch::from_blob((void *)sc, {1, 1});
  bmm_scale = 1 / torch::sqrt(bmm_scale);
  bmm_scale = bmm_scale.to(torch::kBFloat16);
  std::string version = Utils::get_env_var("MLADF_VERSION", "v1");
  if (version == "v0" || version == "v1")
    mladf_v_str_ = version;
  std::map<std::string, std::any> attr = {{"op_version", mladf_v_str_}};
  const float nf[] = {-3.389e38f};
  neg_inf = torch::from_blob((void *)nf, {1, 1});

  bmm1 = std::make_unique<ryzenai::bmm<uint16_t, uint16_t, uint16_t>>(
      "bfloat16", "bfloat16", "bfloat16", false, true, attr);
  bmm1->debug(false);
  std::vector<size_t> bmm1_shape = {32, 3072, 128};
  bmm1->set_params("BMM1", bmm1_shape);

  std::vector<std::vector<int>> sfm_shape_list = {
      {32, 2048, 2048}, {32, 1024, 1024}, {32, 512, 512},
      {32, 256, 256},   {32, 128, 128},   {32, 3072, 3072}};
  std::map<std::string, std::any> sfm_attrs;
  sfm_attrs["shapes"] = sfm_shape_list;
  sfm_attrs["skip_create_input"] = 1;
  sfm_attrs["skip_create_output"] = 1;
  sfm_attrs["op_version"] = mladf_v_str_;

  softmax =
      std::make_unique<ryzenai::masked_softmax<uint16_t, uint16_t, uint16_t>>(
          "bfloat16", true, sfm_attrs);
  softmax->debug(false);

  bmm2 = std::make_unique<ryzenai::bmm<uint16_t, uint16_t, uint16_t>>(
      "bfloat16", "bfloat16", "bfloat16", false, false, attr);
  bmm2->debug(false);
  std::vector<size_t> bmm2_shape = {32, 3072, 3072};
  bmm2->set_params("BMM2", bmm2_shape);

  bmm1_inputs = bmm1->get_inputs();
  bmm1_outputs = bmm1->get_outputs();
  bmm2_inputs = bmm2->get_inputs();
  bmm2_outputs = bmm2->get_outputs();
  softmax_mask = softmax->get_inputs()[1];
  out_ddr = {bmm2_outputs[0].address(), 0, 0};
}

aie::mha_npu_torch::~mha_npu_torch() {}

torch::Tensor aie::mha_npu_torch::execute(torch::Tensor query_states,
                                          torch::Tensor key_states,
                                          torch::Tensor value_states,
                                          torch::Tensor attention_mask,
                                          bool rettorch) {
  int64_t exec_start = 0, exec_end = 0, time0 = 0, time1 = 0, time2 = 0,
          time3 = 0;

  int B = query_states.sizes()[0];
  int M = query_states.sizes()[1];
  int K = query_states.sizes()[2];
  int N = key_states.sizes()[1];

  uint16_t *xCasted;
  uint16_t *yCasted;
  uint16_t *mCasted;
  uint16_t *y2Casted;

  int inputM = M;
  // arbitrary shape support - begin
  if ((M != 256) && (M != 512) && (M != 1024) && (M != 2048) && (M != 3072)) {
    int newM = 0;
    if (M < 256) {
      newM = 256;
    } else if (M < 512) {
      newM = 512;
    } else if (M < 1024) {
      newM = 1024;
    } else if (M < 2048) {
      newM = 2048;
    } else {
      newM = 3072;
    }

    torch::Tensor q_bar = torch::zeros({B, newM, K}).to(torch::kBFloat16);
    torch::Tensor k_bar = torch::zeros({B, newM, K}).to(torch::kBFloat16);
    torch::Tensor v_bar = torch::zeros({B, newM, K}).to(torch::kBFloat16);
    torch::Tensor am_bar = torch::ones({1, newM, newM}) * -3.389e38;
    am_bar = am_bar.to(torch::kBFloat16);
    using namespace torch::indexing;

    q_bar.index_put_({Slice(), Slice(0, M), Slice()}, query_states);
    k_bar.index_put_({Slice(), Slice(0, M), Slice()}, key_states);
    v_bar.index_put_({Slice(), Slice(0, M), Slice()}, value_states);
    am_bar.index_put_({0, Slice(0, M), Slice(0, M)}, attention_mask);

    q_bar = q_bar.contiguous();
    k_bar = k_bar.contiguous();
    v_bar = v_bar.contiguous();
    am_bar = am_bar.contiguous();

    M = q_bar.sizes()[1];
    K = q_bar.sizes()[2];
    N = k_bar.sizes()[1];
    xCasted = static_cast<uint16_t *>(q_bar.data_ptr());
    yCasted = static_cast<uint16_t *>(k_bar.data_ptr());
    mCasted = static_cast<uint16_t *>(am_bar.data_ptr());
    y2Casted = static_cast<uint16_t *>(v_bar.data_ptr());
  } else {
    xCasted = static_cast<uint16_t *>(query_states.data_ptr());
    yCasted = static_cast<uint16_t *>(key_states.data_ptr());
    mCasted = static_cast<uint16_t *>(attention_mask.data_ptr());
    y2Casted = static_cast<uint16_t *>(value_states.data_ptr());
  }
  // arbitrary shape support - end

  std::vector<size_t> bmm1_shape{(size_t)B, (size_t)M, (size_t)K};
  std::vector<size_t> bmm1_wts_shape{(size_t)B, (size_t)K, (size_t)M};
  std::vector<size_t> softmax_shape{(size_t)B, (size_t)M, (size_t)N};
  std::vector<size_t> bmm2_shape{(size_t)B, (size_t)M, (size_t)N};
  std::vector<size_t> bmm2_wts_shape{(size_t)B, (size_t)N, (size_t)K};
  bmm1->set_execute_kernel_shape(bmm1_shape, bmm1_wts_shape);
  bmm2->set_execute_kernel_shape(bmm2_shape, bmm2_wts_shape);
  softmax->set_params("softmax", softmax_shape);
  exec_start = GET_ELAPSED_TIME_NS();

  // auto xCasted = static_cast<uint16_t *>(query_states.data_ptr());
  // auto yCasted = static_cast<uint16_t *>(key_states.data_ptr());
  // auto mCasted = static_cast<uint16_t *>(attention_mask.data_ptr());
  // auto y2Casted = static_cast<uint16_t *>(value_states.data_ptr());
  uint16_t *a_bo_map = bmm1_inputs[0].map<uint16_t *>();
  memcpy((void *)a_bo_map, (void *)xCasted, B * M * K * sizeof(uint16_t));
  uint16_t *b_bo_map = bmm1_inputs[1].map<uint16_t *>();
  memcpy((void *)b_bo_map, (void *)yCasted, B * K * N * sizeof(uint16_t));

  uint16_t *out = bmm2_outputs[0].map<uint16_t *>();

  bmm1_inputs[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bmm1_inputs[1].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bmm1_outputs[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);

  exec_end = GET_ELAPSED_TIME_NS();
  time0 = exec_end - exec_start;

  exec_start = GET_ELAPSED_TIME_NS();
  bmm1->execute(bmm1_inputs, bmm1_outputs, false);

  exec_end = GET_ELAPSED_TIME_NS();
  time1 = exec_end - exec_start;

  exec_start = GET_ELAPSED_TIME_NS();
  uint16_t *mask_bo_map = softmax_mask.map<uint16_t *>();
  memcpy((void *)mask_bo_map, (void *)mCasted, M * N * sizeof(uint16_t));
  softmax_mask.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bmm2_inputs[1].sync(XCL_BO_SYNC_BO_TO_DEVICE);

  std::vector<xrt::bo> inputs = {bmm1_outputs[0], softmax_mask};
  std::vector<xrt::bo> outputs = {bmm2_inputs[0]};
  softmax->execute(inputs, outputs, false);

  exec_end = GET_ELAPSED_TIME_NS();
  time2 = exec_end - exec_start;

  exec_start = GET_ELAPSED_TIME_NS();
  uint16_t *value_bo_map = bmm2_inputs[1].map<uint16_t *>();
  memcpy((void *)value_bo_map, (void *)y2Casted, B * N * K * sizeof(uint16_t));
  bmm2_inputs[1].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bmm2_outputs[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bmm2->execute(bmm2_inputs, bmm2_outputs, rettorch);

  torch::Tensor out_tensor;
  if (rettorch) {
    bmm2_outputs[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    out_tensor =
        torch::from_blob((void *)out, {inputM, B, K}, torch::kBFloat16);
  } else {
    out_ddr[1] = inputM;
    out_ddr[2] = B * K;
    out_tensor = torch::from_blob((void *)out_ddr.data(), {(int)1, (int)3},
                                  torch::kLong);
  }
  return out_tensor;
}
