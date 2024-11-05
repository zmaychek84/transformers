#include "../include/gemm_torch.hpp"
#include <iostream>

namespace {
// Utility function to get the shape of a tensor as a vector of size_t
std::vector<size_t> getTensorShape(const torch::Tensor &tensor) {
  // Get the shape of the tensor as a vector of int64_t
  std::vector<int64_t> int64_shape = tensor.sizes().vec();

  // Create a vector of size_t to hold the shape
  std::vector<size_t> size_t_shape(int64_shape.size());

  // Transform the vector of int64_t to size_t
  if (tensor.dim() == 3) {

    std::transform(int64_shape.begin() + 1, int64_shape.end(),
                   size_t_shape.begin(),
                   [](int64_t val) { return static_cast<size_t>(val); });
  } else {

    std::transform(int64_shape.begin(), int64_shape.end(), size_t_shape.begin(),
                   [](int64_t val) { return static_cast<size_t>(val); });
  }

  return size_t_shape;
}
} // namespace

#define MAX_SEQ_LENGTH 2048

aie::gemm_torch::gemm_torch(bool bb, std::string a_dtype, std::string b_dtype,
                            std::string c_dtype,
                            const std::vector<std::vector<int>> &shape_list)
    : bval(bb), a_dtype_(a_dtype), b_dtype_(b_dtype), c_dtype_(c_dtype),
      shape_list_(shape_list) {}

aie::gemm_torch::~gemm_torch() {}

void aie::gemm_torch::initialize_params(
    torch::Tensor qw, torch::Tensor qz, torch::Tensor sc, torch::Tensor b,
    int gs, const std::map<std::string, int> &attr) {
  group_size = gs;

  qweight = qw.contiguous(); // k x n
  qzero = qz.contiguous();   // k/group_size x n
  scales = sc.contiguous();  // k/group_size x n
  bias = b.contiguous();     // 1 x n
  k = qweight.sizes()[0];
  n = qweight.sizes()[1];
  scales_float = scales.to(torch::kFloat);
  bias_float = bias.to(torch::kFloat);
  wts_shape = {static_cast<size_t>(k), static_cast<size_t>(n)};
  wts_shapes_.push_back(std::make_pair(wts_shape, gs));
  out_ddr = {0, 0, 0};

  // BE WARNED OF MASSIVE HACK
  // GROUP SIZE IS COMMUNICATED TO DD through scale shape

  // DD Tensors
  Tensor weight_tensor = {qweight.data_ptr<int8_t>(), getTensorShape(qweight),
                          b_dtype_};
  Tensor bias_tensor = {bias_float.data_ptr<float>(), getTensorShape(bias),
                        "float"};
  Tensor scales_tensor = {
      scales_float.data_ptr<float>(), {(size_t)gs, 0}, "float"};
  Tensor zeros_tensor = {qzero.data_ptr<int8_t>(), getTensorShape(qzero),
                         b_dtype_};
  std::vector<Tensor> constant_tensors = {weight_tensor, bias_tensor,
                                          scales_tensor, zeros_tensor};
  if (gemm_ == nullptr) {

    std::map<std::string, std::any> attrs;
    std::string mladf_v_str = "v0";
    std::string version = Utils::get_env_var("MLADF_VERSION", "v1");
    if (version == "v0" || version == "v1")
      mladf_v_str = version;
    // attrs["shapes"] = shape_list_;
    attrs["max_m"] = MAX_SEQ_LENGTH;
    attrs["op_version"] = mladf_v_str;
    attrs["group_size"] = gs;
    gemm_ =
        std::make_unique<ryzenai::mladfmatmulbias<int16_t, int8_t, int16_t>>(
            a_dtype_, b_dtype_, c_dtype_, true, attrs);
  }
  std::map<std::string, std::any> attr_tmp;
  for (auto iter : attr)
    attr_tmp.emplace(iter.first, iter.second);
  attr_tmp["group_size"] = gs;
  gemm_->initialize_const_params(constant_tensors, attr_tmp);
}

void aie::gemm_torch::execute_aie(torch::Tensor x, torch::Tensor y,
                                  int wts_index) {

  if (!x.is_contiguous()) {
    std::cout << "Warning: gemm_torch was provided a noncontiguous input "
                 "tensor, this will impact performance!"
              << std::endl;
    x = x.contiguous();
  }
  if (x.dim() == 3) {

    m = x.sizes()[1];
    k = x.sizes()[2];

  } else {

    m = x.sizes()[0];
    k = x.sizes()[1];
  }
  std::vector<size_t> a_shape = {static_cast<size_t>(m),
                                 static_cast<size_t>(k)};
  // DD Tensors
  Tensor input_tensor = {(int16_t *)x.data_ptr<torch::BFloat16>(),
                         getTensorShape(x), a_dtype_};
  Tensor output_tensor = {(int16_t *)y.data_ptr<torch::BFloat16>(),
                          getTensorShape(y), c_dtype_};

  std::vector<Tensor> input_tensors = {input_tensor};
  std::vector<Tensor> output_tensors = {output_tensor};
  gemm_->set_shape(a_shape, wts_shapes_[wts_index].first,
                   wts_shapes_[wts_index].second);
  gemm_->execute_internal(input_tensors, output_tensors, wts_index);
}

torch::Tensor aie::gemm_torch::execute_aie_bo(torch::Tensor x, int wts_index,
                                              bool zerocpy, bool rettensor) {
  int m = 0;
  int k = 0;
  uint64_t input_bo_addr = 0;
  if (zerocpy) {
    auto input_addr = static_cast<uint64_t *>(x.data_ptr());
    m = input_addr[1];
    k = input_addr[2];
    input_bo_addr = input_addr[0];
    std::vector<size_t> a_shape = {static_cast<size_t>(m),
                                   static_cast<size_t>(k)};
    gemm_->set_shape(a_shape, wts_shapes_[wts_index].first,
                     wts_shapes_[wts_index].second);
  } else {
    auto inputs = gemm_->get_inputs(x.sizes()[1]);
    m = x.sizes()[1];
    k = x.sizes()[2];

    uint16_t *in_map = inputs[0].map<uint16_t *>();
    auto in_ptr = static_cast<uint16_t *>(x.data_ptr());
    memcpy((void *)in_map, (void *)in_ptr,
           x.sizes()[1] * x.sizes()[2] * sizeof(uint16_t));
    input_bo_addr = inputs[0].address();
    inputs[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    std::vector<size_t> a_shape = {static_cast<size_t>(m),
                                   static_cast<size_t>(k)};
    gemm_->set_shape(a_shape, wts_shapes_[wts_index].first,
                     wts_shapes_[wts_index].second);
  }

  auto outputs = gemm_->get_outputs(m);
  outputs[0].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto wts = gemm_->get_const();

  std::vector<uint64_t> gate_in = {input_bo_addr, wts[wts_index].address()};
  gemm_->execute(gate_in, outputs);
  torch::Tensor out_tensor;
  if (rettensor) {
    auto c_bo_map = outputs[0].map<uint16_t *>();
    out_tensor = torch::from_blob((void *)c_bo_map,
                                  {1, m, (int)wts_shapes_[wts_index].first[1]},
                                  torch::kBFloat16);
  } else {
    out_ddr[0] = outputs[0].address();
    out_ddr[1] = m;
    out_ddr[2] = wts_shapes_[wts_index].first[1];
    out_tensor = torch::from_blob((void *)out_ddr.data(), {(int)1, (int)3},
                                  torch::kLong);
  }
  return out_tensor;
}
torch::Tensor aie::gemm_torch::execute(torch::Tensor x, int wts_index) {

  if (!x.is_contiguous()) {
    std::cout << "Warning: gemm_torch was provided a noncontiguous input "
                 "tensor, this will impact performance!"
              << std::endl;
    x = x.contiguous();
  }
  torch::Tensor y;
  if (x.dim() == 3) {

    m = x.sizes()[1];
    y = torch::empty({1, m, n}, torch::dtype(torch::kBFloat16));

  } else {

    m = x.sizes()[0];
    y = torch::empty({m, n}, torch::dtype(torch::kBFloat16));
  }
  std::vector<size_t> a_shape = {static_cast<size_t>(m),
                                 static_cast<size_t>(k)};
  // DD Tensors
  Tensor input_tensor = {(int16_t *)x.data_ptr<torch::BFloat16>(),
                         getTensorShape(x), a_dtype_};
  Tensor output_tensor = {(int16_t *)y.data_ptr<torch::BFloat16>(),
                          getTensorShape(y), c_dtype_};

  std::vector<Tensor> input_tensors = {input_tensor};
  std::vector<Tensor> output_tensors = {output_tensor};
  gemm_->set_shape(a_shape, wts_shape, group_size);
  gemm_->execute_internal(input_tensors, output_tensors, wts_index);

  return y;
}
