#include "../include/gemm_dml_torch.hpp"
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

aie::gemm_dml_torch::gemm_dml_torch(bool bb, int mm, int kk, int nn,
                                    std::string a_dtype, std::string b_dtype,
                                    std::string c_dtype) {

  m = mm;
  k = kk;
  n = nn;
  bval = bb;
  a_dtype_ = winrt::hstring(std::wstring(a_dtype.begin(), a_dtype.end()));
  b_dtype_ = winrt::hstring(std::wstring(b_dtype.begin(), b_dtype.end()));
  c_dtype_ = winrt::hstring(std::wstring(c_dtype.begin(), c_dtype.end()));

  op_name_ = "DMLMatMulNBits";
  // setting up DML
  dml_instance_ = &DML_Ops::DMLOps::getInstance();
}

aie::gemm_dml_torch::~gemm_dml_torch() {}

void aie::gemm_dml_torch::initialize_params(torch::Tensor qw, torch::Tensor qz,
                                            torch::Tensor sc, torch::Tensor b,
                                            int gs) {
  group_size = gs;

  qweight = qw.reshape({n, int(k / group_size), int(group_size / 2)})
                .contiguous(); // qw is N x K/2 packed weights as 2
                               // int4 in one int8. reshape qweight to Dim N x
                               // (K / grp_size) x (grp_size / 2)

  qzero = qz.contiguous(); // should be same shape  as qweights

  scales = sc.contiguous()
               .flatten(); // sc is N x (K/grp_size): scales should be flattened
                           // to be of dimension: (K . (N/grp_size))

  bias = b.contiguous(); // 1 x n

  scales_float = scales.to(torch::kFloat);
  bias_float = bias.to(torch::kFloat);

  std::vector<int64_t> input_shape{1, m, k};
  OnnxTensorInfo input_tensorInfo;
  input_tensorInfo.dataType = a_dtype_;
  input_tensorInfo.shape = input_shape;
  input_tensorInfo.pExternalBuffer = nullptr;
  input_tensorInfo.isExternalBufferConstant = false;
  tensor_inputs_.push_back(input_tensorInfo);

  OnnxTensorInfo qweight_tensorInfo;
  qweight_tensorInfo.dataType = b_dtype_;
  qweight_tensorInfo.shape = qweight.sizes().vec();
  qweight_tensorInfo.pExternalBuffer = qweight.data_ptr();
  qweight_tensorInfo.isExternalBufferConstant = true;
  tensor_inputs_.push_back(qweight_tensorInfo);

  OnnxTensorInfo qscales_tensorInfo;
  auto scale_type = L"Float";
  qscales_tensorInfo.dataType = scale_type;
  qscales_tensorInfo.shape = scales.sizes().vec();
  qscales_tensorInfo.pExternalBuffer = scales_float.data_ptr();
  qscales_tensorInfo.isExternalBufferConstant = true;
  tensor_inputs_.push_back(qscales_tensorInfo);

  OnnxTensorInfo qzero_tensorInfo;
  auto ztype = L"Uint8";
  qzero_tensorInfo.dataType = ztype;
  qzero_tensorInfo.shape = qzero.sizes().vec();
  qzero_tensorInfo.pExternalBuffer = qzero.data_ptr();
  qzero_tensorInfo.isExternalBufferConstant = true;
  tensor_inputs_.push_back(qzero_tensorInfo);

  if (bval) {

    OnnxTensorInfo qbias_tensorInfo;
    qbias_tensorInfo.dataType = c_dtype_;
    qbias_tensorInfo.shape = bias.sizes().vec();
    qbias_tensorInfo.pExternalBuffer = bias.data_ptr();
    qbias_tensorInfo.isExternalBufferConstant = true;
    tensor_inputs_.push_back(qbias_tensorInfo);
  }

  std::vector<int64_t> output_shape{1, m, n};
  OnnxTensorInfo out_tensorInfo;
  out_tensorInfo.dataType = c_dtype_;
  out_tensorInfo.shape = output_shape;
  out_tensorInfo.pExternalBuffer = nullptr;
  out_tensorInfo.isExternalBufferConstant = false;
  tensor_outputs_.push_back(out_tensorInfo);

  /*std::cout << "Creating MatMulNBitsOperator" << std::endl;*/

  dml_instance_->CreateMatMulNBitsOperator(op_name_, tensor_inputs_,
                                           tensor_outputs_, group_size, true);
}

torch::Tensor aie::gemm_dml_torch::execute(torch::Tensor x) {

  if (!x.is_contiguous()) {
    std::cout << "Warning: gemm_torch was provided a noncontiguous input "
                 "tensor, this will impact performance!"
              << std::endl;
    x = x.contiguous();
  }

  torch::Tensor y;

  auto x_shape = x.sizes();
  if (x_shape.size() < 3) {
    m = x.sizes()[0];
  } else {
    m = x.sizes()[1];
  }
  y = torch::zeros({1, m, n}, torch::dtype(torch::kFloat)) + 3.14;
  y = y.contiguous();

  tensor_inputs_[0].pExternalBuffer = (uint16_t *)x.data_ptr();

  tensor_outputs_[0].pExternalBuffer = const_cast<void *>(y.data_ptr());
  using namespace torch::indexing;

  //// Debug Prints
  // std::cout << "Input "
  //           << "shape " << x.sizes() << " ";
  //
  // std::cout << "scales "
  //           << "shape " << scales.sizes() << " ";
  //
  //
  // std::cout << "weights "
  //           << "shape " << qweight.sizes() << " ";

  dml_instance_->ComputeMatMulNBitsGPU(op_name_, tensor_inputs_,
                                       tensor_outputs_);

  return y;
}

torch::Tensor aie::gemm_dml_torch::execute_cpu(torch::Tensor x, int wts_index) {

  auto weights = unpack(qweight.reshape({n, int(k / 2)}).contiguous(), k);

  auto sc_ = torch::repeat_interleave(
      scales.reshape({n, int(k / group_size)}), group_size,
      1); // qweight N x K, need to reshape and unpack weights

  weights = weights * sc_;

  auto wts_shape = weights.sizes().vec();

  weights = weights.transpose(
      0, 1); // check weights dim, if (K,N) this statement wont be needed
  weights = weights.to(torch::kBFloat16);

  torch::Tensor out;
  out = torch::matmul(x, weights);
  return out;
}

torch::Tensor aie::gemm_dml_torch::unpack(torch::Tensor compact, int K) {

  using namespace torch::indexing;
  auto qw = torch::empty({compact.size(0), K}).to(torch::kInt8);

  auto refmsb = 0xF0;

  auto reflsb = 0x0F;

  auto masked_compact =
      torch::bitwise_and(compact.index({Slice(), Slice()}), refmsb);
  auto shifted_masked_compact =
      torch::bitwise_right_shift(masked_compact, 4).to(torch::kInt8);
  qw = qw.index_put_({Slice(), Slice(0, None, 2)}, shifted_masked_compact);

  shifted_masked_compact =
      torch::bitwise_and(compact.index({Slice(), Slice()}), reflsb)
          .to(torch::kInt8);

  qw = qw.index_put_({Slice(), Slice(1, None, 2)}, shifted_masked_compact);

  return qw;
}

torch::Tensor aie::gemm_dml_torch::pack(torch::Tensor weights) {
  int64_t qw_shape1 = ceil(weights.size(1) / 2);

  auto qcompact = torch::empty({weights.size(0), qw_shape1}).to(torch::kUInt8);
  auto j = 0;

  using namespace torch::indexing;
  for (int i = 0; i < weights.size(1); i++) {

    if (i % 2 == 0) {
      auto tmp = weights.index({Slice(), i});

      auto rmp = qcompact.index({Slice(), j});

      qcompact = qcompact.index_put_({Slice(), j}, weights.index({Slice(), i}));

      qcompact = qcompact.index_put_(
          {Slice(), j}, qcompact.index({Slice(), j}).bitwise_left_shift(4));

    } else {
      qcompact = qcompact.index_put_(
          {Slice(), j}, torch::bitwise_or(qcompact.index({Slice(), j}),
                                          weights.index({Slice(), i})));

      j++;
    }
  }
  return qcompact;
}
