#include "../include/elemw_mul_torch.hpp"

#include <fstream>
#include <iostream>
#include <tuple>
#include <vector>

#if __has_include(<ryzenai/dyanmic_dispatch/utils/utils.hpp>)
#include <ryzenai/dyanmic_dispatch/utils/utils.hpp>
#else
#include <utils/utils.hpp>
#endif

// AIE Driver headers
#include "xaiengine.h"

// Headers to create Txn binary
#include "op_buf.hpp"
#include "op_types.h"

// dpu kernel metadata
#include "dpu_kernel_metadata.hpp"

aie::elemw_mul_torch::elemw_mul_torch() {}
aie::elemw_mul_torch::~elemw_mul_torch() {}

void aie::elemw_mul_torch::run_elem_mul(uint16_t *aInput, uint16_t *bInput,
                                        uint16_t *aieOut, int M, int N,
                                        bool debug) {

  size_t Ms = static_cast<size_t>(M);
  size_t Ns = static_cast<size_t>(N);

  std::vector<size_t> a_shape = {Ms, Ns};
  std::vector<size_t> b_shape = {Ms, Ns};
  std::vector<size_t> aie_out_shape = {Ms, Ns};
  std::string dtype = "blfoat16";

  std::vector<Tensor> input_Tensors = {{aInput, a_shape, dtype},
                                       {bInput, b_shape, dtype}};
  std::vector<Tensor> output_Tensor = {{aieOut, aie_out_shape, dtype}};

  kernel.execute(input_Tensors, output_Tensor);
}

torch::Tensor aie::elemw_mul_torch::execute(const torch::Tensor &x,
                                            const torch::Tensor &y) {
  if (x.dtype() != torch::kBFloat16 || x.dtype() != y.dtype()) {
    throw std::runtime_error(
        "MLADFADD expects ONLY bfloat16 operands and results");
  }
  if ((x.dim() != 2) || (y.dim() != 2)) {
    throw std::runtime_error(
        "mladfelwmul expects ONLY rank 2 tensors [M,K] for operands");
  }
  if (x.sizes() != y.sizes()) {
    throw std::runtime_error("mladfelwmul expects shame shaped operands");
  }
  size_t M = x.sizes()[0];
  size_t K = x.sizes()[1];

  auto xCasted = static_cast<uint16_t *>(x.data_ptr());
  auto yCasted = static_cast<uint16_t *>(y.data_ptr());
  auto out = torch::empty({static_cast<int>(M), static_cast<int>(K)})
                 .to(torch::kBFloat16);
  auto outCasted = static_cast<uint16_t *>(out.data_ptr());
  if (std::string((Utils::get_env_var("DEVICE"))) == "stx") {
    run_elem_mul(xCasted, yCasted, outCasted, M, K, false);
  }

  return out;
}
