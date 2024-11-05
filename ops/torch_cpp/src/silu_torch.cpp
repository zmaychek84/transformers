#include "../include/silu_torch.hpp"

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

aie::silu_torch::silu_torch() {}
aie::silu_torch::~silu_torch() {}

void aie::silu_torch::run_silu(uint16_t *aInput, uint16_t *aieOut, int M, int N,
                               bool debug) {

  size_t Ms = static_cast<size_t>(M);
  size_t Ns = static_cast<size_t>(N);
  std::vector<size_t> a_shape = {Ms, Ns};
  std::vector<size_t> aie_out_shape = {Ms, Ns};
std:
  string dtype = "bfloat16";

  kernel.debug(debug);
  std::vector<Tensor> input_Tensor = {{aInput, a_shape, dtype}};
  std::vector<Tensor> output_Tensor = {{aieOut, aie_out_shape, dtype}};

  kernel.execute(input_Tensor, output_Tensor);
}

torch::Tensor aie::silu_torch::execute(const torch::Tensor &x) {
  if (x.dtype() != torch::kBFloat16) {
    throw std::runtime_error(
        "mladfsilu expects ONLY bfloat16 operands and results");
  }
  if ((x.dim() != 2)) {
    throw std::runtime_error(
        "mladfsilu expects ONLY rank 2 tensors [M,K] for the operand");
  }
  size_t M = x.sizes()[0];
  size_t K = x.sizes()[1];

  auto xCasted = static_cast<uint16_t *>(x.data_ptr());
  auto out = torch::empty({static_cast<int>(M), static_cast<int>(K)})
                 .to(torch::kBFloat16);
  auto outCasted = static_cast<uint16_t *>(out.data_ptr());
  if (std::string((Utils::get_env_var("DEVICE"))) == "stx") {
    run_silu(xCasted, outCasted, M, K, false);
  }

  return out;
}
