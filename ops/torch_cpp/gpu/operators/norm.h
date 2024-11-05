/* Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include "operator.h"

namespace ryzenai::onnx_utils {

// =====================================================================================================================
// Describes an abstract norm operation.
struct NormParams {
  int32 inputSize[4];     // Input batch, channels, height, and width.
  int32 scaleSize[4];     // Scale batch, channels, height, and width.
  int32 biasSize[4];      // Bias batch, channels, height, and width.
  bool axes[4];           // The axes along which to normalize (N, C, H, W)
  float epsilon;          // Epsilon value
  hstring activationFunc; // The activation function or None if there is no
                          // activation function.
  float activationParam1; // The function's first constant parameter.
  float activationParam2; // The function's second constant parameter.
  bool scale;             // The optional scale input tensor is present.
  bool bias;              // The optional bias input tensor is present.
  bool normalizeVariance; // Whether to normalize variance or not

  hstring dataType; // Which TensorProto DataType to use for our tensors (e.g.,
                    // Float).

  // Only used by shader-based operators.
  int32 instancesX; // How many shader workgroups to execute.
  int32 instancesY; // How many shader workgroups to execute.
  int32 instancesZ; // How many shader workgroups to execute.
};

// =====================================================================================================================
// This class abstracts a DirectML convolution operator.
class NormOperator : public DmlOperator {
public:
  explicit NormOperator(const Context &context, bool disableMetacmds,
                        const NormParams &params);
  virtual ~NormOperator() {}

private:
  void ValidateNormParams(const NormParams &params) const;

  const DML_TENSOR_DATA_TYPE m_dataType; // All tensors must use this data type.
  NormParams m_norm;
  ActivationFuncInfo m_activation;
};

} // namespace ryzenai::onnx_utils
