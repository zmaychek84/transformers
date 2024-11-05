/* Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved. */
#include "operators/norm.h"

#include <sstream>

#include "context.h"

namespace ryzenai::onnx_utils {

// Define defaults and parameter layout metadata for ConvParams
static const NormParams NormDefaults = {
    {1, 1, 1, 1},                 // inputSize
    {1, 1, 1, 1},                 // scaleSize
    {1, 1, 1, 1},                 // biasSize
    {false, false, false, false}, // axes
    0.0,                          // epsilon
    L"None",                      // activationFunc
    0,                            // activationParam1
    0,                            // activationParam2
    true,                         // scale
    true,                         // bias
    true,                         // normalizeVariance
    L"Float",                     // dataType
};

// =====================================================================================================================
NormOperator::NormOperator(const Context &context,
                           bool disableMetacmds, // If metacommands should be
                                                 // disabled for this operator.
                           const NormParams &params)
    : m_norm(params), m_dataType(StringToDataType(params.dataType)) {
  if ((m_dataType != DML_TENSOR_DATA_TYPE_FLOAT32) &&
      (m_dataType != DML_TENSOR_DATA_TYPE_FLOAT16)) {
    std::wstringstream ss;
    ss << "Norm only supports Float and Float16, not "
       << params.dataType.c_str();
    throw std::runtime_error(winrt::to_string(ss.str()));
  }

  // Apply this activation function.
  m_activation.func = StringToActivationFunc(params.activationFunc);
  m_activation.param1 = params.activationParam1;
  m_activation.param2 = params.activationParam2;

  // Build the tensor descriptors. For DirectML's benefit, we must have one
  // pointer for every possible tensor and must use empty pointers as
  // placeholders for unused optional tensors.

  bool isBroadcasted = params.biasSize[0] == 1 && params.biasSize[2] == 1 &&
                       params.biasSize[3] == 1 && params.scaleSize[0] == 1 &&
                       params.scaleSize[2] == 1 && params.scaleSize[3] == 1;

  std::vector<int64> shapeX{m_norm.inputSize[0], m_norm.inputSize[1],
                            m_norm.inputSize[2], m_norm.inputSize[3]};
  std::vector<int64> shapeS{m_norm.scaleSize[0], m_norm.scaleSize[1],
                            m_norm.scaleSize[2], m_norm.scaleSize[3]};
  std::vector<int64> shapeB{m_norm.biasSize[0], m_norm.biasSize[1],
                            m_norm.biasSize[2], m_norm.biasSize[3]};
  std::vector<int64> shapeY{m_norm.inputSize[0], m_norm.inputSize[1],
                            m_norm.inputSize[2], m_norm.inputSize[3]};

  int32 strides[4] = {-1, -1, -1, -1};

  const std::vector<int64> stridesX(strides, strides + shapeX.size());
  const std::vector<int64> stridesY(strides, strides + shapeY.size());

  m_inputTensorDescVec.emplace_back(
      CreateTensorDesc(L"X", L"NCDHW", m_dataType, shapeX, stridesX));
  m_outputTensorDescVec.emplace_back(
      CreateTensorDesc(L"Y", L"NCDHW", m_dataType, shapeY, stridesY));

  if (isBroadcasted) {
    std::vector<int64> stridesS(shapeS.size(), -1);
    std::vector<int64> stridesB(shapeY.size(), -1);

    m_inputTensorDescVec.emplace_back(
        CreateTensorDesc(L"S", L"", m_dataType, shapeS, stridesS));
    m_inputTensorDescVec.emplace_back(
        CreateTensorDesc(L"B", L"", m_dataType, shapeB, stridesB));
  } else {
    std::vector<int64> stridesS(strides, strides + shapeS.size());
    std::vector<int64> stridesB(strides, strides + shapeB.size());

    m_inputTensorDescVec.emplace_back(
        CreateTensorDesc(L"S", L"NCDHW", m_dataType, shapeS, stridesS));
    m_inputTensorDescVec.emplace_back(
        CreateTensorDesc(L"B", L"NCDHW", m_dataType, shapeB, stridesB));
  }

  // Now convert them to the DirectML representation. This is a bit silly but it
  // keeps all of the DML stuff separate.
  DmlTensorDesc dmlTensorX = {};
  DmlTensorDesc dmlTensorS = {};
  DmlTensorDesc dmlTensorB = {};
  DmlTensorDesc dmlTensorY = {};

  ConvertTensorDesc(*m_inputTensorDescVec[0], &dmlTensorX);
  ConvertTensorDesc(*m_inputTensorDescVec[1], &dmlTensorS);
  ConvertTensorDesc(*m_inputTensorDescVec[2], &dmlTensorB);
  ConvertTensorDesc(*m_outputTensorDescVec[0], &dmlTensorY);

  // Build the main DirectML operator descriptor.
  DML_MEAN_VARIANCE_NORMALIZATION_OPERATOR_DESC dmlDesc = {};
  dmlDesc.InputTensor = &dmlTensorX.desc;
  dmlDesc.ScaleTensor = &dmlTensorS.desc;
  dmlDesc.BiasTensor = &dmlTensorB.desc;
  dmlDesc.OutputTensor = &dmlTensorY.desc;
  dmlDesc.CrossChannel =
      !params.axes[0] && params.axes[1] && params.axes[2] && params.axes[3];
  dmlDesc.Epsilon = params.epsilon;
  dmlDesc.NormalizeVariance = params.normalizeVariance;

  // Add on the optional activation func.
  DmlActivationFuncDesc dmlFunc = {};

  if (m_activation.func != ActivationFunc::None) {
    ConvertActivationFuncInfo(m_activation, &dmlFunc);
    dmlDesc.FusedActivation = &dmlFunc.desc;
  }

  // Finally, compile the operator.
  DML_OPERATOR_DESC desc = {};
  desc.Type = DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION;
  desc.Desc = &dmlDesc;

  m_operator = context.CompileOperator(
      &desc, disableMetacmds, (m_dataType == DML_TENSOR_DATA_TYPE_FLOAT16));
}

} // namespace ryzenai::onnx_utils
