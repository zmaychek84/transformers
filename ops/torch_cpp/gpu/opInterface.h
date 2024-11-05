/* Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved. */

#pragma once

#include <iostream>

#include "context.h"
#include "operator.h"
#include "operators/gemm.h"
#include "operators/matMulNBits.h"
#include "tensor.h"

namespace ryzenai::onnx_utils {

struct OnnxTensorInfo {
  std::vector<int64_t> shape;
  hstring dataType;
  void *pExternalBuffer;
  bool isExternalBufferConstant;
};

namespace DML_Ops {

/* This class is the interface class to be called from the Custom OP. It will
 * provide calling functions for all the supported operators*/
class DMLOps {
public:
  ~DMLOps();
  void ComputeMatMulGPU(const std::string &opName,
                        std::vector<OnnxTensorInfo> &,
                        std::vector<OnnxTensorInfo> &);

  void ComputeMatMulNBitsGPU(const std::string &opName,
                             std::vector<OnnxTensorInfo> &,
                             std::vector<OnnxTensorInfo> &);

  void Initialize();
  void CreateMatMulOperator(const std::string &opName,
                            const std::vector<OnnxTensorInfo> &,
                            const std::vector<OnnxTensorInfo> &);

  void CreateMatMulNBitsOperator(const std::string &opName,
                                 const std::vector<OnnxTensorInfo> &,
                                 const std::vector<OnnxTensorInfo> &,
                                 int64_t block_size = 32,
                                 bool hasZeroPoint = false);

  void UpdateConstInfo(const std::string &opName,
                       const std::vector<OnnxTensorInfo> &onnxTensorInput);

  void GetMappedInputOutputTensors(std::string nodeName,
                                   std::vector<void *> &mappedInputTensors,
                                   std::vector<void *> &mappedOutputTensors);
  static DMLOps &getInstance();

  DMLOps(const DMLOps &) = delete;
  DMLOps &operator=(const DMLOps &) = delete;

  /*static auto& instance() {
  static DMLOps dmpOp_;
  return dmpOp_;
  }*/
  const bool &isGPURescMappedForCPU() const;

private:
  explicit DMLOps(const bool &mapGPURescForCPU);
  // Static instance pointer
  static DMLOps *m_instance;
  Context *m_Context;
  bool m_isGPURescMappedForCPU;
  std::map<std::string, std::shared_ptr<DmlOperator>>
      m_OpLists; // List of all the operator objects
};
} // namespace DML_Ops

} // namespace ryzenai::onnx_utils
