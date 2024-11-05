/* Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved. */

#include "opInterface.h"

namespace ryzenai::onnx_utils {

namespace DML_Ops {
DMLOps *DMLOps::m_instance = nullptr;

DMLOps &DMLOps::getInstance() {
#ifdef MAP_GPU_RESC_FOR_CPU
  if (m_instance == nullptr) {
    m_instance = new DMLOps(true);
  }
#else
  if (m_instance == nullptr) {
    m_instance = new DMLOps(false);
  }
#endif // MAP_GPU_RESC_FOR_CPU

  return *m_instance;
}

//=====================================================================================================
DMLOps::DMLOps(const bool &mapGPURescForCPU)
    : m_isGPURescMappedForCPU(mapGPURescForCPU) {
  ContextInfo m_ContextInfo = {
      1,     // The max number of trials in EvaluateOperator.
      0,     // The index of the first adapter to consider.
      true,  // If the D3D12 debug layer should be enabled.
      false, // If the context must use a D3D12 compute queue instead of a
             // direct queue. //changed to false based on discussion with
             // sandeep
      m_isGPURescMappedForCPU, // If all tensors only use persistently mapped
                               // system memory for CPU or NPU to use (no
                               // copies!).
      false,                   // Make the tensor always resident.
      !m_isGPURescMappedForCPU // If we get tensor data externally. This and
                               // systemMem should be opposite
  };

  m_Context = new Context(m_ContextInfo);

  // Print the adapter name to let the user know which GPU was selected.
  std::cout << L"ComputeMatMulGPU: DirectML is executing on: "
            << m_Context->AdapterName().c_str() << std::endl;
}
// create instance of DML_OPS singleton
//  hybrid_llm/matmulnbits.cpp: it instantiates the DML_OPS object and creates
//  the MatMulNbits and computes the operator

//=====================================================================================================
DMLOps::~DMLOps() { delete m_Context; }

//=====================================================================================================
const bool &DMLOps::isGPURescMappedForCPU() const {
  return m_isGPURescMappedForCPU;
}

//=====================================================================================================
void DMLOps::UpdateConstInfo(
    const std::string &opName,
    const std::vector<OnnxTensorInfo>
        &onnxTensorInput) { // surgery replace OnnxTensorInfo by Torch Tensor
                            // info
  TensorVector inputs = m_OpLists[opName]->GetInputTensorVector();
  for (size_t i = 0; i < inputs.size(); i++) {
    if (inputs[i] != nullptr) {
      if (onnxTensorInput[i].isExternalBufferConstant) {
        inputs[i]->SetConstFlag(onnxTensorInput[i].isExternalBufferConstant);
        inputs[i]->UpdateTensorBuffer(onnxTensorInput[i].pExternalBuffer);
      }
    }
  }
  m_Context->UploadInputOrConstData(inputs);
}

//=====================================================================================================
void DMLOps::GetMappedInputOutputTensors(
    std::string nodeName, std::vector<void *> &mappedInputTensors,
    std::vector<void *> &mappedOutputTensors) {
  mappedInputTensors = m_OpLists[nodeName]->GetMappedInputTensors();
  mappedOutputTensors = m_OpLists[nodeName]->GetMappedOutputTensors();
}

//=====================================================================================================
void DMLOps::CreateMatMulOperator(
    const std::string &opName,
    const std::vector<OnnxTensorInfo> &pTensorDescInput,
    const std::vector<OnnxTensorInfo> &pTensorDescOutput) {
  GemmParams params;
  params.m = pTensorDescInput[0].shape[1];
  params.k = pTensorDescInput[0].shape[2]; // surgery
  params.n = pTensorDescInput[1].shape[1]; // surgery
  params.batches = pTensorDescInput[0].shape[0];
  params.dataType = pTensorDescInput[0].dataType;

  // Create a operator from each parameter struct.
  m_OpLists[opName] = std::make_shared<GemmOperator>(*m_Context, 0, params);

  m_OpLists[opName]->InitializeTensors(m_Context);
  m_OpLists[opName]->InitializeAndBindOperator(m_Context);
  UpdateConstInfo(opName, pTensorDescInput);
}

//=====================================================================================================
void DMLOps::CreateMatMulNBitsOperator(
    const std::string &opName,
    const std::vector<OnnxTensorInfo>
        &pTensorDescInput, // surgery replace OnnxTensorInfo by LibTorch
    const std::vector<OnnxTensorInfo> &pTensorDescOutput,
    int64_t block_size, // surgery replace OnnxTensorInfo by LibTorch
    bool hasZeroPoint) {
  MatMulNBitsParams params;
  params.m = pTensorDescInput[0].shape[1];
  params.k = pTensorDescInput[0].shape[2];
  params.n = pTensorDescInput[1].shape[0];
  params.batches = pTensorDescInput[0].shape[0];

  params.dataType = pTensorDescInput[0].dataType;
  params.quantizedBlockB = block_size;
  params.hasZeroPoint = hasZeroPoint; // should come from onnx

  // Debugging Prints
  /*std::cout << "M " << params.m
            << std::endl;
  std::cout << "N " << params.n << std::endl;
  std::cout << "K " << params.k << std::endl;
  std::cout << "batches " << params.batches << std::endl;
  std::cout << "input dtype " ;
  std::wcout << params.dataType.c_str() ;
  std::cout << std::endl;
  std::cout << "block/grp size " << params.quantizedBlockB << std::endl;*/

  // Create a operator from each parameter struct.

  m_OpLists[opName] =
      std::make_shared<MatMulNBitsOperator>(*m_Context, 0, params);

  m_OpLists[opName]->InitializeTensors(m_Context);
  m_OpLists[opName]->InitializeAndBindOperator(m_Context);
  UpdateConstInfo(opName, pTensorDescInput);
}

void DMLOps::Initialize() {
  // Todo: Any init code should go hear.
}

// =====================================================================================================================
void DMLOps::ComputeMatMulNBitsGPU(
    const std::string &opName,
    std::vector<OnnxTensorInfo> &pBufInput, // surgery replace onnx by libtorch
    std::vector<OnnxTensorInfo>
        &pBufOutput) { // surgery replace onnx by libtorch

  std::shared_ptr<DmlOperator> pOperator = m_OpLists[opName];
  try {
    TensorVector inputs =
        pOperator->GetInputTensorVector(); // surgery check if you can replace
                                           // the tensor by a torch tensor
    TensorVector outputs = pOperator->GetOutputTensorVector();
    ;
    // Update the inputs/outputs with external data
    // This loop should only run for inputs which are not constant.
    // Constant inputs should already been uploaded during the custom op
    // kernel's c'tor
    for (size_t i = 0; i < inputs.size(); i++) {
      if (inputs[i] != nullptr)
        inputs[i]->UpdateTensorBuffer(
            pBufInput[i].pExternalBuffer); // extern buffer will have my data
                                           // (wts, input, scales, zeros)
    }
    for (size_t i = 0; i < outputs.size(); i++) {
      if (outputs[i] != nullptr)
        outputs[i]->UpdateTensorBuffer(pBufOutput[i].pExternalBuffer);
    }
    // Upload the inputs and outputs to the GPU. Bind our operator, inputs, and
    // outputs for testing. Finally, execute all trials using EvaluateOperator.

    m_Context->UploadInputOrConstData(inputs);
    m_Context->ExecuteOperator(pOperator);

    m_Context->DownloadData(outputs);

  } catch (const std::exception &exception) {
    std::cout << "Exception: " << exception.what() << std::endl;
  }
}

// =====================================================================================================================
void DMLOps::ComputeMatMulGPU(const std::string &opName,
                              std::vector<OnnxTensorInfo> &pBufInput,
                              std::vector<OnnxTensorInfo> &pBufOutput) {
  std::cout << "Start: Executing Matmul on GPU" << std::endl;
  std::shared_ptr<DmlOperator> pOperator = m_OpLists[opName];

  try {
    TensorVector inputs = pOperator->GetInputTensorVector();
    TensorVector outputs = pOperator->GetOutputTensorVector();
    ;
    // Update the inputs/outputs with external data
    // This loop should only run for inputs which are not constant.
    // Constant inputs should already been uploaded during the custom op
    // kernel's c'tor
    for (size_t i = 0; i < inputs.size(); i++) {
      if (inputs[i] != nullptr)
        inputs[i]->UpdateTensorBuffer(pBufInput[i].pExternalBuffer);
    }
    for (size_t i = 0; i < outputs.size(); i++) {
      if (outputs[i] != nullptr)
        outputs[i]->UpdateTensorBuffer(pBufOutput[i].pExternalBuffer);
    }
    // Upload the inputs and outputs to the GPU. Bind our operator, inputs, and
    // outputs for testing. Finally, execute all trials using EvaluateOperator.

    m_Context->UploadInputOrConstData(inputs);
    m_Context->ExecuteOperator(pOperator);

    m_Context->DownloadData(outputs);

  } catch (const std::exception &exception) {
    std::cout << "Exception: " << exception.what() << std::endl;
  }

  std::cout << "End: Executing Matmul on GPU" << std::endl;
}
} // namespace DML_Ops

} // namespace ryzenai::onnx_utils
