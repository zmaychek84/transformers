/* Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved. */
#include "context.h"

#include <iomanip>
#include <sstream>

#include "operator.h"

namespace ryzenai::onnx_utils {

// =====================================================================================================================
Context::Context(const ContextInfo &info)
    : m_info(info), m_pagingValue(0), m_doneEvent(NULL), m_copyResourceSize(0),
      m_cmdListType(info.useCompute ? D3D12_COMMAND_LIST_TYPE_COMPUTE
                                    : D3D12_COMMAND_LIST_TYPE_DIRECT),
      /* m_descCount(0),
      m_temporaryResSize(0),
      m_persistentResSize(0),*/
      m_gpuTimerFreq(0) {
  CreateDxObjects();

  // Create a DirectML device.
  DML_CREATE_DEVICE_FLAGS flags = DML_CREATE_DEVICE_FLAG_NONE;

  if (m_info.debug) {
    flags |= DML_CREATE_DEVICE_FLAG_DEBUG;
  }

  if (FAILED(
          DMLCreateDevice(m_d3d12Device, flags, IID_PPV_ARGS(&m_dmlDevice)))) {
    throw std::runtime_error("Failed to create a DirectML device.");
  }

  if (FAILED(
          m_dmlDevice->CreateCommandRecorder(IID_PPV_ARGS(&m_dmlRecorder)))) {
    throw std::runtime_error("Failed to create a DirectML command recorder.");
  }
}

// =====================================================================================================================
Context::~Context() {
  if (m_doneEvent != nullptr) {
    CloseHandle(m_doneEvent);
  }
}

// =====================================================================================================================
// Create all D3D12 objects we need to do useful things.
void Context::CreateDxObjects() {
  if (m_info.debug) {
    CComPtr<ID3D12Debug> d3D12Debug;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&d3D12Debug)))) {
      d3D12Debug->EnableDebugLayer();
    }
  }

  // First get a DXGI factory and enumerate a D3D12 device.
  if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&m_dxgiFactory)))) {
    throw std::runtime_error("Failed to create a DXGI factory.");
  }

  CComPtr<IDXGIAdapter1> testAdapter;
  for (UINT idx = static_cast<UINT>(m_info.adapter);
       m_dxgiFactory->EnumAdapters1(idx, &testAdapter) != DXGI_ERROR_NOT_FOUND;
       ++idx) {
    // Make sure this is an IDXGIAdapter4 hardware renderer with D3D12 support.
    if (SUCCEEDED(testAdapter->QueryInterface(IID_PPV_ARGS(&m_dxgiAdapter)))) {
      DXGI_ADAPTER_DESC3 desc = {};
      if (SUCCEEDED(m_dxgiAdapter->GetDesc3(&desc))) {
        CComPtr<ID3D12Device> testDevice;
        if (((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0) &&
            SUCCEEDED(D3D12CreateDevice(m_dxgiAdapter, D3D_FEATURE_LEVEL_11_0,
                                        IID_PPV_ARGS(&testDevice)))) {
          // We require features in ID3D12Device3.
          if (SUCCEEDED(
                  testDevice->QueryInterface(IID_PPV_ARGS(&m_d3d12Device)))) {
            // Save a name for the user to query.
            m_adapterName = desc.Description;
            break;
          } else {
            testDevice.Release();
          }
        }

        if (!m_d3d12Device) {
          // Try again...
          m_dxgiAdapter.Release();
          testAdapter.Release();
        }
      }
    }
  }

  if (!m_d3d12Device) {
    throw std::runtime_error("Failed to create a D3D12 device.");
  }

  if (FAILED(m_d3d12Device->CreateFence(m_pagingValue, D3D12_FENCE_FLAG_NONE,
                                        IID_PPV_ARGS(&m_pagingFence)))) {
    throw std::runtime_error("Failed to create the paging fence.");
  }

  m_doneEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);

  if (m_doneEvent == nullptr) {
    throw std::runtime_error("Failed to create the done event.");
  }

  //
  // Create the copy state
  //

  D3D12_COMMAND_QUEUE_DESC queueDesc = {};

  if (m_info.systemMem == false) {
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_COPY;

    if (FAILED(m_d3d12Device->CreateCommandQueue(&queueDesc,
                                                 IID_PPV_ARGS(&m_copyQueue)))) {
      throw std::runtime_error("Failed to create a copy queue.");
    }

    if (FAILED(m_d3d12Device->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_COPY, IID_PPV_ARGS(&m_copyCmdAlloc)))) {
      throw std::runtime_error("Failed to create the copy command allocator.");
    }

    for (int32 idx = 0; idx < NumCopyState; ++idx) {
      // This command list is rebuilt when tensors are uploaded or downloaded.
      if (FAILED(m_d3d12Device->CreateCommandList(
              0, D3D12_COMMAND_LIST_TYPE_COPY, m_copyCmdAlloc, nullptr,
              IID_PPV_ARGS(&m_copyState[idx].cmdList)))) {
        throw std::runtime_error("Failed to create the copy command list.");
      }

      if (FAILED(m_copyState[idx].cmdList->Close())) {
        throw std::runtime_error("Failed to record the copy command list.");
      }

      // Set this to 1 initially to prevent a deadlock if we wait once before
      // first use.
      if (FAILED(m_d3d12Device->CreateFence(
              1, D3D12_FENCE_FLAG_NONE,
              IID_PPV_ARGS(&m_copyState[idx].fence)))) {
        throw std::runtime_error("Failed to create the copy fence.");
      }

      m_copyState[idx].pBuffer = nullptr;
      m_copyState[idx].needsWait = false;
      m_copyState[idx].pMemcpyDst = nullptr;
      m_copyState[idx].memcpySize = 0;
    }
  }

  //
  // Create the create-time execution state
  //

  queueDesc.Type = m_cmdListType;
  queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_DISABLE_GPU_TIMEOUT;

  if (FAILED(m_d3d12Device->CreateCommandQueue(&queueDesc,
                                               IID_PPV_ARGS(&m_evalQueue)))) {
    throw std::runtime_error("Failed to create a D3D12 queue.");
  }

  if (FAILED(m_d3d12Device->CreateCommandAllocator(
          m_cmdListType, IID_PPV_ARGS(&m_execCmdAlloc)))) {
    throw std::runtime_error("Failed to create the execute command allocator.");
  }

  // These command lists are empty until BindOperator is called.
  if (FAILED(m_d3d12Device->CreateCommandList(0, m_cmdListType, m_execCmdAlloc,
                                              nullptr,
                                              IID_PPV_ARGS(&m_execCmdList)))) {
    throw std::runtime_error("Failed to create the execute command list.");
  }

  if (FAILED(m_execCmdList->Close())) {
    throw std::runtime_error("Failed to record the execute command list.");
  }

  // Set this to 1 initially to prevent a deadlock if we wait once before first
  // use.
  if (FAILED(m_d3d12Device->CreateFence(1, D3D12_FENCE_FLAG_NONE,
                                        IID_PPV_ARGS(&m_evalFence)))) {
    throw std::runtime_error("Failed to create the done fence.");
  }

  //
  // Create the GPU timer state
  //

  D3D12_QUERY_HEAP_DESC queryHeapDesc = {};
  queryHeapDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
  queryHeapDesc.Count = 2 * m_info.maxTrials;

  if (FAILED(m_d3d12Device->CreateQueryHeap(&queryHeapDesc,
                                            IID_PPV_ARGS(&m_timerQueryHeap)))) {
    throw std::runtime_error("Failed to create the timer query heap.");
  }

  D3D12_HEAP_PROPERTIES heapProps = {};
  heapProps.Type = D3D12_HEAP_TYPE_READBACK;

  D3D12_RESOURCE_DESC resourceDesc = {};
  resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
  resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  resourceDesc.Width = 2 * m_info.maxTrials * sizeof(uint64);
  resourceDesc.Height = 1;
  resourceDesc.DepthOrArraySize = 1;
  resourceDesc.MipLevels = 1;
  resourceDesc.SampleDesc.Count = 1;

  if (FAILED(m_d3d12Device->CreateCommittedResource(
          &heapProps, D3D12_HEAP_FLAG_NONE, &resourceDesc,
          D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
          IID_PPV_ARGS(&m_timerBuffer)))) {
    throw std::runtime_error("Failed to create the timer read-back buffer.");
  }

  if (FAILED(m_evalQueue->GetTimestampFrequency(&m_gpuTimerFreq))) {
    throw std::runtime_error("Failed to query the GPU timestamp frequency.");
  }
}

// =====================================================================================================================
// A helper function which resets (0) and signals (1) the given fence on the
// given queue.
void Context::SignalBinaryFence(CComPtr<ID3D12CommandQueue> queue,
                                CComPtr<ID3D12Fence> fence) {
  if (FAILED(fence->Signal(0))) {
    throw std::runtime_error("Failed to set a fence to 0.");
  }

  if (FAILED(queue->Signal(fence, 1))) {
    throw std::runtime_error("Failed to queue a signal to a fence.");
  }
}

// =====================================================================================================================
// Blocks until the fence is equal to a certain value.
void Context::WaitForFence(CComPtr<ID3D12Fence> fence, UINT64 value) {
  // Check if the fence was signaled already. If not block until it is signaled.
  if (fence->GetCompletedValue() != value) {
    if (FAILED(fence->SetEventOnCompletion(value, m_doneEvent))) {
      throw std::runtime_error("Failed to associate an event with a fence.");
    }

    WaitForSingleObject(m_doneEvent, INFINITE);
  }

  HRESULT hr = m_d3d12Device->GetDeviceRemovedReason();

  if (hr != S_OK) {
    switch (hr) {
    case DXGI_ERROR_DEVICE_HUNG:
      throw std::runtime_error("D3D12 Device Removed: DXGI_ERROR_DEVICE_HUNG");
      break;
    case DXGI_ERROR_DEVICE_REMOVED:
      throw std::runtime_error(
          "D3D12 Device Removed: DXGI_ERROR_DEVICE_REMOVED");
      break;
    case DXGI_ERROR_DEVICE_RESET:
      throw std::runtime_error("D3D12 Device Removed: DXGI_ERROR_DEVICE_RESET");
      break;
    case DXGI_ERROR_DRIVER_INTERNAL_ERROR:
      throw std::runtime_error(
          "D3D12 Device Removed: DXGI_ERROR_DEVICE_INTERNAL_ERROR");
      break;
    case DXGI_ERROR_INVALID_CALL:
      throw std::runtime_error(
          "D3D12 Device Removed: DXGI_ERROR_DEVICE_INVALID_CALL");
      break;
    default: {
      std::stringstream ss;
      ss << "D3D12 Device Removed: 0x";
      ss << std::uppercase << std::setfill('0') << std::setw(8) << std::hex
         << hr;
      throw std::runtime_error(ss.str());
    } break;
    };
  }
}

// =====================================================================================================================
// Compiles a DirectML operator.
CComPtr<IDMLCompiledOperator> Context::CompileOperator(
    const DML_OPERATOR_DESC *pDesc,
    bool disableMetacmds, // If DirectML should disable metacommands.
    bool halfPrecision    // If DirectML should use half-precision (FP16)
                          // computation.
) const {
  CComPtr<IDMLOperator> dmlOperator;
  if (FAILED(m_dmlDevice->CreateOperator(pDesc, IID_PPV_ARGS(&dmlOperator)))) {
    throw std::runtime_error("Failed to create a DirectML operator.");
  }

  DML_EXECUTION_FLAGS flags = DML_EXECUTION_FLAG_NONE;

  if (disableMetacmds) {
    flags |= DML_EXECUTION_FLAG_DISABLE_META_COMMANDS;
  }

  if (halfPrecision) {
    // In practice this forces FP16 precision even if the tensors are FP32.
    flags |= DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION;
  }

  CComPtr<IDMLCompiledOperator> compiledOperator;
  if (FAILED(m_dmlDevice->CompileOperator(dmlOperator, flags,
                                          IID_PPV_ARGS(&compiledOperator)))) {
    throw std::runtime_error("Failed to compile a DirectML operator.");
  }

  return compiledOperator;
}

// =====================================================================================================================
// Enqueues an async make resident call. The context will wait for all paging
// requests in BindOperator.
void Context::MakeResident(CComPtr<ID3D12Heap> heap) {
  ID3D12Pageable *const pObject = heap;

  if (FAILED(m_d3d12Device->EnqueueMakeResident(D3D12_RESIDENCY_FLAG_NONE, 1,
                                                &pObject, m_pagingFence,
                                                ++m_pagingValue))) {
    throw std::runtime_error("Failed to make an allocation resident.");
  }
}

////
///=====================================================================================================================
//// Binds a compiled operator and its input and output tensors. This will
/// create / and initialize GPU state.
// void Context::BindOperator(const std::shared_ptr<Operator>& pTargetOp,
//                            const TensorVector& inputs,
//                            const TensorVector& outputs) {
//   // We will execute this operator later.
//   m_operator = pTargetOp->GetOperator();
//
//   // First, make sure any async make resident calls are done.
//   WaitForFence(m_pagingFence, m_pagingValue);
//
//   // Initialize the operator and create the necessary helper resources.
//   CComPtr<IDMLOperatorInitializer> opInitializer;
//   IDMLCompiledOperator* const pOperator = m_operator;
//
//   if (FAILED(m_dmlDevice->CreateOperatorInitializer(
//         1, &pOperator, IID_PPV_ARGS(&opInitializer)))) {
//     throw std::runtime_error(
//       "Failed to create a DirectML operator initializer.");
//   }
//
//   const DML_BINDING_PROPERTIES initializeProps =
//     opInitializer->GetBindingProperties();
//   const DML_BINDING_PROPERTIES executeProps =
//     m_operator->GetBindingProperties();
//
//   // The descriptor set and temporary resource is used for initialization and
//   // execution. The persistent resource is only needed for execution.
//   const UINT neededDescriptors = Max(initializeProps.RequiredDescriptorCount,
//                                      executeProps.RequiredDescriptorCount);
//   const UINT64 neededTempSize = Max(initializeProps.TemporaryResourceSize,
//                                     executeProps.TemporaryResourceSize);
//
//   D3D12_HEAP_PROPERTIES heapProps = {};
//   heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;
//
//   D3D12_RESOURCE_DESC resourceDesc = {};
//   resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
//   resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
//   resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
//   resourceDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
//   resourceDesc.Height = 1;
//   resourceDesc.DepthOrArraySize = 1;
//   resourceDesc.MipLevels = 1;
//   resourceDesc.SampleDesc.Count = 1;
//
//   if (neededTempSize > m_temporaryResSize) {
//     // Recreate the temporary resource.
//     m_temporaryRes.Release();
//     m_temporaryResSize = neededTempSize;
//     resourceDesc.Width = neededTempSize;
//
//     if (FAILED(m_d3d12Device->CreateCommittedResource(
//           &heapProps, D3D12_HEAP_FLAG_NONE, &resourceDesc,
//           D3D12_RESOURCE_STATE_COMMON, nullptr,
//           IID_PPV_ARGS(&m_temporaryRes)))) {
//       throw std::runtime_error(
//         "Failed to create the DirectML temporary buffer.");
//     }
//   }
//
//   if (executeProps.PersistentResourceSize > m_persistentResSize) {
//     // Recreate the persistent resource.
//     m_persistentRes.Release();
//     m_persistentResSize = executeProps.PersistentResourceSize;
//     resourceDesc.Width = executeProps.PersistentResourceSize;
//
//     if (FAILED(m_d3d12Device->CreateCommittedResource(
//           &heapProps, D3D12_HEAP_FLAG_NONE, &resourceDesc,
//           D3D12_RESOURCE_STATE_COMMON, nullptr,
//           IID_PPV_ARGS(&m_persistentRes)))) {
//       throw std::runtime_error(
//         "Failed to create the DirectML persistent buffer.");
//     }
//   }
//
//   if (neededDescriptors > m_descCount) {
//     // Recreate the descriptor heap.
//     m_descHeap.Release();
//     m_descCount = neededDescriptors;
//
//     D3D12_DESCRIPTOR_HEAP_DESC desc = {};
//     desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
//     desc.NumDescriptors = neededDescriptors;
//     desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
//
//     if (FAILED(m_d3d12Device->CreateDescriptorHeap(
//           &desc, IID_PPV_ARGS(&m_descHeap)))) {
//       throw std::runtime_error("Failed to create a descriptor heap.");
//     }
//   }
//
//   DML_BINDING_TABLE_DESC tableDesc = {};
//   tableDesc.Dispatchable = opInitializer;
//   tableDesc.CPUDescriptorHandle =
//     m_descHeap->GetCPUDescriptorHandleForHeapStart();
//   tableDesc.GPUDescriptorHandle =
//     m_descHeap->GetGPUDescriptorHandleForHeapStart();
//   tableDesc.SizeInDescriptors = m_descCount;
//
//   if (m_bindingTable == nullptr) {
//     // Create the binding table the first time we bind an operator.
//     if (FAILED(m_dmlDevice->CreateBindingTable(
//           &tableDesc, IID_PPV_ARGS(&m_bindingTable)))) {
//       throw std::runtime_error("Failed to create a DirectML binding table.");
//     }
//   } else {
//     if (FAILED(m_bindingTable->Reset(&tableDesc))) {
//       throw std::runtime_error("Failed to reset a DirectML binding table.");
//     }
//   }
//
//   // Set up the binding table for initialization, which computes the
//   persistent
//   // buffer contents.
//   if (initializeProps.TemporaryResourceSize > 0) {
//     DML_BINDING_DESC desc = {};
//     DML_BUFFER_BINDING binding = {};
//     desc.Type = DML_BINDING_TYPE_BUFFER;
//     desc.Desc = &binding;
//     binding.Buffer = m_temporaryRes;
//     binding.SizeInBytes = initializeProps.TemporaryResourceSize;
//
//     m_bindingTable->BindTemporaryResource(&desc);
//   }
//
//   if (executeProps.PersistentResourceSize > 0) {
//     DML_BINDING_DESC desc = {};
//     DML_BUFFER_BINDING binding = {};
//     desc.Type = DML_BINDING_TYPE_BUFFER;
//     desc.Desc = &binding;
//     binding.Buffer = m_persistentRes;
//     binding.SizeInBytes = executeProps.PersistentResourceSize;
//
//     m_bindingTable->BindOutputs(1, &desc);
//   }
//
//   // Build, submit, and wait on initialization.
//   if (FAILED(m_execCmdAlloc->Reset())) {
//     throw std::runtime_error(
//       "Failed to reset the execution command allocator.");
//   }
//
//   if (FAILED(m_execCmdList->Reset(m_execCmdAlloc, nullptr))) {
//     throw std::runtime_error(
//       "Failed to reset the execution command list for initialization.");
//   }
//
//   ID3D12DescriptorHeap* const pDescHeap = m_descHeap;
//   m_execCmdList->SetDescriptorHeaps(1, &pDescHeap);
//   m_dmlRecorder->RecordDispatch(m_execCmdList, opInitializer,
//   m_bindingTable);
//
//   if (FAILED(m_execCmdList->Close())) {
//     throw std::runtime_error(
//       "Failed to record the execution command list for initialization.");
//   }
//
//   ID3D12CommandList* pInitCmdLists[] = {m_execCmdList};
//   m_evalQueue->ExecuteCommandLists(
//     sizeof(pInitCmdLists) / sizeof(pInitCmdLists[0]), pInitCmdLists);
//
//   SignalBinaryFence(m_evalQueue, m_evalFence);
//   WaitForFence(m_evalFence, 1);
//
//   // Now reset the binding table and get it ready for execution.
//   tableDesc.Dispatchable = m_operator;
//
//   if (FAILED(m_bindingTable->Reset(&tableDesc))) {
//     throw std::runtime_error(
//       "Failed to reset a DirectML binding table for execution.");
//   }
//
//   if (executeProps.TemporaryResourceSize > 0) {
//     DML_BINDING_DESC desc = {};
//     DML_BUFFER_BINDING binding = {};
//     desc.Type = DML_BINDING_TYPE_BUFFER;
//     desc.Desc = &binding;
//     binding.Buffer = m_temporaryRes;
//     binding.SizeInBytes = executeProps.TemporaryResourceSize;
//
//     m_bindingTable->BindTemporaryResource(&desc);
//   }
//
//   if (executeProps.PersistentResourceSize > 0) {
//     DML_BINDING_DESC desc = {};
//     DML_BUFFER_BINDING binding = {};
//     desc.Type = DML_BINDING_TYPE_BUFFER;
//     desc.Desc = &binding;
//     binding.Buffer = m_persistentRes;
//     binding.SizeInBytes = executeProps.PersistentResourceSize;
//
//     m_bindingTable->BindPersistentResource(&desc);
//   }
//
//   pTargetOp->BindInputs(m_bindingTable, inputs);
//   pTargetOp->BindOutputs(m_bindingTable, outputs);
//
//   if (m_info.systemMem == false) {
//     // Reallocate the staging resources up to some max size. We will do
//     multiple
//     // copies for really big tensors.
//     UINT64 maxTensorSize = 0;
//
//     for (size_t idx = 0; idx < inputs.size(); ++idx) {
//       if (inputs[idx] != nullptr) {
//         maxTensorSize = Max(
//           maxTensorSize,
//           static_cast<UINT64>(inputs[idx]->Desc().totalBytes));
//       }
//     }
//
//     for (size_t idx = 0; idx < outputs.size(); ++idx) {
//       if (outputs[idx] != nullptr) {
//         maxTensorSize = Max(
//           maxTensorSize,
//           static_cast<UINT64>(outputs[idx]->Desc().totalBytes));
//       }
//     }
//
//     const UINT64 resSize = Min(maxTensorSize, MaxCopyResSize);
//
//     if (m_copyResourceSize < resSize) {
//       m_copyResourceSize = resSize;
//       resourceDesc.Width = resSize;
//
//       D3D12_HEAP_PROPERTIES stagingHeapProps = {};
//       stagingHeapProps.Type = D3D12_HEAP_TYPE_CUSTOM;
//       stagingHeapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
//       stagingHeapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
//
//       for (int32 idx = 0; idx < NumCopyState; ++idx) {
//         if (m_copyState[idx].pBuffer != nullptr) {
//           m_copyState[idx].resource->Unmap(0, nullptr);
//           m_copyState[idx].pBuffer = nullptr;
//         }
//
//         m_copyState[idx].resource.Release();
//
//         if (FAILED(m_d3d12Device->CreateCommittedResource(
//               &stagingHeapProps, D3D12_HEAP_FLAG_NONE, &resourceDesc,
//               D3D12_RESOURCE_STATE_COMMON, nullptr,
//               IID_PPV_ARGS(&m_copyState[idx].resource)))) {
//           throw std::runtime_error(
//             "Failed to create the copy staging resource.");
//         }
//
//         if (FAILED(m_copyState[idx].resource->Map(0, nullptr,
//                                                   &m_copyState[idx].pBuffer)))
//                                                   {
//           throw std::runtime_error("Failed to map the copy staging
//           resource.");
//         }
//       }
//     }
//   }
// }

////
///=====================================================================================================================
// void Context::UploadConstData(OpConstants &opConsts)
//{
//     int32 stateIdx = 0;
//     UploadData(&(opConsts.weights), stateIdx);
//     UploadData(&(opConsts.scale), stateIdx);
//
//     // We need to wait for all outstanding copies to finish before returning.
//     for (int32 idx = 0; idx < NumCopyState; ++idx) {
//       if (m_copyState[idx].needsWait) {
//         m_copyState[idx].needsWait = false;
//         WaitForFence(m_copyState[idx].fence, 1);
//       }
//     }
// }
//
////
///=====================================================================================================================
//// Upload tensor to GPU
// void Context::UploadData(Tensor* tensor, int32 &stateIdx)
//{
//     TensorCopyState *copyState = &m_copyState[stateIdx];
//
//     const UINT64 totalSize = static_cast<UINT64>(tensor->Desc().totalBytes);
//
//     for (UINT64 offset = 0; offset < totalSize; offset += m_copyResourceSize)
//     {
//       // Wait for prior accesses to the current copy state to be done.
//       if (copyState->needsWait) {
//         copyState->needsWait = false;
//         WaitForFence(copyState->fence, 1);
//       }
//
//       // This is a two-stage copy, repeating one or more times for each
//       // chunk of m_copyResourceSize bytes.
//       //   1. memcpy from the CPU-side buffer to the mapped staging
//       //   resource.
//       //   2. DMA the staging resource to the appropriate section of the GPU
//       //   resource.
//       const UINT64 copySize = Min(m_copyResourceSize, totalSize - offset);
//
//       memcpy(copyState->pBuffer, VoidPtrInc(tensor->Buffer(), offset),
//              static_cast<size_t>(copySize));
//
//       if (FAILED(
//             copyState->cmdList->Reset(m_copyCmdAlloc, nullptr))) {
//         throw std::runtime_error("Failed to reset the copy command list.");
//       }
//
//       copyState->cmdList->CopyBufferRegion(
//         tensor->Resource(), offset, copyState->resource, 0, copySize);
//
//       if (FAILED(copyState->cmdList->Close())) {
//         throw std::runtime_error("Failed to record the copy command list.");
//       }
//
//       ID3D12CommandList* pCmdLists[] = {copyState->cmdList};
//       m_copyQueue->ExecuteCommandLists(sizeof(pCmdLists) /
//       sizeof(pCmdLists[0]),
//                                        pCmdLists);
//
//       // Signal this copy's fence but don't wait on it yet.
//       SignalBinaryFence(m_copyQueue, copyState->fence);
//       copyState->needsWait = true;
//
//       // Advance to the next copy state.
//         if (++stateIdx >= NumCopyState) {
//         stateIdx = 0;
//         }
//     }
// }

// =====================================================================================================================
// Uploads tensors to the GPU. Must be called on inputs and outputs between
// BindOperator and Evaluate.
void Context::UploadInputOrConstData(const TensorVector &tensors) {
  if (m_info.systemMem == false) {
    if (FAILED(m_copyCmdAlloc->Reset())) {
      throw std::runtime_error("Failed to reset the copy command allocator.");
    }

    int32 stateIdx = 0;

    for (size_t idx = 0; idx < tensors.size(); ++idx) {
      if (tensors[idx] != nullptr) {
        if ((tensors[idx]->Buffer() == nullptr) ||
            (tensors[idx]->GetConstFlag() && tensors[idx]->IsUploaded())) {
          continue;
        }
        // if (tensors[idx]->Desc().isConst)
        const UINT64 totalSize =
            static_cast<UINT64>(tensors[idx]->Desc().totalBytes);

        for (UINT64 offset = 0; offset < totalSize;
             offset += m_copyResourceSize) {
          // Wait for prior accesses to the current copy state to be done.
          if (m_copyState[stateIdx].needsWait) {
            m_copyState[stateIdx].needsWait = false;
            WaitForFence(m_copyState[stateIdx].fence, 1);
          }

          // This is a two-stage copy, repeating one or more times for each
          // chunk of m_copyResourceSize bytes.
          //   1. memcpy from the CPU-side buffer to the mapped staging
          //   resource.
          //   2. DMA the staging resource to the appropriate section of the GPU
          //   resource.
          const UINT64 copySize = Min(m_copyResourceSize, totalSize - offset);

          memcpy(m_copyState[stateIdx].pBuffer,
                 VoidPtrInc(tensors[idx]->Buffer(), offset),
                 static_cast<size_t>(copySize));

          if (FAILED(m_copyState[stateIdx].cmdList->Reset(m_copyCmdAlloc,
                                                          nullptr))) {
            throw std::runtime_error("Failed to reset the copy command list.");
          }

          m_copyState[stateIdx].cmdList->CopyBufferRegion(
              tensors[idx]->Resource(), offset, m_copyState[stateIdx].resource,
              0, copySize);

          if (FAILED(m_copyState[stateIdx].cmdList->Close())) {
            throw std::runtime_error("Failed to record the copy command list.");
          }

          ID3D12CommandList *pCmdLists[] = {m_copyState[stateIdx].cmdList};
          m_copyQueue->ExecuteCommandLists(
              sizeof(pCmdLists) / sizeof(pCmdLists[0]), pCmdLists);

          // Signal this copy's fence but don't wait on it yet.
          SignalBinaryFence(m_copyQueue, m_copyState[stateIdx].fence);
          m_copyState[stateIdx].needsWait = true;

          // Advance to the next copy state.
          if (++stateIdx >= NumCopyState) {
            stateIdx = 0;
          }

          tensors[idx]->SetUploaded(true);
        }
        // UploadData(tensors[idx].get(), stateIdx);
      }
    }

    // We need to wait for all outstanding copies to finish before returning.
    for (int32 idx = 0; idx < NumCopyState; ++idx) {
      if (m_copyState[idx].needsWait) {
        m_copyState[idx].needsWait = false;
        WaitForFence(m_copyState[idx].fence, 1);
      }
    }
  }
}

// =====================================================================================================================
// Downloads tensors from the GPU. Must be called on outputs after Evaluate if
// validation is on.
void Context::DownloadData(const TensorVector &tensors) {
  if (m_info.systemMem == false) {
    if (FAILED(m_copyCmdAlloc->Reset())) {
      throw std::runtime_error("Failed to reset the copy command allocator.");
    }

    int32 stateIdx = 0;

    for (size_t idx = 0; idx < tensors.size(); ++idx) {
      if (tensors[idx] != nullptr) {
        const UINT64 totalSize =
            static_cast<UINT64>(tensors[idx]->Desc().totalBytes);

        for (UINT64 offset = 0; offset < totalSize;
             offset += m_copyResourceSize) {
          // Wait for prior accesses to the current copy state to be done and
          // execute their memcpy.
          if (m_copyState[stateIdx].needsWait) {
            m_copyState[stateIdx].needsWait = false;
            WaitForFence(m_copyState[stateIdx].fence, 1);

            memcpy(m_copyState[stateIdx].pMemcpyDst,
                   m_copyState[stateIdx].pBuffer,
                   m_copyState[stateIdx].memcpySize);
          }

          // This is a two-stage copy, repeating one or more times for each
          // chunk of m_copyResourceSize bytes.
          //   1. DMA a section of the GPU resource to the staging resource.
          //   2. memcpy from the mapped staging resource to the CPU-side
          //   buffer.
          const UINT64 copySize = Min(m_copyResourceSize, totalSize - offset);

          if (FAILED(m_copyState[stateIdx].cmdList->Reset(m_copyCmdAlloc,
                                                          nullptr))) {
            throw std::runtime_error("Failed to reset the copy command list.");
          }

          m_copyState[stateIdx].cmdList->CopyBufferRegion(
              m_copyState[stateIdx].resource, 0, tensors[idx]->Resource(),
              offset, copySize);

          if (FAILED(m_copyState[stateIdx].cmdList->Close())) {
            throw std::runtime_error("Failed to record the copy command list.");
          }

          ID3D12CommandList *pCmdLists[] = {m_copyState[stateIdx].cmdList};
          m_copyQueue->ExecuteCommandLists(
              sizeof(pCmdLists) / sizeof(pCmdLists[0]), pCmdLists);

          // Signal this copy's fence but don't wait on it yet.
          SignalBinaryFence(m_copyQueue, m_copyState[stateIdx].fence);
          m_copyState[stateIdx].needsWait = true;

          // We need to delay the memcpy until later, otherwise we can't
          // pipeline the GPU copy.
          m_copyState[stateIdx].pMemcpyDst =
              VoidPtrInc(tensors[idx]->Buffer(), offset);
          m_copyState[stateIdx].memcpySize = static_cast<size_t>(copySize);

          // Advance to the next copy state.
          if (++stateIdx >= NumCopyState) {
            stateIdx = 0;
          }
        }
      }
    }

    // We need to memcpy all data we DMAed but didn't get around to memcpying.
    // This also waits for idle.
    for (int32 idx = 0; idx < NumCopyState; ++idx) {
      if (m_copyState[idx].needsWait) {
        m_copyState[idx].needsWait = false;
        WaitForFence(m_copyState[idx].fence, 1);

        memcpy(m_copyState[idx].pMemcpyDst, m_copyState[idx].pBuffer,
               m_copyState[idx].memcpySize);
      }
    }
  }
}

// =====================================================================================================================
// Executes the bound operator one for each trial. Outputs will be written to
// the bound output tensors. If we should submit each trial in its own command
// list.
void Context::ExecuteOperator(const std::shared_ptr<DmlOperator> &pTargetOp) {
  // Rebuild the execute command list with the necessary commands.
  if (FAILED(m_execCmdAlloc->Reset())) {
    throw std::runtime_error(
        "Failed to reset the execution command allocator.");
  }

  if (FAILED(m_execCmdList->Reset(m_execCmdAlloc, nullptr))) {
    throw std::runtime_error("Failed to reset the execution command list.");
  }

  CComPtr<IDMLCompiledOperator> m_operator = pTargetOp->GetOperator();

  ID3D12DescriptorHeap *const pDescHeap = pTargetOp->m_descHeap;
  m_execCmdList->SetDescriptorHeaps(1, &pDescHeap);

  D3D12_RESOURCE_BARRIER barrier = {};
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;

  m_execCmdList->EndQuery(m_timerQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, 0);
  m_dmlRecorder->RecordDispatch(m_execCmdList, m_operator,
                                pTargetOp->m_bindingTable);
  m_execCmdList->EndQuery(m_timerQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, 1);
  m_execCmdList->ResourceBarrier(1, &barrier);

  m_execCmdList->ResolveQueryData(m_timerQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP,
                                  0, 2, m_timerBuffer, 0);

  if (FAILED(m_execCmdList->Close())) {
    throw std::runtime_error("Failed to record the execution command list.");
  }

  ID3D12CommandList *pCmdLists[] = {m_execCmdList};
  m_evalQueue->ExecuteCommandLists(sizeof(pCmdLists) / sizeof(pCmdLists[0]),
                                   pCmdLists);

  // We need to wait for the queue to be idle once after all trials.
  SignalBinaryFence(m_evalQueue, m_evalFence);
  WaitForFence(m_evalFence, 1);

  // Now map the readback buffer and compute the GPU execution time.
  void *pData = nullptr;

  if (SUCCEEDED(m_timerBuffer->Map(0, nullptr, &pData))) {
    const uint64 *const pTimes = static_cast<uint64 *>(pData);

    m_gpuTimes.resize(1);

    for (int32 trial = 0; trial < 1; ++trial) {
      const int32 firstTs = 2 * trial;
      m_gpuTimes[trial] = (pTimes[firstTs + 1] - pTimes[firstTs]) /
                          static_cast<float>(m_gpuTimerFreq);
    }

    m_timerBuffer->Unmap(0, nullptr);
  } else {
    throw std::runtime_error("Failed to map the timer read-back buffer.");
  }
}

} // namespace ryzenai::onnx_utils
