/* Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved. */
#include "tensor.h"

#include <iomanip>
#include <type_traits>

#include "context.h"

namespace ryzenai::onnx_utils {

// =====================================================================================================================
// Fills the given tensor with random data from the given generator. Float
// values are in the range [fmin, fmax].
template <DML_TENSOR_DATA_TYPE Type>
static void RandomizeImpl(Tensor *pTensor, std::mt19937 *pGenerator, float fmin,
                          float fmax, int64 maxRandCount) {
  using View = typename TensorViewLookup<Type>::View;
  using DataType = typename View::DataType;
  using StorageType = typename View::StorageType;

  // We want a lot of random values to get a good mix of initial data, but not
  // one for each value in the tensor because that can take many seconds to
  // generate for very large tensors (e.g., tensors with 1 billion values). 1K
  // values seems like a good balance, we go with the first prime number under
  // 1K to avoid pow-2 aligned patterns in our tensor data.
  int64 MaxRandomValues = 4099;
  if (maxRandCount != 0) {
    MaxRandomValues = maxRandCount;
  }
  int64 numValues = 1;

  for (const TensorDim &dim : pTensor->Desc().dims) {
    numValues *= dim.size;
  }

  numValues = Min(numValues, MaxRandomValues);

  std::vector<StorageType> values;
  values.reserve(static_cast<size_t>(numValues));

  if constexpr (std::is_same<DataType, bool>()) {
    // Fill the tensor with random coin flips.
    std::bernoulli_distribution distribution(0.5);

    for (int64 idx = 0; idx < numValues; ++idx) {
      values.push_back(static_cast<StorageType>(distribution(*pGenerator)));
    }
  } else if constexpr (std::is_floating_point<DataType>()) {
    std::uniform_real_distribution<DataType> distribution(fmin, fmax);

    for (int64 idx = 0; idx < numValues; ++idx) {
      values.push_back(static_cast<StorageType>(distribution(*pGenerator)));
    }
  } else if constexpr (std::is_signed<DataType>()) {
    // Fill the tensor with integers in the range [-128, 127].
    std::uniform_int_distribution<int32> distribution(-128, 127);

    for (int64 idx = 0; idx < numValues; ++idx) {
      values.push_back(static_cast<StorageType>(distribution(*pGenerator)));
    }
  } else if constexpr (std::is_unsigned<DataType>()) {
    // Fill the tensor with integers in the range [0, 255].
    std::uniform_int_distribution<uint32> distribution(0, 255);

    for (int64 idx = 0; idx < numValues; ++idx) {
      values.push_back(static_cast<StorageType>(distribution(*pGenerator)));
    }
  }

  // Scan through the values vector and repeatedly copy it into our tensor.
  pTensor->FillPattern(values.data(), numValues * sizeof(StorageType));
}

// =====================================================================================================================
// Computes the size of a data type in bits.
int32 Tensor::ElementSizeBits(DML_TENSOR_DATA_TYPE dataType) {
  int32 size = 0;

  switch (dataType) {
  case DML_TENSOR_DATA_TYPE_FLOAT32:
  case DML_TENSOR_DATA_TYPE_UINT32:
  case DML_TENSOR_DATA_TYPE_INT32:
    size = sizeof(uint32) * 8;
    break;

  case DML_TENSOR_DATA_TYPE_FLOAT16:
  case DML_TENSOR_DATA_TYPE_UINT16:
  case DML_TENSOR_DATA_TYPE_INT16:
    size = sizeof(uint16) * 8;
    break;

  case DML_TENSOR_DATA_TYPE_UINT8:
  case DML_TENSOR_DATA_TYPE_INT8:
    size = sizeof(uint8) * 8;
    break;
  case DML_TENSOR_DATA_TYPE_INT4:
  case DML_TENSOR_DATA_TYPE_UINT4:
    size = 4;
    break;

  default:
    throw std::runtime_error("Unknown tensor data type.");
    break;
  }

  return size;
}

// =====================================================================================================================
// Computes the size of a data type in bytes.
int32 Tensor::ElementSize(DML_TENSOR_DATA_TYPE dataType) {
  int32 size = 0;

  switch (dataType) {
  case DML_TENSOR_DATA_TYPE_FLOAT32:
  case DML_TENSOR_DATA_TYPE_UINT32:
  case DML_TENSOR_DATA_TYPE_INT32:
    size = sizeof(uint32);
    break;

  case DML_TENSOR_DATA_TYPE_FLOAT16:
  case DML_TENSOR_DATA_TYPE_UINT16:
  case DML_TENSOR_DATA_TYPE_INT16:
    size = sizeof(uint16);
    break;

  case DML_TENSOR_DATA_TYPE_UINT8:
  case DML_TENSOR_DATA_TYPE_INT8:
  case DML_TENSOR_DATA_TYPE_INT4:
  case DML_TENSOR_DATA_TYPE_UINT4:
    size = sizeof(uint8);
    break;
  default:
    throw std::runtime_error("Unknown tensor data type.");
    break;
  }

  return size;
}

// =====================================================================================================================
// Finds a DXGI format compatible with the given data type.
DXGI_FORMAT Tensor::ElementFormat(DML_TENSOR_DATA_TYPE dataType) {
  DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;

  switch (dataType) {
  case DML_TENSOR_DATA_TYPE_FLOAT32:
    format = DXGI_FORMAT_R32_FLOAT;
    break;

  case DML_TENSOR_DATA_TYPE_UINT32:
    format = DXGI_FORMAT_R32_UINT;
    break;

  case DML_TENSOR_DATA_TYPE_INT32:
    format = DXGI_FORMAT_R32_SINT;
    break;

  case DML_TENSOR_DATA_TYPE_FLOAT16:
    format = DXGI_FORMAT_R16_FLOAT;
    break;

  case DML_TENSOR_DATA_TYPE_UINT16:
    format = DXGI_FORMAT_R16_UINT;
    break;

  case DML_TENSOR_DATA_TYPE_INT16:
    format = DXGI_FORMAT_R16_SINT;
    break;

  case DML_TENSOR_DATA_TYPE_UINT8:
  case DML_TENSOR_DATA_TYPE_UINT4:
    format = DXGI_FORMAT_R8_UINT;
    break;

  case DML_TENSOR_DATA_TYPE_INT8:
    format = DXGI_FORMAT_R8_SINT;
    break;
  default:
    throw std::runtime_error("Unknown tensor data type.");
    break;
  }

  return format;
}

// =====================================================================================================================
// It's somewhat common to query the size, stride, etc., of a dimension relative
// to its physical memory ordering instead of its logcal ordering. This function
// finds the logical dimension index of a given physical order. Negative values
// start from inside, for example if there are four dimensions then an order of
// -1 really means order 3, the innermost dimension.
int32 Tensor::DimIndex(const TensorDesc &desc, int32 order) {
  // Convert a negative value into a proper non-zero order.
  const int32 size = static_cast<int32>(desc.dims.size());

  if (order < 0) {
    order += size;
  }

  if ((order < 0) || (order >= size)) {
    throw std::runtime_error("Tensor::DimIndex: order is out of bounds!");
  }

  int32 index = -1;

  for (int32 idx = 0; idx < size; ++idx) {
    if (desc.dims[idx].order == order) {
      index = idx;
      break;
    }
  }

  // It should be impossible for us to not find a legal order value.
  assert(index >= 0);

  return index;
}

// =====================================================================================================================
Tensor::Tensor(const TensorDesc &desc, Context *pContext)
    : m_desc(desc), m_systemMem(pContext->GetInfo().systemMem),
      m_pBuffer(nullptr), m_isUploaded(false), m_isBufferConst(false),
      m_ispBufferExternal(pContext->GetInfo().useExternalMemory) {
  CComPtr<ID3D12Device3> d3d12Device = pContext->D3d12Device();

  if (d3d12Device == nullptr) {
    throw std::runtime_error("We must have a D3D12 device to create tensors.");
  }

  // Create a default heap resource to accelerate GPU access during model
  // execution. We create it as a placed resources so that we can do things like
  // disable the heap's zero-init phase.
  D3D12_RESOURCE_DESC resourceDesc = {};
  resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
  resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  resourceDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  resourceDesc.Width = static_cast<UINT64>(m_desc.totalBytes);
  resourceDesc.Height = 1;
  resourceDesc.DepthOrArraySize = 1;
  resourceDesc.MipLevels = 1;
  resourceDesc.SampleDesc.Count = 1;

  const D3D12_RESOURCE_ALLOCATION_INFO allocInfo =
      d3d12Device->GetResourceAllocationInfo(0, 1, &resourceDesc);

  D3D12_HEAP_DESC heapDesc = {};
  heapDesc.SizeInBytes = allocInfo.SizeInBytes;
  heapDesc.Alignment = allocInfo.Alignment;
  heapDesc.Flags =
      D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS | D3D12_HEAP_FLAG_CREATE_NOT_ZEROED;

  if (m_systemMem) {
    // Force the device to allocate system memory.
    heapDesc.Properties.Type = D3D12_HEAP_TYPE_CUSTOM;
    heapDesc.Properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
    heapDesc.Properties.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
  } else {
    // Use the default heap (probably CPU invisible GPU local memory).
    heapDesc.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;

    if (pContext->GetInfo().noPaging == false) {
      // Optimize the runtime's residency paging overhead as well.
      heapDesc.Flags |= D3D12_HEAP_FLAG_CREATE_NOT_RESIDENT;
    }
  }

  if (FAILED(d3d12Device->CreateHeap(&heapDesc, IID_PPV_ARGS(&m_heap)))) {
    throw std::runtime_error("Failed to create a tensor's default heap.");
  }

  if (FAILED(d3d12Device->CreatePlacedResource(
          m_heap, 0, &resourceDesc, D3D12_RESOURCE_STATE_COMMON, nullptr,
          IID_PPV_ARGS(&m_resource)))) {
    throw std::runtime_error("Failed to create a tensor's placed resource.");
  }

  if (m_systemMem) {
    // Just map the buffer resource to get our CPU mapping.
    if (FAILED(m_resource->Map(0, nullptr, &m_pBuffer))) {
      throw std::runtime_error(
          "Failed to map the system memory placed resource.");
    }
  } else {
    if (pContext->GetInfo().noPaging == false) {
      // Kick off an async make resident so that we can keep doing work while
      // it's being paged in.
      pContext->MakeResident(m_heap);
    }

    if (false == m_ispBufferExternal) // if we are using external memory which
                                      // has tensor data.
    {
      // Allocate a full-sized CPU-side buffer for uploading initial data,
      // downloading results, and validation.
      m_pBuffer = malloc(static_cast<size_t>(m_desc.totalBytes));

      if (m_pBuffer == nullptr) {
        throw std::runtime_error(
            "Failed to create a tensor's CPU-side buffer.");
      }
    }
  }
}

void *Tensor::GetMappedTensor() {
  if (m_systemMem) {
    return m_pBuffer;
  }
  return nullptr;
}

// =====================================================================================================================
Tensor::~Tensor() {
  if (m_systemMem) {
    m_resource->Unmap(0, nullptr);
  } else if (false == m_ispBufferExternal) {
    free(m_pBuffer); // only free it, if its not externally passed pointer to
                     // tensor data
  }
}

// =====================================================================================================================
// Fills this tensor with random data from the given generator. Float values are
// in the range [fmin, fmax].
void Tensor::Randomize(std::mt19937 *pGenerator, float fmin, float fmax,
                       int64 maxRandCount) {
  switch (m_desc.dataType) {
  case DML_TENSOR_DATA_TYPE_FLOAT32:
    RandomizeImpl<DML_TENSOR_DATA_TYPE_FLOAT32>(this, pGenerator, fmin, fmax,
                                                maxRandCount);
    break;

  case DML_TENSOR_DATA_TYPE_FLOAT16:
    RandomizeImpl<DML_TENSOR_DATA_TYPE_FLOAT16>(this, pGenerator, fmin, fmax,
                                                maxRandCount);
    break;

  case DML_TENSOR_DATA_TYPE_UINT32:
    RandomizeImpl<DML_TENSOR_DATA_TYPE_UINT32>(this, pGenerator, fmin, fmax,
                                               maxRandCount);
    break;

  case DML_TENSOR_DATA_TYPE_UINT16:
    RandomizeImpl<DML_TENSOR_DATA_TYPE_UINT16>(this, pGenerator, fmin, fmax,
                                               maxRandCount);
    break;

  case DML_TENSOR_DATA_TYPE_UINT8:
    RandomizeImpl<DML_TENSOR_DATA_TYPE_UINT8>(this, pGenerator, fmin, fmax,
                                              maxRandCount);
    break;

  case DML_TENSOR_DATA_TYPE_INT32:
    RandomizeImpl<DML_TENSOR_DATA_TYPE_INT32>(this, pGenerator, fmin, fmax,
                                              maxRandCount);
    break;

  case DML_TENSOR_DATA_TYPE_INT16:
    RandomizeImpl<DML_TENSOR_DATA_TYPE_INT16>(this, pGenerator, fmin, fmax,
                                              maxRandCount);
    break;

  case DML_TENSOR_DATA_TYPE_INT8:
    RandomizeImpl<DML_TENSOR_DATA_TYPE_INT8>(this, pGenerator, fmin, fmax,
                                             maxRandCount);
    break;
  case DML_TENSOR_DATA_TYPE_UINT4:
    RandomizeImpl<DML_TENSOR_DATA_TYPE_UINT4>(this, pGenerator, fmin, fmax,
                                              maxRandCount);
    break;

  default:
    throw std::runtime_error("Unknown tensor data type.");
    break;
  }
}

// =====================================================================================================================
// Fills the tensor with repeating copies of some data, copied from the given
// pattern buffer. The pattern is expected to match the tensor's native storage
// type and have a properly aligned size. The values will be written in an
// implementation defined order so the pattern may look swizzled from the user's
// perspective. Padding may also be written. Basically, this just is just a
// memcpy in a loop that's intended for tensor initialization.
void Tensor::FillPattern(const void *pPattern,
                         int64 size) // The size of the pattern in bytes. Must
                                     // be a multiple of the element size.
{
  assert(size <= SIZE_MAX);

  for (int64 offset = 0; offset < m_desc.totalBytes; offset += size) {
    const int64 copySize = Min(size, m_desc.totalBytes - offset);

    memcpy(VoidPtrInc(m_pBuffer, offset), pPattern,
           static_cast<size_t>(copySize));
  }
}

void Tensor::UpdateTensorBuffer(void *pExternalData) {
  m_pBuffer = pExternalData;
  m_ispBufferExternal = true;
}

} // namespace ryzenai::onnx_utils
