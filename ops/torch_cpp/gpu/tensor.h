/* Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved. */
#pragma once

#include <atlbase.h>
#include <d3d12.h>

#include <ostream>
#include <random>
#include <vector>

#include "DirectML.h"
#include "types.h"

namespace ryzenai::onnx_utils {

class Context;
class Tensor;
struct TensorDesc;

// We will use these vectors frequently so create some typedefs. They need to
// store pointers so that we can put holes in the vector where our Context must
// insert "none" DirectML bindings.
typedef std::vector<std::shared_ptr<TensorDesc>> TensorDescVector;
typedef std::vector<std::shared_ptr<Tensor>> TensorVector;

// All information needed to describe a single dimension in a tensor.
struct TensorDim {
  int64 size;   // The logical size of this dimension of the tensor in elements.
  int64 stride; // The absolute distance between elements in this dimension in
                // elements.
  int32 order;  // Where this dimension is in the tensor's memory packing order.
                // Zero means this is the outer most dimension, one is the next
                // largest, etc. An NCHW tensor has orders of [0, 1, 2, 3].
};

// All information needed to describe a tensor.
struct TensorDesc {
  hstring name; // The name of the tensor.
  std::vector<TensorDim>
      dims; // The size, stride, etc. of each dimension of the tensor.
  DML_TENSOR_DATA_TYPE dataType; // The element data type.
  int64 totalBytes;              // The total size of the tensor in bytes.
  bool isConst;                  // If the tensor is constant.
};

// =====================================================================================================================
// A DirectML buffer tensor. Accessible by the CPU and GPU.
class Tensor {
public:
  static int32 ElementSizeBits(DML_TENSOR_DATA_TYPE dataType);
  static int32 ElementSize(DML_TENSOR_DATA_TYPE dataType);
  static DXGI_FORMAT ElementFormat(DML_TENSOR_DATA_TYPE dataType);

  // Returns the index of the dimension with the given order. Negative values
  // start from inside, for example if there are four dimensions then an order
  // of -1 really means order 3, the innermost dimension.
  static int32 DimIndex(const TensorDesc &desc, int32 order);

  Tensor(const TensorDesc &desc, Context *pContext);
  ~Tensor();

  // Get the cpu accesible pointer to tensor's d3d resource.
  void *GetMappedTensor();

  const TensorDesc &Desc() const { return m_desc; }

  // The tensor's CPU-visible memory.
  void *Buffer() const { return m_pBuffer; }

  // The tensor's GPU-visible resource.
  CComPtr<ID3D12Resource> Resource() const { return m_resource; }

  // Just a helper function to make it easier to call the static version.
  int32 DimIndex(int32 order) const { return DimIndex(m_desc, order); }

  // Compares two tensors of the same type. If there are any differences (within
  // the tolerance if floating point) they are written to the output stream.
  // Returns true if there are no differences. bool Validate(const Tensor&
  // reference, float tolerance, bool hex, std::wostream* pOutput) const;

  // Fills the tensor with random data from the given generator.
  void Randomize(std::mt19937 *pGenerator, float fmin, float fmax,
                 int64 maxRandCount);

  // Fills the tensor with repeating copies of some data, copied from the given
  // pattern buffer. The pattern is expected to match the tensor's native
  // storage type and have a properly aligned size. The values will be written
  // in an implementation defined order so the pattern may look swizzled from
  // the user's perspective. Padding may also be written. Basically, this just
  // is just a memcpy in a loop that's intended for tensor initialization.
  void FillPattern(const void *pPattern, int64 size);
  void UpdateTensorBuffer(void *pExternalData);

  bool IsUploaded() const { return m_isUploaded; }
  void SetUploaded(bool uploaded) { m_isUploaded = uploaded; }

  void SetConstFlag(bool isConst) { m_isBufferConst = isConst; }
  bool GetConstFlag() const { return m_isBufferConst; }

private:
  const TensorDesc m_desc; // The tensor's descriptor.
  const bool
      m_systemMem; // Use a single persistently mapped system memory allocation.
  void *m_pBuffer; // A read/write pointer to the tensor's CPU data buffer.
  bool m_ispBufferExternal; // dont free/delete If m_pBuffer comes externally
                            // from application
  bool m_isBufferConst;
  CComPtr<ID3D12Heap>
      m_heap; // A clone of the staging resource in the default heap.
  CComPtr<ID3D12Resource>
      m_resource; // A placed resource used to access our GPU heap. It must be
                  // manually uploaded and downloaded to keep the CPU data in
                  // sync.
  bool m_isUploaded; // Check if the constants / weight tensor is uploaded
};

// =====================================================================================================================
// A generic base for a class which wraps a tensor with a set of typed set/get
// functions.
class TensorViewBase {
public:
  TensorViewBase(const Tensor &tensor);
  virtual ~TensorViewBase() {}

  // Returns true if the built-in iterator has not reached the end of the
  // tensor.
  bool IterValid() const { return (m_iterAtEnd == false); }

  // Returns the built-in iterator indices.
  const std::vector<int64> &IterIndices() const { return m_iter; }

  // Moves the built-in iterator to the next element in the tensor, possibly
  // reaching the end. The iteration order is implementation defined, use custom
  // iteration indices if you care about the order. For now this will always
  // increment the physically innermost dimension first because it's best for
  // memory locality. This may not be the logical innermost dimension, you have
  // to check the tensor's order specifiers to be sure.
  void IterNext();

  // Moves the built-in iterator to the next element in the tensor, possibly
  // reaching the end. The iteration order is linear logical, meaning it always
  // increments the innermost logical dimension first, then the dimension one
  // step up once we reach the end, and so on. In other words, logical order
  // views the tensor as NCHW layout and increments W first, then H, etc.,
  // regardless of the tensor's strides.
  void IterNextLogical();

  // Moves the built-in iterator back to the first element in the tensor.
  void IterReset();

protected:
  // Converts a set of tensor indices to a flat pointer into the tensor's
  // buffer.
  void *Flatten(const std::vector<int64> &indices) const;

  const Tensor &m_tensor;    // The tensor accessed by this view.
  const int32 m_elementSize; // The size of an element in bytes.
  std::vector<int64> m_iter; // The current position of the built-in iterator.
  void
      *m_pIterPos; // Points to the current position within the tensor's buffer.
  bool m_iterAtEnd; // If the iterator is at the end of the tensor.

  // These are all used to optimize IterNext and are constant after
  // construction.
  int32 m_innerDim;      // The index of the innermost dimension.
  bool m_innerDimPacked; // If the innermost dimension is packed.
  int64 m_innerDimLast;  // The last valid index in the innermost dimension.
};

// =====================================================================================================================
// A templated class which wraps a tensor with a set of typed set/get functions.
template <typename _S, typename _D> class TensorView : public TensorViewBase {
public:
  // Define user-accessable helper types for each of our template types:
  // - StorageType: Holds the raw data of a single element. Can be opaque (e.g.
  // float16).
  // - DataType:    The CPU views each element using this type, converting
  // between the storage type as needed.
  using StorageType = typename _S;
  using DataType = typename _D;

  TensorView(const Tensor &tensor) : TensorViewBase(tensor) {}
  virtual ~TensorView() {}

  // Sets a value in the tensor given one index for each of the tensor's
  // dimensions.
  void SetValueAt(const std::vector<int64> &indices, DataType value) {
    *static_cast<StorageType *>(Flatten(indices)) =
        static_cast<StorageType>(value);
  }

  // Sets the value that the built-in iterator is pointing at.
  void SetValue(DataType value) {
    *static_cast<StorageType *>(m_pIterPos) = static_cast<StorageType>(value);
  }

  // Sets the value that the built-in iterator is pointing at using the Tensor's
  // raw storage type.
  void SetValueRaw(StorageType value) {
    *static_cast<StorageType *>(m_pIterPos) = value;
  }

  // Returns a value from the tensor given one index for each of the tensor's
  // dimensions.
  DataType GetValueAt(const std::vector<int64> &indices) const {
    return static_cast<DataType>(*static_cast<StorageType *>(Flatten(indices)));
  }

  // Returns the value that the built-in iterator is pointing at.
  DataType GetValue() const {
    return static_cast<DataType>(*static_cast<StorageType *>(m_pIterPos));
  }
};

// A helper template struct which translates a data type to a tensor view type
// at compile-time. We should use this struct instead of constructing a
// TensorView directly so that it's easier to tweak the TensorView types.
template <DML_TENSOR_DATA_TYPE Type = DML_TENSOR_DATA_TYPE_UNKNOWN>
struct TensorViewLookup {
  // This is a dummy type and shouldn't be used.
  using View = void;
};

template <> struct TensorViewLookup<DML_TENSOR_DATA_TYPE_FLOAT32> {
  using View = TensorView<float, float>;
};
template <> struct TensorViewLookup<DML_TENSOR_DATA_TYPE_FLOAT16> {
  using View = TensorView<Float16, float>;
};
template <> struct TensorViewLookup<DML_TENSOR_DATA_TYPE_UINT32> {
  using View = TensorView<uint32, uint32>;
};
template <> struct TensorViewLookup<DML_TENSOR_DATA_TYPE_UINT16> {
  using View = TensorView<uint16, uint16>;
};
template <> struct TensorViewLookup<DML_TENSOR_DATA_TYPE_UINT8> {
  using View = TensorView<uint8, uint8>;
};
template <> struct TensorViewLookup<DML_TENSOR_DATA_TYPE_INT32> {
  using View = TensorView<int32, int32>;
};
template <> struct TensorViewLookup<DML_TENSOR_DATA_TYPE_INT16> {
  using View = TensorView<int16, int16>;
};
template <> struct TensorViewLookup<DML_TENSOR_DATA_TYPE_INT8> {
  using View = TensorView<int8, int8>;
};
template <> struct TensorViewLookup<DML_TENSOR_DATA_TYPE_UINT4> {
  using View = TensorView<uint8, uint8>;
}; // TODO: simulate int4
template <> struct TensorViewLookup<DML_TENSOR_DATA_TYPE_INT4> {
  using View = TensorView<uint8, uint8>;
}; // TODO: simulate int4

} // namespace ryzenai::onnx_utils
