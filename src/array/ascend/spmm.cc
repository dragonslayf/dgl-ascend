#include <dgl/array.h>
#include <dgl/aten/csr.h>
#include <dgl/runtime/device_api.h>
#include "../kernel_decl.h"
#include <vector>
#include <dmlc/logging.h>
#include <cstdint>

#ifdef DGL_USE_ASCEND
#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <acl/acl_op.h>
#define ASCEND_CALL(func)                                                \
  {                                                                      \
    aclError e = (func);                                                 \
    CHECK(e == ACL_SUCCESS) << "Ascend Error, code: " << e; \
  }

// Tiling data structure for SpMM sum kernel
struct SpmmSumTilingData {
    uint32_t numSparseRows;  // Number of sparse matrix rows
    uint32_t numSparseCols;  // Number of sparse matrix cols (also dense matrix rows)
    uint32_t numDenseCols;   // Number of dense matrix cols
    uint32_t nnz;            // Number of non-zero elements
};

// Kernel launch function declaration
#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_spmm_sum(uint32_t blockDim, aclrtStream stream, 
                                         void* row_ptr, void* col_ind, 
                                         void* dense_matrix, void* output, 
                                         void* tiling);

#endif

namespace dgl {
namespace aten {
/**
 * @brief Ascend NPU implementation of SpMM on CSR format using AscendC kernel.
 * 
 * This implementation uses AscendC kernel with CopyIn-Compute-CopyOut framework.
 * The computation is performed directly on NPU using optimized AscendC kernels.
 * 
 * @note Only supports copy_lhs + sum operation currently
 */
template <typename IdType, typename DType>
void SpMMCsrAscend(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {

  // Validate operation
  if (op != "copy_lhs" || reduce != "sum") {
    LOG(FATAL) << "SpMMCsrAscend only supports copy_lhs+sum operation. "
               << "Got: op=" << op << ", reduce=" << reduce;
  }

#ifdef DGL_USE_ASCEND
  
  DGLContext ctx = ufeat->ctx;
  CHECK(ctx.device_type == kDGLAscend) << "Expected Ascend device context";
  ASCEND_CALL(aclrtSetDevice(ctx.device_id));
  
  // Extract matrix dimensions
  int64_t num_rows = csr.num_rows;
  int64_t num_cols = csr.num_cols;
  int64_t num_edges = csr.indices->shape[0];
  int64_t out_dim = (out->ndim > 1) ? out->shape[1] : 1;
  
  // Get device pointers (data is already on NPU)
  const IdType* indptr_ptr = static_cast<const IdType*>(csr.indptr->data);
  const IdType* indices_ptr = static_cast<const IdType*>(csr.indices->data);
  const DType* ufeat_ptr = static_cast<const DType*>(ufeat->data);
  DType* out_ptr = static_cast<DType*>(out->data);
  
  // Create or reuse stream for async operations
  static aclrtStream spmm_stream = nullptr;
  if (spmm_stream == nullptr) {
    ASCEND_CALL(aclrtCreateStream(&spmm_stream));
  }
  aclrtStream stream = spmm_stream;
  
  // Prepare tiling data for kernel
  SpmmSumTilingData tiling;
  tiling.numSparseRows = static_cast<uint32_t>(num_rows);
  tiling.numSparseCols = static_cast<uint32_t>(num_cols);
  tiling.numDenseCols = static_cast<uint32_t>(out_dim);
  tiling.nnz = static_cast<uint32_t>(num_edges);
  
  // Allocate and copy tiling data to device
  SpmmSumTilingData* tiling_device = nullptr;
  ASCEND_CALL(aclrtMalloc(reinterpret_cast<void**>(&tiling_device), 
                          sizeof(SpmmSumTilingData), 
                          ACL_MEM_MALLOC_HUGE_FIRST));
  ASCEND_CALL(aclrtMemcpyAsync(tiling_device, sizeof(SpmmSumTilingData),
                               &tiling, sizeof(SpmmSumTilingData),
                               ACL_MEMCPY_HOST_TO_DEVICE, stream));

  // Launch kernel on NPU
  uint32_t blockDim = 32; // Number of cores to use
  aclError launch_err = ACLRT_LAUNCH_KERNEL(spmm_sum)(blockDim, stream,
                                                       const_cast<void*>(static_cast<const void*>(indptr_ptr)),
                                                       const_cast<void*>(static_cast<const void*>(indices_ptr)),
                                                       const_cast<void*>(static_cast<const void*>(ufeat_ptr)),
                                                       out_ptr,
                                                       tiling_device);
  
  if (launch_err != ACL_SUCCESS) {
    LOG(FATAL) << "Kernel launch failed with error code: " << launch_err;
  }
  ASCEND_CALL(aclrtSynchronizeStream(stream));
  
  // Free device memory
  if (tiling_device != nullptr) {
    ASCEND_CALL(aclrtFree(tiling_device));
  }
  
#else
  LOG(FATAL) << "Ascend support is not compiled. Please compile with -DUSE_ASCEND=ON";
#endif
}

// Template specializations for CSR SpMM
template <>
void SpMMCsr<kDGLAscend, int32_t, float>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
  SpMMCsrAscend<int32_t, float>(op, reduce, bcast, csr, ufeat, efeat, out, out_aux);
}

template <>
void SpMMCsr<kDGLAscend, int64_t, float>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
  SpMMCsrAscend<int64_t, float>(op, reduce, bcast, csr, ufeat, efeat, out, out_aux);
}

template <>
void SpMMCsr<kDGLAscend, int32_t, double>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
    LOG(FATAL) << "Double precision not fully supported on Ascend yet.";
}

template <>
void SpMMCsr<kDGLAscend, int64_t, double>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
    LOG(FATAL) << "Double precision not fully supported on Ascend yet.";
}


/**
 * @brief Ascend implementation of SpMM on COO format.
 * 
 * @note Not implemented yet. COO format SpMM operations will fall back to error.
 * @todo Implement COO SpMM with CPU fallback or native Ascend kernels
 */
template <typename IdType, typename DType>
void SpMMCooAscend(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
  LOG(FATAL) << "SpMMCoo on Ascend is not implemented yet. "
             << "Op: " << op << ", Reduce: " << reduce;
}

// Template specializations for COO SpMM (not implemented)
template <>
void SpMMCoo<kDGLAscend, int32_t, float>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
  SpMMCooAscend<int32_t, float>(op, reduce, bcast, coo, ufeat, efeat, out, out_aux);
}

template <>
void SpMMCoo<kDGLAscend, int64_t, float>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
  SpMMCooAscend<int64_t, float>(op, reduce, bcast, coo, ufeat, efeat, out, out_aux);
}

template <>
void SpMMCoo<kDGLAscend, int32_t, double>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
    LOG(FATAL) << "Double precision not fully supported on Ascend yet.";
}

template <>
void SpMMCoo<kDGLAscend, int64_t, double>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
    LOG(FATAL) << "Double precision not fully supported on Ascend yet.";
}

} // namespace aten
} // namespace dgl
