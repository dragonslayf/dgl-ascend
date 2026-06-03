/**
 *  Copyright (c) 2024 by Contributors
 * @file graph/sampling/randomwalk_npu.cc
 * @brief Ascend NPU random walk via RandomWalkCustom aclnn op
 */

#ifdef DGL_USE_ASCEND

#include <acl/acl.h>
#include <aclnn/opdev/op_errno.h>
#include <aclnn_random_walk_custom.h>

#include <dgl/array.h>
#include <dgl/base_heterograph.h>
#include <dgl/random.h>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include "randomwalks_impl.h"

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace sampling {

namespace impl {

namespace {

#define NPU_CHECK(cond, msg)                                          \
  do {                                                                \
    if (!(cond)) {                                                    \
      LOG(FATAL) << "[RandomWalk NPU] " << (msg);                     \
    }                                                                 \
  } while (0)

#define ACL_CHECK(expr)                                               \
  do {                                                                \
    aclError _ret = (expr);                                           \
    if (_ret != ACL_SUCCESS) {                                        \
      const char* err_msg = aclGetRecentErrMsg();                     \
      LOG(FATAL) << "[RandomWalk NPU] ACL call failed, code=" << _ret \
                 << (err_msg != nullptr ? ", msg=" : "")              \
                 << (err_msg != nullptr ? err_msg : "");              \
    }                                                                 \
  } while (0)

#define ACLNN_CHECK(expr)                                             \
  do {                                                                \
    aclnnStatus _ret = (expr);                                        \
    if (_ret != ACLNN_SUCCESS) {                                      \
      const char* err_msg = aclGetRecentErrMsg();                     \
      LOG(FATAL) << "[RandomWalk NPU] aclnn call failed, code="       \
                 << _ret << (err_msg != nullptr ? ", msg=" : "")      \
                 << (err_msg != nullptr ? err_msg : "");              \
    }                                                                 \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t d : shape) {
    n *= d;
  }
  return n;
}

template <typename T>
void CreateAclTensorOnDevice(
    const std::vector<T>& host_data, const std::vector<int64_t>& shape,
    void** device_addr, aclDataType dtype, aclTensor** tensor) {
  const size_t bytes = static_cast<size_t>(GetShapeSize(shape)) * sizeof(T);
  ACL_CHECK(aclrtMalloc(device_addr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMemcpy(
      *device_addr, bytes, host_data.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
  *tensor = aclCreateTensor(
      shape.data(), shape.size(), dtype, nullptr, 0, aclFormat::ACL_FORMAT_ND,
      shape.data(), shape.size(), *device_addr);
  NPU_CHECK(*tensor != nullptr, "aclCreateTensor failed");
}

void DestroyAclTensor(aclTensor* tensor, void* device_addr) {
  if (tensor != nullptr) {
    aclDestroyTensor(tensor);
  }
  if (device_addr != nullptr) {
    aclrtFree(device_addr);
  }
}

template <typename T>
void CopyToHost(const void* src, DGLContext src_ctx, T* dst, size_t count) {
  const size_t bytes = count * sizeof(T);
  if (src_ctx.device_type == kDGLCPU) {
    std::memcpy(dst, src, bytes);
    return;
  }
  NPU_CHECK(src_ctx.device_type == kDGLNPU, "expected NPU tensor context");
  ACL_CHECK(aclrtSetDevice(src_ctx.device_id));
  ACL_CHECK(aclrtMemcpy(dst, bytes, src, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
}

template <typename T>
void CopyToDevice(void* dst, DGLContext dst_ctx, const T* src, size_t count) {
  const size_t bytes = count * sizeof(T);
  if (dst_ctx.device_type == kDGLCPU) {
    std::memcpy(dst, src, bytes);
    return;
  }
  NPU_CHECK(dst_ctx.device_type == kDGLNPU, "expected NPU tensor context");
  ACL_CHECK(aclrtSetDevice(dst_ctx.device_id));
  ACL_CHECK(aclrtMemcpy(dst, bytes, src, bytes, ACL_MEMCPY_HOST_TO_DEVICE));
}

template <typename IdType>
std::vector<IdType> CopyIdArrayToHost(const IdArray& arr) {
  const int64_t n = arr->shape[0];
  std::vector<IdType> host(static_cast<size_t>(n));
  CopyToHost(arr->data, arr->ctx, host.data(), static_cast<size_t>(n));
  return host;
}

template <typename IdType>
void GenerateRandomNumbers(
    int64_t num_seeds, int64_t num_steps, std::vector<uint32_t>* out) {
  out->resize(static_cast<size_t>(num_seeds * num_steps));
  auto* engine = RandomEngine::ThreadLocal();
  // Layout: seed-major, matches kernel randomNumberGm[start * numMetapath].
  for (int64_t seed = 0; seed < num_seeds; ++seed) {
    for (int64_t step = 0; step < num_steps; ++step) {
      (*out)[static_cast<size_t>(seed * num_steps + step)] =
          static_cast<uint32_t>(engine->RandInt(1u << 23));
    }
  }
}

template <typename IdType>
CSRMatrix GetCSRForMetapathStep(
    const HeteroGraphPtr& hg, const IdType* metapath_data, int64_t step) {
  const int64_t etype = static_cast<int64_t>(metapath_data[step]);
  return hg->GetCSRMatrix(etype);
}

template <typename IdType>
void CheckHomogeneousMetapathCSR(
    const HeteroGraphPtr& hg, const TypeArray& metapath) {
  const int64_t num_steps = metapath->shape[0];
  const IdType* metapath_data = metapath.Ptr<IdType>();
  const CSRMatrix ref = GetCSRForMetapathStep<IdType>(hg, metapath_data, 0);
  const int64_t ref_ptr_len = ref.indptr->shape[0];
  const int64_t ref_cols_len = ref.indices->shape[0];
  std::vector<IdType> ref_ptr_host = CopyIdArrayToHost<IdType>(ref.indptr);
  std::vector<IdType> ref_cols_host = CopyIdArrayToHost<IdType>(ref.indices);

  for (int64_t step = 1; step < num_steps; ++step) {
    const CSRMatrix csr = GetCSRForMetapathStep<IdType>(hg, metapath_data, step);
    NPU_CHECK(
        csr.indptr->shape[0] == ref_ptr_len &&
            csr.indices->shape[0] == ref_cols_len,
        "NPU RandomWalkCustom requires identical CSR shape for all metapath "
        "edge types.");
    std::vector<IdType> ptr_host = CopyIdArrayToHost<IdType>(csr.indptr);
    std::vector<IdType> cols_host = CopyIdArrayToHost<IdType>(csr.indices);
    NPU_CHECK(
        ptr_host == ref_ptr_host && cols_host == ref_cols_host,
        "NPU RandomWalkCustom requires identical CSR content for all metapath "
        "edge types.");
  }
}

template <typename IdType>
std::pair<IdArray, IdArray> RandomWalkUniformNPU(
    const HeteroGraphPtr hg, const IdArray seeds, const TypeArray metapath,
    FloatArray restart_prob) {
  CHECK(restart_prob->shape[0] == 0)
      << "NPU random walk does not support restart probability yet.";
  CHECK_EQ(seeds->ctx.device_type, kDGLNPU)
      << "seeds must be on NPU for Ascend random walk.";
  CHECK_EQ(metapath->ctx.device_type, kDGLCPU)
      << "metapath is expected on CPU.";

  const int64_t max_num_steps = metapath->shape[0];
  const int64_t num_seeds = seeds->shape[0];
  const int64_t trace_length = max_num_steps + 1;
  auto ctx = seeds->ctx;

  CheckHomogeneousMetapathCSR<IdType>(hg, metapath);

  const IdType* metapath_data = metapath.Ptr<IdType>();
  const CSRMatrix csr = GetCSRForMetapathStep<IdType>(hg, metapath_data, 0);

  std::vector<IdType> row_ptr_host = CopyIdArrayToHost<IdType>(csr.indptr);
  std::vector<IdType> col_ind_host = CopyIdArrayToHost<IdType>(csr.indices);
  std::vector<IdType> seeds_host = CopyIdArrayToHost<IdType>(seeds);
  std::vector<IdType> metapath_host = CopyIdArrayToHost<IdType>(metapath);

  std::vector<uint32_t> random_host;
  GenerateRandomNumbers<IdType>(num_seeds, max_num_steps, &random_host);

  std::vector<int32_t> row_ptr_i32(row_ptr_host.begin(), row_ptr_host.end());
  std::vector<int32_t> col_ind_i32(col_ind_host.begin(), col_ind_host.end());
  std::vector<int32_t> seeds_i32(seeds_host.begin(), seeds_host.end());
  std::vector<int32_t> metapath_i32(metapath_host.begin(), metapath_host.end());
  std::vector<int32_t> random_i32(random_host.begin(), random_host.end());
  std::vector<int32_t> output_i32(
      static_cast<size_t>(num_seeds * trace_length), 0);

  ACL_CHECK(aclrtSetDevice(ctx.device_id));

  aclrtStream stream = nullptr;
  ACL_CHECK(aclrtCreateStream(&stream));

  void* row_ptr_dev = nullptr;
  void* col_ind_dev = nullptr;
  void* seeds_dev = nullptr;
  void* metapath_dev = nullptr;
  void* random_dev = nullptr;
  void* output_dev = nullptr;
  aclTensor* row_ptr_tensor = nullptr;
  aclTensor* col_ind_tensor = nullptr;
  aclTensor* seeds_tensor = nullptr;
  aclTensor* metapath_tensor = nullptr;
  aclTensor* random_tensor = nullptr;
  aclTensor* output_tensor = nullptr;

  CreateAclTensorOnDevice(
      row_ptr_i32, {static_cast<int64_t>(row_ptr_i32.size())}, &row_ptr_dev,
      aclDataType::ACL_INT32, &row_ptr_tensor);
  CreateAclTensorOnDevice(
      col_ind_i32, {static_cast<int64_t>(col_ind_i32.size())}, &col_ind_dev,
      aclDataType::ACL_INT32, &col_ind_tensor);
  CreateAclTensorOnDevice(
      seeds_i32, {num_seeds}, &seeds_dev, aclDataType::ACL_INT32, &seeds_tensor);
  CreateAclTensorOnDevice(
      metapath_i32, {max_num_steps}, &metapath_dev, aclDataType::ACL_INT32,
      &metapath_tensor);
  CreateAclTensorOnDevice(
      random_i32, {num_seeds * max_num_steps}, &random_dev,
      aclDataType::ACL_INT32, &random_tensor);
  CreateAclTensorOnDevice(
      output_i32, {num_seeds * trace_length}, &output_dev,
      aclDataType::ACL_INT32, &output_tensor);

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  ACLNN_CHECK(aclnnRandomWalkCustomGetWorkspaceSize(
      seeds_tensor, row_ptr_tensor, col_ind_tensor, metapath_tensor,
      random_tensor, output_tensor, &workspace_size, &executor));

  void* workspace = nullptr;
  if (workspace_size > 0) {
    ACL_CHECK(aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST));
  }

  ACLNN_CHECK(aclnnRandomWalkCustom(workspace, workspace_size, executor, stream));
  ACL_CHECK(aclrtSynchronizeStream(stream));

  ACL_CHECK(aclrtMemcpy(
      output_i32.data(), output_i32.size() * sizeof(int32_t), output_dev,
      output_i32.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST));

  if (workspace != nullptr) {
    aclrtFree(workspace);
  }
  DestroyAclTensor(output_tensor, output_dev);
  DestroyAclTensor(random_tensor, random_dev);
  DestroyAclTensor(metapath_tensor, metapath_dev);
  DestroyAclTensor(seeds_tensor, seeds_dev);
  DestroyAclTensor(col_ind_tensor, col_ind_dev);
  DestroyAclTensor(row_ptr_tensor, row_ptr_dev);
  aclrtDestroyStream(stream);

  IdArray traces =
      IdArray::Empty({num_seeds, trace_length}, seeds->dtype, ctx);
  IdArray eids =
      IdArray::Empty({num_seeds, max_num_steps}, seeds->dtype, ctx);

  std::vector<IdType> traces_host(static_cast<size_t>(num_seeds * trace_length));
  for (int64_t seed = 0; seed < num_seeds; ++seed) {
    for (int64_t hop = 0; hop < trace_length; ++hop) {
      traces_host[static_cast<size_t>(seed * trace_length + hop)] =
          static_cast<IdType>(
              output_i32[static_cast<size_t>(seed * trace_length + hop)]);
    }
  }

  std::vector<IdType> eids_host(static_cast<size_t>(num_seeds * max_num_steps), -1);

  CopyToDevice(
      traces->data, ctx, traces_host.data(), static_cast<size_t>(traces_host.size()));
  CopyToDevice(
      eids->data, ctx, eids_host.data(), static_cast<size_t>(eids_host.size()));

  return std::make_pair(traces, eids);
}

}  // namespace

template <DGLDeviceType XPU, typename IdType>
TypeArray GetNodeTypesFromMetapath(
    const HeteroGraphPtr hg, const TypeArray metapath) {
  uint64_t num_etypes = metapath->shape[0];
  TypeArray result = TypeArray::Empty(
      {metapath->shape[0] + 1}, metapath->dtype, metapath->ctx);

  const IdType* metapath_data = metapath.Ptr<IdType>();
  IdType* result_data = result.Ptr<IdType>();

  dgl_type_t curr_type = hg->GetEndpointTypes(metapath_data[0]).first;
  result_data[0] = curr_type;

  for (uint64_t i = 0; i < num_etypes; ++i) {
    auto src_dst_type = hg->GetEndpointTypes(metapath_data[i]);
    dgl_type_t srctype = src_dst_type.first;
    dgl_type_t dsttype = src_dst_type.second;

    if (srctype != curr_type) {
      LOG(FATAL) << "source of edge type #" << i
                 << " does not match destination of edge type #" << i - 1;
      return result;
    }
    curr_type = dsttype;
    result_data[i + 1] = dsttype;
  }
  return result;
}

template <DGLDeviceType XPU, typename IdType>
std::pair<IdArray, IdArray> RandomWalk(
    const HeteroGraphPtr hg, const IdArray seeds, const TypeArray metapath,
    const std::vector<FloatArray>& prob) {
  bool is_uniform = true;
  for (const auto& etype_prob : prob) {
    if (!IsNullArray(etype_prob)) {
      is_uniform = false;
      break;
    }
  }
  CHECK(is_uniform) << "NPU random walk only supports uniform transition.";

  auto restart_prob =
      NDArray::Empty({0}, DGLDataType{kDGLFloat, 32, 1}, DGLContext{kDGLNPU, 0});
  return RandomWalkUniformNPU<IdType>(hg, seeds, metapath, restart_prob);
}

template <DGLDeviceType XPU, typename IdType>
std::pair<IdArray, IdArray> RandomWalkWithRestart(
    const HeteroGraphPtr hg, const IdArray seeds, const TypeArray metapath,
    const std::vector<FloatArray>& prob, double restart_prob) {
  LOG(FATAL) << "RandomWalkWithRestart is not implemented on NPU.";
  return {};
}

template <DGLDeviceType XPU, typename IdType>
std::pair<IdArray, IdArray> RandomWalkWithStepwiseRestart(
    const HeteroGraphPtr hg, const IdArray seeds, const TypeArray metapath,
    const std::vector<FloatArray>& prob, FloatArray restart_prob) {
  LOG(FATAL) << "RandomWalkWithStepwiseRestart is not implemented on NPU.";
  return {};
}

template <DGLDeviceType XPU, typename IdxType>
std::tuple<IdArray, IdArray, IdArray> SelectPinSageNeighbors(
    const IdArray src, const IdArray dst, const int64_t num_samples_per_node,
    const int64_t k) {
  LOG(FATAL) << "SelectPinSageNeighbors is not implemented on NPU.";
  return {};
}

#define INSTANTIATE_NPU_RANDOMWALK(IdType)                                         \
  template TypeArray GetNodeTypesFromMetapath<kDGLNPU, IdType>(                    \
      const HeteroGraphPtr hg, const TypeArray metapath);                            \
  template std::pair<IdArray, IdArray> RandomWalk<kDGLNPU, IdType>(                 \
      const HeteroGraphPtr hg, const IdArray seeds, const TypeArray metapath,        \
      const std::vector<FloatArray>& prob);                                          \
  template std::pair<IdArray, IdArray> RandomWalkWithRestart<kDGLNPU, IdType>(     \
      const HeteroGraphPtr hg, const IdArray seeds, const TypeArray metapath,        \
      const std::vector<FloatArray>& prob, double restart_prob);                   \
  template std::pair<IdArray, IdArray>                                             \
  RandomWalkWithStepwiseRestart<kDGLNPU, IdType>(                                  \
      const HeteroGraphPtr hg, const IdArray seeds, const TypeArray metapath,        \
      const std::vector<FloatArray>& prob, FloatArray restart_prob);                 \
  template std::tuple<IdArray, IdArray, IdArray>                                     \
  SelectPinSageNeighbors<kDGLNPU, IdType>(                                         \
      const IdArray src, const IdArray dst, const int64_t num_samples_per_node,     \
      const int64_t k)

INSTANTIATE_NPU_RANDOMWALK(int32_t);
INSTANTIATE_NPU_RANDOMWALK(int64_t);

}  // namespace impl

}  // namespace sampling

}  // namespace dgl

#endif  // DGL_USE_ASCEND
