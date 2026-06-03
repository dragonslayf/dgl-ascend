/**
 *  Copyright (c) 2024 by Contributors
 * @file npu_device_api.cc
 * @brief Ascend NPU device API for tensors imported from torch_npu.
 */
#ifdef DGL_USE_ASCEND

#include <acl/acl.h>

#include <cstdint>
#include <cstring>
#include <memory>

#include <dgl/runtime/device_api.h>
#include <dgl/runtime/registry.h>

namespace dgl {
namespace runtime {

namespace {

#define ACL_RT_CHECK(expr)                                              \
  do {                                                                  \
    aclError _ret = (expr);                                             \
    CHECK_EQ(_ret, ACL_SUCCESS) << "ACL error code: " << _ret;            \
  } while (0)

}  // namespace

class NPUDeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(DGLContext ctx) final { ACL_RT_CHECK(aclrtSetDevice(ctx.device_id)); }

  void GetAttr(DGLContext ctx, DeviceAttrKind kind, DGLRetValue* rv) final {
    if (kind == kExist) {
      *rv = 1;
      return;
    }
    *rv = 0;
  }

  void* AllocDataSpace(
      DGLContext ctx, size_t nbytes, size_t alignment,
      DGLDataType type_hint) final {
    SetDevice(ctx);
    void* ptr = nullptr;
    if (nbytes == 0) return ptr;
    ACL_RT_CHECK(aclrtMalloc(&ptr, nbytes, ACL_MEM_MALLOC_HUGE_FIRST));
    return ptr;
  }

  void FreeDataSpace(DGLContext ctx, void* ptr) final {
    if (ptr == nullptr) return;
    SetDevice(ctx);
    ACL_RT_CHECK(aclrtFree(ptr));
  }

  void CopyDataFromTo(
      const void* from, size_t from_offset, void* to, size_t to_offset,
      size_t size, DGLContext ctx_from, DGLContext ctx_to,
      DGLDataType type_hint) final {
    if (size == 0) return;
    const char* src = static_cast<const char*>(from) + from_offset;
    char* dst = static_cast<char*>(to) + to_offset;

    if (ctx_from.device_type == kDGLCPU && ctx_to.device_type == kDGLNPU) {
      SetDevice(ctx_to);
      ACL_RT_CHECK(aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_HOST_TO_DEVICE));
    } else if (ctx_from.device_type == kDGLNPU && ctx_to.device_type == kDGLCPU) {
      SetDevice(ctx_from);
      ACL_RT_CHECK(aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST));
    } else if (ctx_from.device_type == kDGLNPU && ctx_to.device_type == kDGLNPU) {
      SetDevice(ctx_from);
      ACL_RT_CHECK(aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_DEVICE));
    } else {
      memcpy(dst, src, size);
    }
  }

  void RecordedCopyDataFromTo(
      void* from, size_t from_offset, void* to, size_t to_offset, size_t size,
      DGLContext ctx_from, DGLContext ctx_to, DGLDataType type_hint,
      void* pytorch_ctx) final {
    CopyDataFromTo(
        from, from_offset, to, to_offset, size, ctx_from, ctx_to, type_hint);
  }

  DGLStreamHandle CreateStream(DGLContext ctx) final {
    SetDevice(ctx);
    aclrtStream stream = nullptr;
    ACL_RT_CHECK(aclrtCreateStream(&stream));
    return static_cast<DGLStreamHandle>(stream);
  }

  void FreeStream(DGLContext ctx, DGLStreamHandle stream) final {
    if (stream == nullptr) return;
    SetDevice(ctx);
    ACL_RT_CHECK(aclrtDestroyStream(static_cast<aclrtStream>(stream)));
  }

  void StreamSync(DGLContext ctx, DGLStreamHandle stream) final {
    SetDevice(ctx);
    ACL_RT_CHECK(aclrtSynchronizeStream(static_cast<aclrtStream>(stream)));
  }

  static const std::shared_ptr<NPUDeviceAPI>& Global() {
    static std::shared_ptr<NPUDeviceAPI> inst = std::make_shared<NPUDeviceAPI>();
    return inst;
  }
};

DGL_REGISTER_GLOBAL("device_api.npu")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      DeviceAPI* ptr = NPUDeviceAPI::Global().get();
      *rv = static_cast<void*>(ptr);
    });

}  // namespace runtime
}  // namespace dgl

#endif  // DGL_USE_ASCEND
