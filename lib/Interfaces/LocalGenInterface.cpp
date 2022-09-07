//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Interfaces/LocalGenInterface.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

using namespace mlir;

namespace tpu_mlir {

constexpr llvm::StringRef LocalGenInterface::kLayerGroupAttrName;
void LocalGenInterface::fixSlice(int64_t &in_idx, int64_t &in_slice,
                                 int64_t in_length) {
  // avoid leak
  auto end_idx = in_idx + in_slice;
  if (in_idx < 0) {
    in_idx = 0;
  }
  if (end_idx > in_length) {
    end_idx = in_length;
  }
  in_slice = end_idx - in_idx;
}

group_info_t LocalGenInterface::getGroupInfo(mlir::Value v, int64_t n_step,
                                             int64_t h_step) {
  return getGroupInfo(v.getDefiningOp(), n_step, h_step);
}

group_info_t LocalGenInterface::getGroupInfo(mlir::Operation *op,
                                             int64_t n_step, int64_t h_step) {
  assert(op != nullptr);
  assert(op->hasAttr(LocalGenInterface::kLayerGroupAttrName));
  auto g_param = op->getAttr(LocalGenInterface::kLayerGroupAttrName)
                     .cast<tpu::LayerGroup>();
  group_info_t ginfo = {0};
  ginfo.id = g_param.id().getInt();
  ginfo.stage = g_param.stage().getInt();
  ginfo.out_addr = g_param.out_addr().getInt();
  ginfo.out_size = g_param.out_size().getInt();
  ginfo.buffer_addr = g_param.buffer_addr().getInt();
  ginfo.buffer_size = g_param.buffer_size().getInt();
  ginfo.eu_align = g_param.eu_align().getValue();
  auto n_idx_v = Module::getI64Array(g_param.n_idx());
  auto n_slice_v = Module::getI64Array(g_param.n_slice());
  auto h_idx_v = Module::getI64Array(g_param.h_idx());
  auto h_slice_v = Module::getI64Array(g_param.h_slice());
  if (n_idx_v->empty() && h_idx_v->empty()) {
    int64_t n, c, h, w;
    ginfo.overstepped = !(n_step == 0 && h_step == 0);
    Module::getNCHW(op->getResult(0), n, c, h, w);
    ginfo.n_slice = n;
    ginfo.h_slice = h;
  } else {
    if (n_step >= (int64_t)n_idx_v->size() ||
        h_step >= (int64_t)h_idx_v->size()) {
      ginfo.overstepped = true;
    } else {
      ginfo.n_idx = n_idx_v->at(n_step);
      ginfo.n_slice = n_slice_v->at(n_step);
      ginfo.h_idx = h_idx_v->at(h_step);
      ginfo.h_slice = h_slice_v->at(h_step);
      ginfo.overstepped = false;
    }
  }
  return ginfo;
}

} // namespace tpu_mlir

#include "tpu_mlir/Interfaces/LocalGenInterface.cpp.inc"
