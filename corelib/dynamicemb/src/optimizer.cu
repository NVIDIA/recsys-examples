/******************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
All rights reserved. # SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
******************************************************************************/

#include "check.h"
#include "optimizer.h"
#include "optimizer_kernel.cuh"
#include "torch_utils.h"
#include "utils.h"
#include <functional>

namespace dyn_emb {

constexpr int MULTIPLIER = 4;
constexpr int WARPSIZE = 32;
constexpr int OPTIMIZER_BLOCKSIZE_VEC = 64;
constexpr int OPTIMIZER_BLOCKSIZE = 1024;

template <typename GradType, typename WeightType, typename IndexType,
          typename OptimizerType>
void launch_update_kernel_for_flat_table(
    GradType *grads, int64_t *table_ptrs, IndexType *indices,
    int64_t *table_ids,
    int64_t *table_value_dims,
    int64_t *table_emb_dims, OptimizerType opt, int64_t const ev_nums,
    uint32_t const grad_stride, uint32_t const max_emb_dim, bool all_dims_vec4,
    int device_id,
    std::function<float(int)> smem_size_f = [](int block_size) { return 0; }) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto &device_prop = DeviceProp::getDeviceProp(device_id);
  if (all_dims_vec4) {
    const int max_grid_size =
        device_prop.num_sms *
        (device_prop.max_thread_per_sm / OPTIMIZER_BLOCKSIZE_VEC);
    const int warp_per_block = OPTIMIZER_BLOCKSIZE_VEC / WARPSIZE;

    int grid_size = 0;
    if (ev_nums / warp_per_block < max_grid_size) {
      grid_size = (ev_nums - 1) / warp_per_block + 1;
    } else if (ev_nums / warp_per_block > max_grid_size * MULTIPLIER) {
      grid_size = max_grid_size * MULTIPLIER;
    } else {
      grid_size = max_grid_size;
    }

    auto kernel = update4_with_index_flat_table_kernel<GradType, WeightType,
                                                       IndexType,
                                                       OptimizerType>;
    kernel<<<grid_size, OPTIMIZER_BLOCKSIZE_VEC, 0, stream>>>(
        ev_nums, grad_stride, grads, table_ptrs, indices, table_ids,
        table_value_dims,
        table_emb_dims, opt);
  } else {
    int block_size =
        max_emb_dim > OPTIMIZER_BLOCKSIZE ? OPTIMIZER_BLOCKSIZE : max_emb_dim;
    int grid_size = ev_nums;

    auto kernel = update_with_index_flat_table_kernel<GradType, WeightType,
                                                      IndexType,
                                                      OptimizerType>;
    kernel<<<grid_size, block_size, smem_size_f(block_size), stream>>>(
        ev_nums, grad_stride, grads, table_ptrs, indices, table_ids,
        table_value_dims,
        table_emb_dims, opt);
  }
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

void sgd_update_for_flat_table(at::Tensor grads, at::Tensor indices,
                               at::Tensor table_ptrs, at::Tensor table_ids,
                               at::Tensor table_value_dims,
                               at::Tensor table_emb_dims,
                               int64_t max_emb_dim, bool all_dims_vec4,
                               float const lr,
                               int64_t table_dtype) {
  int64_t ev_nums = grads.size(0);
  uint32_t grad_stride = grads.size(1);
  if (ev_nums == 0)
    return;
  TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");
  TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");

  uint32_t max_emb_dim_u32 = static_cast<uint32_t>(max_emb_dim);

  auto grad_type = get_data_type(grads);
  auto val_type = static_cast<DataType>(table_dtype);
  auto index_type = get_data_type(indices);
  int device_id = grads.device().index();

  DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(val_type, w_t, [&] {
      DISPATCH_OFFSET_INT_TYPE(index_type, i_t, [&] {
        auto grad_ptr = get_pointer<g_t>(grads);
        auto table_ptrs_ptr = get_pointer<int64_t>(table_ptrs);
        auto index_ptr = get_pointer<i_t>(indices);
        auto tid_ptr = get_pointer<int64_t>(table_ids);
        auto tvd_ptr = get_pointer<int64_t>(table_value_dims);
        auto ted_ptr = get_pointer<int64_t>(table_emb_dims);

        SgdVecOptimizer<g_t, w_t> opt{lr};

        launch_update_kernel_for_flat_table<g_t, w_t, i_t, decltype(opt)>(
            grad_ptr, table_ptrs_ptr, index_ptr, tid_ptr, tvd_ptr,
            ted_ptr, opt, ev_nums, grad_stride, max_emb_dim_u32, all_dims_vec4,
            device_id);
      });
    });
  });
}

void adam_update_for_flat_table(at::Tensor grads, at::Tensor indices,
                                at::Tensor table_ptrs, at::Tensor table_ids,
                                at::Tensor table_value_dims,
                                at::Tensor table_emb_dims, const float lr,
                                const float beta1, const float beta2,
                                const float eps, const float weight_decay,
                                const uint32_t iter_num,
                                int64_t max_emb_dim, bool all_dims_vec4,
                                int64_t table_dtype) {
  int64_t ev_nums = grads.size(0);
  uint32_t grad_stride = grads.size(1);
  if (ev_nums == 0)
    return;
  TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");
  TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");

  uint32_t max_emb_dim_u32 = static_cast<uint32_t>(max_emb_dim);

  auto grad_type = get_data_type(grads);
  auto val_type = static_cast<DataType>(table_dtype);
  auto index_type = get_data_type(indices);
  int device_id = grads.device().index();

  DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(val_type, w_t, [&] {
      DISPATCH_OFFSET_INT_TYPE(index_type, i_t, [&] {
        auto grad_ptr = get_pointer<g_t>(grads);
        auto table_ptrs_ptr = get_pointer<int64_t>(table_ptrs);
        auto index_ptr = get_pointer<i_t>(indices);
        auto tid_ptr = get_pointer<int64_t>(table_ids);
        auto tvd_ptr = get_pointer<int64_t>(table_value_dims);
        auto ted_ptr = get_pointer<int64_t>(table_emb_dims);

        AdamVecOptimizer<g_t, w_t> opt{lr,  beta1,        beta2,
                                       eps, weight_decay, iter_num};

        launch_update_kernel_for_flat_table<g_t, w_t, i_t, decltype(opt)>(
            grad_ptr, table_ptrs_ptr, index_ptr, tid_ptr, tvd_ptr,
            ted_ptr, opt, ev_nums, grad_stride, max_emb_dim_u32, all_dims_vec4,
            device_id);
      });
    });
  });
}

void adagrad_update_for_flat_table(at::Tensor grads, at::Tensor indices,
                                   at::Tensor table_ptrs,
                                   at::Tensor table_ids,
                                   at::Tensor table_value_dims,
                                   at::Tensor table_emb_dims, const float lr,
                                   const float eps,
                                   int64_t max_emb_dim, bool all_dims_vec4,
                                   int64_t table_dtype) {
  int64_t ev_nums = grads.size(0);
  uint32_t grad_stride = grads.size(1);
  if (ev_nums == 0)
    return;
  TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");
  TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");

  uint32_t max_emb_dim_u32 = static_cast<uint32_t>(max_emb_dim);

  auto grad_type = get_data_type(grads);
  auto val_type = static_cast<DataType>(table_dtype);
  auto index_type = get_data_type(indices);
  int device_id = grads.device().index();

  DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(val_type, w_t, [&] {
      DISPATCH_OFFSET_INT_TYPE(index_type, i_t, [&] {
        auto grad_ptr = get_pointer<g_t>(grads);
        auto table_ptrs_ptr = get_pointer<int64_t>(table_ptrs);
        auto index_ptr = get_pointer<i_t>(indices);
        auto tid_ptr = get_pointer<int64_t>(table_ids);
        auto tvd_ptr = get_pointer<int64_t>(table_value_dims);
        auto ted_ptr = get_pointer<int64_t>(table_emb_dims);

        AdaGradVecOptimizer<g_t, w_t> opt{lr, eps};

        launch_update_kernel_for_flat_table<g_t, w_t, i_t, decltype(opt)>(
            grad_ptr, table_ptrs_ptr, index_ptr, tid_ptr, tvd_ptr,
            ted_ptr, opt, ev_nums, grad_stride, max_emb_dim_u32, all_dims_vec4,
            device_id);
      });
    });
  });
}

void rowwise_adagrad_for_flat_table(at::Tensor grads, at::Tensor indices,
                                    at::Tensor table_ptrs,
                                    at::Tensor table_ids,
                                    at::Tensor table_value_dims,
                                    at::Tensor table_emb_dims, const float lr,
                                    const float eps,
                                    int64_t max_emb_dim, bool all_dims_vec4,
                                    int64_t table_dtype) {
  int64_t ev_nums = grads.size(0);
  uint32_t grad_stride = grads.size(1);
  if (ev_nums == 0)
    return;
  TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");
  TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");

  uint32_t max_emb_dim_u32 = static_cast<uint32_t>(max_emb_dim);

  auto grad_type = get_data_type(grads);
  auto val_type = static_cast<DataType>(table_dtype);
  auto index_type = get_data_type(indices);
  int device_id = grads.device().index();

  DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(val_type, w_t, [&] {
      DISPATCH_OFFSET_INT_TYPE(index_type, i_t, [&] {
        auto grad_ptr = get_pointer<g_t>(grads);
        auto table_ptrs_ptr = get_pointer<int64_t>(table_ptrs);
        auto index_ptr = get_pointer<i_t>(indices);
        auto tid_ptr = get_pointer<int64_t>(table_ids);
        auto tvd_ptr = get_pointer<int64_t>(table_value_dims);
        auto ted_ptr = get_pointer<int64_t>(table_emb_dims);

        RowWiseAdaGradVecOptimizer<g_t, w_t> opt{lr, eps};

        launch_update_kernel_for_flat_table<g_t, w_t, i_t, decltype(opt)>(
            grad_ptr, table_ptrs_ptr, index_ptr, tid_ptr, tvd_ptr,
            ted_ptr, opt, ev_nums, grad_stride, max_emb_dim_u32, all_dims_vec4,
            device_id,
            [](int block_size) { return block_size * sizeof(float); });
      });
    });
  });
}

template <typename GradType, typename WeightType, typename OptimizerType>
void launch_update_kernel_for_padded_buffer(
    GradType *grads, WeightType *values, OptimizerType opt,
    int64_t const num_rows, uint32_t const grad_stride,
    uint32_t const value_stride, uint32_t const emb_dim, bool all_dims_vec4,
    int device_id,
    std::function<float(int)> smem_size_f = [](int block_size) { return 0; }) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto &device_prop = DeviceProp::getDeviceProp(device_id);
  if (all_dims_vec4) {
    const int max_grid_size =
        device_prop.num_sms *
        (device_prop.max_thread_per_sm / OPTIMIZER_BLOCKSIZE_VEC);
    const int warp_per_block = OPTIMIZER_BLOCKSIZE_VEC / WARPSIZE;
    int grid_size = 0;
    if (num_rows / warp_per_block < max_grid_size) {
      grid_size = (num_rows - 1) / warp_per_block + 1;
    } else if (num_rows / warp_per_block > max_grid_size * MULTIPLIER) {
      grid_size = max_grid_size * MULTIPLIER;
    } else {
      grid_size = max_grid_size;
    }
    update4_padded_buffer_kernel<GradType, WeightType, OptimizerType>
        <<<grid_size, OPTIMIZER_BLOCKSIZE_VEC, 0, stream>>>(
            num_rows, grad_stride, value_stride, grads, values, opt);
  } else {
    int block_size =
        emb_dim > OPTIMIZER_BLOCKSIZE ? OPTIMIZER_BLOCKSIZE : emb_dim;
    int grid_size = num_rows;
    update_padded_buffer_kernel<GradType, WeightType, OptimizerType>
        <<<grid_size, block_size, smem_size_f(block_size), stream>>>(
            num_rows, grad_stride, value_stride, grads, values, opt);
  }
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

void sgd_update_for_padded_buffer(at::Tensor grads, at::Tensor values,
                                  int64_t emb_dim, int64_t value_dim,
                                  float lr) {
  int64_t num_rows = grads.size(0);
  uint32_t grad_stride = grads.size(1);
  if (num_rows == 0)
    return;
  TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");
  TORCH_CHECK(values.is_cuda(), "values must be a CUDA tensor");
  uint32_t emb_dim_u32 = static_cast<uint32_t>(emb_dim);
  uint32_t value_stride = static_cast<uint32_t>(value_dim);
  bool all_dims_vec4 = (emb_dim % 4 == 0);
  auto grad_type = get_data_type(grads);
  auto val_type = get_data_type(values);
  int device_id = grads.device().index();
  DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(val_type, w_t, [&] {
      SgdVecOptimizer<g_t, w_t> opt{lr};
      launch_update_kernel_for_padded_buffer<g_t, w_t, decltype(opt)>(
          get_pointer<g_t>(grads), get_pointer<w_t>(values), opt, num_rows,
          grad_stride, value_stride, emb_dim_u32, all_dims_vec4, device_id);
    });
  });
}

void adam_update_for_padded_buffer(at::Tensor grads, at::Tensor values,
                                   int64_t emb_dim, int64_t value_dim,
                                   float lr, float beta1, float beta2,
                                   float eps, float weight_decay,
                                   uint32_t iter_num) {
  int64_t num_rows = grads.size(0);
  uint32_t grad_stride = grads.size(1);
  if (num_rows == 0)
    return;
  TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");
  TORCH_CHECK(values.is_cuda(), "values must be a CUDA tensor");
  uint32_t emb_dim_u32 = static_cast<uint32_t>(emb_dim);
  uint32_t value_stride = static_cast<uint32_t>(value_dim);
  bool all_dims_vec4 = (emb_dim % 4 == 0);
  auto grad_type = get_data_type(grads);
  auto val_type = get_data_type(values);
  int device_id = grads.device().index();
  DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(val_type, w_t, [&] {
      AdamVecOptimizer<g_t, w_t> opt{lr, beta1, beta2, eps, weight_decay,
                                     iter_num};
      launch_update_kernel_for_padded_buffer<g_t, w_t, decltype(opt)>(
          get_pointer<g_t>(grads), get_pointer<w_t>(values), opt, num_rows,
          grad_stride, value_stride, emb_dim_u32, all_dims_vec4, device_id);
    });
  });
}

void adagrad_update_for_padded_buffer(at::Tensor grads, at::Tensor values,
                                      int64_t emb_dim, int64_t value_dim,
                                      float lr, float eps) {
  int64_t num_rows = grads.size(0);
  uint32_t grad_stride = grads.size(1);
  if (num_rows == 0)
    return;
  TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");
  TORCH_CHECK(values.is_cuda(), "values must be a CUDA tensor");
  uint32_t emb_dim_u32 = static_cast<uint32_t>(emb_dim);
  uint32_t value_stride = static_cast<uint32_t>(value_dim);
  bool all_dims_vec4 = (emb_dim % 4 == 0);
  auto grad_type = get_data_type(grads);
  auto val_type = get_data_type(values);
  int device_id = grads.device().index();
  DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(val_type, w_t, [&] {
      AdaGradVecOptimizer<g_t, w_t> opt{lr, eps};
      launch_update_kernel_for_padded_buffer<g_t, w_t, decltype(opt)>(
          get_pointer<g_t>(grads), get_pointer<w_t>(values), opt, num_rows,
          grad_stride, value_stride, emb_dim_u32, all_dims_vec4, device_id);
    });
  });
}

void rowwise_adagrad_for_padded_buffer(at::Tensor grads, at::Tensor values,
                                       int64_t emb_dim, int64_t value_dim,
                                       float lr, float eps) {
  int64_t num_rows = grads.size(0);
  uint32_t grad_stride = grads.size(1);
  if (num_rows == 0)
    return;
  TORCH_CHECK(grads.is_cuda(), "grads must be a CUDA tensor");
  TORCH_CHECK(values.is_cuda(), "values must be a CUDA tensor");
  uint32_t emb_dim_u32 = static_cast<uint32_t>(emb_dim);
  uint32_t value_stride = static_cast<uint32_t>(value_dim);
  bool all_dims_vec4 = (emb_dim % 4 == 0);
  auto grad_type = get_data_type(grads);
  auto val_type = get_data_type(values);
  int device_id = grads.device().index();
  DISPATCH_FLOAT_DATATYPE_FUNCTION(grad_type, g_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(val_type, w_t, [&] {
      RowWiseAdaGradVecOptimizer<g_t, w_t> opt{lr, eps};
      launch_update_kernel_for_padded_buffer<g_t, w_t, decltype(opt)>(
          get_pointer<g_t>(grads), get_pointer<w_t>(values), opt, num_rows,
          grad_stride, value_stride, emb_dim_u32, all_dims_vec4, device_id,
          [](int block_size) { return block_size * sizeof(float); });
    });
  });
}

} // namespace dyn_emb

// PYTHON WRAP
void bind_optimizer_kernel_op(py::module &m) {
  m.def("sgd_update_for_flat_table", &dyn_emb::sgd_update_for_flat_table,
        "SGD optimizer for multi-table buffer via table_ptrs", py::arg("grads"),
        py::arg("indices"), py::arg("table_ptrs"), py::arg("table_ids"),
        py::arg("table_value_dims"), py::arg("table_emb_dims"),
        py::arg("max_emb_dim"), py::arg("all_dims_vec4"), py::arg("lr"),
        py::arg("table_dtype"));

  m.def("adam_update_for_flat_table", &dyn_emb::adam_update_for_flat_table,
        "Adam optimizer for multi-table buffer via table_ptrs", py::arg("grads"),
        py::arg("indices"), py::arg("table_ptrs"), py::arg("table_ids"),
        py::arg("table_value_dims"), py::arg("table_emb_dims"), py::arg("lr"),
        py::arg("beta1"), py::arg("beta2"), py::arg("eps"),
        py::arg("weight_decay"), py::arg("iter_num"), py::arg("max_emb_dim"),
        py::arg("all_dims_vec4"), py::arg("table_dtype"));

  m.def("adagrad_update_for_flat_table",
        &dyn_emb::adagrad_update_for_flat_table,
        "Adagrad optimizer for multi-table buffer via table_ptrs", py::arg("grads"),
        py::arg("indices"), py::arg("table_ptrs"), py::arg("table_ids"),
        py::arg("table_value_dims"), py::arg("table_emb_dims"), py::arg("lr"),
        py::arg("eps"), py::arg("max_emb_dim"), py::arg("all_dims_vec4"),
        py::arg("table_dtype"));

  m.def("rowwise_adagrad_for_flat_table",
        &dyn_emb::rowwise_adagrad_for_flat_table,
        "Row Wise Adagrad optimizer for multi-table buffer via table_ptrs",
        py::arg("grads"), py::arg("indices"), py::arg("table_ptrs"),
        py::arg("table_ids"),
        py::arg("table_value_dims"),
        py::arg("table_emb_dims"), py::arg("lr"), py::arg("eps"),
        py::arg("max_emb_dim"), py::arg("all_dims_vec4"),
        py::arg("table_dtype"));

  m.def("sgd_update_for_padded_buffer", &dyn_emb::sgd_update_for_padded_buffer,
        "SGD optimizer for contiguous padded buffer", py::arg("grads"),
        py::arg("values"), py::arg("emb_dim"), py::arg("value_dim"),
        py::arg("lr"));

  m.def("adam_update_for_padded_buffer",
        &dyn_emb::adam_update_for_padded_buffer,
        "Adam optimizer for contiguous padded buffer", py::arg("grads"),
        py::arg("values"), py::arg("emb_dim"), py::arg("value_dim"),
        py::arg("lr"), py::arg("beta1"), py::arg("beta2"), py::arg("eps"),
        py::arg("weight_decay"), py::arg("iter_num"));

  m.def("adagrad_update_for_padded_buffer",
        &dyn_emb::adagrad_update_for_padded_buffer,
        "Adagrad optimizer for contiguous padded buffer", py::arg("grads"),
        py::arg("values"), py::arg("emb_dim"), py::arg("value_dim"),
        py::arg("lr"), py::arg("eps"));

  m.def("rowwise_adagrad_for_padded_buffer",
        &dyn_emb::rowwise_adagrad_for_padded_buffer,
        "Row Wise Adagrad optimizer for contiguous padded buffer",
        py::arg("grads"), py::arg("values"), py::arg("emb_dim"),
        py::arg("value_dim"), py::arg("lr"), py::arg("eps"));
}
