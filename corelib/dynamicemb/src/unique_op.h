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

#ifndef UNIQUE_OP_H
#define UNIQUE_OP_H

#include "ATen/ATen.h"
#include "ATen/AccumulateType.h"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/cuda/DeviceUtils.cuh"
#include "unique_variable.h"
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <memory>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <torch/extension.h>
#include <torch/torch.h>

namespace dyn_emb {

class UniqueOpBase {
public:
  // virtual ~UniqueOpBase() = default;
  virtual void unique(const at::Tensor d_key, const uint64_t len,
                      at::Tensor d_output_index, at::Tensor d_unique_key,
                      at::Tensor d_output_counter, cudaStream_t stream = 0,
                      at::Tensor offset = at::Tensor(),
                      at::Tensor d_frequency_counters = at::Tensor(),
                      at::Tensor d_input_frequencies = at::Tensor()) = 0;

  virtual void reset_capacity(at::Tensor keys, at::Tensor vals,
                              const size_t capacity,
                              cudaStream_t stream = 0) = 0;

  virtual size_t get_capacity() = 0;
};

template <typename KeyType, typename CounterType>
class HashUniqueOp : public UniqueOpBase {
public:
  using UniqueOp_ = dyn_emb::unique_op<KeyType, CounterType,
                                       std::numeric_limits<KeyType>::max(),
                                       std::numeric_limits<CounterType>::max()>;
  HashUniqueOp(at::Tensor keys, at::Tensor vals, at::Tensor counter,
               const size_t capacity)
      : keys_(keys), vals_(vals), counter_(counter), capacity_(capacity) {
    this->unique_op_ = std::make_unique<UniqueOp_>(
        reinterpret_cast<KeyType *>(keys.data_ptr()),
        reinterpret_cast<CounterType *>(vals.data_ptr()),
        reinterpret_cast<CounterType *>(counter.data_ptr()), capacity);
  }
  void unique(const at::Tensor d_key, const uint64_t len,
              at::Tensor d_output_index, at::Tensor d_unique_key,
              at::Tensor d_output_counter, cudaStream_t stream = 0,
              at::Tensor offset = at::Tensor(),
              at::Tensor d_frequency_counters = at::Tensor(),
              at::Tensor d_input_frequencies =
                  at::Tensor()) override { /// TODO: dtype check in runtime.
    if (stream == 0) {
      stream = at::cuda::getCurrentCUDAStream().stream();
    }

    CounterType *offset_ptr = nullptr;
    if (offset.defined() && offset.numel() > 0) {
      // Check if offset is of the same type as CounterType
      if (offset.scalar_type() != at::CppTypeToScalarType<CounterType>::value) {
        throw std::runtime_error(
            "Offset tensor must have the same type as CounterType.");
      }
      offset_ptr = offset.data_ptr<CounterType>();
    }

    CounterType *frequency_counters_ptr = nullptr;
    if (d_frequency_counters.defined() && d_frequency_counters.numel() > 0) {
      // Check if frequency counters is of the same type as CounterType
      if (d_frequency_counters.scalar_type() !=
          at::CppTypeToScalarType<CounterType>::value) {
        throw std::runtime_error("Frequency counters tensor must have the same "
                                 "type as CounterType.");
      }
      frequency_counters_ptr = d_frequency_counters.data_ptr<CounterType>();
    }

    const CounterType *input_frequencies_ptr = nullptr;
    if (d_input_frequencies.defined() && d_input_frequencies.numel() > 0) {
      // Check if input frequencies is of the same type as CounterType
      if (d_input_frequencies.scalar_type() !=
          at::CppTypeToScalarType<CounterType>::value) {
        throw std::runtime_error(
            "Input frequencies tensor must have the same type as CounterType.");
      }
      input_frequencies_ptr = d_input_frequencies.data_ptr<CounterType>();
    }

    this->unique_op_->unique(
        reinterpret_cast<KeyType *>(d_key.data_ptr()), len,
        reinterpret_cast<CounterType *>(d_output_index.data_ptr()),
        reinterpret_cast<KeyType *>(d_unique_key.data_ptr()),
        reinterpret_cast<CounterType *>(d_output_counter.data_ptr()), stream,
        offset_ptr, frequency_counters_ptr, input_frequencies_ptr);
    this->unique_op_->clear(stream);
  }

  void reset_capacity(at::Tensor keys, at::Tensor vals, const size_t capacity,
                      cudaStream_t stream = 0) override {
    if (stream == 0) {
      stream = at::cuda::getCurrentCUDAStream().stream();
    }

    if (keys.scalar_type() != keys_.scalar_type() ||
        vals.scalar_type() != vals_.scalar_type()) {
      throw std::runtime_error("keys and vals must have the same type as the "
                               "original keys and vals.");
    }

    if (keys.size(0) != capacity || vals.size(0) != capacity) {
      throw std::runtime_error(
          "keys and vals must have the same length as the capacity.");
    }

    this->keys_ = keys;
    this->vals_ = vals;
    this->capacity_ = capacity;
    this->unique_op_->reset_capacity(
        reinterpret_cast<KeyType *>(this->keys_.data_ptr()),
        reinterpret_cast<CounterType *>(this->vals_.data_ptr()),
        this->capacity_, stream);
  }

  size_t get_capacity() override { return this->unique_op_->get_capacity(); }

private:
  std::unique_ptr<UniqueOp_> unique_op_;
  // Keep the reference counter.
  at::Tensor keys_;
  at::Tensor vals_;
  at::Tensor counter_;
  size_t capacity_;
};

class UniqueOpFactory {
public:
  static std::shared_ptr<dyn_emb::UniqueOpBase> create(at::Tensor keys,
                                                       at::Tensor vals,
                                                       at::Tensor counter,
                                                       const size_t capacity);
};

} // namespace dyn_emb

#endif // UNIQUE_OP_H
