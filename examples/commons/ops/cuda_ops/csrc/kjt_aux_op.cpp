#include <ATen/ATen.h>
#include <torch/library.h>
#include <tuple>
#include <vector>

namespace {

std::vector<at::Tensor> split_by_lengths_impl(
    const at::Tensor& values,
    const at::Tensor& lengths_1d,
    int64_t num_splits) {
  // Compute split sizes on CPU so we can robustly support:
  // - values on CPU with lengths on CPU/CUDA
  // - values on CUDA with lengths on CPU/CUDA
  const int64_t batch = lengths_1d.numel() / num_splits;
  at::Tensor lengths_i64_cpu = lengths_1d.to(at::kCPU, at::kLong).reshape({num_splits, batch});
  at::Tensor per_split_sizes_cpu = lengths_i64_cpu.sum(/*dim=*/1); // [num_splits] on CPU

    const int64_t total_values = values.size(0);
  const int64_t total_split_sizes = per_split_sizes_cpu.sum().item<int64_t>();
  TORCH_CHECK(
      total_split_sizes == total_values,
      "sum(lengths_1d)=", total_split_sizes, " must equal values.size(0)=", total_values);

  std::vector<at::Tensor> out;
  out.reserve(static_cast<size_t>(num_splits));

  int64_t start = 0;
  for (int64_t i = 0; i < num_splits; ++i) {
    const int64_t len = per_split_sizes_cpu[i].item<int64_t>();
    out.push_back(values.narrow(/*dim=*/0, /*start=*/start, /*length=*/len));
    start += len;
  }

  return out;
}

std::vector<at::Tensor> split_by_lengths_cpu(
    const at::Tensor& values,
    const at::Tensor& lengths_1d,
    int64_t num_splits) {
  TORCH_CHECK(values.device().is_cpu(), "values must be a CPU tensor");
  TORCH_CHECK(
      lengths_1d.device().is_cpu() || lengths_1d.is_cuda(),
      "lengths_1d must be CPU or CUDA, got ", lengths_1d.device());
  TORCH_CHECK(
      values.dim() == 1 || values.dim() == 2,
      "values must be 1D or 2D, got dim=", values.dim());
  TORCH_CHECK(lengths_1d.dim() == 1, "lengths_1d must be 1D, got dim=", lengths_1d.dim());
  TORCH_CHECK(num_splits > 0, "num_splits must be > 0");
  TORCH_CHECK(
      lengths_1d.numel() % num_splits == 0,
      "lengths_1d.numel()=", lengths_1d.numel(), " must be divisible by num_splits=", num_splits);

  return split_by_lengths_impl(values, lengths_1d, num_splits);
}

std::vector<at::Tensor> split_by_lengths_cuda(
    const at::Tensor& values,
    const at::Tensor& lengths_1d,
    int64_t num_splits) {
  TORCH_CHECK(values.device().is_cuda(), "values must be a CUDA tensor");
  TORCH_CHECK(
      lengths_1d.device().is_cpu() || lengths_1d.is_cuda(),
      "lengths_1d must be CPU or CUDA, got ", lengths_1d.device());
  TORCH_CHECK(
      values.dim() == 1 || values.dim() == 2,
      "values must be 1D or 2D, got dim=", values.dim());
  TORCH_CHECK(lengths_1d.dim() == 1, "lengths_1d must be 1D, got dim=", lengths_1d.dim());
  TORCH_CHECK(num_splits > 0, "num_splits must be > 0");
  TORCH_CHECK(
      lengths_1d.numel() % num_splits == 0,
      "lengths_1d.numel()=", lengths_1d.numel(), " must be divisible by num_splits=", num_splits);

  return split_by_lengths_impl(values, lengths_1d, num_splits);
}

at::Tensor lengths_reduce_dim1_impl(
    const at::Tensor& lengths_1d,
    int64_t num_splits,
    bool expect_cuda) {
  TORCH_CHECK(lengths_1d.dim() == 1, "lengths_1d must be 1D, got dim=", lengths_1d.dim());
  TORCH_CHECK(num_splits > 0, "num_splits must be > 0");
  TORCH_CHECK(
      lengths_1d.numel() % num_splits == 0,
      "lengths_1d.numel()=", lengths_1d.numel(), " must be divisible by num_splits=", num_splits);

  if (expect_cuda) {
    TORCH_CHECK(lengths_1d.is_cuda(), "lengths_1d must be a CUDA tensor for CUDA impl");
  } else {
    TORCH_CHECK(lengths_1d.device().is_cpu(), "lengths_1d must be a CPU tensor for CPU impl");
  }

  const int64_t batch = lengths_1d.numel() / num_splits;
  return lengths_1d.view({num_splits, batch}).sum(1);
}

at::Tensor lengths_reduce_dim1_cpu(const at::Tensor& lengths_1d, int64_t num_splits) {
  return lengths_reduce_dim1_impl(lengths_1d, num_splits, /*expect_cuda=*/false);
}

at::Tensor lengths_reduce_dim1_cuda(const at::Tensor& lengths_1d, int64_t num_splits) {
  return lengths_reduce_dim1_impl(lengths_1d, num_splits, /*expect_cuda=*/true);
}

std::vector<at::Tensor> lengths_splits_impl(
    const at::Tensor& lengths_1d,
    int64_t num_splits,
    bool expect_cuda) {
  TORCH_CHECK(lengths_1d.dim() == 1, "lengths_1d must be 1D, got dim=", lengths_1d.dim());
  TORCH_CHECK(num_splits > 0, "num_splits must be > 0");
  TORCH_CHECK(
      lengths_1d.numel() % num_splits == 0,
      "lengths_1d.numel()=", lengths_1d.numel(), " must be divisible by num_splits=", num_splits);

  if (expect_cuda) {
    TORCH_CHECK(lengths_1d.is_cuda(), "lengths_1d must be a CUDA tensor for CUDA impl");
  } else {
    TORCH_CHECK(lengths_1d.device().is_cpu(), "lengths_1d must be a CPU tensor for CPU impl");
  }

  const int64_t batch = lengths_1d.numel() / num_splits;
  std::vector<at::Tensor> out;
  out.reserve(static_cast<size_t>(num_splits));

  for (int64_t i = 0; i < num_splits; ++i) {
    out.push_back(lengths_1d.narrow(/*dim=*/0, /*start=*/i*batch, /*length=*/batch));
  }

  return out;
}

std::vector<at::Tensor> lengths_splits_cpu(const at::Tensor& lengths_1d, int64_t num_splits) {
  return lengths_splits_impl(lengths_1d, num_splits, /*expect_cuda=*/false);
}

std::vector<at::Tensor> lengths_splits_cuda(const at::Tensor& lengths_1d, int64_t num_splits) {
  return lengths_splits_impl(lengths_1d, num_splits, /*expect_cuda=*/true);
}


std::vector<at::Tensor> permute_and_split_impl(
    const at::Tensor& jagged_features,
    const at::Tensor& jagged_lengths,
    const at::Tensor& jagged_offsets,
    int64_t num_static_features,
    int64_t num_dynamic_features,
    const std::vector<int64_t>& features_order,
    bool expect_cuda) {
  int64_t num_features = num_static_features + num_dynamic_features;
  TORCH_CHECK(jagged_features.dim() == 1, "jagged_features must be 1D, got dim=", jagged_features.dim());
  TORCH_CHECK(jagged_lengths.dim() == 1, "jagged_lengths must be 1D, got dim=", jagged_lengths.dim());
  TORCH_CHECK(num_static_features > 0, "num_static_features must be > 0");
  TORCH_CHECK(num_dynamic_features > 0, "num_dynamic_features must be > 0");
  TORCH_CHECK(
      jagged_lengths.numel() % num_features == 0,
      "jagged_lengths.numel()=", jagged_lengths.numel(), " must be divisible by num_features=", num_features);

  if (expect_cuda) {
    TORCH_CHECK(jagged_features.is_cuda(), "jagged_features must be a CUDA tensor for CUDA impl");
    TORCH_CHECK(jagged_lengths.is_cuda(), "jagged_lengths must be a CUDA tensor for CUDA impl");
  } else {
    TORCH_CHECK(jagged_features.device().is_cpu(), "jagged_features must be a CPU tensor for CPU impl");
    TORCH_CHECK(jagged_lengths.device().is_cpu(), "jagged_lengths must be a CPU tensor for CPU impl");
  }
  TORCH_CHECK(num_features == features_order.size(), "features_order size must match total number of features");

  const int64_t batch = jagged_lengths.numel() / num_features;

  std::vector<at::Tensor> permuted_lengths_vec(num_features);
  for (int64_t i = 0; i < num_features; ++i) {
    permuted_lengths_vec[i] = jagged_lengths.narrow(/*dim=*/0, /*start=*/features_order[i] * batch, /*length=*/batch);
  }
  auto permuted_lengths = at::cat(permuted_lengths_vec, /*dim=*/0);

  std::vector<at::Tensor> static_features_vec(num_static_features);
  std::vector<at::Tensor> dynamic_features_vec(num_dynamic_features);

  auto jagged_offsets_cpu = jagged_offsets.to(at::kCPU);

  for (int64_t i = 0; i < num_static_features; ++i) {
    auto numel = permuted_lengths_vec[i].sum().item<int64_t>();
    auto original_index = features_order[i] * batch;
    static_features_vec[i] = jagged_features.narrow(/*dim=*/0, /*start=*/jagged_offsets_cpu[original_index].item<int64_t>(), /*length=*/numel);
  }

  for (int64_t i = 0; i < num_dynamic_features; ++i) {
    auto numel = permuted_lengths_vec[num_static_features + i].sum().item<int64_t>();
    auto original_index = features_order[num_static_features + i] * batch;
    dynamic_features_vec[i] = jagged_features.narrow(/*dim=*/0, /*start=*/jagged_offsets_cpu[original_index].item<int64_t>(), /*length=*/numel);
  }

  auto static_lengths = permuted_lengths.narrow(/*dim=*/0, /*start=*/0, /*length=*/batch * num_static_features);
  auto dynamic_lengths = permuted_lengths.narrow(/*dim=*/0, /*start=*/batch * num_static_features, /*length=*/batch * num_dynamic_features);

  auto static_features = at::cat(static_features_vec, /*dim=*/0);
  auto dynamic_features = at::cat(dynamic_features_vec, /*dim=*/0);

  std::vector<at::Tensor> out{ static_features, dynamic_features, static_lengths, dynamic_lengths };
  return out;
}

std::vector<at::Tensor> permute_and_split_cpu(
    const at::Tensor& jagged_features,
    const at::Tensor& jagged_lengths,
    const at::Tensor& jagged_offsets,
    int64_t num_static_features,
    int64_t num_dynamic_features,
    const std::vector<int64_t>& features_order
) {
  return permute_and_split_impl(jagged_features, jagged_lengths, jagged_offsets, num_static_features, num_dynamic_features, features_order, /*expect_cuda=*/false);
}

std::vector<at::Tensor> permute_and_split_cuda(
    const at::Tensor& jagged_features,
    const at::Tensor& jagged_lengths,
    const at::Tensor& jagged_offsets,
    int64_t num_static_features,
    int64_t num_dynamic_features,
    const std::vector<int64_t>& features_order
) {
  return permute_and_split_impl(jagged_features, jagged_lengths, jagged_offsets, num_static_features, num_dynamic_features, features_order, /*expect_cuda=*/true);
}

std::tuple<at::Tensor, at::Tensor> strip_cached_tokens_impl_v2(
    const at::Tensor& values,
    const at::Tensor& lengths,
    const at::Tensor& length_offsets,
    const at::Tensor& num_cached,
    const std::vector<int64_t>& feature_order,
    bool expect_cuda) {
  const int64_t num_strip_features = feature_order.size();

  TORCH_CHECK(values.dim() == 1, "values must be 1D, got dim=", values.dim());
  TORCH_CHECK(lengths.dim() == 1, "lengths must be 1D, got dim=", lengths.dim());
  TORCH_CHECK(length_offsets.dim() == 1, "length_offsets must be 1D, got dim=", length_offsets.dim());
  TORCH_CHECK(num_cached.dim() == 1, "num_cached must be 1D, got dim=", num_cached.dim());
  TORCH_CHECK(feature_order.size() >= 2, "feature_order must have at least item and action features");
  const int64_t batch_size = num_cached.numel();
  TORCH_CHECK(batch_size > 0, "batch_size must be > 0");
  TORCH_CHECK(
      lengths.numel() == batch_size * num_strip_features,
      "lengths must have shape [batch_size * feature_order.size()]");
  TORCH_CHECK(
      length_offsets.numel() == lengths.numel() + 1,
      "length_offsets must have shape [lengths.numel() + 1]");
  TORCH_CHECK(lengths.device() == values.device(), "lengths must be on the same device as values");
  TORCH_CHECK(length_offsets.device() == values.device(), "length_offsets must be on the same device as values");
  TORCH_CHECK(num_cached.device() == values.device(), "num_cached must be on the same device as values");

  if (expect_cuda) {
    TORCH_CHECK(values.is_cuda(), "values must be a CUDA tensor for CUDA impl");
    TORCH_CHECK(lengths.is_cuda(), "lengths must be a CUDA tensor for CUDA impl");
    TORCH_CHECK(length_offsets.is_cuda(), "length_offsets must be a CUDA tensor for CUDA impl");
    TORCH_CHECK(num_cached.is_cuda(), "num_cached must be a CUDA tensor for CUDA impl");
  } else {
    TORCH_CHECK(values.device().is_cpu(), "values must be a CPU tensor for CPU impl");
    TORCH_CHECK(lengths.device().is_cpu(), "lengths must be a CPU tensor for CPU impl");
    TORCH_CHECK(length_offsets.device().is_cpu(), "length_offsets must be a CPU tensor for CPU impl");
    TORCH_CHECK(num_cached.device().is_cpu(), "num_cached must be a CPU tensor for CPU impl");
  }

  auto lengths_i64 = lengths.to(at::kLong);
  auto offsets_i64 = length_offsets.to(at::kLong);
  auto num_cached_i64 = num_cached.to(at::kLong);
  for (const auto feature_idx : feature_order) {
    TORCH_CHECK(feature_idx >= 0 && feature_idx < num_strip_features, "feature_order contains invalid index: ", feature_idx);
  }

  auto batch_indices = at::arange(batch_size, lengths_i64.options());
  auto remaining_num_cached = num_cached_i64.clone();
  auto strip_counts = at::zeros_like(lengths_i64);
  std::vector<at::Tensor> selected_rows_vec;
  std::vector<at::Tensor> clipped_num_cached_vec;
  std::vector<at::Tensor> new_lengths_vec;
  selected_rows_vec.reserve(static_cast<size_t>(num_strip_features));
  clipped_num_cached_vec.reserve(static_cast<size_t>(num_strip_features));
  new_lengths_vec.reserve(static_cast<size_t>(num_strip_features));

  for (int64_t feature_order_idx = 0; feature_order_idx < num_strip_features - 2; ++feature_order_idx) {
    const auto feature_idx = feature_order[feature_order_idx];
    auto selected_feature_rows = feature_idx * batch_size + batch_indices;
    auto feature_lengths = lengths_i64.index_select(0, selected_feature_rows);
    auto clipped_num_cached = at::minimum(remaining_num_cached, feature_lengths);
    remaining_num_cached = remaining_num_cached - clipped_num_cached;
    strip_counts.narrow(/*dim=*/0, /*start=*/feature_idx * batch_size, /*length=*/batch_size).copy_(clipped_num_cached);
  }

  const auto item_feature_idx = feature_order[num_strip_features - 2];
  const auto action_feature_idx = feature_order[num_strip_features - 1];
  auto item_rows = item_feature_idx * batch_size + batch_indices;
  auto action_rows = action_feature_idx * batch_size + batch_indices;
  auto item_lengths = lengths_i64.index_select(0, item_rows);
  auto action_lengths = lengths_i64.index_select(0, action_rows);
  auto item_num_cached = at::floor_divide(remaining_num_cached + 1, 2);
  auto action_num_cached = at::floor_divide(remaining_num_cached, 2);
  auto item_strip_counts = at::minimum(item_num_cached, item_lengths);
  auto action_strip_counts = at::minimum(action_num_cached, action_lengths);
  strip_counts.narrow(/*dim=*/0, /*start=*/item_feature_idx * batch_size, /*length=*/batch_size).copy_(item_strip_counts);
  strip_counts.narrow(/*dim=*/0, /*start=*/action_feature_idx * batch_size, /*length=*/batch_size).copy_(action_strip_counts);

  for (const auto feature_idx : feature_order) {
    auto selected_feature_rows = feature_idx * batch_size + batch_indices;
    auto feature_lengths = lengths_i64.index_select(0, selected_feature_rows);
    auto clipped_num_cached = strip_counts.index_select(0, selected_feature_rows);
    selected_rows_vec.push_back(selected_feature_rows);
    clipped_num_cached_vec.push_back(clipped_num_cached);
    new_lengths_vec.push_back(feature_lengths - clipped_num_cached);
  }

  auto selected_rows = at::cat(selected_rows_vec, 0);
  auto clipped_num_cached = at::cat(clipped_num_cached_vec, 0);
  auto new_lengths_i64 = at::cat(new_lengths_vec, 0);

  auto zero = at::zeros({1}, new_lengths_i64.options());
  auto new_offsets = at::cat({zero, at::cumsum(new_lengths_i64, 0)}, 0);
  auto source_starts = offsets_i64.index_select(0, selected_rows) + clipped_num_cached;
  auto source_delta = source_starts - new_offsets.narrow(0, 0, lengths_i64.numel());
  auto repeated_delta = at::repeat_interleave(source_delta, new_lengths_i64);
  auto source_indices = at::arange(repeated_delta.numel(), repeated_delta.options()) + repeated_delta;
  return std::make_tuple(values.index_select(0, source_indices), new_lengths_i64.to(lengths.scalar_type()));
}

std::tuple<at::Tensor, at::Tensor> strip_cached_tokens_cpu(
    const at::Tensor& values,
    const at::Tensor& lengths,
    const at::Tensor& length_offsets,
    const at::Tensor& num_cached,
    const std::vector<int64_t>& feature_order) {
  return strip_cached_tokens_impl_v2(values, lengths, length_offsets, num_cached, feature_order, /*expect_cuda=*/false);
}

std::tuple<at::Tensor, at::Tensor> strip_cached_tokens_cuda(
    const at::Tensor& values,
    const at::Tensor& lengths,
    const at::Tensor& length_offsets,
    const at::Tensor& num_cached,
    const std::vector<int64_t>& feature_order) {
  return strip_cached_tokens_impl_v2(values, lengths, length_offsets, num_cached, feature_order, /*expect_cuda=*/true);
}

} // namespace

TORCH_LIBRARY_FRAGMENT(hstu_cuda_ops, m) {
  m.def("split_by_lengths(Tensor values, Tensor lengths_1d, int num_splits) -> Tensor[]");
  m.def("lengths_reduce_dim1(Tensor lengths_1d, int num_splits) -> Tensor");
  m.def("lengths_splits(Tensor lengths_1d, int num_splits) -> Tensor[]");
  m.def("permute_and_split(Tensor jagged_features, Tensor jagged_lengths, Tensor jagged_offsets, int num_static_features, int num_dynamic_features, int[] features_order) -> Tensor[]");
  m.def("strip_cached_tokens(Tensor values, Tensor lengths, Tensor length_offsets, Tensor num_cached, int[] feature_order) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(hstu_cuda_ops, CPU, m) {
  m.impl("split_by_lengths", split_by_lengths_cpu);
  m.impl("lengths_reduce_dim1", lengths_reduce_dim1_cpu);
  m.impl("lengths_splits", lengths_splits_cpu);
  m.impl("permute_and_split", permute_and_split_cpu);
  m.impl("strip_cached_tokens", strip_cached_tokens_cpu);
}

TORCH_LIBRARY_IMPL(hstu_cuda_ops, CUDA, m) {
  m.impl("split_by_lengths", split_by_lengths_cuda);
  m.impl("lengths_reduce_dim1", lengths_reduce_dim1_cuda);
  m.impl("lengths_splits", lengths_splits_cuda);
  m.impl("permute_and_split", permute_and_split_cuda);
  m.impl("strip_cached_tokens", strip_cached_tokens_cuda);
}