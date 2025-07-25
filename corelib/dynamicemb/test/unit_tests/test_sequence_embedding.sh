# Test sequence embedding's forward on 2 GPUs with different [use_index_dedup, dim, output_dtype, dynamicemb_num, multi_hot_sizes]
set -e
export PYHKV_DEBUG=1
export PYHKV_DEBUG_ITER=10

# CUDA_VISIBLE_DEVICES=0,1 torchrun \
#   --nnodes 1 \
#   --nproc_per_node 2 \
#   ./test/unit_tests/test_multi_embedding_collection_fw.py --print_sharding_plan --optimizer_type "adam" --use_index_dedup True

# --log-dir=/Workspace/develop/recsys-examples/corelib/dynamicemb/log \
  # --redirect 3 \

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nnodes 1 \
  --nproc_per_node 2 \
  ./test/unit_tests/test_sequence_embedding_fw.py --print_sharding_plan --optimizer_type "adam" --use_index_dedup True --caching

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nnodes 1 \
  --nproc_per_node 2 \
  ./test/unit_tests/test_sequence_embedding_fw.py --print_sharding_plan --optimizer_type "adam" --use_index_dedup False --caching

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nnodes 1 \
  --nproc_per_node 2 \
  ./test/unit_tests/test_sequence_embedding_fw.py --print_sharding_plan --optimizer_type "adam" --use_index_dedup True --embedding_dim 129 --caching

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nnodes 1 \
  --nproc_per_node 2 \
  ./test/unit_tests/test_sequence_embedding_fw.py --print_sharding_plan --optimizer_type "adam" --use_index_dedup False --embedding_dim 129 --caching

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nnodes 1 \
  --nproc_per_node 2 \
  ./test/unit_tests/test_sequence_embedding_fw.py --print_sharding_plan --optimizer_type "adam" --use_index_dedup False --output_dtype=bf16 --caching

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nnodes 1 \
  --nproc_per_node 2 \
  ./test/unit_tests/test_sequence_embedding_fw.py --print_sharding_plan --optimizer_type "adam" --use_index_dedup False --dynamicemb_num=4 --caching

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nnodes 1 \
  --nproc_per_node 2 \
  ./test/unit_tests/test_sequence_embedding_fw.py --print_sharding_plan --optimizer_type "adam" --use_index_dedup False --multi_hot_sizes=20,0,101,49 --caching

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nnodes 1 \
  --nproc_per_node 2 \
  ./test/unit_tests/test_sequence_embedding_fw.py --print_sharding_plan --optimizer_type "adam" --use_index_dedup False --multi_hot_sizes=20,49,101,0 --caching

# CUDA_VISIBLE_DEVICES=0,1 torchrun \
#   --nnodes 1 \
#   --nproc_per_node 2 \
#   ./test/unit_tests/test_sequence_embedding_fw.py --print_sharding_plan --optimizer_type "adam" --use_index_dedup True

# CUDA_VISIBLE_DEVICES=0,1 torchrun \
#   --nnodes 1 \
#   --nproc_per_node 2 \
#   ./test/unit_tests/test_sequence_embedding_fw.py --print_sharding_plan --optimizer_type "adam" --use_index_dedup False

# CUDA_VISIBLE_DEVICES=0,1 torchrun \
#   --nnodes 1 \
#   --nproc_per_node 2 \
#   ./test/unit_tests/test_sequence_embedding_fw.py --print_sharding_plan --optimizer_type "adam" --use_index_dedup True --embedding_dim 129

# CUDA_VISIBLE_DEVICES=0,1 torchrun \
#   --nnodes 1 \
#   --nproc_per_node 2 \
#   ./test/unit_tests/test_sequence_embedding_fw.py --print_sharding_plan --optimizer_type "adam" --use_index_dedup False --embedding_dim 129

# CUDA_VISIBLE_DEVICES=0,1 torchrun \
#   --nnodes 1 \
#   --nproc_per_node 2 \
#   ./test/unit_tests/test_sequence_embedding_fw.py --print_sharding_plan --optimizer_type "adam" --use_index_dedup False --output_dtype=bf16

# CUDA_VISIBLE_DEVICES=0,1 torchrun \
#   --nnodes 1 \
#   --nproc_per_node 2 \
#   ./test/unit_tests/test_sequence_embedding_fw.py --print_sharding_plan --optimizer_type "adam" --use_index_dedup False --dynamicemb_num=4

# CUDA_VISIBLE_DEVICES=0,1 torchrun \
#   --nnodes 1 \
#   --nproc_per_node 2 \
#   ./test/unit_tests/test_sequence_embedding_fw.py --print_sharding_plan --optimizer_type "adam" --use_index_dedup False --multi_hot_sizes=20,0,101,49

# CUDA_VISIBLE_DEVICES=0,1 torchrun \
#   --nnodes 1 \
#   --nproc_per_node 2 \
#   ./test/unit_tests/test_sequence_embedding_fw.py --print_sharding_plan --optimizer_type "adam" --use_index_dedup False --multi_hot_sizes=20,49,101,0

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
#   --nnodes 1 \
#   --nproc_per_node 4 \
#   ./test/unit_tests/test_sequence_embedding_fw.py --print_sharding_plan --optimizer_type "adam" --use_index_dedup True --batch_size 1024 --num_embeddings_per_feature=8388608,4194304,524288,1048576

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
#   --nnodes 1 \
#   --nproc_per_node 8 \
#   ./test/unit_tests/test_sequence_embedding_fw.py --print_sharding_plan --optimizer_type "adam" --use_index_dedup True --batch_size 1024 --num_embeddings_per_feature=8388608,4194304,524288,1048576

# Test sequence embedding's backward on a single GPU with different ["use_index_dedup", "dim", "batch_size", "multi_hot_sizes"]
# CUDA_VISIBLE_DEVICES=0 torchrun \
#   --nnodes 1 \
#   --nproc_per_node 1 \
#   ./test/unit_tests/test_sequence_embedding_bw.py  --use_index_dedup True --batch_size 1024

# CUDA_VISIBLE_DEVICES=0 torchrun \
#   --nnodes 1 \
#   --nproc_per_node 1 \
#   ./test/unit_tests/test_sequence_embedding_bw.py  --use_index_dedup False --batch_size 1024

# CUDA_VISIBLE_DEVICES=0 torchrun \
#   --nnodes 1 \
#   --nproc_per_node 1 \
#   ./test/unit_tests/test_sequence_embedding_bw.py --embedding_dim 129 --use_index_dedup True --batch_size 1024

# CUDA_VISIBLE_DEVICES=0 torchrun \
#   --nnodes 1 \
#   --nproc_per_node 1 \
#   ./test/unit_tests/test_sequence_embedding_bw.py --embedding_dim 129 --use_index_dedup False --batch_size 1024

# CUDA_VISIBLE_DEVICES=0 torchrun \
#   --nnodes 1 \
#   --nproc_per_node 1 \
#   ./test/unit_tests/test_sequence_embedding_bw.py --embedding_dim 15 --use_index_dedup True --batch_size 1023 --multi_hot_sizes=20,1,101,49

# CUDA_VISIBLE_DEVICES=0 torchrun \
#   --nnodes 1 \
#   --nproc_per_node 1 \
#   ./test/unit_tests/test_sequence_embedding_bw.py --embedding_dim 15 --use_index_dedup False --batch_size 1023  --multi_hot_sizes=20,1,101,49

# CUDA_VISIBLE_DEVICES=0 torchrun \
#   --nnodes 1 \
#   --nproc_per_node 1 \
#   ./test/unit_tests/test_sequence_embedding_bw.py --embedding_dim 15 --use_index_dedup False --batch_size 1023  --multi_hot_sizes=20,49,101,1

  CUDA_VISIBLE_DEVICES=0 torchrun \
  --nnodes 1 \
  --nproc_per_node 1 \
  ./test/unit_tests/test_sequence_embedding_bw.py  --use_index_dedup True --batch_size 1024 --caching

CUDA_VISIBLE_DEVICES=0 torchrun \
  --nnodes 1 \
  --nproc_per_node 1 \
  ./test/unit_tests/test_sequence_embedding_bw.py  --use_index_dedup False --batch_size 1024 --caching

CUDA_VISIBLE_DEVICES=0 torchrun \
  --nnodes 1 \
  --nproc_per_node 1 \
  ./test/unit_tests/test_sequence_embedding_bw.py --embedding_dim 129 --use_index_dedup True --batch_size 1024 --caching

CUDA_VISIBLE_DEVICES=0 torchrun \
  --nnodes 1 \
  --nproc_per_node 1 \
  ./test/unit_tests/test_sequence_embedding_bw.py --embedding_dim 129 --use_index_dedup False --batch_size 1024 --caching

CUDA_VISIBLE_DEVICES=0 torchrun \
  --nnodes 1 \
  --nproc_per_node 1 \
  ./test/unit_tests/test_sequence_embedding_bw.py --embedding_dim 15 --use_index_dedup True --batch_size 1023 --multi_hot_sizes=20,1,101,49 --caching

CUDA_VISIBLE_DEVICES=0 torchrun \
  --nnodes 1 \
  --nproc_per_node 1 \
  ./test/unit_tests/test_sequence_embedding_bw.py --embedding_dim 15 --use_index_dedup False --batch_size 1023  --multi_hot_sizes=20,1,101,49 --caching

CUDA_VISIBLE_DEVICES=0 torchrun \
  --nnodes 1 \
  --nproc_per_node 1 \
  ./test/unit_tests/test_sequence_embedding_bw.py --embedding_dim 15 --use_index_dedup False --batch_size 1023  --multi_hot_sizes=20,49,101,1 --caching
