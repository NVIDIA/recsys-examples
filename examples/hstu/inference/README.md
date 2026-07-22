# HSTU Inference

## Key Features

1. KV Cache Manager

`KVCacheManager` from [`recsys_kvcache_manager`](../../../corelib/recsys_kvcache_manager/README.md) uses GPU memory and host storage for KV-data caches. This reduces KV-data recomputation, and KV-cache-related operations are asynchronous so their overhead can overlap with inference computation.

The GPU KV cache is organized as a paged KV-data table and supports KV-data add/append, lookup, and eviction. When appending new data to the GPU cache, it evicts data from the oldest users according to the LRU policy if there is no empty page. The HSTU attention kernel also accepts KV data from a paged table.

The host KV-data storage supports add/append and lookup. `recsys_kvcache_manager` integrates a `native` backend using pinned host memory only, and a `flexkv` backend based on [`FlexKV`](https://github.com/taco-project/FlexKV/tree/main).

For its API and usage, please see [README.md](../../../corelib/recsys_kvcache_manager/README.md).

2. Asynchronous H2D transfer of host KV data

By using asynchronous data copy on the side CUDA stream, we overlap the host-to-device KV data transfer with HSTU computation layer-wise, to reduce the latency of HSTU inference.
**Note** this feature is only enabled with the `native` backend in the KV-cache manager.

3. Optimization with CUDA graph

We utilize the graph capture and replay support in Torch for convenient CUDA graph optimization on the HSTU layers. This decreases the overhead for kernel launch, especially for input with a small batch size. The input data (hidden states) fed to HSTU layers needs padding to pre-determined batch size and sequence length, due to the requirement of static shape in CUDA graph.

4. Kernel fusion

5. Serving HSTU model with Triton Inference Server Python backend

The Triton path uses the Python backend to load and serve HSTU models. The model consists of a sparse module and a dense module.
With the NVEmbedding backend, the sparse module creates GPU embedding tables or caches for each GPU while sharing the same local parameter-store data. The dense module is served as one instance per GPU. The current Triton dense path uses `forward_nokvcache`; KV-cache inference is available in the standalone Python inference path.

6. End-to-end C++ inference with Torch Export and AOTInductor, covering inference with kvcache (based on `corelib/recsys_kvcache_manager`)

We support end-to-end C++ inference from a PyTorch Python model based on `torch.export` and `torch._inductor.aoti_compile_and_package`.

For the embedding part, our implementation is based on `InferenceEmbeddingTable` from DynamicEmb, using DynamicEmb `ScoredHashTable` and NVEmbedding layers. NVEmbedding implements customized `export_and_aot`, which generates layer metadata and dumped embedding table files together with the model `.pt2` archive, in order to avoid multiple copies of embedding table while loading. The structure of the complete exported and packaged model archive is:

```
path/to/model_archive
        ├── model.pt2                              # AOT-compiled model package for AOTIModelPackageLoader
        ├── metadata.json                          # NVE layer metadata (id, num_embeddings, emb_size, etc.)
        └── weights/{emb_layer_module_name}.nve    # NVE weight data (LinearUVM)
```

Start with the [guide](./GUIDE_TO_RUN_CPP_INFERENCE_DEMO.md) for HSTU Python to C++ inference example.

## How to Setup

1. Install the dependencies for Recsys Examples.

```bash
~$ cd ${WORKING_DIR}
~$ git clone --recursive -b ${TEST_BRANCH} ${TEST_REPO} recsys-examples && cd recsys-examples
~$ docker build \
    --platform linux/amd64 \
    -t recsys-examples:inference \
    -f docker/Dockerfile .
```

## Example: Kuairand-1K

```bash
~$ cd recsys-examples/examples/hstu
~$ export PYTHONPATH=${PYTHONPATH}:$(realpath ../)
~$
~$ # Preprocess the dataset for inference:
~$ python3 ../commons/hstu_data_preprocessor.py --dataset_name "kuairand-1k" --inference
~$
~$ # Run the inference example
~$ python3 ./inference/inference_gr_ranking.py --gin_config_file ./inference/configs/kuairand_1k_inference_ranking.gin --checkpoint_dir ${PATH_TO_CHECKPOINT} --mode eval
```

### Launch and Test the Triton Python Backend

`launch_and_test_triton_python_backend.sh` prepares the HSTU model repository,
starts Triton, waits for the sparse and dense models, and runs the HTTP client
against both the evaluation and training datasets. The server and clients run
in the same container. When testing finishes, the script stops its Triton
process and restores the model configs, model-version directories, Gin config,
and checkpoint `ps_module` directory.

The workflow assumes an image built from this repository's Dockerfile with:

- Triton Server and its Python backend installed under `/opt/tritonserver`.
- `tritonclient[http]` installed in the image.
- The repository available at `/workspace/recsys-examples`.
- One GPU visible in the container, with sufficient shared memory.

The client and server model configuration are fixed at batch size 2. The client
sends one unmeasured warmup batch and then performs three measured runs for
each dataset.

### Input paths

| Input | Default | How to override |
| --- | --- | --- |
| HSTU directory | `/workspace/recsys-examples/examples/hstu` | Set `HSTU_DIR` |
| Checkpoint | `$HSTU_DIR/ckpt/kuairand_1k_ckpt` | Pass argument 1 |
| Preprocessed dataset | `$HSTU_DIR/tmp_data` | Set `DatasetArgs.dataset_path` in the Gin config |
| Server log | `/tmp/hstu-python-backend-tritonserver.log` | Set `SERVER_LOG` |
| Readiness timeout | 300 seconds | Set `READY_TIMEOUT_SECONDS` |

The default checkpoint directory must contain `dynamicemb_module`. The script
uses its `user_id_emb_*` and `video_id_emb_*` files to construct the temporary
`ps_module` layout required by NVEmbedding.

When `DatasetArgs.dataset_path` is not set in
`inference/configs/kuairand_1k_inference_ranking.gin`, the dataset loader uses
`examples/hstu/tmp_data`. Create it from the repository checkout with:

```bash
cd /workspace/recsys-examples/examples/hstu
python3 ../commons/hstu_data_preprocessor.py \
  --dataset_name kuairand-1k \
  --inference
```

Mount or copy the checkpoint at the default location:

```text
/workspace/recsys-examples/examples/hstu/ckpt/kuairand_1k_ckpt
```

Then run the complete test inside the container:

```bash
cd /workspace/recsys-examples/examples/hstu
bash ./inference/launch_and_test_triton_python_backend.sh
```

To use a different checkpoint location, pass it as the only positional
argument:

```bash
bash ./inference/launch_and_test_triton_python_backend.sh \
  /checkpoints/kuairand_1k_ckpt
```

For a non-default repository location, set `HSTU_DIR` as well:

```bash
HSTU_DIR=/opt/recsys-examples/examples/hstu \
  bash /opt/recsys-examples/examples/hstu/inference/launch_and_test_triton_python_backend.sh \
  /checkpoints/kuairand_1k_ckpt
```

Show the path contract without launching Triton:

```bash
bash ./inference/launch_and_test_triton_python_backend.sh --help
```

## Consistency Check for Inference

Currently, we use the evaluation metrics results (e.g. AUC) to check the consistency between training and inference.

1. Evaluation metrics from training

- Add evaluation output in training configs. Make sure `max_train_iters` is a multiple of `eval_interval`, and save a checkpoint at the iteration you want to evaluate.

```
# File: examples/hstu/training/configs/
...
TrainerArgs.eval_interval = 50
TrainerArgs.max_train_iters = 550
TrainerArgs.ckpt_save_interval = 550
...
```

- Get eval metrics from training

```
/workspace/recsys-examples/examples/hstu$ PYTHONPATH=${PYTHONPATH}:$(realpath ../) torchrun --nproc_per_node 1 --master_addr localhost --master_port 6000 ./training/pretrain_gr_ranking.py --gin-config-file ./training/configs/kuairand_1k_ranking.gin
... [training output] ...
[eval] [eval 296 users]:
    Metrics.task0.AUC:0.557266
    Metrics.task1.AUC:0.801949
    Metrics.task2.AUC:0.599034
    Metrics.task3.AUC:0.666739
    Metrics.task4.AUC:0.555904
    Metrics.task5.AUC:0.582272
    Metrics.task6.AUC:0.620481
    Metrics.task7.AUC:0.556170
... [training output] ...
```

2. Evaluation metrics from inference

```
/workspace/recsys-examples/examples/hstu$ PYTHONPATH=${PYTHONPATH}:$(realpath ../) python3 ./inference/inference_gr_ranking.py --gin_config_file ./inference/configs/kuairand_1k_inference_ranking.gin --checkpoint_dir ${PATH_TO_CHECKPOINT} --mode eval
... [inference output] ...
[eval]:
    Metrics.task0.AUC:0.556894
    Metrics.task1.AUC:0.802019
    Metrics.task2.AUC:0.599779
    Metrics.task3.AUC:0.666891
    Metrics.task4.AUC:0.559471
    Metrics.task5.AUC:0.580227
    Metrics.task6.AUC:0.620498
    Metrics.task7.AUC:0.556064
... [inference output] ...
```
