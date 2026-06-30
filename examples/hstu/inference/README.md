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

```
~$ cd recsys-examples/examples/hstu
~$ export PYTHONPATH=${PYTHONPATH}:$(realpath ../)
~$ 
~$ # Preprocess the dataset for inference:
~$ python3 ../commons/hstu_data_preprocessor.py --dataset_name "kuairand-1k" --inference
~$
~$ # Run the inference example
~$ python3 ./inference/inference_gr_ranking.py --gin_config_file ./inference/configs/kuairand_1k_inference_ranking.gin --checkpoint_dir ${PATH_TO_CHECKPOINT}  --mode eval
```

## Consistency Check for Inference

Currently, we use the evaluation metrics results (e.g. AUC) to check the consistency between training and inference.

1. Evaluation metrics from training

* Add evaluation output in training configs. Make sure `max_train_iters` is a multiple of `eval_interval`, and save a checkpoint at the iteration you want to evaluate.

```
# File: examples/hstu/training/configs/
...
TrainerArgs.eval_interval = 50
TrainerArgs.max_train_iters = 550
TrainerArgs.ckpt_save_interval = 550
...
```

* Get eval metrics from training
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

## Example: HSTU Model Inference with Triton Inference Server
1. Build the Docker image for Triton Inference Server serving HSTU model
```
~/recsys-examples$ docker build \
    --build-arg BASE_IMAGE=nvcr.io/nvidia/tritonserver:26.02-py3 \
    --build-arg TRITONSERVER_BUILD=1 \
    -f docker/Dockerfile \
    -t recsys-examples:inference_tritonserver .
```

2. Launch Triton Inference Server

Triton Inference Server reads the model config from `inference/triton/hstu_model/config.pbtxt` and `inference/triton/hstu_sparse/config.pbtxt`.
Setup `HSTU_GIN_CONFIG_FILE` and `HSTU_CHECKPOINT_DIR` in the config before launching the server.

```
~/recsys-examples$ docker run \
    --shm-size=8G --ulimit memlock=-1 -p 8000:8000 -p 8001:8001 -p 8002:8002 --ulimit stack=67108864 \
    --gpus \"device=$NV_GPU\" \
    --volume ${SRC_DIR}:${DST_DIR} \
    -w /workspace/recsys-examples/examples/hstu \
    --hostname $(hostname) \
    --name triton-server-hstu \
    -d -t recsys-examples:inference_tritonserver \
    bash -cx inference/launch_triton_server.sh ${PATH_TO_CHECKPOINT}
```

For development, launch Triton Inference Server in the interactive container as follows:
```
~/recsys-examples$ docker run \
    --shm-size=8G --ulimit memlock=-1 -p 8000:8000 -p 8001:8001 -p 8002:8002 --ulimit stack=67108864 \
    --gpus \"device=$NV_GPU\" \
    --volume ${SRC_DIR}:${DST_DIR} \
    --hostname $(hostname) \
    --name triton-server-hstu \
    -ti recsys-examples:inference_tritonserver
/workspace/recsys-examples$ cd /workspace/recsys-examples/examples/hstu
/workspace/recsys-examples/examples/hstu$ bash ./inference/launch_triton_server.sh ${PATH_TO_CHECKPOINT}
```

3. Launch the HSTU container and run inference with the Triton client
```
~/recsys-examples$ docker run \
    --rm --shm-size 8G --cap-add SYS_NICE --net host \
    --gpus \"device=$NV_GPU\" \
    --volume ${SRC_DIR}:${DST_DIR} \
    --hostname $(hostname) \
    --name triton-client-hstu \
    -ti recsys-examples:inference
```

Within the client container:
```
/workspace/recsys-examples$ # install triton client
/workspace/recsys-examples$ pip3 install tritonclient[all]
/workspace/recsys-examples$ 
/workspace/recsys-examples$ # check Triton server readiness. note: HTTP status code 200 for readiness
/workspace/recsys-examples$ curl -o /dev/null -s -w "%{http_code}\n" localhost:8000/v2/health/ready
200
/workspace/recsys-examples$ 
/workspace/recsys-examples$ cd /workspace/recsys-examples/examples/hstu
/workspace/recsys-examples/examples/hstu$ # start inference with pre-processed dataset
/workspace/recsys-examples/examples/hstu$ PYTHONPATH=${PYTHONPATH}:$(realpath ../) python3 ./inference/triton/hstu_model/client.py --gin_config_file ./inference/configs/kuairand_1k_inference_ranking.gin
...
[eval]:
    Metrics.task0.AUC: 0.556777
    Metrics.task1.AUC: 0.801971
    Metrics.task2.AUC: 0.599631
    Metrics.task3.AUC: 0.666604
    Metrics.task4.AUC: 0.558464
    Metrics.task5.AUC: 0.577246
    Metrics.task6.AUC: 0.620458
    Metrics.task7.AUC: 0.556104
```

## Serve HSTU model using Triton Inference Server with NVEmbedding

The main `docker/Dockerfile` installs [NVEmbedding](https://github.com/NVIDIA/nv-embedding-cache) when building the inference images, so a separate `Dockerfile.nve` step is no longer required.

- Build the Triton server image:
```
~/recsys-examples$ docker build \
    --build-arg BASE_IMAGE=nvcr.io/nvidia/tritonserver:26.02-py3 \
    --build-arg TRITONSERVER_BUILD=1 \
    -f docker/Dockerfile \
    -t recsys-examples:inference_tritonserver .
```

- Launch the Triton server with the NVEmbedding backend enabled:
```
~/recsys-examples$ docker run \
    --shm-size=8G --ulimit memlock=-1 -p 8000:8000 -p 8001:8001 -p 8002:8002 --ulimit stack=67108864 \
    --gpus \"device=$NV_GPU\" \
    --volume ${SRC_DIR}:${DST_DIR} \
    -w /workspace/recsys-examples/examples/hstu \
    --hostname $(hostname) \
    --name triton-server-hstu \
    -d -t recsys-examples:inference_tritonserver \
    bash -cx "printf '\nNetworkArgs.embedding_backend = \"NVEmb\"\n' >> ./inference/configs/kuairand_1k_inference_ranking.gin && bash inference/launch_triton_server.sh ${PATH_TO_CHECKPOINT}"
```

- Launch the HSTU client container and install the Triton client:
```
~/recsys-examples$ docker run \
    --rm --shm-size 8G --cap-add SYS_NICE --net host \
    --gpus \"device=$NV_GPU\" \
    --volume ${SRC_DIR}:${DST_DIR} \
    --hostname $(hostname) \
    --name triton-client-hstu \
    -ti recsys-examples:inference
/workspace/recsys-examples$ pip3 install tritonclient[all]
```

- Run inference with the KuaiRand-1K dataset:
```
/workspace/recsys-examples/examples/hstu$ PYTHONPATH=${PYTHONPATH}:$(realpath ../) python3 ./inference/triton/hstu_model/client.py --gin_config_file ./inference/configs/kuairand_1k_inference_ranking.gin
...
[eval]:
    Metrics.task0.AUC: 0.568207
    Metrics.task1.AUC: 0.746528
    Metrics.task2.AUC: 0.618382
    Metrics.task3.AUC: 0.645711
    Metrics.task4.AUC: 0.529315
    Metrics.task5.AUC: 0.592606
    Metrics.task6.AUC: 0.581823
    Metrics.task7.AUC: 0.556803
```

Note: The NVEmbedding backend may provide different default embedding values for unseen tokens. Use the training dataset for validation:
```
/workspace/recsys-examples/examples/hstu$ PYTHONPATH=${PYTHONPATH}:$(realpath ../) python3 ./inference/triton/hstu_model/client.py --gin_config_file ./inference/configs/kuairand_1k_inference_ranking.gin --train_dataset
...
[eval]:
    Metrics.task0.AUC: 0.975089
    Metrics.task1.AUC: 0.986811
    Metrics.task2.AUC: 0.961327
    Metrics.task3.AUC: 0.961882
    Metrics.task4.AUC: 0.976655
    Metrics.task5.AUC: 0.985043
    Metrics.task6.AUC: 0.970641
    Metrics.task7.AUC: 0.960993
```
