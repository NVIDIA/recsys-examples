# HSTU Inference

## Key Features

1. Asynchronous Cache Manager for KV data

We use GPU memory and host storage for KV data cache as in `AsyncKVCacheManager`. This can help to reduce the recomputation of KV data. All the kvcache related operations are implemented as asynchronous, in order to hide the overhead with inference computation.

The GPU KV cache is organized as a paged KV-data table, and supports KV data adding/appending, lookup and eviction. When appending new data to the GPU cache, we will evict data from the oldest users according to the LRU policy if there is no empty page. The HSTU attention kernel also accepts KV data from a paged table.

The host KV data storage support adding/appending and lookup. We only present an example implementation, since this can be built over other database and can vary widely in the deployment.

2. Asynchronous H2D transfer of host KV data 

By using asynchronous data copy on the side CUDA stream, we overlap the host-to-device KV data transfer with HSTU computation layer-wise, to reduce the latency of HSTU inference.


3. Optimization with CUDA graph

We utilize the graph capture and replay support in Torch for convenient CUDA graph optimization on the HSTU layers. This decreases the overhead for kernel launch, especially for input with a small batch size. The input data (hidden states) fed to HSTU layers needs paddding to pre-determined batch size and sequence length, due to the requirement of static shape in CUDA graph.

4. Kernel fusion

5. Serving HSTU model with Triton Server Python backend

Currently we use the python backend to load and serve hstu models. The hstu model consists of two parts -- the sparse module and the dense module.
The sparse module is served as one instance per node, in which we create a set of gpu embedding tables or caches for each gpu sharing the same PS on the local host node or remote. (NVEmbedding backend only. To get access to NVEmbedding project please contact us.)
The dense module is served as one instance per GPU, and the KV cache is not supported for now.


## KVCache Manager for Inference

### KVCache Usage

1. KVCache Manager supports the following operations:
* `prepare_kvcache_async`: to trigger the allocation for required KV cache pages, kvcache_metadata computation, and onload the KV data from Host KV storage to GPU KVCache in background.
* `prepare_kvcache_wait`: to wait the new KV cache pages allocation and kvcache_metadata computation.
* `paged_kvcache_ops.append_kvcache`: the cuda kernel to copy the `K, V` values into the allocated cache pages.
* `offload_kvcache`: to trigger offloading the KV data from GPU KVCache to Host KV storage in background.
* `evict_kv_cache`: to evict all the KV data in the KVCache Manager.

2. Currently, the KVCache manager need to be access from a single inference stream. No multi-stream support.

3. The KVCache manager accepts full user history sequence as input. The removal of cached tokens in sequences is completed within inference forward pass.


## How to Setup

1. Install the dependencies for Recsys-Examples.

Turn on option `INFERENCEBUILD=1` to skip Megatron installation, which is not required for inference.

```bash
~$ cd ${WORKING_DIR}
~$ git clone --recursive -b ${TEST_BRANCH} ${TEST_REPO} recsys-examples && cd recsys-examples
~$ docker build \
    --platform linux/amd64 \
    --build-arg INFERENCEBUILD=1 \
    -t recsys-examples:inference \
    -f docker/Dockerfile .
```

## Example: Kuairand-1K

```
~$ cd recsys-examples/examples/hstu
~$ export PYTHONPATH=${PYTHONPATH}:$(realpath ../)
~$ 
~$ # Proprocess the dataset for inference:
~$ python3 ../commons/hstu_data_preprocessor.py --dataset_name "kuairand-1k" --inference
~$
~$ # Run the inference example
~$ python3 ./inference/inference_gr_ranking.py --gin_config_file ./inference/configs/kuairand_1k_inference_ranking.gin --checkpoint_dir ${PATH_TO_CHECKPOINT}  --mode eval
```

## Consistency Check for Inference

Currently, we use the evaluation metrics results (e.g. AUC) to check the consistency between training and inference.

1. Evaluation metrics from training

* Add evaluation output in training configs. Make sure `max_train_iters` is a multiple of `max_train_iters`.

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
/workspace/recsys-examples$ PYTHONPATH=${PYTHONPATH}:$(realpath ../) torchrun --nproc_per_node 1 --master_addr localhost --master_port 6000 ./training/pretrain_gr_ranking.py --gin-config-file ./training/configs/kuairand_1k_ranking.gin
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
/workspace/recsys-examples$ PYTHONPATH=${PYTHONPATH}:$(realpath ../) python3 ./inference/inference_gr_ranking.py --gin_config_file ./inference/configs/kuairand_1k_inference_ranking.gin --checkpoint_dir ${PATH_TO_CHECKPOINT} --mode eval
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

## Example: HSTU Model Inference with Triton Server
1. Build the docker image for triton server serving HSTU model
```
~/recsys-examples$ docker build \
    --build-arg BASE_IMAGE=nvcr.io/nvidia/tritonserver:25.06-py3 \
    --build-arg INFERENCEBUILD=1 \
    --build-arg TRITONSERVER_BUILD=1 \
    -f docker/Dockerfile \
    -t recsys-examples:inference_tritonserver .
```

2. Launch the triton server 

The triton server reads the model config from `inference/triton/hstu_model/config.pbtxt` and `inference/triton/hstu_sparse/config.pbtxt`.
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

For development, launch the triton server in the interactive container as following:
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

3. Launch the hstu container and inference with triton client
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
/workspace/recsys-examples$ # check triton server readiness. note: HTTP Status Code 200 for readiness
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

## Serve HSTU model using triton server with NVEmbedding

- **[Important]** Get the NVEmbedding repository

**The NVEmbedding project is open-source limited to authorized customers only**. Please contact Nvidia DevTech (APAC) Team for further discussion.

- Build the server image
```
~$ cd ${NVEMBEDDING_DIR}
~/nve$ git submodule update --init
~/nve$ docker build \
    --build-arg DEVEL_IMAGE=recsys-examples:inference_tritonserver \
    -f ${RECSYS_DIR}/docker/Dockerfile.nve \
    -t recsys-examples:inference_tritonserver .
```

- Launch the triton server 
```
~/recsys-examples$ cd ${PATH_TO_RECSYS_EXAMPLES}
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

- Launch the hstu container and install triton client (showcase in the interactive model)
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

- Inference with KuaiRank-1k Dataset
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

Note: The NVEmbedding backend may provide different default embedding values for unseen tokens. Use the training_dataset for validation
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
