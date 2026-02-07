import argparse
import builtins
import math
import os
import shutil
import urllib.request
import warnings
import zipfile
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from dynamicemb import (
    DynamicEmbDump,
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbLoad,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    FrequencyAdmissionStrategy,
    KVCounter,
)
from dynamicemb.dynamicemb_config import data_type_to_dtype, get_optimizer_state_dim
from dynamicemb.incremental_dump import get_score, incremental_dump
from dynamicemb.optimizer import EmbOptimType, convert_optimizer_type
from dynamicemb.planner import (
    DynamicEmbeddingEnumerator,
    DynamicEmbeddingShardingPlanner,
    DynamicEmbParameterConstraints,
)
from dynamicemb.shard import DynamicEmbeddingCollectionSharder
from fbgemm_gpu.split_embedding_configs import SparseType
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchrec import DataType
from torchrec.distributed.comm import get_local_rank, get_local_size
from torchrec.distributed.fbgemm_qcomm_codec import (
    CommType,
    QCommsConfig,
    get_qcomm_codecs_registry,
)
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import Topology
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)
from torchrec.distributed.planner.types import ShardingPlan
from torchrec.distributed.types import ShardingType
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

# Filter FBGEMM warning, make notebook clean
warnings.filterwarnings(
    "ignore", message=".*torch.library.impl_abstract.*", category=FutureWarning
)

backend = "nccl"
dist.init_process_group(backend=backend)

# Set LOCAL_WORLD_SIZE if not available for proper topology configuration
if "LOCAL_WORLD_SIZE" not in os.environ:
    os.environ["LOCAL_WORLD_SIZE"] = str(torch.cuda.device_count())

# Set LOCAL_RANK if not available (for consistency)
if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = str(get_local_rank())

# Set RANK if not available
if "RANK" not in os.environ:
    os.environ["RANK"] = str(dist.get_rank())

local_rank = dist.get_rank()  # for one node
world_size = dist.get_world_size()
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")
# print with rank info
original_print = builtins.print


def rank_print(*args, **kwargs):
    original_print(f"[RANK {local_rank}] ", *args, **kwargs)


builtins.print = rank_print
cache_ratio = 0.5  # assume we will use 50% of the HBM for cache


def download_movielens(data_dir="./ml-1m"):
    if dist.get_rank() == 0:  # Use global rank for multi-node consistency
        os.makedirs(data_dir, exist_ok=True)
        if os.path.exists(os.path.join(data_dir, "ratings.dat")):
            print(f"MovieLens in {data_dir}")
            return data_dir

        url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
        zip_path = os.path.join(data_dir, "ml-1m.zip")

        print(f"download MovieLens-1M...")
        urllib.request.urlretrieve(url, zip_path)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(os.path.dirname(data_dir))

        extracted_dir = os.path.join(os.path.dirname(data_dir), "ml-1m")
        if extracted_dir != data_dir:
            for file in os.listdir(extracted_dir):
                shutil.move(
                    os.path.join(extracted_dir, file), os.path.join(data_dir, file)
                )
            if os.path.exists(extracted_dir):
                shutil.rmtree(extracted_dir)

        os.remove(zip_path)

    return data_dir


def parse_args():
    parser = argparse.ArgumentParser(description="TorchRec MovieLens with dynamicemb")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--dump", action="store_true")
    parser.add_argument("--incremental_dump", action="store_true")
    parser.add_argument("--caching", action="store_true")
    parser.add_argument("--prefetch_pipeline", action="store_true")
    parser.add_argument("--external_storage", action="store_true")

    parser.add_argument(
        "--data_path",
        type=str,
        default="./ml-1m",
        help="path to dataset MovieLens，and will download if non-existed",
    )
    parser.add_argument("--epochs", type=int, default=5, help="training epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument(
        "--embedding_dim", type=int, default=64, help="embedding dimension"
    )
    parser.add_argument(
        "--num_embeddings", type=int, default=10000, help="number of embeddings"
    )
    parser.add_argument(
        "--mlp_dims",
        type=str,
        default="128,64,32",
        help="dimension of MLP layer，separating with commas",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./model_checkpoints",
        help="path to save the model",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed used for initialization"
    )
    # torchrun --standalone --nproc_per_node=${NGPU} example.py --train "$@" --admission_threshold 5
    parser.add_argument(
        "--admission_threshold",
        type=int,
        default=0,
        help="Frequency threshold for admission strategy (0 disable admission strategy, >0 enable admission strategy and only keys appearing >= threshold will be stored in tables)",
    )
    return parser.parse_args()


class MovieLensDataset(Dataset):
    def __init__(self, data_path: str, split: str = "train"):
        ratings_file = os.path.join(data_path, "ratings.dat")
        if not os.path.exists(ratings_file):
            raise FileNotFoundError(f": {ratings_file}")

        ratings_data = []
        with open(ratings_file, "r", encoding="ISO-8859-1") as f:
            for line in f:
                user_id, movie_id, rating, timestamp = line.strip().split("::")
                ratings_data.append(
                    {
                        "user_id": int(user_id),
                        "movie_id": int(movie_id),
                        "rating": float(rating),
                        "timestamp": int(timestamp),
                    }
                )

        ratings_df = pd.DataFrame(ratings_data)

        users_file = os.path.join(data_path, "users.dat")
        movies_file = os.path.join(data_path, "movies.dat")

        users_data = []
        with open(users_file, "r", encoding="ISO-8859-1") as f:
            for line in f:
                parts = line.strip().split("::")
                user_id = int(parts[0])
                gender = 1 if parts[1] == "M" else 0
                age = int(parts[2])
                occupation = int(parts[3])
                users_data.append(
                    {
                        "user_id": user_id,
                        "gender": gender,
                        "age": age,
                        "occupation": occupation,
                    }
                )

        users_df = pd.DataFrame(users_data)

        movies_data = []
        with open(movies_file, "r", encoding="ISO-8859-1") as f:
            for line in f:
                parts = line.strip().split("::")
                movie_id = int(parts[0])
                year = 0
                if parts[1].endswith(")"):
                    year_start = parts[1].rfind("(")
                    if year_start != -1:
                        year_str = parts[1][year_start + 1 : parts[1].rfind(")")]
                        try:
                            year = int(year_str)
                        except ValueError:
                            year = 0

                movies_data.append({"movie_id": movie_id, "year": year})

        movies_df = pd.DataFrame(movies_data)

        data = pd.merge(ratings_df, users_df, on="user_id", how="left")
        data = pd.merge(data, movies_df, on="movie_id", how="left")

        data = data.sort_values("timestamp")

        split_idx = int(len(data) * 0.8)
        if split == "train":
            self.data = data.iloc[:split_idx]
        else:
            self.data = data.iloc[split_idx:]

        self.max_user_id = data["user_id"].max()
        self.max_movie_id = data["movie_id"].max()
        self.max_age = data["age"].max()
        self.max_occupation = data["occupation"].max()

        print(f": {len(self.data)} ")
        print(f": {self.max_user_id}, : {self.max_movie_id}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        sparse_features = {
            "user_id": torch.tensor([row["user_id"]], dtype=torch.long),
            "movie_id": torch.tensor([row["movie_id"]], dtype=torch.long),
            "gender": torch.tensor([row["gender"]], dtype=torch.long),
            "age": torch.tensor([row["age"]], dtype=torch.long),
            "occupation": torch.tensor([row["occupation"]], dtype=torch.long),
            "year": torch.tensor([row["year"]], dtype=torch.long),
        }

        label = torch.tensor(row["rating"], dtype=torch.float)

        return sparse_features, label


def collate_fn(batch):
    sparse_features = {
        "user_id": [],
        "movie_id": [],
        "gender": [],
        "age": [],
        "occupation": [],
        "year": [],
    }

    labels = []

    for features, label in batch:
        for key in sparse_features:
            sparse_features[key].extend(features[key].tolist())
        labels.append(label)

    lengths = {
        "user_id": torch.tensor([1] * len(batch), dtype=torch.long),
        "movie_id": torch.tensor([1] * len(batch), dtype=torch.long),
        "gender": torch.tensor([1] * len(batch), dtype=torch.long),
        "age": torch.tensor([1] * len(batch), dtype=torch.long),
        "occupation": torch.tensor([1] * len(batch), dtype=torch.long),
        "year": torch.tensor([1] * len(batch), dtype=torch.long),
    }

    values = {
        "user_id": torch.tensor(sparse_features["user_id"], dtype=torch.long),
        "movie_id": torch.tensor(sparse_features["movie_id"], dtype=torch.long),
        "gender": torch.tensor(sparse_features["gender"], dtype=torch.long),
        "age": torch.tensor(sparse_features["age"], dtype=torch.long),
        "occupation": torch.tensor(sparse_features["occupation"], dtype=torch.long),
        "year": torch.tensor(sparse_features["year"], dtype=torch.long),
    }

    kjt = KeyedJaggedTensor(
        keys=list(values.keys()),
        values=torch.cat([values[k] for k in values.keys()]),
        lengths=torch.cat([lengths[k] for k in lengths.keys()]),
    )

    return kjt, torch.tensor(labels, dtype=torch.float)


class MovieLensModel(nn.Module):
    def __init__(
        self,
        embedding_module: EmbeddingCollection,
        dense_in_features: int,
        dense_arch_layer_sizes: List[int],
        over_arch_layer_sizes: List[int],
    ):
        super().__init__()
        self.embedding_module = embedding_module

        dense_arch_layers = []
        for i in range(len(dense_arch_layer_sizes) - 1):
            dense_arch_layers.append(
                nn.Linear(dense_arch_layer_sizes[i], dense_arch_layer_sizes[i + 1])
            )
            dense_arch_layers.append(nn.ReLU())
        self.dense_arch = nn.Sequential(*dense_arch_layers)

        embedding_dim = embedding_module.embedding_configs()[0].embedding_dim
        for config in embedding_module.embedding_configs():
            assert embedding_dim == config.embedding_dim

        over_arch_layers = []
        if dense_in_features == 0:
            input_dim = embedding_dim
        else:
            input_dim = dense_arch_layer_sizes[-1] + embedding_dim

        over_arch_layers.append(
            nn.Linear(
                input_dim,
                over_arch_layer_sizes[0],
            )
        )
        over_arch_layers.append(nn.ReLU())
        for i in range(len(over_arch_layer_sizes) - 1):
            over_arch_layers.append(
                nn.Linear(over_arch_layer_sizes[i], over_arch_layer_sizes[i + 1])
            )
            over_arch_layers.append(nn.ReLU())
        over_arch_layers.append(nn.Linear(over_arch_layer_sizes[-1], 1))
        self.over_arch = nn.Sequential(*over_arch_layers)

    def forward(self, kjt: KeyedJaggedTensor) -> torch.Tensor:
        embeddings = self.embedding_module(kjt)

        sparse_features = torch.cat(
            [embeddings[k].values() for k in embeddings.keys()], dim=0
        )

        prediction = self.over_arch(sparse_features)
        num_features = len(kjt.keys())
        batch = len(kjt.lengths()) // num_features
        hotness = 1
        # batch_size x hotness(1) x num_feature
        x = prediction.view(hotness * num_features, batch)
        return torch.sum(x.t(), dim=-1)


def get_sharder(args, optimizer_type):
    # set optimizer args
    learning_rate = args.lr
    beta1 = 0.9
    beta2 = 0.999
    weight_decay = 0
    eps = 0.001

    # Put args into a optimizer kwargs , which is same usage of torchrec
    optimizer_kwargs = {
        "optimizer": optimizer_type,
        "learning_rate": learning_rate,
        "beta1": beta1,
        "beta2": beta2,
        "weight_decay": weight_decay,
        "eps": eps,
    }

    fused_params = {}
    fused_params[
        "output_dtype"
    ] = (
        SparseType.FP32
    )  # data type of the output after lookup, and can differ from the stored.
    fused_params.update(optimizer_kwargs)
    fused_params[
        "prefetch_pipeline"
    ] = args.prefetch_pipeline  # whether enable prefetch for embedding lookup module

    # precision of all-to-all
    qcomm_codecs_registry = (
        get_qcomm_codecs_registry(
            qcomms_config=QCommsConfig(
                # pyre-ignore
                forward_precision=CommType.FP32,
                # pyre-ignore
                backward_precision=CommType.FP32,
            )
        )
        if backend == "nccl"
        else None
    )

    """
    fused_params: 
        items in fused_params will be finally passed to embedding lookup module. But before that:  
            logic tables in `EmbeddingCollection` will be divided into multiple groups in the `ShardedDynamicEmbeddingCollection`, 
            and the fused_params are equal for tables in the same group. 
        However, we only provide the common for all tables here, but some fields in `DynamicEmbTableOptions` will be merged into fused_params 
            and then be used to group tables(please refer DynamicEmbTableOptions for more details).
        **Performance** issue: Embedding lookup within the same group can be executed in parallel, 
            while embedding lookup between different groups can only be executed sequentially.
    use_index_dedup: 
        Unlike `EmbeddingBagCollection`, there is no reduction operation at the jagged dimension in the input `KeyedJaggedTensor` for `EmbeddingCollection`.
        Therefore, we can deduplicate the input's indices in the input distributor before sparse feature's all-to-all, 
            then it will reduce the bandwidth pressure of NVLink or PCIe when perform embedding's all-to-all, and restore them using inverse information finally.
    qcomm_codecs_registry: used to configure the embeddings(forward) or gradients(backward)' precision when perform all-to-all operation across different ranks 
        in distributed environment. 
    """
    return DynamicEmbeddingCollectionSharder(
        qcomm_codecs_registry=qcomm_codecs_registry,
        fused_params=fused_params,
        use_index_dedup=True,
    )


# use a function warp all the Planner code
def get_planner(
    device, eb_configs, batch_size, optimizer_type, training, caching, args
):
    DATA_TYPE_NUM_BITS: Dict[DataType, int] = {
        DataType.FP32: 32,
        DataType.FP16: 16,
        DataType.BF16: 16,
    }

    hbm_cap = 80 * 1024 * 1024 * 1024  # H100's HBM bytes per GPU
    ddr_cap = 512 * 1024 * 1024 * 1024  # Assume a Node have 512GB memory
    intra_host_bw = 450e9  # Nvlink bandwidth
    inter_host_bw = 25e9  # NIC bandwidth
    bucket_capacity = 1024 if caching else 128

    dict_const = {}

    for eb_config in eb_configs:
        # For HVK  embedding table, need to calculate how many bytes of embedding vector store in GPU HBM
        dim = eb_config.embedding_dim
        tmp_type = eb_config.data_type

        embedding_type_bytes = DATA_TYPE_NUM_BITS[tmp_type] / 8
        emb_num_embeddings = eb_config.num_embeddings
        emb_num_embeddings_next_power_of_2 = 2 ** math.ceil(
            math.log2(emb_num_embeddings)
        )  # hash table needs embedding vector num to be power of 2
        threshold = (bucket_capacity * world_size) / cache_ratio
        threshold_int = math.ceil(threshold)
        if emb_num_embeddings_next_power_of_2 < threshold_int:
            emb_num_embeddings_next_power_of_2 = 2 ** math.ceil(
                math.log2(threshold_int)
            )

        # e.g. for adam, its `x`` embedding + `2x`` optimizer states
        total_dim = dim + get_optimizer_state_dim(
            convert_optimizer_type(optimizer_type), dim, data_type_to_dtype(tmp_type)
        )
        total_hbm_need = (
            embedding_type_bytes * total_dim * emb_num_embeddings_next_power_of_2
        )

        # Setup admission strategy if threshold > 0
        admit_strategy = None
        admission_counter = None
        if args.admission_threshold > 0:
            print(
                f"Admission strategy enabled with threshold={args.admission_threshold}"
            )
            # Create counter config (actual table will be created during sharding)
            admission_counter = KVCounter(
                capacity=emb_num_embeddings_next_power_of_2,
                bucket_capacity=bucket_capacity,
                key_type=torch.int64,
            )

            # Create admission strategy with threshold
            admit_strategy = FrequencyAdmissionStrategy(
                threshold=args.admission_threshold,
                initializer_args=DynamicEmbInitializerArgs(
                    mode=DynamicEmbInitializerMode.CONSTANT,
                    value=0.0,  # Initialize rejected keys to 0
                ),
            )

        const = DynamicEmbParameterConstraints(
            sharding_types=[
                ShardingType.ROW_WISE.value,  # dynamicemb embedding table only support to be sharded in row-wise.
            ],
            use_dynamicemb=True,  # indicate using dynamicemb, and will fallback to raw ParameterConstraints when Fale.
            dynamicemb_options=DynamicEmbTableOptions(
                global_hbm_for_values=total_hbm_need * cache_ratio
                if caching
                else total_hbm_need,
                initializer_args=DynamicEmbInitializerArgs(
                    mode=DynamicEmbInitializerMode.NORMAL
                ),
                score_strategy=DynamicEmbScoreStrategy.STEP,
                caching=caching,
                training=training,
                admit_strategy=admit_strategy,
                admission_counter=admission_counter,
            ),
        )

        dict_const[eb_config.name] = const

    topology = Topology(
        local_world_size=get_local_size(),
        world_size=dist.get_world_size(),
        compute_device=device.type,
        hbm_cap=hbm_cap,
        ddr_cap=ddr_cap,
        intra_host_bw=intra_host_bw,
        inter_host_bw=inter_host_bw,
    )

    # same usage of  torchrec's EmbeddingEnumerator
    enumerator = DynamicEmbeddingEnumerator(
        topology=topology,
        constraints=dict_const,
    )

    # Almost same usage of  torchrec's EmbeddingShardingPlanner, except to input eb_configs,
    #   as dynamicemb need EmbeddingConfig info to help to plan.
    return DynamicEmbeddingShardingPlanner(
        eb_configs=eb_configs,
        topology=topology,
        constraints=dict_const,
        batch_size=batch_size,
        enumerator=enumerator,
        storage_reservation=HeuristicalStorageReservation(percentage=0.05),
        debug=True,
    )


def apply_dmp(model, args, training):
    """
    The initialization of embedding lookup module in dynamicemb is almost consistent with torchrec.
        1. Firstly, you should configure the global parameters of an embedding table using `EmbeddingCollection`.
        2. Then, build a `DynamicEmbeddingCollectionSharder`, and generate `ShardingPlan` from `DynamicEmbeddingShardingPlanner`.
        3. Finally, pass all parameters to the `DistributedModelParallel`, which then handles the embedding sharding and initialization.
    """
    eb_configs = model.embedding_module.embedding_configs()
    optimizer_type = EmbOptimType.ADAM

    """
    After configuring the `EmbeddingCollection`, you need to configure `DynamicEmbeddingCollectionSharder`. 
    It can create an instance of `ShardedDynamicEmbeddingCollection`.
    `ShardedDynamicEmbeddingCollection` provides customized embedding lookup module based on 
        a GPU-optimized scored hash table which can utilize both device and host memory,
        support automatic eviction based on score(per key) while providing better performance.
    Besides, due to differences in deduplication between hash tables and array based static tables, 
        `ShardedDynamicEmbeddingCollection` also provide customized input distributor to support deduplication when `use_index_dedup=True`.
    The actual sharding operation occurs during the initialization of the `ShardedDynamicEmbeddingCollection`, 
        but the parameters used to initialize `DynamicEmbeddingCollectionSharder`  will play a key role in the sharding process.
    By the way, `DynamicEmbeddingCollectionSharder` inherits `EmbeddingCollectionSharder`, 
        and its main job is return an instance of `ShardedDynamicEmbeddingCollection`.
    """
    sharder = get_sharder(args, optimizer_type)

    """
    The next step of preparation is to generate a `ParameterSharding` for each table, describe (configure) the sharding of a parameter. 
    For dynamic embedding table, `DynamicEmbParameterSharding` will be generated, which includes the parameters required from our embedding lookup module.
    We will not expand `DynamicEmbParameterSharding` here. 
    The following steps demonstrate how to obtain `DynamicEmbParameterSharding` by `DynamicEmbeddingShardingPlanner`.
    """
    planner = get_planner(
        device,
        eb_configs,
        args.batch_size,
        optimizer_type=optimizer_type,
        training=training,
        caching=args.caching,
        args=args,
    )
    # get plan for all ranks.
    # ShardingPlan is a dict, mapping table name to ParameterSharding/DynamicEmbParameterSharding.
    plan: ShardingPlan = planner.collective_plan(
        model, [sharder], dist.GroupMember.WORLD
    )

    """
    The final step is to input the `sharder` and `ShardingPlan` to the `DistributedModelParallel`, 
        who will implement the sharded plan through `sharder` and hold the `ShardedDynamicEmbeddingCollection` after sharding.
    Then you can use `dmp` for **training** and **evaluation**, just like using `EmbeddingCollection`.
    """
    dmp = DistributedModelParallel(
        module=model,
        device=device,
        # pyre-ignore
        sharders=[sharder],
        plan=plan,
    )
    return dmp


def create_model(args, training=True):
    # Define the configuration parameters for the embedding table,
    # including its name, embedding dimension, total number of embeddings, and feature name.
    eb_configs = [
        EmbeddingConfig(
            name="user_id",
            embedding_dim=args.embedding_dim,
            num_embeddings=args.num_embeddings,  # `num_embeddings` in `EmbeddingConfig` is the sum of all slices on all GPUs for a table.
            feature_names=[
                "user_id"
            ],  # a list, means different features can share the same table
            data_type=DataType.FP32,  # weight or embedding's data type.
        ),
        EmbeddingConfig(
            name="movie_id",
            embedding_dim=args.embedding_dim,
            num_embeddings=args.num_embeddings,
            feature_names=["movie_id"],
        ),
        EmbeddingConfig(
            name="gender",
            embedding_dim=args.embedding_dim,
            num_embeddings=2,
            feature_names=["gender"],
        ),
        EmbeddingConfig(
            name="age",
            embedding_dim=args.embedding_dim,
            num_embeddings=100,
            feature_names=["age"],
        ),
        EmbeddingConfig(
            name="occupation",
            embedding_dim=args.embedding_dim,
            num_embeddings=50,
            feature_names=["occupation"],
        ),
        EmbeddingConfig(
            name="year",
            embedding_dim=args.embedding_dim,
            num_embeddings=2050,
            feature_names=["year"],
        ),
    ]

    """
    `EmbeddingCollection` is a collection of multiple logical tables.
    It does not allocate memory for embedding tables(device is "meta").
    """
    ec = EmbeddingCollection(
        tables=eb_configs,
        device=torch.device("meta"),  # set device to 'meta
    )

    mlp_dims = [int(dim) for dim in args.mlp_dims.split(",")]

    model = MovieLensModel(
        embedding_module=ec,
        dense_in_features=0,
        dense_arch_layer_sizes=[1, 1],  # placeholder
        over_arch_layer_sizes=mlp_dims,
    )

    model = apply_dmp(model, args, training)

    return model


def train_one_epoch(model, train_loader, optimizer, loss_fn, epoch, total_epochs):
    model.train()
    total_loss = 0

    for batch_idx, (features, labels) in enumerate(train_loader):
        features = features.to(device)
        labels = labels.to(device)

        outputs = model(features)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(
                f"Epoch {epoch+1}/{total_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}"
            )

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{total_epochs}, Average Loss: {avg_loss:.4f}")


def test_one_epoch(model, test_loader, loss_fn, epoch, total_epochs):
    model.eval()
    test_loss = 0
    with torch.inference_mode():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"Epoch {epoch+1}/{total_epochs}, Test Loss: {avg_test_loss:.4f}")


def train(args):
    train_dataset = MovieLensDataset(args.data_path, split="train")
    test_dataset = MovieLensDataset(args.data_path, split="test")
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=dist.get_rank(), shuffle=True
    )
    test_sampler = DistributedSampler(
        test_dataset, num_replicas=world_size, rank=dist.get_rank(), shuffle=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        sampler=train_sampler,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        sampler=test_sampler,
    )

    model = create_model(args)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        train_one_epoch(model, train_loader, optimizer, criterion, epoch, args.epochs)
        test_one_epoch(model, test_loader, criterion, epoch, args.epochs)


def dump(args):
    os.makedirs(args.save_dir, exist_ok=True)
    train_dataset = MovieLensDataset(args.data_path, split="train")
    # Use global rank for proper data distribution across all processes
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=dist.get_rank(), shuffle=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        sampler=train_sampler,
    )

    model = create_model(args)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        train_one_epoch(model, train_loader, optimizer, criterion, epoch, args.epochs)

        # ShardedDyanmicEmbeddingCollection.state_dict() will return a dummy tensor.
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            os.path.join(
                args.save_dir, f"model_epoch_{epoch+1}_rank{dist.get_rank()}.pt"
            ),
        )
    DynamicEmbDump(os.path.join(args.save_dir, "dynamicemb"), model, optim=True)


def load(args):
    os.makedirs(args.save_dir, exist_ok=True)
    test_dataset = MovieLensDataset(args.data_path, split="test")
    # Use global rank for proper data distribution across all processes
    test_sampler = DistributedSampler(
        test_dataset, num_replicas=world_size, rank=dist.get_rank(), shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        sampler=test_sampler,
    )

    model = create_model(args)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # load
    checkpoint = torch.load(
        os.path.join(
            args.save_dir, f"model_epoch_{args.epochs}_rank{dist.get_rank()}.pt"
        ),
        weights_only=True,
    )
    # Must set strict to False, as there is no embedding's weight in model.state_dict()
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    DynamicEmbLoad(os.path.join(args.save_dir, "dynamicemb"), model, optim=True)

    test_one_epoch(model, test_loader, criterion, 0, 1)

    dist.barrier(device_ids=[local_rank])
    # Only global rank 0 should clean up, not local rank 0 on each node
    if dist.get_rank() == 0:
        try:
            shutil.rmtree(args.save_dir)
        except Exception as e:
            print(f"Warning: Failed to remove {args.save_dir}: {e}")
    dist.barrier(device_ids=[local_rank])


def inc_dump(args):
    os.makedirs(args.save_dir, exist_ok=True)
    train_dataset = MovieLensDataset(args.data_path, split="train")
    # Use global rank for proper data distribution across all processes
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=dist.get_rank(), shuffle=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        sampler=train_sampler,
    )

    model = create_model(args)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    undumped_score = get_score(model)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0

        for batch_idx, (features, labels) in enumerate(train_loader):
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                # reset undumped_score here.
                ret_tensors, undumped_score = incremental_dump(model, undumped_score)
                dump_number = 0
                for module_path, named_tensors in ret_tensors.items():
                    for (
                        table_name,
                        tensors,
                    ) in (
                        named_tensors.items()
                    ):  # tensors[0] and tensors[1] are keys and values.
                        dump_number += tensors[0].size(0)
                print(
                    f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, dump number: {dump_number}"
                )

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.4f}")


def main():
    args = parse_args()
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    if dist.get_rank() == 0:  # Use global rank for multi-node consistency
        download_movielens(args.data_path)
    dist.barrier(device_ids=[local_rank])
    if args.train:
        train(args)
    if args.dump:
        dump(args)
    if args.load:
        load(args)
    if args.incremental_dump:
        inc_dump(args)


if __name__ == "__main__":
    main()

dist.destroy_process_group()
