# Introduction
This document consists of two parts, one is the introduction to the API, which can be used as a reference, and the other part prompts specific functions from a functional perspective and which interfaces are related to them.

# DynamicEmb APIs

## APIs
- [DynamicEmbParameterConstraints](#dynamicembparameterconstraints)
- [DynamicEmbeddingEnumerator](#dynamicembeddingenumerator)
- [DynamicEmbeddingShardingPlanner](#dynamicembeddingshardingplanner)
- [DynamicEmbeddingCollectionSharder](#dynamicembeddingcollectionsharder)
- [DynamicEmbCheckMode](#dynamicembcheckmode)
- [DynamicEmbInitializerMode](#dynamicembinitializermode)
- [DynamicEmbInitializerArgs](#dynamicembinitializerargs)
- [DynamicEmbPoolingMode](#dynamicembpoolingmode)
- [DynamicEmbTableOptions](#dynamicembtableoptions)
- [DynamicEmbDump](#dynamicembdump)
- [DynamicEmbLoad](#dynamicembload)
- [incremental_dump](#incremental_dump)
- [get_score](#get_score)
- [set_score](#set_score)
- [Counter](#counter)
- [AdmissionStrategy](#admisssion_strategy)

## DynamicEmbParameterConstraints

The `DynamicEmbParameterConstraints` function inherits from TorchREC's `ParameterConstraints` function. It has the same basic parameters as `ParameterConstraints` and adds members for specifying and configuring dynamic embedding tables. This function serves as the only entry point for users to specify whether an embedding table is a dynamic embedding table..

    ```python
    #How to import
    from dynamicemb.planner import DynamicEmbParameterConstraints

    #API arguments
    @dataclass
    class DynamicEmbParameterConstraints(ParameterConstraints):
        """
        DynamicEmb-specific parameter constraints that extend ParameterConstraints.

        Attributes
        ----------
        use_dynamicemb : Optional[bool]
            A flag indicating whether to use DynamicEmb storage. Defaults to False.
        dynamicemb_options : Optional[DynamicEmbTableOptions]
            Configuration for the dynamic embedding table, including initializer args.
            Common choices include "uniform", "normal", etc. Defaults to "uniform".
        """
        use_dynamicemb: Optional[bool] = False
        dynamicemb_options: Optional[DynamicEmbTableOptions] = DynamicEmbTableOptions()

    ```

## DynamicEmbeddingEnumerator

The `DynamicEmbeddingEnumerator` function inherits from TorchREC's `EmbeddingEnumerator` function and its usage is exactly the same as `EmbeddingEnumerator`. This class differentiates between TorchREC's embedding tables and dynamic embedding tables during enumeration in the sharding plan.

    ```python
    #How to import
    from dynamicemb.planner import DynamicEmbeddingEnumerator

    #API arguments
    class DynamicEmbeddingEnumerator(EmbeddingEnumerator):
    def __init__(
        self,
        topology: Topology,
        batch_size: Optional[int] = BATCH_SIZE,
        #TODO:check the input type is DynamicEmbParameterConstraints or ParameterConstraints
        constraints: Optional[Dict[str, DynamicEmbParameterConstraints]] = None,
        estimator: Optional[Union[ShardEstimator, List[ShardEstimator]]] = None,
    ) -> None:
        """
        DynamicEmbeddingEnumerator extends the EmbeddingEnumerator to handle dynamic embedding tables.

        Parameters
        ----------
        topology : Topology
            The topology of the GPU and Host memory.
        batch_size : Optional[int], optional
            The batch size for training. Defaults to BATCH_SIZE.
            The creation and usage are consistent with the same types in TorchREC.
        constraints : Optional[Dict[str, DynamicEmbParameterConstraints]], optional
            A dictionary of constraints for the parameters. Defaults to None.
        estimator : Optional[Union[ShardEstimator, List[ShardEstimator]]], optional
            An estimator or a list of estimators for estimating shard sizes. Defaults to None.
            The creation and usage are consistent with the same types in TorchREC.
        """
    ```

## DynamicEmbeddingShardingPlanner

Wrapped TorchREC's `EmbeddingShardingPlanner` to perform sharding for dynamic embedding tables. Unlike `EmbeddingShardingPlanner`, it requires an additional eb\_configs parameter to plan the capacity of dynamic embedding tables.

    ```python
    #How to import
    from dynamicemb.planner import DynamicEmbeddingShardingPlanner

    #API arguments
    class DynamicEmbeddingShardingPlanner:
    def __init__(self,
        eb_configs: List[BaseEmbeddingConfig],
        topology: Optional[Topology] = None,
        batch_size: Optional[int] = None,
        enumerator: Optional[Enumerator] = None,
        storage_reservation: Optional[StorageReservation] = None,
        proposer: Optional[Union[Proposer, List[Proposer]]] = None,
        partitioner: Optional[Partitioner] = None,
        performance_model: Optional[PerfModel] = None,
        stats: Optional[Union[Stats, List[Stats]]] = None,
        constraints: Optional[Dict[str, DynamicEmbParameterConstraints]] = None,
        debug: bool = True):

        """
        DynamicEmbeddingShardingPlanner wraps the API of EmbeddingShardingPlanner from the Torchrec repo,
        giving it the ability to plan dynamic embedding tables. The only difference from EmbeddingShardingPlanner
        is that DynamicEmbeddingShardingPlanner has an additional parameter `eb_configs`, which is a list of
        TorchREC BaseEmbeddingConfig. This is because the dynamic embedding table needs to re-plan the number of
        embedding vectors on each rank to align with the power of 2.

        Parameters
        ----------
        eb_configs : List[BaseEmbeddingConfig]
            A list of TorchREC BaseEmbeddingConfig in the TorchREC model
        topology : Optional[Topology], optional
            The topology of GPU and Host memory. If None, a default topology will be created. Defaults to None.
            The creation and usage are consistent with the same types in TorchREC.
            Note: The memory budget does not include the consumption of dynamicemb.
        batch_size : Optional[int], optional
            The batch size for training. Defaults to None, will set 512 in Planner.
        enumerator : Optional[Enumerator], optional
            An enumerator for sharding. Defaults to None.
            The creation and usage are consistent with the same types in TorchREC.
        storage_reservation : Optional[StorageReservation], optional
            Storage reservation details. Defaults to None.
            The creation and usage are consistent with the same types in TorchREC.
        proposer : Optional[Union[Proposer, List[Proposer]]], optional
            A proposer or a list of proposers for proposing sharding plans. Defaults to None.
            The creation and usage are consistent with the same types in TorchREC.
        partitioner : Optional[Partitioner], optional
            A partitioner for partitioning the embedding tables. Defaults to None.
            The creation and usage are consistent with the same types in TorchREC.
        performance_model : Optional[PerfModel], optional
            A performance model for evaluating sharding plans. Defaults to None.
            The creation and usage are consistent with the same types in TorchREC.
        stats : Optional[Union[Stats, List[Stats]]], optional
            Statistics or a list of statistics for the sharding process. Defaults to None.
            The creation and usage are consistent with the same types in TorchREC.
        constraints : Optional[Dict[str, DynamicEmbParameterConstraints]], optional
            A dictionary of constraints for every TorchREC embedding table and Dynamic embedding table. Defaults to None.
        debug : bool, optional
            A flag indicating whether to enable debug mode. Defaults to True.
        """
    ```

## DynamicEmbeddingCollectionSharder

Inherits from TorchREC's `EmbeddingCollectionSharder` and is used in exactly the same way. This API mainly overrides the deduplication process of indices through inheritance, making it compatible with dynamic embedding tables.

    ```python
    #How to import
    from dynamicemb.shard import DynamicEmbeddingCollectionSharder
    ```

## DynamicEmbCheckMode

When the dynamic embedding table capacity is small, a single feature with a large number of indices may lead to issues where the hashtable cannot insert the indices. Enabling safe check allows you to observe this behavior (the number of times indices cannot be inserted and the number of indices each time) to determine if the dynamic embedding table capacity is set too low.

    ```python
    #How to import
    from dynamicemb import DynamicEmbCheckMode

    #API arguments
    class DynamicEmbCheckMode(enum.IntEnum):
        """
        Enumeration for different modes of checking dynamic embedding's insertion behaviors.
        DynamicEmb uses a hashtable as the backend. If the embedding table capacity is small and the number of indices in a single feature is large,
        it is easy for too many indices to be allocated to the same hash table bucket in one lookup, resulting in the inability to insert indices into the hashtable.
        DynamicEmb resolves this issue by setting the lookup results of indices that cannot be inserted to 0.
        Fortunately, in a hashtable with a large capacity, such insertion failures are very rare and almost never occur.
        This issue is more frequent in hashtables with small capacities, which can affect training accuracy.
        Therefore, we do not recommend using dynamic embedding tables for very small embedding tables.

        To prevent this behavior from affecting training without user awareness, DynamicEmb provides a safe check mode.
        Users can set whether to enable safe check when configuring DynamicEmbTableOptions.
        Enabling safe check will add some overhead, but it can provide insights into whether the hash table frequently fails to insert indices.
        If the number of insertion failures is high and the proportion of affected indices is large,
        it is recommended to either increase the dynamic embedding capacity or avoid using dynamic embedding tables for small embedding tables.

        Attributes
        ----------
        ERROR : int
            When there are indices that can't be inserted successfully:
                This mode will throw a runtime error indicating how many indices failed to insert.
                The program will crash.
        WARNING : int
            When there are indices that can't be inserted successfully:
                This mode will give a warning about how many indices failed to insert.
                The program will continue. For uninserted indices, their embeddings' values will be set to 0.0.
        IGNORE : int
            Don't check whether insertion is successful or not, therefore it doesn't bring additional checking overhead.
            For uninserted indices, their embeddings' values will be set to 0.0 silently.
        """
        ERROR = 0
        WARNING = 1
        IGNORE = 2

    ```

## DynamicEmbInitializerMode

The initialization method for each embedding vector in the dynamic embedding table currently supports random UNIFORM distribution, random NORMAL distribution, and non-random constant initialization. The default distribution is UNIFORM.

    ```python
    #How to import
    from dynamicemb import DynamicEmbInitializerMode

    #API arguments
    class DynamicEmbInitializerMode(enum.Enum):
        """
        Enumeration for different modes of initializing dynamic embedding vector values.

        Attributes
        ----------
        NORMAL : str
            Normal Distribution.
        UNIFORM : str
            Uniform distribution of random values.
        CONSTANT : str
            All dynamic embedding vector values are a given constant.
        DEBUG : str
            Debug value generation mode for testing.
        """
        NORMAL = "normal"
        TRUNCATED_NORMAL = "truncated_normal"
        UNIFORM = "uniform"
        CONSTANT = "constant"
        DEBUG = "debug"
    ```

## DynamicEmbInitializerArgs

Parameters for each random initialization method in DynamicEmbInitializerMode.

    ```python
    #How to import
    from dynamicemb import DynamicEmbInitializerArgs

    #API arguments
    @dataclass
    class DynamicEmbInitializerArgs:
        """
        Arguments for initializing dynamic embedding vector values.

        Attributes
        ----------
        mode : DynamicEmbInitializerMode
            The mode of initialization, one of the DynamicEmbInitializerMode values.
        mean : float, optional
            The mean value for normal distributions. Defaults to 0.0.
        std_dev : float, optional
            The standard deviation for normal and distributions. Defaults to 1.0.
        lower : float, optional
            The lower bound for uniform distribution. Defaults to 0.0.
        upper : float, optional
            The upper bound for uniform distribution. Defaults to 1.0.
        value : float, optional
            The constant value for constant initialization. Defaults to 0.0.
        """
        mode: DynamicEmbInitializerMode
        mean: float = 0.0
        std_dev: float = 1.0
        lower: float = None
        upper: float = None
        value: float = 0.0
    ```

## DynamicEmbScoreStrategy

The storage space is limited, but the value range of sparse features is relatively large, 
so dynamicemb introduces the concept of score to perform customized eviction of sparse features within the limited storage space.
dynamicemb provides the following strategies to set the score.

    ```python
    #How to import
    from dynamicemb import DynamicEmbScoreStrategy

    #API arguments
    class DynamicEmbScoreStrategy(enum.IntEnum):
        """
        Enumeration for different modes to set index-embedding's score.
        The index-embedding pair with smaller scores will be more likely to be evicted from the embedding table when the table is full.

        dynamicemb allows configuring scores by table.
        For a table, the scores in the subsequent forward passes are larger than those in the previous ones for modes TIMESTAMP and STEP.
        Users can also provide customized score(mode CUSTOMIZED) for each table's forward pass.
        Attributes
        ----------
        TIMESTAMP:
            In a forward pass, embedding table's scores will be set to global nanosecond timer of device, and due to the timing of GPU scheduling,
            different scores may have slight differences.
            Users must not set scores under TIMESTAMP mode.
        STEP:
            Each embedding table has a member `step` which will increment for every forward pass.
            All scores in each forward pass are the same which is step's value.
            Users must not set scores under STEP mode.
        CUSTOMIZED:
            Each embedding table's score are managed by users.
            Users have to set the score before every forward pass using `set_score` interface.
        """

        TIMESTAMP = 0
        STEP = 1
        CUSTOMIZED = 2
    ```

    Users can specify the `DynamicEmbScoreStrategy` using `score_strategy` in `DynamicEmbTableOptions` per table.

## DynamicEmbPoolingMode

DynamicEmb supports three pooling modes that determine how embedding lookups are aggregated. These modes correspond to how `EmbeddingCollection` (sequence) and `EmbeddingBagCollection` (pooled) work in TorchREC.

All pooling modes use fused CUDA kernels for both forward and backward passes. Tables with different embedding dimensions (mixed-D) are fully supported in `SUM` and `MEAN` modes.

    ```python
    #How to import
    from dynamicemb import DynamicEmbPoolingMode

    #API arguments
    class DynamicEmbPoolingMode(enum.IntEnum):
        """
        Enumeration for pooling modes in dynamic embedding lookup.

        Attributes
        ----------
        SUM : int
            Sum pooling. For each sample, the embeddings of all indices in the bag
            are summed. Output shape: (batch_size, total_D) where total_D is the
            sum of embedding dimensions across all features.
        MEAN : int
            Mean pooling. For each sample, the embeddings of all indices in the bag
            are averaged. Output shape: same as SUM.
        NONE : int
            No pooling (sequence mode). Each index produces its own embedding row.
            Output shape: (total_indices, D).
        """
        SUM = 0
        MEAN = 1
        NONE = 2
    ```

## DynamicEmbTableOptions

Dynamic embedding table parameter class, used to configure the parameters for each dynamic embedding table, to be input into `DynamicEmbParameterConstraints`.

    ```python
    #How to import
    from dynamicemb import DynamicEmbTableOptions

    #API arguments
    @dataclass
    class DynamicEmbTableOptions(_ContextOptions):
        """
        Encapsulates the configuration options for dynamic embedding table.

        This class include parameters that control the behavior and performance of embedding lookup module, specifically tailored for dynamic embeddings.
        `get_grouped_key` will return fields used to group dynamic embedding tables.

        Parameters
        ----------
        training: bool
            Flag to indicate dynamic embedding tables is working on training mode or evaluation mode, default to `True`.
            If in training mode. **dyanmicemb** store embeddings and optimizer states together in the underlying key-value table. e.g.
                key:torch.int64
                value = torch.concat(embedding, opt_states, dim=1)

            Therefore, if `training=True` dynamicemb will allocate memory for optimizer states whose consumption is decided by the `optimizer_type` which got from `fused_params`. 
        initializer_args : DynamicEmbInitializerArgs
            Arguments for initializing dynamic embedding vector values when training, and default using uniform distribution.
            For `UNIFORM` and `TRUNCATED_NORMAL`, the `lower` and `upper` will set to $\pm {1 \over \sqrt{EmbeddingConfig.num\_embeddings}}$.
        eval_initializer_args: DynamicEmbInitializerArgs
            The initializer args for evaluation mode, and will return torch.zeros(...) as embedding by default if index/sparse feature is missing.
        caching: bool
            Flag to indicate dynamic embedding tables is working on caching mode, default to `False`.
            When the device memory on a single GPU is insufficient to accommodate a single shard of the dynamic embedding table, 
                dynamicemb supports the mixed use of device memory and host memory(pinned memory).
            But by default, the values of the entire table are concatenated with device memory and host memory.
            This means that the storage location of one embedding is determined by `hash_function(key)`, and mapping to device memory will bring better lookup performance.
            However, sparse features in training are often with temporal locality.
            In order to store hot keys in device memory, dynamicemb creates two table instances, 
                whose values are stored in device memory and host memory respectively, and store hot keys on the GPU table priorily. 
            If the GPU table is full, the evicted keys will be inserted into the host table.
            If the host table is also full, the key will be evicted(all the eviction is based on the score per key). 
            The original intention of eviction is based on this insight: features that only appear once should not occupy memory(even host memory) for a long time.
            In short:
                set **`caching=True`** will create a GPU table and a host table, and make GPU table serves as a cache;
                set **`caching=False`** will create a hybrid table which use GPU and host memory in a concatenated way to store value.
                All keys and other meta data are always stored on GPU for both cases.
        init_capacity : Optional[int], optional
            The initial capacity of the table. If not set, it defaults to max_capacity after sharding.
            If `init_capacity` is provided, it will serve as the initial table capacity on a single GPU.
            If set, it will be rounded up to the power of 2.
            As the `load_factor` of the table increases, its capacity will gradually double (rehash) until it reaches `max_capacity`.
            Rehash will be done implicitly.
            Note: This is the setting for a single table at each rank.
        max_load_factor : float
            The maximum load factor before rehashing occurs. Default is 0.5.
        score_strategy(DynamicEmbScoreStrategy):
            dynamicemb gives each key-value pair a score to represent its importance.
            Once there is insufficient space, the key-value pair will be evicted based on the score.
            The `score_strategy` is used to configure how to set the scores for keys in each batch.
            Default to DynamicEmbScoreStrategy.TIMESTAMP.
            For the multi-GPUs scenario of model parallelism, every rank's score_strategy should keep the same for one table,
                as they are the same table, but stored on different ranks.
        bucket_capacity : int
            Capacity of each bucket in the hash table, and default is 128 (using 1024 when the table serves as cache).
            A key will only be mapped to one bucket. 
            When the bucket is full, the key with the smallest score in the bucket will be evicted, and its slot will be used to store a new key. 
            The larger the bucket capacity, the more accurate the score based eviction will be, but it will also result in performance loss.
        safe_check_mode : DynamicEmbCheckMode
            Used to check if all keys in the current batch have been successfully inserted into the table.
            Should dynamic embedding table insert safe check be enabled? By default, it is disabled.
            Please refer to the API documentation for DynamicEmbCheckMode for more information.
        global_hbm_for_values : int
            Total GPU memory allocated to store embedding + optimizer states, in bytes. Default is 0.
            It has different meanings under `caching=True` and  `caching=False`.
                When `caching=False`, it decides how much GPU memory is in the total memory to store value in a single hybrid table.
                When `caching=True`, it decides the table capacity of the GPU table.
        external_storage: Storage
            The external storage/ParamterServer which inherits the interface of Storage, and can be configured per table.
            If not provided, will using DynamicEmbeddingTable as the Storage.
        index_type : Optional[torch.dtype], optional
            Index type of sparse features, will be set to DEFAULT_INDEX_TYPE(torch.int64) by default.
        admit_strategy : Optional[AdmissionStrategy], optional
            Admission strategy for controlling which keys are allowed to enter the embedding table.
            If provided, only keys that meet the strategy's criteria will be inserted into the table.
            Keys that don't meet the criteria will still be initialized and used in the forward pass,
            but won't be stored in the table. Default is None (all keys are admitted).
        admission_counter : Optional[Counter], optional
            Counter for tracking the number of keys that have been admitted to the embedding table.
            If provided, the counter will be used to track the number of keys that have been admitted to the embedding table.
            Default is None (no counter is used).    
        
        Notes
        -----
        For detailed descriptions and additional context on each parameter, please refer to the documentation in this repository.
        """

        training: bool = True
        initializer_args: DynamicEmbInitializerArgs = field(
            default_factory=DynamicEmbInitializerArgs
        )
        eval_initializer_args: DynamicEmbInitializerArgs = field(
            default_factory=lambda: DynamicEmbInitializerArgs(
                mode=DynamicEmbInitializerMode.CONSTANT,
                value=0.0,
            )
        )
        caching: bool = False
        init_capacity: Optional[
            int
        ] = None  # if not set then set to max_capcacity after sharded
        max_load_factor: float = 0.5  # max load factor before rehash(double capacity)
        score_strategy: DynamicEmbScoreStrategy = DynamicEmbScoreStrategy.TIMESTAMP
        bucket_capacity: int = 128
        safe_check_mode: DynamicEmbCheckMode = DynamicEmbCheckMode.IGNORE
        global_hbm_for_values: int = 0  # in bytes
        external_storage: Storage = None
        index_type: Optional[torch.dtype] = None
        admit_strategy: Optional[AdmissionStrategy] = None
        admission_counter: Optional[Counter] = None

    ```

## DynamicEmbDump

Automatically find the dynamic embedding tables in the torch model and parallelly dump them into the file system, dumping into a single file.

    ```python
    #How to import
    from dynamicemb import DynamicEmbDump

    #API arguments
    def DynamicEmbDump(path: str, model: nn.Module, table_names: Optional[Dict[str, List[str]]] = None, optim: Optional[bool] = False , pg: Optional[dist.ProcessGroup] = None) -> None:
        """
        Dump the distributed weights and corresponding optimizer states of dynamic embedding tables from the model to the filesystem.
        The weights of the dynamic embedding table will be stored in each EmbeddingCollection or EmbeddingBagCollection folder. 
        The name of the collection is the path of the torch module within the model, with the input module defined as str of model.

        Each dynamic embedding table will be stored as a key binary file and a value binary file, where the dtype of the key is int64_t,
        and the dtype of the value is float. Each optimizer state is also treated as a dynamic embedding table.

        Parameters
        ----------
        path : str
            The main folder for weight files.
        model : nn.Module
            The model contains dynamic embedding tables.
        table_names : Optional[Dict[str, List[str]]], optional
            A dictionary specifying which embedding collection and which table to dump. The key is the name of the embedding collection,
            and the value is a list of dynamic embedding table names within that collection. Defaults to None.
        optim : Optional[bool], optional
            Whether to dump the optimizer states. Defaults to False.
        pg : Optional[dist.ProcessGroup], optional
            The process group used to control the communication scope in the dump. Defaults to None.

        Returns
        -------
        None
        """
    ```

## DynamicEmbLoad

Load embedding weights from the binary file generated by `DynamicEmbDump` into the dynamic embedding tables in the torch model.

    ```python
    #How to import
    from dynamicemb import DynamicEmbLoad

    #API arguments
    def DynamicEmbLoad(path: str, model: nn.Module, table_names: Optional[List[str]] = None, optim: bool = False , pg: Optional[dist.ProcessGroup] = None):
        """
        Load the distributed weights and corresponding optimizer states of dynamic embedding tables from the filesystem into the model.

        Each dynamic embedding table will be stored as a key binary file and a value binary file, where the dtype of the key is int64_t,
        and the dtype of the value is float. Each optimizer state is also treated as a dynamic embedding table.

        Parameters
        ----------
        path : str
            The main folder for weight files.
        model : nn.Module
            The model containing dynamic embedding tables.
        table_names : Optional[Dict[str, List[str]]], optional
            A dictionary specifying which embedding collection and which table to load. The key is the name of the embedding collection,
            and the value is a list of dynamic embedding table names within that collection. Defaults to None.
        optim : bool, optional
            Whether to load the optimizer states. Defaults to False.
        pg : Optional[dist.ProcessGroup], optional
            The process group used to control the communication scope in the load. Defaults to None.

        Returns
        -------
        None
        """
    ```

## incremental_dump {#incremental_dump}

**Background**
In recommendation systems, an incremental dump refers to the process of exporting or updating only the new or changed data since the last data dump, rather than exporting the entire dataset each time. The background for using incremental dumps is that recommendation systems often operate on massive and continuously growing datasets, such as user interactions, item updates, and behavioral logs. Performing a full data dump frequently would be inefficient and resource-intensive, consuming significant storage and processing power.

**Target**
The main purpose of incremental dump is to improve efficiency by reducing the amount of data that needs to be processed and transferred during each update cycle. This allows the recommendation system to stay up-to-date with the latest data changes while minimizing downtime, storage usage, and computational overhead. Incremental dumps enable faster data refresh and model retraining, ensuring that recommendations remain relevant and timely in dynamic, large-scale environments.


**Behavior**
Given a model contains one or more `ShardedDynamicEmbeddingCollection`, this API will dump the eligible indices and embeddings of all ranks, based on the input `score_threshold`.

    ```python
    #How to import
    from dynamicemb.incremental_dump import incremental_dump

    #API arguments
    def incremental_dump(
        model: torch.nn.Module,
        score_threshold: Union[int, Dict[str, Dict[str, int]]],
        pg: Optional[dist.ProcessGroup] = None,
    ) -> Union[
        Tuple[
            Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]],
            Dict[str, Dict[str, int]],
        ],
        None,
    ]:
        """Dump the model's embedding tables incrementally based on the score threshold. The index-embedding pair whose score is not less than the threshold will be returned.

        Args:
            model(nn.Module):The model containing dynamic embedding tables.
            score_threshold(Uinon[int, Dict[str, Dict[str, int]]]):
                int: All embedding table's score threshold will be this integer. It will dump matched results for all tables in the model.
                Dict[str, Dict[str, int]]: the first `str` is the name of embedding collection in the model. 'str' in Dict[str, int] is the name of dynamic embedding table, and `int` in Dict[str, int] is the table's score threshold. It will dump for only tables whose names present in this Dict.
            pg(Optional[dist.ProcessGroup]): optional. The process group used to control the communication scope in the dump. Defaults to None.

        Returns
        -------
        Tuple:
            Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
                The first 'str' is the name of embedding collection.
                The second 'str' is the name of embedding table.
                The first tensor in the Tuple is matched keys on hosts.
                The second tensor in the Tuple is matched values on hosts.
            Dict[str, Dict[str, int]]:
                The first 'str' is the name of embedding collection.
                The second 'str' is the name of embedding table.
                `int` is the current score after finishing the dumping process, which will be used as the score for the next forward pass, and can also be used as the input of the next incremental_dump. If input score_threshold is `int`, the Dict will contain all dynamic embedding tables' current score, otherwise only dumped tables' current score will be returned.
        """
    ```

More usage please see [test](https://github.com/NVIDIA/recsys-examples/blob/main/corelib/dynamicemb/test/unit_tests/incremental_dump/test_distributed_dynamicemb.py)

## get_score

dynamicemb also provides a `get_score` interface whose returns are the current scores which will be used in the next forward pass. 
Users can use get_score’s returns at an earlier time and use it as a threshold for the later incremental dump. 
It's recommended for TIMESTAMP and STEP mode, and can also be used under CUSTOMIZED mode, but please note that get_score will only return the scores used in the next forward.
Under CUSTOMIZED mode, users need to understand the meaning of get_score's returns,  and dynamicemb is not responsible for managing the score any more.  For example, if users call get_score firstly to get the threshold, then decrease the score later to train the model, and finally call incremental_dump using the previous larger threshold, it will not dump anything.

    ```python
    #How to import
    from dynamicemb.incremental_dump import get_score

    #API arguments
    def get_score(model: torch.nn.Module) -> Union[Dict[str, Dict[str, int]], None]:
        """Get score for each dynamic embediing table.

        Args:
            model(torch.nn.Module): The model containing dynamic embedding tables.

        Returns:
            Dict[str, Dict[str,int]]:
                - The first `str` is the name of embedding collection in the model.
                - The second `str` is the name of dynamic embedding table.
                - `int` represents:
                    * TIMESTAMP mode: global timer of device
                    * STEP mode: table's step after last forward pass
                    * CUSTOMIZED mode: score set in last forward pass
                - Returns None if no dynamic embedding tables exist or scores unavailable
        """
    ```

## set_score

Under CUSTOMIZED mode, users have to set the scores for each table.
Users can set the scores one time and use it repeatedly in the several forward passes later.
Generally speaking, the score increases as training progresses, and dynamicemb will throw a warning when the new score is less than the old one.
Setting the environment variable DYNAMICEMB_CSTM_SCORE_CHECK to 0 will not throw the warnings.

    ```python
    #How to import
     from dynamicemb.incremental_dump import set_score

    #API arguments
    def set_score(
        model: torch.nn.Module, table_score: Union[int, Dict[str, Dict[str, int]]]
    ) -> None:
        """Set the score for each dynamic embedding table. It will not reset the scores of each embedding table, but register a score for the

        Args:
            model(torch.nn.Module): The model containing dynamic embedding tables.
            table_score(Union[int, Dict[str, Dict[str, int]]):
                int: all embedding table's scores will be set to this integer.
                Dict[str, Dict[str, int]]: the first `str` is the name of embedding collection in the model. 'str' in Dict[str, int] is the name of dynamic embedding table, and `int` in Dict[str, int] is the table's score which will broadcast to all scores in the same batch for the table.

        Returns:
            None.
        """
    ```

## Counter

**dynamicemb** provides an interface to the Counter which will be used in the embedding admission, and the users can customize the counter implementation by inherit the class `Counter`.


```python
class Counter(abc.ABC):
    """
    Interface of a counter table which maps a key to a counter.
    """

    @abc.abstractmethod
    def add(
        self, keys: torch.Tensor, frequencies: torch.Tensor, inplace: bool
    ) -> torch.Tensor:
        """
        Add keys with frequencies to the `Counter` and get accumulated counter of each key.
        For not existed keys, the frequencies will be assigned directly.
        For existing keys, the frequencies will be accumulated.
        Args:
            keys (torch.Tensor): The input keys, should be unique keys.
            frequencies (torch.Tensor): The input frequencies, serve as initial or incremental values of frequencies' states.
            inplace: If true then store the accumulated_frequencies to counter.
        Returns:
            accumulated_frequencies (torch.Tensor): the frequencies' state in the `Counter` for the input keys.
        """
        accumulated_frequencies: torch.Tensor
        return accumulated_frequencies

    @abc.abstractmethod
    def erase(self, keys) -> None:
        """
        Erase keys form the `Counter`.
        Args:
            keys (torch.Tensor): The input keys to be erased.
        """

    @abc.abstractmethod
    def memory_usage(self, mem_type=MemoryType.DEVICE) -> int:
        """
        Get the consumption of a specific memory type.
        Args:
            mem_type (MemoryType): the specific memory type, default to MemoryType.DEVICE.
        """

    @abc.abstractmethod
    def load(self, key_file, counter_file) -> None:
        """
        Load keys and frequencies from input file path.
        Args:
            key_file (str): the file path of keys.
            counter_file (str): the file path of frequencies.
        """

    @abc.abstractmethod
    def dump(self, key_file, counter_file) -> None:
        """
        Dump keys and frequencies to output file path.
        Args:
            key_file (str): the file path of keys.
            counter_file (str): the file path of frequencies.
        """
        
    @abc.abstractmethod
    def create(self, device: torch.device) -> "Counter":
        """
        Create the counter table on the specified device.
        """
```

**dynamicemb** also provides a built-in counter implementation named `KVCounter`.
There is as capacity limit of `KVCounter` which is bucketized, and the key with the smallest frequency will be evicted from the bucket for a new key if the bucket is full. 

```python

class KVCounter(Counter):
    """
    Interface of a counter table which maps a key to a counter.
    """

    def __init__(
        self,
        capacity: int,
        bucket_capacity: int = 1024,
        key_type: torch.dtype = torch.int64,
    )
```

## AdmissionStrategy

**AdmissionStrategy** is another component for implementing embedding admission.
The keys not in the dynamic embedding table, will first be passed to the `Counter`, after get the accumulated frequencies among the previous training process, the `AdmissionStrategy` will determine which keys will be admitted into the dynamic embedding table.

```python
class AdmissionStrategy(abc.ABC):
    @abc.abstractmethod
    def admit(
        self,
        keys: torch.Tensor,
        frequencies: torch.Tensor,
    ) -> torch.Tensor:
        """
        Admit keys with frequencies >= threshold.
        """

    @abc.abstractmethod
    def get_initializer_args(self) -> Optional[DynamicEmbInitializerArgs]:
        """
        Get the initializer args for keys that are not admitted.
        """
```

**dynamicemn** provides built-in `FrequencyAdmissionStrategy`, which will return keys whose frequencies are not less than the threshold.

```python
class FrequencyAdmissionStrategy(AdmissionStrategy):
    """
    Frequency-based admission strategy.
    Only admits keys whose frequency (score) meets or exceeds a threshold.
    Parameters
    ----------
    threshold : int
        Minimum frequency threshold for admission. Keys with frequency >= threshold
        will be admitted into the embedding table.
    initializer_args: Optional[DynamicEmbInitializerArgs]
        Initializer arguments which determine how to initialize the embedding if the key is not admitted.
    """

    def __init__(
        self,
        threshold: int,
        initializer_args: Optional[DynamicEmbInitializerArgs] = None,
    )
```

# Functionality and User interface

## Distributed embedding training and evaluation

Once the model containing `EmbeddingCollection` is built and initialized through `DistributedModelParallel`, it can be trained and evaluated on each GPU like a single GPU, with torchrec completing communication between different GPUs.

The switching between training and evaluation modes should be consistent with `nn.Module`, while `training` in [DynamicEmbTableOptions](./dynamicemb/dynamicemb_config.py) is used to guide whether to allocate memory to optimizer states when builds the table.

Due to limited resources, the dynamic embedding table does not pre allocate memory for all keys. If a key appears for the first time during training, it will be initialized immediately during the training process. Please see `initializer_args` and `eval_initializer_args` in `DynamicEmbTableOptions` for more information.

## Automatic eviction

The size of the table is finite, but the set of keys during training may be infinite. dynamicemb provides the function of automatic eviction, which constrains the size of tables reasonably when there is no available space. See `score_strategy` and `bucket_capacity` for more information.

## Caching and prefetch

dynamicemb supports caching hot embeddings on GPU memory, and you can prefetch keys from host to device like torchrec. Caching and prefetch work for both sequence mode (`NONE`) and pooling modes (`SUM`/`MEAN`). See `test_prefetch_flush_in_cache` in [test prefetch](./test/test_batched_dynamic_embedding_tables_v2.py) for usage examples.

## External storage

dynamicemb supports external storage once `external_storage` in `DynamicEmbTableOptions` inherits the `Storage` interface under [types.py](./dynamicemb/types.py). 
Refer to demo `PyDictStorage` in [unit test](./test/test_batched_dynamic_embedding_tables_v2.py) for detailed usage.


## Table expansion

Users can specify the initial capacity of a table on a single GPU. When the specified load factor is reached, the capacity of the table will double until the limit is reached. See `init_capacity`, `max_load_factor`, `max_capacity` in `DynamicEmbTableOptions` for more information.


## Dump/Load and Incremental dump

Dump/Load and incremental dump is different from general module in PyTorch, because dynamicemb's underlying implementation is a hash table instead of a dense `torch.Tensor`.

So dynamicemb provides dedicated interface to load/save models' states, and provide conditional dump to support online training.

Please see `DynamicEmbDump`, `DynamicEmbLoad`, `incremental_dump` in [APIs Doc](./DynamicEmb_APIs.md) for more information.

## Deterministic mode

When handling cache eviction and final table eviction, dynamicemb encounters randomness in the eviction of keys with the same score. To eliminate this uncertainty, dynamicemb provides a deterministic mode. Enabling this mode ensures that, under the same training script, the evicted keys will be determined.
This mode is enabled by setting the environment variable `DEMB_DETERMINISM_MODE`.