# DynamicEmb APIs

## APIs
- [DynamicEmbParameterConstraints](#dynamicembparameterconstraints)
- [DynamicEmbeddingEnumerator](#dynamicembeddingenumerator)
- [DynamicEmbeddingShardingPlanner](#dynamicembeddingshardingplanner)
- [DynamicEmbeddingCollectionSharder](#dynamicembeddingcollectionsharder)
- [DynamicEmbCheckMode](#dynamicembcheckmode)
- [DynamicEmbInitializerMode](#dynamicembinitializermode)
- [DynamicEmbInitializerArgs](#dynamicembinitializerargs)
- [DynamicEmbTableOptions](#dynamicembtableoptions)
- [DynamicEmbDump](#dynamicembdump)
- [DynamicEmbLoad](#dynamicembload)

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
            Including HKV Configs and Initializer Args. The initialization method for the parameters.
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
so HKV introduces the concept of score to perform customized evcition of sparse features within the limited storage space.
Based on the score of HKV, dynamicemb provides the following strategies to set the score.

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

## DynamicEmbTableOptions

Dynamic embedding table parameter class, used to configure the parameters for each dynamic embedding table, to be input into `DynamicEmbParameterConstraints`.

    ```python
    #How to import
    from dynamicemb import DynamicEmbTableOptions

    #API arguments
    @dataclass
    class DynamicEmbTableOptions(HKVConfig):
        """
        Encapsulates the configuration options for dynamic embedding tables.

        This class extends HKVConfig to include parameters that control the behavior and performance of
        hierarchical key-value storage systems, specifically tailored for dynamic embeddings in
        recommender systems. The options provided here allow users to customize their embedding tables
        according to their specific requirements.

        Including options:
            1. HKVConfig: explicitly defined by users.
                - Common configs for tables can be fused into a group.
                - Uncommon configs for each table in a group.
                - Some of HKVConfig can be inferred by context (index_type, embedding_dtype, dim, max_capacity, device_id, etc.)
            2. Initializer args.

        Parameters
        ----------
        index_type : Optional[torch.dtype], optional
            The index type of sparse features, will be set to DEFAULT_INDEX_TYPE(torch.int64) by default.
        embedding_dtype : Optional[torch.dtype], optional
            Data (weight) type of dynamic embedding table.
        score_type : torch.dtype
            The score represents how important an embedding item is. This specifies the type of the score.
        device_id : Optional[int], optional
            CUDA device index.

        dim : Optional[int], optional
            The dimensionality of the value vectors. Default is -1, indicating it should be set explicitly.
        max_capacity : Optional[int], optional
            The maximum capacity of the embedding table. Automatically set in the shared planner.
        init_capacity : Optional[int], optional
            The initial capacity of the table. If not set, it defaults to max_capacity after sharding.
        max_load_factor : float
            The maximum load factor before rehashing occurs. The default is 0.5.
        global_hbm_for_values : int
            Total high-bandwidth memory allocated for entire embedding values, in bytes. The default is 0.
        local_hbm_for_values : int
            High-bandwidth memory allocated for local values, in bytes. The default is 0.
        evict_strategy : DynamicEmbEvictStrategy
            Strategy used for evicting entries when the table exceeds its capacity. The default is DynamicEmbEvictStrategy.LRU.
            At present, only DynamicEmbEvictStrategy.LRU and DynamicEmbEvictStrategy.LFU are available.
        bucket_capacity : int
            The number of entries each bucket can hold. The default is 128.
        block_size : int
            The size of blocks used during operations. The default is 128.
        io_block_size : int
            The size of input/output blocks during data transfer operations. The default is 1024.
        io_by_cpu : bool
            Flag indicating whether to use CPU for handling IO operations. The default is False.
        use_constant_memory : bool
            Flag to indicate if constant memory should be utilized. The default is False.
        reserved_key_start_bit : int
            Bit offset for reserved keys in the key space. The default is 0.
        num_of_buckets_per_alloc : int
            Number of buckets allocated per memory allocation request. Default is 1.
        initializer_args : DynamicEmbInitializerArgs
            Arguments for initializing dynamic embedding vector values.
            Default is uniform distribution, and absolute values of upper and lower bound are sqrt(1 / eb_config.num_embeddings).
        score_strategy(DynamicEmbScoreStrategy):
            The strategy to set the score for each indices in forward and backward per table.
            Default to DynamicEmbScoreStrategy.TIMESTAMP.
            For the multi-GPUs scenario of model parallelism, every rank's score_strategy should keep the same for one table,
                as they are the same table, but stored on different ranks.
        safe_check_mode : DynamicEmbCheckMode
            Should dynamic embedding table insert safe check be enabled? By default, it is disabled.
            Please refer to the API documentation for DynamicEmbCheckMode for more information.

        Notes
        -----
        For detailed descriptions and additional context on each parameter, please refer to the documentation at
        https://github.com/NVIDIA-Merlin/HierarchicalKV.
        """

        initializer_args: DynamicEmbInitializerArgs = field(
            default_factory=DynamicEmbInitializerArgs
        )
        score_strategy: DynamicEmbScoreStrategy = DynamicEmbScoreStrategy.TIMESTAMP

    ```
If using `DynamicEmbInitializerMode.UNIFORM`, `DynamicEmbeddingShardingPlanner` will set the `initializer_args.upper` and `initializer_args.lower` to +/- sqrt(1 / eb_config.num_embeddings)
by default(except users provide them explicitly).

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

## incremental_dump

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