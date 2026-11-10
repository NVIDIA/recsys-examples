# Introduction
This document consists of two parts, one is the introduction to the API, which can be used as a reference, and the other part prompts specific functions from a functional perspective and which interfaces are related to them.

# DynamicEmb APIs

## APIs
- [DynamicEmbParameterConstraints](#dynamicembparameterconstraints)
- [DynamicEmbeddingShardingPlanner](#dynamicembeddingshardingplanner)
- [Sharding planner](#sharding-planner)
- [DynamicEmbeddingCollectionSharder](#dynamicembeddingcollectionsharder)
- [DynamicEmbCheckMode](#dynamicembcheckmode)
- [DynamicEmbInitializerMode](#dynamicembinitializermode)
- [DynamicEmbInitializerArgs](#dynamicembinitializerargs)
- [DynamicEmbPoolingMode](#dynamicembpoolingmode)
- [DynamicEmbTableOptions](#dynamicembtableoptions)
- [DynamicEmbDump](#dynamicembdump)
- [DynamicEmbLoad](#dynamicembload)
- [incremental_dump](#incremental_dump)
- [replay_increment](#replay_increment)
- [pop_evicted_keys](#pop_evicted_keys)
- [pop_erased_keys](#pop_erased_keys)
- [erase_by_score_async](#erase_by_score_async)
- [get_score](#get_score)
- [set_score](#set_score)
- [Counter](#counter)
- [AdmissionStrategy and MultiTableAdmitter](#admissionstrategy-and-multitableadmitter)
- [quiesce](#quiesce)
- [node_size](#node_size)

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

## DynamicEmbeddingShardingPlanner

A subclass of TorchREC's `EmbeddingShardingPlanner` that also plans dynamic embedding tables. It takes the same arguments and nothing more; a table is DynamicEmb's when its `DynamicEmbParameterConstraints` has `use_dynamicemb=True`.

Planning happens in two steps. The DynamicEmb tables are settled first -- their capacity comes from `DynamicEmbTableOptions` and their placement from the key-to-rank rule, so there is nothing to search -- and what they cost is taken out of the `Topology` per rank. TorchRec then plans the rest of the model, on a copy of the module with those tables removed, inside the memory that is left. The two halves are returned as one `ShardingPlan`.

The table configs come from the module at plan time, so there is no `eb_configs` argument: state the tables once, where the collection is built. `collective_plan` decides on rank 0 and broadcasts, as TorchREC's own planner does; the per-table options are reattached from each rank's own constraints afterwards, since `external_storage` and `score_function` are local to the process that made them.

For DynamicEmb sharding -- row-wise and table-row-wise alike, since both
bucketize by the same rule -- the supported `dist_type` values are:

- `continuous`
- `roundrobin`
- `hash_roundrobin`

`hash_roundrobin` hashes the raw key before rank assignment and is intended to reduce sensitivity to pathological raw-key patterns that can break plain modulo-based `roundrobin`. It is an opt-in routing mode; the default remains `roundrobin` for compatibility with existing checkpoints. It should be understood as a routing robustness improvement, not as a general solution to arbitrary hot-key or Zipf-skew load balancing.

`roundrobin` sends a key to rank `key % world_size`, and `hash_roundrobin` to rank `murmur3_fmix64(key) % world_size`, where `world_size` is the number of ranks the table is split over. Both read the key as an unsigned 64-bit number. An int32 key is widened to int64 with its sign first, so it goes to the rank of the int64 key with the same value. That is the key the table stores and a checkpoint holds, and checkpoint load and `replay_increment` place each key by it.

The hierarchical dist (`output_dist="hierarchical"`, see [Hierarchical sequence dist](#hierarchical-sequence-dist)) takes `roundrobin` and `hash_roundrobin` only.

    ```python
    #How to import
    from dynamicemb.planner import DynamicEmbeddingShardingPlanner

    #API arguments
    class DynamicEmbeddingShardingPlanner(EmbeddingShardingPlanner):
    def __init__(self,
        topology: Optional[Topology] = None,
        constraints: Optional[Dict[str, DynamicEmbParameterConstraints]] = None,
        storage_reservation: Optional[StorageReservation] = None,
        **kwargs):

        """
        A TorchRec planner that also plans DynamicEmb tables.

        Takes what EmbeddingShardingPlanner takes, and nothing else. Only the three
        arguments below mean anything different here; the rest -- batch_size,
        enumerator, proposer, partitioner, performance_model, stats, debug,
        callbacks, timeout_seconds, plan_loader -- are forwarded unchanged.

        Parameters
        ----------
        topology : Optional[Topology], optional
            The topology of GPU and Host memory. If None, TorchREC builds a default.
            State what the machine has: what the DynamicEmb tables cost is taken out
            of it at plan time, and TorchRec is given the difference.
        constraints : Optional[Dict[str, DynamicEmbParameterConstraints]], optional
            A dictionary of constraints for every TorchREC embedding table and Dynamic
            embedding table. A constraint with `use_dynamicemb=True` marks the table as
            DynamicEmb's. Per-table DynamicEmb options are filled in here at plan time,
            so the options a caller passes are completed in place rather than copied.
        storage_reservation : Optional[StorageReservation], optional
            Storage reservation details. Defaults to None.
            The creation and usage are consistent with the same types in TorchREC. It is
            for what TorchRec reserves for -- dense modules and the input KJT -- and
            sees a Topology the DynamicEmb tables are already out of.
        """
    ```

## Sharding planner

### What happens when you call it

`DynamicEmbeddingShardingPlanner` inherits from TorchREC's `EmbeddingShardingPlanner`. Internally it plans in four steps:

1. **Plan the DynamicEmb tables**, producing a `ParameterSharding` for each.
2. **Modify the planner's `Topology`** to deduct what those tables cost.
3. **Plan the remaining tables** — TorchREC's own planning, over the memory that is left.
4. **Assemble both halves** into one `ShardingPlan`.

**Step 1.** The table configs are read off the module, each DynamicEmb table's `DynamicEmbTableOptions` are filled in place (the table below), and one `ParameterSharding` is written per table: `ROW_WISE` or `TABLE_ROW_WISE` as its constraint asks, one shard of `[max_capacity, embedding_dim]` on each rank that holds it -- every rank for a row-wise table, one node's for a table-row-wise one -- placement by rank. Nothing is searched — a DynamicEmb table's capacity comes from its options and its placement from the key-to-rank rule.

**Step 2.** What each rank spends on those tables is worked out from the shard metadata written in step 1, including optimizer state (read from the sharders' `fused_params` or the parameters' `_optimizer_classes`), and taken out of the `Topology` per rank. Your `Topology` is not modified — say what the machine has, and the planner hands TorchREC the difference.

**Step 3.** TorchREC plans the rest of the model, on a copy of your module with the DynamicEmb tables removed. Only the modules on the path to a pruned collection are copied, so your module is not touched, and the tree keeps its shape so the plan's paths stay yours. Your `storage_reservation`, `enumerator`, `proposer`, `partitioner` and `stats` all apply here, unchanged.

**Step 4.** The DynamicEmb entries are added back to the plan under the paths they live at in your module, and the result is returned as one `ShardingPlan`.

Under `collective_plan`, steps 1 and 2 run on every rank and steps 3–4 on rank 0, whose result is broadcast — as TorchREC's own planner does. Step 2 is deliberately outside the broadcast: it is where a DynamicEmb table that does not fit is refused, and a refusal raised on rank 0 alone would leave every other rank waiting on a broadcast that never comes, turning an error into a hang. The per-table options are reattached from each rank's own constraints afterwards rather than sent, because `external_storage` is a live handle and `score_function` a callable, both local to the process that made them.

So: the DynamicEmb tables never enter TorchREC's search space, their memory is accounted for before TorchREC plans anything, and every rank gets the same placement decisions.

### Naming a table in `constraints`

`constraints` is keyed by string, and a table is named either by its own name or by `collection.name` -- the collection being the path to the module that holds it, as the model sees it. Each table asks for the qualified key first and the bare one second, and takes the first that is there. So **a model whose table names are unique needs nothing new**: a bare name reaches its table as it always has, and a qualified key overrides it for one collection.

The qualified form exists because two collections may declare one table name, and those are two tables: TorchREC shards each collection on its own, so they build separate runtimes with separate storage and separate optimizer state, and neither sees the other's updates. A DynamicEmb table cannot share its constraint with another, because `DynamicEmbTableOptions` is both your policy and the runtime's per-table config -- the planner writes each table's geometry into it -- so one object cannot describe two tables.

`get_constraint_key` writes the key for you:

    from dynamicemb.planner import get_constraint_key

    constraints = {
        get_constraint_key(shared_a, model): ...,   # 'sparse.ebc_a.shared'
        get_constraint_key(shared_b, model): ...,   # 'sparse.ebc_b.shared'
        get_constraint_key(item_id): ...,           # 'item_id'
    }

It takes the config **object** rather than a name, since by name there is nothing to tell two declarations apart, and the second argument is the root the path is relative to -- pass the model, or pass a collection itself to get the bare name back. Omit it and you get `config.name`.

Three things are refused rather than quietly ignored, since each would otherwise leave a constraint doing nothing: a key no table answers to; a qualified key on a table TorchREC plans, because TorchREC's planner looks constraints up by table name alone and would never find it; and one `DynamicEmbTableOptions` object reached by two tables.

**Not for `get_planner`.** Its `table_row_wise_embedding_table_names`, `data_parallel_embedding_table_names` and `dynamicemb_options_dict` are keyed by the bare table name -- it takes a flat `eb_configs` with no module to take a path from. A qualified key matches none of them, and `get_planner` refuses a name that `eb_configs` does not declare rather than planning the table some other way. A model that needs a qualified key builds its constraints and the planner directly.

### What the planner writes into your options

At plan time, `DynamicEmbeddingShardingPlanner` reads the table configs off the module and resolves each one against `constraints` (see **Naming a table in `constraints`** above). Then, **for each table with** `DynamicEmbParameterConstraints.use_dynamicemb == True`, it updates that table’s `DynamicEmbTableOptions` in **`dynamicemb_options`** via the internal routine `_prepare_dynemb_table_options` (order matters):

| Step | Field(s) | What happens |
|------|-----------|----------------|
| 1 | `initializer_args` | **`complete_initializer_args`** returns a new `DynamicEmbInitializerArgs` when needed. For **`UNIFORM`** initialization only: if `lower` or `upper` is `None`, they are filled. With a TorchREC `embedding_config`, bounds are `±sqrt(1 / num_embeddings)`; without it, `0.0` and `1.0`. Other modes are returned unchanged. |
| 2 | `bucket_capacity`, `max_capacity` | **`_sharded_table_bucket_layout(embedding_config, world_size, bucket_capacity)`** (internal) returns **`(num_buckets, effective_bucket_width)`** per rank. The planner overwrites **`bucket_capacity`** with the **effective** width (after `MAX_BUCKET_CAPACITY` / alignment rules). **`max_capacity`** is set to **`num_buckets * effective_bucket_width`**, i.e. the same value as **`get_sharded_table_capacity(embedding_config, world_size, bucket_capacity)`**. **`init_capacity`**: if unset, set to **`max_capacity`**; if set, align to **`bucket_capacity`**, then clamp to **`max_capacity`** if larger. **User input:** `bucket_capacity` on `DynamicEmbTableOptions` (multiple of **`BUCKET_ALIGNMENT` (16)** unless **`MAX_BUCKET_CAPACITY`** = `2**63 - 1`). **Sentinel:** layout is `(1, aligned_per_rank_rows)` — one bucket spanning the shard. **Otherwise:** `num_buckets = align_to_table_size(ceil(N/world), bucket_capacity) // bucket_capacity`. |
| 3 | `local_hbm_for_values` | Overwritten to **`ceil(global_hbm_for_values / fanout)`** so each rank holding the table gets an equal byte budget from the user-provided **`global_hbm_for_values`** (set on `DynamicEmbTableOptions` before planning). **It is the table's own budget, even when the runtime fuses it with others:** the fused group takes one layout decision on the sums of its sizes and budgets -- if every row fits the summed budget the whole group is HBM-only -- but when it does not fit, each table keeps `min(its size, its own budget)` in HBM and the rest of its rows go to the host tier; a spare budget is not lent to groupmates. The planner reserves memory the same way. A budget that resolves to zero also turns `caching` off for the table, with a warning, before grouping and cost see it. |

**The divisor (`fanout`) is per table.** It is how many pieces the table's rows are split into: `world_size` for a `ROW_WISE` table, which sits on every rank, and `local_world_size` for a `TABLE_ROW_WISE` one, which sits on a single node. The same number decides the shard metadata, so capacity and placement cannot disagree. Steps 2 and 3 both use it.

**Asking for `TABLE_ROW_WISE`.** Through `get_planner`, list the table in `table_row_wise_embedding_table_names`. Constructing constraints yourself, name it in `sharding_types`:

    DynamicEmbParameterConstraints(
        sharding_types=[ShardingType.TABLE_ROW_WISE.value],
        use_dynamicemb=True,
        dynamicemb_options=DynamicEmbTableOptions(...),
    )

**Which node it lands on is chosen for you.** The planner packs the table-row-wise tables across nodes by what they weigh, fitting them into the room the row-wise tables leave behind. You do not have to say.

To override, set **`node_index`** on `DynamicEmbTableOptions` to a node numbered `0 .. world_size // local_world_size - 1`. It is a pin, not a requirement: pinned tables are placed first and the rest are fitted around them, so you can name the nodes you care about and leave the others alone. Setting it on a row-wise table is an error rather than ignored — such a table is on every rank, so there is no node to name.

To replace the rule, pass a `NodePlacer` to the planner:

    from dynamicemb.planner import BalancedNodePlacer, NodePlacer, TableKey, TableToPlace

    class MyPlacer(NodePlacer):
        def place(self, tables, topology, committed) -> Dict[TableKey, int]:
            ...   # {table.key: node index}, deterministic on every rank

A `TableKey` is the pair `(collection, name)` -- the collection being the path to the module the table was declared in -- because a name alone is not an identity when two collections declare one. Each `TableToPlace` carries its own as `table.key`, so a placer returns the keys it was given.

    DynamicEmbeddingShardingPlanner(topology=..., constraints=..., node_placer=MyPlacer())

`BalancedNodePlacer` is the default: largest table first onto the emptiest node, measured against the tightest rank of a node rather than the node's total, because a table-row-wise table charges every rank of its node the same.

**What it costs.** A table-row-wise table lives on one node, so each of that node's ranks holds `1/local_world_size` of its rows rather than `1/world_size` — the same table needs `world_size / local_world_size` times the memory per rank it would need row-wise. For a DynamicEmb table that is not only memory: the table is bounded and evicts, so holding N times the rows per rank changes the eviction rate, and with it accuracy.

**What it buys, and what it does not.** The saving is on the **output** side. A row-wise table's pooled result is reduce-scattered across every rank, so the reduction itself crosses nodes; a table-row-wise table reduces inside its node first and only then sends one pooled result per node across — `O(B · D · world_size)` of cross-node output traffic becomes `O(B · D)`.

**Input keys still cross nodes.** The input distribution is a KJT all-to-all over the *global* process group under both sharding types; what table-row-wise changes is the number of buckets the keys are sorted into (`local_world_size` rather than `world_size`) and a staggered shuffle before the exchange — not its scope. A rank that does not sit on the node holding the table still sends that table's keys across the fabric. Choose table-row-wise for a table whose cross-node *output* traffic is a measured bottleneck, not to keep its lookups on one node.

**User-supplied values that should be set before planning** (typical DMP path) include at least:

- **`global_hbm_for_values`**: global HBM byte budget for the table’s values; the planner only splits it across ranks.
- **`bucket_capacity`**: requested hashtable bucket size in rows (see rules above), or `MAX_BUCKET_CAPACITY` for “single bucket per rank table”.
- **`initializer_args`**: optional partial bounds for `UNIFORM`; missing bounds are completed as in step 1.

**Downstream (after planning, when modules are built):** the batched embedding path uses **`max_capacity`** from **`DynamicEmbTableOptions`** (e.g. allocation size and consistency checks against TorchREC shard row counts in `_get_dynamicemb_options_per_table` in `batched_dynamicemb_compute_kernel.py`).

**Public helpers** (same rules as the planner):

- `from dynamicemb.dynamicemb_config import complete_initializer_args` (initializer completion; not re-exported from `dynamicemb` top-level today)
- `from dynamicemb import get_sharded_table_capacity` — returns **per-rank row capacity** after sharding and bucket alignment (``num_buckets * effective_bucket_width``), matching **`max_capacity`** set by the planner. With **`MAX_BUCKET_CAPACITY`**, the table is one bucket per rank whose width is the aligned shard row count.
- `from dynamicemb import get_table_value_bytes` — total bytes for **all ranks**’ value storage (embedding + optimizer state rows), using the same row layout as `get_sharded_table_capacity` for the given `bucket_capacity` (including **`MAX_BUCKET_CAPACITY`**). Use this to size **`global_hbm_for_values`** before planning; apply your own **caching** fraction or **HBM budget scale** on top if needed (as in benchmarks / examples).
- `from dynamicemb import BUCKET_ALIGNMENT, MAX_BUCKET_CAPACITY`

## DynamicEmbeddingCollectionSharder

Inherits from TorchREC's `EmbeddingCollectionSharder` and is used in exactly the same way. This API mainly overrides the deduplication process of indices through inheritance, making it compatible with dynamic embedding tables.

With `use_index_dedup=True`, each rank deduplicates its indices per table before the input distribution. The indices may be int32 or int64. int32 indices are deduplicated as int64, the keys the tables store, and are passed on as int32.

    ```python
    #How to import
    from dynamicemb.shard import DynamicEmbeddingCollectionSharder
    ```

With `use_index_dedup=True` and an LFU score strategy or an admission strategy, each rank sends its deduplicated keys together with how often each one occurs (as the KJT weights), so that a table counts every occurrence.

A training call on the hierarchical dist (`output_dist="hierarchical"` or `"auto"`) deduplicates the sharding's keys in the same way whatever `use_index_dedup` is, and its tables count the keys as on TorchRec's dists under the same flag: without it the keys carry the counts for LFU tables alone, and an admission strategy of a table that does not evict by LFU counts each key once per call. Tables that clip gradients need `use_index_dedup=True` there ([docs/hier_output_dist.md](./docs/hier_output_dist.md), sections 6 and 7.4).

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
        Users can also provide customized score(mode CUSTOMIZED) for each table's forward pass,
        or a score for every key (mode PRECISE_CUSTOMIZED).
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
        LFU:
            If there are not enough slots inside the bucket to store new keys, the least used key in the bucket will be evicted.
        PRECISE_CUSTOMIZED:
            Like CUSTOMIZED, but every key has a score of its own: the module is called with a `ScoredKeyedJaggedTensor`
            whose `scores` hold an int64 for each key. See "Per-key score" below.
        NO_EVICTION:
            The table’s capacity doubles whenever there are not enough slots for new keys, and this continues until available memory is exhausted.
            When the memory resources are insufficient, there will be a warning message, and training can continue but the accuracy of eviction cannot be guaranteed.
            Used alone, it keeps no score. It can be combined with `CUSTOMIZED` or `PRECISE_CUSTOMIZED` as `(NO_EVICTION, CUSTOMIZED)` (either order), which keeps the user's score. With such a combination **do not use a score of 0**: an erased key's slot is cleared to 0, and a full bucket gives up the slot with the smallest score, so a key scored 0 ties with an erased slot and can be the one that is given up.
            `incremental_dump` skips such tables (with a warning) and `replay_increment` refuses them.
        """

        TIMESTAMP = 0
        STEP = 1
        CUSTOMIZED = 2
        LFU = 3
        NO_EVICTION = 4
        PRECISE_CUSTOMIZED = 5
    ```

    Users can specify the `DynamicEmbScoreStrategy` using `score_strategy` in `DynamicEmbTableOptions` per table.

### Compound (multi-score) strategies and score order

A table normally keeps **one** score per key, but a **compound** `score_strategy`
given as a tuple keeps **more than one**. The supported compound
`(DynamicEmbScoreStrategy.TIMESTAMP, DynamicEmbScoreStrategy.LFU)` keeps **two**
scores per key — a last-access timestamp and an access frequency — so the table
can rank eviction by frequency while still supporting a time-based
`incremental_dump`.

The **order** you write the tuple in does **not** change eviction behavior:
`(TIMESTAMP, LFU)` and `(LFU, TIMESTAMP)` evict identically. It only controls the
order of the score columns **as you see them**, in two places that always stay
consistent with each other:

1. **Scores dumped to file** (`DynamicEmbDump` / `load`): each key's scores are
   written in the tuple order. `(TIMESTAMP, LFU)` writes `[timestamp, frequency]`
   per key; `(LFU, TIMESTAMP)` writes `[frequency, timestamp]`. `load` reads them
   back in the same order, so a checkpoint is interchangeable only between tables
   configured with the same tuple order.
2. **`score_function` indexing**: inside a `score_function`, `scores[0]`,
   `scores[1]`, ... follow the same tuple order. For `(TIMESTAMP, LFU)`,
   `scores[0]` is the timestamp and `scores[1]` the frequency; for
   `(LFU, TIMESTAMP)` the two are swapped.

Choose the order that matches how you want checkpoint columns laid out and how you
index `scores` in your `score_function` — the two always agree.

`(NO_EVICTION, CUSTOMIZED)` (either order) is the other supported combination. It is
not a second score: `NO_EVICTION` is a mode (the table never evicts, it grows), and
the table keeps **one** score, `CUSTOMIZED`'s. Set it with `set_score` before a forward
pass exactly as for a plain `CUSTOMIZED` table. `NO_EVICTION` used alone keeps **no**
score. Neither form supports `incremental_dump` / `replay_increment`: such a table has
no timestamp to take a delta by, and its erased keys are the user's to manage.

With `(NO_EVICTION, CUSTOMIZED)` or `(NO_EVICTION, PRECISE_CUSTOMIZED)`, **scores must be at
least 1**. An erased key leaves its slot with score 0, and when a bucket has no free slot
left the key with the smallest score is the one given up, so an erased slot is always
taken before a key that is in use -- unless that key is also scored 0, in which case the
two tie and the key can be lost. (`NO_EVICTION` alone has no score to set; the table
writes 1 for each key it stores, which gives it the same guarantee.)

### Customized eviction via `score_function`

`LFU` alone evicts by raw access frequency. For finer control — e.g. a
time-decayed LFU that also favors recently-used keys — configure a table with the
**compound** score strategy `(DynamicEmbScoreStrategy.TIMESTAMP,
DynamicEmbScoreStrategy.LFU)` (either order) and pass a Python `score_function` in
`DynamicEmbTableOptions`. When a bucket is full, the key(s) with the **lowest**
returned score are evicted.

**Signature:** `score_function(scores, cur_timestamp) -> float64`

- `scores` indexes the table's two scores in the **same order as your
  `score_strategy` tuple** (see "Compound (multi-score) strategies and score order"
  above). For `(TIMESTAMP, LFU)`: `scores[0]` is the last-access timestamp (device
  nanosecond timer) and `scores[1]` is the access frequency; for `(LFU, TIMESTAMP)`
  the two are swapped.
- `cur_timestamp` is the current device nanosecond timer at eviction time.
- The function is compiled to device code with **numba-cuda**, so it must be
  numba-compilable: index `scores` with **integer constants** only, and return a
  numeric value (the return is treated as `float64`, so an integer expression is
  cast). It should return a **finite** value; a `NaN` return is treated as
  evict-first. Any decay constant is written directly in the body. If
  `score_function` is omitted, eviction uses the default frequency-ranked evictor
  (ties broken by the older timestamp).

**Constraints:** `score_function` is supported **only** for the two-word compound
`{TIMESTAMP, LFU}` strategy — exactly one `TIMESTAMP` column plus one `LFU` column,
so `scores` always has **length 2** (`scores[0]`/`scores[1]`). It is **not**
available for single strategies (`TIMESTAMP`, `STEP`, `LFU`, `CUSTOMIZED`,
`NO_EVICTION`) or any other score combination or count. Setting `score_function`
with a non-compound `score_strategy` raises `ValueError`. `numba-cuda` must be
installed.

```python
import math
from dynamicemb import DynamicEmbScoreStrategy, DynamicEmbTableOptions

def lru_lfu_decay_score(scores, cur_timestamp):
    # scores[0] = last-access timestamp (ns), scores[1] = access frequency.
    # Time-decayed LFU: reward frequency, penalize staleness (~0.9 per second).
    age_seconds = (cur_timestamp - scores[0]) * 1e-9
    return math.log(max(scores[1], 1)) + age_seconds * math.log(0.9)

table_options = DynamicEmbTableOptions(
    score_strategy=(
        DynamicEmbScoreStrategy.TIMESTAMP,
        DynamicEmbScoreStrategy.LFU,
    ),
    score_function=lru_lfu_decay_score,
)
```

### Per-key score (`PRECISE_CUSTOMIZED`)

By default a CUSTOMIZED table scores every key of a batch with one number
(`set_score`). With `score_strategy=DynamicEmbScoreStrategy.PRECISE_CUSTOMIZED`, the input
carries **a score for each key** instead, and the table stores it as that key's
score. A key's score then decides its eviction (lowest first) like any other
CUSTOMIZED score.

The scores travel beside the keys, not in `weights`, in a
`ScoredKeyedJaggedTensor` -- a `KeyedJaggedTensor` with one extra int64 tensor
`scores`, 1-D, one entry per value:

```python
from dynamicemb import ScoredKeyedJaggedTensor

features = ScoredKeyedJaggedTensor(
    keys=["f0", "f1"], values=values, lengths=lengths, scores=scores
)
embeddings = model(features)
```

Scores are sent to the table's rank by their own all2all, kept in step with the
keys through the bucketize, and never mixed with the frequency counters that LFU
and admission use.

Rules:

- **All or none.** Every table of an `EmbeddingCollection` /
  `EmbeddingBagCollection` uses it, or none does, and a collection
  that sets it holds DynamicEmb tables only. While training, when set, the input must be a
  `ScoredKeyedJaggedTensor` (a plain `KeyedJaggedTensor` raises); evaluation reads
  no score, so it takes either. When not set,
  the scores of a `ScoredKeyedJaggedTensor` are ignored.
- **Score strategy.** `PRECISE_CUSTOMIZED`, alone or with `NO_EVICTION` as
  `(NO_EVICTION, PRECISE_CUSTOMIZED)`. Not with `admit_strategy`. It is its own
  strategy, so it cannot be combined with LFU.
- **Sharding.** Row-wise only (EC and EBC), on TorchRec's dists: the
  [hierarchical sequence dist](#hierarchical-sequence-dist) carries no score, so
  in an `EmbeddingCollection` `output_dist="hierarchical"` raises `ValueError`
  when the model is sharded (a one-rank sharding warns instead) and `"auto"`
  takes TorchRec's dists; an `EmbeddingBagCollection` ignores `output_dist`. No
  variable batch size per key
  (VBE). Tables with a host/hybrid storage tier (`HybridStorage`) are not
  supported yet. Neither is an `external_storage` that is the only
  tier (no cache and no HBM for values); with a cache above it, it is fine.
- **Training pipeline.** Not with TorchRec's `TrainPipelineSparseDist` or its
  variants while training: they fuse the splits all2all of the keys, which has
  no channel for scores, and the input dist raises `NotImplementedError`.
  Evaluation sends no scores, so it is not affected.
- **Repeated keys.** Within one batch -- including across ranks, and with
  `use_index_dedup` -- a key takes the **largest** score the batch gave it.
- **Across batches.** The score is assigned on every hit, so the latest batch
  wins, even if its score is lower; a key a batch does not contain keeps its
  score. Backward does not change scores.
- **`set_score`** raises for such a table: the scores come from the input.
- Scores must be int64. A negative score is read by the table as a very large
  one, so with `safe_check_mode` `ERROR` it raises, with `WARNING` it warns, and
  with `IGNORE` it is not looked for.

## DynamicEmbPoolingMode

DynamicEmb supports three pooling modes that determine how embedding lookups are aggregated. These modes correspond to how `EmbeddingCollection` (sequence) and `EmbeddingBagCollection` (pooled) work in TorchREC.

All pooling modes use fused CUDA kernels for both forward and backward passes. Tables with different embedding dimensions (mixed-D) are fully supported in `SUM` and `MEAN` modes.

    ```python
    #How to import
    from dynamicemb import DynamicEmbPoolingMode

    #API arguments
    @enum.unique
    class DynamicEmbPoolingMode(enum.IntEnum):
        """
        Enumeration for pooling modes in dynamic embedding lookup.

        The values are taken from the bound C++ enum ``dyn_emb::PoolingMode``
        (``src/utils.h``), which is the single source of truth, in the same way
        ``DynamicEmbEvictStrategy`` takes its values from ``EvictStrategy``. It
        is an ``IntEnum`` because a pooling mode is handed straight to the
        kernels and compared as an integer, so it stays usable wherever a plain
        ``int`` was before (``SUM == 0``, dict keys, JSON).

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
        SUM = BagPoolingMode.KSum    # 0
        MEAN = BagPoolingMode.KMean  # 1
        NONE = BagPoolingMode.KNone  # 2
    ```

**Weighted SUM pooling** — `SUM` pooling supports optional per-position float32 weights: `out[b] = Σ wᵢ · embᵢ`. Supported in both training and
eval mode, only for `SUM` (no weighted-MEAN variant), and works with mixed-D tables. Two ways to use it:

- **Through TorchRec's `EmbeddingBagCollection`**: build the collection with
  `is_weighted=True` and attach the weights to the features KJT (`KeyedJaggedTensor(keys=..., values=indices, weights=weights, lengths=...)`);
  call `model(features)` as usual. The weights ride the KJT through the all2all distribution and reach the DynamicEmb lookup as pooling weights.

  **`is_weighted=True` is required, not merely conventional.** DynamicEmb only reads the KJT weights as pooling weights when the collection
  declares itself weighted; on an `is_weighted=False` collection the weights are ignored and the result is a plain unweighted sum, with no
  error raised. The weights channel of a `KeyedJaggedTensor` is shared — TorchRec also uses it to carry per-feature scores for virtual-table
  eviction, and the sequence (`EmbeddingCollection`) path reads it as LFU frequency counters — so the flag is what disambiguates them.
- **Through `BatchedDynamicEmbeddingTablesV2.forward` directly**:
  `module(indices, offsets, pooling_weights=w)` where `w` is a float32 tensor aligned with `indices` (`w.numel() == indices.numel()`).

**The weights are not differentiable.** No gradient is computed for them, so learned per-position weights are not supported: passing a tensor with
`requires_grad=True` while grad mode is enabled raises `ValueError` rather than letting the producing module train against a gradient that is
silently zero. Weights that carry `requires_grad` are still accepted under `torch.no_grad()`, which is what keeps eval working.

Weighted pooling raises `ValueError` when: `pooling_mode != SUM`, the weights are not float32, `weights.numel() != indices.numel()`, the weights
require grad while grad mode is enabled, or `frequency_counters` is passed alongside them (both are carried by the single KJT weights channel, so
a caller has to pick one).

## DynamicEmbOptimType

The `optimizer` entry in `fused_params` normally takes an FBGEMM `EmbOptimType`. DynamicEmb also implements
optimizers that `EmbOptimType` has no member for; those live in `DynamicEmbOptimType` and are accepted
anywhere an optimizer type is. The two enums never compare equal, so an `EmbOptimType` check elsewhere
cannot accidentally match one of them.

    ```python
    #How to import
    from dynamicemb import DynamicEmbOptimType

    @enum.unique
    class DynamicEmbOptimType(enum.Enum):
        FTRL = "ftrl"
    ```

### FTRL

FTRL-Proximal, Algorithm 1 of *Ad Click Prediction: a View from the Trenches* (McMahan et al., KDD 2013),
with the paper's fixed square root generalized to an arbitrary exponent. Per coordinate:

```
new_accum = accum + grad²
linear   += grad − (new_accum^(−p) − accum^(−p)) / lr · weight        p = learning_rate_power
weight    = 0                                                          if |linear| ≤ l1_reg
          = (sign(linear)·l1_reg − linear)
            / ((ftrl_beta + new_accum^(−p)) / lr + l2_reg)             otherwise
accum     = new_accum
```

`lr`, `ftrl_beta`, `l1_reg` and `l2_reg` are the paper's α, β, λ1 and λ2; `p = -0.5` makes `n^(−p)` a
square root and so recovers its `α / (β + √n)` learning rate exactly. Pass them through `fused_params` alongside `optimizer`:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `learning_rate` | `0.01` | α. Must be positive -- FTRL divides by it. |
| `learning_rate_power` | `-0.5` | Exponent on the accumulator **in the learning rate**, so it must be `<= 0`: negative decays the rate, `0` holds it fixed, and a positive value would make it grow without bound (rejected). `-0.5` is the paper's own choice and takes a faster kernel path; other values follow TensorFlow's generalization. |
| `ftrl_beta` | `0.0` | β. Keeps the per-coordinate learning rate finite while `accum` is still small -- the paper's way of bounding the first steps. |
| `initial_accumulator_value` | `0.0` | Seeds `accum`; `linear` always starts at 0. Read the note below before setting it non-zero. |
| `l1_reg` | `0.0` | λ1. |
| `l2_reg` | `0.0` | λ2. |
| `group_lasso_reg` | `0.0` | λ of G-FTRL -- see [FTRL with group sparsity](#ftrl-with-group-sparsity). `0` is plain FTRL. Must be `>= 0`. Combines with `l1_reg` (sparse group lasso). |

Two behaviours worth knowing before choosing FTRL:

- **Per-row state is `2 × embedding_dim`**, laid out as `linear` then `accum` right after the embedding --
  the same shape of layout Adam uses for its `m` and `v`. Checkpoints store it at full width.
- **The weight is re-solved from `(linear, accum)` each step** rather than nudged from its previous value.
  That is what lets `l1_reg` drive a weight to exactly zero instead of merely shrinking it.

  It is also why seeding the state deserves care here. FTRL was written for linear regression, whose
  weights start at zero; an embedding's do not, and a state of `linear = 0` is inconsistent with the
  weight already sitting in the row. The first update reconciles them, keeping a fraction

  ```
  (√n₁ − √n₀) / (ftrl_beta + √n₁)        n₀ = initial_accumulator_value,  n₁ = n₀ + g²
  ```

  of the initializer. **Both** knobs that bound the early steps cost retention, and for the small
  gradients typical of embeddings they cost nearly all of it:

  | `n0` | `ftrl_beta` | `g = 1.0` | `g = 0.01` | `g = 0.0001` |
  | --- | --- | --- | --- | --- |
  | `0.0` | `0.0` | 100% | 100% | 100% |
  | `0.1` | `0.0` | 69.8% | 0.05% | 0.000005% |
  | `0.0` | `1.0` | 50% | 0.99% | 0.01% |
  | `0.1` | `1.0` | 35.8% | 0.012% | 0.0000012% |

  Only `initial_accumulator_value = 0` together with `ftrl_beta = 0` keeps the initializer whole. That
  combination has its own cost: with `n₀ = 0` the first step is `w − α·sign(g)`, a full α-sized move
  however small the gradient. Which matters more is a modelling choice -- if the initializer is doing
  real work, keep both at 0 and pick α accordingly; if the early steps need damping, expect the
  initializer to be mostly overwritten and size it as if rows started near zero.

#### FTRL with group sparsity

Setting `group_lasso_reg > 0` adds a group-lasso term to FTRL, following **G-FTRL** (Algorithm 1 and Eq. 3 of *Feature
Selection for Facebook Feed Ranking System via a Group-Sparsity-Regularized Training Algorithm*, Ni et al., CIKM 2019),
with **one embedding row of width `d` as the group**. `linear` and `accum` are updated exactly as in FTRL; only the
weight solve differs.

`l1_reg`, `l2_reg` and `group_lasso_reg` are independent parameters. Each step rebuilds a row from its FTRL state
(`z = linear`, `n = accum`) with the update below, which takes this objective as its reference:

```
min_w   z·w + ½ Σᵢ aᵢ wᵢ² + l1_reg·‖w‖₁ + (l2_reg / 2)·‖w‖₂² + τ·‖w‖₂
aᵢ = (ftrl_beta + nᵢ^(−p)) / lr          τ = group_lasso_reg · √d          ‖·‖₂ over the row
```

The update is the papers' common-shrink form, **not** the exact minimizer of this objective. It is exact in the zero
test and when every `hᵢ` in the row is equal; otherwise it is an approximation with no error bound and does not
satisfy the objective's first-order conditions (see the last bullet). The configurations it gives:

| Set | Effect |
| --- | --- |
| `l1_reg` only | element-level sparsity |
| `group_lasso_reg` only | row-level sparsity |
| `l1_reg` + `group_lasso_reg` | row-level and element-level sparsity together |
| plus `l2_reg` | bounds the weights on top of any of the above |

With `s = sign(z)·max(|z| − l1_reg, 0)` and `hᵢ = aᵢ + l2_reg`, the update is

```
weight = 0                                         if ‖s‖₂ ≤ τ
       = −(1 − τ / ‖s‖₂) · sᵢ / hᵢ                  otherwise
```

- The threshold carries the group-size factor `√d`, a scaling convention taken from the paper. It does not guarantee
  that rows of different widths end up with the same sparsity rate: how often a row clears `λ·√d` still depends on the
  gradients it receives, so `λ` may need tuning per table when the widths differ.
- With `l1_reg = 0` (the default) `s = z`, which is the paper's closed form. With `l1_reg > 0` the row norm is taken
  after the L1 soft-threshold, the sparse-group-lasso step of *Adaptive Optimizers with Sparse Group Lasso for Neural
  Networks in CTR Prediction* (Yue et al., arXiv:2107.14432, Algorithm 3), so L1 zeroes elements inside a row that
  survives the group test.
- The paper's own `γ_g`, which only adds `λ·γ_g` to the denominator that `l2_reg` already feeds, is not a separate
  knob; `l2_reg = λ·γ_g` reproduces it.
- Eq. 3 as printed in the paper is typeset with a prefactor that reads as `1/η_t`; this follows Algorithm 1 and the
  minimizer of the objective, which give `η_t`.
- **The zero test is exact, the shrink factor is not always.** The stationarity condition of the objective gives
  `wᵢ = −sᵢ / (hᵢ + τ/‖w‖₂)`, which is zero for the whole row exactly when `‖s‖₂ ≤ τ`. In a non-zero row the update above
  uses one shrink factor `1 − τ/‖s‖₂` for the row and the per-element `hᵢ`, as the papers do; that equals the exact
  solution when every `hᵢ` in the row is the same and is an approximation when they differ. For example, with
  `z = [3, 4]`, `h = [1, 2]`, `τ = 1` the update gives `[−2.4, −1.6]` while the exact minimizer is about
  `[−2.207, −1.695]`.

`group_lasso_reg = 0` takes the original single-pass kernel path and is bit-identical to plain FTRL. The solve needs
the whole row's norm, so `group_lasso_reg > 0` makes the kernel two passes over the row.

How the group threshold is set, and what surrounds the update:

- **The group size scaling is fixed at `√d`.** `τ = group_lasso_reg · √d`, with `d` the row's real embedding width (not
  a padded buffer's widest row), so a table's own `dim` is what enters. There is no option to change it. As above, this
  is a convention, not a promise of equal sparsity across widths.
- **Regularization lives in the update, not the gradient.** `l1_reg`, `l2_reg` and `group_lasso_reg` are applied when the
  row is rebuilt from `(linear, accum)`; the gradient handed to the optimizer must not also carry their gradients. On the
  way there a gradient is clipped (when clipping is on), summed over repeated lookups of the same key, cast to the
  embedding dtype, and only then given to the optimizer, so a key looked up several times in a step is updated **once**
  with the summed gradient. That is not the same as updating once per lookup, since `(g₁+g₂)² ≠ g₁²+g₂²`.
- **Only rows touched in a step are updated.** A row that no key in the step looks up is not visited: its `linear` and
  `accum` do not change and it is not re-solved, so this is not equivalent to a pass over every row at each step.
- **The learning rate is read at each step but never applied retroactively.** `σ = (n_new^(−p) − n_old^(−p)) / lr`
  is computed with the current `lr`, and `linear` already holds the history built under earlier values. Changing the
  learning rate with `set_learning_rate` partway through training therefore does not reproduce a run that used the new
  value from the start. This is the behaviour of FTRL here, not specific to the group term.
- **State precision.** `linear` and `accum` live in the table's value row at the table's dtype, so a fp16 or bf16 table
  keeps them at that precision (`optimizer_state_dtype` is only for row-wise Adagrad). The group path rounds the
  updated state to that dtype *before* taking the row norm, so the zero test is of what is stored. A row whose norm is
  very close to `τ` can therefore fall on the other side than it would in fp32.
- **Checkpoints.** The meta records `group_lasso_reg` and loading restores it, so a resumed run keeps training with the
  group term instead of falling back to plain FTRL. Checkpoints written before this option existed carry no
  `group_lasso_reg` and load as `group_lasso_reg = 0`.

**A freshly initialized row is the one most likely to be zeroed.** The threshold is on the row's accumulated
`linear`, not on its weights, and `linear` starts at 0 -- so on the first update a row's initializer is not what
is compared against the threshold, only the gradient history it has accumulated so far. The first step is, per element
(`ftrl_beta` does not enter `linear`; it only changes the weight solved from it),

```
linear = g − (n₁^(−p) − n₀^(−p)) / lr · weight         p = learning_rate_power,  n₀ = initial_accumulator_value,  n₁ = n₀ + g²
```

which depends on `learning_rate_power`:

- **`learning_rate_power = −0.5` (the default) with `initial_accumulator_value = 0`:** `linear = g − |g| · weight / lr`.
  A row whose gradients are small has a small `‖s‖₂`, and if it does not exceed `λ·√d` the whole row, initializer
  included, is written as zero. This is the same effect as the initializer-retention table above, only all-or-nothing
  per row, and a larger `group_lasso_reg` makes it more likely.
- **`learning_rate_power = 0`:** the two powers are both 1 and cancel, so `linear = g` whatever the initializer or
  `initial_accumulator_value`; the first step's `‖s‖₂` is just the gradient's own, and the formula above does not apply.
- **Any other power:** `linear = g − |g|^(−2p) · weight / lr` for `initial_accumulator_value = 0`.

Choose `group_lasso_reg` below what a row with a typical first-step gradient reaches under your `learning_rate_power`
(`‖s‖₂ / √d`), or expect the rows that are touched only a few times to come out as zero. A zeroed row is not removed:
its `linear` and `accum` keep accumulating, so it can become non-zero again once enough gradient has built up.

Unlike the `EmbOptimType` optimizers, FTRL has no FBGEMM counterpart, so `construct_twin_module` cannot
build a TorchRec twin for a model that uses it.

### Optimizer state precision

`optimizer_state_dtype`, a field of [DynamicEmbTableOptions](#dynamicembtableoptions), sets the
precision the optimizer state is kept at. It is also what a dump records as `optim_state_dtype`. It is
a per-table option rather than a `fused_params` entry because `fused_params` also reaches the tables
FBGEMM shards, which do not accept it.

| Optimizer | Allowed values | Default (`None`) |
| --- | --- | --- |
| `EXACT_ROWWISE_ADAGRAD` | `torch.float32`, or the table's `embedding_dtype` | `torch.float32` |
| all others | the table's `embedding_dtype` | the table's `embedding_dtype` |

Row-wise Adagrad defaults to `float32` because its accumulator is a running sum of mean squared
gradients, which for embeddings is often below fp16's smallest subnormal (~6e-8). Kept in a 16-bit
table's dtype it rounds to zero on every update, so the learning rate never decays. Setting it to the
table's dtype restores the old behaviour. The accumulator always fits in the row's fixed 16-byte state
slot, so this choice changes neither the row layout nor memory use. A checkpoint loads into a table
with either setting; the accumulator is converted to the table's precision.

## DynamicEmbTableOptions

Per-table configuration for dynamic embedding, passed into `DynamicEmbParameterConstraints` as `dynamicemb_options`. The authoritative definition lives in `dynamicemb.dynamicemb_config.DynamicEmbTableOptions` (this section mirrors its docstring).

Fields declared first (through `device_id`) are **planner/runtime-heavy**: `DynamicEmbeddingShardingPlanner` fills them via `_prepare_dynemb_table_options` together with internal `_sharded_table_bucket_layout` (and thus the same per-rank row count as `get_sharded_table_capacity`). User-facing knobs such as `training`, `bucket_capacity`, `global_hbm_for_values`, and `initializer_args` follow. Hash-table **scores** are driven by `score_strategy` and kernels, not by a separate score-dtype field on this dataclass.

    ```python
    #How to import
    from dynamicemb import DynamicEmbTableOptions

    #API arguments
    @dataclass
    class DynamicEmbTableOptions:
        """
        Encapsulates the configuration options for dynamic embedding table.

        This class includes parameters that control the behavior and performance of the embedding lookup module, specifically tailored for dynamic embeddings.
        `get_grouped_key` will return fields used to group dynamic embedding tables.

        Fields listed first (through ``device_id``) are often filled by the planner or runtime rather than
        being the main user configuration knobs. Score handling for the hash table follows score
        policies and kernels, not a separate score-dtype field on this dataclass.

        Parameters
        ----------
        embedding_dtype : Optional[torch.dtype], optional
            Data (weight) type of dynamic embedding table. Also the precision a checkpoint stores:
            ``DynamicEmbDump`` writes value files at this dtype and records it in the table's meta JSON as
            ``embedding_dtype`` (alongside ``embedding_dim`` / ``optim_state_dtype``), which is how
            ``DynamicEmbLoad`` reads them back. Loading a checkpoint whose precision differs converts the
            values and warns; a differing dim is an error.
        dim : Optional[int], optional
            Value vector dimension. With ``DynamicEmbeddingShardingPlanner``, ``_prepare_dynemb_table_options``
            sets it from ``BaseEmbeddingConfig.embedding_dim``. The embedding kernel only warns if it
            differs from the sharded ``local_cols`` (see ``_get_dynamicemb_options_per_table``).
        max_capacity : Optional[int], optional
            Per-shard maximum table rows on one GPU. With ``DynamicEmbeddingShardingPlanner``,
            ``_prepare_dynemb_table_options`` sets ``max_capacity`` to
            per-rank row count from ``get_sharded_table_capacity``.
            You may omit ``max_capacity`` on ``DynamicEmbTableOptions`` before planning and let the planner set it.
            If ``init_capacity`` is unset it becomes ``max_capacity``; if set and aligned,
            it is clamped to at most ``max_capacity``.
            The embedding kernel checks consistency with TorchREC shard metadata (see
            ``_get_dynamicemb_options_per_table``).
        evict_strategy : DynamicEmbEvictStrategy
            Strategy used for evicting entries when the table exceeds its capacity.
            Default is ``DynamicEmbEvictStrategy.LRU``.
        local_hbm_for_values : int
            High-bandwidth memory allocated for local values, in bytes. Default is 0.
            With ``DynamicEmbeddingShardingPlanner``, this is set to
            ``ceil(global_hbm_for_values / world_size)`` per rank.
        device_id : Optional[int], optional
            CUDA device index.
        training: bool
            Flag to indicate dynamic embedding tables is working on training mode or evaluation mode, default to `True`.
            If in training mode. **dynamicemb** stores embeddings and optimizer states together in the underlying key-value table. e.g.
                key:torch.int64
                value = torch.concat(embedding, opt_states, dim=1)

            Therefore, if `training=True` the module allocates memory for optimizer states; the per-row state size follows the `optimizer` entry in `fused_params` (FBGEMM `EmbOptimType`), as used by `BatchedDynamicEmbeddingTablesV2`.
        initializer_args : DynamicEmbInitializerArgs
            Arguments for initializing dynamic embedding vector values when training, and default using uniform distribution.
            For ``UNIFORM`` and ``TRUNCATED_NORMAL``, ``lower`` and ``upper`` default to
            ``±1/sqrt(N)`` where ``N`` is ``EmbeddingConfig.num_embeddings``.
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
            A table whose resolved `local_hbm_for_values` is zero has caching turned off, with a warning, at plan/construction time: a cache is an HBM tier, and without a budget it would be a single bucket whose overflow cannot absorb a batch's admitted keys.
        init_capacity : Optional[int], optional
            The initial capacity of the table. If not set, it defaults to max_capacity after sharding.
            ``BatchedDynamicEmbeddingTablesV2`` also sets ``init_capacity`` to ``max_capacity`` when it is still ``None``.
            If `init_capacity` is provided, it will serve as the initial table capacity on a single GPU.
            With ``DynamicEmbeddingShardingPlanner``, it is rounded up to a multiple of the effective
            ``bucket_capacity`` in ``_prepare_dynemb_table_options``, then capped at ``max_capacity`` if the aligned value is larger.
            As the `load_factor` of the table increases, its capacity will gradually double (rehash) until it reaches `max_capacity`.
            Rehash will be done implicitly.
            Note: This is the setting for a single table at each rank.
        max_load_factor : float
            The maximum load factor before rehashing occurs. Default is 0.5.
            In NO_EVICTION mode, this option is ignored: the implementation uses a fixed effective max load factor of 0.5 for the key_index_map (initial sizing and expansion). See the "Table expansion" section for NO_EVICTION trigger conditions.
        score_strategy(DynamicEmbScoreStrategy or Tuple[DynamicEmbScoreStrategy, ...]):
            dynamicemb gives each key-value pair a score to represent its importance.
            Once there is insufficient space, the key-value pair will be evicted based on the score.
            The `score_strategy` is used to configure how to set the scores for keys in each batch.
            Default to DynamicEmbScoreStrategy.TIMESTAMP.
            For the multi-GPUs scenario of model parallelism, every rank's score_strategy should keep the same for one table,
                as they are the same table, but stored on different ranks.
            A *compound* score strategy may be given as a tuple to keep **more than
            one** score per key. Only
            `(DynamicEmbScoreStrategy.TIMESTAMP, DynamicEmbScoreStrategy.LFU)` and
            `(DynamicEmbScoreStrategy.NO_EVICTION, DynamicEmbScoreStrategy.CUSTOMIZED)` (in
            either order) are supported for now. The first ranks eviction by the LFU access
            count while ALSO keeping a per-key last-access timestamp (two scores per
            key, +8 bytes/key) so items touched since the last dump can be selected by
            a time-based `incremental_dump`. Both tuple orders evict identically; the
            order only controls how the score columns are laid out when dumped to file
            and how `score_function` indexes them (see "Compound (multi-score)
            strategies and score order" under `DynamicEmbScoreStrategy`). A one-element
            tuple `(X,)` is treated as the single strategy `X`.
        score_function(Optional[Callable]):
            Optional custom eviction score for a compound `{TIMESTAMP, LFU}` table.
            Signature `score_function(scores, cur_timestamp) -> float64`; `scores` is
            indexed in the configured (logical) tuple order, and the key(s) with the
            lowest returned score are evicted first. It is compiled to device code
            with numba-cuda, so subscripts must be integer constants. Supported
            **only** for the two-word `{TIMESTAMP, LFU}` compound strategy (`scores`
            has length 2); any other (single or non-compound) `score_strategy` with a
            `score_function` raises `ValueError`. Defaults to None (the built-in
            frequency → older-timestamp evictor). See the "Customized eviction via
            `score_function`" section under `DynamicEmbScoreStrategy`.
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
            To match planner row counts and optimizer state width, size the **global** budget with
            ``get_table_value_bytes(embedding_config, EmbOptimType, world_size, bucket_capacity)``
            (same ``bucket_capacity`` you pass into ``get_sharded_table_capacity``, e.g. ``128`` or ``MAX_BUCKET_CAPACITY``),
            then multiply by a cache ratio or scale if desired. The planner overwrites **per-rank**
            ``local_hbm_for_values`` to ``ceil(global_hbm_for_values / world_size)``.
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
            Anything deciding takes -- a frequency counter, an initializer for
            the rows it rejects -- is configured on the strategy itself, e.g.
            ``FrequencyAdmissionStrategy(threshold=..., counter=KVCounter(...))``.
            The module turns the strategies of the tables it fuses into one
            ``MultiTableAdmitter``, which is what actually decides.
        admission_counter : Optional[Counter], optional
            Deprecated, and warns when set. Pass the counter to the strategy
            that uses it instead.
        evicted_item_mode : EvictedItemMode, optional
            How the *last-tier* storage handles an item it evicts. ``DISCARD``
            (default) drops evicted keys with zero overhead. ``RETAIN_KEY`` retains
            the keys it evicts so they can be read back with ``pop_evicted_keys``.
            Only the final tier that truly discards a key records it -- intermediate
            cache / HBM tiers spill their evictions to the next tier and are not
            recorded. Records the (key, table_id) only, no value/score. Default
            ``EvictedItemMode.DISCARD``.
        output_dist : str, optional
            How a row-wise sequence (``EmbeddingCollection``) sharding moves keys to their
            owners and the looked-up rows back. ``"torchrec"`` (default): TorchRec's input and
            output dists. ``"hierarchical"``: DynamicEmb's hierarchical dist, which needs
            ``output_dist_capacity``, DynamicEmb built against NCCL >= 2.30.7 and tables of
            ``dist_type`` ``roundrobin`` or ``hash_roundrobin``; a call on it deduplicates
            the sharding's keys, whatever the sharder's ``use_index_dedup`` (tables that clip
            gradients need that flag). It carries no score per key (``PRECISE_CUSTOMIZED``
            tables raise ``ValueError``). ``"auto"``: ``"hierarchical"`` where the build has
            it, ``output_dist_capacity`` is set and no table is ``continuous``, takes a score
            per key or clips gradients without index dedup, else TorchRec's dists. See
            ``docs/hier_output_dist.md``, section 4.
        node_dedup : bool, optional
            With ``"hierarchical"`` (or ``"auto"``) across nodes, a node sends each key once
            and adds up its gradient rows before they cross back. ``False`` sends every
            occurrence. Default ``True``. See ``docs/hier_output_dist.md``, section 4.3.
        output_dist_capacity : Optional[Union[int, Dict[str, int]]], optional
            The capacities of ``"hierarchical"`` (and ``"auto"``), in keys: an int for both,
            or a dict of ``"lookup_keys"`` and, across nodes, ``"remote_keys"``. Other names
            raise ``ValueError``. Default ``None``. See ``docs/hier_output_dist.md``, section 5.
        optimizer_state_dtype : Optional[torch.dtype], optional
            Precision the optimizer state is kept at, and the ``optim_state_dtype`` a dump
            records. ``None`` (default) keeps each optimizer's default. Only
            ``EXACT_ROWWISE_ADAGRAD`` accepts anything besides the table's ``embedding_dtype``:
            its accumulator defaults to ``torch.float32`` and may be set to the table's
            ``embedding_dtype``. See "Optimizer state precision". Tables differing in it are not
            grouped onto shared storage.
        
        Notes
        -----
        See ``DynamicEmb_APIs.md`` (this file) and ``dynamicemb/planner/planner.py`` for planner integration.
        """

        embedding_dtype: Optional[torch.dtype] = None
        dim: Optional[int] = None
        max_capacity: Optional[int] = None
        evict_strategy: DynamicEmbEvictStrategy = DynamicEmbEvictStrategy.LRU
        local_hbm_for_values: int = 0  # in bytes
        device_id: Optional[int] = None

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
        score_strategy: Union[
            DynamicEmbScoreStrategy, Tuple[DynamicEmbScoreStrategy, ...]
        ] = DynamicEmbScoreStrategy.TIMESTAMP
        bucket_capacity: int = 128
        safe_check_mode: DynamicEmbCheckMode = DynamicEmbCheckMode.IGNORE
        global_hbm_for_values: int = 0  # in bytes
        external_storage: Storage = None
        index_type: Optional[torch.dtype] = None
        admit_strategy: Optional[AdmissionStrategy] = None
        admission_counter: Optional[Counter] = None  # deprecated
        evicted_item_mode: EvictedItemMode = EvictedItemMode.DISCARD
        output_dist: str = "torchrec"
        node_dedup: bool = True
        output_dist_capacity: Optional[Union[int, Dict[str, int]]] = None
        optimizer_state_dtype: Optional[torch.dtype] = None

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
        and the dtype of the value is the table's own ``embedding_dtype`` (float32, float16 or bfloat16) -- values are stored at the
        precision the table holds them at, not widened to float32. Each optimizer state is also treated as a dynamic embedding table,
        stored at the optimizer's state precision: the table's ``embedding_dtype``, except row-wise Adagrad's accumulator, which
        defaults to float32 (see ``optimizer_state_dtype``).

        The value files carry no header, so the per-table meta JSON records ``embedding_dtype``, ``embedding_dim`` and
        ``optim_state_dtype``; :func:`DynamicEmbLoad` reads the files by them. A checkpoint written before those keys existed is read
        as float32 with the row width recovered from the value file's size.

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
        and the dtype of the value is the table's own ``embedding_dtype``. Each optimizer state is also treated as a dynamic embedding
        table, stored at the optimizer's state precision: the table's ``embedding_dtype``, except row-wise Adagrad's accumulator,
        which defaults to float32 (see ``optimizer_state_dtype``).

        The value files are read at the precision and row width the per-table meta JSON records (``embedding_dtype`` /
        ``embedding_dim`` / ``optim_state_dtype``). A checkpoint written before those keys existed is read as float32 with the row
        width recovered from the value file's size. A dim that disagrees with the runtime table is an error; a differing precision is
        converted on load, with a warning.

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

The meaning of the threshold depends on the table's `score_strategy`:
- Strategies that carry a device-timestamp column — the single `TIMESTAMP` strategy, or a compound strategy that includes `TIMESTAMP` such as `(TIMESTAMP, LFU)` — produce a **time-based** incremental dump: only items whose last-access timestamp crosses the threshold (i.e. items that were touched/changed since the reference time) are dumped. Use the value returned by `get_score` as the reference threshold.
- Strategies without a timestamp column (e.g. `STEP`, single `LFU`) instead threshold on the **absolute score value**: items whose score is not less than the threshold are dumped. This is not a time-based increment.
- Tables whose strategy contains `NO_EVICTION` (alone or as `(NO_EVICTION, CUSTOMIZED)`) are **skipped** with a `UserWarning` and are absent from the result; nothing is drained from them. They have no timestamp to take a delta by, and their erased keys are the user's to manage.

> **Limitation — `dist_type`:** `incremental_dump` supports only `roundrobin` and `hash_roundrobin` sharding. A table sharded with `dist_type="continuous"` raises `NotImplementedError`. The returned `slot_index` is meant for precise `replay_increment`, which reconstructs each key's owning rank from the key via `(key or hash(key)) % world_size`; `continuous` uses a different, range-based key→rank mapping that this path does not implement. Use `roundrobin` or `hash_roundrobin` if you need incremental dump.

    ```python
    #How to import
    from dynamicemb.incremental_dump import incremental_dump

    #API arguments
    def incremental_dump(
        model: torch.nn.Module,
        score_threshold: Union[int, Dict[str, Dict[str, int]]],
        pg: Optional[dist.ProcessGroup] = None,
    ) -> Dict[str, "DeltaDumpResult"]:
        """Dump the model's embedding tables incrementally based on the score threshold. The index-embedding pair whose score is not less than the threshold will be returned.

        Args:
            model(nn.Module):The model containing dynamic embedding tables.
            score_threshold(Union[int, Dict[str, Dict[str, int]]]):
                int: All embedding table's score threshold will be this integer. It will dump matched results for all tables in the model.
                Dict[str, Dict[str, int]]: the first `str` is the name of embedding collection in the model. 'str' in Dict[str, int] is the name of dynamic embedding table, and `int` in Dict[str, int] is the table's score threshold. It will dump for only tables whose names present in this Dict.
            pg(Optional[dist.ProcessGroup]): optional. The process group used to control the communication scope in the dump (the all_gather of keys/values/slot_index). Defaults to None.

        Returns
        -------
        Dict[str, DeltaDumpResult]:
            ``{collection_path: DeltaDumpResult}`` -- one ``DeltaDumpResult`` per
            embedding collection. Each ``DeltaDumpResult`` holds column-aligned
            per-table lists (element ``i`` refers to ``table_names[i]``):

            - ``table_names: List[str]`` -- the dumped table names.
            - ``keys: List[torch.Tensor]`` -- per-table matched keys on host.
            - ``values: List[torch.Tensor]`` -- per-table `[N, dim]` embeddings on
              host. **Embeddings only**: the rest of the stored row is in
              `optimizer_states`, and concatenating the two along dim 1
              reproduces the row as the table holds it.
            - ``optimizer_states: List[Optional[torch.Tensor]]`` -- per-table
              optimizer state on host, at the **same width the file checkpoint
              uses** — narrower than the runtime row for rowwise Adagrad, whose
              fused layout reserves 16 bytes per row but fills one scalar.
              ``None`` for a table whose optimizer keeps no per-row state, e.g.
              plain SGD.
            - ``scores: List[torch.Tensor]`` -- per-table `[N, num_scores]` score
              words on host, in the table's configured (logical) column order.
              Timestamp columns hold an **age** (`current_score - score`), not a
              raw timestamp, since `%globaltimer` is per device and resets across
              boots; a consumer rebases them onto its own clock. Every other
              column (LFU frequency, STEP, CUSTOMIZED) is
              carried verbatim.
            - ``erased_keys: List[Optional[torch.Tensor]]`` -- per-table keys an
              explicit erase removed, for tables whose ``evicted_item_mode``
              erase asked to have recorded (each `erase` call passes its own
              `EvictedItemMode`), so always present and possibly empty. These
              are the removals `replay_increment` applies.
            - ``evicted_keys: List[Optional[torch.Tensor]]`` -- per-table retained
              evicted keys on host for tables with ``evicted_item_mode=RETAIN_KEY``,
              else ``None``. Returning them drains that table's retained-evicted
              buffer (each evicted key reported once across successive calls).
            - ``meta: List[Dict[str, Any]]`` -- per-table metadata, a flat dict:
                - ``"current_score": int`` -- the table's current score after this
                  dump; usable as the next forward's score and as the next
                  ``incremental_dump`` threshold.
                - ``"slot_index": torch.Tensor`` -- int64 host tensor aligned with
                  ``keys``; the storage slot each dumped key occupies (for
                  ``replay_increment``). A storage keeps a value row per key,
                  apart from the slot, so it is an ``[N, 2]`` int64 tensor: the key
                  slot and the value row.
                - ``"num_rows": List[int]`` -- how many value rows each source
                  rank had handed out, in rank order; the receiver picks the entry
                  of its own shard. A delta names only the rows that changed, so
                  replay takes the row counter from here.
                - ``"erased_slot_index": List[torch.Tensor]`` -- per source rank,
                  the ``[M, 2]`` int64 (slot, row) of every slot a retained erase left
                  reclaimed since the last dump (an erased slot keeps its row), so a
                  replica can recreate them. Only complete if the erases were
                  retained (`RETAIN_KEY`); see ``erase_untracked``.
                - ``"erase_untracked": bool`` -- the table erased keys without
                  retaining them (on any rank of the dumping group), so
                  ``erased_slot_index`` misses the slots they left and the delta
                  cannot be replayed exactly. Sticky. ``replay_increment`` refuses
                  such a delta.
                - ``"current_capacity": int`` -- key-map slots: the modulus that
                  decides a key's home bucket.
                - ``"row_capacity": Tuple[int, ...]`` -- value-buffer rows per
                  storage tier, which is what bounds a row write.
                - ``"bucket_capacity": int`` -- slots per hash bucket.
                - ``"num_scores": int`` -- score words per key; part of the
                  slot layout ``replay_increment`` compares against.
                - ``"world_size": int`` -- ranks the table was sharded across.
                - ``"table_options": DynamicEmbTableOptions`` -- the table config.
        """
    ```

More usage please see [test](https://github.com/NVIDIA/recsys-examples/blob/main/corelib/dynamicemb/test/unit_tests/incremental_dump/test_distributed_dynamicemb.py)

## replay_increment

**Background**
`incremental_dump` produces a delta; `replay_increment` is the other half of that pipeline — it writes the delta back into a model. The typical deployment is delta replication: a training job dumps periodically, the delta is shipped to a serving replica, and the replica replays it to catch up without reloading a full checkpoint.

**Behavior**
For every table in the delta, `replay_increment` erases the table's `erased_keys` (so the target converges to the source) and then upserts `keys` / `values` at the slots in `meta["slot_index"]`. `evicted_keys` is never applied: the key that took an evicted key's slot is in the same delta and overwrites it.

*Write-back is by slot.* Every key is written at the slot and value row it held in the source table, leaving the target layout-identical to it. A key can only be found inside its own home bucket, and that bucket is `hash(key) % capacity / bucket_capacity`, so this requires the target's layout to match the source's. `replay_increment` compares the delta's `meta` (`current_capacity`, `row_capacity`, `bucket_capacity`, `num_scores`, `world_size`, and the `table_options` fields `score_strategy` / `dim` / `dist_type`) against the target table, and checks that every column has the shape the write path needs. Both happen for **all** of the delta's tables before any of them is written, so a mismatch anywhere raises `ValueError` naming the first offending field with the collection untouched — a delta spans a whole collection, and a partly applied one is worse to recover from than a rejected one. Configure the target to match the source, or rebuild it from a full checkpoint (`DynamicEmbLoad`) instead.

The same rule is enforced per key inside the kernel — a slot that does not land in its key's home bucket raises rather than dropping the key.

> **Precondition:** writing a key at its source slot **overwrites whatever occupies that slot** in the target. That is what makes a replica converge — if the source evicted key `B` to make room for `A`, a same-capacity replica must do the same. It also means a replayed table must be built *only* by loading/replaying from its source: a table that also takes independent writes can lose a key whose slot a delta key claims.

*Rows and erases.* A storage keeps a value row per key, apart from the key's slot, and an erased key leaves its slot reclaimed, still owning its row. For a replica to keep the same rows, an erase that should be replayed has to be **retained**: `erase_keys(..., mode=EvictedItemMode.RETAIN_KEY)`. A table that erased keys **without** `RETAIN_KEY` (on any rank of the dumping group) -- by `erase_keys`, or by [`erase_by_score_async`](#erase_by_score_async), which retains nothing -- is marked, `meta["erase_untracked"]` is set, and `replay_increment` raises `ValueError` for its deltas; the mark is sticky, so rebuild the replica from a full checkpoint after such an erase. Within a delta the order is: erase by key, recreate the reclaimed slots (`meta["erased_slot_index"]`), then upsert.

*The source growing ends the delta chain.* A target's capacity does not change on its own. Once the source table expands (rehash), its capacity no longer matches the target's, and every delta dumped from then on is rejected (`capacity` mismatch). Rebuild the target from a full checkpoint of the grown source and replay what comes after it.

*`NO_EVICTION` tables* are not replayed: `incremental_dump` skips them, and a delta that names one is refused with `NotImplementedError`.

*Scores and optimizer state travel under `ReplayContent`,* which asks for both by default. Drop `SCORE` and a restored key is scored as if it had just been inserted into the target, so the replica ranks its own future evictions by when it received each key rather than by how the source ranked it — the two can evict in different orders, but never disagree on the value of a key they both hold. Drop `OPTIMIZER_STATE` and a key keeps its state only if it already occupies the target row. 

*Sharding.* Replay keeps only the keys this rank owns, recomputing ownership from the key with **this** model's global world size — the fan-out the tables were sharded over. There is deliberately no process-group argument: replay is local (filter, then write), so a group could only narrow the modulus and mis-route every key. A delta gathered over a process group (`incremental_dump(..., pg)`) holds the whole group's keys and so fans out correctly when the same delta is handed to every rank. A per-rank delta (`pg=None`) holds only the producing rank's keys and should be replayed there — replaying it on another rank is not an error, every key simply belongs to someone else and is skipped, which `ReplayStats.skipped` reports. Like `incremental_dump`, only `roundrobin` and `hash_roundrobin` are supported; `continuous` raises `NotImplementedError`.

*Optimizer state* is not part of a delta. A key that already occupies its target row keeps its optimizer state; a row taken over from another key (or a brand-new one) is reset to the table's initial optimizer state.

    ```python
    #How to import
    from dynamicemb import replay_increment

    #API arguments
    def replay_increment(
        model: torch.nn.Module,
        deltas: Dict[str, "DeltaDumpResult"],
        content: ReplayContent = ReplayContent.ALL,
    ) -> Dict[str, Dict[str, "ReplayStats"]]:
        """Write incremental_dump results back into a model's dynamic embedding tables.

        Args:
            model(nn.Module): The model containing dynamic embedding tables.
            deltas(Dict[str, DeltaDumpResult]): `incremental_dump`'s return value, keyed by embedding-collection path. Collections or tables the model does not have are skipped with a warning.
            content(ReplayContent): what travels with each key besides its embedding, which always does — optimizer state, score words, or both. Defaults to both; `ReplayContent.EMBEDDING_ONLY` for neither.

        Returns
        -------
        Dict[str, Dict[str, ReplayStats]]:
            `{collection_path: {table_name: ReplayStats}}`, where `ReplayStats` has:

            - `upserted: int` -- keys written back at their source slot.
            - `erased: int` -- keys actually removed before the upsert, counted as really removed (a delta can name keys this replica never held).
            - `skipped: int` -- delta keys this rank does not own.

        Raises
        ------
        ValueError: a target table's layout does not match the source's, or a delta is missing the per-key data replay needs.
        NotImplementedError: a table is sharded with `dist_type="continuous"`.
        """
    ```

Example — replicate a training model's deltas into a serving model:

    ```python
    from dynamicemb import replay_increment
    from dynamicemb.incremental_dump import get_score, incremental_dump

    threshold = get_score(train_model)          # reference point for the next window
    ...                                          # train for a while
    deltas = incremental_dump(train_model, threshold, pg)
    threshold = {c: {r.table_names[i]: r.meta[i]["current_score"]
                     for i in range(len(r.table_names))}
                 for c, r in deltas.items()}     # threshold for the NEXT dump

    stats = replay_increment(serve_model, deltas)       # raises if layouts differ
    for collection, per_table in stats.items():
        for name, s in per_table.items():
            print(collection, name, s.upserted, "keys replayed at their source slot")
    ```

## pop_evicted_keys

**Background**
When a table's last-tier storage is full, evicting a key drops it from the system entirely. With `evicted_item_mode=RETAIN_KEY` (see [DynamicEmbTableOptions](#dynamicembtableoptions)), the last tier instead retains the keys it evicts so they can be read back -- e.g. to feed a downstream key-value store, a cold-tier archive, or an offline pipeline.

**Behavior**
Returns, per table, the keys evicted since the previous call, deduplicated within a table. This is a read-and-clear (incremental) operation: each evicted key is reported exactly once across successive calls, and returning a table's keys drains its retained-evicted buffer on this rank. Tables without `evicted_item_mode=RETAIN_KEY` are omitted from the result.

Keys removed by an *explicit erase* are not here — they live in a separate buffer read by [pop_erased_keys](#pop_erased_keys). The two are kept apart because a consumer usually wants different things from them: an eviction says the table ran out of room, an erase says someone asked for the key to go.

    ```python
    #How to import
    from dynamicemb import pop_evicted_keys

    #API arguments
    def pop_evicted_keys(
        model: torch.nn.Module,
        table_names: Optional[Dict[str, List[str]]] = None,
        pg: Optional[dist.ProcessGroup] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Return (and clear) the keys evicted and retained by last-tier storage, per table.

        Only tables created with evicted_item_mode=RETAIN_KEY are included; all other tables are omitted.

        Args:
            model(nn.Module): the model containing dynamic embedding tables.
            table_names(Optional[Dict[str, List[str]]]): optional filter, keyed by
                embedding-collection path -> [table_name, ...]. None pops every
                retain-enabled table in the model.
            pg(Optional[dist.ProcessGroup]): optional. None returns each rank's LOCAL
                evicted keys (row-wise sharded, hence disjoint across ranks; zero
                communication). When given, keys are all_gathered within pg so every
                rank in the group receives the group-wide union. Clearing always
                affects only this rank's buffer, regardless of pg.

        Returns
        -------
        Dict[str, Dict[str, torch.Tensor]]:
            {collection_path: {table_name: keys}} where keys is a 1-D host
            tensor of table-unique evicted keys. Empty dict if the model has no
            tables retaining evictions.
        """
    ```

## pop_erased_keys

**Background**
The counterpart of [pop_evicted_keys](#pop_evicted_keys) for the other way a key leaves a table: an explicit removal, rather than the table running out of room.

Unlike evictions, this is **not** a table setting. Retaining evictions has to be decided up front — it swaps in a collecting insert kernel and costs an extra kernel output plus a device sync on every evicting insert. An erase already holds its keys, so recording them costs a copy and nothing has to be prepared; each `erase` call therefore passes its own `EvictedItemMode` and decides for itself.

**Behavior**
Same arguments and the same read-and-clear semantics as `pop_evicted_keys`, draining the erased-key buffer instead. Tables whose mode retains no erases are omitted.

Only these keys are removals a replica has to perform for itself, which is why `replay_increment` applies `erased_keys` and never `evicted_keys`.

    ```python
    #How to import
    from dynamicemb import pop_erased_keys

    #API arguments
    def pop_erased_keys(
        model: torch.nn.Module,
        table_names: Optional[Dict[str, List[str]]] = None,
        pg: Optional[dist.ProcessGroup] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Return (and clear) the keys an explicit erase removed, per table.

        Every table is included -- whether an erase was recorded is that
        erase call's decision, not the table's, so there is nothing to
        filter on. Arguments are identical to pop_evicted_keys.
        """
    ```

## erase_by_score_async

**Background**
Remove, in the background of training, every key whose score is below a threshold you give: an expiry ("drop what has not been seen for a week"), or making room in a table that never evicts. For a table with [`NO_EVICTION`](#dynamicembscorestrategy) it is a way to free keys: the slots and value rows an erase frees are what the keys that come next take, so a table that is erased as it is filled stops growing once it holds what the window holds.

The call returns at once with a handle; the scan runs on a thread of its own, on a CUDA stream of its own, beside the forward, the backward and the prefetch of the training.

**Behavior**
- *Which tables.* Those whose score strategy has the one you name as a member, so `CUSTOMIZED` takes `CUSTOMIZED` and `(NO_EVICTION, CUSTOMIZED)` tables, and `PRECISE_CUSTOMIZED` takes `PRECISE_CUSTOMIZED` and `(NO_EVICTION, PRECISE_CUSTOMIZED)` tables. A table that is `NO_EVICTION` alone keeps no score and is never picked; naming `NO_EVICTION` is an error. With no table that qualifies the call is not an error: the handle is done at once and says `(0, 0)`. A table that keeps two score words (`(TIMESTAMP, LFU)`, in either order) is taken by both `TIMESTAMP` and `LFU`, and the one you name is the word compared: `TIMESTAMP` its last-access timestamp (an expiry, whatever the key's frequency), `LFU` its frequency (whatever its age). The other word is not looked at, and an erased key loses both. `table_names` narrows the call to the tables you name, each of which must qualify; a name that is in no module, a table that does not qualify, or an empty list is an error, raised before anything starts. With `strict=False` a name that no module of this rank holds is skipped instead, so the same call works on every rank when a table is not on every rank (`TABLE_ROW_WISE`); a held name that does not qualify is still an error. Tables that need different thresholds take one call each (example below).
- *The threshold.* A key is erased when its score is **strictly below** `threshold`, an int in `[0, 2**64)`, in the unit the score has: the number you set or gave per key, a step, a frequency, or for `TIMESTAMP` the device clock, in ns, which you read with `dynamicemb.device_timestamp()` on the tables' device. An `LFU` frequency is the raw access count: it never decays (a `score_function` only changes how evictions rank), and a key just admitted starts at its first count, so an erase by `LFU` alone also removes the keys that came in recently. A frequency floor usually wants an age condition with it, which this call does not combine. For `(NO_EVICTION, CUSTOMIZED)` and `(NO_EVICTION, PRECISE_CUSTOMIZED)` tables the scores of the keys you keep must be at least 1: an erased slot is cleared to 0, so a key scored 0 ties with one.
- *Layouts.* Supported: GPU-only tables, and a cache over host storage (only the storage behind the cache is erased; the cache is neither read nor changed, so a key it still holds can be written back when the cache lets it go). Refused when a table of the module would be scanned: host-only tables, `HybridStorage`, and any layout with an `external_storage`.
- *What is spared.* Best effort. A key a prefetched batch is using -- from its prefetch to its backward -- is not erased even if its score is below the threshold; it is counted as spared instead. A key whose score is raised or lowered while the scan runs is erased or not by the score it has when the scan reaches it. If the table is expanded while it is scanned (the training inserted more keys than it has room for), that table is scanned again from its first bucket.
- *What it overlaps.* Forward, backward and prefetch. **Not** dump, load, `incremental_dump`, `replay_increment`, `pop_evicted_keys`, `pop_erased_keys` or evaluation: call `wait()` on the handle first; this is not checked. The prefetch of a batch waits for the chunk the scan is in and the few it has in flight, not for the whole scan.
- *Which stream.* By default each module makes one stream of its own and keeps it, and the scan overlaps the training. You can pass a stream of the tables' device. The one the training runs on, the default stream or the prefetch stream also work, but the scan is then queued among the training's kernels, does not overlap them, and a `UserWarning` says so.
- *Chunks.* The scan goes a chunk of at most `chunk_slots` slots at a time, with at most `max_in_flight` chunks launched and not yet finished on the device. They set how much the scan disturbs the training and how long it takes: smaller and fewer disturb it less and take longer. The defaults (`2**22` slots, `2`) were measured on an H100 with a table of `2**29` slots at half load and training steps of 0.7 ms, scans running back to back: a step was slowed by about 10%; with `2**24` slots it was slowed by 110%, with `2**26` it took about eight times as long, and with no limit on chunks in flight (`max_in_flight` large) a step took about six times as long as without a scan. A smaller chunk makes the scan take longer (in the same 300 steps, 13 to 15 scans finished with `2**22` slots and 115 with `2**24`). Those scans only read; erasing is slower (134 million keys removed in 21 ms, with chunks of `2**24` slots, against 3.4 ms to look at the table), and the disturbance while it erases was not measured. A real training step is longer than the 0.7 ms used, so the same disturbance should cost it proportionally less; that was not measured either. Both parameters are worth tuning for your tables.
- *Ranks.* The call is local: each rank erases what it holds, there is no collective in it, and `result()` is what this rank erased. Add it up across the ranks with a collective if you need the total.
- *What it does not do.* The keys it erases are not recorded for `pop_erased_keys` or `replay_increment`. So a table it removed any key from is marked (`meta["erase_untracked"]` of an [incremental_dump](#incremental_dump) delta, sticky), and `replay_increment` refuses that table's deltas: a replica would otherwise keep the keys the source erased, and their rows. A call that removes nothing does not mark. `NO_EVICTION` tables are not replayed at all. After such an erase, rebuild the replica from a full checkpoint.

The handle (`EraseHandle`, also importable from `dynamicemb`):

| Call | Meaning |
| --- | --- |
| `done()` | Whether the erase has finished on the device. Does not block. |
| `wait()` | Blocks the calling thread until it has finished on the device. |
| `wait_stream(stream=None)` | Makes `stream` (default: the current one) wait for it, blocking the calling thread only until everything has been launched. |
| `result()` | `(erased, spared)`: keys removed, and keys left because a batch held them. Blocks until it has finished. |

Anything the background thread raised is raised by whichever of these you call. Given a model, the call finds every dynamic embedding table in it the way `DynamicEmbDump` does and returns one handle for all of them (its `result()` adds theirs up). Everything is checked before anything starts, so an error leaves nothing erased.

    ```python
    #How to import
    from dynamicemb import erase_by_score_async, EraseHandle

    #API arguments
    def erase_by_score_async(
        model: nn.Module,
        score_strategy: DynamicEmbScoreStrategy,
        threshold: int,
        stream: Optional[torch.cuda.Stream] = None,
        chunk_slots: int = 1 << 22,
        max_in_flight: int = 2,
        table_names: Optional[Sequence[str]] = None,
        strict: bool = True,
    ) -> EraseHandle:
        """Erase, in the background of training, the keys whose score is below ``threshold``
        from every dynamic embedding table of ``model`` that scores its keys by ``score_strategy``,
        or only from the tables named in ``table_names``.

        Parameters
        ----------
        model : nn.Module
            The model containing dynamic embedding tables.
        score_strategy : DynamicEmbScoreStrategy
            Which tables to erase from, and how their scores read. A table qualifies when its
            score strategy has this as a member; in a ``(TIMESTAMP, LFU)`` table it is also
            the score word compared. ``NO_EVICTION`` alone is an error.
        threshold : int
            A key is erased when its score is strictly below it. In [0, 2**64).
        stream : Optional[torch.cuda.Stream]
            The stream the erase launches on, on the tables' device. Default: one of its own per
            module, which lets the erase overlap the training.
        chunk_slots : int
            The most slots one chunk of the scan covers, in whole buckets. Default 2**22.
        max_in_flight : int
            At most this many chunks launched and not yet finished on the device. Default 2.
        table_names : Optional[Sequence[str]]
            Erase only from these tables, by name; each must qualify for ``score_strategy``.
            Default ``None``: every table that qualifies. A name in no module, a table that does
            not qualify, and an empty list are errors.
        strict : bool
            ``False``: a name no module of this rank holds is skipped, not an error, for tables
            that are not on every rank (``TABLE_ROW_WISE``). A held name that does not qualify
            is still an error. Default ``True``.

        Returns
        -------
        EraseHandle
            done(), wait(), wait_stream(stream) and result() -> (erased, spared).
        """
    ```

The same call is a method of `BatchedDynamicEmbeddingTablesV2`, `module.erase_by_score_async(score_strategy, threshold, stream=None, chunk_slots=..., max_in_flight=..., table_names=None)`, for the tables of that one module. It has no `strict`: every name must be a table of the module.

    ```python
    # Keep the last 1000 steps. Every key is given the step it was seen in, as its score
    # (a table of (NO_EVICTION, PRECISE_CUSTOMIZED), scores at least 1).
    handle = None
    for step, batch in enumerate(loader, start=1):
        loss = model(batch).sum()
        loss.backward()
        optimizer.step()
        if step % 100 == 0 and step > 1000:
            if handle is not None:
                handle.wait()            # the previous one is done before another starts
            handle = erase_by_score_async(
                model, DynamicEmbScoreStrategy.PRECISE_CUSTOMIZED, threshold=step - 1000
            )

    # Before a dump: the erase must be finished.
    if handle is not None:
        erased, spared = handle.result()   # this rank's
    DynamicEmbDump(path, model)
    ```

Tables that need different thresholds take one call each, with `table_names`. Here two `(TIMESTAMP, LFU)` tables expire keys at different ages: `item_id` after a day without being seen, `creator_id` after 90 days. The calls on one module run one after another, in the order they were made, and each handle has its own counts. For counts per table, make one call per table: each table is scanned once either way.

    ```python
    from dynamicemb import DynamicEmbScoreStrategy, device_timestamp, erase_by_score_async

    DAY_NS = 24 * 3600 * 10**9
    now = device_timestamp()   # this rank's device clock, in ns
    short = erase_by_score_async(
        model, DynamicEmbScoreStrategy.TIMESTAMP, now - DAY_NS, table_names=["item_id"]
    )
    long = erase_by_score_async(
        model, DynamicEmbScoreStrategy.TIMESTAMP, now - 90 * DAY_NS, table_names=["creator_id"]
    )
    # Before a dump: both must be finished.
    erased_items, _ = short.result()
    erased_creators, _ = long.result()
    DynamicEmbDump(path, model)
    ```

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

Under CUSTOMIZED mode, users have to set the scores for each table. This holds for a table configured with `(NO_EVICTION, CUSTOMIZED)` (either order) as well; `set_score` is refused for a table whose score part is anything else.
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

A counter maps a key to an accumulated count. A strategy that admits by
frequency owns one; the framework only carries it, reporting its memory and
writing it into checkpoints, which is what `AdmissionStrategy.state()` hands
over. Custom counters inherit `Counter`.

`add` and `erase` are called with the arrays of a step: per-key arrays of a
capacity `N` and `num_valid`, the number of live leading entries (`None`: all
of them). Only those entries may be read, what the methods return has the
capacity `N`, and neither may wait for the device; see
[Writing a Storage, Cache, Counter or MultiTableAdmitter](#writing-a-storage-cache-counter-or-multitableadmitter).

```python
class Counter(abc.ABC):
    """Interface of a counter table which maps a key to a counter."""

    @abc.abstractmethod
    def add(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        frequencies: torch.Tensor,
        *,
        num_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Add frequencies to these keys and return their accumulated counts.

        The result has the length of ``keys``; its entries past ``num_valid``
        are undefined.
        """

    @abc.abstractmethod
    def erase(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        *,
        num_valid: Optional[torch.Tensor] = None,
    ) -> None:
        """Erase these keys (the first ``num_valid`` of them)."""

    @abc.abstractmethod
    def memory_usage(self, mem_type=MemoryType.DEVICE) -> int:
        """Consumption of one kind of memory."""

    @abc.abstractmethod
    def load(
        self,
        key_file,
        counter_file,
        table_id: int,
        dist_type: str,
        shard_index: Optional[int] = None,
        num_shards: Optional[int] = None,
    ) -> None:
        """Load one table's keys and counts from these files."""

    @abc.abstractmethod
    def dump(self, key_file, counter_file, table_id: int) -> None:
        """Dump one table's keys and counts to these files."""
```

**dynamicemb** provides `KVCounter`, which sizes one table's share of a counter.
It is configuration, not a `Counter`: the strategy holding it turns the shares
of the tables it serves into a single fused `MultiTableKVCounter` when a module
materializes it. The table is bucketized, and a full bucket evicts its
smallest-frequency key to make room, so size the capacity for the keys still
waiting to be admitted rather than for the embedding table.

```python
class KVCounter:
    def __init__(
        self,
        capacity: int,
        bucket_capacity: int = 1024,
        key_type: torch.dtype = torch.int64,
    )
```

## AdmissionStrategy and MultiTableAdmitter

Admission is two classes: what a table is configured with, and what a module
runs.

`AdmissionStrategy` is the configuration. It is inert -- it allocates nothing,
touches no device, and may be given to as many tables as you like -- and it
decides nothing. A fused module turns the configurations of the tables it fuses
into the one thing that decides by calling `create_admitter`, which is where
anything device-side comes into being.

```python
class AdmissionStrategy(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def create_admitter(
        cls,
        table_strategies: List["AdmissionStrategy"],
        device: torch.device,
    ) -> "MultiTableAdmitter":
        """The admitter these tables share, with its device state allocated.

        Called once, by the module, with one configuration per table it fuses.
        """

    @classmethod
    def one_configuration(cls, table_strategies) -> "AdmissionStrategy":
        """The single configuration these tables agree on.

        They were grouped on get_grouped_key, so they are already
        interchangeable and the first stands for all.
        """

    def get_grouped_key(self):
        """What has to match for two tables to share one fused module.

        The default is the instance itself, so only the same object groups.
        Override it to let equal but separately constructed configurations
        share a module, returning everything that changes the decision and, for
        what is resolved per table, only what the tables must agree on.
        """
```

`MultiTableAdmitter` is what the module runs, and holds everything deciding
takes. Only `admit` has to be implemented; the other two have defaults, so an
admitter that merely decides is one method.

`admit` is called with the arrays of a step: `keys`, `table_ids` and
`frequencies` have a capacity `N`, and `num_valid` is the number of live
leading entries (`None`: all of them). The mask it returns has the length of
`keys`; only its first `num_valid` entries are read. The `MultiTableInitializer`
that `non_admitted_initializer` returns is called the same way,
`initializer(buffer, keys, table_ids, indices, num_valid=count)`: `indices`
selects the rows to write, at the capacity of the batch, and only its first
`count` entries are live, so an initializer of your own has to take `num_valid`.
See
[Writing a Storage, Cache, Counter or MultiTableAdmitter](#writing-a-storage-cache-counter-or-multitableadmitter).

```python
class MultiTableAdmitter(abc.ABC):
    @abc.abstractmethod
    def admit(
        self,
        keys: torch.Tensor,
        table_ids: torch.Tensor,
        frequencies: Optional[torch.Tensor] = None,
        *,
        num_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Which of these missing keys may enter the table.

        ``frequencies`` is how often each key occurred in *this batch*, where
        the module counts occurrences at all; None means treat each key as one.
        It is an increment, not a running total -- any total is the admitter's
        own to keep.

        Only the first ``num_valid`` entries of the arrays are live (all of
        them if None); the rest is undefined and is never to be read. The
        result is a boolean mask over ``keys``.
        """

    def state(self) -> Optional[Counter]:
        """Persistent state the framework has to carry, or None.

        Whatever an admitter accumulates across steps has to be reported in
        memory accounting and survive a checkpoint. Return it here and the
        framework does both.
        """

    @property
    def non_admitted_initializer(self):
        """What writes the rows this admitter rejects, or None for the table's.

        A rejected key still takes part in the forward, so its row has to be
        written by something. Return a ``MultiTableInitializer`` to take that
        on; the module does the calling, with ``num_valid`` (only the first
        ``num_valid`` entries of ``indices`` are live).
        """
```

**dynamicemb** provides two pairs.

`FrequencyAdmissionStrategy` admits a key once it has been seen often enough,
and carries the counter it accumulates into, so admission is configured in one
place. Its admitter erases a key from that counter the moment it is admitted.

```python
class FrequencyAdmissionStrategy(AdmissionStrategy):
    def __init__(
        self,
        threshold: int,
        counter: Optional[KVCounter] = None,
        initializer_args: Optional[DynamicEmbInitializerArgs] = None,
    )
    # create_admitter -> MultiTableFrequencyAdmitter
```

`ProbabilisticAdmissionStrategy` admits a key by a coin toss instead, and its
admitter keeps no state at all -- `state()` is None and no counter is built for
it. Admission is consulted only for a key that is missing, so a key gets a
fresh toss every time it turns up and is not yet in the table: it takes
`1 / probability` appearances on average to get in. A batch holding a key `k`
times counts as `k` tosses.

```python
class ProbabilisticAdmissionStrategy(AdmissionStrategy):
    def __init__(
        self,
        probability: float,
        initializer_args: Optional[DynamicEmbInitializerArgs] = None,
    )
    # create_admitter -> MultiTableProbabilisticAdmitter
```

For both, `initializer_args` says how to initialize the rows the admitter
rejects; None leaves them to the table's own initializer, which is also the only
way to get UNIFORM bounds derived from a table's row count.

## quiesce

Lets a model on the hierarchical dist (`output_dist="hierarchical"`, or `"auto"`) free its communication buffers at once when it is dropped. Call it on every rank, after the model's last call and before the model is dropped.

    ```python
    #How to import
    from dynamicemb.shard import quiesce

    #API arguments
    def quiesce(model: torch.nn.Module) -> None:
        """
        For every hierarchical dist of ``model`` (a DistributedModelParallel, say), in the
        same order on every rank, its operator synchronizes the device and meets the other
        ranks over its own communicator. After it, no kernel of the sharding runs on any
        rank, and nothing is still being sent between ranks. So dropping the model frees its
        communication buffers at once.

        Without it, dropping the model waits until every node peer and rail peer has dropped
        its own, for at most ``min(timeout, 30 s)``. If one does not, the buffers stay until
        the process exits. A call of the model after ``quiesce`` undoes it. Shardings on
        TorchRec's dists are left as they are.
        """
    ```

## node_size

The GPUs of a node of the hierarchical dist (`output_dist="hierarchical"`, or `"auto"`) on the ranks of a process group: NCCL's LSA team there, which the dist reads when it builds a sharding's operator. Use it where you count with the dist's nodes, as `hier_needs` in `benchmark/benchmark_utils.py` does.

    ```python
    #How to import
    from dynamicemb.shard import node_size

    #API arguments
    def node_size(pg: Optional[torch.distributed.ProcessGroup] = None) -> int:
        """
        The ranks of NCCL's LSA team on the ranks of ``pg`` (the default group if None):
        those that reach each other's memory over NVLink. A host of 8 GPUs on H100
        machines, with the ranks numbered host by host; on GB200 and GB300 NVL72, with
        multi-node NVLink on, the job's GPUs of a rack when the job runs inside one rack.
        Not ``LOCAL_WORLD_SIZE``, the GPUs of a host.

        Collective: every rank of ``pg`` calls it, at the same point. It makes a short-lived
        NCCL communicator of the group's ranks, as the dist makes its own, reads its LSA
        team, ends it, and checks that the ranks read the same size and that it divides
        them. The answer is kept per group of ranks. Raises RuntimeError, the same on every
        rank, where a rank cannot read it (a build without the hierarchical dist, an NCCL
        error) or the ranks disagree; but an NCCL error inside the communicator's
        initialization on some ranks only leaves the other ranks waiting there, as with any
        NCCL communicator. Warns, on every rank, when the team splits the GPUs of a host
        between nodes (ranks not numbered host by host).
        """
    ```

# Functionality and User interface

## Distributed embedding training and evaluation

Once the model containing `EmbeddingCollection` is built and initialized through `DistributedModelParallel`, it can be trained and evaluated on each GPU like a single GPU, with torchrec completing communication between different GPUs.

The switching between training and evaluation modes should be consistent with `nn.Module`, while `training` in [DynamicEmbTableOptions](./dynamicemb/dynamicemb_config.py) is used to guide whether to allocate memory to optimizer states when builds the table.

Due to limited resources, the dynamic embedding table does not pre allocate memory for all keys. If a key appears for the first time during training, it will be initialized immediately during the training process. Please see `initializer_args` and `eval_initializer_args` in `DynamicEmbTableOptions` for more information.

## Hierarchical sequence dist

`output_dist="hierarchical"` in `DynamicEmbTableOptions` replaces TorchRec's input and output dists of row-wise `EmbeddingCollection` shardings with DynamicEmb's hierarchical dist.

- Set `output_dist_capacity` from what the model's batches need: `hier_needs` and `hier_capacities` in `benchmark/benchmark_utils.py` count it.
- Use tables of `dist_type` `roundrobin` (the default) or `hash_roundrobin`. With a `continuous` table, `"hierarchical"` raises `ValueError` at the first `input_dist()`, and `"auto"` takes TorchRec's dists ([docs/hier_output_dist.md](./docs/hier_output_dist.md), section 6).
- A call on the dist deduplicates the sharding's keys (index dedup), whatever the sharder's `use_index_dedup`; without that flag the sharding warns so once, at its first `input_dist()` ([docs/hier_output_dist.md](./docs/hier_output_dist.md), section 7.4). Tables that clip gradients need `use_index_dedup=True`: without it, `"hierarchical"` raises `ValueError` at the first `input_dist()`, and `"auto"` takes TorchRec's dists ([docs/hier_output_dist.md](./docs/hier_output_dist.md), section 6).
- The dist carries no score per key. With tables that take one (`PRECISE_CUSTOMIZED`), `"hierarchical"` raises `ValueError` when the model is sharded, and `"auto"` takes TorchRec's dists ([docs/hier_output_dist.md](./docs/hier_output_dist.md), section 6).
- A node of the dist is NCCL's LSA team, the GPUs that reach each other's memory over NVLink: a host of 8 GPUs on H100 machines, and on GB200 and GB300 NVL72 with multi-node NVLink the job's GPUs of a rack when the job runs inside one rack. The dist reads it at its first `input_dist()` ([node_size](#node_size) gives it too); `LOCAL_WORLD_SIZE`, the GPUs of a host, does not set it. Across racks it is what the NCCL you run gives: check it there ([docs/hier_output_dist.md](./docs/hier_output_dist.md), section 4.2).
- Keys may be `int32` or `int64`. The dist gives an `int32` key the owner of the `int64` key of its value, as DynamicEmb's bucketize, checkpoint loading and `replay_increment` do ([docs/hier_output_dist.md](./docs/hier_output_dist.md), section 10.1).
- Give every DynamicEmb table of a sharding the same `output_dist`, `output_dist_capacity` and `node_dedup`, and every rank the same options and environment variables ([docs/hier_output_dist.md](./docs/hier_output_dist.md), section 4.1).
- Set `NCCL_CROSS_NIC=0` for multi-node jobs, before the first NCCL communicator is created, so that NCCL connects GIN on the rails alone.
- Call [quiesce](#quiesce) on every rank before dropping the model.
- Every rank must make the same calls in the same order, as with NCCL collectives.
- The dist queues each step on the current stream and orders nothing across streams. TorchRec's training loop and train pipelines order its steps; a training loop of your own must order them the same way ([docs/hier_output_dist.md](./docs/hier_output_dist.md), sections 4.2 and 16).

Design, capacities, configuration, environment variables and fallbacks: [docs/hier_output_dist.md](./docs/hier_output_dist.md). Benchmark: [benchmark/README.md](./benchmark/README.md), section 3.

## Weighted EmbeddingBagCollection

Weighted-sum pooling for `EmbeddingBagCollection` tables backed by DynamicEmb is supported in training and eval (see
[DynamicEmbPoolingMode](#dynamicembpoolingmode)). Gradients are scaled per-position: `grad_embᵢ = wᵢ · grad_pooled`, verified against an exact SGD
reference in `test/unit_tests/test_weighted_pooled_embedding_v2.py` and end-to-end, sharded, forward and backward, against a twin TorchRec model in `test/unit_tests/test_twin_module.py` (`is_weighted=True`).

### Breaking API change: `per_sample_weights` split

`BatchedDynamicEmbeddingTablesV2.forward` previously took `per_sample_weights`, which was consumed as **LFU frequency counters**. That parameter is now split:

- `frequency_counters: Optional[Tensor]` — per-position LFU frequency counters (the old `per_sample_weights` semantics).
- `pooling_weights: Optional[Tensor]` — per-position float32 weights for weighted-SUM pooling (new; SUM only). The two are mutually exclusive.

Callers that passed `per_sample_weights=...` for LFU counting must pass `frequency_counters=...` instead.

`InferenceEmbeddingTable`'s `per_sample_weights` (inference/export path) is a separate API and is unchanged.

## Automatic eviction

The size of the table is finite, but the set of keys during training may be infinite. dynamicemb provides the function of automatic eviction, which constrains the size of tables reasonably when there is no available space. See `score_strategy` and `bucket_capacity` for more information.

## Caching and prefetch

dynamicemb supports caching hot embeddings on GPU memory, and you can prefetch keys from host to device like torchrec. Caching and prefetch work for both sequence mode (`NONE`) and pooling modes (`SUM`/`MEAN`). See `test_prefetch_flush_in_cache` in [test prefetch](./test/test_batched_dynamic_embedding_tables_v2.py) for usage examples.

## External storage

dynamicemb supports external storage once `external_storage` in `DynamicEmbTableOptions` inherits the `Storage` interface under [types.py](./dynamicemb/types.py). Its `find` and `insert` are called with the capacity-sized arrays and the `num_valid` of the lookup step: see [Writing a Storage, Cache, Counter or MultiTableAdmitter](#writing-a-storage-cache-counter-or-multitableadmitter).
Refer to demo `PyDictStorage` in [unit test](./test/test_batched_dynamic_embedding_tables_v2.py) for detailed usage.


## Table expansion

Users can specify the initial capacity of a table on a single GPU. When the table needs more space, the implementation may double the key_index_map and embedding table capacity (per table, only for tables that need it) before insert. Expansion is triggered in these paths so that insert does not fail for lack of capacity:

- **Prefetch HBM direct**: before inserting admitted keys into `DynamicEmbStorage`.
- **Cache write-back**: before writing evicted keys back to storage (only `DynamicEmbStorage` uses cache mode; `HybridStorage` does not).
- **Generic forward (DEFAULT mode)**: before `storage.insert()` when using `DynamicEmbStorage` or `HybridStorage` (for the latter, only the host tier is expanded).
- **HybridStorage.load**: before inserting keys evicted from HBM into the host tier.

**Trigger conditions:**

- **Non–NO_EVICTION**: When `max_load_factor` would be exceeded or (if set) at `max_capacity`, the table(s) that need it are doubled. See `init_capacity`, `max_load_factor`, `max_capacity` in `DynamicEmbTableOptions`.
- **NO_EVICTION**: The option `max_load_factor` is not used. The key_index_map and the embedding (value) buffer grow **separately**, each by what it has to hold. At creation the key_index_map has ceil(init_capacity/0.5) slots rounded up to whole buckets and the embedding buffer half that many rows.
  - The **key_index_map** is sized by the keys it holds, with a fixed effective max load factor of **0.5**: it grows when `keys held + new unique keys` exceed half its capacity. Erasing keys therefore makes room for new ones without growing it.
  - The **embedding buffer** is sized by the rows it has handed out. A key takes a row when it takes a slot that has never held one; an erased key's slot keeps its row, and a later key that takes that slot uses it. So after erases the rows handed out can outnumber the keys held, and the buffer grows when `rows handed out + new unique keys` exceed its rows -- at least doubling, but never beyond the number of slots in the key_index_map, since a slot owns one row and no more rows than slots can be handed out. A table whose keys keep being replaced therefore settles at one row per slot instead of growing without bound.
  The growth is not capped by `max_capacity`.
- **Loading a checkpoint** (`DynamicEmbStorage.load`): the table is sized for the keys **this rank keeps** from the file (counted from the key file with the rank's ownership filter), not for every key in it, since a checkpoint may hold other ranks' keys as well.


## Host syncs in a step

The number of unique keys of a batch is only known to the device, and the lookup step (the prefetch, the forward and the backward of a batch) does not read it back. The arrays of the step -- the unique keys, their table ids, the misses, the rows to write back, ... -- keep the length of the input, `N`, the *capacity*, and carry the number of live leading entries beside them as `num_valid`, a one-element int64 tensor on the device. Every op of the step is launched over `N`, reads `num_valid`, and does its work for the live entries only; what lies past them is undefined and never read. Compaction (the misses, the admitted keys, the evicted keys) writes its result at the same length with a count of its own, so a step whose keys all hit runs its miss path for zero keys instead of deciding on the host to skip it. The price is memory: the arrays of the step are allocated for the `N` input keys, not for the number of unique or missing keys.

This is the one path of every storage layout (HBM only, cache over storage, hybrid, host only), in training and in eval, and the classes you write follow the same contract: see [Writing a Storage, Cache, Counter or MultiTableAdmitter](#writing-a-storage-cache-counter-or-multitableadmitter).

A step makes no blocking host sync. It waits for the device only when:

- a table can grow (`init_capacity` below `max_capacity`, or `NO_EVICTION`): in training, deciding whether to expand needs the number of unique keys per table on the host; see [Table expansion](#table-expansion). A table that cannot grow never reads it.
- `evicted_item_mode` retains evicted keys: the eviction count is read to keep the keys.
- `DEMB_DETERMINISM_MODE` is set (see [Deterministic mode](#deterministic-mode)), or `DYNAMICEMB_DEBUG` (see [Environment Variables](#environment-variables)).
- a class you wrote reads the count itself, because it cannot work from the device (a store that keeps its rows in host memory, say): the wait is that class's own.
- the dump, load, replay, flush, pop and erase functions, and the per-key score check when `safe_check_mode` asks for it, which return host data or ask a question of the data.

A probabilistic admitter (`ProbabilisticAdmissionStrategy`) draws one random number per entry of the arrays of the step, that is one per input key of the batch, whatever the number of keys that are missing, and it does so on every step, also on one whose keys all hit, since the miss path runs for zero keys instead of being skipped. The draws follow `torch.manual_seed`; the generator advances by that many per step, so which key gets which draw depends on the size of the batch.

The over-subscription check of the prefetched keys (`Outstanding prefetched keys ... exceed cache capacity`) reads the number of unique keys only when the number of input keys of the batches in flight is larger than the cache.

The blocking syncs of a step can be counted with `--sync-count` of the many-collections benchmark, see the [benchmark README](./benchmark/README.md#3-benchmark-many-embedding-collections-with-a-small-dense-part).

## Writing a Storage, Cache, Counter or MultiTableAdmitter

A `Storage` (the `external_storage` of a table), a `MultiTableAdmitter` (what the `create_admitter` of your own `AdmissionStrategy` returns) and a `Counter` (what an admitter counts in, which its `state()` hands out) are the classes of [types.py](./dynamicemb/types.py) that you can implement yourself, and so is the `MultiTableInitializer` (of [initializer.py](./dynamicemb/initializer.py)) that such an admitter hands out as its `non_admitted_initializer`. A `Cache` cannot be plugged in: the module builds a `DynamicEmbCache` itself, and the step uses more of it than its interface, namely its `increment_counter` and `decrement_counter`, which pin and release a slot, and its table state. The `lookup` and `insert_and_evict` of the interface are listed here with the others because they follow the same contract. The lookup step calls all of them with the arrays of [Host syncs in a step](#host-syncs-in-a-step): per-key arrays of a capacity `N`, and `num_valid`, the number of live leading entries.

**This is a breaking change of these interfaces.** Their methods take a keyword-only `num_valid` (the `__call__` of a `MultiTableInitializer` takes a trailing `num_valid=None`), and what they return has the capacity of the input, with counts on the device, instead of being cut to the number of keys. A class written against the earlier signatures (exact-length tensors, no `num_valid`, `num_missing` and `num_evicted` as Python ints) fails at the first step with a `TypeError` for the unexpected `num_valid`. A method that accepts `num_valid` and ignores it does not fail, it works on entries that are undefined. There is no fallback that cuts the arrays to size for such a class, so migrate it as described below. Code that calls `find`, `insert`, `lookup` or `insert_and_evict` of the built-in `DynamicEmbStorage`, `HybridStorage` or `DynamicEmbCache` directly sees the same change in what it gets back: `num_missing` and `num_evicted` are tensors on the device, and the `missing_*` and `evicted_*` arrays have the length of the input, with only their first `num_missing` or `num_evicted` entries live (`int(count.item())` reads the count on the host; `num_valid` can be left out for exact-length arrays).

| Class | Method | Returns |
| --- | --- | --- |
| `Storage` | `find(unique_keys, table_ids, copy_mode, lfu_accumulated_frequency=None, key_scores=None, *, num_valid=None)` | `(num_missing, missing_keys, missing_indices, missing_table_ids, missing_scores, founds, output_scores, values)` |
| `Storage` | `insert(keys, table_ids, values, scores=None, preserve_existing=False, *, num_valid=None)` | `None` |
| `Cache` | `lookup(unique_keys, table_ids, lfu_accumulated_frequency=None, *, num_valid=None)` | `(score_out, founds, indices)` |
| `Cache` | `insert_and_evict(keys, table_ids, scores=None, *, num_valid=None)` | `(indices, num_evicted, evicted_keys, evicted_indices, evicted_scores, evicted_table_ids)` |
| `Counter` | `add(keys, table_ids, frequencies, *, num_valid=None)` | `accumulated_frequencies` |
| `Counter` | `erase(keys, table_ids, *, num_valid=None)` | `None` |
| `MultiTableAdmitter` | `admit(keys, table_ids, frequencies=None, *, num_valid=None)` | a boolean mask over `keys` |
| `MultiTableInitializer` | `__call__(buffer, keys, table_ids, indices, num_valid=None)` | `None` |

The other methods of the classes (dump, load, the dimensions, ...) are unchanged. What each argument and result means is in the docstrings of [types.py](./dynamicemb/types.py); the rules they all follow are these:

- **Live entries.** `num_valid` is a one-element int64 CUDA tensor: the number of live *leading* entries of every per-key array of the call. `None` means that all of them are live, an exact-length call such as the explicit APIs and tests make. Read, write and return only those entries. What lies past them is undefined -- memory that can hold anything, not necessarily a valid key or table id -- so never read it and never use it as an index.
- **Capacity-sized results.** Return per-key results with the capacity of the inputs. A method that selects some of the keys returns arrays of the capacity whose first `count` entries are the selection, with the count (`num_missing`, `num_evicted`) as a one-element int64 CUDA tensor: never a Python int, and also when `num_valid` is `None`. The missing keys of `find` are in the order of the input, and `missing_indices` are their positions in the input arrays; `founds` and `output_scores` have an entry per input key, and `values` has a row per input key (the row of a key that was not found is not written). The evicted keys of `insert_and_evict` are the keys that left the cache and any key that could not be placed, in no defined order: entry `j` of the four `evicted_*` arrays describes one key.
- **Keys.** The keys of `insert`, `insert_and_evict` and `Counter.add` are unique within each table, and so are those of `find` and `lookup` in a training step and in the evaluation of a pooled table. The evaluation of a table that is not pooled (`DynamicEmbPoolingMode.NONE`) looks up its raw ids, repeats included, and inserts nothing: answer every entry, a repeat as any other.
- **Arrays.** The step hands what you return on to the counted ops of `dynamicemb_extensions`, which check what they get. Every array is a CUDA tensor on the device of the inputs, 1-D and contiguous (`values`: 2-D with contiguous rows); keys and scores have 8-byte elements (`int64` or `uint64`), table ids and indices are `int64`, `founds` is `bool`.
- **Values.** The `values` of `find` is a tensor of `embedding_dtype()` with `max_embedding_dim()` columns for `CopyMode.EMBEDDING` and `max_value_dim()` for `CopyMode.VALUE`, laid out as `CopyMode` says (see [types.py](./dynamicemb/types.py)): the embedding of every table padded to `max_embedding_dim()` columns and, for `VALUE`, the optimizer states from that column on, whatever the table's own `embedding_dim`, because the step finds them there. The step writes into it (it initializes the rows of the new keys, resets their optimizer states and, in the backward, applies the gradients to the rows), so return a tensor of your own, never a view of your memory. The `values` of `insert` are the caller's, and their rows past `num_valid` are undefined: copy the rows you keep.
- **An initializer** (the `__call__` of a `MultiTableInitializer`) gets the `buffer` to write, with `keys` and `table_ids` alongside it, one entry per row of the buffer, and `indices`, the rows to write, at the capacity of the batch. Only the first `num_valid` entries of `indices` are live; never use the others as a row.
- **No wait.** Do not read a device tensor on the host: no `.item()`, `.cpu()` or `.tolist()`, and no torch op whose output size depends on device data (`nonzero`, `masked_select`, boolean-mask indexing, `unique`). `dynamicemb_extensions` has the building blocks: `flagged_compact(flags, inputs, num_valid)` (the entries where a flag is set, with their count: it returns `(count, positions, outs)`, where `positions[j]` is the input position of the `j`-th selected entry and `outs[k][j] == inputs[k][positions[j]]`, a `None` input staying `None`), `counted_gather(src, idx, num_valid)`, `counted_scatter(dst, idx, src, num_valid)`, `counted_scatter_fill(dst, idx, value, num_valid)` and `counted_count_true(flags, num_valid)`. Each does its work for the live entries only and reads nothing back; the index ops skip a negative index, so `torch.where(mask, idx, -1)` makes an index array select the entries where `mask` holds, with no compaction.
- **A class that cannot work from the device** -- a store that keeps its rows in a Python dict or in host memory -- reads the count once, `n = capacity if num_valid is None else int(num_valid.item())`, and works on the first `n` entries. That wait is the only one the step makes for the class, and it is the class's own cost.

An admitter that decides key by key needs nothing from `num_valid`: a mask over all `N` entries is correct, because only its live entries are read.

```python
class OddKeyAdmitter(MultiTableAdmitter):
    def admit(self, keys, table_ids, frequencies=None, *, num_valid=None):
        return keys % 2 == 1
```

To migrate a class that was written for exact-length tensors, add `*, num_valid=None` to its methods, and `key_scores=None` to a `find` that lacks it (it is in the signature above, and a table that takes a score per key passes it), take `n = keys.numel() if num_valid is None else int(num_valid.item())`, run the old code on `keys[:n]`, `table_ids[:n]` and the other per-key inputs cut the same way, and return what it produced padded to the capacity, with each count as a one-element int64 tensor on the device. For a `find` that is:

```python
import torch
from dynamicemb.types import Storage


class MyStore(Storage):
    def find(self, unique_keys, table_ids, copy_mode,
             lfu_accumulated_frequency=None, key_scores=None, *, num_valid=None):
        capacity = unique_keys.numel()
        n = capacity if num_valid is None else int(num_valid.item())  # waits for the device

        def pad(x):  # the first entries of an array of the capacity, on the device
            out = torch.empty((capacity, *x.shape[1:]), dtype=x.dtype,
                              device=unique_keys.device)
            out[: x.size(0)] = x
            return out

        # The old code, run on unique_keys[:n] and table_ids[:n], gives founds [n],
        # values [n, dim] and, for the m keys it did not find, their keys,
        # positions, table ids and scores: in the dtypes and the layout of the
        # rules above, and values a tensor of its own.
        ...
        num_missing = torch.tensor([m], dtype=torch.int64, device=unique_keys.device)
        return (num_missing, pad(missing_keys), pad(missing_indices),
                pad(missing_table_ids),
                None if missing_scores is None else pad(missing_scores),
                pad(founds), pad(output_scores), pad(values))
```

`PyDictStorage` in the [unit test](./test/test_batched_dynamic_embedding_tables_v2.py) is a complete store written against this contract: a Python dict in host memory.

## Erasing keys by score

A hash table has no dense index to slice, so removing the keys whose score is below a threshold -- an expiry, or room for new keys in a table that never evicts -- is a call of its own that scans the table. It runs in the background of training, on a stream and a thread of its own, and a prefetch waits for it for at most a chunk.

Please see `erase_by_score_async` in [APIs Doc](./DynamicEmb_APIs.md) for more information, and [DynamicEmbScoreStrategy](#dynamicembscorestrategy) for which strategies keep a score it can use.

## Dump/Load and Incremental dump

Dump/Load and incremental dump is different from general module in PyTorch, because dynamicemb's underlying implementation is a hash table instead of a dense `torch.Tensor`.

So dynamicemb provides dedicated interface to load/save models' states, and provide conditional dump to support online training.

Please see `DynamicEmbDump`, `DynamicEmbLoad`, `incremental_dump` in [APIs Doc](./DynamicEmb_APIs.md) for more information.

## Deterministic mode

When handling cache eviction and final table eviction, dynamicemb encounters randomness in the eviction of keys with the same score. To eliminate this uncertainty, dynamicemb provides a deterministic mode. Enabling this mode ensures that, under the same training script, the evicted keys will be determined.
This mode is enabled by setting the environment variable `DEMB_DETERMINISM_MODE`.

## Environment Variables

The following environment variables control runtime behavior of dynamicemb:

| Variable | Default | Effect |
|----------|---------|--------|
| `DYNAMICEMB_CSTM_SCORE_CHECK` | `1` | When set to `0`, suppresses warnings in `set_score` when a new score is less than the previous one (CUSTOMIZED eviction mode). |
| `DEMB_DETERMINISM_MODE` | unset | When set, enables deterministic eviction of keys with equal scores during cache eviction and final table eviction. |
| `DYNAMICEMB_DEBUG` | unset | When set to any non-empty value, enables extra runtime validation in `segmented_unique_cuda`. Copies `segmented_range` to CPU and verifies: (1) `segmented_range[0] == 0`, (2) `segmented_range[num_tables] == num_keys`, (3) monotonically non-decreasing. Raises an error with a descriptive message on violation. Intended for development and testing only; incurs one device-to-host copy per call. |
| `DYNAMICEMB_HIER_PROXY_MB` | `512` | Hierarchical dist (`output_dist="hierarchical"`): cap, in MiB, of the rows a GPU passes between its node and the other nodes (in both directions); a call whose rows do not fit moves them in laps of its ring. |
| `DYNAMICEMB_HIER_TIMEOUT_S` | the process group's timeout | Hierarchical dist: how long a kernel waits for a peer before it prints what it waited for and traps. |
| `DYNAMICEMB_HIER_OUT_CTAS`, `DYNAMICEMB_HIER_REDUCE_CTAS`, `DYNAMICEMB_HIER_PAYLOAD_KB` | – | Hierarchical dist: its CTA counts and the payload of a chunk across nodes: [docs/hier_output_dist.md](./docs/hier_output_dist.md), section 4.4, which also gives the build option of its trace. Its node is NCCL's LSA team, which no variable of DynamicEmb's sets ([node_size](#node_size)). |
