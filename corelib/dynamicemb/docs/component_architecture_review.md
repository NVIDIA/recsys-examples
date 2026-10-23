# dynamicemb 组件划分评审

状态：**仅分析，尚未改动任何代码。** 记录于 `lw-cache-wip`（HEAD 00020fd，已 rebase 到 upstream/main 1ba40f2）。

依据：阅读各模块的类、函数清单和几处关键代码（`key_value_table.py`、`batched_dynamicemb_function.py`、`scored_hashtable.py`、`types.py`、`batched_dynamicemb_tables.py`），没有逐行通读约 17,000 行 Python，所以个别细节可能有偏差；动手前应对相应位置重新核对。

## 1. 现有组件

| # | 组件 | 位置 | 行数 | 职责 |
|---|---|---|---|---|
| 1 | 配置 | `dynamicemb_config.py` | 1135 | 选项、score 策略、容量和字节数计算 |
| 2 | 哈希索引 | `scored_hashtable.py` + `src/table_operation` | 1582 + C++ | key 到 slot 的映射，带 score 和 ref_counter |
| 3 | 表状态 | `DynamicEmbTableState`（`key_value_table.py`） | dataclass | 索引、值缓冲、维度、optimizer、score、淘汰保留缓冲 |
| 4 | 分层存储 | `DynamicEmbStorage`（可带 `lower`）、`HybridStorage`（`key_value_table.py`） | 与其他内容共 4418 | HBM、host、PS 三层 |
| 5 | GPU 缓存 | `DynamicEmbCache` + `src/embedding_cache` | Python 薄封装 + C++ | `lookup_pin` / `allocate` / `publish` / `release` |
| 6 | 编排 | `batched_dynamicemb_function.py` | 1505 | prefetch 两条路径、forward/backward、准入 |
| 7 | 模块 | `BatchedDynamicEmbeddingTablesV2`（`batched_dynamicemb_tables.py`） | 2325 | 构造、forward、dump/load、replay、内存打印 |
| 8 | 周边 | admission、optimizer、initializer、incremental_dump、exportable_tables、dump_load | 各 300 到 1000 | |

## 2. 接口不清晰的地方

1. **`Storage` 基类太薄，实际靠 isinstance 和 hasattr 绕过。**
   - 基类只有 `find` / `insert` / `dump` / `load` 和维度查询。
   - 代码里有 `hasattr(storage, "collect_table_sizes")`、`hasattr(storage, "expand_if_need")`。
   - `isinstance(..., (DynamicEmbStorage, HybridStorage))` 在 `batched_dynamicemb_tables.py` 里出现 8 处以上。
   - `storage.lower` 不在基类里，却被编排层直接访问。
   - 结论：真正的接口是具体类的并集，不是基类。

2. **分层有两套写法。**
   - `HybridStorage` 把 HBM 和 host 写死成两层，自己有 800 多行。
   - `DynamicEmbStorage(lower=...)` 是链式。
   - PS 作为第三层加入后，存在两种组合层的方式，且互不可组合。

3. **未命中的处理重复了两份。**
   - `_prefetch_cache_path`（约 300 行）和 `_prefetch_hbm_direct_path`（约 250 行）都内嵌同一套流程：
     - 向 lower 要值，计算 `from_lower`、`admit_mask`；
     - 用初始值或 lower 的值填充；
     - 被淘汰的行写回 `lower.insert`。
   - 这是"层与层之间的策略"，不该内嵌在两个编排函数里。
   - 前面修复的哈希折叠 bug（`take_values_of_keys_evicted_in_the_same_insert`）就出在这条链路的辅助函数上。

4. **保护 slot 不被淘汰有两套机制。**
   - 缓存用 node pin 加 `EvictionGuard`。
   - HBM_DIRECT 用哈希表里的 `ref_counter`（prefetch 加、backward 减，insert/evict kernel 检查 `counter > 0`）。
   - 各自合理，但读代码的人要同时记两套。

## 3. 划分上的问题

- **`key_value_table.py` 4418 行，至少装了 6 件事**：表状态创建、层存储、缓存封装、checkpoint（dump/load/replay，约 1000 行）、扩容、淘汰保留缓冲与 flush / eval_lookup。
- **`DynamicEmbTableState` 是大杂烩。**
  - 同时装着索引、值缓冲、optimizer、score 状态、`no_eviction_next_index`、大小估计和 4 组 evicted/erased chunk。
  - 缓存只用到值缓冲，却要拿整个对象。
  - "淘汰保留"只属于最后一层，却长在每一层的状态里。
- **checkpoint 散在 6 个文件**：`scored_hashtable.dump/load`、`key_value_table.py`、`dump_load.py`、`incremental_dump.py`、`exportable_tables.py`、`batched_dynamicemb_tables.py`。score 的编码细节（`_scores_to_age`、`get_find_score_arg`）又分散在 config 和 key_value_table 之间。
- **模块类越界**：`BatchedDynamicEmbeddingTablesV2` 既做 TorchRec 对接，又管 replay 任务、文件路径编码和内存表格打印。
- **`dynamicemb_config.py` 混了三种东西**：用户选项、score 策略逻辑、容量/字节数算术。

## 4. 建议的重新划分

```
Index (ScoredHashTable)   只管 key→slot、score，不管值
RowStore                  值缓冲、维度、扩容、optimizer 状态布局        <- 从 TableState 拆出
Tier                      Index + RowStore，统一接口：
                          find / insert(返回 victims) / erase / export
    HbmTier, HostTier, PsTier（PS 适配器）
TierChain                 有序 Tier 列表 + 未命中策略
                          （问下一层、准入、初值、淘汰下沉）
Cache                     pin 协议，持有 RowStore 的行池，不碰 Index
Checkpoint                dump / load / incremental / replay，只依赖 Tier 的 export / import
Module                    只做装配和 nn.Module 对接
```

预期收益：
- `HybridStorage` 变成 `TierChain([Hbm, Host])`，PS 是再加一层，三层可任意组合。
- 两条 prefetch 路径里重复的未命中处理收成一处。
- `Storage` 基类收窄到真正被调用的方法，去掉 isinstance / hasattr。
- `evicted_item_mode` 只存在于最后一层。

## 5. 建议的实施顺序

1. **只搬家，不改行为**：把 `key_value_table.py` 按职责拆成 5 到 6 个文件（checkpoint、expansion、cache、tier storage 等）。风险最低，现有测试可直接验证。
2. **把重复的未命中处理抽成一处。** 最可能再出 bug 的地方，收益最直接。
3. **收窄 `Storage` 接口，消除 isinstance / hasattr**，并决定 `HybridStorage` 是否并入链式写法。
4. **拆 `DynamicEmbTableState`**，放最后。调用点最多，要等前三步让依赖关系清晰后再动。

## 6. 取舍与风险

- 第 4 步和统一的 `Tier` / `TierChain` 抽象改动大。按"没有调用者的机制不要"的原则，不建议一次性引入这套新抽象；先做第 2 步，看抽出来的东西是否真有多个使用者，再决定是否继续。
- 每一步都应跑完整测试集（主测试文件分片、各 unit test、torchrun 的 `test_table_dump_load.py`、dump/load 矩阵、C++ 套件），并重跑 benchmark 确认没有性能回退。
- 已知未解决的问题（与本评审无关，但拆分时不要遗漏）：PS 的 score 历史在回到前层表时丢失；`erase` 不会传到 PS；dump/load 不包含 PS；binding 里的 CUDA event 连线没有测试覆盖。
