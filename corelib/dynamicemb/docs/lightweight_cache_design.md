# 轻量化 Cache 设计

> 状态：**设计草案，尚未实现。**
> 适用范围：DynamicEmb **训练**路径下 `caching=True` 的 GPU 侧 embedding 缓存。
> 目标：替换现有的 `DynamicEmbCache`（类名不变，内部重写；代码独立到 `dynamicemb/cache.py`，不再放在 4000 多行的 `key_value_table.py` 里，所以导入路径从 `dynamicemb.key_value_table` 改为 `dynamicemb.cache`）。
> 关联代码：`corelib/dynamicemb/`。
> 参考：NVIDIA/recsys-examples PR #506 的设计文档。本文是独立设计，取舍见 §11。

---

## 0. 摘要

现有 `DynamicEmbCache` 与 `DynamicEmbStorage` 共用同一套 `ScoredHashTable`：每个 slot 带 score、引用计数，另有溢出区和 JIT 的 score 策略。但在 main 现在的流程里，**score 只存在于 storage**，cache 只按时间戳排序，所以这些能力对 cache 都是多余的。

本设计把 cache 收缩成三件事：

1. **索引**：key → slot（行池里的行号）。
2. **保护**：被 in-flight step 引用的 slot 不会被淘汰（引用计数 `ref`）。
3. **替换**：在 `ref == 0` 的 slot 里按 FIFO 选受害者。

关键决策：**slot 的分配与 key 的索引相互独立。**

- **分配**：每表一个"淘汰游标"（`evict_cursor`），从游标起取前 N 个 `ref == 0` 的 slot，即 FIFO。第 r 个新 key 先看游标后第 r 个位置，空闲就直接用；位置被 pin 的 key 再从后面的位置里按 warp 成批地用一个计数器 CAS 领取（§5.2）。不需要 ring，不需要全表扫描，不需要 host 循环。
- **索引**：桶链哈希，单向链，桶锁。

**保证**：只要整张表里还有 `ref == 0` 的 slot，分配就一定成功。失败当且仅当没有足够的未被引用的 slot。

元数据约 **20~24 B/slot**。

---

## 1. 背景

### 1.1 main 里 cache 的使用方式（读代码得到）

`_prefetch_cache_path`（`batched_dynamicemb_function.py`）每步：

1. `cache.lookup` 命中的 key 立即 `increment_counter`（pin）。
2. **所有** unique key（含命中）对 storage 做一次只读 `_find_keys`：score 的更新只发生在 storage。
3. miss 的 key 经 storage / 下层存储 / 准入 / 初始化，得到值。
4. `cache.insert_and_evict` 为 miss 分配 slot，返回被淘汰的 (key, slot, table_id)。
5. 先 `load_from_flat` 读出被淘汰 slot 的最新值，再 `store_to_flat` 把新值写入 slot。
6. 被淘汰的值经 `_write_back_evicted` 无条件写回 storage（storage 已淘汰的 key 进 `external_storage` 或被丢弃）。
7. `increment_counter` 钉住新分配的 slot。

backward 在更新行之后调用 `decrement_counter`。

由此得出：

- **cache 里的副本总是最新的**，被淘汰时无条件写回，不需要 dirty 位。
- **storage 的淘汰不看 cache**，不需要 backing 侧的保护接口。
- cache 不需要 score、overflow、扩容、dump/load。

### 1.2 现有实现的问题

- 索引是 `LinearBucketTable(enable_overflow=True)`：每 slot 带 score，另有溢出区。overflow 是 storage"不允许插入失败"的产物，cache 本不需要。
- cache 与 storage 共用 `DynamicEmbTableState`，继承了它全部的生命周期能力。

---

## 2. 目标与非目标

### 2.1 目标

| # | 目标 |
|---|---|
| G1 | 元数据 ≤ 24 B/slot |
| G2 | **只要整表还有 `ref == 0` 的 slot，分配必然成功**（没有 set 冲突导致的失败） |
| G3 | 分配的代价与 pin 密度之间有界：先看的位置是连续访存；被 pin 挡住的 key 以 32 个位置为一步往后找，最坏整表看一遍 `refs[]` |
| G4 | `ref > 0` 的 slot 永远不被淘汰 |
| G5 | 没有需要定期重建的累积状态 |
| G6 | 不改动 storage 侧的任何接口与 kernel |
| G7 | step 时间不低于现有实现 |

### 2.2 非目标

- 不保证确定性（`DEMB_DETERMINISM_MODE` 对 cache 不起作用，见 §9）。
- 不做 score / LRU / LFU / CLOCK：只做 FIFO。
- 不做容量扩展、dump/load：这些属于 storage。
- 不覆盖推理部署（`ref` 的生命周期绑定"前向 pin、反向释放"）。
- 不做跨 rank 共享，不做 backing 的分层改动，不引入 PS 第三层。

---

## 3. 数据结构

设表 `t` 的容量为 `C`（slot 数，沿用 main 的 `int(max_capacity × cap_scale)`，不再向桶容量取整），行宽 `D` 由行池决定（行池不属于本设计，见 §8.2）。

| 名称 | 类型 / 形状 | 含义 |
|---|---|---|
| `keys` | `int64 [C]` | slot 当前承载的 key；空位为 `EMPTY = -1`（全 1，与 ScoredHashTable 的 `EmptyKey` 一致）；哈希复用 `murmur3_fmix64`（`murmur_hash.cuh`） |
| `refs` | `int32 [C]` | 引用计数：有多少个 in-flight step 钉着这个 slot |
| `next` | `uint32 [C]` | 桶链的后继；`NIL = 0x7FFFFFFF` |
| `heads` | `uint32 [B]` | 桶头；`B = next_pow2(C)`；**bit 31 是桶锁** |
| `evict_cursor` | `int64` 标量 | FIFO 的淘汰游标，下一次分配时第 0 个位置所在的 slot |

约束：`C < 2^31 - 1`（slot 号要放进 31 位）。

- **行号 = slot 号**，由分配决定，与 hash 无关。
- 每张逻辑表持有独立的一套结构（行宽、容量各自独立）。
- 元数据：`8 (key) + 4 (ref) + 4 (next) + 4~8 (heads)` = **20~24 B/slot**。

---

## 4. 不变量

| # | 内容 |
|---|---|
| INV-1 | `keys[s] != EMPTY` 当且仅当 `s` 恰好出现在 `bucket(keys[s])` 的链上一次；`keys[s] == EMPTY` 的 slot 不在任何链上 |
| INV-2 | `ref > 0` 的 slot 永不被选为受害者，其 key 与行都不变 |
| INV-3 | **`ref` 从 0 变为非 0 只发生在 `lookup`（命中）和 `insert_and_evict`（新 slot）里**，两者都只被 prefetch 调用；prefetch 在 `TableAccessGate` 内、在同一条流链上串行执行。backward（`release`）只做减一。因此一个 slot 一旦被分配 kernel 观察到 `ref == 0`，在本次分配完成前不会被别人占走，**不需要对 `ref` 做 CAS**（分配里唯一的 CAS 是对每表的推进计数 `advanced[t]`，见 §5.2） |
| INV-4 | 每个 unique key 每步恰好 `+1`（`lookup` 命中，或 `insert_and_evict` 新分配，二者互斥）、恰好 `-1`（`release`）（前向已去重）。`release` 的 debug 构建断言 `ref > 0` |
| INV-5 | **命中先于分配**：一次 prefetch 里 `lookup` 总在 `insert_and_evict` 之前，命中的 slot 已经 pin 住，不会被本步分配当作受害者淘汰。这一条现在由接口本身保证，不再依赖调用顺序 |
| INV-6 | 读索引（lookup）不与修改索引（`insert_and_evict` 的索引更新、`erase`、`reset`）并发。由 `TableAccessGate` 与流顺序保证；eval 的只读 lookup 只在 `not training` 时发生，此时 `prefetch` 直接返回 |
| INV-7 | `insert_and_evict` 返回时，新 slot 的 `ref` 已经是 1：分配与 pin 之间没有空窗 |

INV-3、INV-6 是本设计免去对 `ref` 的 CAS 和 event 的依据，**实现前要在代码里逐项复核**（见 §10 待确认项）。

---

## 5. 操作

### 5.1 lookup

输入 `keys[U]`、`table_ids[U]`（key 必须 unique），一个 kernel 覆盖所有表。**命中的 slot 加一个引用（pin）**：

```
b = hash(key) & (B_t - 1)
n = heads[b] & ~LOCK_BIT
while n != NIL:
    if keys[n] == key: atomicAdd(&refs[n], 1); return found, slot = n
    n = next[n]
return not found, slot = -1
```

- 无锁：读链，不与修改并发（INV-6）。
- 训练的 prefetch 与 eval 共用这一个入口。eval 读完行之后调用 `release` 把 pin 还回去（`release` 跳过 `-1`，可以直接传整个 slots 数组）。
- hash 用 64 位 finalizer（如 murmur3 fmix64）。

### 5.2 insert_and_evict（分配 + 淘汰）

输入：本步需要新 slot 的 miss `keys[N]`，`table_ids[N]`，**按表号升序分组**（`flagged_compact` 保序）。整个分配没有 host 循环，也没有按表的 launch：所有表在同一组 kernel 里处理，host 只在最后同步一次，读出这次调用是否成功。

**K0：划分每表的 key 区间**（每个 key 一个线程，一次 launch）

每个线程只看自己的表号和前一个的表号：不同就是一张表的起点，同时填上两者之间空表的起点，最后一个线程补齐 `out[T] = n`。表号出现降序，或不在 `[0, T)` 里，写一个状态值，host 在最后那次同步之后报错。

**K1：选 slot（`alloc_slots_kernel`，每个 key 一个线程）**。每次调用的工作内存是一个 `AllocWork` 结构（`table_range`、`ids_status`、`advanced`、`fail`、`ok`，一块连续的清零显存），各 kernel 按名字读写。

位置是相对游标的偏移 `0 .. C-1`，每个位置只看一次：

```
key r（它在本表 N 个 key 里的序号）先看位置 r：
    refs[(cursor + r) % C] == 0 → 这就是它的 slot，不需要任何原子操作
    否则它"需要重试"
需要重试的 key 从位置 N 起领取，advanced[t] 记着已经看到了位置 N + advanced[t] 为止：
    一个 warp 里同属一张表、都需要重试的 key 成组（共 m 个），
    组长读 advanced[t]，整个 warp 一起读后面 32 个位置的 ref，ballot 出空闲的，
    空闲数 f >= m → 取前 m 个，把 advanced[t] 推到第 m 个空闲位置之后；
    f < m         → 取全部 f 个，把 advanced[t] 推过整个窗口（这 32 个位置里其余的都被 pin）；
    组长对 advanced[t] 做一次 CAS，成功才分配给组员，失败则整组重读再来
    N + advanced[t] >= C（整表都看过了）→ 这张表失败
```

- 选出的 slot 集合恰好是"从游标起的前 N 个空闲 slot"，新游标是 `(cursor + N + advanced[t]) % C`，即第 N 个空闲 slot 之后一位；**越过的只有亲眼看到被 pin 的 slot**，空闲 slot 不会被漏掉，FIFO 的语义不变。
- 哪个 key 拿到哪个 slot：没有 pin 挡路时第 r 个 key 拿第 r 个位置；有 pin 时，没找到自己位置的 key 之间的对应不指定（本来也不要求确定性）。
- CAS 在 `advanced[t]`（单调递增的相对计数）上，没有 ABA 问题。竞争只出现在需要重试的那部分 key 之间。
- 失败不修改任何状态：K1 只写 `slots` 和这次调用的工作缓冲（`advanced`、失败标志）。
- `ref == 0` 的 slot 包括空位（`EMPTY`）、以及已驻留但空闲的 key：后者**仍可被命中**，只是允许被淘汰。

**K1.5：收尾**（一个 block）

没有任何表失败、表号也没问题，才把每张有 key 的表的游标推进到 `(cursor + N + advanced) % C`，并把"成功"标志置 1；否则游标不动。K2 读这个标志，失败就什么也不做。

**K2：更新索引**（每个 key 一个线程）

```
s = slots[i]; old = keys[s]; new = miss_keys[i]
refs[s] = 1                      # 新 slot 已 pin；它刚才是空闲的，没有别人会改它
if old != EMPTY:
    lock(bucket(old)); 从头遍历链，找到 s，摘除; unlock
keys[s] = new
lock(bucket(new)); next[s] = head; head = s; unlock
```

- 任何时候只持一把锁，不会死锁。
- 桶锁是 `heads[b]` 的 bit 31，自旋获取。持锁区很短（链长约 1），冲突概率约 `N / B`。
- 摘除靠遍历：链短，不需要 `prev`（省 4 B/slot）。

**返回**：`slots[N]`（每个新 slot 的 `ref` 已是 1），被淘汰项 `(evicted_keys, evicted_slots, evicted_table_ids)`，`num_evicted`。K1 之后、K2 之前后者就已确定，K2 不改行数据。

被淘汰 slot 的**行数据**由调用方在 `store_to_flat` 覆盖之前读出（与现在的步骤 8 一致）。

### 5.3 release

```
release(slots, table_ids):  对每个元素（slot >= 0）atomicSub(&refs[slot], 1)
                            debug: 断言旧值 > 0
```

取代原来的 `increment_counter` / `decrement_counter`：加一已经并入 `lookup` 和 `insert_and_evict`。backward 对每步的 slots 调用它，`-1`（未被准入、没有 slot 的 key）被跳过；eval 的 `lookup` 读完行后也调用它。

### 5.4 erase（replay 后失效）

```
在桶锁内按 key 找到 slot（不加引用）
lock(bucket); 摘除; unlock
keys[slot] = EMPTY       # ref 须为 0；不为 0 时不改 ref，只给 err 计数（check_errors 可见）
```

不回写：这正是"后端刚写成新值，缓存副本过期"。留下的空位在游标下一次推进经过时被分配。

### 5.5 export / reset / size

- `export`（`flush_cache` 用）：每表对 `keys != EMPTY` 做流压缩，输出 `(keys, slots)`。
- `reset`：`keys = EMPTY`，`refs = 0`，`heads = NIL`，`evict_cursor = 0`。
- `size`：统计 `keys != EMPTY`。

---

## 6. 与现有流程的接合

**原则**：`_prefetch_cache_path` 等调用点保持原样，只删去用不到的参数和返回值。

保留类名 `DynamicEmbCache` 和下列方法名（模块换成 `dynamicemb/cache.py`，行池是 `mixed_dim_buffers.py` 里的 `MixedDimBuffers`；`flush_cache` 是 storage 侧的写回逻辑，仍在 `key_value_table.py`）；`Cache` 抽象基类（`types.py`）同步收窄：

| 方法 | 变化 |
|---|---|
| `lookup(keys, table_ids)` → `(founds, slots)` | 去掉 `lfu_accumulated_frequency` 参数与 `score_out` 返回值。**命中的 slot 加一个引用（pin）** |
| `increment_counter`、`decrement_counter` | **删除**：加一并入 `lookup` 和 `insert_and_evict`，减一由新增的 `release` 完成 |
| `release(slots, table_ids)` | **新增**：每个 slot 减一，`-1` 跳过 |
| `insert_and_evict(keys, table_ids)` → `(slots, num_evicted, evicted_keys, evicted_slots, evicted_table_ids)` | 去掉 `scores` 参数与 `evicted_scores` 返回值。**新 slot 已 pin（ref = 1）** |
| `erase(keys, table_ids)` | 新增，取代 `cache.key_index_map.erase` |
| `export(table_id, batch_size)` → 迭代 `(keys, slots)` | 新增，取代 `flush_cache` 里对 `key_index_map._batched_export_keys_scores` 的用法 |
| `reset()`、`set_record_cache_metrics` | 不变 |
| `num_tables`、`capacity(table_id=None)`、`size(table_id=None)` | **加入 `Cache` 抽象基类**：表数、槽位数、已缓存的 key 数（某一张表或全部） |
| `cache_metrics` | 返回 `CacheMetrics` dataclass 的快照（`num_lookups`、`num_hits`、`num_inserts`、`num_evictions`），没有记录时为 `None`；不再是 10 个元素的张量。调用方每次要读时重新取 |
| `training`、`set_score` | 删除（cache 没有 score） |

**调用点的改动清单**：

| 位置 | 改动 |
|---|---|
| `_prefetch_cache_path` 步骤 1 | `_, founds, cache_indices = cache.lookup(...)` → `founds, cache_indices = cache.lookup(...)`，并**删掉**其后的"命中项 `flagged_compact` + `increment_counter`"（一次压缩加一次同步） |
| `_prefetch_cache_path` 步骤 7 | `cache.insert_and_evict(cache_keys, cache_tids, None)` 去掉 `None`，解包少一个 `evicted_scores` 的 `_` |
| `_prefetch_cache_path` 步骤 9 | **删掉** `cache.increment_counter(cache_slots, cache_tids)`（新 slot 已在 `insert_and_evict` 里 pin 住） |
| `_eval_lookup_cached` | 解包同步骤 1；读完行之后加 `cache.release(cache_indices, table_ids)` |
| backward（`counter_owner.decrement_counter`） | cache 模式改调 `cache.release(...)`，storage 模式仍是 `storage.decrement_counter(...)` |
| `flush_cache` | 循环源换成 `cache.export(...)` |
| `dynamicemb_prefetch`、`_prefetch_cache_path`、`_initialize_rows` | 各加一个 `optimizer` 参数；`batched_dynamicemb_tables.prefetch` 里传入 `self._optimizer` |
| `_initialize_rows` | 第一个参数从 `state` 改为 `emb_dims`（设备上各表的 embedding 维度） |
| `_prefetch_cache_path` 步骤 7、8 | `load_from_flat(state, ...)` / `store_to_flat(state, ...)` → `cache.load_padded(...)` / `cache.store_padded(...)` |
| forward（取 `unique_embs`） | cache 分支 `cache.load_padded(...)`，storage 分支仍是 `load_from_flat(storage._state, ...)` |
| backward（融合优化器更新） | cache 分支的基地址、维度等取自 `cache.buffers_view`，storage 分支取自 `storage._state` |
| `_create_cache_storage` | 算好 `capacities` 和 `MixedDimBuffers(..., "fixed")`，`DynamicEmbCache(values, key_dtype=...)`；不再复制 `options` 去改 `bucket_capacity` 等 |
| `flush_cache`、`_eval_lookup_cached` | 用 `cache.load_padded(...)` |
| 显存报告（`_print_memory_consume` 里两处） | `_tier_bytes_from_state(cache._state)` → `cache.value_bytes()`（`DynamicEmbCache` 的方法） |
| `_invalidate_cache` | `self._cache.key_index_map.erase(...)` → `self._cache.erase(...)` |
| `batched_dynamicemb_tables.py` 里的 `cache.training = ...`、`cache.set_score(...)` | 删除（约 3 处） |

**行池**：`_prefetch_cache_path` 里原来的 `state = cache._state` 加 `load_from_flat` / `store_to_flat`，改成 `cache.load_padded` / `cache.store_padded`；`_initialize_rows` 取 `cache.buffers_view.emb_dims`。forward 里 cache 分支用 `cache.load_padded`；backward 里传给融合优化器的基地址、维度等取自 `cache.buffers_view`。storage 分支保持原样。细节见 §8.2。

**不变的部分**：storage 的 `_find_keys`、准入、`_initialize_rows`、`load_from_flat` / `store_to_flat`、`_write_back_evicted`，以及各步骤的先后顺序。

**好处与代价**：prefetch 里不再需要"命中项 `flagged_compact` + `increment_counter`"，也没有"分配之后、pin 之前"的空窗。代价是 eval 这类只读的调用者要在读完之后 `release`。

`outstanding_keys` 检查从"全局总数 ≤ 总容量"改为**每表**：`unique_size_per_table` 已在 host。全局检查对每表容量并不充分。

---

## 7. 容量与失败语义

- 每表容量仍由 `cap_scale` 决定，不再向 1024 取整，cache 选项里的 `bucket_capacity = 1024` 覆盖随之删除。
- 分配失败当且仅当该表没有足够的 `ref == 0` 的 slot，抛出明确的容量错误（需要增大 cache、减小预取深度、或减小 batch）。这与现有"cache 装不下一个 step 的 key"同类。
- 失败时 K1 不修改任何状态（先找位置，找够了再置 `ref`）。

**分配的最坏代价**：

- 常见情形（游标前方 pin 很少）：每个 key 读一个位置，连续访存，没有原子操作。
- 被 pin 挡住的 key：以 32 个位置为一步往后找，一长段连续被 pin 的 slot 约需 `M / 32` 步串行，每步一次对 `advanced[t]` 的 CAS。
- 许多 warp 同时重试（pin 很多时）会竞争同一个 `advanced[t]`：每次成功的 CAS 让其余的重读 32 个 ref。这是估算，没有实测，见 §10.5。
- 整表都被看过（失败）时最多 `C / 32` 步。

---

## 8. 内存与接入面

### 8.1 元数据对比

| | 每 slot | 备注 |
|---|---|---|
| 本设计 | 20~24 B | key 8 + ref 4 + next 4 + heads 4~8 |
| PR #506 | 约 32 B | 见其文档 §4 |

相对典型行宽（dim=128、fp32、Adam，1536 B）约占 1.5%，所以**内存不是本设计的主要卖点**，简单性与有界性才是。

### 8.2 行池：`MixedDimBuffers`

cache 的值（行）是一个 `MixedDimBuffers`（`dynamicemb/mixed_dim_buffers.py`），由调用方构造后交给 cache，cache 通过 `cache.values` 持有它，并在 `DynamicEmbCache` 上转发行读写：`load_padded` / `store_padded` / `load_exact` / `store_exact`，以及 `buffers_view`（即 `values.view`）。这些转发只在具体类上，不在 `Cache` 抽象基类里；调用点因此不必经过 `cache.values`。它是每表一个 `[行数, value_dim]` 的缓冲，加上行读写 kernel 需要的东西：各缓冲的基地址 `ptrs`、各表的 `emb_dims` / `value_dims`（设备和 host 各一份）、最宽的 `max_emb_dim` / `max_value_dim`、`all_dims_vec4`。它提供：

| 方法 | 作用 |
|---|---|
| `load_padded(rows, table_ids, copy_mode)` / `store_padded(rows, table_ids, values)` | 按 (行, 表) 读写；结果/输入补齐到最宽的表，每行只有本表宽度有效，尾部未定义 |
| `load_exact(rows, table)` / `store_exact(rows, table, values)` | 读写某一张表的整行，宽度精确等于这张表的 `value_dim`，不补齐 |
| `view` | 返回 `BuffersView`（`ptrs`、`dtype`、`value_dims`、`emb_dims`、`max_emb_dim`、`max_value_dim`、`all_dims_vec4`），给 optimizer 用 |
| `num_rows(table)` | 该表 buffer 当前的行数 |
| `grow_to(table, num_rows)` | 把该表 buffer 增长到至少 `num_rows` 行（够了则无操作，返回是否增长），并刷新 `ptrs`；仅可扩容的类型（`"device"`、`"host"`），`"fixed"` 需要增长时抛 `NotImplementedError` |

`kind` 决定缓冲放哪、能不能增长：`"device"`、`"host"` 是可扩容的，**`"fixed"` 是构造时定大小的设备池，cache 用它**。`"fixed"` 的缓冲是 `FixedDeviceBuffer`（`extendable_tensor.py`）：一个普通的 torch 张量，不是可扩容的 VMM 映射，`extend()` 直接报错。cache 的容量在构造时就定了，永远在 HBM、不扩容。

**构造函数**：`DynamicEmbCache(values, key_dtype=torch.int64)`，不再传 `options`、optimizer，也不再单独传容量。

| 参数 | 来源 |
|---|---|
| `values` | 行池。必须是 `kind="fixed"`。每表的行数就是它的槽位数，索引（`CacheCore`）按这些行数建，所以容量不用另外传 |
| `key_dtype` | 调用方 key 的类型（int64 或 uint64）。索引内部存 int64 位模式，被淘汰的 key 和 `export` 的 key 按这个类型还给调用方。它无法从 `values` 推出，所以单独给，默认 int64 |

之前 cache 从 `options` / optimizer 里读的东西现在都由调用方算好并放进 `values`：容量（行数）、各表 `dim`、行宽（`dim + optimizer.get_state_dim(dim)`）、dtype。这些在 `batched_dynamicemb_tables._create_cache_storage` 里算（它本来就有 `options` 和 `self._optimizer`）。

**不再有 `CacheState`**：`load_from_flat` / `store_to_flat` 这一套对 cache 换成 `values.load_padded` / `values.store_padded`；storage 路径不变，仍用 `DynamicEmbTableState`。`optimizer` 不放在 cache 里：`_initialize_rows` 需要时作为参数传入（`dynamicemb_prefetch` → `_prefetch_cache_path` → `_initialize_rows`），它只取 `values.emb_dims`。显存报告调用 `DynamicEmbCache.value_bytes()`，它返回每表的值缓冲字节数（cache 不扩容，峰值就是当前大小）。它是具体类的方法，不在 `Cache` 抽象基类里。`_print_memory_consume` 目前只统计值缓冲；cache 元数据的字节数由 `DynamicEmbCache.meta_bytes()` 提供，尚未接入报告。

---

## 9. 确定性

不保证。原因有两个：

1. K1 看到的 `refs[]` 会被另一条流上的 backward 并发地减小，看到哪些空闲 slot 取决于时序；同一张表里没找到自己位置的 key 拿到哪个 slot 也取决于 warp 到达计数器的顺序。
2. 桶链的顺序取决于线程到达桶锁的顺序（不影响任何 key 的值）。

`DEMB_DETERMINISM_MODE` 对 cache 无效；storage 的确定性插入不受影响，因为 storage 的淘汰与 cache 无关。

---

## 10. 待确认与风险

1. **`release` → 分配的跨流可见性。** backward 先更新行、再 `atomicSub(ref)`（同一条流）；prefetch 流上的 K1 观察到 `ref == 0` 后读受害者的行。两条流之间没有显式事件，这是 main 现在的计数方案同样依赖的契约。计划：`release` kernel 在 `atomicSub` 之前加 `__threadfence()`，K1 读 `refs[]` 用 acquire 语义，成本很小，契约变明确。**实现时要再核对 main 是否有额外的事件保证。**
2. ~~**行池对象的构造**~~ 已定：`MixedDimBuffers`，由调用方构造后交给 cache（§8.2）。
3. **同步次数。** `insert_and_evict` 在 C++ 里只在最后同步一次（读边界状态、各表失败标志）；`flagged_compact` 取被淘汰个数还有一次，在 Python 封装里。分配本身没有按表的 launch，每步 launch 数是常数。
4. **重试路径的竞争（已实测，确实是问题）。** 需要重试的 key 很多时，各 warp 竞争同一个 `advanced[t]` 的 CAS。`insert_and_evict` 单测（H100，估计值）：没有 pin 时 0.17–0.46 ms；一半 slot 随机被 pin 时 15–76 ms；90% 随机被 pin 时 100–505 ms；游标前方一整段连续被 pin 时第一次调用 72–837 ms（每 32 个位置约 2.7 µs，串行），之后的调用恢复正常。端到端 benchmark 里 cache 远没被 pin 满，这个代价不出现。可选的缓解：让每个 warp 先领一段更大的范围，或改成分块计数再前缀和；接口不变。
5. **性能。** 端到端结果见 §12.1。
6. **桶锁在热点下的行为。** 同一桶里大量 key 时自旋会变慢。`B = next_pow2(C)` 保证平均链长 ≤ 1，热点来自 hash 质量。
7. **分块汇总**（每 1024 个 slot 记一个空闲数，跳过全被 pin 的块）已决定先不做，最坏代价见 §7，留作后续优化。

---

## 11. 与 PR #506 的取舍

| 项 | PR #506 | 本设计 | 理由 |
|---|---|---|---|
| 范围 | 缓存 + backing 的急切流程 + 守卫 + PS 第三层 | 只换 cache | main 里 storage 已独占 score，backing 淘汰不看 cache |
| 分配 | 全局 ring：释放时入队，出队时 CAS 复核 | 按游标顺序取空闲 slot（位置直接取，被 pin 的用 CAS 计数器领取） | 免去 `in_ring` 位、出队复核、两个 event；只有 prefetch 把 ref 从 0 抬起（INV-3） |
| 索引 | 桶链，双向，桶锁 | 桶链，单向，桶锁 | 链长约 1，遍历摘除代替 `prev` |
| 状态字 | `ref + 状态位 + in_ring + used` 打包 | 独立的 `ref` | 没有 `LOADING`：值的写入与读取都在 prefetch 流的顺序里 |
| 替换 | FIFO 或 CLOCK | 仅 FIFO | 决策 |
| 失败语义 | 容量错误 | 容量错误 | 同 |
| 接口 | `lookup_pin / allocate / publish / release` | `lookup`（命中即 pin）、`insert_and_evict`（新 slot 已 pin）、`release`，另有 `erase / export` | 去掉 `increment_counter / decrement_counter`；没有 publish：值由 Python 写 |
| 元数据 | 约 32 B | 约 20~24 B | |

**放弃的，以及放弃的代价**：

- ring 能让分配在任何 pin 密度下都是 O(N)；本设计在 pin 接近 100% 时是 O(C)，换来没有并发协议。
- FIFO 在游标前方有被重新 pin 的老 slot 时会有轻微的乱序，不影响正确性。

---

## 12. 验证与实施

### 12.1 微基准（先做，决定是否继续）

独立的 CUDA 程序，只测 `lookup`、`insert_and_evict`、`release` 这几个 kernel（M0 当时的原型还是 `increment_counter` / `decrement_counter` 的形式，语义等价）：

- **对象**：本设计、main 现有 cache（必要时加 PR 的核心）。
- **参数**：unique 数 40 万；容量取 `0.5×` 与 `4×` 的 unique 数；命中率取 0%、50%、90%；**加一个高 pin 比例（90%、99%）的场景**，测 §7 的分配代价（pin 的分布也要分两种：随机散布、和游标前方一整段连续）。
- **判据**：不比现有实现慢；高 pin 场景的分配代价在每步毫秒量级之下。不达标就回到 PR 的 ring 结构。

**结果（单张 H100，TorchRec 关闭，每批 100 万 key，dim 128，训练前向，ms，新 / 旧）**：

| 表数 / 优化器 / cache 比例 | 前向 | eval |
|---|---|---|
| 1 / sgd / 0.8 | 8.6 / 12.7 | 2.0 / 2.1 |
| 1 / sgd / 1.0 | 1.1 / 6.6 | 0.8 / 0.97 |
| 10 / adam / 0.8 | 15.9 / 21.0 | 2.3 / 2.4 |
| 10 / adam / 1.0 | 2.4–2.6 / 9.7 | 0.7 / 1.05 |
| 10 / sgd / 0.8 | 8.9 / 15.7 | 2.0 / 2.3 |
| 10 / sgd / 1.0 | 2.3 / 7.4 | 0.7 / 1.05 |
| 100 / sgd / 0.8 | 9.0 / 17.6 | 2.0 / 2.5 |
| 100 / sgd / 1.0 | 2.2 / 8.1 | 0.6 / 1.07 |

反向两边持平（0.6–1.0 ms）。cache 比例 0.8 时新 cache 的命中率更低（如 10 张表 0.492 对 0.556，FIFO 对 LRU），仍然更快；命中率到第 49 次迭代还在上升，不是稳态；旧 cache 取整容量，两边行数不完全相同；benchmark 只有一个 dim，变长未对比。高 pin 场景的结果见 §10.4：没有达到"每步毫秒量级之下"的判据，需要优化。

### 12.2 测试

- **单元**：不变量（INV-1~5）、FIFO 顺序（含被 pin 跳过）、空位优先、失败时无副作用、erase、export、reset。
- **并发**：两条流，一条 prefetch、一条 `release`，压力下断言 INV-2 与"静止时 ref 全为 0"。
- **集成**：现有 `test_batched_dynamic_embedding_tables_v2.py` 的 caching 用例（含 `test_prefetch_flush_in_cache`）、async erase 在 cache 布局下的用例。
- **变异检验**：去掉 §10.1 的 fence，看测试是否能发现问题。

### 12.3 步骤

| 阶段 | 内容 |
|---|---|
| M0 | 微基准与判定（§12.1） |
| M1 | C++/CUDA 核心 + torch 绑定 + 单元测试 |
| M2 | Python：新 `DynamicEmbCache`，改 `Cache` 抽象与调用点 |
| M3 | 删除旧 cache 不再使用的代码与测试里对 `key_index_map` 的依赖 |
| M4 | 文档（`DynamicEmb_APIs.md`）与 benchmark |

---

## 附录：一个完整的例子

`C = 8`，`B = 4`，`home(k) = k mod 4`。预取深度为 2：下一步的 prefetch 发生在上一步 backward 之前。

**Step 1**，unique = {5, 9, 6}，全部 miss。从 `evict_cursor=0` 取 slot 0、1、2，`insert_and_evict` 把它们的 `ref` 置 1，`evict_cursor→3`。索引：桶 1 为 `1(9) → 0(5)`，桶 2 为 `2(6)`。

```
key  5  9  6  ∅  ∅  ∅  ∅  ∅
ref  1  1  1  0  0  0  0  0      evict_cursor = 3
```

**Step 2**，unique = {5, 7, 12, 3, 10}。5 命中并 pin，`ref[0]=2`。其余 4 个 miss，取 slot 3、4、5、6，`evict_cursor→7`。

```
key  5  9  6  7 12  3 10  ∅
ref  2  1  1  1  1  1  1  0      evict_cursor = 7
```

**Step 1 的 backward**：对 slot 0、1、2 做 `release`。slot 1、2 空闲但仍可命中，slot 0 仍被 Step 2 钉着。

```
ref  1  0  0  1  1  1  1  0
```

**Step 3**，unique = {7, 20, 26}。先 pin：7 命中，`ref[3]=2`。miss 为 20、26，N=2，游标 P=7，位置 r 对应 slot `(7+r)%8`：

- 第 0 个 key（20）先看位置 0，即 slot 7：`ref==0`，直接取（空位）。
- 第 1 个 key（26）先看位置 1，即 slot 0：`ref==1`，被 pin，成为"需要重试"的 key。
- 它从位置 N=2 起领取（`advanced[t]=0`）：位置 2 即 slot 1，`ref==0`，取，受害者 key 9；`advanced[t]→1`。

游标 `→(7+N+advanced)%8=(7+2+1)%8=2`。20→slot 7，26→slot 1，slot 0 因被 pin 而越过。调用方先读出 slot 1 里 key 9 的值写回 storage，再覆盖。索引更新：

- 摘 9：桶 1 加锁，`heads[1] = next[1] = 0`。
- 插 20：`next[7] = heads[0] = 4`，`heads[0] = 7`。
- 插 26：`next[1] = heads[2] = 6`，`heads[2] = 1`（slot 1 从桶 1 搬到桶 2）。

```
key  5 26  6  7 12  3 10 20
ref  1  1  0  2  1  1  1  1      evict_cursor = 2

b0: 7→4    b1: 0    b2: 1→6→2    b3: 5→3
```

此时 8 个 slot 里 7 个被钉住，分配仍然成功。再来一步需要 2 个新 slot 就会失败，因为只剩 slot 2 空闲：这才是失败条件。
