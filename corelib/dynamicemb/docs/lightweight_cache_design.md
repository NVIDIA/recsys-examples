# 轻量化 Embedding Cache 设计提案

> 状态：已实现并接入 `DynamicEmbCache`（见 §17）；其余章节是设计依据，与实现不一致处以 §17 和各节顶部的说明为准
> 适用范围：DynamicEmb **训练**路径的 GPU 侧 embedding 缓存（不覆盖推理部署，见 §2.2）
> 关联代码：`corelib/dynamicemb/`
> 实现：`src/embedding_cache/`。与本文的差异，以实现为准，已回写到各节：
> 去掉 `EVICTING` 状态（§4.3）；Step 4 合并为一个 kernel，靠桶锁（§6.1）；去掉 pin 后的身份复查（§6.3）；
> ring 的槽位数取 `2C`，tail 快照在 kernel 启动前取，并作为 push/pop 两个阶段的屏障（§6.4、§7）。
> 行号不存进 node；可在同一次 prefetch 调用内携带，但不得跨 `expand_if_need`（§4.5）。backing 是 score、准入、key 淘汰的唯一权威（§6.5）。
> 多表：块集体的 kernel 按表分别 launch，只有给 backing 用的几个融合（§4.4）。
> 附录 C 的逐步走查仍按"两趟 + `EVICTING`"、并把 backing 探测画在摘链趟里的早期表述画出，数据流不受影响。

---

## 0. 摘要

本文提出一套**重新设计**的 GPU embedding 缓存，而非在现有 `DynamicEmbCache` 上改造。

核心思路是：缓存只做缓存该做的事——把 key 映射到一个 GPU 常驻的定长行池，保证在用的行不被回收，在行被复用时把脏数据换出去。所有与"存储"相关的能力（容量扩展、dump/load、跨层迁移）都不进入缓存，留给 backing 表。

结构上由四组张量构成：**行池（EmbeddingTable）+ 节点元数据（Node）+ 哈希索引 + 可回收节点环（Ring）**。其中 Ring 用 `arange` 初始化，使"首次分配"与"回收复用"合并为同一条代码路径；节点的引用计数与状态打包进单个 32 位字，使"命中时钉住"与"淘汰时占用"通过一次 CAS 天然互斥——这是全部并发安全性的基础。

元数据开销约 **32 B/node**，相对典型行宽（dim=128、fp32、Adam，1536 B）约占 **2%**。

---

## 1. 背景与动机

### 1.1 现有实现

现有 `caching=True` 路径下，`DynamicEmbCache` 与 `DynamicEmbStorage` **共用同一个** `create_table_state()` 构造的 `DynamicEmbTableState`（`dynamicemb/key_value_table.py:1423`）。这带来几个结构性后果：

- 缓存继承了 storage 的全部生命周期能力：容量扩展、`dump` / `load`、`collect_table_sizes`。缓存并不需要这些。
- 索引结构是 `LinearBucketTable(enable_overflow=True)`：每 slot 携带 key、digest、score，另有覆盖 `capacity + 3 × bucket_capacity × num_tables` 的 int32 引用计数，以及一整块 overflow buffer（`dynamicemb/scored_hashtable.py:410-448`）。overflow 是 storage 语义（不允许插入失败）的产物，缓存本可以直接淘汰。
- value 行宽取所有表的 `max(emb_dim + optim_state_dim)`（`dynamicemb/key_value_table.py:277-279`），混合维度场景下存在行内碎片。
- `flush_cache` 需要先把 backing storage 预扩容到 `storage_size + cache_size`（`dynamicemb/key_value_table.py:2562-2640`），产生内存尖峰。

### 1.2 为什么重新设计

上述问题的根因是**缓存被按存储的规格构造**。逐条改造需要把 `DynamicEmbTableState` 拆开，而它同时服务于 storage 路径，改动面会溢出到 hybrid / 单层等所有模式。

重新设计的边界更清晰：新缓存不复用 `DynamicEmbTableState`，只依赖 backing 暴露的两个能力（key→row 查询、host 行的读写）。现有路径保持不动，通过配置项切换，可以并行验证。

---

## 2. 设计目标与非目标

### 2.1 目标

| # | 目标 | 验收方式 |
|---|---|---|
| G1 | 元数据开销 ≤ 40 B/node | 静态计算 + 运行期实测 |
| G2 | 命中路径（lookup + pin）无指针写、无 host 同步 | 代码审查 + nsys 时间线 |
| G3 | 淘汰不影响任何正在被引用的 key | 不变量断言 + 压力测试 |
| G4 | 替换策略可配置且对训练结果透明 | FIFO / CLOCK 切换后 loss 逐位一致 |
| G5 | 无容量扩展路径、无内存尖峰 | checkpoint 期间峰值显存监控 |
| G6 | E2E step time 不低于现有实现 | HSTU / MovieLens benchmark |

### 2.2 非目标

- **不覆盖推理部署场景。** 本设计的若干前提在纯前向路径下不成立：`ref` 的生命周期绑定在"前向 pin、反向释放"上；行布局包含 optimizer state；"每个被缓存的 key 必然被反向更新"这一前提（INV-9 与无条件回写的依据）不成立；prefetch 依赖 key 在计算前已知。推理侧的 embedding 缓存需要单独设计。文中提到的 eval（§9.4）指训练过程中的周期性验证，不是独立部署的推理服务。
- 不改动 backing 表（host storage）的内部实现
- **不做 backing 索引的分层或下沉。** 索引恒常驻 GPU（§3、§10.3），key 总数因此受索引显存预算约束
- 不做 KV 量化 / 压缩（如需，另立提案）
- 不做跨 rank 的缓存共享或一致性
- 不做运行时的跨表容量再平衡（见 §10.1）

---

## 3. 总体架构

```
┌─────────────────────────── GPU ────────────────────────────┐
│                                                             │
│   ┌── 缓存（本文范围）──────────────────────────────┐        │
│   │  emb_table  [C_t, D_t]   行池                   │        │
│   │  nodes      key/state/prev/next                 │        │
│   │  hash_heads [B_t]        key → node             │        │
│   │  ring       [C_t]        可回收 node 队列       │        │
│   └─────────────────────────────────────────────────┘        │
│                                                             │
│   ┌── backing 索引 ─────────────────────────────────┐        │
│   │  key → host row id                              │        │
│   └─────────────────────────────────────────────────┘        │
└──────────────────────────────┬──────────────────────────────┘
                               │ PCIe（zero-copy）
┌──────────────────────────────┴──────────────────────────────┐
│                          Host                               │
│   backing 行数据：embedding + optimizer state               │
│   mapped pinned 内存，kernel 直接寻址                       │
└─────────────────────────────────────────────────────────────┘
```

两条关键的分层决策：

**backing 索引在 GPU，backing 行数据在 host。** miss 路径因此完全没有 CPU 参与——GPU 侧查索引拿到 row id，再由 warp 通过 zero-copy 直接把该行从 host 拉进 `emb_table`。这与现有实现的分层一致（`create_table_state` 中 `key_index_map` 无条件建在 `device`，值缓冲走 `HostExtendableBuffer` / `HostVMMTensor`，见 `dynamicemb/extendable_tensor.py:131-149`）。

**每张逻辑表持有独立的一套结构，但所有表在同一个 kernel 内处理。** 见 §4.3。

---

## 4. 数据结构

### 4.1 每表结构

设表 `t` 的缓存容量为 `C_t`（node 数），行宽 `D_t = emb_dim_t + optim_state_dim_t`。

| 名称 | 类型 / 形状 | 含义 |
|---|---|---|
| `emb_table_t` | `[C_t, D_t]` | 行池。行号即 node id；embedding 与 optimizer state **同行连续** |
| `key_t` | `int64 [C_t]` | 该 node 当前承载的 key |
| `state_t` | `uint32 [C_t]` | 打包状态字，见 §4.2 |
| `prev_t` / `next_t` | `int32 [C_t]` | **哈希桶链**的前驱 / 后继 |
| `hash_heads_t` | `int32 [B_t]`，`B_t = next_pow2(2 C_t)` | 桶头 node id，`-1` 为空 |
| `ring_t` | `int32 [C_t]` | 可回收 node 队列 |
| `ring_head_t` / `ring_tail_t` | `uint64` 标量 | 单调递增计数器 |

节点元数据采用 **SoA**：`key` 与 `state` 在不同 kernel 中被独立访问，分离存储才能保证合并访存。

**embedding 与 optimizer state 同行**是刻意的：反向的优化器更新是访存密集型，两者分离会让访存次数翻倍且都退化为非合并访问；同时淘汰回写必须携带整行，同行布局让回写成为一次连续搬运。两者的 dtype 一致，因此 `emb_table_t` 是一个同构的 typed tensor。

**node 中不存 backing 行号。** 淘汰时的行号由 backing 按 key 给出，不随 node 长期保存。这换来两件事：省 4 B/node，以及**对 backing 重哈希免疫**——行号一旦长期保存，backing 扩容重哈希后它就是野指针，回写会写到别的 key 的行上，造成静默的数值损坏。行号只在一次 prefetch 调用内使用，不跨 `expand_if_need`。详见 §4.5。

### 4.2 状态字位布局

```
 31 30 | 29   | 28      | 27   | 26 .......... 0
 状态  | 保留 | in_ring | used | ref_counter
```

| 字段 | 宽度 | 含义 |
|---|---|---|
| 状态 | 2 bit | `EMPTY` / `LOADING` / `RESIDENT`（第四个取值空闲） |
| `in_ring` | 1 bit | 当前在 ring 中存在一条记录 |
| `used` | 1 bit | 自上次被表针扫过以来是否被命中（CLOCK 用，见 §8） |
| `ref_counter` | 27 bit | 当前有多少个 in-flight step 引用该 node（上限 ~1.3 亿） |

**没有 `dirty` 位。** 训练路径下每个被缓存的 key 必然走完反向并被更新，因此 `dirty` 恒为 1，留着只是徒增一类"忘了置位"的缺陷——而这类缺陷的表现是 optimizer state 静默丢失、模型照常训练但收敛变差，极难定位。改为**淘汰时无条件回写**（§9.1），依据是 INV-9。

**打包成单字是必需的，不是优化。** 命中时的"钉住"（ref+1）与淘汰时的"占用"（判定 ref==0 并改状态）必须互斥。若 ref 与状态位分处两个字，就存在"刚 ref+1 完、同时被判定为可淘汰"的窗口。打包后两者 CAS 同一个字，互斥由硬件保证。

顺带的收益：CLOCK 的 `used` 位可以直接合并进 pin 的那次 CAS，**不产生任何额外访存**。

⚠️ 由此推出一个硬化要求：`atomicSub(&state_t[n], 1)` 是对**整个字**减一，仅当 `ref > 0` 时才等价于"ref 减一"。若 INV-6 被破坏、出现多减，`ref` 会从 0 借位到高位，**`used`、`in_ring`、状态位同时被腐蚀**。debug 构建的释放路径应断言 `REF(old) > 0`。

### 4.3 状态机

```
EMPTY ──占用──> LOADING ──加载完成──> RESIDENT ──被占用──> LOADING (复用)
                                          │
                                          └── ref 在 RESIDENT 内部 0↔N 变化
```

- `EMPTY`：初始状态，行内无有效数据
- `LOADING`：已归属某 key，H2D 或初始化未完成，**既不可消费也不可淘汰**。占用一个 node 的那次 CAS 直接把它置为 `LOADING` 并持有一个引用
- `RESIDENT`：数据有效。`ref > 0` 表示在用；`ref == 0` 表示空闲，但**仍然可以命中**

**没有单独的 `EVICTING`。** 它只会存在于"占用"与"改派"之间，而这两步在同一个 kernel 的同一个线程里完成，没有任何其他线程能观察到；且 pin 与 claim 对它的处理与对 `LOADING` 完全一致（都拒绝）。多一个状态只是多一个没人读的取值。

### 4.4 多表分派

每张表是一套完整、互不相干的结构：行池、node、桶链、ring，行宽各自独立。分派分两类，取决于 kernel 是否做 ring 的块集体操作。

**按表分别 launch：** lookup+pin、allocate、搬运（写回、加载、发布）、release。输入 `unique_keys[U]` 按表分段，`table_range[T+1]` 由 `segmented_unique_cuda` 直接给出，每张表取自己那一段，用的就是单表的 kernel，一行也不改。每表的 unique 数 host 本来就有（`unique_size_per_table` 是 CPU 张量），所以不需要同步。

为什么这几个不融合成一次 launch：allocate 和 release 里的 ring 操作是**块集体**的（一个 block 共用一个计数器的聚合 ticket，内含 `__syncthreads`）。miss 压紧之后顺序被打乱，同一个 block 里会混着不同的表，聚合就不成立。要融合就得做"block 到表"的映射，让每个 block 只属于一张表，这是一块不小的新机制，现在没有证据表明启动开销是瓶颈：每步的 launch 数是 `T × kernel 数`，在预取与计算重叠的前提下是 host 侧的开销，而不是 GPU 的。留到 profile 说需要时再做。

**一次 launch 覆盖所有表：** 守卫（`EvictionGuard`）、`find_nodes`、`invalidate`、`fetch_rows`。原因是 backing 的淘汰 kernel 本身是跨表的，交回来的被淘汰 key 列表混着各个表、没有顺序，没有"每表一段"可切。这几个都不是块集体操作，按元素分派就很便宜：device 侧传入每表的 `HashMap` 数组与行池数组，每个元素按自己的 table id 取对应那一张。取值的输出用每个元素一个偏移（各表行宽不同，调用方累加），不是固定步长。

**每表独立结构的收益与代价：**

- ✅ `D_t` 逐表独立，不存在行内碎片
- ✅ 桶链天然按表隔离，node 无需存储 table id，比对时少一次访存
- ✅ `ring_head` / `ring_tail` 从单一计数器拆成 `T` 个，`atomicAdd` 竞争摊到 T 路
- ⚠️ 容量无法跨表共享，需要容量规划环节（见 §10.1）

### 4.5 backing 扩容与重哈希

DynamicEmb 的 backing 表按每步各表的 unique 数由 host 驱动扩容（`expand_if_need`），扩容会 rehash，key 的槽位与行号随之改变。**重哈希是常态，不是异常路径**，本设计不得假设它不发生，也不得假设它只发生在某个特定时刻。

由此得到两条规则：

1. **node 里不存行号；行号只在一次 prefetch 调用内携带，不得跨 `expand_if_need`。** 行号来自本步 backing 的 find/insert，以及搬运前对受害者 key 做的一次只读 lookup（Const 策略，不更新 score）。扩容只发生在一次 prefetch 调用的开头，所以调用内的行号不会过期；任何要跨调用保存行号的做法都是错的。
2. **读 backing 的 kernel 不得与扩容并发。** 这是所有 backing 使用者共有的约束，缓存不例外：claim（find-or-insert）、搬运、flush 都是这样的 kernel。规则 1 消除的是"跨 kernel 的过期"，消除不了"kernel 运行时 backing 被搬动"。

规则 2 在现有后端下已经成立，不需要额外的 event，依据是三点：

- 扩容由 `dynamicemb_prefetch` 在开头调用 `storage.expand_if_need(...)` 同步触发，位于 prefetch 流上，先于该 step 的缓存路径（`batched_dynamicemb_function.py`）。
- 扩容值缓冲前，`ExtendableBuffer.extend()` 先执行 `torch.cuda.synchronize()`，整设备排空（`extendable_tensor.py`）。
- 扩容在 host 线程里同步完成，期间不会向其他流发射新的 kernel。

所以只要缓存里读 backing 的 kernel 与 `expand_if_need` 发射在同一条 prefetch 流、同一次 prefetch 调用里，二者天然有序。反向的 release 不读 backing，不受影响。

**扩容会整体替换 backing 的视图**：它新建一个 `key_index_map`（新的设备张量），把所有 key 重新插入，再把值行从旧槽搬到新槽，并刷新 `table_ptrs_dev`（`_expand_tables_impl`）。因此缓存 kernel 用的 backing 视图（索引表指针、各表值缓冲指针）必须在每次 launch 前从当前状态里取，不能跨 step 保存。

扩容也可能让 backing 淘汰一些 key。搬运 kernel 探测不到某个 key 时按"backing 没有这个 key 的记录"处理，这个契约在 backing 接口定下来时一并明确（早期原型的做法见 §16），不在缓存里另设分支。

---

## 5. 不变量

以下不变量应在 debug 构建中以断言形式落地。它们是并发正确性的完整依据。

| # | 内容 | 保障手段 |
|---|---|---|
| INV-1 | `ref > 0` 的 node 永不被淘汰 | 出队时 CAS 校验 |
| INV-2 | node 仅在 `ref` 发生 `1→0` 跃迁且 `in_ring == 0` 时入队 | `atomicOr(IN_RING)` 的返回值判定 |
| INV-3 | 从 ring 弹出后必须重新校验，`CAS(RESIDENT\|ref=0 → LOADING)` 成功才算占用 | 与并发 pin 互斥 |
| INV-4 | `tail - head ≤ C_t` | 由 INV-2 推出（每 node 至多一条记录），入队端无需检查 |
| INV-5 | `head ≤ tail` | 出队越界时必须 `atomicSub` 归还 |
| INV-6 | 每个 unique key 每步恰好 `+1 / -1` | 前向必须先去重 |
| INV-7 | 替换策略对训练结果完全透明 | 见下 |
| INV-8 | pin 成功后必须复查 `key_t[n] == key` | 防 node 被改派（§6.3） |
| INV-9 | **`RESIDENT` 的行永不旧于 backing 中对应的行** | 见下 |

**关于 INV-6。** 这是最容易在后续迭代中被破坏的一条。一旦某条路径（被 mask 的 token、提前退出的分支、异常路径）漏掉 `-1`，该 node 的 `ref` 将永久大于 0，从此无法被淘汰，表现为缓存有效容量的缓慢泄漏——症状是命中率随训练时长单调下降。更严重的是反向情形：多减一次会因借位腐蚀整个状态字（§4.2）。

校验有两处：debug 构建的释放路径断言 `REF(old) > 0`；以及 checkpoint 排空流水之后，此刻所有 `ref` 必须为 0（§9.5），这是一次精确的全量校验，不需要额外的巡检 kernel。

**关于 INV-7。** 缓存是纯性能设施：被淘汰的 key 会带着完整的行回写 backing，重新加载后值不变。因此 FIFO 与 CLOCK 的切换只影响 PCIe 流量与 step time，不影响 loss 与收敛。

这条不变量决定了该配置项的验收方式——**只测性能，不需要数值对齐测试**。同时它也是一条回归防线：若切换替换策略导致 loss 变化，说明缓存存在真实缺陷（最可能是回写漏了 optimizer state，或 ref 计数泄漏导致行被丢弃）。

**关于 INV-9。** 缓存中的行只有两个来源：从 backing 加载（与 backing 相等），或在缓存内被优化器更新（比 backing 新）。因此回写永远是"用新覆盖旧"。

这条不变量支撑了两个设计决定：

- **淘汰时无条件回写**（§9.1）——不需要 `dirty` 位来判断"这行比 backing 新吗"，答案恒为是
- **flush 时回写全部 `RESIDENT` 行**（§9.5）——不得因 `ref > 0` 或任何其他条件跳过。一个长期命中、从未被淘汰的热 key，其 backing 副本可能已落后数千步；跳过它意味着丢失这期间的全部更新，而 `ref > 0` 恰好意味着它正在被使用，也就是最热、最不该丢的那批

---

## 6. 运行时流程

### 6.1 前向

| 步 | 操作 | 可复用 |
|---|---|---|
| 1 | 去重：`keys[N]` → `unique_keys[U]`、`inverse[N]`、`table_range[T+1]` | `src/unique_op.cu` |
| 2 | 哈希查找 + pin | 新写 |
| 3 | 对 backing 里已有、或准入的 miss 从 ring 弹出 node（backing 的 find 与准入判定见 §6.5 的第 3、4 步） | 新写 |
| 4 | 受害者摘链、新 key 插入桶头，一个 kernel | 新写 |
| 5 | 双向搬运：受害者行 → host，新行 host → `emb_table`；全新 key 走初始化 | `src/initializer.cu` |
| 5b | backing 插入本步准入的新 key；缓存同时淘汰被 backing 淘汰的 key（§6.5 的第 7、8 步） | 新写 |
| 6 | `out[i] = emb_table[node_ids[inverse[i]], :emb_dim]` | `src/lookup_forward.cu` |

**Step 4 是一个 kernel。** 线程甲把受害者从桶 b 摘出、线程乙往桶 b 头部插入，是会发生的并发修改，这是无锁双向链表的一般性难题。解法是桶级锁：桶头自身兼作锁位，摘链和插链各自在持锁期间完成整个修改，而一次修改碰到的所有 node（受害者的前驱后继、原桶头）都在被锁的桶里，因为链不跨桶。所以不必把两者拆成两个阶段，省掉一次 launch。需要分阶段的是**查找**：查找无锁，读桶头与链都是普通读，因此不能与任何修改并发，这条才是与 ring 的读写相分离（§7.2）、pin 的阶段分离（§6.3）同一套纪律。

**未被准入的 key 走旁路。** 准入判断由 backing 表负责（`dynamicemb/embedding_admission.py`），被拒绝的 key 拿不到 backing 行号，因而不可缓存：它们**不占 node、不进 ring、不计 ref**，前向通过临时 buffer + 初始化产出结果。Step 3 需要把这类 key 从 miss 集合中分离出去。

**Step 2 的查找与 pin：**

```cuda
h = hash(key) & (B_t - 1);
n = hash_heads_t[h];
while (n != -1) {
    if (key_t[n] == key) {
        if (try_pin(n)) {                 // CAS: 状态==RESIDENT 且 ref+1
            node_ids[u] = n; found[u] = 1; return;
        }
        break;                            // 撞上 LOADING，降级为 miss
    }
    n = next_t[n];
}
node_ids[u] = -1; found[u] = 0;
```

pin 失败时**降级为 miss，不自旋**。代价仅是多一次 H2D，且发生概率极低；而在 GPU 上自旋等待一个可能夹着 PCIe 传输的操作是死路。

**命中的定义包含"躺在 ring 里的 node"。** `ref` 归零不等于数据失效——只要还没被弹出复用，命中它就是完整的缓存命中。这是 ring 相对"归零即释放"的核心收益。

**Step 1–5 整体提前若干 step 执行（prefetch）**，与之前 step 的 dense 计算重叠，用于掩盖 PCIe 延迟。流水深度见 §6.4。

### 6.2 反向

| 步 | 操作 | 可复用 |
|---|---|---|
| 1 | 按 `inverse` 规约梯度 → `[U, emb_dim]` | `src/lookup_backward.cu` |
| 2 | 原地优化器更新：读整行 → 计算 → 写回原行 | `src/optimizer_kernel.cuh` |
| 3 | `ref -1`；若发生 `1→0` 且 `in_ring` CAS 成功则入队尾 | 新写 |

Step 3：

```cuda
uint32_t old = atomicSub(&state_t[n], 1);
assert(REF(old) > 0);                                   // debug：防借位腐蚀
if (REF(old) == 1) {                                    // 1 → 0 跃迁
    if (atomicOr(&state_t[n], IN_RING) & IN_RING) return;  // 已在环中
    uint64_t pos = atomicAdd(&ring_tail_t, 1ULL);
    ring_t[pos & MASK] = n;
}
```

"由观察到 `1→0` 跃迁的线程负责入队"这一归属规则，配合 `in_ring` 位，共同保证入队恰好一次。

### 6.3 pin 的身份竞态

读到桶链中的 node `n` 之后、CAS pin 之前，`n` 可能已被淘汰并改派给**另一个** key。此时它的状态又是 `RESIDENT`，pin 的 CAS 会**成功**——线程以为拿到了 key K，实际持有的是 K′ 的行。

唯一的防线是**阶段分离**：Step 2（查找 + pin）与 Step 3–4（分配 + 淘汰）处于**不同的 kernel**，由流序保证不并发。状态字防不住这件事——它可以回到逐位相同的取值，pin 后也无从分辨节点是否被改派过——所以不设"pin 后复查 key"的兜底：阶段分离成立时它永远不触发，不成立时（例如融合 lookup 与 allocate、或多条 prefetch stream）要改的是这条分离，而不是在这里补一道检查。

pin 失败也不是"多一次 H2D"那么便宜：key 仍绑定在原 node 上，把它当 miss 会让 allocate 再给它绑一个新 node，同一个 key 对应两个 node。阶段分离成立时 pin 不会失败；这条同样依赖它。

### 6.4 prefetch 流水深度

支持深度 `d > 1`。ring 的 MPMC 隐患只有一种形态：

- **出队的 tail 快照包含了"已预留、未写入"的位置** —— 危险。入队已 `atomicAdd(tail)` 拿到位置 `p` 但尚未写入 `ring[p]`，出队的快照却已包含 `p`，读到的是该槽的上一个占用者
- **入队写在快照之上** —— 安全。入队写的位置 `p ≥ 出队的 tail 快照`，出队只消费 `< 快照`，二者不相交

关键是快照必须是**整个 kernel 的同一个值**，并在 kernel 启动之前取好。若放在 kernel 里读，各 block 启动时刻不同，后启动的 block 可能读到已被一个迟到的 release 推进过的 `tail`，等于从"安全"的方向走进了"危险"。所以快照由 kernel 之前的一次流序拷贝取得（`ReclaimQueueStorage::take_snapshot`），kernel 只读它。

这次拷贝就是 push 阶段与 pop 阶段之间的**屏障**，两侧都要排序，"已发起"按的是 host 的发射顺序，不是设备上的执行顺序：

> **快照之前**：allocate 所在的流先等待所有已发起的 release 完成（一个 stream event wait）。
> **快照之后**：之后发起的 release 必须等待"快照已取"这个 event（在快照拷贝后立即 record）。

两侧缺一不可：缺前者，快照可能覆盖正在写入的位置；缺后者，一个较晚发起、却先被设备执行的 release，同样会让快照读到"已预留、未写入"的 `tail`。排好之后，release 可自由地与运行中的 allocate kernel 重叠。

这条约束不限制流水深度，且在稳态下**不产生气泡**：深度 `d` 时，`allocate(N+d)` 大约发生在 step N 的前向期间，而 `release(N-1)` 在 step N-1 反向末尾就已完成；"快照已取"的 event 在 allocate 一开始就满足，反向的 release 几乎不会等它。

配套要求：**所有 step 的 prefetch 工作走同一条 prefetch stream**，使各 step 的 lookup / allocate kernel 之间天然串行，§6.3 的阶段分离在 `d > 1` 下依然成立。若使用多条 prefetch stream，该保证失效，需要额外设计。

深度的代价体现在容量下界（§10.1）：深度越大，本步释放的 node 越晚可用，`C_t` 需相应放大。`ref_counter` 的 27 位对任何现实深度都绰绰有余。

### 6.5 与 backing 的分工

**缓存里没有 score，也不做准入和 key 的淘汰。** 这些全部由 backing 的逻辑决定；缓存只做三件事：GPU 上的行副本与放置、ref 保护、在 ref 为 0 的 node 里按 FIFO 选换出对象。所以每一步**所有 unique key，不论缓存是否命中，都先过一遍 backing**：更新 score，缺失的过准入，准入的新 key 插入 backing。

**保护规则（backing 淘汰前先查缓存）。** backing 要淘汰某个 key 时，先在缓存里找它：**不在缓存里，或在缓存里且 ref 为 0，才允许淘汰；ref 不为 0 则跳过这个候选。** 这一条同时覆盖了 in-flight 的 step 和本步已经拿到 node 的 key。实现上是 backing 的 `insert_and_evict` 在选 victim 时多带一个判定函数（查缓存桶链、读状态字 refs）。

每步顺序（顺序本身就是正确性的一部分，见下）：

1. 去重，`expand_if_need`。
2. 缓存 lookup + pin：命中的 key 先 pin，ref 加一。
3. backing 对**所有** key 做 find，更新 score，得到每个 key 是否存在及其行号。
4. 对 backing 里缺失的 key 做准入判定（只判定，不插入）。
5. 缓存对所有需要 node 的 miss（backing 里已有的，加上准入的新 key）做 allocate：FIFO 选受害者，rebind 到新 key，此时这些 node 的 ref 已是 1。
6. 搬运：受害者行写回 backing，backing 已有的 key 从 backing 加载，新 key 初始化。
7. **backing 插入本步准入的新 key**，判定函数保护 ref 不为 0 的缓存 key；得到被淘汰的 key 列表。插入失败（一个桶里所有可淘汰的 key 都被守卫保护）时抛出容量错误，不降级为未准入（见 §17.5）。
8. **缓存处理被淘汰的 key，与 backing 同时淘汰**：key 若在缓存里，必然是 ref 为 0 的空闲 node。先取下它最新的 embedding 与 optimizer state，交给保留逻辑（见下），再从桶链摘除并把 node 置为 EMPTY，**不写回 backing**（它的行正被新 key 占用）。
9. 反向：缓存 refs 减一。

**为什么是这个顺序。**

- 2 先于 7：命中的 key 要先有 ref，才受保护。
- 5、6 先于 7：本步在 backing 里已有、但不在缓存的 key，必须先拿到 node（ref 为 1）才受保护；受害者也必须先写回 backing，它的行才不会被 7 的插入重新分配。
- 4 与 7 分开：准入只判定，插入放在 node 都就绪之后；这样新 key 的 node 在插入前就已存在，判定函数能保护它们。
- 8 中"查找"与"摘除并改状态"要分成两个 kernel：桶链的 lookup 不能与任何修改并发（HashMap 的调用约定，见 `hashmap.h`）。7 里 backing 的判定函数会读缓存桶链，这没有问题，因为此时缓存不做修改。

**取下来的值交给保留逻辑。** 第 8 步取下的 embedding 与 optimizer state，交给 `evicted_item_mode` 的保留逻辑（`_append_evicted` 一路），与 backing 自己淘汰、没有缓存副本的 key 走同一条路。缓存只负责把值取下来并交过去，要不要保留由模式决定，所以只有模式要求保留值时才需要取。目前 `EvictedItemMode` 只有 `RETAIN_KEY`，保留值需要在其中新增一个标志；这是保留逻辑那一侧的改动。

**取最新值只发生在第 8 步，对象只能是空闲 node。** 空闲 node 没有人在写，取到的值是确定的最新值；这也是要求 ref 不为 0 的 key 不被淘汰的原因之一，否则要读的行可能正被反向修改。

---

## 7. Ring 协议

### 7.1 表示与初始化

```
ring_t = arange(C_t);   head = 0;   tail = C_t
所有 node：state = EMPTY | in_ring=1 | ref=0
```

`arange` 初始化让**"首次分配"与"回收复用"成为同一条代码路径**。传统实现需要分别维护 free-list（从未使用的槽）与 eviction-queue（用过待回收的槽）；这里两者合一——环中装的一律是"可被占用的 node"，只是有的尚未写过数据（`EMPTY`），有的装着有效但无人引用的数据（`RESIDENT`，命中它即缓存命中）。分配方不关心两者区别，只关心 `ref == 0`。

元素个数为 `tail - head`，空即 `head == tail`。使用无符号单调计数器配合掩码索引，比维护独立的 size 变量更好——后者在并发下需要额外原子操作，而 `tail - head` 是免费的。

**槽位数组的长度不是 `C`，而是 `2C` 向上取到 2 的幂。** 位置 `p` 落在槽 `p & mask`，当 pusher 写到 `p + 槽数` 时才会覆盖它。`in_ring` 位把有效条目数限制在 `C` 以内，但这不是 push 需要避开的量，要避开的是"最早未读位置到 `tail`"的跨度。出队线程是先领 ticket、后读槽的；二者之间别的线程会丢弃或领走它之后的条目，在跨度中间留下空洞，而 push 只写 `tail`，不填空洞。于是跨度可以比有效条目数更宽，只有 `C` 个槽时，push 会绕回到一个已被领走 ticket、却还没读的位置：那个出队线程读到别人的 node，而原本在该槽的 node 仍带着 `in_ring`，再也不会入队——永久丢一个槽位。

跨度有硬上界。出队 kernel 开始时跨度至多为 `C`（此时所有未消费位置都是有效条目）；其间启动的 push 写在快照之上，且每个 node 至多 push 一次（push 之后它带着 `in_ring`，新条目在快照之上，没人会把它清掉），至多再加 `C`。所以跨度不超过 `2C`，`2C` 个槽不会在被读之前被覆盖。取成 2 的幂，索引就是一次掩码，省掉每次 push 与每轮 pop 里的 64 位除法；代价是每槽 4 字节，相对一行 embedding 可以忽略。

计数器使用 **uint64**。uint32 在 `C ≤ 2^31` 时靠无符号减法也能正确处理回绕，但那需要额外论证；uint64 在任何真实训练时长内都不会绕，省下的推理成本远大于 8 字节。

### 7.2 核心约束：读写在时间上相分离

> **同一时刻，ring 只被"纯出队 kernel"或"纯入队 kernel"中的一种访问，由流序保证。**

这条约束是整套机制得以保持简单的根本原因。若出队与入队真正并发，就会撞上经典的 MPMC 环形队列问题：入队线程先 `atomicAdd(tail, 1)` 占位 `p`，但尚未把 node id 写入 `ring[p]`；此时出队线程看到 `tail` 已增长，认为 `p` 有效，读到的却是上一轮的陈旧值。修复需要为每个槽引入序列号（Vyukov 方案），复杂度与调试成本陡增。

训练循环天然满足该约束：

```
step N:   [prefetch / allocate ← 只出队]  forward  backward  [release ← 只入队]
step N+1: [prefetch / allocate ← 只出队]  ...
```

启用深度 `d > 1` 的 prefetch 流水时，该约束由围绕 tail 快照的两条 stream event 依赖维持，详见 §6.4。代价仅是"本步释放的 node 要过若干步才可用"——这是容量问题而非正确性问题，已被 §10 的容量下界覆盖。

该约束带来一个直接好处：**出队 kernel 的边界是一个固定的快照，入队 kernel 中 `head` 是常量**。边界检查只读一个启动前就定下的值，全程精确，无需反复读取被高频原子写的变量。注意快照必须在 kernel 启动前取，不能在 kernel 里各线程各读一次，理由见 §6.4。

### 7.3 出队

```cuda
__device__ int ring_pop(uint64_t tail_snapshot, int t) {
    for (;;) {
        uint64_t pos = atomicAdd(&ring_head[t], 1ULL);
        if (pos >= tail_snapshot) {
            atomicSub(&ring_head[t], 1ULL);   // 归还，维持 INV-5
            return -1;                        // 环已抽干 → 上层报错 / 反压
        }
        int cand = ring_t[pos & MASK];
        uint32_t st = state_t[cand];

        if (REF(st) != 0) {                   // 陈旧条目：该 node 已被重新命中
            clear_in_ring(cand);
            continue;                         // 丢弃，重领
        }
        if (try_claim(cand)) {                // INV-3
            clear_in_ring(cand);
            return cand;
        }
        clear_in_ring(cand);                  // 刚被 pin 走，同样丢弃
    }
}
```

三个易错点：

**陈旧条目是丢弃，而非重新入队。** 直觉上会想"它还有效，放回队尾"，但这会在缓存被 pin 满时造成活锁——所有条目 `ref > 0`，线程将绕环无限循环。丢弃则使环单调收缩，最坏情况是抽干并返回 `-1`，上层立即得到明确错误。丢弃不会泄漏：该 node 会在下次真正归零时重新入队。

**越界时必须把 head 减回去。** 否则多个线程同时撞到空环会把 head 推过 tail，`tail - head` 下溢为巨大的无符号数，后续所有判断全部失效。此缺陷仅在缓存打满时触发，测试极难覆盖，因此需要 INV-5 的断言兜底。

**CAS 失败与陈旧条目走同一处理。** 两者都意味着"该 node 现在不归我"，无需区分。

### 7.4 入队

见 §6.2 Step 3。

**入队无需溢出检查**，由 INV-4 保证：`in_ring` 位使任一 node 在环中至多一条记录，因此环中记录数不超过 `C_t`；而出队只会减少记录数（丢弃陈旧条目亦然）。

### 7.5 `in_ring` 位的必要性

这是整套设计中最容易写错的地方，单独说明：

> node X 的 `ref` 归零 → 入 ring。此后 X 被再次命中，`ref 0→1`（**X 仍留在 ring 中，成为陈旧条目**）。反向释放，`ref 1→0`。若此时无条件再次入队，ring 中就有了 X 的**两条**记录。第一条被弹出、成功淘汰并复用为新 key 后，第二条即成为悬垂引用——会把一个正在服役的新 key 当作垃圾回收。

`in_ring` 位保证任一 node 在环中至多一条记录：入队走 `CAS(in_ring: 0→1)`，出队清零。陈旧条目在 §7.3 被识别并丢弃，该 node 将在下次真正归零时重新入队。

等价方案是为每个 node 维护 generation 计数、ring 中存 `(node_id, gen)` 二元组，出队时比对。两者择一即可，`in_ring` 方案节省一半 ring 内存。

### 7.6 一个 C=4 的完整 trace

初始 `ring=[n0,n1,n2,n3]`，`head=0`，`tail=4`，全部 `EMPTY`。

| 步 | 动作 | 结果 |
|---|---|---|
| 1 | key A miss | pop `pos0`→`n0`。A→n0，ref=1。`head=1` |
| | 反向释放 | ref→0，in_ring 0→1，push `ring[4&3]=n0`，`tail=5` |
| 2 | key B miss | pop `pos1`→`n1`。`head=2`。释放后 push，`tail=6` |
| 3 | key C miss | pop `pos2`→`n2`。`head=3`。释放后 push，`tail=7` |
| 4 | key D miss | pop `pos3`→`n3`。`head=4`。释放后 push，`tail=8` |
| | | 环内容 = 位置 `[4,8)` = `[n0,n1,n2,n3]` = A,B,C,D |
| 5 | key A **hit** | ref 0→1，`used(n0)=1`。释放时 in_ring 已置位 → **不重复入队** |
| 6 | key E miss | pop `head=4` → `n0`（A）。FIFO 下 ref==0 → 淘汰 A |

第 6 步正是 FIFO 的病理点：A 在第 5 步刚被访问，却因"进来得最早"被淘汰。§8 讨论如何用 CLOCK 规避。

---

## 8. 替换策略

### 8.1 FIFO 与 CLOCK 的差别

- **FIFO** 的淘汰依据是：*进来多久了*
- **CLOCK** 的淘汰依据是：*进来之后有没有被再用过*

CLOCK 只有两条规则：

1. **插入时** `used = 0`；**命中时** `used = 1`
2. **淘汰时**检查队头：`used == 1` 则清零并放回队尾（给予第二次机会），继续看下一个；`used == 0` 才淘汰

推论：**一个 node 要被淘汰，必须在表针走完一整圈期间一次都没被命中过。** 热 key 每圈都会重新置位，因此永远轮不到它。

对照关系：**`ring_head` 即表针，`ring` 数组即钟面**。

### 8.2 对照 trace

访问序列 `A B C D A E A F`，容量 4，A 为热 key：

| 步 | 访问 | FIFO | CLOCK |
|---|---|---|---|
| 1–4 | A B C D | 依次装入，满 `[A B C D]` | 依次装入，`used` 全为 0 |
| 5 | **A** | **hit**，队列顺序不变 | **hit**，`used(A)=1` |
| 6 | E | miss → 淘汰队头 **A** ❌ | miss → 队头 A `used=1`，清零并放回队尾；下一个 B `used=0` → 淘汰 B ✓ |
| 7 | **A** | **miss**（第 6 步刚被扔掉）→ 淘汰 B | **hit** ✓ |
| 8 | F | miss → 淘汰 C | miss → 队头 C `used=0` → 淘汰 C ✓ |

8 步的例子里差距只有一次 miss，但规模效应显著：**FIFO 的表针每转一圈就会把整个缓存换一遍，热 key 也不例外**。因此 FIFO 的命中率大致等于「缓存容量 ÷ 一圈内出现的 unique key 数」，几乎与 key 的热度分布无关——等于把分布倾斜带来的全部红利浪费掉。推荐场景的 key 分布是重尾的，倾斜越强，浪费越大。

### 8.3 实现

配置项独立于现有的 `EvictStrategy`（后者是 HKV 的 score 语义，本设计不使用）：

```python
class ReplacementPolicy(Enum):
    FIFO  = 0    # 默认
    CLOCK = 1
```

**采用运行期开关，而非模板参数。** 理由：出队循环不是热路径（每个 miss 走一次，而非每个 key 走一次），一个全 kernel 一致、完美预测的分支开销可忽略；模板化的唯一实质收益是省下 `retry_buf` 显存，而按需分配已解决该问题。运行期开关少一份编译产物、少一条独立测试路径。

作用域为全局而非逐表。逐表混用在技术上几乎免费（kernel 本就在读逐表参数），但缺乏实际需求，且会让命中率回归问题的排查空间乘以 `2^T`。

CLOCK 开启后，出队循环增加一个分支：

```cuda
if (st & USED) {
    atomicAnd(&state_t[cand], ~USED);
    retry_buf[atomicAdd(&retry_cnt, 1)] = cand;   // 攒起来，稍后统一入队尾
    continue;
}
```

配合一个跟随的小 kernel，把 `retry_buf` 顺序追加到 `ring_tail`。**不能在出队 kernel 内直接写 `tail`**，那会破坏 §7.2 的读写相分离约束。

### 8.4 终止性：必须限次回退

上述实现有一个终止性缺陷，必须一并处理：

> 进入 `retry_buf` 的条目在本次出队 kernel 内**不可见**。若 ring 中的活跃条目恰好全部 `used == 1`，它们会被逐个搬进 `retry_buf`，`head` 一路推到 `tail_snapshot`——`ring_pop` 返回 `-1`，报"缓存打满"。但实际上有大量可淘汰 node，只是都躺在 `retry_buf` 里等待下一个 kernel。

这是**误报**，且仅在缓存打满且访问模式突然变均匀时触发，测试极难覆盖。

经典 CLOCK 没有这个问题——它的表针在**固定**环上循环，转一圈清零后第二圈必然找到受害者（上界 2 圈）；而本设计的 ring 是队列，`head` 推进即意味着条目被消费。

修复采用**限次回退**：

```
每个线程最多授予 k 次第二次机会（k = 1 或 2）；
超过后无视 used 位，直接 claim 下一个 ref == 0 的条目。
```

这保证循环有界、不会误报，且降级优雅：**本步退化为 FIFO，下一步恢复 CLOCK 语义**（第一次扫描已把那些 `used` 位清零）。真正触发降级的场景是"缓存中所有内容都是热的"，此时任何替换策略表现都一样差，损失可忽略。

已评估但不推荐的两个替代：让出队 kernel 直接写 `ring_tail`（破坏读写相分离）；把 `retry_buf` 做成可被同一 kernel 消费的第二级环（同样的竞态，只是换了位置）。为一个病理场景引入并发队列的复杂度并不划算。

### 8.5 `used` 位无条件置位：一个免费的在线指标

即便运行在 FIFO 模式，`used` 位也照常置位——它合并进 pin 的 CAS，零额外访存。

> **在出队时统计 `used == 1` 的条目占比，即等于"若启用 CLOCK，有多少次淘汰会被推迟"。**

这个比例可以直接指示是否值得启用 CLOCK，**无需离线模拟、无需 trace dump**，跑一次正常训练即可得到：

- 比例接近 0 → 访问几乎没有复用，CLOCK 无意义
- 比例 40%+ → 大量热 key 正被 FIFO 误杀，应启用

需要说明它是**筛选指标而非精确预测**：推迟一次淘汰会改变后续的受害者选择，二阶效应无法由此推算。但作为"值不值得试"的判据完全够用。

建议将该指标与命中率、ring 抽干次数一并做成固定的缓存健康度指标组，异步拷出、每数百步读取一次，**热路径上不得有任何 host 同步**。

---

## 9. 写回

### 9.1 触发时机：仅淘汰与 checkpoint

不做后台主动刷。理由比"全双工顺风车"更根本：

> 训练中最热的行，特点是**每步都被更新、但永远不会被淘汰**。它们在缓存中反复累积更新，这正是缓存存在的全部意义——把 N 次更新吸收为最终淘汰时的 1 次回写。后台主动刷会把这些行反复写回 host，而每一次写都会被下一步的更新作废。这是纯粹的带宽浪费，且越热的行浪费越多。

同一逻辑排除了"`ref` 归零时立即回写"的方案。若绑定在 `ref → 0`，则每步 D2H 行数从 `M`（每步 miss / 淘汰数）涨到 `U`（每步 unique key 数），而 `M = U × (1 - 命中率)`——命中率 90% 时流量是 10 倍，**缓存越有效，惩罚越重**。

**淘汰时无条件回写，不设 `dirty` 判断。** 依据是 INV-9：`RESIDENT` 的行永不旧于 backing。训练路径下每个被缓存的 key 必然走完反向并被更新，`dirty` 恒为 1，判断没有意义；去掉它同时消除了"某条路径忘了置位 → optimizer state 静默丢失"这一整类缺陷。

唯一保留的判断是 `EMPTY` 与 `RESIDENT` 的区分——从未装过数据的 node 被弹出时不能把垃圾写进 host。该信息本就在状态位中。

### 9.2 与加载合并为双向 kernel

稳态下每次 miss 恰好触发一次淘汰：

```
每步 PCIe 流量：H2D = M 行（miss 加载），D2H = M 行（受害者回写）
```

方向相反、量级相等——这正是全双工的理想输入。合并后每个 warp 处理一对：

```
warp 处理 (受害者 node n, 受害者行 r_old, 新 key 的行 r_new)：
  1. emb_table[n] → 寄存器                （device 读）
  2. host[r_new]  → 寄存器                （PCIe H2D）┐ 同时在飞
  3. 寄存器(旧行) → host[r_old]           （PCIe D2H）┘
  4. 新行寄存器   → emb_table[n]          （device 写）
```

由于回写是无条件的，kernel 内**没有 `if (dirty)` 分支，warp 不产生分化**——每个 warp 都执行完整的双向搬运。`r_new` 来自 backing 的 find/insert，`r_old` 来自搬运前对受害者 key 的一次只读 lookup，二者都在同一次 prefetch 调用内取得（§4.5）；全新 key 没有历史值，走初始化。

寄存器开销不成问题：`dim=128` fp32 加 Adam 两个 state 为 1536 B/行 = 96 个 `float4`，摊到 32 lane 上是每 lane 3 个 `float4`，两行共 24 个寄存器。可按 chunk 流水以进一步重叠。

由于旧行在第 1 步就已进入寄存器，第 4 步随时可以覆写——**D2H 不阻塞 H2D**，不存在"等待写回"的问题。

**实测结果：链路确实全双工，但我们的访问形态只能拿到其中一部分。** 随机行下双向 1.19×（§12.1），而真实密度下排序几乎无用（§12.2），所以 1.19× 就是可期待的数字——约省 20% 的传输时间。值得做，无需前置工作。

窄行（≤ 512 B）应关闭融合：实测那里双向反而更慢（dim=32 时 0.80×），行太窄时 kernel 受限于延迟与占用率而非带宽，每个 warp 兼顾两个方向反而减少在飞的行数。

### 9.3 准入即在 backing 分配行

> 新 key 在准入时，立即在 backing 中 **insert**，保证它在 backing 中有行。此时**不写任何值**，仅占位。缓存不参与准入，也不参与 key 的淘汰（§6.5）。

代价接近于零——miss 路径**本来就要探测 backing 哈希**（以判断该 key 是已有 key 还是全新 key），find-or-insert 只是同一次探测的自然延伸。

收益是三重的：

| | 惰性分配 | 准入即分配 |
|---|---|---|
| 淘汰 | 可能触发 backing 插入与扩容 | **纯值写**，行号可探测得到 |
| checkpoint flush | 批量插入 + 扩容尖峰 | **纯值 dump**，无分配 |
| 回写 kernel | 需处理插入失败 / 扩容分支 | 无分支，纯数据搬运 |

这直接消除了 §1.1 中现有实现的内存尖峰问题（G5）。淘汰路径上不再有任何"可能失败、可能触发分配"的操作，使双向 kernel 得以保持为纯粹的数据搬运——无分支、无错误处理、无 host 参与。

配套语义：backing 的行一旦分配即**终身绑定该 key**，缓存淘汰不释放它。因此 backing 表的容量按总 key 数规划，而非活跃 key 数。

注意行号**不缓存在 node 中**（§4.1、§4.5）：每次由 backing 给出，只在一次 prefetch 调用内使用。这使设计对 backing 重哈希免疫。

### 9.4 训练内 eval 路径

指训练过程中周期性的验证前向（`module.eval()`），不是独立部署的推理服务——后者已由 §2.2 排除。

**与现有 `eval_lookup` 语义对齐**（`dynamicemb/key_value_table.py:2777`）：

```
eval：查找 → 命中则直接读；miss 则从 backing zero-copy 直读
      不插入、不准入、不淘汰、不占用 node
      不 pin、不置 `used`
      只拷贝 embedding 列，不触碰 optimizer state
```

三点说明：

- **不能绕过缓存直读 backing。** 缓存中的行普遍比 backing 新（INV-9），绕过会读到陈旧值。必须先查缓存。
- **不置 `used`。** eval 是全量遍历，若置位会把所有 `used` 刷成 1，CLOCK 的表针下一轮将对每个条目都给出第二次机会，直接退化为 FIFO 且出队循环变长——扫描污染换了个形式又回来了。
- **不 pin** 的前提是 eval 期间没有并发的 allocate。eval 本就是独立阶段，该前提成立，但需在实现中确保。

由于不准入，全量遍历不会冲刷热数据；热 key 依然享受命中。相比"先全量 flush 再 bypass"，省掉一整次全量 dump。

### 9.5 checkpoint flush

```
flush(checkpoint)：
  1. 排空流水，等所有 in-flight step 完成反向
  2. assert 所有 node 的 ref == 0
  3. 回写全部 RESIDENT 的行（EMPTY 跳过）
```

**Step 1 的排空是必需的。** flush 需要把行读出送往 host，这次读不能与两类 kernel 并发：反向的优化器更新（会读到半新半旧的撕裂行）、以及 allocate 的双向搬运（会读到正被新 key 覆写的行）。排空后两者都不在运行。也可用 event 依赖精确插入而不排空，但 checkpoint 间隔通常数千步，排空的代价是 `d` 个 step 的流水气泡（毫秒级），不值得为此增加复杂度。

**Step 2 是白拿的校验。** 排空之后所有 `ref` 必须为 0，这是 INV-6 的一次精确全量校验。ref 泄漏平时只表现为"命中率随训练时长缓慢下降"，极难定位；把断言放在这里等于给 INV-6 装了周期性哨兵，无需额外的巡检 kernel。

**Step 3 不得有任何跳过条件**（INV-9）。特别是不能跳过 `ref > 0` 的行——一个长期命中、从未被淘汰的热 key，其 backing 副本可能已落后数千步。

成本：最坏情况整个缓存都要写，即 `C_t × D_t` 的 D2H。4M node × 1536 B ≈ 6 GB，按 20 GB/s 约 300 ms。

**实现。** `resident_slots_kernel` 为每个 RESIDENT 节点生成一个槽位，节点自己的 key 当作受害者，格式与淘汰产生的槽位一致；调用方像对真正的受害者一样去 backing 查这些 key 的行号，然后直接交给 `writeback_kernel`，写回的拷贝不重复实现。同一个 kernel 数出仍持有引用的节点数（`referenced`），排空之后它必须为零，这就是上面的 Step 2。节点不论有没有引用都会被写，测试里专门核对了带着引用 flush 时被持有的 key 在 backing 里是最新的。

### 9.6 崩溃一致性

仅在淘汰时回写意味着崩溃会丢失缓存中尚未回写的全部更新。但训练本就依赖周期性 checkpoint 恢复，"丢失上次 checkpoint 以来的进度"是既有语义，缓存并未使其变差。因此不需要 WAL 或额外持久化机制。

### 9.7 回写粒度

整行（embedding + optimizer state）。

**只回写 embedding 会导致 optimizer state 在淘汰时静默丢失**——模型照常训练，只是收敛变差。这类缺陷极难定位，实现时应有针对性测试。

---

## 10. 容量与内存

### 10.1 容量约束

```
C_t ≥ (表 t 单步 unique key 数) × (prefetch 流水深度) × 安全系数
```

否则 `ring_pop` 返回 `-1`，即缓存里每个 node 都被 in-flight 的 step 占着，训练无法继续。（FIFO 下每个分配线程只消耗一个环条目。CLOCK 每个线程最坏消耗 `1 + k` 个，容量要乘上这个因子，见 §16。）

**这个约束在哪里检查。** 公式里"每表单步 unique 数"是随数据变的，配置期并不知道，所以真正起作用的保护是**运行时的每表检查**：每次 prefetch，对每张表累计 in-flight 的 key 数，超过 `C_t` 就报明确的错误，而不是等到 `ring_pop` 抽干。现有 `dynamicemb_prefetch` 里已经有一个全局的同类检查（`outstanding_keys` 与缓存容量比较），新缓存把它改成每表一份即可。每表的 unique 数 host 本来就有（`unique_size_per_table` 是 CPU 张量），不需要同步。这是 Python 编排层的算术，缓存的 C++ 部分不需要一个对应的校验组件。

由于容量无法跨表共享（§4.4），需要容量分配环节：

- **初版**：按各表预估 key 数或历史访问频次静态按比例分配
- **P1**：观测每表命中率与 ring 抽干次数，做离线再平衡
- **不做**：运行时动态再平衡（挪动 node）复杂度过高

现有 `dynamicemb/planner/` 已在做类似的分片规划，可挂靠。

**行宽应作为容量规划的输入之一。** §12.2 实测：随机行访问下 1536 B 的行拿到 39.6 GiB/s，512 B 的行只有 14.0，128 B 的行只有 3.85——每字节的 miss 代价随行变窄急剧恶化（PCIe 事务粒度以下的行，每次传输大部分是浪费）。

所以**窄表需要比按 key 数等比分配更多的容量**：它更依赖高命中率来回避 miss，而不是靠 miss 走得快。§12.2 的密度复测进一步表明，**这个差距在真实密度下无法靠排序弥补**：随机访问带宽与批大小基本无关，1536 B 行恒在 ~39 GiB/s、512 B 行恒在 ~13.9 GiB/s。窄表只能靠更高的命中率回避 miss。

### 10.2 元数据开销

| 项 | 字节/node |
|---|---|
| `key` | 8 |
| `state` | 4 |
| `prev` + `next` | 8 |
| `hash_heads`（`B_t = 2 C_t`） | 8 |
| `ring` | 4 |
| **合计** | **32** |

对比典型行宽（dim=128、fp32、Adam）1536 B，占比约 **2.1%**，满足 G1。

这一数字同时说明：后续优化的重点不在元数据，而在行宽本身。

### 10.3 backing 索引的显存占用

**backing 索引恒常驻 GPU，这是设计前提，不设下沉 host 的退化路径。** 由此得到一个必须在容量规划阶段处理的约束：**总 key 数 `N` 的上界由索引的显存预算决定。**

每 key 的索引开销：

```
每槽 = key 8 B + digest 1 B          （槽位即行号，行号本身不占空间）
槽数 = N / 负载因子
每 key ≈ 9 B / 负载因子 ≈ 18 B（负载因子 0.5），计对齐后取 20 B
```

注意 backing 索引**不需要 score 字段**——backing 不做淘汰，key 一旦插入即终身驻留。这比仓库现有的 `LinearBucketTable`（key + digest + score）省下 8 B/槽。

| 索引显存预算 | 可容纳的总 key 数 |
|---|---|
| 2 GB | 1.0 × 10⁸ |
| 4 GB | 2.0 × 10⁸ |
| 8 GB | 4.0 × 10⁸ |
| 16 GB | 8.0 × 10⁸ |
| 32 GB | 1.6 × 10⁹ |

超出预算时的处理方式是**限制可训练的 key 总数**（收紧 backing 的准入门槛，见 §6.1 的旁路路径），而不是改变索引的存放位置。

### 10.4 显存预算的切分：索引 vs 缓存

索引与缓存在同一块 HBM 上竞争，两者的单位成本相差两个数量级：

```
backing 索引 ：  ~20 B / key        （决定能训练多少 key）
缓存行池    ： 1536 B / node        （决定命中率，dim=128、fp32、Adam）
```

即 **1 GB 显存可以换 5000 万个可索引的 key，或者 65 万个缓存行**。索引每 key 便宜约 75 倍。

但两者的**数量级不同**，所以实际占用未必悬殊。以 `N = 4 × 10⁸`、`C = 4M` 为例：

```
backing 索引 ： 8 GB
缓存行池    ： 6 GB
缓存元数据  ： 0.13 GB
```

二者相当。这意味着容量规划是一个真实的权衡：**多给索引 = 能容纳更多 key；多给缓存 = 更高命中率、更少 PCIe 流量**。建议的做法是先由业务侧的 key 总数定死索引预算（它是硬约束——装不下就训练不了），剩余显存全部给缓存，再用 §12.3 的命中率指标验证缓存容量是否够用。

---

## 11. 接口

### 11.1 配置

```python
@dataclass
class LightweightCacheOptions:
    capacity: int                      # 每表 node 数 C_t
    replacement: ReplacementPolicy = ReplacementPolicy.FIFO
    clock_max_second_chances: int = 1  # §8.4 的 k
    prefetch_depth: int = 1            # 支持 > 1，见 §6.4
    record_metrics: bool = False
```

### 11.2 核心 API

```python
lookup(unique_keys, table_range)            -> node_ids, found
allocate(miss_keys, table_range)            -> node_ids            # 含淘汰与双向搬运
release(node_ids, table_range)              -> None
flush(table_id=None)                        -> None                # checkpoint 用
metrics()                                   -> CacheMetrics
```

接口面积刻意保持在 5 个方法以内。缓存不提供 `expand` / `dump` / `load`——这些属于 backing 表。

---

## 12. 待验证事项

### 12.1 PCIe 双向带宽 — 已实测

**结论：链路是全双工的，但我们的访问形态只有在行有序时才吃得到。**

H100（PCIe Gen5 x16）实测，`corelib/dynamicemb/debug_pcie_bidir_bw/`（本地目录，已被 `.gitignore` 忽略，不在仓库中）：

| 形态 | H2D | D2H | 双向 | 增益 |
|---|---|---|---|---|
| copy engine，连续 | 51.97 | 51.33 | **94.44** | **1.82×** |
| kernel，零拷贝，连续 | 47.85 | 49.01 | 78.53 | 1.60× |
| **gather，随机行，1536 B/行** | 39.56 | 39.03 | **47.08** | **1.19×** |
| **gather，有序行，1536 B/行** | 52.15 | 49.16 | **83.15** | **1.59×** |

（GiB/s；gather 行即 §9.2 的 warp-per-row 形态，是缓存实际做的事）

copy engine 拿到 1.82×，说明链路本身接近理想全双工。但缓存走的是随机 gather，**未排序时双向只有 1.19×**——融合两个方向只省 19%，远低于设计时的隐含预期。排序后回到 1.59×。

§12.2 的复测显示排序在真实密度下几乎无收益，因此**真实工作点上的行等同于无序**，融合能拿到的是 **1.19×** 那一行，不是 1.59×。

折算成传输时间约省 20%（两趟串行 `2MB/39.2` 对融合 `2MB/47.1`）。**仍然值得做**——miss 路径的 PCIe 开销是这条链路的主要成本——但远不是原设计隐含的接近一倍，也不需要任何前置工作。

### 12.2 回写行排序 — 已实测，结论是**不做**

初测（`pcie_bidir_gather`）显示排序有 1.32×–3.55× 的收益，据此一度把排序提到融合之前。**复测推翻了这个结论**：那个收益是测量条件的产物，不是真实工作点的性质。

初测排的是 400 万个索引、落在 1120 万行上——**密度 36%**，排完几乎就是顺序扫描。而且它排的是**全部**索引，这是缓存做不到的：缓存只看得见一步的 miss。

复测（`pcie_density_sweep`）改成**只在步内排序**，并走一个大索引池以免同一批页的地址翻译在迭代间保持常驻：

| 表 / 行宽 | 每步行数 | 密度 | 未排序 | 已排序 | 增益 |
|---|---|---|---|---|---|
| 64 GiB / 1536 B | 4 096 | 0.0092% | 30.44 | 30.66 | **1.01×** |
| 64 GiB / 1536 B | 16 384 | 0.037% | 36.59 | 38.64 | **1.06×** |
| 64 GiB / 1536 B | 262 144 | 0.59% | 37.90 | 47.03 | 1.24× |
| 64 GiB / 512 B | 16 384 | 0.012% | 13.79 | 14.03 | **1.02×** |
| 64 GiB / 512 B | 1 048 576 | 0.78% | 13.00 | 22.80 | 1.75× |
| 16 GiB / 512 B | 1 048 576 | 3.1% | 13.92 | 48.40 | 3.48× |

（GiB/s，H100 PCIe Gen5 x16）

**真实工作点的密度约 1e-4**：每步数万个 miss，对应数亿行的 backing。落在表格最上面几行，**收益 1–6%**。收益要到 0.2% 密度以上才显现，1% 以上才可观。

**结论：不做。** 为 1–6% 去重构 allocate 各趟、外加每步两次排序，不划算。相应地，原 P1.5 里程碑取消。

#### 一个反而被加强的结论

未排序带宽在各密度下**几乎恒定**（1536 B 行 ~39 GiB/s，512 B 行 ~13.9 GiB/s）——这就是随机访问的地板，与批大小无关。

于是行宽的 **2.8 倍**差距在真实密度下**无法靠排序弥补**。§10.1 关于窄表需要更多容量的结论因此比初测时更强，而不是更弱。

另可见每步行数过小（4096）时带宽也会掉（30.4 vs 37.9）——批太小喂不满并行度。这给"prefetch 深度换更大的 miss 批"提供了一个额外理由。

### 12.3 CLOCK 的实际收益

**测什么**：FIFO 模式下 §8.5 的 `used == 1` 占比指标，以及两种策略下的命中率与 E2E step time。

**如何影响设计**：决定 CLOCK 是否作为推荐默认值。注意由 INV-7，此项只需性能测试。

### 12.4 缓存容量是否够用

**测什么**：命中率、ring 抽干次数、以及每步 miss 数 `M` 与 unique key 数 `U` 的比值。

**如何影响设计**：`N` 由业务侧的 key 总数定死索引预算后，剩余显存全部给缓存（§10.4）。若命中率仍不理想，说明工作集超出了可分配的缓存容量，此时的选项是收紧 backing 准入门槛（减少 `N`，把显存从索引挪给缓存），而不是调整缓存本身的结构。

### 12.5 每节点自旋锁 vs 无锁状态字 — 完工后再对比

当前实现是**无锁**的：状态打包进一个 32 位字，`try_pin` / `try_claim` / `release` 各是一个 CAS 循环。备选方案是给每个 node 加一把 32 位自旋锁，锁内用普通读写改状态。**留待整体跑通后实测对比，不在此之前切换。**

**这把锁不占额外内存。** 当前 node 布局是 `key(8) + prev(4) + next(4) + state(4) = 20`，按 8 字节对齐后是 24，浪费 4 字节；加上锁正好是 24，浪费 0。

**它能买到两样东西：**

1. **位打包整体消失**。打包存在的唯一理由是"多字段修改必须原子"。有锁之后 `refs` 可以是完整 32 位、residency 可以是 `uint8_t`、加第五个状态或第二个计数器都是零成本。`node_state.h` 里的掩码、移位、进位借位边界分析和对应的穷举测试都不再需要。
2. **pin 身份竞态（§6.3）消失**，前提是这把锁同时锁住 `key`。当前流程是"CAS 成功 pin → 再依赖读一次 `key` 复查 → 不符则 unpin 降级为 miss"，因为状态字会回到位模式完全相同的值（ABA），CAS 分辨不出节点是否已改派给别的 key。锁住 key 之后，`residency == RESIDENT && key == wanted` 和 `refs+1` 在同一个临界区里完成。

**性能上不一定更慢，这是需要实测的原因。** `key` 与 `state` 同在一个 24 字节节点、同一条 cache line，而链表遍历本来就已经读过 `node.key`。当前那次身份复查是**排在 CAS 完成之后的依赖读**，是一次真实的串行延迟；锁的版本把它挪进临界区，用的是同一次加载的数据。净差可能只是多一次 release store。

**它的代价：**

- 临界区从"一条 CAS 指令"变宽到"取锁 + 比较 + 写 + 放锁"，争用窗口更大。争用发生在热 key 上的 `pin ∥ release`（§6.4，两者跨 stream 且无序），这正是要测的场景。
- 桶锁与节点锁并存，`insert` 改 key 时需同时持有两者，必须定死加锁顺序。这一项是复杂度**增加**。

**测什么**：热 key 上 N 线程并发 pin/release 的微基准，两种实现各一遍；以及 E2E step time。

**触发条件（无需实测即可决定切换）**：状态装不进一个 32 位字的时候——需要第五个 residency、第二个计数器、或任何必须与 `refs` 原子地一同修改的每节点元数据。届时"一个 CAS 搞定"不再成立，锁反而更便宜。

---

## 13. 里程碑

| 阶段 | 内容 | 验收 |
|---|---|---|
| **P0** | 单表、FIFO、链式哈希、深度 1、串行加载 / 回写；单元测试覆盖全部不变量 | INV-1..9 断言通过；与 backing-only 路径 loss 逐位一致 |
| **P1** | 多表融合 kernel；双向搬运 kernel；prefetch 流水；健康度指标 | HSTU / MovieLens E2E 不低于现有实现（G6） |
| **P2** | CLOCK 可选；容量规划挂靠 planner（含 §10.1 的行宽项） | 命中率与 step time 对比报告 |

链式哈希在 P0 采用，因其删除逻辑清晰、易于调试。**开放寻址 + digest 列为 P2 优化候选**：可省去 `prev`/`next` 共 8 B/node，且把桶链的"依赖性随机访问"（每跳一次不可掩盖的数百周期延迟）压成一两次连续访存；仓库现有的 `LinearBucketTable` 已验证过该结构。若 profile 显示 lookup 占比显著，这是第一个应替换的部件。

---

## 14. 风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| INV-6 被破坏导致 ref 泄漏或状态字腐蚀 | 命中率单调下降；或 node 状态错乱 | 释放路径 `assert(REF(old) > 0)`；checkpoint 排空后全量校验（§9.5） |
| 桶链并发修改 | 链表结构损坏，查找失效或死循环 | 桶级锁（§6.1）；查找与任何修改不并发 |
| 多条 prefetch stream 破坏阶段分离 | ring 竞态、pin 身份竞态 | 约束为单条 prefetch stream（§6.4） |
| 读 backing 的 kernel 与 backing 扩容并发 | 读到被搬动的索引或行，静默数值损坏 | 现有后端已满足（§4.5 规则 2）；缓存 kernel 须与 `expand_if_need` 在同一条 prefetch 流 |
| backing 淘汰了仍在 in-flight 的缓存 key | 行被重新分配，反向写到别人的行上 | §6.5 的保护：backing 选 victim 时先查缓存，ref 不为 0 则跳过 |
| 一个桶里的 key 全部受保护，backing 插入失败 | 新 key 无法准入 | 抛出容量错误（§6.5 第 7 步、§17.5）；容量规划保证 backing 远大于缓存与 in-flight 的 key 数 |
| 容量规划不当，热表 ring 抽干 | 训练中断 | 配置期校验；ring 抽干计数作为告警指标 |
| PCIe 非全双工 | §9.2 收益落空 | §12.1 前置验证；退化路径已设计 |
| key 总数超出索引显存预算 | 无法训练全部 key | 配置期按 §10.3 的公式校验；收紧 backing 准入门槛 |
| CLOCK 出队循环退化 | 分配延迟抖动 | §8.4 限次回退；默认 FIFO |

---

## 15. 被否决的备选：索引融合

**思路。** 不让缓存自己维护 key→node 的哈希，而是在 backing 索引的每个槽位上加一个 `node_id` 字段（`-1` 为不在缓存），node 里存 backing 槽号作为反向指针。一次 backing 探测同时回答"这个 key 的行在哪"和"它在不在缓存"；淘汰时清 `node_id[slot]` 是一次 O(1) 写；缓存侧的索引结构（桶链、`prev/next`）整个消失，每 node 元数据可降到约 12 B。

**为什么不采用。** 它的前提是 backing 槽位稳定，而 DynamicEmb 的扩容会 rehash，所有反向指针同时失效，且 node 里没有 key，无法自行修复。三种应对都不成立：

| 应对 | 问题 |
|---|---|
| 预分配到上界，永不 rehash | 需要预知 key 总数上界，且扩容是常态，不能假设它不发生 |
| rehash 时整体 flush 并清空缓存 | 每次扩容都付一次全量脏行 dump 与重新预热 |
| node 里额外存 key，rehash 后按 key 重新探测修正 | 每 node 多 8 B，扩容时多一遍修正，且此时缓存侧又有了 key，与本设计只差一个索引，收益所剩无几 |

另外它的成本按 backing 槽数计（每槽 4 B），收益按缓存容量计，`N/C`（总 key 数与缓存容量之比）很大时得不偿失，且把 backing 索引锁死在 GPU 上。

**结论。** 重哈希是硬需求，因此不采用。本设计让缓存自带索引、node 里存 key，换来的正是对 backing 重哈希免疫（§4.1、§4.5）。完整的原设计文档 `cache_fused_index_design.md` 在 git 历史中。

---

## 16. 被取代的原型：`lw_cache`

本文最初对应 `src/lw_cache/` 与 `test/lw_cache/`，现已由 `src/embedding_cache/` 取代并删除。完整代码在提交 `0c2ee1a` 里（`git show 0c2ee1a:corelib/dynamicemb/src/lw_cache/...`；若历史被改写，用 `git log --all -S lw_cache` 找）。取代的原因与取走了什么：

- **取代的原因**：`embedding_cache` 在同一设计上把并发协议收得更紧（`release` 与 `claim` 各为一次 CAS，节点一行、桶锁带 acquire/release 内存序，lookup 与压紧融合，ring 的聚合原子），并避开了读 `lw_cache` 代码时发现的两处问题（未复现，是按代码推出来的）：`release_ref` 先减后置位的两步窗口，以及桶锁成功后无 acquire 就读 `prev/next`。
- **还没有对应实现、需要时从历史里取的**：
  - **CLOCK**（`absorb_retry_kernel`、retry buffer）。有意推迟：先用 `used` 位占比测出是否值得做。取的时候带上它 README 里记的"计数没清零"缺陷，以及"CLOCK 的容量需求是 FIFO 的 `(1+k)` 倍"。
  - **指标**（`metrics.h`）与**容量校验**（`CacheConfig::validate`）。
  - **flush**（`flush_kernel`、`count_referenced_kernel`）。
- **有意不采用**：多表融合分派（见 §4.4）、`backing.h` 的 `row_ptr` 契约（现在 backing 是 Python 现有的 `key_index_map` 加各表值缓冲指针，行号由外部给，见 §4.5）。
- **值得借鉴的测试做法**：对着一个纯 Python 模型逐步比较的差分测试（`test_differential.py`、`test_multi_table.py`）。写 `embedding_cache` 的 Python 绑定时，端到端测试应照这个办法做。

---

## 17. 接入现有系统：方案、假设与待定项

目标：用 `src/embedding_cache/` 替换现有的 `DynamicEmbCache`。这一节记录接入的方案、做出的假设，以及目前不确定、先按假设推进的点。每完成一个阶段就更新状态。

### 17.1 现有系统里缓存是怎么被使用的（读代码得到）

- `forward/backward` 只通过三样东西使用缓存：`cache._state`（各表行池的指针 `table_ptrs_dev`、各表维度、`optimizer`）、`value_row_indices`（就是缓存里的行号）、反向里的 `cache.decrement_counter(slot_indices, table_ids)`。所以新缓存的行池只要仍以"每表一个张量加 `table_ptrs_dev`"的形式暴露，前向与反向几乎不用改；`decrement_counter` 换成 `release`。
- `dynamicemb_prefetch` 里 `storage.expand_if_need(...)` 在缓存路径之前（§4.5 已核对）。
- 缓存还被这些地方用到：显存报告（`_print_memory_consume` 读 `cache._state`）、`flush`/`flush_cache`、`reset_cache_states`、`cache_metrics`、eval 路径（`_eval_lookup_cached`）、增量回放后的失效（`_invalidate_cache` 调 `cache.key_index_map.erase`）、`outstanding_keys` 的容量检查（全局，对比缓存容量）。
- **后端原有两类**：`DynamicEmbStorage`（行可寻址，有 `key_index_map`），以及外部存储 PS（只实现 `Storage` 接口：按 key 的 `find`/`insert`，拷贝值），PS 当时直接取代 host 表。现在后端只有 `DynamicEmbStorage`，PS 是它下面的一层，见 §17.8。

### 17.2 方案

- 保留类名与导入路径：`DynamicEmbCache` 仍在 `dynamicemb.key_value_table`，只换内部实现。这样 `isinstance(cache, DynamicEmbCache)` 一类的检查不变。
- 新增 C++ 绑定 `EmbeddingCacheCore`（`src/embedding_cache/binding.cu`），持有每表的桶链、node、ring，提供：`lookup_pin`、`allocate`、`publish`、`release`、`find_nodes`、`invalidate`、`resident_slots`、`reset`。行池由 Python 持有（每表一个张量，沿用 `table_ptrs_dev`），绑定只拿指针。
- 分两阶段做（阶段 1 是为了能先验证绑定与编排，已在阶段 2 完成后删除）：
  - **阶段 1：通用流程（拷贝式）。** 数据搬运沿用 `storage.find` / `storage.insert`，后端在缓存淘汰时才被插入（惰性）。当时用来承载 PS 后端；PS 改为第三层之后没有调用方，已删除。
  - **阶段 2：行可寻址流程（§6.5）。** 每步所有 key 先过 backing（更新 score），新 key 准入后立刻插入 backing，淘汰前用 `EvictionGuard` 保护。需要改后端的 `insert` kernel 与 LruLfu cubin（§17.4）。现在是唯一的流程。

### 17.3 假设与待定项

不确定的点先按下面的假设推进，之后如果证明不对再回来改。

1. ~~**PS 后端只走阶段 1 的流程**~~ （作废：PS 成为第三层，缓存的后端永远是 `DynamicEmbStorage`，见 §17.8。）
2. **缓存没有 score，`training` 与 `set_score` 变成空操作**，只为兼容调用方保留。待定：阶段 1 下回写后端时的 score 如何给（先传 `None`，由后端按当前 step 的默认值处理；LFU 策略下频率信息会丢失，这是阶段 1 的已知缺口，阶段 2 才补上）。
3. **`cache_metrics` 保留四个槽位的含义**：`[0]` 本步 unique 数，`[1]` 命中数，`[2]` 插入数，`[3]` 淘汰数。
4. **`key_index_map` 属性不再存在**，测试里直接读它的地方要改；`size()` 用统计 RESIDENT 节点数实现。
5. **容量**：每表缓存容量仍是 `int(max_capacity * cap_scale)`，不再有 overflow 区，显存报告与对齐测试里关于 overflow 的断言要改。（实现时改为仍向上取整到桶容量，见 17.5。）
6. **`outstanding_keys` 的容量检查改成每表一份**，用 host 已有的 `unique_size_per_table`。
7. **每步一次 D2H 同步**（读各表的 miss 数）。现有路径本来就有多处同步，先接受，等有 profile 再去。
8. **多流**：allocate 在 prefetch 流上，release 在反向所在的流上。§6.4 要求的 tail 快照屏障（快照前 allocate 的流等已发起的 release，快照后发起的 release 等"快照已取"事件）在绑定里用 CUDA event 实现。这是阶段 1 就必须正确的部分，因为旧缓存靠原子计数，没有这条要求。
9. **eval 路径**：缓存里只读（`find_nodes`，不 pin、不准入），不在缓存里的从 backing 取，不插入。
10. **增量回放后的失效**（`_invalidate_cache`）：`find_nodes` 加 `invalidate`，不回写。这正是"缓存副本比后端新"的反面（后端刚被写成新值，缓存副本过期），语义与淘汰时的作废一致。
11. **dtype**：阶段 1 搬运全部沿用现有按 dtype 的 kernel，缓存自己的 kernel 不接触值。阶段 2 的零拷贝搬运按字节拷贝，不按元素类型实例化（先按 4 字节对齐，向量化等 profile）。
12. **旧实现的删除**：新实现通过现有测试之后，删除旧 `DynamicEmbCache` 内部用到的、不再被任何路径使用的 kernel 与代码。哪些算"不再使用"在删除时逐一核对。

### 17.4 阶段 2 对后端的改动（设计，已完成，见 17.6）

后端的普通 `insert` 在选 victim 的两处检查计数：选候选时（`reduce`、`reduce_ranked`），以及锁住之后再检查一次。改动是把"计数是否大于零"泛化成一个判定对象，计数仍是默认行为，缓存模式下再加一条"缓存是否允许淘汰这个 key"。只改普通 `insert` 与带收集的 `insert`（缓存模式下急切插入用的）及其在 LruLfu cubin 里的对应入口；`insert_and_evict` 不动。判定对象是不含指针以外成员的 POD，放在 cubin 与 AoT 共用的小头文件里，因此 `embedding_cache` 里的视图类型要拆成"只含视图"和"含存储"两部分，前者才能被 cubin 包含。

### 17.5 实现中做出的决定（与 17.3 的假设对照）

- **缓存容量取整**：每表 `C_t` 向上取到桶容量（缓存是 1024）的整数倍，与旧缓存的索引一致，只是不再有 overflow 区。原因：缓存必须同时装下一个 step 的全部 key，极小的配置（几百行）原来靠取整才能跑，新实现若恰好取配置值，现有测试里一批这样的配置会直接报容量错误。17.3 里"恰好是 `C_t`"的假设作废。
- **只有一种驱动方式**（`_prefetch_cache_path`）：**急切流程**（§6.5）。每个 key 先过后端（`_find_keys`，更新 score），新 key 准入后立刻插入后端（带守卫、收集被淘汰的 key），缓存再据此作废。LFU/LRU 的 score 因此对每次访问都准确，不论缓存是否命中。确定性模式原先走不了，因为后端的确定性插入不能报告淘汰了哪些 key；现在 `_deterministic_insert` 逐波用 `table_insert_collect_evicted` 并按波顺序拼接淘汰列表，所以也走这条。
- **急切流程里后端插入失败**（一个桶里所有可淘汰的 key 都被守卫保护）：抛出明确的容量错误，不降级为"未准入"。这是容量规划错误，与缓存装不下一个 step 的 key 同类。已定：保持报错；降级路径目前没有调用方，需要时再加。
- **`flush`**：原地写行（每个缓存里的 key 在后端都有行），不再 `insert`，也不再预先扩容后端；若发现缓存里有后端已没有的 key（理论上不该有），回退到 `insert`。
- **取值交给保留逻辑**（§6.5 第 8 步）：目前 `EvictedItemMode` 只有 `RETAIN_KEY`，只需要 key，所以急切流程里只把被后端淘汰的 key 交给 `_append_evicted`，没有取值。`rows.cuh` 的 `fetch_rows_kernel` 已备好，但没有任何调用者，所以没有暴露到绑定里；等 `EvictedItemMode` 增加保留值的标志时再接。
- **`DEMB_DETERMINISM_MODE`**：新缓存里哪个 key 占哪个 node 取决于线程到达 reclaim queue 的顺序，不固定，也不影响任何 key 的值。急切流程下后端的插入顺序与缓存无关，由后端的确定性插入决定，所以后端的淘汰是确定的。

- **事件接线没有测试证明**：绑定里用两个 CUDA event 实现 §6.4 的快照屏障（allocate 等已发起的 release，release 等已取的快照）。`test_embedding_cache_core.py` 里的两流压力测试检查了"被持有的节点没有被复用"和"静止时队列条目数等于带标志的节点数"，但**去掉这两个事件后它仍然通过**（五次），因为那个窗口只有几条指令宽。所以事件是否必要，只有 `test/embedding_cache/gpu` 里显式控制流序的设备测试在覆盖；绑定层的接线是按设计写的，没有被变异检验证明。

### 17.6 后端的守卫（已完成）

后端普通 `insert` 的两处计数检查（`reduce`、`reduce_ranked`）和锁后复查，都增加了对 `EvictionGuardView`（`embedding_cache/guard.h`）的询问；AoT 的 `table_insert_kernel`、`table_insert_collect_kernel` 与 LruLfu cubin 的 `dyn_emb_insert_entry`、`dyn_emb_insert_collect_entry` 都接上了。`ScoredHashTable.insert(..., guard=...)` 与两个绑定函数多了两个默认为 0 的参数（守卫的设备指针与表数），不传即与原来一致。测试：`test/unit_tests/table_operation/test_eviction_guard.py`，覆盖两条淘汰路径、带与不带 `collect_evicted`、"没有守卫时这些 key 会被淘汰"的对照，以及一个桶里全是受保护 key 时插入被拒。

### 17.8 PS 作为第三层

外部存储（PS）不再取代 host 表，而是接在表的下面。HBM 表、host 表、"HBM 缓存 + host 表"三种布局下都一样：前面那张表（缓存的后端就是这张 host 表）按现在没有 PS 时的大小建（`max_capacity` 不变），PS 接住它淘汰出来的行，并在它没有某个 key 时被查。`StorageLayout` 里原来的 `CACHING_PS`、`HOST_PS` 去掉了，PS 是 `DynamicEmbStorage.lower`。

- **读（表里没有的 key）**：向 PS `find`。PS 里有的 key 不算新 key，不过准入器，值用 PS 的；它们随后被插入表里（"升上来"），并在缓存布局下进缓存。通用路径（host 表、eval）由 `DynamicEmbStorage.find` 合并 PS 的结果；HBM 直连与缓存的急切流程在各自的步骤里显式查。
- **写（表淘汰的行）**：连同值写进 PS。值要在新 key 的行被写之前读出，所以后端的 `collect_evicted` 多返回了被淘汰 key 原来占的槽位（新 key 占了同一个槽，它的值行还是旧值）。通用路径与 HBM 直连用 `insert_and_evict`（本来就返回槽位、score），缓存的急切流程用带守卫的 collect。缓存里有副本的被淘汰 key，值取缓存的（比后端新），再作废缓存项。
- **同一次插入里自己被淘汰的 key**：后面的 key 可能淘汰前面刚插入的 key，它的行还没写，读出来是别的 key 的旧值。这时它自己的值是正在插入的那个（`take_values_of_keys_evicted_in_the_same_insert`）。去掉这一步，确定性模式下的测试会失败（一次插入的各"波"按序执行，这种情况必然出现）；非确定性模式下取决于竞争，测试没有稳定复现。
- **后端的"保持已有"插入**（反向里刷新值）：表里没有、是从 PS 读出来的 key，按新 key 插入（有自己的 score），而不是用不变分数的 CONST 策略——后者给的分数最低，会被下一个 key 立刻淘汰。这是在确定性模式的端到端测试里发现的。
- **PS 里的旧副本**：key 升上表之后，PS 里那份是过期的，不删（`Storage` 没有删除接口）。表是权威：先问表，写表在后，所以不影响正确性。代价是 PS 里可能留着已升上来的 key 的旧值，在表淘汰它时被覆盖。
- **PS 不参与淘汰决策，也没有 score 的契约**：回写时带上表给的 score，但读回来时表按自己的策略给分，不沿用 PS 的。LFU 的历史频率因此在经过 PS 一次后丢失。已知缺口。
- **dump / load / 导出 / 增量 dump / `erase` 只覆盖表，不含 PS**。PS 是外部存储，自己负责持久化。`erase` 不会删掉 PS 里的副本，所以被擦除的 key 之后仍可能被 PS 读回来。已知缺口。
- **`RETAIN_KEY`（保留被淘汰的 key）**：表下面有 PS 时，被淘汰的 key 没有离开系统，所以不保留（表的 `evicted_item_mode` 被置为 `DISCARD`）。
- **HybridStorage 仍忽略 `external_storage`**（警告）。
- **规划**：`get_group_value_bytes_by_tier` 不再把 PS 当成"把后端从这台机器上拿走"，host 字节数与没有 PS 时一样。

测试：`test/unit_tests/test_ps_tier.py`，三种布局 × 默认/确定性模式，让工作集（6000 个 key）大于表（2048 行），用"每个 key 初始 1.0、loss 为输出之和、SGD"检查每个 key 最终都是 `1 - lr × 出现次数`，并对一个丢弃一切写入的 PS 做对照（必须失败）。另有一个单独的测试覆盖"同一次插入里自己被淘汰"。

### 17.7 进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| 0 | 在 EOS 上建立构建与测试流程，记录基线 | 完成 |
| 1a | C++ 绑定 `EmbeddingCacheCore` | 完成 |
| 1b | Python：新 `DynamicEmbCache`，通用流程 | 完成 |
| 1c | 现有测试跑通，更新直接依赖旧内部结构的测试 | 主测试文件与大部分单测通过，见下 |
| 2a | 后端 `insert` 的守卫与 cubin 入口 | 完成 |
| 2b | 急切流程，`DynamicEmbStorage` 后端切换过去 | 完成，LFU 测试脚本通过 |
| 3 | 删除旧实现，更新文档与示例 | 完成：旧缓存代码已无引用 |
| 4 | PS 作为表下面的第三层，删除 lazy 流程 | 实现完成，见 §17.8；全量测试与 benchmark 见下 |

性能（EOS H100）。旧实现 → 新实现；两棵树同时各占一块 GPU（0 与 4）跑，所以数字带有同机干扰，只能说明方向。

`benchmark_batched_dynamicemb_tables.py`（单卡，10 表，D=128，每步约 43 万唯一 key；ms）：

| 配置 | forward | backward | train | eval |
|---|---|---|---|---|
| gpu adam | 1.0 → 1.0 | 0.9 → 0.9 | 1.9 → 1.9 | 0.6 → 0.6 |
| gpu sgd | 1.0 → 1.0 | 0.6 → 0.6 | 1.6 → 1.6 | 0.6 → 0.6 |
| caching adam cfr=0.8 | 23.9 → 20.1 | 1.0 → 1.0 | 24.9 → 21.1 | 3.4 → 3.7 |
| caching adam cfr=1.0 | 10.8 → 2.9 | 1.0 → 1.0 | 11.8 → 3.9 | 1.1 → 0.7 |
| caching sgd cfr=0.8 | 19.4 → 17.2 | 0.7 → 0.7 | 20.1 → 17.9 | 3.2 → 3.3 |
| caching sgd cfr=1.0 | 9.7 → 2.7 | 0.7 → 0.7 | 10.4 → 3.4 | 1.1 → 0.7 |
| no_caching adam | 16.5 → 16.3 | 16.2 → 16.2 | 32.7 → 32.5 | 20.2 → 20.1 |
| no_caching sgd | 13.3 → 13.3 | 13.4 → 13.3 | 26.6 → 26.6 | 17.2 → 17.2 |
| no_hbm adam | 21.6 → 21.3 | 21.3 → 21.0 | 43.0 → 42.3 | 28.5 → 27.9 |
| no_hbm sgd | 17.4 → 17.2 | 17.3 → 17.3 | 34.7 → 34.5 | 23.4 → 23.3 |

没有退化。只有缓存配置变了：cfr=0.8 时 train 快约 15%，cfr=1.0（工作集全在缓存里）时快约 3 倍。cfr=0.8 的 eval 3.4 → 3.7 ms 是个小幅变慢，一次运行分不清是不是噪声。

`benchmark_embedding_collection.py`（8 卡，批 65536，每次迭代 ms）：

| 配置 | 旧 | 新 |
|---|---|---|
| 动态表，开 dedup（三次重复） | 13.28 | 13.29 |
| 动态表，关 dedup | 19.83 | 19.87 |
| TorchRec 原表，开 dedup | 12.36 | 12.27 |

第一次运行里开 dedup 的新实现是 13.66，随后三次重复都是 13.29，是噪声。

---

## 附录 A：可复用的现有代码

| 用途 | 位置 |
|---|---|
| 分段去重（含 table_range 输出） | `src/unique_op.cu`、`dynamicemb/batched_dynamicemb_function.py:74` |
| 全新 key 的初始化 | `src/initializer.cu`、`src/initializer.cuh` |
| 优化器原地更新 | `src/optimizer_kernel.cuh` |
| 前 / 反向 gather-scatter | `src/lookup_forward.cu`、`src/lookup_backward.cu` |
| host 侧可扩展缓冲（mapped pinned） | `dynamicemb/extendable_tensor.py:131`、`src/vmm_tensor.cu` |
| 逐表指针数组模式 | `dynamicemb/key_value_table.py:160` |
| PCIe 双向带宽基准 | `corelib/dynamicemb/debug_pcie_bidir_bw/pcie_bidir_gather.cu` |

需要新写的仅有三部分：链式哈希 kernel（查找 / 插入 / 摘链）、ring 的领取与校验、状态字的原子协议。

## 附录 B：与现有实现的对照

| 维度 | 现有 `DynamicEmbCache` | 本设计 |
|---|---|---|
| 索引结构 | `LinearBucketTable`，开放寻址 + digest + score | 链式哈希（P2 考虑换回开放寻址） |
| 每 slot 元数据 | key + digest + score + ref_counter + overflow 计数 | key + state + prev/next + row |
| overflow buffer | `3 × bucket_capacity × num_tables` | 无（缓存允许淘汰，不需要） |
| 替换依据 | score（int64 时间戳 / LFU 频次） | `used` 1 bit（CLOCK）或纯 FIFO |
| 行宽 | 所有表取 `max(emb_dim + optim_state_dim)` | 逐表独立 `D_t` |
| 与 storage 的关系 | 共用 `DynamicEmbTableState` | 完全解耦，仅依赖 key→row 查询与 host 行读写 |
| 容量扩展 | 支持（继承自 storage） | 不支持（缓存定长） |
| flush 内存尖峰 | backing 需预扩容 | 无（准入即分配行） |
| 脏位管理 | `dirty` 位，条件回写 | 无（INV-9，淘汰时无条件回写） |

---

## 附录 C：完整示例走查

用一个极小的配置走完三个 training step，覆盖冷启动、命中复用、`in_ring` 防重复入队、陈旧条目丢弃、淘汰与双向回写。

### C.0 例子设定

```
C = 4           node n0 .. n3
B = 8           桶 b0 .. b7，hash(k) = k & 7
D = 6           emb 4 列 + optimizer state 2 列
单表，prefetch 深度 = 1，替换策略 = FIFO
```

**初始状态**（`ring = arange(4)`，全部 `EMPTY` 且 `in_ring = 1`）：

```
nodes                                  ring
┌────┬─────┬───────┬─────┬────────┐        idx  0    1    2    3
│ id │ key │ state │ ref │in_ring │            ┌────┬────┬────┬────┐
├────┼─────┼───────┼─────┼────────┤            │ n0 │ n1 │ n2 │ n3 │
│ n0 │  -  │ EMPTY │  0  │   1    │            └────┴────┴────┴────┘
│ n1 │  -  │ EMPTY │  0  │   1    │              ↑head=0        tail=4
│ n2 │  -  │ EMPTY │  0  │   1    │
│ n3 │  -  │ EMPTY │  0  │   1    │        hash_heads : b0..b7 全为 -1
└────┴─────┴───────┴─────┴────────┘        emb_table  : 4 行未初始化
```

---

### C.1 Step 1 —— 冷启动，三次 miss

**输入** `keys = [10, 18, 3, 10, 18]`

**① 去重**（`segmented_unique_cuda`）

```
输入 keys : [ 10, 18,  3, 10, 18 ]
                 ↓  去重
unique    : [ 10, 18,  3 ]              U = 3
inverse   : [  0,  1,  2,  0,  1 ]      前向 gather 用它还原到每个位置
```

**② 查找** —— `hash_heads` 全空，三个 key 全部 miss。

**③ ring pop ×3**

```
        idx  0    1    2    3
           ┌────┬────┬────┬────┐
           │ n0 │ n1 │ n2 │ n3 │
           └────┴────┴────┴────┘
             ↑    ↑    ↑
           pop  pop  pop           head: 0 → 1 → 2 → 3      tail = 4

三个候选都是 EMPTY 且 ref == 0，CAS claim 全部成功
分配结果：key 10 → n0     key 18 → n1     key 3 → n2
```

**④ backing find-or-insert**（准入即分配行，§9.3）

```
key 10 → 行 100        key 18 → 行 205        key 3 → 行 7

行号只在本次分配中使用（决定从哪加载），不写入 node（§4.1）
```

**⑤ 挂桶链**（Step 4b 的插链趟；本步无受害者，故摘链趟为空）—— `hash(10) = 2`，`hash(18) = 2` **同桶冲突**，`hash(3) = 3`

```
hash_heads                桶链（头插法，18 后插故在链头）
┌────┬─────┐
│ b0 │ -1  │
│ b1 │ -1  │
│ b2 │ n1  │──→ ┌────────┐  next  ┌────────┐
│ b3 │ n2  │    │n1 key18│ ⇄─────→│n0 key10│──→ nil
│ b4 │ -1  │    └────────┘  prev  └────────┘
│ .. │ ..  │
└────┴─────┘    ┌────────┐
      └────────→│n2 key 3│──→ nil
                └────────┘
```

**⑥ 加载**（本例假设三个 key 在 backing 中已存在；全新 key 则走 `initializer` 在 GPU 上直接初始化，不产生 H2D）

```
host[100] ──── PCIe H2D ────→ emb_table[0]
host[205] ──── PCIe H2D ────→ emb_table[1]
host[  7] ──── PCIe H2D ────→ emb_table[2]

状态迁移： EMPTY → LOADING → RESIDENT，  ref = 1
```

**⑦ 前向** `out[i] = emb_table[node_ids[inverse[i]], :4]` —— 5 个输入位置由 3 行数据供给。

**⑧ 反向** —— 优化器原地更新三行（emb 与 optim state 同行，一次访存拿全），`ref 1→0` 后入队。**没有 `dirty` 置位**——三行现在都比 backing 新（INV-9），将来淘汰时无条件回写：

```
push n0 → ring[4 & 3] = ring[0] = n0     tail: 4 → 5
push n1 → ring[5 & 3] = ring[1] = n1     tail: 5 → 6
push n2 → ring[6 & 3] = ring[2] = n2     tail: 6 → 7
```

**Step 1 结束**

```
nodes                                    ring
┌────┬─────┬──────────┬─────┬───────┐        idx  0    1    2    3
│ id │ key │  state   │ ref │in_ring│            ┌────┬────┬────┬────┐
├────┼─────┼──────────┼─────┼───────┤            │ n0 │ n1 │ n2 │ n3 │
│ n0 │ 10  │ RESIDENT │  0  │   1   │            └────┴────┴────┴────┘
│ n1 │ 18  │ RESIDENT │  0  │   1   │                        ↑head=3
│ n2 │  3  │ RESIDENT │  0  │   1   │                              tail=7
│ n3 │  -  │  EMPTY   │  0  │   1   │
└────┴─────┴──────────┴─────┴───────┘   node 中不存行号（§4.1）

环的逻辑顺序（从 head 到 tail，含环绕）：
   idx3  →  idx0  →  idx1  →  idx2
    n3       n0       n1       n2
   (空)    (key10)  (key18)  (key3)
   队头 ←──────────────────────→ 队尾
```

---

### C.2 Step 2 —— 命中复用 + `in_ring` 防重复入队

**输入** `keys = [10, 25]`

**① 查找**

```
key 10:  hash = 2 → b2 链
            ┌────────┐        ┌────────┐
         →  │n1 key18│  ✗  →  │n0 key10│  ✓ 命中
            └────────┘        └────────┘
         pin:  CAS state(n0):  RESIDENT|ref=0  →  RESIDENT|ref=1|used=1

key 25:  hash = 1 → hash_heads[1] = -1  →  miss
```

> 注意：n0 的 `ref` 归零后一直躺在 ring 里，**数据始终有效**——命中它是完整的缓存命中，不产生任何 PCIe 流量。这是 ring 相对"归零即释放"的核心收益。

**② ring pop ×1**

```
        idx  0    1    2    3
           ┌────┬────┬────┬────┐
           │ n0 │ n1 │ n2 │ n3 │
           └────┴────┴────┴────┘
                            ↑ head=3  → pop 得 ring[3&3] = n3
           校验：ref(n3) == 0 ✓   state == EMPTY ✓   → claim 成功
           head: 3 → 4
```

**③ 分配与挂链** —— `key 25 → row 88`，`hash(25) = 1`

```
hash_heads[1]  →  ┌────────┐
                  │n3 key25│ ──→ nil
                  └────────┘
host[88] ──── PCIe H2D ────→ emb_table[3]     ref = 1
```

**④ 反向释放** —— 两个 node 都被更新，但入队行为不同：

```
n0:  ref 1 → 0
     atomicOr(state, IN_RING) 的返回值中 IN_RING 已置位
       →  ★ 不重复入队。它在 Step 1 放进 ring 的那条记录（idx0）仍然有效

n3:  ref 1 → 0
     atomicOr 返回值中 IN_RING 为 0  →  CAS 成功，入队
       push n3 → ring[7 & 3] = ring[3] = n3     tail: 7 → 8
```

★ 这正是 §7.5 所述：若此处无条件入队，ring 中会出现 n0 的**两条**记录，第一条被淘汰复用后，第二条即成为悬垂引用。

**Step 2 结束**

```
        idx  0    1    2    3
           ┌────┬────┬────┬────┐
           │ n0 │ n1 │ n2 │ n3 │        head = 4    tail = 8
           └────┴────┴────┴────┘
环的逻辑顺序：
   idx0  →  idx1  →  idx2  →  idx3
    n0       n1       n2       n3
  (key10)  (key18)  (key3)  (key25)
  队头 ←──────────────────────→ 队尾

四个 node 全部 RESIDENT，ref 全为 0，环满（tail - head = 4 = C，满足 INV-4）
```

---

### C.3 Step 3 —— 陈旧条目丢弃 + 淘汰 + 双向回写

**输入** `keys = [10, 41]`

**① 查找 + pin**

```
key 10:  b2 链 → [n1 key18] ✗ → [n0 key10] ✓ 命中
         pin:  ref 0 → 1，used = 1

key 41:  hash = 1 → b1 链 → [n3 key25] ✗ → nil  →  miss
```

> ★ 关键：n0 此刻 `ref = 1`（正在被本步使用），而它在 ring 的 **idx0 —— 正是 head 指向的位置**。它成了一条陈旧条目。

**② ring pop —— 第一次撞上陈旧条目**

```
        idx  0    1    2    3
           ┌────┬────┬────┬────┐
           │ n0 │ n1 │ n2 │ n3 │
           └────┴────┴────┴────┘
             ↑ head=4 → pop 得 ring[4&3] = ring[0] = n0

  校验 ref(n0) = 1  ≠ 0   →  ★ 陈旧条目
       clear in_ring(n0)，【丢弃该记录，不放回队尾】，head: 4 → 5
       （n0 会在它下次真正归零时重新入队，不会泄漏）

             ↑ head=5 → pop 得 ring[5&3] = ring[1] = n1  (key=18, ref=0)
  校验 ref(n1) = 0 ✓
       CAS(state: RESIDENT|ref=0 → LOADING) 成功   ← INV-3
       clear in_ring(n1)，head: 5 → 6
```

> 丢弃而非回收，是为了避免缓存被 pin 满时线程绕环活锁（§7.3）。

**③ 淘汰 n1：摘链趟（Step 4a）**

```
改前  hash_heads[2] ──→ ┌────────┐ ⇄ ┌────────┐
                        │n1 key18│   │n0 key10│ ──→ nil
                        └────────┘   └────────┘

摘除 n1（借助 prev/next 双向指针，O(1)，无需从桶头遍历）

改后  hash_heads[2] ──→ ┌────────┐
                        │n0 key10│ ──→ nil        n0.prev = -1
                        └────────┘

同趟内探测 backing 取受害者行号：key 18 → 行 205
（node 中不存行号，现场探测，因而对 backing 重哈希免疫，§4.1）
```

> 摘链与插链必须分属两趟。若同处一个 kernel，就会出现"甲正把 n1 从桶 2 摘出、乙正往桶 2 插入新 node"的并发双向链表修改（§6.1）。本例中 key 41 挂在桶 1，恰好不冲突，但一般情况下会。

**④ 双向搬运**（§9.2）—— `key 41 → find-or-insert → 行 300`

```
                        一个 warp 内完成
   ┌──────────────────────────────────────────────────────────┐
   │ ① emb_table[1] ─────────────→ 寄存器    device 读（旧行）│
   │                                                          │
   │ ② host[300]    ──── H2D ────→ 寄存器   ┐                 │
   │                                        │ PCIe 双向同时在飞 │
   │ ③ 寄存器(旧行) ──── D2H ────→ host[205]┘                 │
   │                                                          │
   │ ④ 寄存器(新行) ─────────────→ emb_table[1]  device 写    │
   └──────────────────────────────────────────────────────────┘

旧行在 ① 就已进入寄存器，④ 随时可覆写 —— D2H 不阻塞 H2D，不存在"等待写回"
```

**⑤ 挂新桶：插链趟（Step 4b）** —— `hash(41) = 1`

```
改前  hash_heads[1] ──→ ┌────────┐ ──→ nil
                        │n3 key25│
                        └────────┘

改后  hash_heads[1] ──→ ┌────────┐ ⇄ ┌────────┐
                        │n1 key41│   │n3 key25│ ──→ nil
                        └────────┘   └────────┘

n1: key=41, state=RESIDENT, ref=1, in_ring=0
```

**Step 3 结束**

```
nodes                                    ring
┌────┬─────┬──────────┬─────┬───────┐        idx  0    1    2    3
│ id │ key │  state   │ ref │in_ring│            ┌────┬────┬────┬────┐
├────┼─────┼──────────┼─────┼───────┤            │ n0 │ n1 │ n2 │ n3 │
│ n0 │ 10  │ RESIDENT │  0  │   1   │            └────┴────┴────┴────┘
│ n1 │ 41  │ RESIDENT │  0  │   1   │                   ↑head=6  tail=10
│ n2 │  3  │ RESIDENT │  0  │   1   │
│ n3 │ 25  │ RESIDENT │  0  │   1   │        key 18 已被淘汰，
└────┴─────┴──────────┴─────┴───────┘        其整行已回写至 host[205]
```

---

### C.4 一个 step 内的时序全貌

```
   ┌──── prefetch（只出队）─────┐   ┌── fwd/bwd ──┐   ┌─ release（只入队）─┐
   │ ① 去重                     │   │             │   │                    │
   │ ② 查找 + pin      ← 独立   │   │ ⑦ gather    │   │ ⑨ ref - 1          │
   │ ③ ring pop（含校验/丢弃）  │   │             │   │ ⑩ 若 1→0 入队尾    │
   │ ④a 摘链趟 + 探测受害者行号 │   │ ⑧ 梯度规约  │   │                    │
   │ ④b 插链趟         ← 独立   │   │    优化器   │   │                    │
   │ ⑤ 双向搬运（H2D + D2H）    │   │    原地更新 │   │                    │
   └────────────────────────────┘   └─────────────┘   └────────────────────┘

   ② 与 ④ 必须是不同 kernel（pin 身份竞态，§6.3）
   ④a 与 ④b 必须是不同 kernel（桶链并发修改，§6.1）
       写 ring_head                    只读 emb_table      写 ring_tail
       写 hash / nodes / emb_table      写 emb_table       写 state

   └──────────────── 由流序保证：出队与入队永不并发（§7.2）─────────────┘

   与 step N-1 的 dense 计算重叠 ↗
```

`ring_head` 只在 prefetch 阶段被写、`ring_tail` 只在 release 阶段被写——这正是 §7.2 那条约束的具体形态，也是出队 kernel 中 `tail` 可被当作常量快照的原因。

---

### C.5 不变量在本例中的落点

| 不变量 | 在哪一步体现 |
|---|---|
| INV-1 `ref > 0` 永不被淘汰 | C.3 ② —— n0 因 `ref = 1` 被跳过 |
| INV-2 仅在 `1→0` 且 `in_ring == 0` 时入队 | C.2 ④ —— n0 因 `in_ring` 已置位而不重复入队 |
| INV-3 弹出后 CAS 重新校验 | C.3 ② —— n1 经 `CAS(RESIDENT\|ref=0 → LOADING)` 才被占用 |
| INV-4 `tail - head ≤ C` | C.2 结束时取等（环满），全程未超 |
| INV-5 `head ≤ tail` | 本例未触发抽干；若 Step 3 中 n1..n3 也都被 pin，`ring_pop` 将返回 `-1` 并归还 head |
| INV-6 每 unique key 每步恰好 `+1 / -1` | key 10 在三个 step 中各 `+1/-1` 一次，`ref` 始终回到 0 |
| INV-7 替换策略对结果透明 | key 18 被淘汰后其整行（emb + optim state）已落 host[205]，再次访问时值不变 |
| INV-8 pin 后复查 key | C.2 ① 与 C.3 ① —— pin 成功后确认 `key_t[n0] == 10` |
| INV-9 `RESIDENT` 行永不旧于 backing | C.3 ④ —— n1 无条件回写 host[205]，无需判断是否"脏" |

**若启用 CLOCK**：C.3 ② 中 n0 会先被检查 `used` 位——但 `ref != 0` 的判定在前，仍按陈旧条目丢弃，结果相同。差异出现在另一种情形：若 n0 当时 `ref == 0` 且 `used == 1`，FIFO 会直接淘汰它，而 CLOCK 会清零 `used`、把它送进 `retry_buf` 放回队尾，转而淘汰下一个——key 10 得以留在缓存中。
