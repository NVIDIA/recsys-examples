# Admission Strategy — Design & Change Document

> Branch: `fix/dynamicemb-admission-init`, on top of `upstream/main` @ `97062d9`.
> Goal: stop a fused module from silently collapsing its tables' per-table
> configuration onto table 0, and give admission one coherent shape — one
> object that owns the decision, the state the decision needs, and how the rows
> it rejects get initialized.

---

## 1. Background

`BatchedDynamicEmbeddingTablesV2` fuses several logical tables into one module
with one value buffer. Which tables may be fused is decided by
`DynamicEmbTableOptions.get_grouped_key()`: tables whose keys are equal share a
module.

Everything per-table therefore arrives as a *list*, and something has to turn
that list into one object the module can run. One of those already existed:

| per-table configuration | one per module |
| --- | --- |
| `KVCounter` | `MultiTableKVCounter` |
| `DynamicEmbInitializerArgs` | *(nothing — table 0's was used for all)* |
| `AdmissionStrategy` | *(nothing — table 0's was used for all)* |

The two empty rows are this document.

## 2. What is wrong today

### 2.1 Every table initializes as table 0

`_create_initializers` built one initializer per table, but every consumer in
`batched_dynamicemb_function.py` indexed `initializers[0]` — all six call
sites, training and eval alike. Tables 1..N-1 were initialized with table 0's
distribution.

This is not a corner case. `get_grouped_key` deliberately left
`initializer_args` out, so tables with different initializers *are* fused; and
the planner resolves an unbounded `UNIFORM` to `±sqrt(1 / num_embeddings)`
(`complete_initializer_args`), which differs per table by construction. Two
tables of different row counts, configured identically, already get different
parameters — and the smaller one was then initialized with the larger one's.

The tests missed it because all four of their tables have
`num_embeddings=10000` and an explicit `CONSTANT` initializer, so the four
initializers are equivalent.

Eval initializers were collapsed the same way. `__post_init__` forces
`eval_initializer_args.mode` to `CONSTANT`, so their modes always agree and no
bound resolution is ever needed — but their *values* may differ, and table 0's
was used for all.

### 2.2 The same collapse in the fused counter

`MultiTableKVCounter` passes each table's own `capacity` to the fused table —
correct — but reads `bucket_capacity` and `key_type` from `kv_counters[0]`.
Those two describe the single physical table, so they genuinely have to agree;
what was missing is a rule saying so, not per-table support.

### 2.3 Admission is configured in two places that must agree

```python
DynamicEmbTableOptions(
    admit_strategy=FrequencyAdmissionStrategy(threshold=4),
    admission_counter=KVCounter(capacity=1 << 20),
)
```

Nothing tied these together. Set the first and forget the second and the model
built fine, then died on the first forward that missed, inside
`_apply_admission`, with `AttributeError: 'NoneType' object has no attribute
'add'` — the three admission paths called `admission_counter.add(...)`
unconditionally. And a counter's capacity is a property of *frequency-based
admission*, not of an embedding table: a probabilistic strategy needs no counter
at all yet would still have to be given one.

### 2.4 The strategy answers for tables it was not asked about

Each table carries its own `admit_strategy`; the module used
`self._dynamicemb_options[0].admit_strategy` and dropped the rest. This happens
to be correct — `get_grouped_key` puts `threshold` in the key, so a module's
strategies are interchangeable — but nothing stated the guarantee, and the same
`[0]` pattern is a real bug two rows up.

### 2.5 Who writes a rejected row was negotiated per batch

A rejected key still takes part in the forward, so its row must be written.
`initialize_non_admitted_embeddings` returned a bool meaning "I wrote them", and
the caller then covered whatever was left. That handshake was inverted, which is
the bug the first commit on this branch fixes: one branch wrote the rows twice,
the other never wrote them and let `storage.find`'s `torch.empty` buffer reach
the forward. Whether a strategy has an initializer is fixed at construction;
asking per call only created the opportunity to get it wrong.

### 2.6 Smaller defects found on the way

- `DynamicEmbInitializerArgs.__eq__` never compared `mode`, so
  `CONSTANT(0.0) == NORMAL(0.0, 1.0)`; both it and
  `DynamicEmbTableOptions.__eq__` returned the `NotImplementedError` *class*
  (truthy) instead of the `NotImplemented` singleton, making `args == 5` true.
- `FrequencyAdmissionStrategy` rebuilt its initializer on every forward, which
  for the random modes means a `CurandStateContext` — a device allocation plus
  a grid-wide init kernel — per step.
- The fallback bounds for an unbounded `UNIFORM` (`0.0`, `1.0`) were written out
  in two places.

## 3. Design

Three rules, applied uniformly.

**Per-table configuration becomes one multi-table object, never element 0.**
Turning N configurations into the one thing a module runs is a named step with
a stated precondition, not a subscript.

**Group on what must agree; resolve the rest per table.** A grouping key
carries only what tables cannot differ in, and each object answers for itself
what that is. An initializer answers with its mode, not its parameters — those
are looked up by table id in the kernel. A counter answers with its bucket
layout and key type, not its capacity.

**Put a thing where it belongs, not where it is used.** The counter is state
that only the admission decision needs, so it moves onto the strategy. The
initializer for rejected rows exists only because admission exists, so it lives
with the strategy too — but the *module* runs it, so buffer layout stays out of
the strategy's interface.

### 3.1 The admission interface

Two classes, because there are two things: what a table is configured with, and
what a module runs. One class was both, which left a configuration carrying an
`admit` that only raised, a `state` that was always None and an initializer that
was never there -- half an object, right only because nobody called that half.

```python
class AdmissionStrategy(abc.ABC):
    """How a table's admission is configured. Inert, shared, decides nothing."""

    @classmethod
    @abc.abstractmethod
    def create_admitter(cls, table_strategies, device) -> "MultiTableAdmitter":
        """The admitter these tables share, with its device state allocated.

        Called once, by the module, with one configuration per table it fuses.
        A classmethod so each configuration picks the admitter it needs, rather
        than a base class enumerating them.
        """

    @classmethod
    def one_configuration(cls, table_strategies) -> "AdmissionStrategy":
        """The single configuration these tables agree on.

        Grouping already made them interchangeable, so the first stands for
        all; this says so rather than take element zero and leave the reader to
        wonder.
        """

    def get_grouped_key(self):
        """What must match for two tables to share a module."""
        return id(self)


class MultiTableAdmitter(abc.ABC):
    """What a fused module runs, holding everything deciding takes."""

    @abc.abstractmethod
    def admit(self, keys, table_ids, frequencies=None) -> torch.Tensor:
        """Which of these missing keys may enter the table.

        ``frequencies`` is how often each key occurred in *this batch*, where
        the module counts occurrences at all; None means treat each key as one.
        An increment, not a running total -- any total is the admitter's own.
        """

    def state(self) -> Optional[Counter]:
        """Persistent state the framework has to carry, or None."""
        return None

    @property
    def non_admitted_initializer(self):
        """What writes the rows this admitter rejects; None for the table's."""
        return None
```

One abstract method each. Everything else has a default, so an admitter that
merely decides is one method, and a probabilistic one needs no framework change
at all: `state()` stays None and `admit` is one line.

`state()` exists because the counter is not private bookkeeping: it is device
memory the framework sizes, reports through `memory_usage`, and checkpoints.
`Counter` already is that protocol; it simply gains a second audience —
`add`/`erase` for the admitter, `memory_usage`/`dump`/`load` for the module.

### 3.2 Configuration after the change

```python
FrequencyAdmissionStrategy(
    threshold=4,
    counter=KVCounter(capacity=1 << 20),   # was DynamicEmbTableOptions.admission_counter
    initializer_args=None,                 # None: rejected rows use the table's initializer
)
```

`DynamicEmbTableOptions.admission_counter` is deprecated (§5). Admission is
configured in one place, so it cannot be configured inconsistently.

Each configuration has an admitter of its own:
`FrequencyAdmissionStrategy` builds a `MultiTableFrequencyAdmitter`, holding the
fused counter, and `ProbabilisticAdmissionStrategy` a
`MultiTableProbabilisticAdmitter`, holding nothing. Four classes where a single
one would do less, and the reason to prefer them is that none of the eight
members between them is dead.

### 3.3 Lifecycle

```
caller writes configuration          inert: no device memory, shareable across tables
        |
planner                              resolves table fields only; never writes into
        |                            a strategy
module construction
        |-- MultiTableInitializer.create(...)          train, from the tables' args
        |-- MultiTableInitializer.create(...)          eval, likewise
        `-- type(s[0]).create_admitter(...)            admission
                 |-- MultiTableKVCounter               from each strategy's KVCounter
                 `-- MultiTableInitializer.create      for rejected rows
forward                              admit() decides and keeps its own books
```

The planner touching only table fields is what keeps a configuration object
inert, and is why nothing along this path has to copy one. The one `copy` that
remains is in `DynamicEmbTableOptions.__post_init__`, and belongs to the
deprecated field alone (§5).

The cost is that an unbounded `UNIFORM` on a *strategy's* initializer cannot be
resolved against `num_embeddings`: only the planner still knows it, since by
module construction `max_capacity` has become the per-rank aligned row count,
and the strategy belongs to no one table. That case warns and falls back to
`DEFAULT_UNIFORM_LOWER` / `DEFAULT_UNIFORM_UPPER`, pointing at the two
configurations that do work — give explicit bounds, or omit `initializer_args`
and inherit the table's initializer, which *is* resolved per table.

Those two constants live in `types.py` and are now the only definition of that
fallback; `complete_initializer_args` and the initializer both read them.

### 3.4 Per-table parameters in the kernel

`MultiTableInitializer.create` decides once, at construction, whether its
tables agree. They usually do, and then each subclass calls its mode's plain
kernel with the parameters as scalars — the same call the code made before this
change. When they do not, it calls that mode's `_table_params` kernel with a
`[num_tables, num_params]` float32 tensor and the table each buffer row belongs
to. Both write the same multi-table buffer; only where the parameters come from
differs, which is why the second form is named for the parameters and not
`_multi_table`.

The two are **separate entry points, hence separate kernels**, so the common
path carries nothing of the other's. In CUDA that is one templated holder:

```cuda
template <bool kPerTable, int kNumParams> struct InitParams;

template <int N> struct InitParams<false, N> {
  float values[N];
  DEVICE_INLINE float get(int64_t, int slot) const { return values[slot]; }
};

template <int N> struct InitParams<true, N> {
  const float *table_params; // [num_tables, N], row-major
  const int64_t *table_ids;  // buffer row -> table
  DEVICE_INLINE float get(int64_t vec_id, int slot) const {
    return table_params[table_ids[vec_id] * N + slot];
  }
};
```

Each generator is `template <bool kPerTable> struct XxxEmbeddingGenerator` and
reads `params_.get(vec_id, slot)`. The shared instantiation keeps its
parameters in registers, reads no memory for them, and carries neither pointer;
there is no runtime branch in either.

`keys` and `table_ids` run alongside the value buffer, one entry per row of
it, and `indices` picks out the rows to write -- which is why it comes last in
an initializer's signature, and why the launchers check the other two against
the buffer's row count rather than trust the convention to be remembered.
Per mode: `UNIFORM`/`NORMAL` 2 parameters, `TRUNCATED_NORMAL` 4, `CONSTANT` 1.
`DEBUG` derives the value from the key, so it takes no parameters, its tables
cannot disagree, and it has no per-table form.

A lookup happens per element rather than per row, since `generate()` is called
once per element. Consecutive threads cover one row and read one table id, so it
broadcasts out of L1; hoisting it would mean restructuring the kernel loop and
is not worth it here.

### 3.5 Grouping

Every object that takes part answers for itself, through one helper:

```python
def group_key_of(obj):
    """Ask an object what has to match for its tables to be fused."""
    if obj is None:
        return None
    get_grouped_key = getattr(obj, "get_grouped_key", None)
    return get_grouped_key() if callable(get_grouped_key) else obj
```

```python
# DynamicEmbTableOptions.get_grouped_key
grouped_key["initializer"]      = group_key_of(self.initializer_args)
grouped_key["eval_initializer"] = group_key_of(self.eval_initializer_args)
grouped_key["admit_strategy"]   = group_key_of(self.admit_strategy)
```

| object | answers with | leaves out, because it is resolved per table |
| --- | --- | --- |
| `DynamicEmbInitializerArgs` | its mode | its parameters |
| `KVCounter` | type, `bucket_capacity`, `key_type` | its `capacity` |
| `FrequencyAdmissionStrategy` | type, `threshold`, and the two above | — |
| anything else | itself | — |

Falling back to the object itself keeps today's behavior (identity) for an
implementation that does not answer. `group_key_of` lives in `types.py`, the
lowest layer, because `dynamicemb_config` and `embedding_admission` both need
it.

`MultiTableInitializer.create` and `AdmissionStrategy.one_configuration` both
validate their inputs by *this* key rather than by hand, so what they check
cannot drift from what actually decided the tables may be fused.

One consequence worth stating: tables configured with *equal but separately
constructed* strategies group together now, where identity comparison kept them
apart. That is the common case, and it changes sharding plans. Checkpoints are
unaffected — `dump`/`load` address tables by name, not by module.

### 3.6 The initializer classes

`MultiTableInitializer` is the base class, with `create` as its factory; the
five modes are its subclasses. Each subclass writes down its own two calls and
nothing else:

```python
class UniformInitializer(MultiTableInitializer):
    def __init__(self, args, table_params=None):
        super().__init__(args, table_params)
        self._curand_state = CurandStateContext()

    @staticmethod
    def table_param_rows(args_list):
        return [[args.lower, args.upper] for args in args_list]

    def __call__(self, buffer, keys, table_ids, indices):
        if self._table_params is None:
            uniform_init(buffer, indices, self._curand_state,
                         self._args.lower, self._args.upper)
        else:
            uniform_init_table_params(buffer, indices, self._curand_state,
                                      self._table_params,
                                      self._table_ids_for_kernel(table_ids))
```

`table_param_rows` is where the column order is written down, four
lines from the scalar call that has to agree with it and with the slot indices
the matching generator reads. That adjacency is the whole point: before this
change the order existed in one place only, and adding a second far from the
first is what would make it rot.

`DebugInitializer` overrides `__call__` outright — it is the only mode that
reads `keys`, and doing so keeps that argument out of every other mode.

### 3.7 Probabilistic admission, the first strategy built on this shape

`ProbabilisticAdmissionStrategy` admits a missing key with a fixed chance, and
needed no framework change at all: `state()` stays None so no counter is ever
built for it, `create_admitter` only has the rejected-row initializer to open,
and `admit` is a comparison.

```python
draws = torch.rand(num_keys, device=keys.device)
return draws < self.probability
```

Admission is consulted only for a key that is *missing*, so a key gets a fresh
toss every time it turns up and is not yet in the table: it takes `1 / p`
appearances on average to get in. That is frequency filtering with no counter
and no state. Deciding by `hash(key)` instead would be reproducible and
rank-stable, but it would settle each key's fate forever — at `p = 0.1`, nine
keys in ten could never get in however hot they are. Rank stability buys
nothing here anyway: row-wise sharding already gives a key to exactly one rank.

**Repeats within a batch.** A batch holding a key `k` times deserves `k` tosses.
Tossing `k` times and taking any success is exactly one toss against
`1 - (1 - p)^k`, so that is what it compares to — one draw per key, no ragged
loop. `frequencies` is where `k` comes from, and is 1 when the module is not
counting.

Three details that are easy to get backwards:

- **`<`, not `<=`.** `torch.rand` draws from `[0, 1)`, so a strict comparison
  admits nothing at `p = 0` and everything at `p = 1`. The initializers'
  `curand_uniform` is `(0, 1]` and wants the opposite; the two conventions sit
  in one codebase and should not be "unified".
- **`log1p(-1)` has no value**, so `p = 0` and `p = 1` short-circuit before any
  arithmetic.
- **`1 - p` loses a small `p` outright in float32** — at `p = 1e-8` it rounds to
  1, and `1 - (1 - p)^k` becomes 0, admitting nothing ever. The threshold is
  computed as `-expm1(k · log1p(-p))`, which is exact at both ends of the range.

## 4. Changes by area

All of the following is implemented in the working tree. **Nothing has been
compiled or run**: there is no CUDA on the machine the work was done on.

**`src/initializer.cuh`** — `InitParams<kPerTable, kNumParams>` and its two
specializations; the uniform, normal, truncated-normal and constant generators
templated on `kPerTable` and reading `params_.get(vec_id, slot)`;
`MappingEmbeddingGenerator` (DEBUG) unchanged.

**`src/initializer.cu`** — `check_table_params` validating dtype, shape and
contiguity; two launchers per parameterized mode; nine bindings, of which the
five pre-existing ones keep their names and signatures unchanged, the new ones
being `<mode>_init_table_params`.

**`initializer.py`** — `MultiTableInitializer` as base plus `create`; five
subclasses as in §3.6; `_with_default_bounds` returning a copy rather than
filling the caller's args in place. `BaseDynamicEmbInitializer` and
`create_initializer_from_args` are gone — they had no users outside this file.

**`types.py`** — `AdmissionStrategy` and `MultiTableAdmitter` as in §3.1;
`group_key_of`;
`DynamicEmbInitializerArgs.get_grouped_key`; `DEFAULT_UNIFORM_LOWER` /
`DEFAULT_UNIFORM_UPPER`; the `__eq__` fix (already committed).

**`embedding_admission.py`** — `KVCounter.get_grouped_key`;
`FrequencyAdmissionStrategy` taking its own counter, implementing
`create_admitter` and `get_grouped_key`, warning on an unbounded `UNIFORM`;
and their admitters `MultiTableFrequencyAdmitter` and
`MultiTableProbabilisticAdmitter` implementing `admit`, `state` and
`non_admitted_initializer`. The frequency admitter does the counter's `add`
and `erase` itself.

**`dynamicemb_config.py`** — grouping as in §3.5; `complete_initializer_args`
reading the shared fallback constants; the `admission_counter` deprecation and
its fold in `__post_init__` (§5).

**`planner.py`** — unchanged but for a docstring. It resolves table fields and
nothing else: `eval_initializer_args` needs no resolution, being `CONSTANT` by
construction, and a strategy is never touched.

**`batched_dynamicemb_tables.py`** — `_create_initializers` builds the train
and eval initializers; `_create_admitter` builds the admitter and takes
`_admission_counter` from its `state()`, which the dump/load paths still read.

**`batched_dynamicemb_function.py`** — six call sites pass `table_ids`; the
three admission paths lose their counter bookkeeping to `admit()`; rejected
rows are written by `strategy.non_admitted_initializer or initializer`, one
call, no branch on a return value; `admission_counter` is gone from every
signature.

**`key_value_table.py`** — the two eval lookup sites pass `table_ids`.

**`example.py`, `test_batched_dynamic_embedding_tables_v2.py`,
`test_embedding_dump_load.py`, `test_embedding_admission.py`,
`DynamicEmb_APIs.md`** — construct the counter on the strategy.

## 5. Public API changes

| before | after | migration |
| --- | --- | --- |
| `DynamicEmbTableOptions.admission_counter=KVCounter(...)` | `FrequencyAdmissionStrategy(counter=KVCounter(...))` | still works, warns |
| `AdmissionStrategy.initialize_non_admitted_embeddings` | `MultiTableAdmitter.non_admitted_initializer` | removed |
| `AdmissionStrategy.admit` | `MultiTableAdmitter.admit` | a configuration no longer decides; `create_admitter` builds what does |
| `admit(keys, frequencies)` | `admit(keys, table_ids, frequencies=None)` | `frequencies` is now a per-batch increment, not a running total |

`admission_counter` keeps working: `__post_init__` warns, then folds it into a
*copy* of the strategy. A copy because one strategy is commonly handed to every
table and each may have sized its counter differently, which folding in place
would flatten to whichever table was configured last; and only into a
`FrequencyAdmissionStrategy`, since another implementation's `counter` would be
that author's own thing. The whole shim is one `if` block, to be deleted with
the field.

`initialize_non_admitted_embeddings` is gone outright. It is an abstract
method, so an implementation that still has one keeps importing and simply
stops being called — a silent change, and the reason to mention it in release
notes rather than rely on an error.

## 6. Relationship to the four commits already on the branch

They stand. Commits 1 and 2 fix the initialization bug as it exists today, and
this design removes the shape that allowed it. Commit 3's reject-all test keeps
working — it asserts what a rejected key's lookup returns, which is exactly the
behavior being restructured, though it needs the constructor change of §3.2.
Commit 4 is a prerequisite: `create` compares resolved parameters, and
`DynamicEmbTableOptions.__eq__` compares grouping keys.

## 7. Tests

1. **Heterogeneous initializers.** One module, table A `CONSTANT(1.0)`, table B
   `CONSTANT(2.0)`; check per table after a forward. Fails today: B reads 1.0.
2. **Default UNIFORM bounds.** Two tables an order of magnitude apart in row
   count, no explicit initializer; assert each stays within its own
   `±sqrt(1 / num_embeddings)`.
3. **Heterogeneous eval initializers.** Same, through the eval path, where the
   mode is fixed at `CONSTANT` but the values differ.
4. **Rejected rows.** Extend the reject-all matrix from commit 3 to a strategy
   whose tables were given different initializer parameters.
5. **No counter needed.** A strategy whose `state()` is None runs without any
   counter being built — the case that raises `AttributeError` today.
6. **Grouping.** Equal-but-distinct strategies fuse; different thresholds do
   not; different initializer *parameters* fuse, different *modes* do not.
7. **Deprecation.** `admission_counter` still admits the same keys, and warns.
8. **Equality.** `CONSTANT(0.0) != NORMAL(0.0, 1.0)`; `args == 5` is False.
9. Existing admission, dump/load and incremental-dump suites, for regression.

1-3, 6 and 8 are new files; 4 and 7 extend `test_embedding_admission.sh`.

`test/unit_tests/admission/test_probabilistic_admission.py` covers §3.7 and is
written:
the rate, both ends of the range, repeats compounding to `1 - (1 - p)^k`, a
seeded run repeating, and a probability small enough that computing
`(1 - p)^k` directly would admit nothing — which separates the two forms
without needing a large sample.

## 8. Commit plan

Continuing from the four already on the branch:

5. `fix(dynamicemb): return NotImplemented from DynamicEmbTableOptions comparisons`
6. `refactor(dynamicemb): let each object say what fusing its tables requires`
7. `feat(dynamicemb): give each mode a per-table initializer kernel`
8. `feat(dynamicemb): give a fused module one initializer over all its tables`
9. `refactor(dynamicemb): let an admission strategy own its counter and its initializer`
10. `test(dynamicemb): cover per-table initializers and counter-free admission`

7 and 8 split the mechanism from the behavior change; 9 is the API change and
carries the deprecation.

## 9. Out of scope

- **Per-table thresholds.** `admit` receives `table_ids`, so a strategy could
  hold a threshold per table and drop `threshold` from its grouping key. That is
  a feature, not a fix.
- **`initializers[0]` elsewhere.** Only the initializer family is addressed here.
- **`evict_strategy` taking element 0** — it is in the grouping key, so a
  module's tables agree on it by construction, and one physical table has one
  score layout either way.
