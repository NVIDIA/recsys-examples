# The `score` package

Everything about a table's *score* -- the number that ranks its keys for
eviction -- lives in `dynamicemb/score/`. This page says what each module holds
and how a score travels from the user's input to the hash table and into a
checkpoint.

| Module | Holds |
|---|---|
| `strategy.py` | `DynamicEmbScoreStrategy`, the compound form, validation, the physical column order and the dump/load permutations |
| `spec.py` | `ScoreSpec` (the physical score of a table), `score_policy_num_scores`, `get_score_policy` |
| `args.py` | `ScoreArg` (the score one call is given), `score_source`, `get_find_score_arg`, `get_insert_score_arg` |
| `checkpoint_columns.py` | the score columns of a dump and of a replay: ages instead of timestamps, logical column order |
| `kjt.py` | `ScoredKeyedJaggedTensor`, the input that carries one score per key |
| `jit.py` | the JIT glue of the `(TIMESTAMP, LFU)` eviction kernel: it links a user's score function into the packaged fatbins |
| `fatbin/` | the eviction fatbins `setup.py` builds from `src/jit/` (generated, not in git) |

The old import paths (`dynamicemb_config`, `key_value_table`, `scored_hashtable`)
still re-export the names that moved, so existing code keeps working.

## Strategies (`strategy.py`)

A table's `score_strategy` is a single `DynamicEmbScoreStrategy` or a compound
tuple:

| Strategy | The score of a key |
|---|---|
| `TIMESTAMP` | the device timer at the batch that touched it (LRU) |
| `STEP` | the table's step counter |
| `CUSTOMIZED` | what the user set with `set_score`, one value per table per batch |
| `PRECISE_CUSTOMIZED` | what the input carries for the key (`ScoredKeyedJaggedTensor`) |
| `LFU` | the count of how often the key was seen |
| `NO_EVICTION` | none -- the table grows instead of evicting |

Only these compounds are supported, in either order: `(TIMESTAMP, LFU)`,
`(NO_EVICTION, CUSTOMIZED)` and `(NO_EVICTION, PRECISE_CUSTOMIZED)`.
`normalize_score_strategy` validates a value and returns it in canonical form.

Two orders exist for a compound, and they are not the same:

* the **logical** order is the tuple the user wrote; checkpoints are written in
  it;
* the **physical** order is how the words sit on the device
  (`get_physical_score_order`), fixed whatever the user wrote: `(TIMESTAMP, LFU)`
  is always word 0 = last access, word 1 = frequency.

`score_dump_permutation` and `score_load_permutation` convert between them.

## The physical score (`spec.py`)

`get_score_policy` maps a strategy to a `ScoreSpec`: its name and the kernel's
`ScorePolicy` (`GLOBAL_TIMER`, `ASSIGN`, `ACCUMULATE`, `LRU_LFU`, or `FLAG` for
`NO_EVICTION` alone, whose single word marks a slot as in use). A table's
`score_policy` is this.

## What a call is given (`args.py`)

A key can come with two inputs: a **frequency** (how often the batch counted it)
and a **score** (what the input assigned it). `score_source(evict_strategy,
frequency, key_scores)` is the one place that says which of them a table takes:
LFU takes the frequency, `CUSTOMIZED` the score, any other strategy none. The
caller chooses once and hands the tables below it that single tensor, called
`score_input`; a table or an external store never sees the other.

`get_find_score_arg` and `get_insert_score_arg` turn `score_input` into a
`ScoreArg`, the policy and the values of one kernel call:

* in evaluation, or for a read-only lookup, a find uses `CONST`: no score is
  touched;
* in training, an LFU find `ACCUMULATE`s the frequency (one per key if none is
  given), a `CUSTOMIZED` or `STEP` find `ASSIGN`s;
* an insert sets the score of a new key: a `TIMESTAMP` table takes the timer, any
  other `ASSIGN`s (an `ACCUMULATE` table assigns on insert); re-inserting a
  key that already exists, such as a backward update, uses `CONST` so its score
  is not overwritten.

## Score columns in a checkpoint (`checkpoint_columns.py`)

A dump or a delta carries the scores in the logical column order. A device
timestamp is meaningless off the device that wrote it, so timestamp columns are
dumped as an **age** (`now - score`) and turned back into a timestamp against
the target's clock on load or replay (`_scores_to_age`, `_age_to_scores`).
`_dump_score_block` and `_replay_score_block` do both steps and check that a
block is as wide as the table's score layout; `_fresh_score_block` makes the
scores of keys replayed without `ReplayContent.SCORE`.

## Per-key scores in the input (`kjt.py`)

`ScoredKeyedJaggedTensor` carries one int64 score per key beside `weights`, so a
collection can still use `weights` for pooling weights or frequency counters.
The `KeyedJaggedTensor` methods a batch goes through on its way to a module
(`to`, `split`, `permute`, ...) return the subclass again with the scores moved
the same way as the values. `get_scores` reads them and `without_scores` drops
them.

## Dependencies

`strategy.py` depends on the standard library only. `spec.py`, `args.py` and
`checkpoint_columns.py` depend on `strategy.py`, `dynamicemb.types` and the
extension; they name `DynamicEmbTableState` for typing only, so the package
never imports `key_value_table`. The tables, the lookup paths and the
checkpoint code import `score`, not the other way round.
