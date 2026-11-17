# Receiver resolution: first HBM slice

Native row-wise sequence training resolves HBM keys after receiver completion and hands prepared rows to the standalone TorchRec forward adapter. Hash slots index pins; physical rows index values/moments; compact lookup rows are distinct. DynamicEmb retains its optimizer.

Raw calls, evaluation, cache/Host/Hybrid/Storage, pooled/processor, TW-RW and hierarchical paths retain existing lookup behavior in this slice. Explicit resolve_input is a training-only HBM capability; cancellation releases the returned state once. Retained graphs retain pins, including rehash/stream ordering through the lifetime prerequisite.

Subsequent dependent slices enable Cache, coherent Storage snapshots, other layouts and hierarchical ownership. No new global flag or temporary public configuration is needed. Upstream #557 supplies atomic pins; #558 is superseded and is not a new prerequisite. this diff must be reviewed against the declared prerequisite base. True NVLink/GIN and network PS validation remain separate.

## Cache extension

Pinned Cache and cache-over-Storage sequence receivers now use prepared rows too. Snapshot-backed Host/Hybrid/PS and additional layouts still keep their existing receiver behavior. Cache replacement initialization, overlap and reverse backward are tested separately.

## Storage extension

Host/Hybrid/Storage sequence receivers now resolve complete live snapshots. Writes publish embedding and optimizer bytes to overlapping batches by complete table/key identity and preserve existing scores. Local Storage-interface tests do not validate a network PS or concurrent remote writers. Evaluation is resolved read-only; raw/FIFO preparation also receives coherent snapshots.
