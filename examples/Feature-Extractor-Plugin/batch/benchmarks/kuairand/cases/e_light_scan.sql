-- Case E (counter-example): read the log and do almost nothing (one conversion + one division).
-- Expected: the plugin may fuse it, but the Parquet scan dominates, so no gain.
SELECT
  user_id,
  video_id,
  is_click,
  play_time_ms / 1000D                                        AS play_sec
FROM log
