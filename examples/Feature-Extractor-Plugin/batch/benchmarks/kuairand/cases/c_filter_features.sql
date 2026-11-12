-- Case C: sample filtering + features (drop invalid / too-short exposures, keep feed traffic).
SELECT
  user_id,
  video_id,
  least(play_time_ms / (duration_ms + 1D), 5D)                AS watch_ratio,
  log(1D + play_time_ms / 1000D)                              AS log_play_sec,
  CASE WHEN duration_ms < 15000 THEN 0
       WHEN duration_ms < 60000 THEN 1 ELSE 2 END             AS duration_bucket,
  is_like + is_follow + is_comment + is_forward               AS interact_cnt,
  is_click
FROM log
WHERE duration_ms > 0
  AND play_time_ms >= 0
  AND is_rand = 0
  AND (play_time_ms > 3000 OR is_click = 1)
