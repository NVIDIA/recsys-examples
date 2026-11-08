-- Case A: per-exposure sample features on the KuaiRand log (numeric, row-wise).
-- Sources of each feature are listed in ../README.md.
SELECT
  user_id,
  video_id,
  CAST(hourmin / 100 AS INT)                                  AS hour_of_day,
  dayofweek(CAST(time_ms DIV 1000 AS TIMESTAMP))              AS day_of_week,
  least(play_time_ms / (duration_ms + 1D), 5D)                AS watch_ratio,
  log(1D + play_time_ms / 1000D)                              AS log_play_sec,
  CASE WHEN duration_ms < 7000   THEN 0
       WHEN duration_ms < 15000  THEN 1
       WHEN duration_ms < 30000  THEN 2
       WHEN duration_ms < 60000  THEN 3
       WHEN duration_ms < 120000 THEN 4 ELSE 5 END            AS duration_bucket,
  CASE WHEN play_time_ms >= duration_ms THEN 1 ELSE 0 END     AS complete_play,
  is_like + is_follow + is_comment + is_forward               AS interact_cnt,
  is_click,
  long_view,
  pmod(user_id * 1000003 + tab, 1048576)                      AS user_tab_cross
FROM log
