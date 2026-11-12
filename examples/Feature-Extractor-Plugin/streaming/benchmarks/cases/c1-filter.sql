-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0
-- Case c1-filter. Table `events`: seq, user_id, video_id, time_ms, is_click, is_like, is_follow, is_comment, is_forward,
-- is_hate, long_view, play_time_ms, duration_ms, is_rand, tab (all BIGINT NOT NULL), ts = event time (time_ms, TIMESTAMP_LTZ(3)),
-- proc_time = PROCTIME(). The driver defines it; see REPRODUCE.md.
SELECT seq, user_id, video_id, time_ms,
       CAST(play_time_ms AS DOUBLE) / CAST(duration_ms + 1 AS DOUBLE) AS watch_ratio,
       CASE WHEN duration_ms < 15000 THEN 0 WHEN duration_ms < 60000 THEN 1 ELSE 2 END AS duration_bucket,
       is_like + is_follow + is_comment + is_forward AS interact_cnt,
       is_click
FROM events
WHERE duration_ms > 0 AND play_time_ms >= 0 AND is_rand = 0
  AND (play_time_ms > 3000 OR is_click = 1)
