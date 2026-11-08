-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0
-- Case c2-tumble. Table `events`: seq, user_id, video_id, time_ms, is_click, is_like, is_follow, is_comment, is_forward,
-- is_hate, long_view, play_time_ms, duration_ms, is_rand, tab (all BIGINT NOT NULL), ts = event time (time_ms, TIMESTAMP_LTZ(3)),
-- proc_time = PROCTIME(). The driver defines it; see REPRODUCE.md.
SELECT user_id, window_start, window_end, COUNT(*) AS expo_cnt,
       SUM(is_click) AS click_cnt, SUM(play_time_ms) AS play_ms
FROM TABLE(TUMBLE(TABLE events, DESCRIPTOR(ts), INTERVAL '5' MINUTE))
GROUP BY user_id, window_start, window_end
