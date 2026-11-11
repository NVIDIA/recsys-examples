-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0
-- Case c3-topn-rn. Table `events`: seq, user_id, video_id, time_ms, is_click, is_like, is_follow, is_comment, is_forward,
-- is_hate, long_view, play_time_ms, duration_ms, is_rand, tab (all BIGINT NOT NULL), ts = event time (time_ms, TIMESTAMP_LTZ(3)),
-- proc_time = PROCTIME(). The driver defines it; see REPRODUCE.md.
-- Variant of c3-topn that also outputs the rank.
SELECT seq, user_id, video_id, play_time_ms, rn FROM (
  SELECT seq, user_id, video_id, play_time_ms,
         ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY play_time_ms DESC, seq ASC) AS rn
  FROM events) WHERE rn <= 10
