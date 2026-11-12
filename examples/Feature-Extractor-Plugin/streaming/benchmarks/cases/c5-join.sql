-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0
-- Case c5-join. Table `events`: seq, user_id, video_id, time_ms, is_click, is_like, is_follow, is_comment, is_forward,
-- is_hate, long_view, play_time_ms, duration_ms, is_rand, tab (all BIGINT NOT NULL), ts = event time (time_ms, TIMESTAMP_LTZ(3)),
-- proc_time = PROCTIME(). The driver defines it; see REPRODUCE.md.
-- Interval join of the impression stream and the click stream (is_click = 1), key = user_id * 10^8 + video_id.
CREATE TEMPORARY VIEW imp AS
  SELECT seq, user_id * 100000000 + video_id AS uv, user_id, duration_ms, ts FROM events;
CREATE TEMPORARY VIEW clk AS
  SELECT seq, user_id * 100000000 + video_id AS uv, play_time_ms, ts FROM events WHERE is_click = 1;
SELECT i.seq, c.seq, i.uv, c.play_time_ms, i.duration_ms
FROM imp i JOIN clk c ON i.uv = c.uv AND c.ts BETWEEN i.ts AND i.ts + INTERVAL '30' MINUTE
