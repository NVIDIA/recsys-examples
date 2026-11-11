-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0
-- Case c6-lookup. Table `events`: seq, user_id, video_id, time_ms, is_click, is_like, is_follow, is_comment, is_forward,
-- is_hate, long_view, play_time_ms, duration_ms, is_rand, tab (all BIGINT NOT NULL), ts = event time (time_ms, TIMESTAMP_LTZ(3)),
-- proc_time = PROCTIME(). The driver defines it; see REPRODUCE.md.
-- Needs the temporary table video_dim (video_id BIGINT, author_id BIGINT, PRIMARY KEY (video_id) NOT ENFORCED), a custom in-memory
-- connector defined by driver/src/main/java/fep/FepVideoDimFactory.java; only the videos present in the replayed slice are loaded.
SELECT e.seq, e.video_id, d.author_id
FROM events AS e
JOIN video_dim FOR SYSTEM_TIME AS OF e.proc_time AS d
  ON e.video_id = d.video_id
