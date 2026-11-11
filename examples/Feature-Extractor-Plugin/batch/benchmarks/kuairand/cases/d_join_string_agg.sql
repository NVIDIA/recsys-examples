-- Case D (counter-example): join user/video tables, string features, per-group aggregation.
-- Expected: little or nothing for the plugin to fuse.
SELECT
  u.user_active_degree,
  v.video_type,
  substring(v.tag, 1, 2)                                      AS tag_prefix,
  count(*)                                                    AS exposures,
  avg(l.is_click)                                             AS ctr,
  avg(l.long_view)                                            AS long_view_rate
FROM log l
JOIN user u        ON l.user_id = u.user_id
JOIN video_basic v ON l.video_id = v.video_id
GROUP BY u.user_active_degree, v.video_type, substring(v.tag, 1, 2)
