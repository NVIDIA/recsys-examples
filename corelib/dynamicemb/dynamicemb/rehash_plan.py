# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host arithmetic of a rehash that moves values (no torch, so that it can be tested
without a GPU).

A key hashes to the bucket ``(hash % table_slots) // bucket_capacity`` of its table (the
mapping of ``insert_body`` and ``find_body`` in ``src/table_operation/kernels.cuh``). When
a table grows to a whole multiple ``k`` of its size, ``hash % (k * old_slots)`` is
``hash % old_slots`` plus ``j * old_slots`` for some ``0 <= j < k``, so a key of old bucket
``b`` lands in bucket ``b + j * old_buckets`` of the new table: the bucket it came from
(``j == 0``) or one of the buckets that the growth added after the old table's last one.
"""


def chunks_are_independent(old_slots: int, new_slots: int) -> bool:
    """Whether the values of a table that keeps each value at the slot of its key can be
    moved one chunk of the rehash at a time.

    Every value of a bucket of the old table goes to the rows of that same bucket (which
    the chunk has just read) or to rows past the old table, which hold nothing; so a store
    of one chunk never lands on a row that a later chunk has yet to read. That holds when the
    new table is a whole multiple of the old one, which is every growth that doubles. A
    new size that is not one (the last growth of a table, which ``max_capacity`` clamps)
    sends the keys of a bucket to unrelated buckets, and every value has to be read
    before any is written.
    """
    return old_slots > 0 and new_slots % old_slots == 0


def lands_in_the_bucket_of_origin_or_a_new_one(
    src_slot: int, dst_slot: int, bucket_capacity: int, old_buckets: int
) -> bool:
    """The invariant :func:`chunks_are_independent` stands on, for one key: its new
    bucket is its old bucket plus a whole number of old tables. The device asserts it for
    every lane of every chunk of such a move."""
    moved = dst_slot // bucket_capacity - src_slot // bucket_capacity
    return moved % old_buckets == 0
