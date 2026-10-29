# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from dynamicemb.exportable_embedding.indexer import FusedIdentityIndexer


@pytest.mark.parametrize("state_sidecar", [False, True])
def test_fused_identity_invalid_ids_cannot_alias_another_table(state_sidecar):
    device = torch.device("cuda", 0)
    indexer = FusedIdentityIndexer([2, 2], state_sidecar=state_sidecar, device=device)
    # Table 0 key 3 previously aliased table 1 key 0 (storage row 4).
    keys = torch.tensor([0, 1, -1, 2, 3, 0, 1, -3, 2], device=device)
    tables = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1], device=device)
    expected = torch.tensor([1, 2, 0, 0, 0, 4, 5, 3, 3], device=device)
    torch.testing.assert_close(indexer(keys, tables), expected)
