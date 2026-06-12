# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Lightweight filesystem abstraction for DynamicEmb checkpoint I/O.

Supports local POSIX filesystems natively and HDFS via the optional
``fsspec[arrow]`` package.  The factory :func:`get_filesystem` selects
the backend based on the path prefix (``hdfs://`` → Hdfs, otherwise
local), so no public API signatures need to change.
"""

from __future__ import annotations

import abc
import glob as _glob
import os
from typing import IO, Any, List


class AbstractFileSystem(abc.ABC):
    """Minimal filesystem interface covering the operations used by
    DynamicEmb dump / load.
    """

    @abc.abstractmethod
    def open(self, path: str, mode: str = "rb") -> IO[Any]:
        ...

    @abc.abstractmethod
    def exists(self, path: str) -> bool:
        ...

    @abc.abstractmethod
    def isdir(self, path: str) -> bool:
        ...

    @abc.abstractmethod
    def makedirs(self, path: str, exist_ok: bool = True) -> None:
        ...

    @abc.abstractmethod
    def ls(self, path: str) -> List[str]:
        ...

    @abc.abstractmethod
    def glob(self, pattern: str) -> List[str]:
        ...

    @abc.abstractmethod
    def size(self, path: str) -> int:
        ...


# ---------------------------------------------------------------------------
# Local filesystem (always available, zero extra dependencies)
# ---------------------------------------------------------------------------


class LocalFileSystem(AbstractFileSystem):
    """Thin wrapper around Python's built-in file I/O and ``os.path``."""

    def open(self, path: str, mode: str = "rb") -> IO[Any]:
        return open(path, mode)

    def exists(self, path: str) -> bool:
        return os.path.exists(path)

    def isdir(self, path: str) -> bool:
        return os.path.isdir(path)

    def makedirs(self, path: str, exist_ok: bool = True) -> None:
        os.makedirs(path, exist_ok=exist_ok)

    def ls(self, path: str) -> List[str]:
        return os.listdir(path)

    def glob(self, pattern: str) -> List[str]:
        return _glob.glob(pattern)

    def size(self, path: str) -> int:
        return os.path.getsize(path)


# ---------------------------------------------------------------------------
# HDFS filesystem (requires ``fsspec[arrow]``)
# ---------------------------------------------------------------------------


class HdfsFileSystem(AbstractFileSystem):
    """Filesystem backed by an HDFS cluster via *fsspec* / *pyarrow*.

    The *fsspec* + *pyarrow* dependency is only imported when the first
    I/O operation is performed.  Host and port are parsed from ``hdfs://``
    URIs and passed explicitly to *pyarrow*, so the Hadoop configuration
    files (``core-site.xml``) do **not** need to set ``fs.defaultFS``.
    """

    def __init__(self) -> None:
        self._fs = None  # lazily initialised on first use
        self._host: str | None = None
        self._port: int = 0

    # -- lazy init helper -----------------------------------------------

    def _ensure_fs(self) -> None:
        if self._fs is not None:
            return
        try:
            import fsspec  # noqa: F401
        except ImportError:
            raise ImportError(
                "HDFS support requires the 'fsspec[arrow]' package. "
                "Install it with:  pip install fsspec[arrow]"
            )
        try:
            import pyarrow.fs as _pafs
        except ImportError:
            raise ImportError(
                "HDFS support requires the 'pyarrow' package (bundled with "
                "fsspec[arrow]). Install it with:  pip install fsspec[arrow]"
            )

        # pyarrow HadoopFileSystem needs explicit host:port when the
        # environment's core-site.xml does not set fs.defaultFS correctly.
        # Otherwise libhdfs falls back to RawLocalFileSystem (file:///).
        if self._host:
            arrow_fs = _pafs.HadoopFileSystem(host=self._host, port=self._port)
        else:
            arrow_fs = _pafs.HadoopFileSystem()

        from fsspec.implementations.arrow import ArrowFSWrapper

        self._fs = ArrowFSWrapper(arrow_fs)

    def _parse_uri(self, path: str) -> None:
        """Extract host + port from an ``hdfs://host:port/...`` URI."""
        if self._host is not None or not path.startswith("hdfs://"):
            return
        rest = path[len("hdfs://") :]
        host_part = rest.split("/", 1)[0]
        if ":" in host_part:
            self._host, port_s = host_part.rsplit(":", 1)
            self._port = int(port_s)
        else:
            self._host = host_part
            self._port = 8020  # default HDFS NameNode port

    # -- I/O methods ----------------------------------------------------

    def open(self, path: str, mode: str = "rb") -> IO[Any]:
        self._parse_uri(path)
        self._ensure_fs()
        return self._fs.open(path, mode)

    def exists(self, path: str) -> bool:
        self._parse_uri(path)
        self._ensure_fs()
        return self._fs.exists(path)

    def isdir(self, path: str) -> bool:
        self._parse_uri(path)
        self._ensure_fs()
        return self._fs.isdir(path)

    def makedirs(self, path: str, exist_ok: bool = True) -> None:
        self._parse_uri(path)
        self._ensure_fs()
        self._fs.makedirs(path, exist_ok=exist_ok)

    def ls(self, path: str) -> List[str]:
        self._parse_uri(path)
        self._ensure_fs()
        return self._fs.ls(path)

    def glob(self, pattern: str) -> List[str]:
        self._parse_uri(pattern)
        self._ensure_fs()
        return self._fs.glob(pattern)

    def size(self, path: str) -> int:
        self._parse_uri(path)
        self._ensure_fs()
        return self._fs.size(path)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_HDFS: HdfsFileSystem | None = None
_LOCAL: LocalFileSystem | None = None


def get_filesystem(path: str) -> AbstractFileSystem:
    """Return the appropriate filesystem for *path*.

    Parameters
    ----------
    path : str
        A filesystem path.  If it starts with ``"hdfs://"`` an
        :class:`HdfsFileSystem` is returned; otherwise the default
        :class:`LocalFileSystem` is returned.

    Returns
    -------
    AbstractFileSystem
    """
    if path.startswith("hdfs://"):
        global _HDFS
        if _HDFS is None:
            _HDFS = HdfsFileSystem()
        return _HDFS
    global _LOCAL
    if _LOCAL is None:
        _LOCAL = LocalFileSystem()
    return _LOCAL
