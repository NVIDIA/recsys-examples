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

import os
import socket
import struct
import threading
import time
from array import array
from ctypes import CDLL, get_errno
from typing import List, Optional


DEFAULT_LAYERWISE_EVENTFD_SOCKET = "/tmp/flexkv_layerwise_eventfd.sock"


class FlexKVLayerwiseEventfdSender:
    """Creates mock layerwise eventfds and sends them to FlexKV's worker."""

    def __init__(
        self,
        num_layers: int,
        socket_path: str,
        num_counters: int = 3,
        timeout_s: float = 180.0,
    ) -> None:
        self.num_layers = int(num_layers)
        self.socket_path = socket_path
        self.num_counters = int(num_counters)
        self.timeout_s = float(timeout_s)
        self._eventfds: Optional[List[List[int]]] = None
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def _create_eventfd() -> int:
        if hasattr(os, "eventfd"):
            return os.eventfd(0, 0)
        libc = CDLL(None, use_errno=True)
        fd = libc.eventfd(0, 0)
        if fd < 0:
            raise OSError(get_errno(), "eventfd creation failed")
        return int(fd)

    def create_eventfds(self) -> List[List[int]]:
        if self._eventfds is None:
            # FlexKV layerwise worker expects counter sets for triple buffering.
            self._eventfds = [
                [self._create_eventfd() for _ in range(self.num_layers)]
                for _ in range(self.num_counters)
            ]
        return self._eventfds

    def layer_eventfds(self, counter_id: int) -> List[int]:
        eventfds = self.create_eventfds()
        if counter_id < 0 or counter_id >= len(eventfds):
            raise ValueError(
                f"Invalid layerwise counter_id={counter_id}, "
                f"expected [0, {len(eventfds)})"
            )
        return eventfds[counter_id]

    def start(self) -> None:
        if self._thread is not None:
            return
        self.create_eventfds()
        self._thread = threading.Thread(
            target=self._send_eventfds,
            name="flexkv-layerwise-eventfd-sender",
            daemon=True,
        )
        self._thread.start()

    def _send_eventfds(self) -> None:
        eventfds = self.create_eventfds()
        deadline = time.time() + self.timeout_s
        last_error: Optional[Exception] = None
        while time.time() < deadline:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.connect(self.socket_path)
                metadata = struct.pack(
                    "iiii",
                    0,
                    1,
                    self.num_layers,
                    len(eventfds),
                )
                sock.sendall(metadata)
                for counter_id, fds in enumerate(eventfds):
                    fd_array = array("i", fds)
                    sock.sendmsg(
                        [struct.pack("i", counter_id)],
                        [
                            (
                                socket.SOL_SOCKET,
                                socket.SCM_RIGHTS,
                                fd_array.tobytes(),
                            )
                        ],
                    )
                ack = sock.recv(1)
                if ack != b"\x01":
                    raise RuntimeError(
                        f"FlexKV layerwise eventfd receiver returned ack={ack!r}"
                    )
                return
            except (FileNotFoundError, ConnectionRefusedError, socket.timeout) as e:
                last_error = e
                time.sleep(0.05)
            finally:
                sock.close()
        raise RuntimeError(
            "Timed out sending mock layerwise eventfds to FlexKV "
            f"socket {self.socket_path}: {last_error}"
        )
